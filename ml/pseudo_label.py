"""Pseudo-labels for unlabelled road frames (WP3): current segmenter (+ optional second model) -> YOLO-seg labels
that a human reviews before they are ever trained on.

    python -m ml.pseudo_label --images app_data/store/frames --weights models/base_model/pothole_seg11n.pt \
        --out datasets/extracted/pseudo/bus_frames_2026w46 [--verifier <second model .pt>] \
        [--mask auto|sam2|grabcut|ellipse] [--limit 500]

For every image (``*.jpg|*.jpeg|*.png``, searched recursively; read with OpenCV only, never executed):

* the segmenter predicts detections (canonical classes of ``configs/classes.yaml``);
* a detection-only model (the D9 bootstrap weights) gets its mask from ``measure.detect.box_to_mask`` (``--mask``,
  default ``ellipse``: no model download; ``auto`` tries SAM2.1 then GrabCut);
* each detection is **accepted** when conf >= ``--conf-accept`` and (with ``--verifier``) the second model has a box
  of the same class with IoU >= ``--agree-iou``; it goes to **review** when conf >= ``--conf-review`` but not
  accepted; below that it is dropped;
* accepted detections are written as YOLO-seg lines (``cls x1 y1 x2 y2 ...`` normalised); images with review items
  also get an overlay JPEG in ``review/`` (green = accepted, orange = review) and are listed in ``review.csv``.

Output (``--out``): ``images/`` (copies), ``labels/`` (YOLO-seg, accepted only), ``review/``, ``review.csv``,
``manifest.json`` (weights + sha256, thresholds, counts, ``status: needs_human_review``). Nothing here touches
``datasets``: merging reviewed pseudo-labels into a training split is a separate, explicit step
(``dataset_tools/build_dataset.py``), so an unreviewed label can never reach a model by accident.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from ml.common import class_index, rel, resolve_repo_path, sha256_file, utc_now_iso, write_json

IMG_EXT = {".jpg", ".jpeg", ".png"}


@dataclass
class Decision:
    cls: str
    conf: float
    bbox: list[float]
    polygon: Optional[list[list[float]]]
    mask_method: str
    verdict: str  # accept | review | drop
    reason: str


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def decide(dets: list[Any], verifier_dets: Optional[list[Any]], conf_accept: float, conf_review: float,
           agree_iou: float) -> list[tuple[Any, str, str]]:
    """(detection, verdict, reason) for each detection of the primary model."""
    out = []
    for d in dets:
        if d.conf < conf_review:
            out.append((d, "drop", f"conf {d.conf:.2f} < {conf_review}"))
            continue
        agree = None
        if verifier_dets is not None:
            agree = max((box_iou(d.bbox, v.bbox) for v in verifier_dets if v.cls == d.cls), default=0.0)
        if d.conf >= conf_accept and (agree is None or agree >= agree_iou):
            why = f"conf {d.conf:.2f}" + (f", verifier IoU {agree:.2f}" if agree is not None else "")
            out.append((d, "accept", why))
        elif agree is not None and agree < agree_iou:
            out.append((d, "review", f"verifier disagrees (IoU {agree:.2f} < {agree_iou})"))
        else:
            out.append((d, "review", f"conf {d.conf:.2f} < {conf_accept}"))
    return out


def yolo_seg_line(cls_idx: int, polygon: list[list[float]], w: int, h: int) -> str:
    pts = " ".join(f"{min(1.0, max(0.0, x / w)):.6f} {min(1.0, max(0.0, y / h)):.6f}" for x, y in polygon)
    return f"{cls_idx} {pts}"


def polygon_for(img: np.ndarray, d: Any, mask_method: str) -> tuple[Optional[list[list[float]]], str]:
    if d.polygon and len(d.polygon) >= 3:
        return [list(map(float, p)) for p in d.polygon], "detector_seg"
    from measure.detect import box_to_mask
    from ml.segmenter import mask_to_polygon

    try:
        mk, used = box_to_mask(img, list(d.bbox), method=mask_method)
    except RuntimeError as e:
        return None, f"no mask: {e}"
    return mask_to_polygon(mk.astype(bool)), used


def overlay(img: np.ndarray, items: list[Decision]) -> np.ndarray:
    import cv2

    out = img.copy()
    for it in items:
        if it.verdict == "drop":
            continue
        col = (60, 200, 60) if it.verdict == "accept" else (0, 150, 255)
        x1, y1, x2, y2 = (int(round(v)) for v in it.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        if it.polygon:
            cv2.polylines(out, [np.round(np.asarray(it.polygon)).astype(np.int32)], True, col, 2)
        cv2.putText(out, f"{it.cls} {it.conf:.2f} {it.verdict}", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1, cv2.LINE_AA)
    return out


def iter_images(root: Path, limit: int) -> list[Path]:
    imgs = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
    return imgs[:limit] if limit else imgs


def run(a: argparse.Namespace) -> dict[str, Any]:
    import cv2

    from ml.segmenter import load_segmenter

    src = Path(a.images)
    if not src.is_dir():
        raise SystemExit(f"{src} is not a directory")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise SystemExit(f"{out} is not empty (use --overwrite)")
    for sub in ("images", "labels", "review"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    seg = load_segmenter(a.weights, conf=min(a.conf_review, 0.25), device=a.device)
    ver = load_segmenter(a.verifier, conf=0.1, device=a.device) if a.verifier else None
    counts = {"images": 0, "unreadable": 0, "with_accepted": 0, "with_review": 0, "accept": 0, "review": 0,
              "drop": 0, "no_mask": 0}
    t0 = time.perf_counter()
    review_rows: list[dict[str, Any]] = []
    for path in iter_images(src, a.limit):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            counts["unreadable"] += 1
            continue
        counts["images"] += 1
        h, w = img.shape[:2]
        dets = [d for d in seg.predict(img) if class_index(d.cls) is not None]
        vdets = [d for d in ver.predict(img)] if ver is not None else None
        items: list[Decision] = []
        for d, verdict, why in decide(dets, vdets, a.conf_accept, a.conf_review, a.agree_iou):
            poly, how = (None, "-") if verdict == "drop" else polygon_for(img, d, a.mask)
            if verdict != "drop" and poly is None:
                counts["no_mask"] += 1
                verdict, why = "review", f"{why}; {how}"
            items.append(Decision(d.cls, float(d.conf), list(map(float, d.bbox)), poly, how, verdict, why))
            counts[verdict] += 1
        stem = f"{path.parent.name}__{path.stem}" if a.flatten_names else path.stem
        accepted = [it for it in items if it.verdict == "accept" and it.polygon]
        review = [it for it in items if it.verdict == "review"]
        shutil.copy2(path, out / "images" / f"{stem}{path.suffix.lower()}")
        lines = [yolo_seg_line(int(class_index(it.cls)), it.polygon, w, h) for it in accepted]  # type: ignore[arg-type]
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts["with_accepted"] += bool(accepted)
        if review:
            counts["with_review"] += 1
            cv2.imwrite(str(out / "review" / f"{stem}.jpg"), overlay(img, items))
            for it in review:
                review_rows.append({"image": f"images/{stem}{path.suffix.lower()}", "cls": it.cls,
                                    "conf": round(it.conf, 3), "bbox": " ".join(f"{v:.0f}" for v in it.bbox),
                                    "mask": it.mask_method, "reason": it.reason})
    seg.close()
    if ver is not None:
        ver.close()
    with (out / "review.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["image", "cls", "conf", "bbox", "mask", "reason"])
        wr.writeheader()
        wr.writerows(review_rows)
    manifest = {
        "schema": "gunthameter.pseudo_labels/1", "created_at": utc_now_iso(), "status": "needs_human_review",
        "source_images": rel(src), "weights": rel(resolve_repo_path(a.weights)) if a.weights else None,
        "weights_sha256": sha256_file(resolve_repo_path(a.weights)) if a.weights else None,
        "verifier": rel(resolve_repo_path(a.verifier)) if a.verifier else None,
        "thresholds": {"conf_accept": a.conf_accept, "conf_review": a.conf_review, "agree_iou": a.agree_iou},
        "mask_method": a.mask, "counts": counts, "seconds": round(time.perf_counter() - t0, 1),
        "note": "labels/ holds only accepted detections; review/ + review.csv list the rest. Review before use."}
    write_json(out / "manifest.json", manifest)
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RoadFix pseudo-labelling (WP3)")
    ap.add_argument("--images", required=True, help="directory of frames (searched recursively)")
    ap.add_argument("--weights", default=None, help="segmenter (.pt/.onnx); default SEG_WEIGHTS / bootstrap")
    ap.add_argument("--verifier", default=None, help="second model: accept only when both agree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf-accept", type=float, default=0.5)
    ap.add_argument("--conf-review", type=float, default=0.25)
    ap.add_argument("--agree-iou", type=float, default=0.3)
    ap.add_argument("--mask", choices=("auto", "sam2", "grabcut", "ellipse"), default="ellipse",
                    help="mask for detection-only models (auto/sam2 download SAM2.1 on first use)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flatten-names", action="store_true", help="prefix file names with their folder name")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    m = run(a)
    print(f"{m['counts']} in {m['seconds']} s -> {a.out} (status: {m['status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
