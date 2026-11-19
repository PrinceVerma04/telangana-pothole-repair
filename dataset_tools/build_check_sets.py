"""Check sets for height / lens / night invariance (D48): fixed, generated variants of existing test sets.

    make check-sets        # -> test_datasets/check_<set>_<variant>/ + entries in configs/datasets.yaml

Variants (deterministic, seeded per image; written once, so every model sees exactly the same pictures):
  - night: exposure cut to 3-10 % in linear light, shot + read noise, one headlight pool, slight blur, JPEG.
    Implemented independently of the training augmentation (ml/augment/hard.py) so a model cannot score by having
    learnt that exact transform.
  - zoomout: the photo shrunk to 60 % and centred on grey (camera higher / wider lens: potholes smaller).
  - zoomin: the centre 70 % cropped and enlarged (camera lower / narrower lens: potholes larger, some cut).
Sources: Nagar Netra test, web photos (Kaggle), your phone outlines; night also for eYantra clean roads (false alarms
at night). Labels move with the picture (polygons clipped to the crop).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

TESTS = REPO / "test_datasets"
DATASETS_YAML = REPO / "configs" / "datasets.yaml"
IMG_EXT = (".jpg", ".jpeg", ".png")
# (source test-set id in configs/datasets.yaml, short name, variants)
PLAN = [("base_test", "nagar", ("night", "zoomout", "zoomin")),
        ("web_test", "web", ("night", "zoomout", "zoomin")),
        ("phone_street_marked", "phone", ("night", "zoomout", "zoomin")),
        ("eyantra_negatives", "eyantra", ("night",))]
ZOOM_OUT, ZOOM_IN = 0.6, 0.7


def seed_of(name: str, variant: str) -> int:
    return int(hashlib.sha256(f"{variant}:{name}".encode()).hexdigest()[:8], 16)


def night_image(img: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    lin = (img.astype(np.float32) / 255.0) ** 2.2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0.4, 0.6) * w, h * 1.1
    pool = np.exp(-(((xx - cx) / (0.6 * w)) ** 2 + ((yy - cy) / (0.8 * h)) ** 2) * 2.0)
    gain = rng.uniform(0.03, 0.10) + rng.uniform(0.2, 0.6) * pool  # headlight pool stays below daylight
    lin = lin * gain[..., None]
    lin = rng.poisson(np.clip(lin, 0, 1) * 800.0) / 800.0 + rng.normal(0, 0.004, lin.shape)  # shot + read noise
    out = (np.clip(lin, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    return cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])[1], cv2.IMREAD_COLOR)


def clip_poly_to_rect(poly: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Sutherland-Hodgman clipping of a polygon (k, 2) to a rectangle."""
    def clip(pts: list, inside, cut) -> list:
        out = []
        for i in range(len(pts)):
            a, b = pts[i - 1], pts[i]
            if inside(b):
                if not inside(a):
                    out.append(cut(a, b))
                out.append(b)
            elif inside(a):
                out.append(cut(a, b))
        return out

    def cx(x):
        return lambda a, b: (x, a[1] + (b[1] - a[1]) * (x - a[0]) / ((b[0] - a[0]) or 1e-9))

    def cy(y):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (y - a[1]) / ((b[1] - a[1]) or 1e-9), y)

    pts = [tuple(p) for p in np.asarray(poly, float)]
    for inside, cut in ((lambda p: p[0] >= x0, cx(x0)), (lambda p: p[0] <= x1, cx(x1)),
                        (lambda p: p[1] >= y0, cy(y0)), (lambda p: p[1] <= y1, cy(y1))):
        pts = clip(pts, inside, cut) if pts else []
    return np.asarray(pts, float).reshape(-1, 2)


def read_polys(label: Path, keep_ids: set[int]) -> list[np.ndarray]:
    out = []
    for line in label.read_text(encoding="utf-8").splitlines() if label.is_file() else []:
        t = line.split()
        if len(t) < 5 or int(float(t[0])) not in keep_ids:
            continue
        v = np.asarray(t[1:], float)
        if len(v) == 4:
            cx_, cy_, bw, bh = v
            v = np.array([cx_ - bw / 2, cy_ - bh / 2, cx_ + bw / 2, cy_ - bh / 2, cx_ + bw / 2, cy_ + bh / 2,
                          cx_ - bw / 2, cy_ + bh / 2])
        out.append(v[: len(v) // 2 * 2].reshape(-1, 2))
    return out


def transform(img: np.ndarray, polys: list[np.ndarray], variant: str, seed: int) -> tuple[np.ndarray, list]:
    """polys normalised (k, 2) in, normalised out."""
    h, w = img.shape[:2]
    if variant == "night":
        return night_image(img, seed), polys
    if variant == "zoomout":
        small = cv2.resize(img, (round(w * ZOOM_OUT), round(h * ZOOM_OUT)), interpolation=cv2.INTER_AREA)
        canvas = np.full_like(img, 114)
        ox, oy = (w - small.shape[1]) // 2, (h - small.shape[0]) // 2
        canvas[oy: oy + small.shape[0], ox: ox + small.shape[1]] = small
        return canvas, [p * ZOOM_OUT + [ox / w, oy / h] for p in polys]
    if variant == "zoomin":
        m0, m1 = (1 - ZOOM_IN) / 2, (1 + ZOOM_IN) / 2
        crop = img[round(h * m0): round(h * m1), round(w * m0): round(w * m1)]
        out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        res = []
        for p in polys:
            q = clip_poly_to_rect(p, m0, m0, m1, m1)
            if len(q) >= 3:
                q = (q - m0) / ZOOM_IN
                area = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1)))
                if area > 1e-4:  # keep potholes still visibly inside the crop
                    res.append(q)
        return out, res
    raise ValueError(variant)


def build(tests: Path = TESTS, datasets_yaml: Path = DATASETS_YAML, root: Path = REPO, overwrite: bool = False,
          plan: list = PLAN) -> list[dict]:
    from ml import benchmark as B

    ds = yaml.safe_load(datasets_yaml.read_text(encoding="utf-8"))
    specs = {t["id"]: t for t in ds["tests"]}
    made = []
    for src_id, short, variants in plan:
        spec = specs[src_id]
        ts = B.load_testset(spec, root)
        for v in variants:
            out = tests / f"check_{short}_{v}"
            if out.exists():
                if not overwrite:
                    made.append({"id": f"check_{short}_{v}", "skipped": True})
                    continue
                shutil.rmtree(out)
            (out / "images").mkdir(parents=True)
            has_labels = spec["kind"] == "yolo"
            if has_labels:
                (out / "labels").mkdir()
            keep = {int(i) for i in spec.get("pothole_ids", [0])}
            for p, _ in ts["items"]:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                polys = read_polys(root / spec["labels"] / f"{p.stem}.txt", keep) if has_labels else []
                im2, polys2 = transform(img, polys, v, seed_of(p.name, v))
                cv2.imwrite(str(out / "images" / f"{p.stem}.jpg"), im2, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if has_labels:
                    (out / "labels" / f"{p.stem}.txt").write_text(
                        "".join("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in q) + "\n" for q in polys2))
            entry = {"id": f"check_{short}_{v}", "name": f"CHECK {v}: {spec.get('name', src_id)}",
                     "kind": spec["kind"], "images": str(out.relative_to(root) / "images"), "check": v,
                     "note": f"D48 synthetic {v} variant of {src_id} (dataset_tools/build_check_sets.py)"}
            if has_labels:
                entry.update(labels=str(out.relative_to(root) / "labels"), pothole_ids=[0])
            made.append(entry)
    # register (replace earlier check entries, keep everything else as it is)
    text = datasets_yaml.read_text(encoding="utf-8")
    marker = "\n  # --- D48 check sets (generated by dataset_tools/build_check_sets.py; do not edit by hand)\n"
    if marker in text:
        text = text[: text.index(marker)] + "\n"
    entries = [m for m in made if not m.get("skipped")] or []
    existing = {m["id"] for m in made if m.get("skipped")}
    if entries or existing:
        allm = []
        for src_id, short, variants in plan:
            for v in variants:
                eid = f"check_{short}_{v}"
                e = next((m for m in entries if m["id"] == eid), None)
                if e is None and eid in existing:
                    e = {"id": eid, "name": f"CHECK {v}: {specs[src_id].get('name', src_id)}",
                         "kind": specs[src_id]["kind"], "images": f"test_datasets/{eid}/images", "check": v,
                         "note": f"D48 synthetic {v} variant of {src_id} (dataset_tools/build_check_sets.py)"}
                    if specs[src_id]["kind"] == "yolo":
                        e.update(labels=f"test_datasets/{eid}/labels", pothole_ids=[0])
                if e:
                    allm.append(e)
        text = text.rstrip("\n") + marker + "".join(
            "  - " + json.dumps(e, ensure_ascii=False) + "\n" for e in allm)
        datasets_yaml.write_text(text, encoding="utf-8")
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    for m in build(overwrite=a.overwrite):
        print(m["id"], "(kept)" if m.get("skipped") else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
