"""Model evaluation -> ``reports/model_eval.json`` (PLAN 3.4 eval metrics, WP3 acceptance).

    python -m ml.eval --weights models/training_runs/seg/y26m_seg_india_v2ft_geo/weights/best_pothole.pt --stage trained_y26m_seg_india_v2ft_geo
    python -m ml.eval --weights models/training_runs/seg/<name>/weights/best.pt --stage smoke --compare runs/.../best.onnx --limit 50

Ground truth
  * Box GT for the test split (RDD-India D40 held-out, 514 images) defaults to the **original RDD2022 boxes**
    (``datasets/extracted/rdd2022/india/labels/<id>.txt``, class ids from ``meta.json``) when every image resolves; the
    dataset label polygons are SAM2 pseudo-masks whose bounding boxes differ from the annotated boxes. The polygon-bbox
    numbers are reported as well (``vs_label_polygons``) because Ultralytics' own val uses that convention.
  * Mask GT = the dataset polygons. On the test split these are SAM2.1 pseudo-masks from RDD boxes (962 of 1,038
    pothole instances; 76 box-polygon fallbacks), not human masks; eYantra val masks contain no pothole class (D13).
    Mask numbers are therefore "agreement with SAM2 pseudo-masks", reported as such.

Metrics: per class AP50 / AP50-95 (COCO-greedy matching, 101-point AP), P/R at conf 0.25, best F1; for seg models
mask AP50, mean IoU of box-matched masks at conf >= 0.25 and dataset pixel IoU; mean over the classes the model can
predict that have ground truth. Timings: wall ms per image (pre + infer + post), device, imgsz.
The report keeps one entry per (stage, weights, split, limit); re-running replaces that entry.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ml.common import (
    DATASET_CFG,
    REPO_ROOT,
    REPORTS,
    class_index,
    class_names,
    label_path_for,
    read_json,
    rel,
    sha256_file,
    split_images,
    utc_now_iso,
    write_json,
)
from ml.metrics import Evaluator, eval_hw

REPORT_PATH = REPORTS / "model_eval.json"
RDD_INTERIM = REPO_ROOT / "datasets" / "extracted" / "rdd2022"
TARGETS = {
    "box_map50_pothole_rdd_india_test": {"floor": 0.70, "target": 0.75, "source": "PLAN WP3 acceptance / 3.4"},
    "box_map50_95_pothole_rdd_india_test": {"target": 0.45, "source": "PLAN 3.4"},
    "mask_iou": {"floor": 0.65, "target": 0.70, "source": "PLAN WP3 acceptance / 3.4"},
    "onnx_map50_delta": {"max_abs": 0.02, "source": "PLAN WP3 acceptance (<= 2 % mAP delta)"},
}


# ============================================================================================ ground truth
def read_label_file(path: Path, hw: tuple[int, int], id_map: Optional[dict[int, int]] = None
                    ) -> tuple[list[int], np.ndarray, list[np.ndarray]]:
    """YOLO label (polygon ``cls x y x y ...`` or box ``cls cx cy w h``, normalised) -> (cls, boxes xyxy px,
    polygons px). ``id_map``: dataset label id -> canonical class index (a dataset may keep only a subset of the
    classes, D15); ids missing from the map are skipped. None = ids are already canonical."""
    H, W = hw
    cls, boxes, polys = [], [], []
    if not path.is_file():
        return cls, np.zeros((0, 4)), polys
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        c = int(float(t[0]))
        if id_map is not None:
            if c not in id_map:
                continue
            c = id_map[c]
        v = np.asarray(t[1:], float)
        if len(v) == 4:
            cx, cy, w, h = v * [W, H, W, H]
            poly = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2],
                             [cx - w / 2, cy + h / 2]])
        else:
            poly = v[: len(v) // 2 * 2].reshape(-1, 2) * [W, H]
        cls.append(c)
        polys.append(poly)
        boxes.append([poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()])
    return cls, np.asarray(boxes, float).reshape(-1, 4), polys


_RDD_MAP: Optional[dict[int, Optional[int]]] = None


def rdd_class_map() -> dict[int, Optional[int]]:
    """RDD interim class id -> frozen class index (D40->pothole, D20->alligator_crack, D00/D10->crack, OTHER->None)."""
    global _RDD_MAP
    if _RDD_MAP is None:
        import yaml

        meta = read_json(RDD_INTERIM / "meta.json", {}) or {}
        ids = meta.get("class_ids") or {"0": "D00", "1": "D10", "2": "D20", "3": "OTHER", "4": "D40"}
        smap = (yaml.safe_load((REPO_ROOT / "configs" / "classes.yaml").read_text(encoding="utf-8"))
                .get("source_maps", {}).get("rdd2022", {}))
        _RDD_MAP = {int(k): (class_index(smap[v]) if v in smap else None) for k, v in ids.items()}
    return _RDD_MAP


def rdd_original_label(img: Path) -> Optional[Path]:
    stem = img.stem
    if not stem.startswith("rddin_"):
        return None
    p = RDD_INTERIM / "india" / "labels" / f"{stem.split('_', 1)[1]}.txt"
    return p if p.is_file() else None


def read_rdd_original(path: Path, hw: tuple[int, int]) -> tuple[list[int], np.ndarray]:
    H, W = hw
    mp = rdd_class_map()
    cls, boxes = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.split()
        if len(t) != 5:
            continue
        c = mp.get(int(float(t[0])))
        if c is None:
            continue
        cx, cy, w, h = np.asarray(t[1:], float) * [W, H, W, H]
        cls.append(c)
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return cls, np.asarray(boxes, float).reshape(-1, 4)


def choose_gt_boxes(images: Sequence[Path], mode: str) -> str:
    """'rdd_original' when any image has an original RDD label (per image: original RDD boxes where they exist,
    dataset labels for the others, e.g. eYantra manhole frames in the test split, D15); else 'labels'."""
    if mode != "auto":
        return mode
    return "rdd_original" if images and any(rdd_original_label(p) is not None for p in images) else "labels"


# ============================================================================================ evaluation
# Test images by source (uid prefix written by dataset_tools/build_dataset.py), scored separately next to the whole split.
# RDD-India is the acceptance basis (``rdd_india_subset``, original RDD boxes); the others use the label polygons.
SOURCE_SUBSETS = {"bharat": "bh_", "webseg_human_masks": "ws_", "eyantra": "ey_"}


def sample_evenly(images: Sequence[Path], n: int) -> list[Path]:
    """``n`` images spread evenly over the sorted list (``--limit``): sorted names group by source prefix, so the first
    n would all come from one source."""
    if not n or n >= len(images):
        return list(images)
    step = len(images) / n
    return [images[int(i * step)] for i in range(n)]


def evaluate(seg: Any, images: Sequence[Path], gt_boxes: str = "auto", conf: float = 0.001, masks: bool = True,
             mask_max_side: int = 256, progress: bool = True, label_ids: Optional[dict[int, int]] = None
             ) -> dict[str, Any]:
    """Run ``seg.predict`` over ``images`` and score it. Returns a dict (metrics + timings).

    ``label_ids``: dataset label id -> canonical index (``ml.common.dataset_class_ids``); None = canonical ids.
    When some images have original RDD labels, ``rdd_india_subset`` scores those images alone: the PLAN WP3 target
    ("box mAP50 on the RDD-India D40 test") stays comparable after other images join the test split (D15)."""
    import cv2

    names = class_names()
    mode = choose_gt_boxes(images, gt_boxes)
    ev_box = Evaluator(names)  # chosen box GT
    ev_lab = Evaluator(names, mask_max_side=mask_max_side)  # label polygons (box + mask)
    ev_rdd = Evaluator(names)  # original RDD boxes, RDD-India images only
    n_rdd = 0
    ev_sub = {k: Evaluator(names, mask_max_side=mask_max_side) for k in SOURCE_SUBSETS}
    n_sub: dict[str, int] = dict.fromkeys(SOURCE_SUBSETS, 0)
    capable = sorted({class_index(c) for c in seg.canonical_names().values() if c and class_index(c) is not None})
    do_masks = masks and str(getattr(seg, "task", "")) == "segment"
    times, n_det = [], 0
    t_start = time.perf_counter()
    for i, p in enumerate(images):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cannot read {p}")
        hw = img.shape[:2]
        mhw = eval_hw(hw[0], hw[1], mask_max_side)
        t0 = time.perf_counter()
        dets = seg.predict(img, conf=conf, mask_hw=mhw if do_masks else None)
        times.append((time.perf_counter() - t0) * 1000.0)
        dets = [d for d in dets if d.cls_id is not None]
        n_det += len(dets)
        pc = [d.cls_id for d in dets]
        pf = [d.conf for d in dets]
        pb = np.asarray([d.bbox for d in dets], float).reshape(-1, 4)
        lc, lb, lp = read_label_file(label_path_for(p), hw, label_ids)
        rp = rdd_original_label(p) if mode == "rdd_original" else None
        if rp is not None:
            bc, bb = read_rdd_original(rp, hw)
            ev_rdd.add_image(bc, bb, pc, pf, pb)
            n_rdd += 1
        else:
            bc, bb = lc, lb
        ev_box.add_image(bc, bb, pc, pf, pb)
        if do_masks:
            from ml.segmenter import polygon_to_mask

            sx, sy = mhw[1] / hw[1], mhw[0] / hw[0]
            gm = [polygon_to_mask(np.asarray(q) * [sx, sy], mhw) for q in lp]
            pm = [d.mask if d.mask is not None else np.zeros(mhw, bool) for d in dets]
            ev_lab.add_image(lc, lb, pc, pf, pb, gt_masks=gm, pred_masks=pm, image_hw=mhw)
        else:
            gm = pm = None
            ev_lab.add_image(lc, lb, pc, pf, pb)
        for k, pre in SOURCE_SUBSETS.items():
            if p.name.startswith(pre):
                n_sub[k] += 1
                if do_masks:
                    ev_sub[k].add_image(lc, lb, pc, pf, pb, gt_masks=gm, pred_masks=pm, image_hw=mhw)
                else:
                    ev_sub[k].add_image(lc, lb, pc, pf, pb)
        if progress and (i + 1) % 50 == 0:
            el = time.perf_counter() - t_start
            print(f"  {i + 1}/{len(images)} images, {el:.0f} s, {np.mean(times):.0f} ms/img", flush=True)
    wall = time.perf_counter() - t_start
    tt = np.asarray(times[3:] if len(times) > 6 else times)
    box = ev_box.summary(capable)
    lab = ev_lab.summary(capable)
    pot = box["per_class"].get("pothole", {})
    res: dict[str, Any] = {
        "n_images": len(images), "n_detections": n_det, "gt_boxes": mode,
        "gt_masks": "dataset label polygons (test: SAM2.1 pseudo-masks from RDD boxes)" if do_masks else None,
        "model_classes_capable": [names[c] for c in capable],
        "pothole": {k: pot.get(k) for k in ("n_gt", "n_pred", "box_ap50", "box_ap50_95", "p_at_025", "r_at_025",
                                            "best_f1", "best_f1_conf")},
        "capable_classes": {"mean_over": box["mean_over"], "map50": box["map50"], "map50_95": box["map50_95"]},
        "per_class": box["per_class"],
        "vs_label_polygons": {"pothole_box_ap50": lab["per_class"].get("pothole", {}).get("box_ap50"),
                              "pothole_box_ap50_95": lab["per_class"].get("pothole", {}).get("box_ap50_95"),
                              "map50": lab["map50"], "mean_over": lab["mean_over"]},
        "timing": {"wall_s": round(wall, 1), "ms_per_image_mean": _r(tt.mean()), "ms_per_image_p50": _r(np.median(tt)),
                   "ms_per_image_p95": _r(np.percentile(tt, 95)), "note": "predict() wall time incl. pre/post; "
                   "first 3 images excluded as warm-up"},
    }
    if n_rdd and n_rdd < len(images):
        rdd = ev_rdd.summary(capable)
        rpot = rdd["per_class"].get("pothole", {})
        res["rdd_india_subset"] = {
            "n_images": n_rdd, "gt_boxes": "rdd_original",
            "pothole": {k: rpot.get(k) for k in ("n_gt", "n_pred", "box_ap50", "box_ap50_95", "p_at_025", "r_at_025",
                                                 "best_f1", "best_f1_conf")},
            "map50": rdd["map50"], "mean_over": rdd["mean_over"],
            "note": "RDD-India test images only: the PLAN WP3 acceptance basis (box mAP50 pothole >= 0.70)"}
    subs: dict[str, Any] = {}
    for k, ev in ev_sub.items():
        if not n_sub[k] or n_sub[k] == len(images):
            continue
        sm = ev.summary(capable)
        sp = sm["per_class"].get("pothole", {})
        subs[k] = {"n_images": n_sub[k], "gt": "human-drawn polygons" if k == "webseg_human_masks" else "label polygons",
                   "pothole": {x: sp.get(x) for x in ("n_gt", "n_pred", "box_ap50", "box_ap50_95", "p_at_025",
                                                       "r_at_025", "best_f1", "best_f1_conf")},
                   "map50": sm["map50"], "mean_over": sm["mean_over"]}
        if do_masks:
            subs[k]["mask"] = {"pothole_mask_ap50": sp.get("mask_ap50"),
                               "pothole_mean_matched_mask_iou": sp.get("mean_matched_mask_iou"),
                               "pothole_n_matched_masks": sp.get("n_matched_masks"), "pothole_pixel_iou": sp.get("pixel_iou")}
    if subs:
        res["source_subsets"] = subs
    if do_masks:
        lp_ = lab["per_class"].get("pothole", {})
        res["mask"] = {"pothole_mask_ap50": lp_.get("mask_ap50"), "pothole_mask_ap50_95": lp_.get("mask_ap50_95"),
                       "pothole_mean_matched_mask_iou": lp_.get("mean_matched_mask_iou"),
                       "pothole_n_matched_masks": lp_.get("n_matched_masks"),
                       "pothole_pixel_iou": lp_.get("pixel_iou"), "mask_map50_capable": lab["mask_map50"],
                       "eval_mask_max_side": mask_max_side,
                       "note": "IoU against the label polygons at <= %d px (SAM2.1 pseudo-masks for RDD/BharatPotHole, "
                               "human-drawn for the ws_ subset: see source_subsets); matched = box IoU >= 0.5 at "
                               "conf >= 0.25" % mask_max_side}
    else:
        res["mask"] = {"status": "n/a", "reason": "detection-only model: no masks; mask IoU is not defined"}
    return res


def parity(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare two evaluate() results on the same images (e.g. .pt vs .onnx)."""
    def g(r: dict, *k: str) -> Optional[float]:
        for kk in k:
            r = (r or {}).get(kk)
        return r

    pa, pb = g(a, "pothole", "box_ap50"), g(b, "pothole", "box_ap50")
    ma, mb = g(a, "capable_classes", "map50"), g(b, "capable_classes", "map50")
    out = {"pothole_ap50": [pa, pb], "pothole_ap50_delta": _r(pb - pa) if pa is not None and pb is not None else None,
           "map50_capable": [ma, mb], "map50_delta": _r(mb - ma) if ma is not None and mb is not None else None,
           "n_detections": [a.get("n_detections"), b.get("n_detections")],
           "ms_per_image": [g(a, "timing", "ms_per_image_mean"), g(b, "timing", "ms_per_image_mean")]}
    if a.get("mask", {}).get("pothole_mask_ap50") is not None:
        out["pothole_mask_ap50"] = [g(a, "mask", "pothole_mask_ap50"), g(b, "mask", "pothole_mask_ap50")]
    d = out["map50_delta"] if out["map50_delta"] is not None else out["pothole_ap50_delta"]
    out["within_2pct"] = None if d is None else bool(abs(d) <= 0.02)
    return out


def detection_agreement(seg_a: Any, seg_b: Any, images: Sequence[Path], conf: float = 0.25) -> dict[str, Any]:
    """Per-detection agreement at an operating point: count delta, greedy match (same class, IoU>=0.5), box IoU."""
    import cv2

    from ml.metrics import box_iou

    na = nb = nm = 0
    ious, conf_d = [], []
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        da, db = seg_a.predict(img, conf=conf), seg_b.predict(img, conf=conf)
        na, nb = na + len(da), nb + len(db)
        taken = set()
        for x in da:
            best, bi = 0.0, None
            for j, y in enumerate(db):
                if j in taken or y.cls != x.cls:
                    continue
                iou = float(box_iou(np.asarray([x.bbox]), np.asarray([y.bbox]))[0, 0])
                if iou > best:
                    best, bi = iou, j
            if bi is not None and best >= 0.5:
                taken.add(bi)
                nm += 1
                ious.append(best)
                conf_d.append(abs(x.conf - db[bi].conf))
    return {"conf": conf, "n_images": len(images), "n_a": na, "n_b": nb, "n_matched": nm,
            "count_delta_pct": _r(100.0 * (nb - na) / na) if na else None,
            "mean_box_iou_matched": _r(np.mean(ious)) if ious else None,
            "mean_abs_conf_delta": _r(np.mean(conf_d)) if conf_d else None}


def ultralytics_val(weights: Path, split: str, imgsz: int, device: str, batch: int = 4,
                    data_cfg: Path | str = DATASET_CFG) -> dict[str, Any]:
    """Cross-check with Ultralytics' own val (box GT = polygon bboxes). Only for weights whose classes are exactly the
    dataset's classes, in the same order."""
    from ml.common import load_dataset_cfg, names_tuple, ultralytics_env, write_resolved_data_yaml

    ultralytics_env()
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = tuple(model.names[i] for i in sorted(model.names))
    ds_names = names_tuple(load_dataset_cfg(data_cfg)["names"])
    if names != ds_names:
        return {"status": "skipped", "reason": f"model classes {names} != dataset classes {ds_names}"}
    data = write_resolved_data_yaml(REPO_ROOT / "models" / "training_runs" / "seg" / "_data" / "eval_data.yaml", data_cfg)
    t0 = time.perf_counter()
    m = model.val(data=str(data), split=split, imgsz=imgsz, batch=batch, device=device, plots=False, verbose=False,
                  project=str(REPO_ROOT / "models" / "training_runs" / "seg" / "_val"), name=weights.stem, exist_ok=True, workers=0,
                  half=False)
    out: dict[str, Any] = {"wall_s": round(time.perf_counter() - t0, 1), "box_map50_all": _r(m.box.map50),
                           "box_map50_95_all": _r(m.box.map)}
    pid = ds_names.index("pothole") if "pothole" in ds_names else -1  # model/dataset id, not the canonical one
    try:
        idx = list(m.box.ap_class_index).index(pid)
        out["pothole_box_ap50"] = _r(m.box.all_ap[idx, 0])
        out["pothole_box_ap50_95"] = _r(m.box.all_ap[idx].mean())
        if hasattr(m, "seg"):
            out["pothole_mask_ap50"] = _r(m.seg.all_ap[idx, 0])
            out["pothole_mask_ap50_95"] = _r(m.seg.all_ap[idx].mean())
    except (ValueError, AttributeError, IndexError):
        out["pothole"] = "no pothole predictions/GT"
    return out


# ============================================================================================ report
def load_report(path: Path = REPORT_PATH) -> dict[str, Any]:
    rep = read_json(path, None) or {}
    rep.setdefault("schema", "gunthameter.model_eval/1")
    rep.setdefault("runs", [])
    rep["targets"] = TARGETS
    return rep


def upsert_run(rep: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    key = (run.get("stage"), run.get("weights"), run.get("split"), run.get("limit"))
    rep["runs"] = [r for r in rep["runs"] if (r.get("stage"), r.get("weights"), r.get("split"), r.get("limit")) != key]
    rep["runs"].append(run)
    rep["updated_at"] = utc_now_iso()
    rep["acceptance"] = acceptance(rep)
    return rep


def acceptance(rep: dict[str, Any]) -> dict[str, Any]:
    """PLAN WP3 acceptance against the best *full test split* run of a trained (non-baseline, non-smoke) model."""
    full = [r for r in rep.get("runs", []) if r.get("split") == "test" and not r.get("limit")
            and not str(r.get("stage", "")).startswith(("baseline", "smoke"))]
    st: dict[str, Any] = {"evaluated_on": None, "box_map50_pothole": None, "mask_iou_pothole": None,
                          "onnx_delta": None, "status": "NOT VERIFIED: no full test-split evaluation of a trained "
                                                        "model yet (long training not run)"}
    if full:
        def pothole_ap(run: dict[str, Any]) -> Optional[float]:
            mt_ = run.get("metrics", {})  # RDD-India images only when the test split also holds other sources (D15)
            return (mt_.get("rdd_india_subset") or mt_).get("pothole", {}).get("box_ap50")

        r = max(full, key=lambda r: pothole_ap(r) or 0)
        mt = r.get("metrics", {})
        ap = pothole_ap(r)
        iou = mt.get("mask", {}).get("pothole_mean_matched_mask_iou")
        st.update({"evaluated_on": f"{r['stage']} ({r['weights']})", "box_map50_pothole": ap,
                   "mask_iou_pothole": iou,
                   "box_map50_floor_ok": None if ap is None else ap >= 0.70,
                   "mask_iou_floor_ok": None if iou is None else iou >= 0.65})
        st["status"] = "evaluated (see *_ok flags)"
    par = [r for r in rep.get("runs", []) if r.get("parity")]
    if par:
        st["onnx_delta"] = par[-1]["parity"]
    return st


def _r(x: Any) -> Optional[float]:
    try:
        return None if x is None or np.isnan(float(x)) else round(float(x), 4)
    except (TypeError, ValueError):
        return None


def run_meta(seg: Any, weights: Path, args: argparse.Namespace, device: str) -> dict[str, Any]:
    from ml.common import device_info

    di = device_info() if device.startswith("cuda") else {"name": "cpu"}
    return {"stage": args.stage, "weights": rel(weights), "weights_sha256": sha256_file(weights),
            "format": weights.suffix.lstrip("."), "task": str(getattr(seg, "task", "")), "split": args.split,
            "limit": args.limit or None, "imgsz": args.imgsz, "conf": args.conf, "max_det": args.max_det,
            "iou_nms": 0.7, "device": device, "device_name": di.get("name"),
            "providers": getattr(seg, "providers", None), "host": platform.node(), "created_at": utc_now_iso()}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True, help=".pt or .onnx (repo-relative ok)")
    ap.add_argument("--stage", required=True, help="label, e.g. baseline_bootstrap | smoke | trained_y26n_laptop")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--data", default=str(DATASET_CFG))
    ap.add_argument("--gt-boxes", default="auto", choices=("auto", "rdd_original", "labels"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="N images spread evenly over the split; 0 = all")
    ap.add_argument("--device", default=None, help="cuda:0 | cpu (default: GPU if usable and free)")
    ap.add_argument("--compare", default=None, help="second weights (e.g. the ONNX export) for a parity check")
    ap.add_argument("--compare-stage", default=None)
    ap.add_argument("--ultralytics-val", action="store_true", help="also run Ultralytics val as a cross-check")
    ap.add_argument("--out", default=str(REPORT_PATH))
    ap.add_argument("--note", action="append", default=[])
    args = ap.parse_args(argv)

    from ml.common import resolve_repo_path
    from ml.segmenter import load_segmenter

    weights = resolve_repo_path(args.weights)
    images = split_images(args.split, args.data)
    from ml.common import dataset_class_ids, load_dataset_cfg, names_tuple

    label_ids = dataset_class_ids(args.data)  # dataset label id -> canonical class (subset datasets, D15)
    if args.limit:
        images = sample_evenly(images, args.limit)
    if args.device is None:
        from measure.depth_backends.base import pick_device

        args.device = pick_device("cuda:0", 1.0)
    seg = load_segmenter(weights, device=args.device, conf=args.conf, imgsz=args.imgsz, max_det=args.max_det)
    print(f"eval {rel(weights)} on {len(images)} {args.split} images, device {args.device}", flush=True)
    t0 = time.perf_counter()
    res = evaluate(seg, images, args.gt_boxes, conf=args.conf, label_ids=label_ids)
    run = {**run_meta(seg, weights, args, args.device), "metrics": res, "notes": list(args.note),
           "dataset": {"config": rel(Path(args.data)), "classes": list(names_tuple(load_dataset_cfg(args.data)["names"]))}}
    if seg.task != "segment":
        run["notes"].append("detection-only weights: mask metrics are n/a")
    if args.ultralytics_val and weights.suffix == ".pt":
        try:
            run["ultralytics_val"] = ultralytics_val(weights, args.split, args.imgsz, args.device,
                                                     data_cfg=args.data)
        except Exception as e:  # noqa: BLE001 - cross-check only
            run["ultralytics_val"] = {"status": "failed", "error": f"{type(e).__name__}: {e}"[:300]}
    seg.close()
    rep = load_report(Path(args.out))
    if args.compare:
        cw = resolve_repo_path(args.compare)
        seg2 = load_segmenter(cw, device=args.device, conf=args.conf, imgsz=args.imgsz, max_det=args.max_det)
        res2 = evaluate(seg2, images, args.gt_boxes, conf=args.conf, label_ids=label_ids)
        a2 = argparse.Namespace(**{**vars(args), "stage": args.compare_stage or f"{args.stage}_{cw.suffix[1:]}"})
        run2 = {**run_meta(seg2, cw, a2, args.device), "metrics": res2, "notes": list(args.note),
                "parity": {"reference": rel(weights), **parity(res, res2)}}
        seg_a = load_segmenter(weights, device=args.device, imgsz=args.imgsz)
        run2["parity"]["agreement_at_conf_025"] = detection_agreement(seg_a, seg2, images[: min(len(images), 50)])
        seg_a.close()
        seg2.close()
        upsert_run(rep, run)
        upsert_run(rep, run2)
        print(json.dumps(run2["parity"], indent=1))
    else:
        upsert_run(rep, run)
    write_json(args.out, rep)
    print(json.dumps({"pothole": res["pothole"], "capable": res["capable_classes"], "mask": res.get("mask"),
                      "timing": res["timing"]}, indent=1))
    print(f"wrote {rel(args.out)} in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    raise SystemExit(main())
