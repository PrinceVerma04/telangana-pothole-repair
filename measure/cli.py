"""RoadFix measurement CLI.

  python -m measure.cli run  --image X.jpg --calib configs/cameras/phone_demo.json \\
                             [--mask m.png | --detect models/training_runs/seg/y26m_seg_india_v2ft_geo/weights/best_pothole.pt] --json out.json
  python -m measure.cli demo --out reports/measure_demo

``run`` writes a JSON with the ``server.app.schemas.api.MeasureResponse`` shape (measurements,
severities, depth_model_version, calib_id, certified, elapsed_ms, overlay_url).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _load_masks(path: str, shape: tuple[int, int], split: bool) -> list[np.ndarray]:
    import cv2

    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise SystemExit(f"cannot read mask {path}")
    if m.ndim == 3:
        m = m.max(axis=2)
    if m.shape[:2] != shape:
        raise SystemExit(f"mask {m.shape[:2]} does not match image {shape}")
    labels = [v for v in np.unique(m) if v != 0]
    if 1 < len(labels) < 64:
        return [m == v for v in labels]  # instance-id PNG (0 = background)
    mk = m > 0
    if split:
        n, lab = cv2.connectedComponents(mk.astype(np.uint8), connectivity=8)
        return [lab == i for i in range(1, n) if (lab == i).sum() > 0]
    return [mk]


def run_measure(image: str, calib: str, *, mask: Optional[str] = None, detect: Optional[str] = None,
                cfg_path: Optional[str] = None, overrides: Optional[dict[str, Any]] = None, conf: float = 0.25,
                max_det: int = 5, mask_method: str = "auto", roadseg: bool = True, split: bool = False,
                overlay: Optional[str] = None, backend_name: Optional[str] = None) -> dict[str, Any]:
    import cv2

    from measure import irc
    from measure.config import load_measure_cfg
    from measure.detect import box_to_mask
    from measure.detect import detect as run_detect
    from measure.pipeline import measure_frames
    from measure.types import CalibCert

    t0 = time.perf_counter()
    cfg = load_measure_cfg(cfg_path, overrides)
    if backend_name:
        cfg["depth"]["backend"] = backend_name
    cert = CalibCert.load(calib)
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image {image}")
    H, W = img.shape[:2]
    masks: list[np.ndarray] = []
    confs: list[float] = []
    det_info: list[dict[str, Any]] = []
    if mask:
        masks = _load_masks(mask, (H, W), split)
        confs = [1.0] * len(masks)
        det_info = [{"source": "mask_file", "mask_method": "provided"} for _ in masks]
    elif detect:
        for d in run_detect(img, detect, conf=conf)[:max_det]:
            if "mask" in d:
                mk, how = d["mask"], d["mask_method"]
            else:
                mk, how = box_to_mask(img, d["bbox"], mask_method)
            if mk.sum() == 0:
                continue
            masks.append(mk)
            confs.append(d["conf"])
            det_info.append({"source": "detector", "bbox": d["bbox"], "det_conf": d["conf"], "cls": d["cls"],
                             "mask_method": how, "detect_ms": d.get("detect_ms")})
    else:
        raise SystemExit("give --mask or --detect")
    resp: dict[str, Any] = {"measurements": [], "severities": [], "depth_model_version": "none",
                            "calib_id": cert.calib_id, "certified": bool(cert.certified), "elapsed_ms": 0.0,
                            "overlay_url": None}
    if not masks:
        resp["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
        return resp
    notroad = None
    timings: dict[str, Any] = {}
    from measure.pipeline import size_only

    if roadseg and not size_only(cfg):  # the road mask only helps the depth-based plane fit
        try:
            from measure.roadseg import notroad_mask

            notroad = notroad_mask(img, timings=timings)
        except Exception as e:  # noqa: BLE001 - optional stage; recorded
            timings["roadseg_error"] = f"{type(e).__name__}: {e}"
    fms, ctx = measure_frames(img, masks, cert, cfg, mask_confs=confs, notroad_mask=notroad)
    out = []
    for fm, di in zip(fms, det_info, strict=False):
        ex = dict(fm.extras)
        ex.update(detection=di, roadseg=timings or None)
        out.append(fm.model_copy(update={"extras": ex}))
    resp["measurements"] = [f.to_dict() for f in out]
    resp["severities"] = [irc.severity(f.depth_p95_mm, f.width_m * 1000.0, f.water_filled).value for f in out]
    resp["depth_model_version"] = out[0].depth_model_version
    if overlay:
        from measure.overlay import render_overlay

        render_overlay(ctx.img, ctx.masks, out, ctx.depth, overlay, maps=ctx.maps, title=Path(image).name)
        resp["overlay_url"] = str(overlay)
    resp["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    return resp


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m measure.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure potholes in one image")
    r.add_argument("--image", required=True)
    r.add_argument("--calib", required=True, help="calibration certificate JSON (configs/cameras/*.json)")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--mask", help="PNG mask: nonzero = pothole, or instance ids")
    g.add_argument("--detect", help="ultralytics weights (.pt); box models get masks via --mask-method")
    r.add_argument("--mask-method", default="auto", choices=["auto", "sam2", "grabcut", "ellipse"])
    r.add_argument("--split", action="store_true", help="split a binary mask into connected components")
    r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--max-det", type=int, default=5)
    r.add_argument("--cfg", help="measure.yaml (default configs/measure.yaml)")
    r.add_argument("--backend", help="override depth.backend (off = size only, D27 | da2_metric | da3_metric | ...)")
    r.add_argument("--no-roadseg", action="store_true", help="skip SegFormer road mask for the plane fit")
    r.add_argument("--json", required=True, help="output JSON path")
    r.add_argument("--overlay", help="optional overlay PNG path")
    d = sub.add_parser("demo", help="synthetic validation + real-image demo -> reports/measure_demo")
    d.add_argument("--out", default="reports/measure_demo")
    d.add_argument("--images", default="datasets/extracted/measure_demo_imgs")
    d.add_argument("--n-images", type=int, default=5)
    d.add_argument("--weights", default=None, help="segmenter weights (default: ml.common.DEFAULT_SEG_WEIGHTS)")
    d.add_argument("--trials", type=int, default=200)
    d.add_argument("--skip-real", action="store_true")
    d.add_argument("--skip-synthetic", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        resp = run_measure(a.image, a.calib, mask=a.mask, detect=a.detect, cfg_path=a.cfg, conf=a.conf,
                           max_det=a.max_det, mask_method=a.mask_method, roadseg=not a.no_roadseg, split=a.split,
                           overlay=a.overlay, backend_name=a.backend)
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(resp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for m, s in zip(resp["measurements"], resp["severities"], strict=False):
            print(f"{s:14s} depth_p95={m['depth_p95_mm']} mm  area={m['area_m2']:.3f} m2  volume={m['volume_l']} L  "
                  f"q={m['quality']:.2f}  reject={m['reject_reason']}")
        print(f"wrote {a.json} ({len(resp['measurements'])} measurements, certified={resp['certified']}, "
              f"{resp['elapsed_ms']:.0f} ms)")
        return 0
    from measure.demo import run_demo

    run_demo(a.out, images_dir=a.images, n_images=a.n_images, weights=a.weights, trials=a.trials,
             skip_real=a.skip_real, skip_synthetic=a.skip_synthetic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
