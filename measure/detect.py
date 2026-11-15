"""Detection + box->mask for the CLI / demo (the production path uses ml/segmenter.py masks).

* ``detect(img, weights)``: ultralytics model; seg models return masks directly; box models
  (e.g. weights/bootstrap_pothole_yolo26m.pt, class 'Pothole') return boxes only.
* ``box_to_mask(img, box, method)``: 'sam2' (facebook/sam2.1-hiera-small via transformers, box prompt;
  GPU only if >= 1.5 GB free, else CPU), 'grabcut' (cv2.grabCut initialised with the box), 'ellipse'
  (inscribed ellipse). 'auto' tries sam2 -> grabcut -> ellipse; the method used is always returned.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from measure.depth_backends.base import cuda_usable, pick_device

POTHOLE_NAMES = ("pothole", "d40", "pothole_water")
SAM2_ID = "facebook/sam2.1-hiera-small"
_MODELS: dict[str, Any] = {}


def _yolo(weights: str) -> Any:
    if weights not in _MODELS:
        from ultralytics import YOLO

        _MODELS[weights] = YOLO(weights)
    return _MODELS[weights]


def detect(img_bgr: np.ndarray, weights: str, conf: float = 0.25, device: Optional[str] = None,
           names: tuple[str, ...] = POTHOLE_NAMES, imgsz: int = 640) -> list[dict[str, Any]]:
    """Return [{bbox:[x1,y1,x2,y2], conf, cls, mask?}] for pothole classes only, sorted by conf."""
    model = _yolo(weights)
    dev = device or pick_device("cuda:0", 1.0)
    t0 = time.perf_counter()
    res = model.predict(img_bgr, conf=conf, device=dev, imgsz=imgsz, verbose=False)[0]
    dt = (time.perf_counter() - t0) * 1000.0
    out = []
    H, W = img_bgr.shape[:2]
    masks = None
    if getattr(res, "masks", None) is not None and res.masks is not None:
        import cv2

        mk = res.masks.data.cpu().numpy()
        masks = [cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5 for m in mk]
    for i, b in enumerate(res.boxes):
        cname = str(model.names[int(b.cls)]).lower()
        if names and cname not in names:
            continue
        d = {"bbox": [float(v) for v in b.xyxy[0].tolist()], "conf": float(b.conf), "cls": cname,
             "detector": weights, "detect_ms": dt, "device": dev}
        if masks is not None:
            d["mask"] = masks[i]
            d["mask_method"] = "detector_seg"
        out.append(d)
    return sorted(out, key=lambda d: -d["conf"])


def _sam2(device: str) -> tuple[Any, Any]:
    key = f"sam2:{device}"
    if key not in _MODELS:
        from transformers import Sam2Model, Sam2Processor

        _MODELS[key] = (Sam2Processor.from_pretrained(SAM2_ID), Sam2Model.from_pretrained(SAM2_ID).to(device).eval())
    return _MODELS[key]


def sam2_box_mask(img_bgr: np.ndarray, box: list[float], device: Optional[str] = None) -> tuple[np.ndarray, float]:
    import torch

    dev = device or pick_device("cuda", 1.5)
    proc, model = _sam2(dev)
    rgb = np.ascontiguousarray(img_bgr[..., ::-1])
    inputs = proc(images=rgb, input_boxes=[[list(map(float, box))]], return_tensors="pt").to(dev)
    with torch.inference_mode():
        out = model(**inputs, multimask_output=True)
    masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]  # (1, 3, H, W)
    scores = out.iou_scores.cpu().numpy().reshape(-1)
    m = masks.numpy().reshape(-1, *masks.shape[-2:])
    # prefer the best-scoring mask that stays inside the box (the box is the detector's extent)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    inside = np.zeros(m.shape[-2:], bool)
    inside[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = True
    best, best_s = None, -1.0
    for k in range(len(m)):
        mk = m[k] > 0
        if mk.sum() == 0:
            continue
        frac_in = (mk & inside).sum() / mk.sum()
        sc = float(scores[k]) * (1.0 if frac_in > 0.9 else 0.5)
        if sc > best_s:
            best, best_s = mk & inside, sc
    if best is None:
        raise RuntimeError("SAM2 produced no mask")
    return best, best_s


def grabcut_mask(img_bgr: np.ndarray, box: list[float], iters: int = 5) -> np.ndarray:
    import cv2

    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
    m = np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, m, (x1, y1, max(2, x2 - x1), max(2, y2 - y1)), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    out = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
    if n > 1:
        out = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    return out


def ellipse_mask(shape: tuple[int, int], box: list[float]) -> np.ndarray:
    import cv2

    m = np.zeros(shape[:2], np.uint8)
    x1, y1, x2, y2 = box
    cv2.ellipse(m, (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))),
                (max(1, int(round((x2 - x1) / 2))), max(1, int(round((y2 - y1) / 2)))), 0, 0, 360, 1, -1)
    return m.astype(bool)


def box_to_mask(img_bgr: np.ndarray, box: list[float], method: str = "auto",
                min_frac: float = 0.15) -> tuple[np.ndarray, str]:
    """Returns (mask, method_used). A mask covering < ``min_frac`` of its box is treated as a failure."""
    x1, y1, x2, y2 = box
    box_area = max(1.0, (x2 - x1) * (y2 - y1))
    order = {"auto": ["sam2", "grabcut", "ellipse"], "sam2": ["sam2", "ellipse"],
             "grabcut": ["grabcut", "ellipse"], "ellipse": ["ellipse"]}[method]
    errors = []
    for m in order:
        try:
            if m == "sam2":
                mk, _ = sam2_box_mask(img_bgr, box)
            elif m == "grabcut":
                mk = grabcut_mask(img_bgr, box)
            else:
                mk = ellipse_mask(img_bgr.shape, box)
            if mk.sum() >= min_frac * box_area or m == "ellipse":
                return mk, m if not errors else f"{m} (after {'; '.join(errors)})"
            errors.append(f"{m}: mask {mk.sum() / box_area:.0%} of box")
        except Exception as e:  # noqa: BLE001 - fall through to the next method, recorded
            errors.append(f"{m}: {type(e).__name__}: {e}")
    raise RuntimeError("; ".join(errors))


def polygon_to_mask(poly_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly_xy, float)).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def release() -> None:
    _MODELS.clear()
    try:
        import torch

        if cuda_usable():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
