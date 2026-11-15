"""Optional road segmentation for the plane-fit candidates (PLAN 4.4.1 'minus road-seg not-road pixels').

SegFormer fine-tuned on Cityscapes (``nvidia/segformer-b2-finetuned-cityscapes-1024-1024``, b0 on CPU).
Cityscapes train id 0 = road. Everything else (sidewalk, vehicles incl. the ego hood, buildings, sky,
people) becomes ``notroad``. A pothole is usually labelled road or 'terrain'; detection masks are
excluded separately, so that does not matter here.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from measure.depth_backends.base import pick_device

MODEL_GPU = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
MODEL_CPU = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
ROAD_IDS = (0,)
_CACHE: dict[str, Any] = {}


def _load(device: str) -> tuple[Any, Any, str]:
    mid = MODEL_GPU if device.startswith("cuda") else MODEL_CPU
    key = f"{mid}:{device}"
    if key not in _CACHE:
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        proc = AutoImageProcessor.from_pretrained(mid)
        model = SegformerForSemanticSegmentation.from_pretrained(mid).to(device).eval()
        _CACHE[key] = (proc, model)
    p, m = _CACHE[key]
    return p, m, mid


def road_mask(img_bgr: np.ndarray, device: Optional[str] = None, timings: Optional[dict] = None) -> np.ndarray:
    """Boolean (H, W) road mask."""
    import torch

    dev = device or pick_device("cuda", 1.0)
    proc, model, mid = _load(dev)
    H, W = img_bgr.shape[:2]
    t0 = time.perf_counter()
    inputs = proc(images=np.ascontiguousarray(img_bgr[..., ::-1]), return_tensors="pt").to(dev)
    with torch.inference_mode():
        logits = model(**inputs).logits
        up = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        lab = up.argmax(1)[0].cpu().numpy()
    if timings is not None:
        timings.update(roadseg_ms=(time.perf_counter() - t0) * 1000.0, roadseg_model=mid, roadseg_device=dev)
    return np.isin(lab, ROAD_IDS)


def notroad_mask(img_bgr: np.ndarray, device: Optional[str] = None, timings: Optional[dict] = None,
                 erode_px: int = 5) -> np.ndarray:
    """Pixels NOT usable for the road plane (road mask eroded a little to drop kerb/vehicle edges)."""
    import cv2

    rm = road_mask(img_bgr, device, timings)
    if erode_px > 0:
        rm = cv2.erode(rm.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=erode_px).astype(bool)
    return ~rm


def release() -> None:
    _CACHE.clear()
