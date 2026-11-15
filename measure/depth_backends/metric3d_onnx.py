"""Metric3D v2 via ONNX Runtime (optional backend; onnxruntime-gpu is installed).

The ONNX file is not shipped. Export or download a Metric3D-v2 ViT-S/L ONNX (input 1x3x616x1064 RGB,
ImageNet mean/std in 0-255 units) and point ``depth.metric3d_onnx_path`` (code default) or the env var
``METRIC3D_ONNX`` at it. Metric3D predicts depth for a *canonical camera* with focal 1000 px; metric depth
is ``Z = Z_canonical * fx_input / 1000`` where fx_input is the focal at the network input resolution.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import numpy as np

from measure.depth_backends.base import DepthBackend
from measure.types import Method

INPUT_HW = (616, 1064)
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


class Metric3DOnnxBackend(DepthBackend):
    name = "metric3d_onnx"
    method = Method.mono

    def __init__(self, dcfg: dict[str, Any], path: Optional[str] = None) -> None:
        super().__init__()
        self.path = path or dcfg.get("metric3d_onnx_path") or os.environ.get("METRIC3D_ONNX")
        if not self.path or not os.path.isfile(self.path):
            raise NotImplementedError(
                "Metric3D ONNX backend needs a model file: set depth.metric3d_onnx_path or METRIC3D_ONNX to a "
                "Metric3D-v2 ONNX export (input 1x3x616x1064). Default backend da2_metric needs nothing extra.")
        import onnxruntime as ort

        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(self.path, providers=prov)
        self.inp = self.sess.get_inputs()[0].name

    @property
    def version(self) -> str:
        return f"metric3d_onnx:{os.path.basename(self.path)}"

    @staticmethod
    def preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
        import cv2

        H, W = img_bgr.shape[:2]
        sc = min(INPUT_HW[0] / H, INPUT_HW[1] / W)
        h, w = int(round(H * sc)), int(round(W * sc))
        rgb = cv2.resize(img_bgr[..., ::-1], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        ph, pw = INPUT_HW[0] - h, INPUT_HW[1] - w
        top, left = ph // 2, pw // 2
        pad = np.empty((INPUT_HW[0], INPUT_HW[1], 3), np.float32)
        pad[:] = MEAN
        pad[top:top + h, left:left + w] = rgb
        x = ((pad - MEAN) / STD).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(x), sc, (top, left, h, w)

    def predict(self, img_bgr: np.ndarray, fx_px: float) -> tuple[np.ndarray, Optional[np.ndarray]]:  # pragma: no cover
        import cv2

        H, W = img_bgr.shape[:2]
        x, sc, (top, left, h, w) = self.preprocess(img_bgr)
        t0 = time.perf_counter()
        out = self.sess.run(None, {self.inp: x})
        d = np.asarray(out[0], np.float32).reshape(INPUT_HW)[top:top + h, left:left + w]
        d = d * (fx_px * sc) / 1000.0
        self.last_timing_ms["forward_ms"] = (time.perf_counter() - t0) * 1000.0
        return cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR), None
