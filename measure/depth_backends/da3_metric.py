"""Depth-Anything-3 METRIC backend (optional, DECISIONS D4).

Needs the DA3 git package, which is NOT installed in .venv (it would otherwise risk pulling torch).
Install without touching torch:

    .venv/bin/python -m pip install --no-deps "git+https://github.com/ByteDance-Seed/Depth-Anything-3"
    # then install any missing pure-python deps it reports, one by one, again with --no-deps

Scale conventions (``depth.scale_convention`` in configs/measure.yaml):
  * ``fx_over_300`` (DA3METRIC README):  Z = fx' * net / 300, fx' = focal at the inference resolution
  * ``fx``:                              Z = fx' * net
  * ``relative``:                        Z = net (scale fixed by the height lock)
Validate once on the staged target by minimising |h_hat - h_true| before the scale lock (PLAN 4.2).
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from measure.depth_backends.base import DepthBackend, pick_device
from measure.types import Method

INSTALL_HINT = (
    "DA3 backend needs the Depth-Anything-3 package: "
    ".venv/bin/python -m pip install --no-deps 'git+https://github.com/ByteDance-Seed/Depth-Anything-3' "
    "(never reinstall torch). Or set depth.backend: da2_metric in configs/measure.yaml (default)."
)


class DA3MetricBackend(DepthBackend):
    name = "da3_metric"
    method = Method.mono

    def __init__(self, dcfg: dict[str, Any], device: Optional[str] = None) -> None:
        super().__init__()
        try:
            from depth_anything_3.api import DepthAnything3  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - depends on optional package
            raise NotImplementedError(INSTALL_HINT) from e
        self._cls = DepthAnything3
        self.cfg = dcfg
        self.model_id = dcfg.get("da3_model_id", "depth-anything/DA3METRIC-LARGE")
        self.convention = dcfg.get("scale_convention", "fx_over_300")
        self.long_side = int(dcfg.get("infer_long_side", 1036))
        self.device = pick_device(device or dcfg.get("device", "cuda"), float(dcfg.get("min_free_gpu_gb", 2.0)),
                                  bool(dcfg.get("cpu_fallback", True)))
        self._model = None

    @property
    def version(self) -> str:
        return f"{self.model_id}@{self.convention}"

    def predict(self, img_bgr: np.ndarray, fx_px: float) -> tuple[np.ndarray, Optional[np.ndarray]]:  # pragma: no cover
        import cv2
        import torch

        if self._model is None:
            self._model = self._cls.from_pretrained(self.model_id).to(self.device).eval()
        H, W = img_bgr.shape[:2]
        sc = self.long_side / max(H, W)
        h14, w14 = max(14, int(round(H * sc / 14)) * 14), max(14, int(round(W * sc / 14)) * 14)
        rgb = cv2.resize(img_bgr[..., ::-1], (w14, h14), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        with torch.inference_mode():
            pred = self._model.inference([np.ascontiguousarray(rgb)])
        net = np.asarray(pred.depth[0], np.float32)
        conf = np.asarray(pred.conf[0], np.float32) if getattr(pred, "conf", None) is not None else None
        fx_inf = fx_px * (w14 / W)
        if self.convention == "fx_over_300":
            z = fx_inf * net / 300.0
        elif self.convention == "fx":
            z = fx_inf * net
        else:
            z = net
        z = cv2.resize(z, (W, H), interpolation=cv2.INTER_LINEAR)
        if conf is not None:
            conf = cv2.resize(conf, (W, H), interpolation=cv2.INTER_LINEAR)
            conf = conf / max(float(conf.max()), 1e-6)
        self.last_timing_ms.update(forward_ms=(time.perf_counter() - t0) * 1000.0, device=self.device)
        return z, conf
