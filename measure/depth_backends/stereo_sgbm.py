"""Tier-B stereo via OpenCV semi-global block matching (PLAN 4.11).

Input must be a *rectified* pair. ``predict(img_bgr, fx)`` accepts either a side-by-side frame
(left | right, width 2W) or a left frame plus ``right_bgr=`` / ``set_right()``. Output metric depth
Z = fx * B / disparity; conf = 1 where the disparity is valid and passes a left-right check, else 0.
Steps 4.4-4.8 still run on stereo depth (tilt removal; the scale lock should give s in [0.9, 1.1],
else the rig needs recalibration -- pipeline flags it).
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from measure.depth_backends.base import DepthBackend
from measure.types import Method


class StereoSGBMBackend(DepthBackend):
    name = "stereo_sgbm"
    method = Method.stereo

    def __init__(self, dcfg: Optional[dict[str, Any]] = None, baseline_m: Optional[float] = None,
                 num_disp: int = 128, block: int = 5, lr_check_px: float = 1.5) -> None:
        super().__init__()
        dcfg = dcfg or {}
        self.baseline = float(baseline_m if baseline_m is not None else dcfg.get("stereo_baseline_m", 0.12))
        self.num_disp = int(np.ceil(num_disp / 16) * 16)
        self.block = int(block)
        self.lr_check_px = float(lr_check_px)
        self._right: Optional[np.ndarray] = None

    @property
    def version(self) -> str:
        return f"sgbm_B{self.baseline:.3f}_nd{self.num_disp}_b{self.block}"

    def set_right(self, right_bgr: np.ndarray) -> None:
        self._right = right_bgr

    def disparity(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
        gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
        b = self.block
        kw = dict(minDisparity=0, numDisparities=self.num_disp, blockSize=b, P1=8 * b * b, P2=32 * b * b,
                  disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                  mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        mL = cv2.StereoSGBM_create(**kw)
        dl = mL.compute(gl, gr).astype(np.float32) / 16.0
        # right-reference disparity via horizontal flip trick
        dr = mL.compute(gr[:, ::-1].copy(), gl[:, ::-1].copy()).astype(np.float32)[:, ::-1] / 16.0
        H, W = dl.shape
        valid = dl > 0.5
        u = np.arange(W)[None, :].repeat(H, 0)
        ur = np.clip(np.round(u - dl).astype(int), 0, W - 1)
        dr_at = dr[np.arange(H)[:, None], ur]
        valid &= np.abs(dl - dr_at) <= self.lr_check_px
        return dl, valid

    def predict(self, img_bgr: np.ndarray, fx_px: float, right_bgr: Optional[np.ndarray] = None
                ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        right = right_bgr if right_bgr is not None else self._right
        left = img_bgr
        if right is None:
            if img_bgr.shape[1] % 2:
                raise ValueError("stereo_sgbm needs a right image (set_right / right_bgr) or a side-by-side frame")
            W2 = img_bgr.shape[1] // 2
            left, right = img_bgr[:, :W2], img_bgr[:, W2:]
        t0 = time.perf_counter()
        disp, valid = self.disparity(left, right)
        with np.errstate(divide="ignore"):
            z = np.where(valid, fx_px * self.baseline / np.maximum(disp, 1e-6), np.nan).astype(np.float32)
        self.last_timing_ms["forward_ms"] = (time.perf_counter() - t0) * 1000.0
        return z, valid.astype(np.float32)
