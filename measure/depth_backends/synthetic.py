"""Synthetic backend: returns a rendered (optionally noisy) depth map -- for tests and the demo."""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import numpy as np

from measure.depth_backends.base import DepthBackend
from measure.types import Method

DepthSource = Union[np.ndarray, Callable[[np.ndarray, float], np.ndarray]]


class SyntheticBackend(DepthBackend):
    name = "synthetic"
    method = Method.synthetic

    def __init__(self, dcfg: Optional[dict[str, Any]] = None, depth: Optional[DepthSource] = None,
                 conf: Optional[np.ndarray] = None, tag: str = "exact") -> None:
        super().__init__()
        self.depth = depth
        self.conf = conf
        self.tag = tag

    @property
    def version(self) -> str:
        return f"synthetic-{self.tag}"

    def set_depth(self, depth: DepthSource, conf: Optional[np.ndarray] = None) -> None:
        self.depth, self.conf = depth, conf

    def predict(self, img_bgr: np.ndarray, fx_px: float) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self.depth is None:
            raise RuntimeError("SyntheticBackend has no depth source; call set_depth() first")
        d = self.depth(img_bgr, fx_px) if callable(self.depth) else self.depth
        if d.shape[:2] != img_bgr.shape[:2]:
            raise ValueError(f"synthetic depth {d.shape} does not match image {img_bgr.shape[:2]}")
        return np.asarray(d, np.float64), self.conf
