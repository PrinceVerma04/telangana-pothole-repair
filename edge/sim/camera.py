"""Pinhole camera over a flat road, from a calibration certificate (``configs/cameras/*.json``, ``CalibCert``).

Conventions are those of ``measure/`` (``measure.camera.world_to_camera_rotation``): road frame X right, Y up,
Z forward, origin on the road directly below the camera; the camera sits at height ``h_true_m`` and looks forward
pitched down by ``pitch_deg``; image x right, y down. Lens distortion is ignored (the simulated camera is the
ideal one of the certificate: ``dist`` must be zero) and roll must be zero.

For a road point (X, 0, Z):  x_c = X,  y_c = h cos p - Z sin p,  z_c = h sin p + Z cos p,  u = fx x_c/z_c + cx,
v = fy y_c/z_c + cy. The road plane maps to the image by the homography ``G = K @ M`` with
``M = [[1, 0, 0], [0, -sin p, h cos p], [0, cos p, h sin p]]`` acting on (X, Z, 1).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMERAS_DIR = PROJECT_ROOT / "configs" / "cameras"


@dataclass(frozen=True)
class SimCamera:
    calib_id: str
    fx: float
    fy: float
    cx: float
    cy: float
    W: int
    H: int
    h: float
    pitch_deg: float
    gnss_latency_s: float = 0.3
    certified: bool = False

    @classmethod
    def from_cert_dict(cls, d: dict[str, Any]) -> "SimCamera":
        if any(abs(float(x)) > 1e-12 for x in (d.get("dist") or [])):
            raise ValueError(f"{d.get('calib_id')}: the simulator camera has no lens distortion")
        if abs(float(d.get("roll_deg") or 0.0)) > 1e-9:
            raise ValueError(f"{d.get('calib_id')}: roll must be 0 for the simulator camera")
        return cls(calib_id=str(d["calib_id"]), fx=float(d["fx"]), fy=float(d["fy"]), cx=float(d["cx"]),
                   cy=float(d["cy"]), W=int(d["img_w"]), H=int(d["img_h"]), h=float(d["h_true_m"]),
                   pitch_deg=float(d["pitch_deg"]), gnss_latency_s=float(d.get("gnss_latency_s") or 0.3),
                   certified=bool(d.get("certified", False)))

    @classmethod
    def load(cls, calib_id_or_path: str | Path = "sim_bus") -> "SimCamera":
        p = Path(calib_id_or_path)
        if not p.suffix:
            p = CAMERAS_DIR / f"{calib_id_or_path}.json"
        return cls.from_cert_dict(json.loads(p.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ geometry
    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def M(self) -> np.ndarray:
        p = math.radians(self.pitch_deg)
        s, c = math.sin(p), math.cos(p)
        return np.array([[1.0, 0.0, 0.0], [0.0, -s, self.h * c], [0.0, c, self.h * s]])

    @property
    def G(self) -> np.ndarray:
        """Road plane (X, Z, 1) -> homogeneous pixel."""
        return self.K @ self.M

    @property
    def horizon_v(self) -> float:
        return self.cy - self.fy * math.tan(math.radians(self.pitch_deg))

    def ground_to_pixel(self, X: Any, Z: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(u, v, z_cam) for road points; z_cam <= 0 means behind the camera."""
        X = np.asarray(X, float)
        Z = np.asarray(Z, float)
        p = math.radians(self.pitch_deg)
        yc = self.h * math.cos(p) - Z * math.sin(p)
        zc = self.h * math.sin(p) + Z * math.cos(p)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * X / zc + self.cx
            v = self.fy * yc / zc + self.cy
        return u, v, zc

    def pixel_to_ground(self, u: Any, v: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(X, Z, ok) where the pixel ray meets the road; ok False above the horizon."""
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        p = math.radians(self.pitch_deg)
        s, c = math.sin(p), math.cos(p)
        xn = (u - self.cx) / self.fx
        yn = (v - self.cy) / self.fy
        # camera ray d_c = (xn, yn, 1); in road frame: X = xn, Y = -yn c - s, Z = -yn s + c
        dy = -yn * c - s
        dz = -yn * s + c
        ok = dy < -1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(ok, -self.h / dy, np.nan)
        return xn * t, dz * t, ok

    def footprint_pixels(self, pts_xz: Sequence[Sequence[float]]) -> Optional[np.ndarray]:
        """Project a road polygon [(X, Z), ...] to pixels (N, 2); None if any vertex is behind the camera."""
        a = np.asarray(pts_xz, float)
        u, v, zc = self.ground_to_pixel(a[:, 0], a[:, 1])
        if np.any(zc <= 0.05) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            return None
        return np.stack([u, v], axis=1)

    def in_view(self, px: np.ndarray, margin_px: float = 2.0) -> bool:
        return bool(np.all((px[:, 0] >= margin_px) & (px[:, 0] <= self.W - 1 - margin_px)
                           & (px[:, 1] >= margin_px) & (px[:, 1] <= self.H - 1 - margin_px)))
