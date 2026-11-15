"""Multi-frame fusion: track keyframes -> one PassMeasurement (PLAN 4.8 + PassMeasurement docstring).

For each metric m in {depth_p95, depth_max, depth_mean, area, volume, length, width}:
  1. keep frames with q >= min_quality_for_fusion (0.2) and not rejected;
  2. M = median(m_j); MAD = median|m_j - M|; reject |m_j - M| > max(3 * 1.4826 * MAD, 3 * sigma_j);
  3. fused = sum(w_j m_j) / sum(w_j), w_j = q_j / sigma_j^2;
  4. fused sigma = sqrt(1 / sum(1 / sigma_j^2)), floored at 3 mm (depth) and 5 % (area, volume, L, W).
The IMU depth hint (``ImuResult.imu_depth_mm``, sigma 25 mm) joins depth_p95 only when at least one camera
frame supplied a depth (PLAN 4.9: never the sole source for a severity class).
Water: the pass is water-filled if the q-weighted majority of usable frames is; depth/volume are then None.
ground_xz / range are taken from the best-quality frame (each frame has its own road frame, so averaging
ground_xz across frames would be meaningless); lane_offset is the q-weighted mean.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from measure.config import ensure_cfg
from measure.types import DepthQuality, FrameMeasurement, ImuResult, Method, PassMeasurement

_METRICS = {
    # name: (value attr, sigma attr or callable, floor kind)
    "depth_p95_mm": ("depth_p95_mm", "depth_sigma_mm", "depth"),
    "depth_max_mm": ("depth_max_mm", "_depth_max_sigma", "depth"),
    "depth_mean_mm": ("depth_mean_mm", "_depth_mean_sigma", "depth"),
    "area_m2": ("area_m2", "area_sigma", "rel"),
    "volume_l": ("volume_l", "volume_sigma", "rel"),
    "length_m": ("length_m", "length_sigma_m", "rel"),
    "width_m": ("width_m", "width_sigma_m", "rel"),
}


def _sigma(f: FrameMeasurement, name: str) -> Optional[float]:
    if name == "_depth_max_sigma":
        v = f.extras.get("depth_max_sigma_mm")
        return float(v) if v is not None else f.depth_sigma_mm
    if name == "_depth_mean_sigma":
        if f.volume_sigma is not None and f.area_m2 and f.area_m2 > 0:
            return float(f.volume_sigma / f.area_m2)  # litres / m^2 = mm
        return f.depth_sigma_mm
    v = getattr(f, name)
    return float(v) if v is not None else None


def fuse_metric(values: Sequence[float], sigmas: Sequence[float], qualities: Sequence[float], *,
                mad_k: float = 3.0, sigma_k: float = 3.0, floor: float = 0.0) -> tuple[Optional[float], Optional[float], list[int]]:
    """Weighted mean with MAD outlier rejection. Returns (fused, sigma, kept indices)."""
    v = np.asarray(values, float)
    s = np.asarray(sigmas, float)
    q = np.asarray(qualities, float)
    ok = np.isfinite(v) & np.isfinite(s) & (q > 0)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return None, None, []
    s = np.where(s > 0, s, max(floor, 1e-9))
    vv = v[idx]
    M = float(np.median(vv))
    mad = float(np.median(np.abs(vv - M)))
    thr = np.maximum(mad_k * 1.4826 * mad, sigma_k * s[idx])
    keep = idx[np.abs(vv - M) <= thr]
    if len(keep) == 0:
        keep = idx
    w = q[keep] / s[keep] ** 2
    fused = float(np.sum(w * v[keep]) / np.sum(w))
    sig = float(math.sqrt(1.0 / np.sum(1.0 / s[keep] ** 2)))
    return fused, max(sig, floor), [int(i) for i in keep]


def fuse_frames(frames: Sequence[FrameMeasurement], imu: Optional[ImuResult] = None,
                cfg: Optional[dict[str, Any]] = None, *, trip_id: Optional[str] = None,
                pothole_id: Optional[int] = None) -> PassMeasurement:
    cfg = ensure_cfg(cfg)
    fc, qmin = cfg["fusion"], float(cfg["quality"]["min_quality_for_fusion"])
    dfloor, rfloor = float(fc["depth_sigma_floor_mm"]), float(fc["rel_sigma_floor"])
    total = len(frames)
    usable = [f for f in frames if f.quality >= qmin and f.reject_reason is None]
    base: dict[str, Any] = dict(n_frames_total=total, trip_id=trip_id, pothole_id=pothole_id)
    if not usable:
        return PassMeasurement(n_frames=0, quality=0.0, depth_quality=DepthQuality.rejected,
                               reject_reason="no_frame_passed_quality_gate", **base)
    q = np.array([f.quality for f in usable])
    water_w = float(np.sum(q * np.array([f.water_filled for f in usable], float)))
    water = water_w > 0.5 * float(q.sum())
    out: dict[str, Any] = {}
    kept_by_metric: dict[str, list[int]] = {}
    for name, (attr, sattr, kind) in _METRICS.items():
        if water and name in ("depth_p95_mm", "depth_max_mm", "depth_mean_mm", "volume_l"):
            continue
        cand = [(i, getattr(f, attr), _sigma(f, sattr), f.quality) for i, f in enumerate(usable)]
        cand = [c for c in cand if c[1] is not None and c[2] is not None]
        if not cand:
            continue
        vals = [c[1] for c in cand]
        sigs = [c[2] for c in cand]
        qs = [c[3] for c in cand]
        if name == "depth_p95_mm" and imu is not None and imu.imu_depth_mm is not None:
            vals.append(float(imu.imu_depth_mm))
            sigs.append(float(imu.imu_depth_sigma_mm))
            qs.append(1.0)
        floor = dfloor if kind == "depth" else 0.0
        fused, sig, keep = fuse_metric(vals, sigs, qs, mad_k=fc["mad_k"], sigma_k=fc["sigma_k"], floor=floor)
        if fused is None:
            continue
        if kind == "rel":
            sig = max(sig, rfloor * abs(fused))
        out[name] = (fused, sig)
        kept_by_metric[name] = [cand[k][0] for k in keep if k < len(cand)]
    imu_used = (imu is not None and imu.imu_depth_mm is not None and "depth_p95_mm" in out)
    primary = kept_by_metric.get("depth_p95_mm") or kept_by_metric.get("area_m2") or list(range(len(usable)))
    best = max(usable, key=lambda f: f.quality)
    lane = [(f.lane_offset_m, f.quality) for f in usable if f.lane_offset_m is not None]
    lane_off = (sum(v * w for v, w in lane) / sum(w for _, w in lane)) if lane else None
    methods = {f.method for f in usable}
    method = Method.mono_imu if imu_used and methods <= {Method.mono, Method.mono_imu} else best.method
    pm = PassMeasurement(
        depth_p95_mm=out.get("depth_p95_mm", (None, None))[0],
        depth_sigma_mm=out.get("depth_p95_mm", (None, None))[1],
        depth_max_mm=out.get("depth_max_mm", (None, None))[0],
        depth_mean_mm=out.get("depth_mean_mm", (None, None))[0],
        area_m2=out.get("area_m2", (0.0, 0.0))[0],
        area_sigma=out.get("area_m2", (0.0, 0.0))[1],
        volume_l=out.get("volume_l", (None, None))[0],
        volume_sigma=out.get("volume_l", (None, None))[1],
        length_m=out.get("length_m", (0.0, None))[0],
        length_sigma_m=out.get("length_m", (None, None))[1],
        width_m=out.get("width_m", (0.0, None))[0],
        width_sigma_m=out.get("width_m", (None, None))[1],
        water_filled=water,
        depth_quality=DepthQuality.unknown_water if water else DepthQuality.ok,
        quality=float(np.clip(np.max(q), 0, 1)),
        method=method,
        imu_corroborated=(imu.corroborated if imu is not None else None),
        imu_depth_mm=(imu.imu_depth_mm if imu is not None else None),
        n_frames=len(primary),
        range_m=best.range_m,
        lane_offset_m=lane_off,
        ground_xz=best.ground_xz,
        scale_factor=best.scale_factor,
        depth_model_version=best.depth_model_version,
        certified=all(f.certified for f in usable),
        frames_used=[usable[i].seq if usable[i].seq is not None else i for i in primary],
        **base,
    )
    return pm
