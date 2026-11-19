"""measure_frame(img, mask, cert, cfg) -> FrameMeasurement  (used by the GPU worker and the CLI).

Chain (PLAN 4.1-4.7): match intrinsics to the image -> undistort frame + masks -> depth backend ->
back-project -> RANSAC road plane + height scale-lock (once per frame, all detections excluded) ->
per mask: local quadratic ring fit -> mask integral (depth, area, volume, L/W, sigma, quality, water).
Pure library: no DB / server imports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from measure import irc
from measure.calib import has_distortion, undistort_image, undistort_mask
from measure.camera import backproject, cert_for_image
from measure.config import ensure_cfg
from measure.depth_backends.base import DepthBackend, get_backend
from measure.pothole_geom import measure_mask
from measure.surface import fit_road_surface, with_local_quadratic
from measure.types import CalibCert, FrameMeasurement, Method, SurfaceFit


@dataclass
class FrameContext:
    """Everything computed once per frame (handy for overlays / debugging)."""

    depth: Optional[np.ndarray]  # raw backend depth (undistorted frame); None in size-only mode (D27)
    K: np.ndarray
    cert: CalibCert  # intrinsics matched to the image
    surface: SurfaceFit  # global plane fit (before per-mask quadratic)
    img: np.ndarray  # undistorted image
    masks: list[np.ndarray]
    timings_ms: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    maps: list[dict[str, Any]] = field(default_factory=list)


def measure_frames(img: np.ndarray, masks: Sequence[np.ndarray], cert: CalibCert, cfg: Optional[dict[str, Any]] = None,
                   *, backend: Optional[DepthBackend] = None, depth: Optional[np.ndarray] = None,
                   conf: Optional[np.ndarray] = None, mask_confs: Optional[Sequence[float]] = None,
                   verifier_classes: Optional[Sequence[Optional[str]]] = None,
                   notroad_mask: Optional[np.ndarray] = None, seq: Optional[int] = None,
                   t_frame_ms: Optional[int] = None, method: Optional[Method] = None,
                   depth_model_version: Optional[str] = None,
                   masks_are_raw: bool = True) -> tuple[list[FrameMeasurement], FrameContext]:
    """Measure every mask of one frame with a single depth pass and a single road-plane fit."""
    cfg = ensure_cfg(cfg)
    t0 = time.perf_counter()
    H, W = img.shape[:2]
    cert_i, info = cert_for_image(cert, W, H)
    masks = [np.asarray(m).astype(bool) for m in masks]
    for m in masks:
        if m.shape != (H, W):
            raise ValueError(f"mask shape {m.shape} != image {(H, W)}")
    # --- undistort (masks/polygons arrive in raw pixels: PLAN_CRITIQUE (c))
    K = cert_i.K_np()
    if cfg.get("undistort", True) and has_distortion(cert_i):
        img_u, K = undistort_image(img, cert_i)
        if masks_are_raw:
            masks = [undistort_mask(m, cert_i) for m in masks]
        info["undistorted"] = True
    else:
        img_u = img
        info["undistorted"] = False
    timings: dict[str, Any] = {}
    if depth is None and backend is None and size_only(cfg):
        return _measure_frames_ground(img_u, masks, K, cert_i, cfg, info, timings, t0, mask_confs, verifier_classes,
                                      seq, t_frame_ms)
    # --- depth
    if depth is None:
        backend = backend or get_backend(cfg)
        td = time.perf_counter()
        depth, conf_b = backend.predict(img_u, float(K[0, 0]))
        conf = conf if conf is not None else conf_b
        timings["depth_ms"] = (time.perf_counter() - td) * 1000.0
        timings.update({f"depth_{k}": v for k, v in backend.last_timing_ms.items()})
        method = method or backend.method
        depth_model_version = depth_model_version or backend.version
    method = method or Method.mono
    depth_model_version = depth_model_version or "external"
    depth = np.asarray(depth, np.float64)
    # --- plane + scale lock
    ts = time.perf_counter()
    pts = backproject(depth, K)
    excl = np.zeros((H, W), bool)
    for m in masks:
        excl |= m
    surf = fit_road_surface(pts, None, cfg, h_true=cert_i.h_true_m, exclude_mask=excl,
                            pitch_prior_deg=cert_i.pitch_deg, yaw_deg=cert_i.yaw_deg, certified=cert_i.certified,
                            notroad_mask=notroad_mask)
    timings["surface_ms"] = (time.perf_counter() - ts) * 1000.0
    if method == Method.stereo and surf.ok and not (0.9 <= surf.s <= 1.1):
        surf.extras.setdefault("warnings", []).append(f"stereo scale {surf.s:.3f} outside [0.9, 1.1]: recalibrate rig")
    ctx = FrameContext(depth=depth, K=K, cert=cert_i, surface=surf, img=img_u, masks=masks, timings_ms=timings,
                       info=info)
    out: list[FrameMeasurement] = []
    pts_s = pts * surf.s if surf.ok else None
    for i, m in enumerate(masks):
        tm = time.perf_counter()
        sf = with_local_quadratic(surf, pts_s, m, excl, K, cfg) if surf.ok and m.any() else surf
        maps: dict[str, Any] = {}
        fm = measure_mask(depth, m, K, sf, cfg, img_bgr=img_u, cert=cert_i, conf=conf,
                          mask_conf=float(mask_confs[i]) if mask_confs and i < len(mask_confs) else 1.0,
                          verifier_cls=verifier_classes[i] if verifier_classes and i < len(verifier_classes) else None,
                          method=method, depth_model_version=depth_model_version, maps_out=maps)
        sev = irc.severity(fm.depth_p95_mm, fm.width_m * 1000.0, fm.water_filled)
        ex = dict(fm.extras)
        ex.update(severity=sev.value, severity_labels=irc.labels(sev), intrinsics=info,
                  timings_ms={**timings, "mask_ms": (time.perf_counter() - tm) * 1000.0})
        if fm.water_filled:
            lb = irc.water_lower_bound(fm.width_m * 1000.0)
            ex["severity_lower_bound"] = lb.value
        fm = fm.model_copy(update={"extras": ex, "seq": seq, "t_frame_ms": t_frame_ms})
        out.append(fm)
        ctx.maps.append(maps)
    timings["total_ms"] = (time.perf_counter() - t0) * 1000.0
    return out, ctx


def size_only(cfg: dict[str, Any]) -> bool:
    """D27: ``depth.backend: off`` (YAML may read a bare off as False) -> no depth model, size from the ground plane."""
    b = (cfg.get("depth") or {}).get("backend")
    return b is False or str(b).strip().lower() in ("off", "none", "false", "")


def _measure_frames_ground(img_u: np.ndarray, masks: list[np.ndarray], K: np.ndarray, cert_i: CalibCert,
                           cfg: dict[str, Any], info: dict[str, Any], timings: dict[str, Any], t0: float,
                           mask_confs: Optional[Sequence[float]], verifier_classes: Optional[Sequence[Optional[str]]],
                           seq: Optional[int], t_frame_ms: Optional[int]) -> tuple[list[FrameMeasurement], FrameContext]:
    """Size-only measurement of every mask (D27): no depth pass, no RANSAC; the calibrated plane is the road."""
    from measure.ground_geom import ground_surface, measure_mask_ground

    surf = ground_surface(cert_i.h_true_m, cert_i.pitch_deg, cert_i.roll_deg, cert_i.yaw_deg)
    ctx = FrameContext(depth=None, K=K, cert=cert_i, surface=surf, img=img_u, masks=masks, timings_ms=timings,
                       info={**info, "size_only": True})
    out: list[FrameMeasurement] = []
    for i, m in enumerate(masks):
        tm = time.perf_counter()
        fm = measure_mask_ground(m, K, cert_i, cfg, img_bgr=img_u,
                                 mask_conf=float(mask_confs[i]) if mask_confs and i < len(mask_confs) else 1.0,
                                 verifier_cls=verifier_classes[i] if verifier_classes and i < len(verifier_classes)
                                 else None)
        sev = irc.severity(None, fm.width_m * 1000.0, fm.water_filled)
        ex = dict(fm.extras)
        ex.update(severity=sev.value, severity_labels=irc.labels(sev), severity_basis="width (IRC:82, no depth)",
                  intrinsics=info, timings_ms={**timings, "mask_ms": (time.perf_counter() - tm) * 1000.0})
        if fm.water_filled:
            ex["severity_lower_bound"] = irc.water_lower_bound(fm.width_m * 1000.0).value
        out.append(fm.model_copy(update={"extras": ex, "seq": seq, "t_frame_ms": t_frame_ms}))
        ctx.maps.append({})
    timings["total_ms"] = (time.perf_counter() - t0) * 1000.0
    return out, ctx


def measure_frame(img: np.ndarray, mask: np.ndarray, cert: CalibCert, cfg: Optional[dict[str, Any]] = None,
                  **kw: Any) -> FrameMeasurement:
    """PLAN 10 WP2 interface. Extra keyword args go to :func:`measure_frames` (backend=, depth=, ...).

    ``other_masks=`` (list) excludes further detections of the same frame from the road fit.
    """
    others = list(kw.pop("other_masks", []) or [])
    res, _ = measure_frames(img, [mask] + others, cert, cfg, **kw)
    return res[0]
