"""Size-only measurement without a depth model (D27, ``depth.backend: off``).

The pothole outline (detector mask) is projected onto the road plane known from the camera calibration: tape-measured
height ``h_true_m``, ``pitch_deg``, ``roll_deg``, ``yaw_deg`` (``measure.camera.plane_from_extrinsics``). Every mask
pixel ray meets that plane at its aperture point, so the outline's plan footprint is exact for a flat road and a
camera at its calibrated pose. Output: area (m2), length / width (m), range, lane offset, each size with sigma.
Depth, volume and the depth-based water test are not available: those fields stay None, method = ``mono_ground``.

Uncertainty: the pose of a bus camera moves (braking, load, suspension), so sigma is propagated by re-projecting the
mask with the camera height +- ``height_sigma_m`` and the pitch +- ``pitch_sigma_deg`` (configs/measure.yaml
``ground``), plus a +-``boundary_px`` outline error, all added in quadrature.

Assumptions (also in the result's extras): flat road around the pothole; the mask outline is the pothole rim at road
level. A ridge, a camber or a wrong calibration shows up as a size error, never as a depth.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from measure.camera import plane_from_extrinsics, rays_at
from measure.config import ensure_cfg
from measure.pothole_geom import _erode, _rect_lw, image_gates, range_gate, specular_score
from measure.surface import intersect_rays, road_frame
from measure.types import CalibCert, DepthQuality, FrameMeasurement, Method, SurfaceFit, SurfaceModel

DEFAULTS = {"height_sigma_m": 0.02, "pitch_sigma_deg": 0.75, "min_valid_frac": 0.9, "max_far_m": 30.0,
            "max_area_m2": 25.0}


def ground_surface(h_m: float, pitch_deg: float, roll_deg: float = 0.0, yaw_deg: float = 0.0) -> SurfaceFit:
    """The calibrated road plane as a SurfaceFit (scale 1, no RANSAC)."""
    n, d = plane_from_extrinsics(h_m, pitch_deg, roll_deg)
    return SurfaceFit(n=n, d=d, h_hat=d, s=1.0, R_road=road_frame(n, yaw_deg), quad_coeffs=None, plane_rms_mm=0.0,
                      ring_rms_mm=None, inlier_ratio=1.0, surface_model=SurfaceModel.plane, pitch_est_deg=pitch_deg,
                      extras={"source": "calibration"})


def _footprint(mask: np.ndarray, K: np.ndarray, surf: SurfaceFit) -> Optional[dict[str, Any]]:
    """Plan area, L, W and placement of the mask on the plane; None when too much of it is above the horizon."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    vs, us = np.nonzero(mask)
    d = rays_at(K, us, vs)
    tA, A = intersect_rays(surf, d)
    valid = np.isfinite(tA)
    if valid.sum() < 10:
        return None
    n = np.asarray(surf.n, float)
    a = tA ** 2 / (fx * fy * np.maximum(np.abs(d @ n), 1e-9))  # exact plan footprint of each pixel beam
    bnd = mask & ~_erode(mask, 1)
    bv, bu = np.nonzero(bnd)
    cu = (bu[:, None] + np.array([-0.5, 0.5, 0.5, -0.5])[None, :]).ravel()
    cv = (bv[:, None] + np.array([-0.5, -0.5, 0.5, 0.5])[None, :]).ravel()
    _, C = intersect_rays(surf, rays_at(K, cu, cv))
    C = C.reshape(-1, 4, 3)
    okc = np.all(np.isfinite(C[..., 0]), axis=1)
    L, W, ell_x, ell_z, perim = _rect_lw(C[okc])
    # far edge from the outline's boundary (pixels under-sample the far part): 98th pct. of corner distances
    far = float(np.percentile(np.linalg.norm(C[okc].reshape(-1, 3), axis=1), 98)) if okc.any() else float("inf")
    return {"area": float(a[valid].sum()), "L": L, "W": W, "ell_x": ell_x, "ell_z": ell_z, "perim": perim,
            "a_bnd": float(a[valid & bnd[vs, us]].sum()), "valid_frac": float(valid.mean()),
            "range": float(np.median(tA[valid])), "range_far": far,
            "lane": float(np.median(A[valid, 0])),
            "gxz": (float(np.median(A[valid, 0])), float(np.median(A[valid, 2])))}


def measure_mask_ground(mask: np.ndarray, K: np.ndarray, cert: CalibCert, cfg: Optional[dict[str, Any]] = None, *,
                        img_bgr: Optional[np.ndarray] = None, mask_conf: float = 1.0,
                        verifier_cls: Optional[str] = None) -> FrameMeasurement:
    """One mask in one (undistorted) frame -> size-only FrameMeasurement (depth fields None)."""
    cfg = ensure_cfg(cfg)
    g = {**DEFAULTS, **(cfg.get("ground") or {})}
    mask = np.asarray(mask).astype(bool)
    mask_px = int(mask.sum())
    base = dict(method=Method.mono_ground, depth_model_version="none", mask_px=mask_px, calib_id=cert.calib_id,
                certified=bool(cert.certified), scale_factor=1.0, plane_inlier_ratio=1.0)
    if mask_px == 0:
        return FrameMeasurement(quality=0.0, reject_reason="empty_mask", depth_quality=DepthQuality.rejected, **base)
    h, pitch, roll, yaw = cert.h_true_m, cert.pitch_deg, cert.roll_deg, cert.yaw_deg
    fp = _footprint(mask, K, ground_surface(h, pitch, roll, yaw))
    if fp is None or fp["valid_frac"] < float(g["min_valid_frac"]):
        return FrameMeasurement(quality=0.0, reject_reason="above_horizon", depth_quality=DepthQuality.rejected,
                                **base)

    # ---- sigma: re-project with the pose perturbed by its 1-sigma, plus the outline boundary error
    dh, dp = float(g["height_sigma_m"]), float(g["pitch_sigma_deg"])
    deltas: dict[str, list[float]] = {"area": [], "L": [], "W": []}
    for hh, pp in ((h + dh, pitch), (h, pitch + dp), (h, max(0.1, pitch - dp))):
        alt = _footprint(mask, K, ground_surface(hh, pp, roll, yaw))
        for k in deltas:
            deltas[k].append(0.0 if alt is None else alt[k] - fp[k])
    # height: one-sided step is the 1-sigma change; pitch: half the +-sigma spread (symmetric estimate)
    pose = {k: math.hypot(v[0], 0.5 * (abs(v[1]) + abs(v[2]))) for k, v in deltas.items()}
    b_px = float(cfg["uncertainty"]["boundary_px"])
    sig_A = math.hypot(pose["area"], b_px * fp["a_bnd"])
    sig_L = math.hypot(pose["L"], 2 * b_px * fp["ell_z"])
    sig_W = math.hypot(pose["W"], 2 * b_px * fp["ell_x"])

    # ---- quality gates (same as the depth path, minus the road-plane fit)
    qcfg = cfg["quality"]
    gates: dict[str, Any] = {"q_range": range_gate(fp["range"], qcfg), "q_mask_conf": float(np.clip(mask_conf, 0, 1)),
                             "q_area": 1.0 if mask_px >= int(cfg["mask"]["min_mask_px"]) else 0.0}
    if img_bgr is not None:
        gates.update(image_gates(img_bgr, mask, qcfg))
    else:
        gates.update(q_blur=1.0, q_illum=1.0, image_gates="not_evaluated")
    q = 1.0
    for kq in ("q_range", "q_mask_conf", "q_area", "q_blur", "q_illum"):
        q *= float(gates[kq])
    q = float(np.clip(q, 0.0, 1.0))
    reject = None
    if gates["q_area"] == 0.0:
        reject = f"mask_px<{cfg['mask']['min_mask_px']}"
    elif gates["q_range"] == 0.0:
        reject = "range_outside_gate"
    elif fp["range_far"] > float(g["max_far_m"]):  # D51: the far edge near the horizon blows the area up
        reject = "outline_too_far"
    elif fp["area"] > float(g["max_area_m2"]):  # D51: no single pothole is this big; a wrong outline or pose
        reject = "implausible_area"

    # water: no depth to test shallowness, so only the verifier class decides; the specular score is kept as a hint
    water = verifier_cls == "pothole_water"
    water_info = specular_score(img_bgr, mask, cfg["water"]) if img_bgr is not None else {}
    extras: dict[str, Any] = {
        "size_only": True,
        "assumptions": "flat road; outline = rim at road level; camera at its calibrated height and pitch",
        "sigma_parts": {"pose_area_m2": pose["area"], "pose_length_m": pose["L"], "pose_width_m": pose["W"],
                        "height_sigma_m": dh, "pitch_sigma_deg": dp, "boundary_px": b_px},
        "perimeter_m": fp["perim"], "valid_px_frac": fp["valid_frac"], "far_edge_m": fp["range_far"],
        "gates": gates, "water": water_info,
        "offset_from_vehicle_centre_m": fp["lane"] + cert.lateral_offset_m,
    }
    return FrameMeasurement(
        area_m2=fp["area"], area_sigma=sig_A, length_m=fp["L"], width_m=fp["W"], length_sigma_m=sig_L,
        width_sigma_m=sig_W, water_filled=water,
        depth_quality=DepthQuality.unknown_water if water else DepthQuality.not_measured,
        quality=q if reject is None else 0.0, range_m=fp["range"], lane_offset_m=fp["lane"], ground_xz=fp["gxz"],
        surface_model=SurfaceModel.plane, plane_rms_mm=None, reject_reason=reject, extras=extras, **base)
