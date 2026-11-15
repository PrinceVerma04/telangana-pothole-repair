"""Mask integral: depth statistics, area, volume, L/W, range, lane offset, sigma, quality, water (PLAN 4.5-4.7, 4.10).

Geometry (per mask pixel i, unnormalised ray d_i with z = 1, global road normal n):
  * F_i  = s * Z_i * d_i  -- the observed floor point (scale-locked), in the road frame (x_F, y_F, z_F).
  * A_i  = ray  x  reference surface (plane or local quadratic): the *aperture* point at the road level.
  * a_i  = t_A^2 / (fx * fy * |n . d_i|): exact plan (road-plane) cross-section of the pixel beam at
           A_i. (PLAN 4.5 writes (Z/fx)(Z/fy)/cos(theta); that omits the off-axis factor
           cos(alpha) = 1/|d|. The form used here is exact for a pinhole: |n.d| = cos(theta)/cos(alpha).)
  * delta_i     = y_s(x_F, z_F) - y_F   (vertical cavity depth below the surface at the floor point,
                  clipped >= 0) -> depth statistics.
  * delta_vol_i = y_s(A_i) - y_F        -> volume. Each pixel beam is a cone from the camera centre;
                  its slice from the surface at A_i down to F_i is a frustum of horizontal cross-section
                  a_i * (t / t_A)^2, i.e. volume a_i * delta_vol_i * (1 + r + r^2/3), r = (Z_i - t_A)/t_A.
                  The sum is the exact volume of the part of the cavity the camera sees.

Self-occlusion: at grazing angles the near wall hides a wedge under the ray through the near rim.
Per image column the hidden wedge is bounded by [0, S_col], S_col = 1/2 * gap * delta_first * w_col
(gap = plan distance rim->first floor hit). We report V = V_visible + f*S (f = 0.5) and add
S/sqrt(12) (uniform prior) to sigma_V. Fully visible potholes have S ~ 0.

Sigma model (PLAN 4.7, with PLAN_CRITIQUE (b): the outer surface_rms term of sigma_d95 double
counted surface_rms already inside sigma_delta_i, so it is dropped):
  sigma_delta_i^2 = (k Z_i^2 |n.d_i|)^2 + surface_rms^2 + (delta_i sigma_s)^2
  sigma_s = sqrt((tape/h_true)^2 + (plane_rms/h_true)^2)
  sigma_d95 = sqrt(mean_{top 5 %} sigma_delta_i^2)
  sigma_A = sqrt((2 A sigma_s)^2 + (b_px * sum_{boundary px} a_i)^2)   (boundary +-b_px pixel)
  sigma_V = sqrt((sum a_i sigma_delta_i)^2 + (depth_mean sigma_A)^2 + sigma_occl^2)
The decomposition sigma_d95^2 = k^2 * zc2 + other2 is stored in extras['sigma_parts'] so that
measure.truth can re-fit k per device.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from measure.camera import rays_at
from measure.config import ensure_cfg
from measure.surface import intersect_rays, surface_height, to_road
from measure.types import CalibCert, DepthQuality, FrameMeasurement, Method, SurfaceFit


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    if len(x) == 0:
        return float("nan")
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    c = np.cumsum(ws) - 0.5 * ws
    c /= ws.sum()
    return float(np.interp(q, c, xs))


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2

    if px <= 0:
        return mask.copy()
    return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(px)).astype(bool)


# ----------------------------------------------------------------------------- quality gates
def range_gate(D: float, qcfg: dict[str, Any]) -> float:
    lo, hi = qcfg["range_full_m"]
    far, far_q = qcfg["range_far_m"], qcfg["range_far_q"]
    near0 = qcfg.get("range_near_zero_m", 1.5)
    if not np.isfinite(D):
        return 0.0
    if D < lo:
        return float(np.clip((D - near0) / max(lo - near0, 1e-6), 0.0, 1.0))
    if D <= hi:
        return 1.0
    if D <= far:
        return float(1.0 - (1.0 - far_q) * (D - hi) / (far - hi))
    return 0.0


def image_gates(img_bgr: np.ndarray, mask: np.ndarray, qcfg: dict[str, Any]) -> dict[str, float]:
    import cv2

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    vs, us = np.nonzero(mask)
    H, W = mask.shape
    pad = 16
    y0, y1 = max(0, vs.min() - pad), min(H, vs.max() + pad + 1)
    x0, x1 = max(0, us.min() - pad), min(W, us.max() + pad + 1)
    roi = gray[y0:y1, x0:x1]
    lap = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    luma = float(gray[mask].mean())
    q_blur = float(np.clip(lap / float(qcfg["blur_lap_var_ref"]), 0.0, 1.0))
    lo, hi = qcfg["luma_ok"]
    ramp = float(qcfg.get("illum_ramp", 20.0))
    if luma < lo:
        q_ill = float(np.clip(1 - (lo - luma) / ramp, 0, 1))
    elif luma > hi:
        q_ill = float(np.clip(1 - (luma - hi) / ramp, 0, 1))
    else:
        q_ill = 1.0
    return {"lap_var": lap, "q_blur": q_blur, "mask_luma": luma, "q_illum": q_ill}


def specular_score(img_bgr: np.ndarray, mask: np.ndarray, wcfg: dict[str, Any]) -> dict[str, Any]:
    """PLAN 4.10 specular test: mask mean luma > 0.6 * sky luma AND low texture (local std)."""
    import cv2

    gray = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr).astype(np.float32)
    H = gray.shape[0]
    sky = float(gray[: max(1, int(H * wcfg.get("sky_rows_frac", 0.15)))].mean())
    m1 = cv2.blur(gray, (7, 7))
    m2 = cv2.blur(gray * gray, (7, 7))
    std = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    tex = float(np.median(std[mask])) if mask.any() else float("nan")
    luma = float(gray[mask].mean()) if mask.any() else float("nan")
    bright = luma > float(wcfg["luma_vs_sky"]) * sky
    smooth = tex < float(wcfg["texture_std_max"])
    return {"sky_luma": sky, "mask_luma": luma, "texture_std": tex, "specular": bool(bright and smooth)}


# ----------------------------------------------------------------------------- main
def measure_mask(depth_m: np.ndarray, mask: np.ndarray, K: np.ndarray, surface: SurfaceFit,
                 cfg: Optional[dict[str, Any]] = None, *, img_bgr: Optional[np.ndarray] = None,
                 cert: Optional[CalibCert] = None, conf: Optional[np.ndarray] = None, mask_conf: float = 1.0,
                 depth_is_scaled: bool = False, verifier_cls: Optional[str] = None,
                 stereo_disagreement: Optional[float] = None, method: Method = Method.mono,
                 depth_model_version: str = "unknown", h_true: Optional[float] = None,
                 maps_out: Optional[dict[str, Any]] = None) -> FrameMeasurement:
    """One mask in one (undistorted) frame -> FrameMeasurement.

    ``depth_m`` is the raw backend depth (camera z, metres); ``surface.s`` is applied unless
    ``depth_is_scaled``. ``K`` is the undistorted intrinsic matrix the depth map lives in.
    ``maps_out`` (optional dict) receives per-pixel ``vs, us, delta_mm`` for overlays.
    """
    cfg = ensure_cfg(cfg)
    mask = mask.astype(bool)
    mask_px = int(mask.sum())
    base = dict(method=method, depth_model_version=depth_model_version, mask_px=mask_px,
                calib_id=cert.calib_id if cert else None, certified=bool(cert.certified) if cert else False)
    if h_true is None:
        h_true = cert.h_true_m if cert else float(surface.d)
    if not surface.ok:
        return FrameMeasurement(quality=0.0, reject_reason=f"surface:{surface.reject_reason}",
                                depth_quality=DepthQuality.rejected, plane_inlier_ratio=surface.inlier_ratio,
                                extras={"surface": _surface_extras(surface)}, **base)
    if mask_px == 0:
        return FrameMeasurement(quality=0.0, reject_reason="empty_mask", depth_quality=DepthQuality.rejected,
                                **base)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    s = 1.0 if depth_is_scaled else float(surface.s)
    n = np.asarray(surface.n, float)
    k_sigma = float(cert.truth_fit.k_sigma) if cert else float(cfg["uncertainty"]["k_sigma_default"])

    vs, us = np.nonzero(mask)
    d = rays_at(K, us, vs)
    Z = depth_m[vs, us].astype(np.float64) * s
    valid = np.isfinite(Z) & (Z > 0)
    if conf is not None:
        valid &= conf[vs, us] >= float(cfg["mask"]["depth_conf_min"])
    tA, A = intersect_rays(surface, d)
    valid &= np.isfinite(tA)
    if valid.sum() < 10:
        return FrameMeasurement(quality=0.0, reject_reason="no_valid_depth", depth_quality=DepthQuality.rejected,
                                **base)
    F = to_road(Z[:, None] * d, surface)
    ndot = np.abs(d @ n)
    a = tA ** 2 / (fx * fy * np.maximum(ndot, 1e-9))
    ys_F = surface_height(surface, F[:, 0], F[:, 2])
    delta = np.clip(ys_F - F[:, 1], 0.0, None)
    delta_vol = np.clip(A[:, 1] - F[:, 1], 0.0, None)
    # pixel beams are cones from the camera centre: the slice between the surface (camera z = t_A) and
    # the floor (z = Z) is a frustum whose horizontal cross-section grows as (t / t_A)^2 ->
    # volume = a_A * delta * (1 + r + r^2/3), r = (Z - t_A) / t_A.
    rr = np.clip((Z - tA) / tA, 0.0, None)
    vol_px = a * delta_vol * (1.0 + rr + rr * rr / 3.0)

    er = _erode(mask, int(cfg["mask"]["erode_px"]))
    stat = valid & er[vs, us]
    if stat.sum() < 10:
        stat = valid.copy()
    a_v, dv_v = a[valid], delta_vol[valid]
    area = float(a_v.sum())
    v_vis = float(vol_px[valid].sum())

    # ---- near-wall self-occlusion (hidden wedge under the ray through the near rim)
    occ = cfg["occlusion"]
    S_occ, S_hat, occ_info = 0.0, 0.0, {"mode": "disabled"}
    if occ.get("enabled", True):
        S_occ, S_hat, occ_info = occlusion_estimate(us[valid], vs[valid], A[valid], F[valid], dv_v, tA[valid],
                                                    fx, occ)
    volume = v_vis + S_hat
    # truth lies in [V_vis, V_vis + S]; uniform on the larger side -> |err| <= sqrt(3) * sigma
    sig_occ = max(S_hat, S_occ - S_hat) / math.sqrt(3.0)

    med = int(cfg["mask"].get("delta_median_px", 5))
    if med >= 3:
        delta_stat = _median_delta(depth_m, s, K, surface, mask, vs, us, delta, med)
    else:
        delta_stat = delta
    ds, ws = delta_stat[stat], a[stat]
    p95 = weighted_quantile(ds, ws, 0.95)
    p99 = weighted_quantile(ds, ws, 0.99)
    depth_mean = volume / area if area > 0 else float("nan")

    # ---- geometry: L/W from boundary-pixel corners on the surface
    er1 = _erode(mask, 1)
    bnd = mask & ~er1
    bv, bu = np.nonzero(bnd)
    corners_u = (bu[:, None] + np.array([-0.5, 0.5, 0.5, -0.5])[None, :]).ravel()
    corners_v = (bv[:, None] + np.array([-0.5, -0.5, 0.5, 0.5])[None, :]).ravel()
    tc, C = intersect_rays(surface, rays_at(K, corners_u, corners_v))
    C = C.reshape(-1, 4, 3)
    okc = np.all(np.isfinite(C[..., 0]), axis=1)
    L, Wd, ell_x, ell_z, perim = _rect_lw(C[okc])
    in_bnd = bnd[vs, us]
    a_bnd = float(a[valid & in_bnd].sum())

    # ---- sigma model
    unc = cfg["uncertainty"]
    surf_rms_m = (surface.ring_rms_mm if surface.ring_rms_mm is not None else surface.plane_rms_mm) / 1000.0
    plane_rms_m = surface.plane_rms_mm / 1000.0
    sigma_s = math.sqrt((float(unc["tape_sigma_m"]) / h_true) ** 2 + (plane_rms_m / h_true) ** 2)
    zc = Z * Z * ndot  # sigma_Z * |n.d| per unit k
    sig_d_sq = (k_sigma * zc) ** 2 + surf_rms_m ** 2 + (delta * sigma_s) ** 2
    top = stat & (delta_stat >= p95)
    if top.sum() == 0:
        top = stat
    zc2 = float(np.mean(zc[top] ** 2))
    other2 = float(np.mean(surf_rms_m ** 2 + (delta[top] * sigma_s) ** 2))
    sig_d95 = math.sqrt(k_sigma ** 2 * zc2 + other2)
    top1 = stat & (delta_stat >= p99)
    sig_dmax = math.sqrt(float(np.mean(sig_d_sq[top1]))) if top1.any() else sig_d95
    b_px = float(unc["boundary_px"])
    sig_A = math.sqrt((2 * area * sigma_s) ** 2 + (b_px * a_bnd) ** 2)
    sig_V = math.sqrt(float(np.sum(a_v * np.sqrt(sig_d_sq[valid]))) ** 2 + (depth_mean * sig_A) ** 2 + sig_occ ** 2)
    # a +-b_px boundary error is correlated around the mask (dilation/erosion moves both ends) -> 2*b*ell
    sig_L = math.sqrt((L * sigma_s) ** 2 + (2 * b_px * ell_z) ** 2)
    sig_W = math.sqrt((Wd * sigma_s) ** 2 + (2 * b_px * ell_x) ** 2)

    # ---- placement
    rng_m = float(np.median(Z[valid]))
    lane_off = float(np.median(A[valid, 0]))
    gxz = (float(np.median(A[valid, 0])), float(np.median(A[valid, 2])))

    # ---- quality gates (PLAN 4.7)
    qcfg = cfg["quality"]
    gates: dict[str, Any] = {"q_range": range_gate(rng_m, qcfg), "q_mask_conf": float(np.clip(mask_conf, 0, 1)),
                             "q_plane": float(np.clip(surface.inlier_ratio / 0.8, 0, 1)),
                             "q_area": 1.0 if mask_px >= int(cfg["mask"]["min_mask_px"]) else 0.0}
    if img_bgr is not None:
        gates.update(image_gates(img_bgr, mask, qcfg))
    else:
        gates.update(q_blur=1.0, q_illum=1.0, image_gates="not_evaluated")
    warnings = list(surface.extras.get("warnings", []))
    gates["q_pitch"] = float(qcfg.get("pitch_warn_q", 0.5)) if any("pitch_dev" in w for w in warnings) else 1.0
    gates["q_scale"] = float(qcfg.get("pitch_warn_q", 0.5)) if any("scale_factor" in w for w in warnings) else 1.0
    q = 1.0
    for kq in ("q_range", "q_mask_conf", "q_plane", "q_area", "q_blur", "q_illum", "q_pitch", "q_scale"):
        q *= float(gates[kq])
    q = float(np.clip(q, 0.0, 1.0))
    reject = None
    if gates["q_area"] == 0.0:
        reject = f"mask_px<{cfg['mask']['min_mask_px']}"
    elif gates["q_range"] == 0.0:
        reject = "range_outside_gate"

    # ---- water (PLAN 4.10)
    wcfg = cfg["water"]
    water_info: dict[str, Any] = {}
    water = verifier_cls == "pothole_water"
    if img_bgr is not None:
        water_info = specular_score(img_bgr, mask, wcfg)
        if (p95 * 1000.0 < float(wcfg["shallow_depth_mm"]) and water_info["specular"]
                and area > float(wcfg["min_area_m2"])):
            water = True
    if stereo_disagreement is not None and stereo_disagreement > 0.30:
        water = True
    water_info["stereo_disagreement"] = stereo_disagreement

    extras: dict[str, Any] = {
        "sigma_parts": {"k_sigma": k_sigma, "zc2": zc2, "other2": other2, "sigma_s": sigma_s,
                        "surface_rms_mm": surf_rms_m * 1000.0},
        "depth_max_sigma_mm": sig_dmax * 1000.0,
        "occlusion": {"visible_l": v_vis * 1000.0, "hidden_bound_l": S_occ * 1000.0,
                      "hidden_est_l": S_hat * 1000.0, "sigma_l": sig_occ * 1000.0, **occ_info},
        "perimeter_m": perim,
        "n_valid_px": int(valid.sum()),
        "n_stat_px": int(stat.sum()),
        "gates": gates,
        "water": water_info,
        "offset_from_vehicle_centre_m": lane_off + (cert.lateral_offset_m if cert else 0.0),
        "volume_cuboid_l": L * Wd * p99 * 1000.0,
        "surface": _surface_extras(surface),
    }
    if warnings:
        extras["warnings"] = warnings

    fm = dict(
        depth_p95_mm=p95 * 1000.0, depth_max_mm=p99 * 1000.0, depth_mean_mm=depth_mean * 1000.0,
        depth_sigma_mm=sig_d95 * 1000.0, area_m2=area, area_sigma=sig_A, volume_l=volume * 1000.0,
        volume_sigma=sig_V * 1000.0, length_m=L, width_m=Wd, length_sigma_m=sig_L, width_sigma_m=sig_W,
        water_filled=water, depth_quality=DepthQuality.ok, quality=q if reject is None else 0.0,
        range_m=rng_m, lane_offset_m=lane_off, ground_xz=gxz, surface_model=surface.surface_model,
        plane_rms_mm=surface.plane_rms_mm, ring_rms_mm=surface.ring_rms_mm,
        plane_inlier_ratio=surface.inlier_ratio, scale_factor=surface.s, reject_reason=reject, extras=extras,
        **base,
    )
    if maps_out is not None:
        maps_out.update(vs=vs[valid], us=us[valid], delta_mm=delta_stat[valid] * 1000.0)
    if water:
        fm.update(depth_p95_mm=None, depth_max_mm=None, depth_mean_mm=None, depth_sigma_mm=None, volume_l=None,
                  volume_sigma=None, depth_quality=DepthQuality.unknown_water)
        extras["water_raw_depth_p95_mm"] = p95 * 1000.0
    return FrameMeasurement(**fm)


def _median_delta(depth_m: np.ndarray, s: float, K: np.ndarray, surface: SurfaceFit, mask: np.ndarray,
                  vs: np.ndarray, us: np.ndarray, delta: np.ndarray, ksize: int) -> np.ndarray:
    """Median-filter the cavity-depth image (bbox of the mask) to remove per-pixel noise that would
    otherwise inflate upper percentiles (p95 of a noisy flat floor = d + 1.64 sigma). Kernel shrinks to 3
    for masks < 25 rows tall. Pixels outside the mask use their own delta (road ~ 0)."""
    from scipy.ndimage import median_filter

    y0, y1, x0, x1 = vs.min(), vs.max() + 1, us.min(), us.max() + 1
    if (y1 - y0) < 25:
        ksize = 3
    pad = ksize
    H, W = mask.shape
    y0p, y1p, x0p, x1p = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
    from measure.camera import pixel_rays

    Kc = K.copy()
    Kc[0, 2] -= x0p
    Kc[1, 2] -= y0p
    rays = pixel_rays(Kc, y1p - y0p, x1p - x0p)
    Zb = depth_m[y0p:y1p, x0p:x1p].astype(np.float64) * s
    Fb = to_road(Zb[..., None] * rays, surface)
    db = surface_height(surface, Fb[..., 0], Fb[..., 2]) - Fb[..., 1]
    db = np.where(np.isfinite(db), np.clip(db, 0.0, None), 0.0)
    db[vs - y0p, us - x0p] = delta
    filt = median_filter(db, size=ksize, mode="nearest")
    return filt[vs - y0p, us - x0p]


def occlusion_estimate(us: np.ndarray, vs: np.ndarray, A: np.ndarray, F: np.ndarray, dvol: np.ndarray,
                       tA: np.ndarray, fx: float, occ: dict[str, Any]) -> tuple[float, float, dict[str, Any]]:
    """Hidden near-wall wedge volume: (upper bound S, estimate S_hat, info), m^3.

    Per image column (nearest pixel = largest row): the ray through the near rim A0 reaches the floor at
    F0 (depth delta0) after a plan gap g. Nothing below that ray is seen. Assuming the floor under the
    wedge is no deeper than delta0, the hidden cross-section is bounded by S_col = g*delta0/2 (vertical
    near wall). 'mirror' mode estimates the hidden part by reflecting the (fully visible) far-wall
    profile delta(u) -- u = plan distance from the far rim -- onto the near wall:
    hidden = int_0^g clip(min(delta_far(u), delta0) - delta0*u/g, 0) du. A box pothole (vertical far
    wall) gets ~S_col, a bowl a small fraction. 'fraction' mode uses fill_fraction * S_col.
    Columns with too few far-wall samples fall back to fill_fraction.
    """
    mode = occ.get("mode", "mirror")
    f_fill = float(occ.get("fill_fraction", 0.5))
    order = np.lexsort((-vs, us))
    u_sorted = us[order]
    starts = np.r_[0, np.nonzero(u_sorted[1:] != u_sorted[:-1])[0] + 1, len(order)]
    S_tot, S_hat, n_occ_cols, n_fallback = 0.0, 0.0, 0, 0
    for c0, c1 in zip(starts[:-1], starts[1:], strict=False):
        idx = order[c0:c1]  # near -> far
        i0 = idx[0]
        g_vec = F[i0, [0, 2]] - A[i0, [0, 2]]
        gap = float(np.hypot(*g_vec))
        d0 = float(dvol[i0])
        w = float(tA[i0] / fx)
        S_col = 0.5 * gap * d0 * w
        if S_col <= 0:
            continue
        S_tot += S_col
        n_occ_cols += 1
        if mode != "mirror" or len(idx) < 4 or gap < 1e-6:
            S_hat += f_fill * S_col
            n_fallback += 1
            continue
        e = g_vec / gap
        far_rim = F[idx[-1], [0, 2]]
        u = (far_rim - F[idx, :][:, [0, 2]]) @ e  # distance back from the far rim
        dd = dvol[idx]
        sel = (u >= -1e-9) & (u <= gap * 1.05)
        if sel.sum() < 3:
            S_hat += f_fill * S_col
            n_fallback += 1
            continue
        uo = np.argsort(u[sel])
        us_, ds_ = u[sel][uo], dd[sel][uo]
        if len(ds_) >= 3:  # light median smoothing, then monotone (depth grows away from the rim)
            pad = np.r_[ds_[0], ds_, ds_[-1]]
            ds_ = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)
        ds_ = np.maximum.accumulate(ds_)
        ug = np.linspace(0.0, gap, 17)
        dm = np.interp(ug, np.r_[0.0, us_], np.r_[0.0, ds_])
        hid = np.clip(np.minimum(dm, d0) - d0 * ug / gap, 0.0, None)
        S_hat += float(np.trapezoid(hid, ug)) * w
    info = {"mode": mode, "occluded_columns": n_occ_cols, "fallback_columns": n_fallback}
    return S_tot, min(S_hat, S_tot), info


def _rect_lw(C: np.ndarray) -> tuple[float, float, float, float, float]:
    """C: (M, 4, 3) boundary-pixel corner points on the surface (road frame).

    Returns L (along travel, z), W (across), mean footprint extents ell_x, ell_z, perimeter (m).
    """
    import cv2

    if len(C) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    xz = C[..., [0, 2]].reshape(-1, 2)
    rect = cv2.minAreaRect((xz * 1000.0).astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float64) / 1000.0
    e1, e2 = box[1] - box[0], box[2] - box[1]
    l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    # L = side most aligned with the travel axis (z)
    if abs(e1[1]) / max(l1, 1e-12) >= abs(e2[1]) / max(l2, 1e-12):
        L, W = l1, l2
    else:
        L, W = l2, l1
    ell_x = float(np.mean(C[..., 0].max(1) - C[..., 0].min(1)))
    ell_z = float(np.mean(C[..., 2].max(1) - C[..., 2].min(1)))
    hull = cv2.convexHull((xz * 1000.0).astype(np.float32))
    perim = float(cv2.arcLength(hull, True)) / 1000.0
    return L, W, ell_x, ell_z, perim


def _surface_extras(surface: SurfaceFit) -> dict[str, Any]:
    out = {"h_hat": surface.h_hat, "s": surface.s, "pitch_est_deg": surface.pitch_est_deg,
           "n_road_pts": surface.n_road_pts, "surface_model": surface.surface_model.value,
           "reject_reason": surface.reject_reason}
    for k in ("roll_est_deg", "pitch_dev_deg", "quadratic_fallback", "n_ring_px", "candidate_gate"):
        if k in surface.extras:
            out[k] = surface.extras[k]
    return out
