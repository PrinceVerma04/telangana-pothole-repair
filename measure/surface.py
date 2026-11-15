"""Road surface: RANSAC plane, height scale-lock, road frame, local quadratic ring fit (PLAN 4.4).

Conventions
-----------
* Plane ``n.p + d = 0`` in the camera frame with ``n`` the unit road *up* normal, oriented so the
  camera origin is on the positive side (``d > 0``); then ``d`` is the camera height and
  ``n.p + d`` is the signed height of ``p`` above the plane.
* Scale lock: ``s = h_true / h_hat`` with ``h_hat`` the RANSAC height in raw depth units; callers
  multiply depth by ``s``. ``SurfaceFit.d`` is reported *after* scale lock (``= h_true``).
* Road frame (measure/types.py): X right, Y up (= n), Z forward = vehicle axis projected onto the
  plane; origin = camera foot point ``p0 = -d n``. ``R_road`` rows are (X, Y, Z) expressed in camera
  coordinates; because the contract's road frame is left-handed while OpenCV's camera frame is
  right-handed, ``det(R_road) = -1`` (orthonormal, distances/areas preserved).
* Local quadratic ``y_s(x, z) = a + b x + c z + d x^2 + e z^2 + f x z`` in road-frame metres,
  stored in ``quad_coeffs`` in *absolute* road coordinates (centre also kept in ``extras``).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Optional

import numpy as np

from measure.config import ensure_cfg
from measure.types import SurfaceFit, SurfaceModel


# ----------------------------------------------------------------------------- plane RANSAC
def _plane_from_3(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """p: (B, 3, 3) -> normals (B, 3) unit, offsets (B,)."""
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 1e-12
    n[ok] /= nn[ok, None]
    d = -np.einsum("bi,bi->b", n, p[:, 0])
    return n, np.where(ok, d, np.nan)


def lsq_plane(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares plane through ``pts`` (N, 3), oriented so d > 0 (origin on + side)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    d = -float(n @ c)
    if d < 0:
        n, d = -n, -d
    return n / np.linalg.norm(n), d


def ransac_plane(pts: np.ndarray, thr: float, iterations: int = 1000, confidence: float = 0.99,
                 rng: Optional[np.random.Generator] = None, batch: int = 128,
                 up_hint: Optional[np.ndarray] = None, max_tilt_deg: float = 60.0
                 ) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Numpy RANSAC (3-point hypotheses evaluated in batches, adaptive early exit at ``confidence``).

    ``up_hint`` (unit vector) discards hypotheses tilted more than ``max_tilt_deg`` from it (walls,
    vehicles). Returns (n, d, inlier_mask, iterations_used) after a least-squares refit on inliers.
    """
    rng = rng or np.random.default_rng(0)
    N = len(pts)
    if N < 3:
        raise ValueError("need >= 3 points")
    best_cnt, best = -1, None
    it, needed = 0, iterations
    cos_tilt = math.cos(math.radians(max_tilt_deg))
    while it < min(iterations, needed):
        b = min(batch, iterations - it)
        idx = rng.integers(0, N, size=(b, 3))
        n, d = _plane_from_3(pts[idx])
        valid = np.isfinite(d)
        if up_hint is not None:
            valid &= np.abs(n @ up_hint) >= cos_tilt
        if valid.any():
            n, d = n[valid], d[valid]
            dist = np.abs(pts @ n.T + d[None, :])  # (N, b')
            cnt = (dist < thr).sum(axis=0)
            j = int(np.argmax(cnt))
            if cnt[j] > best_cnt:
                best_cnt, best = int(cnt[j]), (n[j], d[j])
                w = best_cnt / N
                if w >= 1.0:
                    needed = it + b
                elif w > 0:
                    needed = math.ceil(math.log(1 - confidence) / math.log(max(1e-12, 1 - w ** 3)))
        it += b
    if best is None:
        raise RuntimeError("RANSAC found no valid plane hypothesis")
    n, d = best
    inl = np.abs(pts @ n + d) < thr
    if inl.sum() >= 3:
        n, d = lsq_plane(pts[inl])
        inl = np.abs(pts @ n + d) < thr
    if d < 0:
        n, d = -n, -d
    return n, float(d), inl, it


# ----------------------------------------------------------------------------- road frame
def _rodrigues(axis: np.ndarray, deg: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    t = math.radians(deg)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * Kx + (1 - math.cos(t)) * Kx @ Kx


def road_frame(n: np.ndarray, yaw_deg: float = 0.0) -> np.ndarray:
    """R_road (rows X, Y, Z in camera coordinates). Vehicle axis = camera z rotated by -yaw about n."""
    Y = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    Z = z - (z @ Y) * Y
    Z /= np.linalg.norm(Z)
    if abs(yaw_deg) > 1e-9:
        # camera yawed right by yaw -> vehicle forward is yaw to the camera's left. Rotating about the
        # 'up' axis Y (which points to -y_cam) by +yaw turns +Z toward -X (left).
        Z = _rodrigues(Y, yaw_deg) @ Z
    X = np.cross(Z, Y)
    X /= np.linalg.norm(X)
    return np.stack([X, Y, Z])


def pitch_roll_from_normal(n: np.ndarray) -> tuple[float, float]:
    """For n = up in camera frame: pitch (deg, + = looking down), roll (deg)."""
    pitch = math.degrees(math.asin(float(np.clip(-n[2], -1, 1))))
    roll = math.degrees(math.atan2(float(n[0]), float(-n[1])))
    return pitch, roll


def to_road(points_cam: np.ndarray, fit: SurfaceFit) -> np.ndarray:
    """Scaled camera-frame points (..., 3) -> road frame (..., 3)."""
    p0 = -fit.d * np.asarray(fit.n)
    return (points_cam - p0) @ np.asarray(fit.R_road).T


def surface_height(fit: SurfaceFit, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Reference surface height y_s(x, z) in the road frame (0 for the plane)."""
    q = fit.quad_coeffs
    if q is None or fit.surface_model == SurfaceModel.plane:
        return np.zeros_like(np.asarray(x, float))
    a, b, c, d, e, f = q
    return a + b * x + c * z + d * x * x + e * z * z + f * x * z


def surface_gradient(fit: SurfaceFit, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if fit.surface_model == SurfaceModel.plane or fit.quad_coeffs is None:
        zero = np.zeros_like(np.asarray(x, float))
        return zero, zero
    _, b, c, d, e, f = fit.quad_coeffs
    return b + 2 * d * x + f * z, c + 2 * e * z + f * x


def intersect_rays(fit: SurfaceFit, rays_cam: np.ndarray, iters: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Intersect unnormalised camera rays (N, 3; z = 1) with the reference surface.

    Returns (t, road_xyz) where ``t`` = camera-frame z of the hit (the ray has z = 1) and road_xyz
    (N, 3) the hit in the road frame. Plane: closed form; quadratic: Newton from the plane solution.
    Rays that never meet the surface ahead get t = nan.
    """
    R = np.asarray(fit.R_road)
    o = np.array([0.0, fit.d, 0.0])  # camera origin in the road frame
    r = rays_cam @ R.T  # ray directions in road frame
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(r[:, 1] < -1e-9, -fit.d / r[:, 1], np.nan)
        if fit.surface_model == SurfaceModel.quadratic and fit.quad_coeffs is not None:
            for _ in range(iters):
                p = o + t[:, None] * r
                ys = surface_height(fit, p[:, 0], p[:, 2])
                gx, gz = surface_gradient(fit, p[:, 0], p[:, 2])
                g = p[:, 1] - ys
                dg = r[:, 1] - gx * r[:, 0] - gz * r[:, 2]
                t = t - g / np.where(np.abs(dg) > 1e-12, dg, np.nan)
    t = np.where(t > 0, t, np.nan)
    return t, o + t[:, None] * r


# ----------------------------------------------------------------------------- candidates
def road_candidates(points_cam: np.ndarray, cfg: dict[str, Any], exclude_mask: Optional[np.ndarray] = None,
                    scale: float = 1.0, notroad_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """PLAN 4.4.1: lower 65 % rows, minus dilated detections, Z*scale in [z_min, z_max], minus not-road."""
    import cv2

    rc = cfg["road_candidates"]
    H, W = points_cam.shape[:2]
    m = np.zeros((H, W), bool)
    m[int(round(H * (1 - rc["lower_frac"]))):, :] = True
    Z = points_cam[..., 2] * scale
    m &= np.isfinite(Z) & (Z >= rc["z_min_m"]) & (Z <= rc["z_max_m"])
    if exclude_mask is not None and exclude_mask.any():
        k = int(rc.get("mask_dilate_px", 20))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        m &= ~cv2.dilate(exclude_mask.astype(np.uint8), ker).astype(bool)
    if notroad_mask is not None:
        m &= ~notroad_mask
    return m


# ----------------------------------------------------------------------------- main fit
def fit_road_surface(points_cam: np.ndarray, road_mask: Optional[np.ndarray] = None,
                     cfg: Optional[dict[str, Any]] = None, *, h_true: float,
                     exclude_mask: Optional[np.ndarray] = None, pitch_prior_deg: Optional[float] = None,
                     yaw_deg: float = 0.0, certified: bool = False,
                     pothole_mask: Optional[np.ndarray] = None, K: Optional[np.ndarray] = None,
                     notroad_mask: Optional[np.ndarray] = None,
                     rng: Optional[np.random.Generator] = None) -> SurfaceFit:
    """RANSAC plane + height scale-lock (+ local quadratic if ``pothole_mask`` and ``K`` given).

    ``points_cam``: (H, W, 3) raw back-projected points (preferred: candidates are then rebuilt after
    scale lock with the metric range gate) or (N, 3) with ``road_mask`` (N,) selecting candidates.
    """
    cfg = ensure_cfg(cfg)
    rs = cfg["ransac"]
    rng = rng or np.random.default_rng(int(rs.get("seed", 0)))
    grid = points_cam.ndim == 3
    extras: dict[str, Any] = {}

    def _select(scale: float) -> np.ndarray:
        if grid:
            m = road_candidates(points_cam, cfg, exclude_mask, scale, notroad_mask)
            if road_mask is not None:
                m &= road_mask
            return points_cam[m]
        sel = np.ones(len(points_cam), bool) if road_mask is None else road_mask.astype(bool)
        p = points_cam[sel]
        return p[np.all(np.isfinite(p), axis=1)]

    def _reject(reason: str, **kw: Any) -> SurfaceFit:
        base = dict(n=np.array([0.0, -1.0, 0.0]), d=float(h_true), h_hat=float("nan"), s=float("nan"),
                    R_road=np.eye(3), quad_coeffs=None, plane_rms_mm=float("nan"), ring_rms_mm=None,
                    inlier_ratio=0.0, reject_reason=reason, extras=extras)
        base.update(kw)
        return SurfaceFit(**base)

    minp = int(rs.get("min_points", 300))
    maxp = int(cfg["road_candidates"]["max_points"])

    def _sub(p: np.ndarray) -> np.ndarray:
        return p if len(p) <= maxp else p[rng.choice(len(p), maxp, replace=False)]

    up_hint = None
    if pitch_prior_deg is not None:
        from measure.camera import plane_from_extrinsics

        up_hint, _ = plane_from_extrinsics(1.0, pitch_prior_deg)
    thr_m = float(rs["inlier_mm"]) / 1000.0
    iters_used = 0
    # --- pass 0: scale-free bootstrap. Raw monocular depth can be off by 2-4x, so the metric range
    # gate cannot be applied yet: use the [p5, p75] band of raw candidate depth and a threshold
    # relative to depth (bootstrap_rel_thr * median Z).
    if grid:
        nogate = copy.deepcopy(cfg)
        nogate["road_candidates"]["z_min_m"], nogate["road_candidates"]["z_max_m"] = 0.0, float("inf")
        m0 = road_candidates(points_cam, nogate, exclude_mask, 1.0, notroad_mask)
        if road_mask is not None:
            m0 &= road_mask
        z0 = points_cam[..., 2][m0]
        if len(z0) >= minp:
            lo0, hi0 = np.percentile(z0, [5, 75])
            m0 &= (points_cam[..., 2] >= lo0) & (points_cam[..., 2] <= hi0)
        pts0 = points_cam[m0]
    else:
        pts0 = _select(1.0)
    pts0 = pts0[np.all(np.isfinite(pts0), axis=1) & (pts0[:, 2] > 0)] if len(pts0) else pts0
    if len(pts0) < minp:
        return _reject("too_few_road_points", n_road_pts=int(len(pts0)))
    pts0 = _sub(pts0)
    thr0 = float(rs.get("bootstrap_rel_thr", 0.004)) * float(np.median(pts0[:, 2]))
    try:
        n, d, inl, it0 = ransac_plane(pts0, thr0, int(rs["iterations"]), float(rs["confidence"]), rng,
                                      int(rs.get("batch", 128)), up_hint=up_hint)
    except (RuntimeError, ValueError) as e:
        return _reject(f"ransac_failed:{e}", n_road_pts=int(len(pts0)))
    iters_used += it0
    s0 = h_true / d
    extras["bootstrap"] = {"h_hat": float(d), "s": float(s0), "thr_raw_m": thr0, "inliers": float(inl.mean())}
    # --- pass 1: metric range gate (Z*s0 in [z_min, z_max]) and metric inlier threshold
    pts = pts0
    if grid:
        p1 = _select(s0)
        if len(p1) < minp:
            wide = copy.deepcopy(cfg)
            wide["road_candidates"]["z_max_m"] *= 2.0
            p1 = road_candidates(points_cam, wide, exclude_mask, s0, notroad_mask)
            p1 = points_cam[p1 & road_mask] if road_mask is not None else points_cam[p1]
            extras["candidate_gate"] = "widened_z_max_x2"
        if len(p1) >= minp:
            pts = _sub(p1)
    try:
        n, d, inl, it1 = ransac_plane(pts, thr_m / s0, int(rs["iterations"]), float(rs["confidence"]), rng,
                                      int(rs.get("batch", 128)), up_hint=up_hint)
    except (RuntimeError, ValueError) as e:
        return _reject(f"ransac_failed:{e}", n_road_pts=int(len(pts)))
    iters_used += it1
    extras["ransac_iterations"] = iters_used
    s = h_true / d
    # refine: re-gate candidates in metres with the current s, LS refit on inliers
    for _ in range(int(rs.get("refine_passes", 2))):
        if grid:
            p2 = _select(s)
            if len(p2) >= minp:
                pts = _sub(p2)
        inl = np.abs(pts @ n + d) < thr_m / s
        if inl.sum() < 3:
            break
        n, d = lsq_plane(pts[inl])
        s = h_true / d
        inl = np.abs(pts @ n + d) < thr_m / s
    h_hat = float(d)
    res = (pts[inl] @ n + d) * s
    plane_rms_mm = float(np.sqrt(np.mean(res ** 2)) * 1000.0) if inl.any() else float("nan")
    inlier_ratio = float(inl.mean())
    pitch_est, roll_est = pitch_roll_from_normal(n)
    R_road = road_frame(n, yaw_deg)
    fit = SurfaceFit(n=n, d=float(h_true), h_hat=h_hat, s=float(s), R_road=R_road, quad_coeffs=None,
                     plane_rms_mm=plane_rms_mm, ring_rms_mm=None, inlier_ratio=inlier_ratio,
                     surface_model=SurfaceModel.plane, pitch_est_deg=pitch_est, n_road_pts=int(len(pts)),
                     extras=extras)
    extras["roll_est_deg"] = roll_est
    lo, hi = cfg["scale_lock"]["clip"]
    reasons = []
    if inlier_ratio < float(rs["min_inlier_ratio"]):
        reasons.append(f"plane_inlier_ratio<{rs['min_inlier_ratio']}")
    if plane_rms_mm > float(rs["max_rms_mm"]):
        reasons.append(f"plane_rms>{rs['max_rms_mm']}mm")
    if not (lo <= s <= hi):
        act = cfg["scale_lock"].get("clip_action", "auto")
        if act == "reject" or (act == "auto" and certified):
            reasons.append(f"scale_factor_outside[{lo},{hi}]")
        else:
            extras.setdefault("warnings", []).append(
                f"scale_factor {s:.2f} outside [{lo}, {hi}] (uncertified cert: wrong intrinsics/height or depth-model "
                f"domain gap; measurement kept, quality reduced)")
    if pitch_prior_deg is not None:
        dev = abs(pitch_est - pitch_prior_deg)
        extras["pitch_dev_deg"] = dev
        if dev > float(rs["max_pitch_dev_deg"]):
            action = rs.get("pitch_dev_action", "auto")
            if action == "reject" or (action == "auto" and certified):
                reasons.append(f"pitch_dev>{rs['max_pitch_dev_deg']}deg")
            else:
                extras.setdefault("warnings", []).append(f"pitch_dev {dev:.1f} deg > {rs['max_pitch_dev_deg']}")
    if reasons:
        fit.reject_reason = ";".join(reasons)
    if pothole_mask is not None and K is not None and grid and fit.ok:
        fit = with_local_quadratic(fit, points_cam * s, pothole_mask, exclude_mask, K, cfg)
    return fit


# ----------------------------------------------------------------------------- local quadratic
def _expand_centered(cc: np.ndarray, x0: float, z0: float) -> np.ndarray:
    a, b, c, d, e, f = cc
    return np.array([a - b * x0 - c * z0 + d * x0 * x0 + e * z0 * z0 + f * x0 * z0,
                     b - 2 * d * x0 - f * z0,
                     c - 2 * e * z0 - f * x0,
                     d, e, f])


def footprint_boundary_xz(fit: SurfaceFit, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Plan (x, z) of the mask boundary pixels intersected with the global plane."""
    import cv2

    from measure.camera import rays_at

    er = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    vs, us = np.nonzero(mask & ~er)
    if len(us) == 0:
        vs, us = np.nonzero(mask)
    plane_fit = copy.copy(fit)
    plane_fit.surface_model = SurfaceModel.plane
    t, p = intersect_rays(plane_fit, rays_at(K, us, vs))
    ok = np.isfinite(t)
    return p[ok][:, [0, 2]]


def with_local_quadratic(fit: SurfaceFit, points_cam_scaled: np.ndarray, pothole_mask: np.ndarray,
                         exclude_mask: Optional[np.ndarray], K: np.ndarray,
                         cfg: Optional[dict[str, Any]] = None) -> SurfaceFit:
    """PLAN 4.4.5: fit y_s(x,z) on the ring 0.3-1.2 m around the footprint (not in any mask).

    Falls back to the plane (``surface_model='plane'``) if < min_ring_px ring pixels or the
    (column-normalised) design matrix condition number exceeds max_condition. Robust: two passes
    rejecting residuals > max(3*1.4826*MAD, 5 mm).
    """
    from scipy.spatial import cKDTree

    cfg = ensure_cfg(cfg)
    qc = cfg["quadratic"]
    out = copy.copy(fit)
    out.extras = dict(fit.extras)
    bxz = footprint_boundary_xz(fit, pothole_mask, K)
    if len(bxz) < 3:
        out.extras["quadratic_fallback"] = "no_footprint"
        return out
    pr = to_road(points_cam_scaled, fit)
    excl = pothole_mask.copy()
    if exclude_mask is not None:
        excl |= exclude_mask
    valid = np.all(np.isfinite(pr), axis=-1) & ~excl & (points_cam_scaled[..., 2] > 0)
    lo, hi = bxz.min(0) - qc["ring_outer_m"], bxz.max(0) + qc["ring_outer_m"]
    valid &= (pr[..., 0] >= lo[0]) & (pr[..., 0] <= hi[0]) & (pr[..., 2] >= lo[1]) & (pr[..., 2] <= hi[1])
    # also drop points far off the plane (vehicles, kerbs): |y| < 0.15 m
    valid &= np.abs(pr[..., 1]) < 0.15
    cand = pr[valid]
    if len(cand) == 0:
        out.extras["quadratic_fallback"] = "no_ring_pixels"
        return out
    dist, _ = cKDTree(bxz).query(cand[:, [0, 2]])
    ring = cand[(dist >= qc["ring_inner_m"]) & (dist <= qc["ring_outer_m"])]
    out.extras["n_ring_px"] = int(len(ring))
    if len(ring) < int(qc["min_ring_px"]):
        out.extras["quadratic_fallback"] = f"ring_px<{qc['min_ring_px']}"
        out.ring_rms_mm = None
        return out
    x0, z0 = float(np.median(bxz[:, 0])), float(np.median(bxz[:, 1]))
    u, w, y = ring[:, 0] - x0, ring[:, 2] - z0, ring[:, 1]
    A = np.stack([np.ones_like(u), u, w, u * u, w * w, u * w], axis=1)
    colscale = np.maximum(np.abs(A).max(axis=0), 1e-12)
    An = A / colscale
    cond = float(np.linalg.cond(An))
    out.extras["quad_condition"] = cond
    if not np.isfinite(cond) or cond > float(qc["max_condition"]):
        out.extras["quadratic_fallback"] = "ill_conditioned"
        return out
    keep = np.ones(len(y), bool)
    coef = None
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(An[keep], y[keep], rcond=None)
        r = y - An @ coef
        mad = np.median(np.abs(r[keep] - np.median(r[keep])))
        thr = max(3 * 1.4826 * mad, 0.005)
        new_keep = np.abs(r) <= thr
        if new_keep.sum() < int(qc["min_ring_px"]) or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    cc = coef / colscale
    r = y[keep] - A[keep] @ cc
    out.quad_coeffs = _expand_centered(cc, x0, z0)
    out.surface_model = SurfaceModel.quadratic
    out.ring_rms_mm = float(np.sqrt(np.mean(r ** 2)) * 1000.0)
    out.extras.update(quad_centre_xz=[x0, z0], quad_centered_coeffs=cc.tolist(), n_ring_inliers=int(keep.sum()))
    return out
