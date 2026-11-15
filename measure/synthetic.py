"""Synthetic road scenes with exact ground truth (PLAN 4.12 level 1).

World/road frame: X right, Y up, Z forward; origin on the road surface directly below the camera;
camera centre at (0, h, 0) looking forward and pitched down by ``pitch_deg``.

Road: plane ``y = 0`` or a parabolic crown ``y = -alpha*((x - xc)^2 - xc^2)`` whose cross-slope is
``crown_slope`` at ``crown_halfwidth`` from the crown line ``x = xc`` (so 2 % at the lane edge).

Cavity: spherical cap of aperture radius ``a`` and depth ``d`` below the road at its centre. The
renderer ray-casts every pixel exactly (road first, then the sphere exit point if the road hit lies
inside the ball), so self-occlusion of the near wall at grazing angles is physically correct.
Truth is integrated on a 1 mm plan grid (and checked against the analytic cap volume for the plane).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from measure.camera import pixel_rays, world_to_camera_rotation
from measure.types import CalibCert

SKY_DEPTH_M = 80.0  # DA-V2 metric outdoor max_depth; sky pixels get this


@dataclass
class Cap:
    x: float  # lateral centre (m, +right)
    z: float  # forward centre (m)
    a: float  # aperture radius (m)
    d: float  # depth at centre (m)

    @property
    def R(self) -> float:
        return (self.a ** 2 + self.d ** 2) / (2.0 * self.d)

    def analytic_volume_m3(self) -> float:
        return math.pi * self.d ** 2 * (3 * self.R - self.d) / 3.0


@dataclass
class Box:
    """Cylindrical 'box' pothole: vertical walls, flat floor at depth d (worst case for occlusion)."""

    x: float
    z: float
    a: float  # radius (m)
    d: float  # depth (m)

    def analytic_volume_m3(self) -> float:
        return math.pi * self.a ** 2 * self.d


@dataclass
class SceneSpec:
    h: float = 1.4
    pitch_deg: float = 12.0
    roll_deg: float = 0.0
    W: int = 1280
    H: int = 720
    fx: float = 1280.0
    fy: float = 1280.0
    cx: Optional[float] = None
    cy: Optional[float] = None
    road: str = "plane"  # plane | crown
    crown_slope: float = 0.02
    crown_halfwidth: float = 3.5
    crown_center_x: float = 1.75
    caps: list[Any] = field(default_factory=list)  # Cap | Box
    texture_seed: int = 7
    texture_offset_x: float = 0.0  # world x offset of the texture (render a stereo right view by shifting)

    @property
    def K(self) -> np.ndarray:
        cx = self.W / 2.0 if self.cx is None else self.cx
        cy = self.H / 2.0 if self.cy is None else self.cy
        return np.array([[self.fx, 0, cx], [0, self.fy, cy], [0, 0, 1.0]])

    @property
    def alpha(self) -> float:
        return 0.0 if self.road == "plane" else self.crown_slope / (2.0 * self.crown_halfwidth)

    def road_y(self, x: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x, float)
        if self.road == "plane":
            return np.zeros_like(x)
        xc = self.crown_center_x
        return -self.alpha * ((x - xc) ** 2 - xc ** 2)

    def cert(self, calib_id: str = "synthetic") -> CalibCert:
        K = self.K
        return CalibCert(calib_id=calib_id, device_id="synthetic", vehicle_kind="sim", fx=self.fx, fy=self.fy,
                         cx=float(K[0, 2]), cy=float(K[1, 2]), img_w=self.W, img_h=self.H, h_true_m=self.h,
                         pitch_deg=self.pitch_deg, roll_deg=self.roll_deg, certified=False,
                         notes="synthetic scene (measure/synthetic.py)")


@dataclass
class RenderResult:
    depth: np.ndarray  # (H, W) camera-z depth, metres
    masks: list[np.ndarray]  # per cap (H, W) bool aperture masks
    truth: list[dict[str, Any]]
    image: Optional[np.ndarray]  # (H, W, 3) uint8 BGR or None
    spec: SceneSpec
    points_world: np.ndarray  # (H, W, 3) visible surface points, world frame


def _cap_centre_world(spec: SceneSpec, cap: Cap) -> np.ndarray:
    y0 = float(spec.road_y(np.array([cap.x]))[0])
    return np.array([cap.x, y0 - cap.d + cap.R, cap.z])


def render(spec: SceneSpec, with_image: bool = True) -> RenderResult:
    K = spec.K
    H, W = spec.H, spec.W
    R_wc = world_to_camera_rotation(spec.pitch_deg, spec.roll_deg)  # world -> camera
    d_cam = pixel_rays(K, H, W)
    r = d_cam @ R_wc  # (H,W,3) world ray directions (row-vector form of R_wc.T @ d)
    C = np.array([0.0, spec.h, 0.0])
    rx, ry, rz = r[..., 0], r[..., 1], r[..., 2]

    # --- road intersection: t = camera z (since d_cam.z == 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        if spec.road == "plane":
            t = np.where(ry < -1e-9, -spec.h / ry, np.inf)
        else:
            al, xc = spec.alpha, spec.crown_center_x
            A2 = al * rx ** 2
            B = ry - 2 * al * xc * rx
            Cc = spec.h
            disc = B ** 2 - 4 * A2 * Cc
            sq = np.sqrt(np.maximum(disc, 0))
            # A2 >= 0 and Cc > 0: both roots share the sign of -B; the smaller positive root in the
            # numerically stable form 2C/(-B + sq) (reduces to -C/B when A2 -> 0).
            t = np.where((B < -1e-12) & (disc >= 0), 2 * Cc / (-B + sq), np.inf)
    t = np.where(np.isfinite(t) & (t < SKY_DEPTH_M), t, np.inf)
    P = C + np.where(np.isfinite(t), t, 0.0)[..., None] * r
    masks = []
    for cap in spec.caps:
        if isinstance(cap, Box):
            y0 = float(spec.road_y(np.array([cap.x]))[0])
            inside = np.isfinite(t) & ((P[..., 0] - cap.x) ** 2 + (P[..., 2] - cap.z) ** 2 < cap.a ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_floor = np.where(ry < 0, (y0 - cap.d - spec.h) / ry, np.inf)
            # exit from the vertical cylinder (2D circle in x-z)
            ox, oz = -cap.x, -cap.z
            a2 = rx * rx + rz * rz
            b = 2 * (rx * ox + rz * oz)
            c = ox * ox + oz * oz - cap.a ** 2
            t_wall = (-b + np.sqrt(np.maximum(b * b - 4 * a2 * c, 0))) / (2 * a2)
            t = np.where(inside, np.minimum(t_floor, t_wall), t)
            masks.append(inside)
            continue
        S = _cap_centre_world(spec, cap)
        CS = C - S
        inside = np.isfinite(t) & (np.sum((P - S) ** 2, axis=-1) < cap.R ** 2)
        a2 = np.sum(r * r, axis=-1)
        b = 2 * np.sum(r * CS, axis=-1)
        c = float(CS @ CS - cap.R ** 2)
        disc = b * b - 4 * a2 * c
        t_exit = (-b + np.sqrt(np.maximum(disc, 0))) / (2 * a2)
        t = np.where(inside, t_exit, t)
        masks.append(inside)
    P = C + np.where(np.isfinite(t), t, SKY_DEPTH_M)[..., None] * r
    depth = np.where(np.isfinite(t), t, SKY_DEPTH_M)
    img = _shade(spec, P, depth, masks) if with_image else None
    truth = [cap_truth(spec, cap) for cap in spec.caps]
    return RenderResult(depth=depth, masks=masks, truth=truth, image=img, spec=spec, points_world=P)


_TEX_CACHE: dict[int, np.ndarray] = {}


def _texture(seed: int) -> np.ndarray:
    if seed not in _TEX_CACHE:
        import cv2

        rng = np.random.default_rng(seed)
        base = rng.normal(size=(1024, 1024)).astype(np.float32)
        t = cv2.GaussianBlur(base, (0, 0), 1.2) * 1.0 + cv2.GaussianBlur(base, (0, 0), 6.0) * 3.0
        t = (t - t.mean()) / (t.std() + 1e-9)
        _TEX_CACHE[seed] = t
    return _TEX_CACHE[seed]


def _shade(spec: SceneSpec, P: np.ndarray, depth: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Procedural asphalt texture (5 mm texels, 5.12 m tile) -> BGR uint8. Used for stereo/blur/luma tests."""
    tex = _texture(spec.texture_seed)
    n = tex.shape[0]
    gx = np.mod((P[..., 0] + spec.texture_offset_x) / 0.005, n)
    gz = np.mod(P[..., 2] / 0.005, n)
    i0, j0 = np.floor(gz).astype(int) % n, np.floor(gx).astype(int) % n
    fz, fx = gz - np.floor(gz), gx - np.floor(gx)
    i1, j1 = (i0 + 1) % n, (j0 + 1) % n
    v = (tex[i0, j0] * (1 - fz) * (1 - fx) + tex[i1, j0] * fz * (1 - fx)
         + tex[i0, j1] * (1 - fz) * fx + tex[i1, j1] * fz * fx)
    g = 110 + 22 * v
    for m in masks:
        g = np.where(m, g * 0.7, g)
    g = np.where(depth >= SKY_DEPTH_M - 1e-6, 205.0, g)
    g = np.clip(g, 0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    c = np.cumsum(ws) - 0.5 * ws
    c /= ws.sum()
    return float(np.interp(q, c, xs))


def cap_truth(spec: SceneSpec, cap: Any, cell_m: float = 0.001) -> dict[str, Any]:
    """Exact truth on a 1 mm plan grid: area, volume, depth_p95/p99/mean (area weighted), L, W."""
    half = cap.a * 1.2 + 0.02
    xs = np.arange(cap.x - half, cap.x + half, cell_m) + cell_m / 2
    zs = np.arange(cap.z - half, cap.z + half, cell_m) + cell_m / 2
    X, Z = np.meshgrid(xs, zs)
    if isinstance(cap, Box):
        inside = (X - cap.x) ** 2 + (Z - cap.z) ** 2 < cap.a ** 2
        y0 = float(spec.road_y(np.array([cap.x]))[0])
        yf = np.where(inside, y0 - cap.d, np.inf)
    else:
        S = _cap_centre_world(spec, cap)
        r2 = (X - S[0]) ** 2 + (Z - S[2]) ** 2
        inside = r2 < cap.R ** 2
        yf = np.where(inside, S[1] - np.sqrt(np.maximum(cap.R ** 2 - r2, 0)), np.inf)
    delta = spec.road_y(X) - yf
    cav = inside & (delta > 0)
    dl = delta[cav]
    w = np.full(dl.shape, cell_m ** 2)
    area = float(cav.sum() * cell_m ** 2)
    vol = float(dl.sum() * cell_m ** 2)
    zz, xx = Z[cav], X[cav]
    out = {
        "area_m2": area,
        "volume_l": vol * 1000.0,
        "volume_l_analytic": cap.analytic_volume_m3() * 1000.0 if spec.road == "plane" else None,
        "depth_p95_mm": _weighted_quantile(dl, w, 0.95) * 1000.0,
        "depth_max_mm": _weighted_quantile(dl, w, 0.99) * 1000.0,
        "depth_mean_mm": vol / area * 1000.0,
        "depth_centre_mm": cap.d * 1000.0,
        "length_m": float(zz.max() - zz.min() + cell_m),
        "width_m": float(xx.max() - xx.min() + cell_m),
        "ground_xz": (cap.x, cap.z),
        "range_m": None,
    }
    return out


def add_depth_noise(depth: np.ndarray, k: float, rng: np.random.Generator, mode: str = "mixed",
                    smooth_frac: float = 0.5, corr_px: int = 24, scale_sigma: float = 0.0,
                    valid_max: float = SKY_DEPTH_M - 1e-6,
                    region: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Monocular-like depth noise: sigma_Z = k Z^2 (PLAN 4.7), optional global scale error.

    mode: 'iid' per-pixel Gaussian; 'smooth' spatially correlated field (correlation ~corr_px);
    'mixed' sqrt(1-f^2)*iid + f*smooth with f = smooth_frac (unit total variance per pixel).
    A global multiplicative scale error ~ N(1, scale_sigma) is applied first (the height scale-lock
    must cancel it).
    """
    import cv2

    H, W = depth.shape
    g = 1.0 + (rng.normal() * scale_sigma if scale_sigma > 0 else 0.0)
    if mode == "region_offset":
        # one correlated draw over ``region`` (e.g. the cavity): the model mis-estimates the whole relief by
        # eps * k * Z^2 -- exactly the error the sigma_d95 k-term describes; road stays exact (scale g only).
        if region is None:
            raise ValueError("region_offset noise needs region=")
        eps = rng.normal()
        out = np.where(depth < valid_max, depth * g, depth)
        out = np.where(region, out + eps * k * depth ** 2, out)
        return out, {"scale": g, "mode": mode, "k": k, "eps": eps}
    iid = rng.normal(size=(H, W))
    if mode == "iid":
        field = iid
    else:
        lo = rng.normal(size=(max(2, H // corr_px + 2), max(2, W // corr_px + 2))).astype(np.float32)
        sm = cv2.resize(lo, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float64)
        sm /= sm.std() + 1e-12
        if mode == "smooth":
            field = sm
        else:
            f = float(np.clip(smooth_frac, 0, 1))
            field = math.sqrt(1 - f * f) * iid + f * sm
    valid = depth < valid_max
    out = np.where(valid, depth * g + field * k * depth ** 2, depth)
    out = np.where(valid, np.maximum(out, 0.05), out)
    return out, {"scale": g, "mode": mode, "k": k}


def jitter_mask(mask: np.ndarray, rng: np.random.Generator, p: float = 1.0 / 3.0) -> np.ndarray:
    """Simulate a +-1 px segmentation boundary error: erode or dilate by one pixel with prob p each."""
    import cv2

    u = rng.random()
    k = np.ones((3, 3), np.uint8)
    if u < p:
        return cv2.erode(mask.astype(np.uint8), k).astype(bool)
    if u < 2 * p:
        return cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    return mask.copy()


def bowl_scene(range_m: float, *, h: float = 1.4, pitch_deg: float = 12.0, a: float = 0.4, d: float = 0.05,
               lateral: float = 0.0, road: str = "plane", W: int = 1280, H: int = 720, fx: float = 1280.0,
               **kw: Any) -> SceneSpec:
    """One spherical-cap pothole centred ``range_m`` ahead (plan distance along Z)."""
    return SceneSpec(h=h, pitch_deg=pitch_deg, W=W, H=H, fx=fx, fy=fx, road=road,
                     caps=[Cap(x=lateral, z=range_m, a=a, d=d)], **kw)


def stereo_pair(spec: SceneSpec, baseline_m: float) -> tuple[RenderResult, RenderResult]:
    """Rectified stereo pair: left = ``spec``; right camera displaced +baseline along camera/world x
    (rendered by shifting the scene by -baseline and offsetting the texture). Plane roads only (a crown
    centre would move too, which is handled by shifting crown_center_x)."""
    import copy

    right = copy.deepcopy(spec)
    for c in right.caps:
        c.x -= baseline_m
    right.crown_center_x -= baseline_m
    right.texture_offset_x += baseline_m
    if spec.roll_deg != 0:
        raise ValueError("stereo_pair assumes zero roll (rectified along image rows)")
    return render(spec), render(right)


def cap_visibility(spec: SceneSpec, cap: Cap) -> dict[str, float]:
    """Is the near wall self-occluded? Compares the grazing angle of the ray through the near rim with
    the wall slope at the rim (plane road, central slice)."""
    z_near = cap.z - cap.a
    dist = math.hypot(z_near, cap.x)
    graze = math.degrees(math.atan2(spec.h, dist))
    rim_slope = math.degrees(math.asin(min(1.0, cap.a / cap.R)))
    return {"grazing_deg": graze, "rim_slope_deg": rim_slope, "occluded": float(rim_slope > graze)}
