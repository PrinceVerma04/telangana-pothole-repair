"""Truth ingest, accuracy per range bin, per-device fits of k_sigma and g_imu, certificate issue
(PLAN 4.11, 4.12 levels 2-3).

* ``evaluate(pairs)``: depth MAE / bias, volume & area MAPE, 2-sigma coverage, IRC class accuracy per
  range bin (3-5, 5-7, 7-10, 10-12 m).
* ``fit_k_sigma(pairs)``: maximum-likelihood k in sigma_d95^2 = k^2 * zc2 + other2, using the sigma
  decomposition every FrameMeasurement stores in ``extras['sigma_parts']`` (pothole_geom.py).
* ``fit_g_imu(peaks_g, depths_mm)``: least squares through the origin, only with >= 20 truth potholes.
* ``load_point_cloud`` / ``measure_point_cloud``: phone-LiDAR .ply/.obj -> depth/area/volume
  (method 'lidar_phone'); pure parsing, nothing from the file is executed.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from measure import irc
from measure.types import CalibCert, TruthFit

TRUTH_CSV_COLUMNS = ["pothole_ref", "device_id", "method", "depth_p95_mm", "depth_max_mm", "area_m2", "volume_l",
                     "length_m", "width_m", "lat", "lon", "measured_at", "measured_by", "mesh_key", "notes"]
RANGE_BINS = (3.0, 5.0, 7.0, 10.0, 12.0)
TRUTH_SIGMA = {"manual_gauge": {"depth_mm": 3.0}, "lidar_phone": {"volume_rel": 0.05}}


@dataclass
class TruthPair:
    truth: dict[str, Any]  # depth_p95_mm, volume_l, area_m2, width_m (any subset)
    measured: Any  # FrameMeasurement | PassMeasurement
    range_m: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def rng(self) -> Optional[float]:
        return self.range_m if self.range_m is not None else getattr(self.measured, "range_m", None)


# ----------------------------------------------------------------------------- CSV
def write_template(path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(TRUTH_CSV_COLUMNS)


def load_truth_csv(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out: dict[str, Any] = {}
            for k in TRUTH_CSV_COLUMNS:
                v = (r.get(k) or "").strip()
                if k in ("depth_p95_mm", "depth_max_mm", "area_m2", "volume_l", "length_m", "width_m", "lat", "lon"):
                    out[k] = float(v) if v else None
                else:
                    out[k] = v or None
            if out["method"] not in (None, "manual_gauge", "lidar_phone", "stereo"):
                raise ValueError(f"unknown truth method {out['method']!r}")
            rows.append(out)
    return rows


# ----------------------------------------------------------------------------- evaluation
def _bin_label(r: Optional[float], bins: Sequence[float]) -> str:
    if r is None or not np.isfinite(r):
        return "unknown"
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        if lo <= r < hi:
            return f"{lo:g}-{hi:g} m"
    return f"<{bins[0]:g} m" if r < bins[0] else f">={bins[-1]:g} m"


def evaluate(pairs: Sequence[TruthPair], bins: Sequence[float] = RANGE_BINS, k_sigma: float = 2.0) -> dict[str, Any]:
    groups: dict[str, list[TruthPair]] = {}
    for p in pairs:
        groups.setdefault(_bin_label(p.rng, bins), []).append(p)
    groups["all"] = list(pairs)
    out: dict[str, Any] = {}
    for g, ps in groups.items():
        de, ve, ae, cov, cls_ok = [], [], [], [], []
        for p in ps:
            m, t = p.measured, p.truth
            if m.depth_p95_mm is not None and t.get("depth_p95_mm") is not None:
                e = m.depth_p95_mm - t["depth_p95_mm"]
                de.append(e)
                if m.depth_sigma_mm:
                    cov.append(abs(e) <= k_sigma * m.depth_sigma_mm)
            if m.volume_l is not None and t.get("volume_l"):
                ve.append(abs(m.volume_l - t["volume_l"]) / t["volume_l"])
            if m.area_m2 and t.get("area_m2"):
                ae.append(abs(m.area_m2 - t["area_m2"]) / t["area_m2"])
            if t.get("depth_p95_mm") is not None and t.get("width_m") is not None and m.depth_p95_mm is not None:
                cls_ok.append(irc.severity(m.depth_p95_mm, m.width_m * 1000.0) ==
                              irc.severity(t["depth_p95_mm"], t["width_m"] * 1000.0))
        out[g] = {
            "n": len(ps),
            "depth_mae_mm": float(np.mean(np.abs(de))) if de else None,
            "depth_bias_mm": float(np.mean(de)) if de else None,
            "volume_mape": float(np.mean(ve)) if ve else None,
            "area_mape": float(np.mean(ae)) if ae else None,
            "depth_coverage_2sigma": float(np.mean(cov)) if cov else None,
            "irc_class_accuracy": float(np.mean(cls_ok)) if cls_ok else None,
        }
    return out


# ----------------------------------------------------------------------------- k_sigma fit
def _parts(m: Any) -> Optional[tuple[float, float]]:
    sp = (getattr(m, "extras", None) or {}).get("sigma_parts")
    if not sp:
        return None
    return float(sp["zc2"]), float(sp["other2"])


def fit_k_sigma(pairs: Sequence[TruthPair], k_bounds: tuple[float, float] = (0.0, 5e-3)) -> dict[str, Any]:
    """MLE of k for depth_p95 errors: e_j ~ N(0, k^2 zc2_j + other2_j) (units: metres internally).

    Returns k, n, the 2-sigma coverage and z-RMS before (stored k) and after the fit. If the errors are
    smaller than the k-independent floor (other2) the MLE is k = lower bound -- reported as such.
    """
    from scipy.optimize import minimize_scalar

    E, A, B, k0 = [], [], [], []
    for p in pairs:
        m = p.measured
        pr = _parts(m)
        if pr is None or m.depth_p95_mm is None or p.truth.get("depth_p95_mm") is None:
            continue
        E.append((m.depth_p95_mm - p.truth["depth_p95_mm"]) / 1000.0)
        A.append(pr[0])
        B.append(pr[1])
        k0.append(float(m.extras["sigma_parts"]["k_sigma"]))
    if len(E) < 3:
        raise ValueError(f"need >= 3 truth pairs with sigma_parts, got {len(E)}")
    e, a, b = np.array(E), np.array(A), np.array(B)

    def nll(k: float) -> float:
        v = k * k * a + b
        return float(np.sum(np.log(v) + e * e / v))

    res = minimize_scalar(nll, bounds=k_bounds, method="bounded", options={"xatol": 1e-8})
    k = float(res.x)

    def diag(kk: np.ndarray | float) -> tuple[float, float]:
        s = np.sqrt(np.asarray(kk) ** 2 * a + b)
        return float(np.mean(np.abs(e) <= 2 * s)), float(np.sqrt(np.mean((e / s) ** 2)))

    cov0, z0 = diag(np.array(k0))
    cov1, z1 = diag(k)
    floor_limited = k <= k_bounds[0] + 1e-6  # errors already explained by the k-independent terms
    return {"k_sigma": k, "n": len(e), "coverage_2sigma_before": cov0, "z_rms_before": z0,
            "coverage_2sigma_after": cov1, "z_rms_after": z1, "floor_limited": floor_limited,
            "depth_mae_mm": float(np.mean(np.abs(e)) * 1000.0)}


def fit_g_imu(peaks_g: Sequence[float], depths_mm: Sequence[float], min_n: int = 20) -> dict[str, Any]:
    """depth_mm ~ g_imu * peak_g (through origin). None if fewer than ``min_n`` truth potholes (PLAN 4.9)."""
    x = np.asarray(peaks_g, float)
    y = np.asarray(depths_mm, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    if len(x) < min_n:
        return {"g_imu": None, "n": int(len(x)), "reason": f"need >= {min_n} truth potholes"}
    g = float(np.sum(x * y) / np.sum(x * x))
    resid = y - g * x
    return {"g_imu": g, "n": int(len(x)), "rmse_mm": float(np.sqrt(np.mean(resid ** 2))),
            "r2": float(1 - np.sum(resid ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))}


def issue_cert(cert: CalibCert, *, k_sigma: Optional[float] = None, g_imu: Optional[float] = None,
               n_truth: int = 0, depth_mae_mm: Optional[float] = None, volume_mape: Optional[float] = None) -> CalibCert:
    tf = cert.truth_fit.to_dict()
    tf.update(k_sigma=float(k_sigma) if k_sigma is not None else tf["k_sigma"], g_imu=g_imu, n_truth=int(n_truth),
              depth_mae_mm=depth_mae_mm, volume_mape=volume_mape,
              fitted_at=datetime.now(timezone.utc).isoformat())
    d = cert.to_dict()
    d["truth_fit"] = TruthFit.from_dict(tf).to_dict()
    return CalibCert.from_dict(d)


# ----------------------------------------------------------------------------- phone LiDAR meshes
def load_point_cloud(path: str | Path) -> np.ndarray:
    """Vertices (N, 3) from ASCII/binary-little-endian PLY or OBJ ('v x y z' lines). Parsing only."""
    p = Path(path)
    raw = p.read_bytes()
    if p.suffix.lower() == ".obj":
        pts = [list(map(float, ln.split()[1:4])) for ln in raw.decode("utf-8", "replace").splitlines()
               if ln.startswith("v ")]
        return np.asarray(pts, float)
    if not raw.startswith(b"ply"):
        raise ValueError("not a PLY/OBJ file")
    end = raw.index(b"end_header") + len(b"end_header")
    while raw[end:end + 1] in (b"\r", b"\n"):
        end += 1
    header = raw[:end].decode("ascii", "replace").splitlines()
    fmt = next(ln.split()[1] for ln in header if ln.startswith("format"))
    n_vert, props, in_vertex = 0, [], False
    tmap = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
            "char": "i1", "int8": "i1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2", "int": "i4",
            "int32": "i4", "uint": "u4", "uint32": "u4"}
    for ln in header:
        t = ln.split()
        if t[:2] == ["element", "vertex"]:
            n_vert, in_vertex = int(t[2]), True
        elif t[:1] == ["element"]:
            in_vertex = False
        elif t[:1] == ["property"] and in_vertex:
            if t[1] == "list":
                raise ValueError("list properties in the vertex element are not supported")
            props.append((t[2], tmap[t[1]]))
    names = [n for n, _ in props]
    if fmt == "ascii":
        lines = raw[end:].decode("ascii", "replace").split("\n")[:n_vert]
        arr = np.array([list(map(float, ln.split()[:len(props)])) for ln in lines], float)
        return arr[:, [names.index("x"), names.index("y"), names.index("z")]]
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format {fmt}")
    dt = np.dtype([(n, "<" + t) for n, t in props])
    v = np.frombuffer(raw, dtype=dt, count=n_vert, offset=end)
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(float)


def measure_point_cloud(points: np.ndarray, cell_m: float = 0.005, min_depth_mm: float = 3.0,
                        ransac_thr_m: float = 0.004, seed: int = 0) -> dict[str, Any]:
    """Cavity metrics from a dense surface scan of a pothole and its surroundings.

    RANSAC plane on all points (the surrounding road dominates), heights above the plane, plan grid of
    ``cell_m`` (mean height per cell), cavity = connected cells deeper than ``min_depth_mm`` containing the
    deepest cell. Returns depth_p95/max (p99)/mean (mm), area (m^2), volume (l), method 'lidar_phone'.
    """
    import cv2

    from measure.surface import ransac_plane

    pts = np.asarray(points, float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    rng = np.random.default_rng(seed)
    sub = pts if len(pts) <= 50000 else pts[rng.choice(len(pts), 50000, replace=False)]
    n, d, _, _ = ransac_plane(sub, ransac_thr_m, 1000, 0.999, rng)
    # orient n so the cavity (the side with more points well off the plane) has negative height
    hgt = pts @ n + d
    if np.sum(hgt > 0.01) > np.sum(hgt < -0.01):
        n, d, hgt = -n, -d, -hgt
    # in-plane basis
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = a - (a @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    u, v = pts @ e1, pts @ e2
    iu = np.floor((u - u.min()) / cell_m).astype(int)
    iv = np.floor((v - v.min()) / cell_m).astype(int)
    H, W = iv.max() + 1, iu.max() + 1
    s = np.zeros((H, W))
    c = np.zeros((H, W))
    np.add.at(s, (iv, iu), hgt)
    np.add.at(c, (iv, iu), 1)
    with np.errstate(invalid="ignore"):
        hmap = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    depth = np.where(np.isfinite(hmap), np.clip(-hmap, 0, None), 0.0)
    cav = (depth * 1000.0 > min_depth_mm).astype(np.uint8)
    nlab, lab = cv2.connectedComponents(cav, connectivity=8)
    if nlab <= 1:
        return {"depth_p95_mm": 0.0, "depth_max_mm": 0.0, "depth_mean_mm": 0.0, "area_m2": 0.0, "volume_l": 0.0,
                "method": "lidar_phone"}
    deepest = np.unravel_index(np.argmax(depth), depth.shape)
    region = lab == lab[deepest]
    # fill unobserved holes inside the region (cells with no points) by closing
    region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    dl = depth[region]
    area = float(region.sum() * cell_m ** 2)
    vol = float(dl.sum() * cell_m ** 2)
    srt = np.sort(dl)
    return {"depth_p95_mm": float(np.quantile(srt, 0.95) * 1000.0), "depth_max_mm": float(np.quantile(srt, 0.99) * 1000.0),
            "depth_mean_mm": vol / area * 1000.0 if area else 0.0, "area_m2": area, "volume_l": vol * 1000.0,
            "method": "lidar_phone", "volume_sigma_l": 0.05 * vol * 1000.0, "n_points": int(len(pts))}


def k_from_coverage(errors_mm: Sequence[float], zc2: Sequence[float], other2: Sequence[float],
                    target: float = 0.95) -> float:
    """Alternative to MLE: smallest k reaching ``target`` 2-sigma coverage (bisection)."""
    e = np.asarray(errors_mm, float) / 1000.0
    a, b = np.asarray(zc2, float), np.asarray(other2, float)
    lo, hi = 0.0, 0.05
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.mean(np.abs(e) <= 2 * np.sqrt(mid * mid * a + b)) >= target:
            hi = mid
        else:
            lo = mid
    return hi if math.isfinite(hi) else float("nan")
