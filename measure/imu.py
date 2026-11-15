"""IMU corroboration (PLAN 4.9): existence corroboration; depth hint only after a per-vehicle fit.

* Predicted wheel pass time: t_pass = t_frame + D / v (D = range to the pothole, v = speed).
* Wheel match: |lane_offset - wheel_track_{left,right}| < 0.35 m (cert wheel tracks, camera-axis frame).
* Jolt: peak = max |az - g| in [t_pass - 0.5 s, t_pass + 0.5 s], in g. g is estimated as the median of
  az over the whole +-1 s window (robust to phone tilt; ``gravity='nominal'`` uses 9.80665).
* Threshold: 0.25 g * (v / 8 m/s)^0.5. Corroborated iff wheel match and peak > threshold.
* Depth hint: imu_depth_mm = g_imu * peak_g only when ``cert.truth_fit.g_imu`` was fitted on >= 20
  truth potholes, else None. Enters fusion with sigma 25 mm.
* Jolt-only observations: peak > 0.35 g at speed > 15 km/h with no camera detection.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from measure.config import ensure_cfg
from measure.types import CalibCert, ImuResult, ImuWindow

G = 9.80665


def load_window_npz(path: str | Path) -> ImuWindow:
    """Read an ``imu.window_key`` .npz (contract in measure.types.ImuWindow). allow_pickle is off."""
    with np.load(path, allow_pickle=False) as z:
        kw: dict[str, Any] = {"t_mono_ms": z["t_mono_ms"].astype(np.int64).tolist(),
                              "az": z["az"].astype(float).tolist()}
        if "rate_hz" in z:
            kw["rate_hz"] = float(np.asarray(z["rate_hz"]).reshape(-1)[0])
        for k in ("ax", "ay"):
            if k in z:
                kw[k] = z[k].astype(float).tolist()
    return ImuWindow(**kw)


def save_window_npz(path: str | Path, w: ImuWindow) -> None:
    arrs: dict[str, Any] = {"t_mono_ms": np.asarray(w.t_mono_ms, np.int64), "az": np.asarray(w.az, np.float32),
                            "rate_hz": np.float32(w.rate_hz)}
    if w.ax is not None:
        arrs["ax"] = np.asarray(w.ax, np.float32)
    if w.ay is not None:
        arrs["ay"] = np.asarray(w.ay, np.float32)
    np.savez_compressed(path, **arrs)


def predict_pass_time(t_frame_ms: int, range_m: float, speed_mps: float) -> Optional[int]:
    if speed_mps is None or speed_mps <= 0.1 or range_m is None:
        return None
    return int(round(t_frame_ms + 1000.0 * range_m / speed_mps))


def speed_threshold_g(speed_mps: Optional[float], cfg: Optional[dict[str, Any]] = None) -> float:
    c = ensure_cfg(cfg)["imu"]
    v = float(speed_mps) if speed_mps else float(c["speed_ref_mps"])
    return float(c["base_threshold_g"]) * (max(v, 0.1) / float(c["speed_ref_mps"])) ** float(c["speed_exp"])


def wheel_match(lane_offset_m: Optional[float], cert: CalibCert, tol_m: float = 0.35) -> Optional[str]:
    if lane_offset_m is None:
        return None
    dl = abs(lane_offset_m - cert.wheel_track_left_m)
    dr = abs(lane_offset_m - cert.wheel_track_right_m)
    if min(dl, dr) >= tol_m:
        return None
    return "left" if dl <= dr else "right"


def peak_g(window: ImuWindow, t0_ms: float, t1_ms: float, gravity: str = "median") -> Optional[float]:
    t = np.asarray(window.t_mono_ms, float)
    az = np.asarray(window.az, float)
    if len(t) == 0:
        return None
    g = float(np.median(az)) if gravity == "median" else G
    sel = (t >= t0_ms) & (t <= t1_ms)
    if not sel.any():
        return None
    return float(np.max(np.abs(az[sel] - g)) / G)


def corroborate(window: Optional[ImuWindow], t_pass: Optional[int], lane_offset: Optional[float], cert: CalibCert,
                *, speed_mps: Optional[float] = None, cfg: Optional[dict[str, Any]] = None,
                gravity: str = "median") -> ImuResult:
    c = ensure_cfg(cfg)["imu"]
    thr = speed_threshold_g(speed_mps, cfg)
    wheel = wheel_match(lane_offset, cert, float(c["wheel_tol_m"]))
    if window is None or t_pass is None:
        return ImuResult(corroborated=False, t_pass_ms=t_pass, threshold_g=thr, wheel=wheel,
                         lane_match=wheel is not None, reason="no_imu_window" if window is None else "no_pass_time")
    half = float(c["window_s"]) * 1000.0
    pk = peak_g(window, t_pass - half, t_pass + half, gravity)
    if pk is None:
        return ImuResult(corroborated=False, t_pass_ms=t_pass, threshold_g=thr, wheel=wheel,
                         lane_match=wheel is not None, reason="pass_time_outside_window")
    corr = wheel is not None and pk > thr
    tf = cert.truth_fit
    depth = None
    if tf.g_imu is not None and tf.n_truth >= int(c["min_truth_for_gain"]):
        depth = float(tf.g_imu * pk)
    reason = None if corr else ("not_in_wheel_path" if wheel is None else "peak_below_threshold")
    return ImuResult(corroborated=corr, t_pass_ms=t_pass, peak_g=pk, threshold_g=thr, wheel=wheel,
                     lane_match=wheel is not None, imu_depth_mm=depth,
                     imu_depth_sigma_mm=float(c["hint_sigma_mm"]), reason=reason)


def detect_jolts(t_ms: np.ndarray, az: np.ndarray, speed_mps: np.ndarray | float, *,
                 cfg: Optional[dict[str, Any]] = None, min_gap_ms: int = 500,
                 gravity: str = "median") -> list[dict[str, Any]]:
    """Jolt-only events: |az - g| > 0.35 g at speed > 15 km/h; events closer than ``min_gap_ms`` merge."""
    c = ensure_cfg(cfg)["imu"]
    t = np.asarray(t_ms, float)
    a = np.asarray(az, float)
    v = np.broadcast_to(np.asarray(speed_mps, float), t.shape)
    g = float(np.median(a)) if gravity == "median" else G
    dev = np.abs(a - g) / G
    hit = (dev > float(c["jolt_only_peak_g"])) & (v * 3.6 > float(c["jolt_only_min_kmh"]))
    events: list[dict[str, Any]] = []
    for i in np.nonzero(hit)[0]:
        if events and t[i] - events[-1]["t_ms"] < min_gap_ms:
            if dev[i] > events[-1]["peak_g"]:
                events[-1].update(t_ms=int(t[i]), peak_g=float(dev[i]), speed_mps=float(v[i]))
            continue
        events.append({"t_ms": int(t[i]), "peak_g": float(dev[i]), "speed_mps": float(v[i]), "source": "jolt"})
    return events


def synth_window(t_center_ms: int, rate_hz: float = 100.0, peak_g_val: float = 0.0, at_ms: Optional[int] = None,
                 noise_g: float = 0.02, tilt_deg: float = 0.0, seed: int = 0) -> ImuWindow:
    """Test helper: +-1 s window, gravity * cos(tilt) baseline, a 40 ms half-sine jolt at ``at_ms``."""
    rng = np.random.default_rng(seed)
    n = int(2 * rate_hz) + 1
    t = t_center_ms + np.round((np.arange(n) - n // 2) * 1000.0 / rate_hz).astype(np.int64)
    az = G * math.cos(math.radians(tilt_deg)) + rng.normal(0, noise_g * G, n)
    if peak_g_val > 0 and at_ms is not None:
        bump = np.clip(1 - np.abs(t - at_ms) / 20.0, 0, None)
        az = az + peak_g_val * G * np.sin(0.5 * np.pi * bump)
    return ImuWindow(t_mono_ms=t.tolist(), az=az.tolist(), rate_hz=rate_hz)
