"""Ground point of a detection + small metric helpers used by every geo module (PLAN 5.1).

Metric maths uses a local tangent-plane (equirectangular) approximation around a reference point, with
the WGS84 meridional / parallel radii at the reference latitude. Over the <= 200 m extents used here the
error is below 1 mm, far below GNSS noise, and it needs no pyproj transformer per call.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# WGS84 series expansions for metres per degree at latitude phi
_M_LAT = (111132.954, -559.822, 1.175)  # a0 + a1 cos 2phi + a2 cos 4phi
_M_LON = (111412.84, -93.5)  # b0 cos phi + b1 cos 3phi

EARTH_R_M = 6371008.8  # mean radius, for haversine
DEPTH_SIGMA_FRAC = 0.05  # PLAN 5.1: sigma_pos = sqrt(gps_acc^2 + (0.05 * D)^2)
DEFAULT_GPS_ACCURACY_M = 8.0


def m_per_deg(lat: float) -> tuple[float, float]:
    """(metres per degree of latitude, metres per degree of longitude) at ``lat``."""
    p = math.radians(lat)
    mlat = _M_LAT[0] + _M_LAT[1] * math.cos(2 * p) + _M_LAT[2] * math.cos(4 * p)
    mlon = _M_LON[0] * math.cos(p) + _M_LON[1] * math.cos(3 * p)
    return mlat, mlon


def to_local(lat0: float, lon0: float, lat, lon):
    """(east, north) metres of ``lat``/``lon`` (scalars or arrays) relative to the origin ``lat0``/``lon0``."""
    mlat, mlon = m_per_deg(lat0)
    return (np.asarray(lon) - lon0) * mlon, (np.asarray(lat) - lat0) * mlat


def from_local(lat0: float, lon0: float, east, north):
    """Inverse of :func:`to_local`: returns (lat, lon)."""
    mlat, mlon = m_per_deg(lat0)
    return lat0 + np.asarray(north) / mlat, lon0 + np.asarray(east) / mlon


def offset(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    """Move a point by (east, north) metres. Returns (lat, lon) floats."""
    la, lo = from_local(lat, lon, east_m, north_m)
    return float(la), float(lo)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (numpy-broadcasting)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    d = 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(d) if np.ndim(d) == 0 else d


def bearing_of(east: float, north: float) -> float:
    """Bearing (deg clockwise from north, 0-360) of a local (east, north) vector."""
    return math.degrees(math.atan2(east, north)) % 360.0


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, 0-180 deg."""
    d = abs((a - b) % 360.0)
    return 360.0 - d if d > 180.0 else d


def ground_point(lat: float, lon: float, heading_deg: Optional[float], speed_mps: Optional[float],
                 ground_xz: Optional[tuple[float, float]], gnss_latency_s: float = 0.3,
                 gps_accuracy_m: Optional[float] = DEFAULT_GPS_ACCURACY_M) -> tuple[float, float, float]:
    """Pothole ground position from the GPS fix at frame time (PLAN 5.1).

    The fix is advanced along the heading by ``gnss_latency_s * speed`` (the GNSS fix lags the frame),
    then offset by the pothole centroid in the road frame: ``z`` metres forward along the heading and
    ``x`` metres to the right of travel (``ground_xz = (x, z)`` from ``FrameMeasurement``).

    Returns ``(lat, lon, sigma_m)`` with ``sigma_m = sqrt(gps_accuracy^2 + (0.05 * D)^2)``, where
    ``D = hypot(x, z)`` is the ground range to the pothole. Without a heading no offset can be applied,
    so the raw fix is returned and ``D`` is added to sigma (the pothole is somewhere within D of the fix).
    """
    acc = DEFAULT_GPS_ACCURACY_M if gps_accuracy_m is None or not math.isfinite(gps_accuracy_m) else gps_accuracy_m
    x, z = (0.0, 0.0) if ground_xz is None else (float(ground_xz[0]), float(ground_xz[1]))
    rng = math.hypot(x, z)
    if heading_deg is None or not math.isfinite(heading_deg):
        return float(lat), float(lon), math.sqrt(acc**2 + (DEPTH_SIGMA_FRAC * rng) ** 2) + rng
    speed = 0.0 if speed_mps is None or not math.isfinite(speed_mps) else max(0.0, float(speed_mps))
    fwd = gnss_latency_s * speed + z
    h = math.radians(heading_deg)
    # forward unit = (sin h, cos h); right-of-travel unit = (cos h, -sin h)   [east, north]
    east = fwd * math.sin(h) + x * math.cos(h)
    north = fwd * math.cos(h) - x * math.sin(h)
    la, lo = offset(lat, lon, east, north)
    return la, lo, math.sqrt(acc**2 + (DEPTH_SIGMA_FRAC * rng) ** 2)
