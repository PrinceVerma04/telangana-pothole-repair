"""Small local-tangent-plane geodesy (metres) for the simulator. Accurate to < 0.1 % over tens of km."""

from __future__ import annotations

import math

R_EARTH = 6_371_008.8
_K = math.pi / 180.0 * R_EARTH  # metres per degree of latitude


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(min(1.0, math.sqrt(a)))


class LocalFrame:
    """Equirectangular east/north metres around an origin (lat0, lon0)."""

    def __init__(self, lat0: float, lon0: float) -> None:
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self.kx = _K * math.cos(math.radians(lat0))
        self.ky = _K

    def to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        return self.lat0 + y / self.ky, self.lon0 + x / self.kx


def bearing_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compass bearing (0 = north, 90 = east) of the vector (x1, y1) -> (x2, y2) in a local frame."""
    return math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360.0


def heading_unit(heading_deg: float) -> tuple[float, float, float, float]:
    """(fwd_x, fwd_y, right_x, right_y) unit vectors (east, north) for a compass heading."""
    h = math.radians(heading_deg)
    fx, fy = math.sin(h), math.cos(h)
    return fx, fy, fy, -fx


def angdiff(a: float, b: float) -> float:
    """Signed smallest difference a - b in degrees, in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d
