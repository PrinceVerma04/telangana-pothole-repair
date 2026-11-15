"""Small shared helpers: UTC/IST time, haversine, bbox parsing, slippy-tile maths."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), name="IST")
EARTH_R_M = 6_371_008.8


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_ist(dt: datetime | None) -> datetime | None:
    dt = as_utc(dt)
    return dt.astimezone(IST) if dt else None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_M * math.asin(min(1.0, math.sqrt(a)))


def radius_box(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) box enclosing a circle; use before an exact haversine filter."""
    dlat = math.degrees(radius_m / EARTH_R_M)
    dlon = math.degrees(radius_m / (EARTH_R_M * max(0.01, math.cos(math.radians(lat)))))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def parse_bbox(s: str | None) -> tuple[float, float, float, float] | None:
    """``"minlon,minlat,maxlon,maxlat"`` -> tuple; raises ValueError on malformed input."""
    if s is None or s == "":
        return None
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be minlon,minlat,maxlon,maxlat")
    x1, y1, x2, y2 = parts
    if not (-180 <= x1 <= 180 and -180 <= x2 <= 180 and -90 <= y1 <= 90 and -90 <= y2 <= 90):
        raise ValueError("bbox out of range")
    if x1 > x2 or y1 > y2:
        raise ValueError("bbox min must be <= max")
    return x1, y1, x2, y2


def tile_bbox(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Web-Mercator slippy tile -> (min_lon, min_lat, max_lon, max_lat)."""
    n = 2 ** z

    def lon(xx: float) -> float:
        return xx / n * 360.0 - 180.0

    def lat(yy: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))

    return lon(x), lat(y + 1), lon(x + 1), lat(y)


def bbox_of_wkt(wkt: str | None) -> tuple[float, float, float, float] | None:
    if not wkt:
        return None
    from shapely import wkt as swkt

    g = swkt.loads(wkt)
    if g.is_empty:
        return None
    return tuple(float(v) for v in g.bounds)  # type: ignore[return-value]
