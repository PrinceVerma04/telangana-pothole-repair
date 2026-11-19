"""GTFS route shapes for the simulator: load one route from ``gtfs_routes.json``, clean it, cut a window.

``gtfs_routes.json`` (WP10 geo build) is a list of ``{route_id, short_name, long_name, shape: [[lon, lat], ...],
trips_per_day}``. Only the requested routes are kept in memory (the 454k road segments are never read here).

TGSRTC shapes contain digitising artefacts: repeated points and one-point "spikes" where the line jumps a few
metres back and forth. A bus cannot drive those, and they flip the heading the map-matcher uses, so
``clean_shape`` removes duplicates (< 0.5 m) and spikes (turn > 150 deg on a leg < 60 m) before driving.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from edge.sim.geodesy import LocalFrame, bearing_deg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GTFS = PROJECT_ROOT / "datasets" / "extracted" / "geo" / "gtfs_routes.json"
SPIKE_TURN_DEG = 150.0
SPIKE_MAX_LEG_M = 60.0
DUP_M = 0.5


@lru_cache(maxsize=2)
def _load_all(path: str) -> tuple[dict[str, Any], ...]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of routes")
    return tuple(data)


def load_routes_json(path: Path | str = DEFAULT_GTFS) -> list[dict[str, Any]]:
    return list(_load_all(str(Path(path).resolve())))


def find_route(route_id: str, path: Path | str = DEFAULT_GTFS) -> dict[str, Any]:
    """The route whose ``route_id`` (exact) or else ``short_name`` equals ``route_id``. KeyError with the
    closest ids when it does not exist (route ids must exist in gtfs_routes.json)."""
    routes = load_routes_json(path)
    for key in ("route_id", "short_name"):
        for r in routes:
            if str(r.get(key)) == route_id:
                return r
    near = sorted({str(r.get("route_id")) for r in routes if str(r.get("route_id", "")).startswith(route_id[:2])})[:12]
    raise KeyError(f"route {route_id!r} not in {path}; ids starting like it: {near}")


def _turn(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    """Absolute heading change (deg) at b for the path a -> b -> c."""
    h1 = math.atan2(bx - ax, by - ay)
    h2 = math.atan2(cx - bx, cy - by)
    d = math.degrees(abs((h2 - h1 + math.pi) % (2 * math.pi) - math.pi))
    return d


def clean_shape(lonlat: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Drop duplicate points and one-point spikes (see module doc). Input/output ``[(lon, lat), ...]``."""
    pts = [(float(p[0]), float(p[1])) for p in lonlat if p is not None and len(p) >= 2]
    if len(pts) < 2:
        return pts
    fr = LocalFrame(pts[0][1], pts[0][0])
    xy = [fr.to_xy(la, lo) for lo, la in pts]
    keep_xy, keep = [xy[0]], [pts[0]]
    for p, q in zip(pts[1:], xy[1:], strict=True):
        if math.hypot(q[0] - keep_xy[-1][0], q[1] - keep_xy[-1][1]) >= DUP_M:
            keep_xy.append(q)
            keep.append(p)
    changed = True
    while changed and len(keep) >= 3:
        changed = False
        i = 1
        while i < len(keep) - 1:
            a, b, c = keep_xy[i - 1], keep_xy[i], keep_xy[i + 1]
            leg = min(math.hypot(b[0] - a[0], b[1] - a[1]), math.hypot(c[0] - b[0], c[1] - b[1]))
            if _turn(*a, *b, *c) > SPIKE_TURN_DEG and leg < SPIKE_MAX_LEG_M:
                del keep[i], keep_xy[i]
                changed = True
                continue
            i += 1
    return keep


@dataclass
class Route:
    """A drivable polyline in a local metric frame. ``chainage`` in metres from the first point."""

    route_id: str
    name: str
    lonlat: list[tuple[float, float]]
    trips_per_day: float = 0.0

    def __post_init__(self) -> None:
        if len(self.lonlat) < 2:
            raise ValueError(f"route {self.route_id}: need >= 2 points")
        self.frame = LocalFrame(self.lonlat[0][1], self.lonlat[0][0])
        self.xy = [self.frame.to_xy(la, lo) for lo, la in self.lonlat]
        self.cum = [0.0]
        for (x1, y1), (x2, y2) in zip(self.xy, self.xy[1:], strict=False):
            self.cum.append(self.cum[-1] + math.hypot(x2 - x1, y2 - y1))

    @property
    def length_m(self) -> float:
        return self.cum[-1]

    def xy_at(self, c: float) -> tuple[float, float]:
        c = min(max(c, 0.0), self.length_m)
        i = max(0, min(len(self.cum) - 2, bisect.bisect_right(self.cum, c) - 1))
        seg = self.cum[i + 1] - self.cum[i]
        t = 0.0 if seg <= 0 else (c - self.cum[i]) / seg
        (x1, y1), (x2, y2) = self.xy[i], self.xy[i + 1]
        return x1 + t * (x2 - x1), y1 + t * (y2 - y1)

    def latlon_at(self, c: float) -> tuple[float, float]:
        return self.frame.to_latlon(*self.xy_at(c))

    def heading_at(self, c: float, half_window_m: float = 5.0) -> float:
        """Compass heading of travel at chainage ``c`` (bearing over +-``half_window_m``)."""
        a = max(0.0, c - half_window_m)
        b = min(self.length_m, c + half_window_m)
        if b - a < 0.5:
            a, b = max(0.0, b - 1.0), min(self.length_m, a + 1.0)
        return bearing_deg(*self.xy_at(a), *self.xy_at(b))

    def sample(self, step_m: float) -> list[tuple[float, float, float]]:
        """(lat, lon, heading) every ``step_m`` along the route (both ends included)."""
        n = max(1, int(self.length_m // step_m))
        cs = [i * step_m for i in range(n + 1)]
        if self.length_m - cs[-1] > 1e-6:
            cs.append(self.length_m)
        return [(*self.latlon_at(c), self.heading_at(c)) for c in cs]

    def window(self, start_m: float, length_m: Optional[float]) -> "Route":
        """Sub-route from ``start_m`` for ``length_m`` metres (None = to the end)."""
        start = min(max(0.0, start_m), max(0.0, self.length_m - 1.0))
        end = self.length_m if length_m is None else min(self.length_m, start + max(1.0, length_m))
        pts = [self.latlon_at(start)]
        for c, (lo, la) in zip(self.cum, self.lonlat, strict=True):
            if start < c < end:
                pts.append((la, lo))
        pts.append(self.latlon_at(end))
        return Route(self.route_id, self.name, [(lo, la) for la, lo in pts], self.trips_per_day)

    def bbox(self, pad_m: float = 0.0) -> tuple[float, float, float, float]:
        lons = [p[0] for p in self.lonlat]
        lats = [p[1] for p in self.lonlat]
        dlat = pad_m / self.frame.ky
        dlon = pad_m / self.frame.kx
        return min(lons) - dlon, min(lats) - dlat, max(lons) + dlon, max(lats) + dlat


def route_from_record(rec: dict[str, Any]) -> Route:
    return Route(str(rec["route_id"]), str(rec.get("long_name") or rec.get("short_name") or rec["route_id"]),
                 clean_shape(rec["shape"]), float(rec.get("trips_per_day") or 0.0))


def load_route(route_id: str, path: Path | str = DEFAULT_GTFS, start_m: float = 0.0,
               length_m: Optional[float] = None) -> Route:
    r = route_from_record(find_route(route_id, path))
    return r.window(start_m, length_m) if (start_m > 0 or length_m is not None) else r


def outside_stretches(route: Route, contains: Any, step_m: float = 25.0, min_len_m: float = 800.0) -> \
        list[tuple[float, float]]:
    """Chainage intervals [(start, end)] where ``contains(lat, lon)`` is False for >= ``min_len_m``."""
    out: list[tuple[float, float]] = []
    start: Optional[float] = None
    c = 0.0
    while c <= route.length_m:
        la, lo = route.latlon_at(c)
        inside = bool(contains(la, lo))
        if not inside and start is None:
            start = c
        if inside and start is not None:
            if c - start >= min_len_m:
                out.append((start, c))
            start = None
        c += step_m
    if start is not None and route.length_m - start >= min_len_m:
        out.append((start, route.length_m))
    return out


def admin_area_contains(circles_geojson: Path | str) -> Any:
    """``contains(lat, lon)`` for the union of the admin circles (gis/layers/circles.geojson). Roads outside every
    circle get no owner from the admin-unit rule, so potholes there go to the jurisdiction queue."""
    import shapely
    from shapely.geometry import Point, shape

    with open(circles_geojson, encoding="utf-8") as f:
        fc = json.load(f)
    union = shapely.union_all([shape(ft["geometry"]) for ft in fc.get("features", []) if ft.get("geometry")])
    shapely.prepare(union)
    return lambda lat, lon: bool(union.contains(Point(lon, lat)))


def find_unowned_stretches(routes: Iterable[dict[str, Any]], contains: Any, *, min_len_m: float = 1500.0,
                           bbox: Optional[tuple[float, float, float, float]] = None, limit: int = 5) -> \
        list[tuple[Route, float, float]]:
    """Busiest routes with a stretch of >= ``min_len_m`` outside the admin area (and inside ``bbox``)."""
    cands = sorted(routes, key=lambda r: -(float(r.get("trips_per_day") or 0.0)))
    out: list[tuple[Route, float, float]] = []
    for rec in cands:
        shape = rec.get("shape") or []
        if len(shape) < 2:
            continue
        if bbox is not None and not any(bbox[0] <= p[0] <= bbox[2] and bbox[1] <= p[1] <= bbox[3] for p in shape):
            continue
        r = route_from_record(rec)
        for s, e in outside_stretches(r, contains, min_len_m=min_len_m):
            mid = r.latlon_at((s + e) / 2)
            if bbox is not None and not (bbox[0] <= mid[1] <= bbox[2] and bbox[1] <= mid[0] <= bbox[3]):
                continue
            out.append((r, s, e))
            break
        if len(out) >= limit:
            break
    return out
