"""DB rows <-> pure ``geo`` dataclasses, plus process-wide caches for the spatial indexes (WP6, DECISIONS D1).

``geo/*`` is a pure library (no DB, no server imports). This module is the only place that turns ORM rows into
its inputs:

* ``segment_rec(row)``       RoadSegment row -> ``geo.types.SegmentRec`` (coords parsed from ``geom_wkt``)
* ``obs_point(...)``         one verified detection of an observation -> ``geo.types.ObsPoint``
* ``pothole_point(row)``     Pothole row -> ``geo.types.PotholePoint``
* ``fix_point(row, trip)``   TripFix row -> ``geo.types.FixPoint``
* ``match_result(fix_row)``  a TripFix that already carries way_id/direction -> ``geo.types.MatchResult``
* ``road_index_for(db, pts)``  a cached ``geo.index.RoadIndex`` that contains every segment within ``margin_m`` of
  the given points. Only 0.01 deg tiles (~1.1 km) touched by the points are loaded from ``road_segment``, so the
  454k-segment demo DB is never loaded whole. The cache is keyed by database URL and expires after ``TTL_S``.
* ``jurisdiction_inputs(db)`` admin units / contracts as ``(id, agency_id, shapely geometry)`` lists (cached).

``geo`` modules are imported lazily inside the functions: they are written in parallel (WP5) and tests may
substitute fakes in ``sys.modules``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.app.db import models as m

log = logging.getLogger("gunthameter.workers.adapters")

TILE_DEG = 0.01  # ~1.1 km at Hyderabad's latitude
TTL_S = 900.0  # road data rarely changes; re-seeding / admin uploads are picked up after this
MAX_CACHED_SEGMENTS = 250_000  # beyond this the cache for one database is reset (RAM guard, D12)


def geo_types() -> Any:
    import importlib

    return importlib.import_module("geo.types")


# =========================================================================== row -> dataclass
def parse_linestring(wkt: Optional[str]) -> list[tuple[float, float]]:
    """``LINESTRING (lon lat, lon lat, ...)`` -> [(lon, lat), ...] (fast path without shapely)."""
    if not wkt:
        return []
    s = wkt.strip()
    i, j = s.find("("), s.rfind(")")
    if i < 0 or j <= i or not s[:i].strip().upper().startswith("LINESTRING"):
        try:
            from shapely import wkt as swkt

            g = swkt.loads(s)
            if g.geom_type == "MultiLineString":
                return [tuple(c[:2]) for part in g.geoms for c in part.coords]  # type: ignore[misc]
            return [tuple(c[:2]) for c in g.coords]  # type: ignore[misc]
        except Exception:
            return []
    out: list[tuple[float, float]] = []
    for part in s[i + 1:j].split(","):
        xy = part.split()
        if len(xy) >= 2:
            out.append((float(xy[0]), float(xy[1])))
    return out


def segment_rec(row: m.RoadSegment) -> Any:
    T = geo_types()
    return T.SegmentRec(id=int(row.id), way_id=int(row.osm_way_id), coords=parse_linestring(row.geom_wkt),
                        oneway=bool(row.oneway), lanes=row.lanes, highway=str(row.highway or "unclassified"),
                        owner_agency_id=row.owner_agency_id, contract_id=row.contract_id,
                        admin_unit_id=row.admin_unit_id)


def lane_band(lane_offset_m: Optional[float]) -> Optional[int]:
    """``geo.cluster.lane_band`` (PLAN 5.3: round(offset / 1.75 m))."""
    if lane_offset_m is None:
        return None
    import importlib

    return importlib.import_module("geo.cluster").lane_band(lane_offset_m)


def obs_point(key: int, obs: m.Observation, lat: float, lon: float, sigma_m: float,
              lane_offset_m: Optional[float]) -> Any:
    """``key`` is the detection id: one observation (frame) may show several potholes at different ground points."""
    T = geo_types()
    return T.ObsPoint(obs_id=int(key), trip_id=obs.trip_id, lat=float(lat), lon=float(lon), sigma_m=float(sigma_m),
                      way_id=obs.way_id, direction=obs.direction, lane_offset_m=lane_offset_m, ts=obs.captured_at)


def pothole_point(p: m.Pothole) -> Any:
    T = geo_types()
    band = p.lane_band if p.lane_band is not None else lane_band(p.lane_offset_m)
    return T.PotholePoint(pothole_id=int(p.id), lat=float(p.lat), lon=float(p.lon), way_id=p.way_id,
                          direction=p.direction, lane_band=band, status=str(p.status))


def fix_point(f: m.TripFix, trip_id: Any = None) -> Any:
    T = geo_types()
    return T.FixPoint(trip_id=trip_id if trip_id is not None else f.trip_id, ts=f.ts, lat=float(f.lat),
                      lon=float(f.lon), heading_deg=f.heading_deg, speed_mps=f.speed_mps, accuracy_m=f.accuracy_m,
                      cam_active=bool(f.cam_active))


def match_result(f: m.TripFix, road_segment_id: Optional[int] = None) -> Any:
    """MatchResult rebuilt from a TripFix whose way_id/direction were stored by the trip-close hook (or None)."""
    if f.way_id is None:
        return None
    T = geo_types()
    return T.MatchResult(road_segment_id=road_segment_id if road_segment_id is not None else -1,
                         way_id=int(f.way_id), direction=int(f.direction or 0), chainage_m=0.0, dist_m=0.0,
                         bearing_deg=float(f.heading_deg or 0.0), score=0.0)


# =========================================================================== road index cache
def _db_key(db: Session) -> str:
    try:
        return str(db.get_bind().url)
    except Exception:  # pragma: no cover - unbound session
        return f"session:{id(db)}"


def _tiles_for(points: Iterable[tuple[float, float]], margin_m: float) -> set[tuple[int, int]]:
    tiles: set[tuple[int, int]] = set()
    for lat, lon in points:
        if lat is None or lon is None or not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        dlat = margin_m / 111_320.0
        dlon = margin_m / (111_320.0 * max(0.1, math.cos(math.radians(lat))))
        for ty in range(math.floor((lat - dlat) / TILE_DEG), math.floor((lat + dlat) / TILE_DEG) + 1):
            for tx in range(math.floor((lon - dlon) / TILE_DEG), math.floor((lon + dlon) / TILE_DEG) + 1):
                tiles.add((tx, ty))
    return tiles


@dataclass
class _IndexEntry:
    created: float = field(default_factory=time.monotonic)
    tiles: set[tuple[int, int]] = field(default_factory=set)
    segments: dict[int, Any] = field(default_factory=dict)  # road_segment.id -> SegmentRec
    index: Any = None
    dirty: bool = True


_LOCK = threading.RLock()
_INDEX: dict[str, _IndexEntry] = {}
_JURIS: dict[str, tuple[float, Any]] = {}


def clear_caches() -> None:
    with _LOCK:
        _INDEX.clear()
        _JURIS.clear()


def _load_tile_segments(db: Session, tiles: Iterable[tuple[int, int]]) -> list[m.RoadSegment]:
    S = m.RoadSegment
    rows: dict[int, m.RoadSegment] = {}
    for tx, ty in tiles:
        x1, y1 = tx * TILE_DEG, ty * TILE_DEG
        x2, y2 = x1 + TILE_DEG, y1 + TILE_DEG
        for r in db.scalars(select(S).where(S.max_lon >= x1, S.min_lon <= x2, S.max_lat >= y1, S.min_lat <= y2)):
            rows[r.id] = r
    return list(rows.values())


def road_index_for(db: Session, points: Sequence[tuple[float, float]], margin_m: float = 300.0) -> Any:
    """Cached ``geo.index.RoadIndex`` covering ``points`` (lat, lon) + margin. Returns None when no segment is near."""
    import importlib

    RoadIndex = importlib.import_module("geo.index").RoadIndex
    key = _db_key(db)
    need = _tiles_for(points, margin_m)
    with _LOCK:
        ent = _INDEX.get(key)
        if ent is None or time.monotonic() - ent.created > TTL_S or len(ent.segments) > MAX_CACHED_SEGMENTS:
            ent = _INDEX[key] = _IndexEntry()
        missing = need - ent.tiles
        if missing:
            for r in _load_tile_segments(db, missing):
                if r.id not in ent.segments:
                    rec = segment_rec(r)
                    if len(rec.coords) >= 2:
                        ent.segments[r.id] = rec
                        ent.dirty = True
            ent.tiles |= missing
        if not ent.segments:
            return None
        if ent.dirty or ent.index is None:
            ent.index = RoadIndex(list(ent.segments.values()))
            ent.dirty = False
        return ent.index


def cached_segment(db: Session, segment_id: Optional[int]) -> Any:
    """SegmentRec for ``segment_id`` from the cache (or the DB)."""
    if segment_id is None:
        return None
    with _LOCK:
        ent = _INDEX.get(_db_key(db))
        if ent is not None and segment_id in ent.segments:
            return ent.segments[segment_id]
    row = db.get(m.RoadSegment, segment_id)
    return segment_rec(row) if row is not None else None


# =========================================================================== jurisdiction inputs
def jurisdiction_inputs(db: Session) -> tuple[list[tuple[int, Optional[int], Any]], list[tuple[int, Optional[int], Any]]]:
    """(admin_units, contracts) as lists of ``(id, agency_id, shapely geometry)`` (current units only; cached)."""
    from shapely import wkt as swkt

    key = _db_key(db)
    with _LOCK:
        hit = _JURIS.get(key)
        if hit is not None and time.monotonic() - hit[0] <= TTL_S:
            return hit[1]

    def _rows(model: Any, extra: Any = None) -> list[tuple[int, Optional[int], Any]]:
        stmt = select(model).where(model.geom_wkt.is_not(None))
        if extra is not None:
            stmt = stmt.where(extra)
        out = []
        for r in db.scalars(stmt):
            try:
                out.append((int(r.id), r.agency_id, swkt.loads(r.geom_wkt)))
            except Exception:  # malformed upload: ignore (it is visible in /admin)
                continue
        return out

    units = _rows(m.AdminUnit, m.AdminUnit.valid_to.is_(None))
    # most specific first so a resolver that takes the first hit gets a ward before its circle/zone
    rank = {"ward": 0, "circle": 1, "division": 2, "zone": 3}
    kinds = {int(i): k for i, k in db.execute(select(m.AdminUnit.id, m.AdminUnit.kind))}
    units.sort(key=lambda u: rank.get(kinds.get(u[0], ""), 9))
    res = (units, _rows(m.Contract))
    with _LOCK:
        _JURIS[key] = (time.monotonic(), res)
    return res


__all__ = [
    "clear_caches", "fix_point", "jurisdiction_inputs", "lane_band", "match_result", "obs_point", "parse_linestring",
    "pothole_point", "road_index_for", "segment_rec", "cached_segment",
]
