"""PostGIS adapter for map-matching (production path, DECISIONS D1). Same signatures as ``geo.mapmatch``.

The first argument is a DB-API connection (psycopg 3 or psycopg2), not a ``RoadIndex``. PostGIS returns
the candidate segments. Scoring and Viterbi smoothing are the same code as the in-memory path
(``geo.mapmatch.score_candidates`` / ``viterbi``), so both give identical results for the same candidates.
This module imports without psycopg. Only :func:`connect` imports it.

Schema assumed on PostgreSQL (the SQLite demo keeps ``geom_wkt`` text only):

    ALTER TABLE road_segment ADD COLUMN geom geometry(LineString, 4326)
        GENERATED ALWAYS AS (ST_GeomFromText(geom_wkt, 4326)) STORED;
    CREATE INDEX ix_road_segment_geom ON road_segment USING gist (geom);
    CREATE INDEX ix_road_segment_geog ON road_segment USING gist ((geom::geography));

Candidate query (``CANDIDATES_SQL``; KNN on the geometry index, exact distance and radius on geography):

    WITH q AS (SELECT ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326) AS g),
    c AS (
      SELECT s.*, ST_LineLocatePoint(s.geom, q.g) AS frac, q.g AS qg
      FROM road_segment s, q
      WHERE ST_DWithin(s.geom::geography, q.g::geography, %(radius)s)
      ORDER BY s.geom <-> q.g
      LIMIT 16
    )
    SELECT id, osm_way_id, oneway,
           ST_Distance(geom::geography, qg::geography) AS dist_m,
           frac,
           ST_Length(geom::geography) AS length_m,
           degrees(ST_Azimuth(ST_LineInterpolatePoint(geom, GREATEST(frac - 0.001, 0))::geography,
                              ST_LineInterpolatePoint(geom, LEAST(frac + 0.001, 1))::geography)) AS bearing_deg,
           ST_X(ST_StartPoint(geom)), ST_Y(ST_StartPoint(geom)),
           ST_X(ST_EndPoint(geom)), ST_Y(ST_EndPoint(geom)),
           ST_X(ST_ClosestPoint(geom, qg)), ST_Y(ST_ClosestPoint(geom, qg))
    FROM c;

Trip fixes of one trip (for :func:`match_trip_id`):

    SELECT trip_id, ts, lat, lon, heading_deg, speed_mps, accuracy_m, cam_active
    FROM trip_fix WHERE trip_id = %(trip_id)s ORDER BY ts;

Valhalla Meili or OSRM ``/match`` can replace the Viterbi step in production. They return the same
``MatchResult`` once their edge ids are mapped back to ``road_segment.id``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Optional

from geo.index import DEFAULT_RADIUS_M, Candidate
from geo.locate import to_local
from geo.mapmatch import score_candidates, to_result, viterbi_in_time_order
from geo.types import FixPoint, MatchResult

# Same text as in the module docstring (geo/tests/test_mapmatch_pg.py keeps them in sync).
CANDIDATES_SQL = """
WITH q AS (SELECT ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326) AS g),
c AS (
  SELECT s.*, ST_LineLocatePoint(s.geom, q.g) AS frac, q.g AS qg
  FROM road_segment s, q
  WHERE ST_DWithin(s.geom::geography, q.g::geography, %(radius)s)
  ORDER BY s.geom <-> q.g
  LIMIT 16
)
SELECT id, osm_way_id, oneway,
       ST_Distance(geom::geography, qg::geography) AS dist_m,
       frac,
       ST_Length(geom::geography) AS length_m,
       degrees(ST_Azimuth(ST_LineInterpolatePoint(geom, GREATEST(frac - 0.001, 0))::geography,
                          ST_LineInterpolatePoint(geom, LEAST(frac + 0.001, 1))::geography)) AS bearing_deg,
       ST_X(ST_StartPoint(geom)), ST_Y(ST_StartPoint(geom)),
       ST_X(ST_EndPoint(geom)), ST_Y(ST_EndPoint(geom)),
       ST_X(ST_ClosestPoint(geom, qg)), ST_Y(ST_ClosestPoint(geom, qg))
FROM c;
"""

TRIP_FIX_SQL = ("SELECT trip_id, ts, lat, lon, heading_deg, speed_mps, accuracy_m, cam_active "
                "FROM trip_fix WHERE trip_id = %(trip_id)s ORDER BY ts;")


def connect(dsn: str) -> Any:  # pragma: no cover - needs a PostgreSQL server
    """Open a psycopg 3 connection (psycopg is imported here only)."""
    import psycopg  # type: ignore[import-not-found]

    return psycopg.connect(dsn)


def _rows(conn: Any, sql: str, params: dict[str, Any]) -> list[Sequence[Any]]:
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        return list(cur.fetchall())
    finally:
        cur.close()


def row_to_candidate(r: Sequence[Any], lat: float, lon: float) -> Candidate:
    """One row of ``CANDIDATES_SQL`` (in SELECT column order) -> ``Candidate`` for the query point lat/lon."""
    seg_id, way_id, oneway, dist_m, frac, length_m, bearing, sx, sy, ex, ey, cx, cy = r[:13]
    length = float(length_m or 0.0)
    d = float(dist_m)
    brg = float(bearing or 0.0) % 360.0
    # side of the query point relative to the digitised direction: + = right (2D cross product sign)
    ce, cn = to_local(float(cy), float(cx), lat, lon)
    h = math.radians(brg)
    cross = math.sin(h) * float(cn) - math.cos(h) * float(ce)
    return Candidate(seg_id=int(seg_id), way_id=int(way_id), oneway=bool(oneway), dist_m=d, bearing_deg=brg,
                     chainage_m=float(frac or 0.0) * length, length_m=length,
                     offset_m=d if cross < 0 else -d, start=(float(sx), float(sy)), end=(float(ex), float(ey)))


def candidates(conn: Any, lat: float, lon: float, radius_m: float = DEFAULT_RADIUS_M) -> list[Candidate]:
    """Candidate segments within ``radius_m`` from PostGIS, nearest first."""
    rows = _rows(conn, CANDIDATES_SQL, {"lat": float(lat), "lon": float(lon), "radius": float(radius_m)})
    out = [row_to_candidate(r, lat, lon) for r in rows]
    out.sort(key=lambda c: (c.dist_m, c.seg_id))
    return out


def match_point(conn: Any, lat: float, lon: float, heading_deg: Optional[float], sigma_m: float = 8.0,
                radius_m: float = DEFAULT_RADIUS_M) -> Optional[MatchResult]:
    """Same as ``geo.mapmatch.match_point`` with PostGIS candidates."""
    scored = score_candidates(candidates(conn, lat, lon, radius_m), heading_deg, sigma_m)
    return to_result(scored[0]) if scored else None


def match_trip(conn: Any, fixes: list[FixPoint], radius_m: float = DEFAULT_RADIUS_M) -> list[Optional[MatchResult]]:
    """Same as ``geo.mapmatch.match_trip`` with PostGIS candidates (one query per fix)."""
    return viterbi_in_time_order(fixes, lambda f: candidates(conn, f.lat, f.lon, radius_m))


def match_trip_id(conn: Any, trip_id: Any, radius_m: float = DEFAULT_RADIUS_M
                  ) -> tuple[list[FixPoint], list[Optional[MatchResult]]]:
    """Load a trip's ``trip_fix`` rows and match them (PLAN WP5 ``match_trip(trip_id)``)."""
    fixes = [FixPoint(trip_id=r[0], ts=r[1], lat=float(r[2]), lon=float(r[3]),
                      heading_deg=None if r[4] is None else float(r[4]),
                      speed_mps=None if r[5] is None else float(r[5]),
                      accuracy_m=None if r[6] is None else float(r[6]), cam_active=bool(r[7]))
             for r in _rows(conn, TRIP_FIX_SQL, {"trip_id": trip_id})]
    return fixes, match_trip(conn, fixes, radius_m)
