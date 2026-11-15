"""The PostGIS adapter imports without psycopg and gives the same matches as the in-memory index when fed
the same candidates (a fake DB-API connection answers the SQL from the synthetic RoadIndex)."""

import re
import subprocess
import sys

import pytest
import shapely

from geo import mapmatch, mapmatch_pg
from geo.tests.synth import drive, dual_carriageway, ll, two_lane_network


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params):
        self.conn.calls.append((sql, params))
        assert sql == mapmatch_pg.CANDIDATES_SQL
        self.rows = []
        for c in self.conn.index.candidates(params["lat"], params["lon"], params["radius"]):
            line = shapely.LineString(c.seg.coords)
            pt = shapely.Point(params["lon"], params["lat"])
            cp = line.interpolate(line.project(pt))
            self.rows.append((c.seg_id, c.way_id, c.oneway, c.dist_m, c.chainage_m / c.length_m, c.length_m,
                              c.bearing_deg, *c.start, *c.end, cp.x, cp.y))

    def fetchall(self):
        return self.rows

    def close(self):
        pass


class FakeConn:
    def __init__(self, index):
        self.index = index
        self.calls = []

    def cursor(self):
        return FakeCursor(self)


def test_imports_without_psycopg():
    code = ("import sys; sys.modules['psycopg'] = None; sys.modules['psycopg2'] = None; "
            "import geo.mapmatch_pg as m; print(callable(m.match_point) and callable(m.match_trip))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True"


def test_sql_constant_is_documented_in_module_docstring():
    norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
    assert norm(mapmatch_pg.CANDIDATES_SQL) in norm(mapmatch_pg.__doc__)
    assert "%(lat)s" in mapmatch_pg.CANDIDATES_SQL and "ST_DWithin" in mapmatch_pg.CANDIDATES_SQL


@pytest.mark.parametrize("heading", [90.0, 270.0, None])
def test_match_point_parity_with_index(heading):
    idx = dual_carriageway()
    conn = FakeConn(idx)
    lat, lon = ll(20.0, -2.0)
    a = mapmatch.match_point(idx, lat, lon, heading)
    b = mapmatch_pg.match_point(conn, lat, lon, heading)
    assert (a.road_segment_id, a.direction) == (b.road_segment_id, b.direction)
    assert a.chainage_m == pytest.approx(b.chainage_m) and a.score == pytest.approx(b.score)
    assert conn.calls and conn.calls[0][1]["radius"] == 25.0


def test_candidate_side_sign_matches_index():
    idx = two_lane_network()
    conn = FakeConn(idx)
    for y in (5.0, -5.0):
        lat, lon = ll(100.0, y)
        ci = idx.candidates(lat, lon)[0]
        cp = mapmatch_pg.candidates(conn, lat, lon)[0]
        assert ci.offset_m == pytest.approx(cp.offset_m, abs=0.05)


def test_match_trip_parity_with_index():
    idx = two_lane_network()
    xs = [-60.0, -50.0, -40.0, -30.0, -20.0, -10.0, -4.0, -1.0, -4.0, 6.0, 16.0]
    ys = [1.5] * len(xs)
    ys[7] = 5.0
    fixes = drive("T", xs, ys, 90.0, 10.0)
    for k in (6, 7, 8):
        fixes[k].speed_mps = 0.0
    a = mapmatch.match_trip(idx, fixes)
    b = mapmatch_pg.match_trip(FakeConn(idx), fixes)
    assert [(r.road_segment_id, r.direction) for r in a] == [(r.road_segment_id, r.direction) for r in b]
