import json
from pathlib import Path

import numpy as np
import pytest
import shapely
from pyproj import Transformer

from geo.index import RoadIndex, coords_from_wkt, iter_geojson_features, iter_segments_geojson, segment_from_wkt
from geo.tests.synth import ll, two_lane_network

FIXTURE = Path(__file__).resolve().parents[2] / "tests/fixtures/geo_small/road_segments.geojson"


def test_nearest_returns_distance_and_digitised_bearing():
    idx = two_lane_network()
    lat, lon = ll(100.0, 5.0)
    hits = idx.nearest(lat, lon, radius_m=25)
    assert [h[0].id for h in hits] == [2]
    seg, d, brg = hits[0]
    assert d == pytest.approx(5.0, abs=0.02)
    assert brg == pytest.approx(90.0, abs=0.1)
    c = idx.candidates(lat, lon)[0]
    assert c.chainage_m == pytest.approx(100.0, abs=0.05)
    assert c.length_m == pytest.approx(400.0, abs=0.2)
    assert c.offset_m == pytest.approx(-5.0, abs=0.02)  # north of an eastbound line = left


def test_nearest_radius_and_ordering_near_junction():
    idx = two_lane_network()
    lat, lon = ll(3.0, 10.0)
    ids = [s.id for s, _, _ in idx.nearest(lat, lon, radius_m=25)]
    assert ids[0] == 4  # 3 m from the north-south road
    assert set(ids) == {1, 2, 3, 4}
    assert idx.nearest(*ll(0.0, 150.0), radius_m=5)[0][0].id == 4
    assert idx.nearest(*ll(150.0, 150.0), radius_m=25) == []


def test_segment_from_wkt_and_empty_index():
    s = segment_from_wkt(7, 70, "LINESTRING (78.45 17.43, 78.451 17.43)", oneway=True)
    assert s.coords == [(78.45, 17.43), (78.451, 17.43)] and s.oneway
    assert coords_from_wkt("LINESTRING EMPTY") == []
    assert RoadIndex([]).nearest(17.43, 78.45) == []


def test_streaming_geojson_reader_single_line_and_multiline(tmp_path):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"osm_way_id": i, "owner_agency": "GHMC"},
         "geometry": {"type": "LineString", "coordinates": [[78.45, 17.43 + i * 1e-4], [78.46, 17.43 + i * 1e-4]]}}
        for i in range(50)]}
    one = tmp_path / "one.geojson"
    one.write_text(json.dumps(fc), encoding="utf-8")
    multi = tmp_path / "multi.geojson"
    multi.write_text('{"type":"FeatureCollection","features":[\n'
                     + ",\n".join(json.dumps(f) for f in fc["features"]) + "\n]}\n", encoding="utf-8")
    for p in (one, multi):
        feats = list(iter_geojson_features(p, chunk_chars=97))  # tiny chunks exercise the refill path
        assert [f["properties"]["osm_way_id"] for f in feats] == list(range(50))
    segs = list(iter_segments_geojson(one, agency_ids={"GHMC": 3}, first_id=10))
    assert segs[0].id == 10 and segs[-1].id == 59 and segs[5].owner_agency_id == 3


def test_fixture_network_metric_distances_match_utm():
    """Real OSM fixture (1.7k segments): nearest == brute force in UTM 44N, distances within 5 cm."""
    segs = list(iter_segments_geojson(FIXTURE))
    assert len(segs) > 1000
    idx = RoadIndex(segs)
    to_m = Transformer.from_crs(4326, 32644, always_xy=True)
    lines_m = [shapely.LineString(np.column_stack(to_m.transform(*np.array(s.coords).T))) for s in segs]
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(30):
        s = segs[int(rng.integers(len(segs)))]
        lon, lat = s.coords[0]
        lat += float(rng.normal(0, 8e-5))
        lon += float(rng.normal(0, 8e-5))
        hits = idx.nearest(lat, lon, radius_m=40)
        px, py = to_m.transform(lon, lat)
        brute = sorted((shapely.Point(px, py).distance(g), k) for k, g in enumerate(lines_m))
        brute_in = [(d, k) for d, k in brute if d <= 40]
        assert len(hits) == len(brute_in)
        if hits:
            # UTM has ~0.04 % scale error at this longitude; distances < 40 m -> < 2 cm
            assert hits[0][1] == pytest.approx(brute_in[0][0], abs=0.05)
            assert hits[0][0].id == segs[brute_in[0][1]].id or abs(brute_in[0][0] - brute_in[1][0]) < 0.05
            checked += 1
    assert checked >= 20
