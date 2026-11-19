"""WP10 geo build: OSM splitting, tag normalisation, ownership, GTFS shapes/trips, layers, fixture contract."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, Point, box, shape

from geo import build_geo as B
from geo import gtfs as G
from geo import osm as O

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests/fixtures/geo_small"
FULL = ROOT / "datasets/extracted/geo"
L = B.layers_module()

SEG_TYPES = {
    "osm_way_id": (int,), "seg_idx": (int,), "name": (str, type(None)), "name_te": (str, type(None)),
    "highway": (str,), "ref": (str, type(None)), "lanes": (int, type(None)), "oneway": (bool,),
    "lit": (str, type(None)), "surface": (str,), "length_m": (float, int), "bearing_deg": (float, int),
    "junction_start": (bool,), "junction_end": (bool,), "gtfs_trips_per_day": (float, int),
    "owner_agency": (str, type(None)), "admin_unit_code": (str, type(None)), "contract_code": (str, type(None)),
}
AGENCIES = {"GHMC", "CMC", "MMC", "RNB", "HMDA", "NHAI", "SCB"}
HIGHWAYS = set(O.DRIVABLE_HIGHWAYS)


def check_segment_props(p: dict) -> None:
    assert set(SEG_TYPES) <= set(p), set(SEG_TYPES) - set(p)
    for k, t in SEG_TYPES.items():
        assert isinstance(p[k], t), (k, p[k])
    assert not isinstance(p["osm_way_id"], bool)
    assert p["highway"] in HIGHWAYS
    assert p["surface"] in {"BT", "CC", "unknown"}
    assert p["lit"] in {"yes", "no", None}
    assert p["length_m"] > 0 and 0 <= p["bearing_deg"] < 360
    assert p["gtfs_trips_per_day"] >= 0
    assert p["owner_agency"] is None or p["owner_agency"] in AGENCIES


# ----------------------------------------------------------------------------- splitting
def test_split_ways_at_shared_nodes_and_junction_degree():
    # A: 1-2-3-4-5 ; B crosses A at 3 ; C continues A from 5 (degree 2, not a junction)
    ways = [[1, 2, 3, 4, 5], [10, 3, 11], [5, 20]]
    sp, flat, _ = O.split_ways(ways)
    segs = [tuple(flat[s:e + 1]) for s, e in zip(sp.start, sp.end, strict=True)]
    assert segs == [(1, 2, 3), (3, 4, 5), (10, 3), (3, 11), (5, 20)]
    assert sp.seg_idx.tolist() == [0, 1, 0, 1, 0]
    assert sp.way_row.tolist() == [0, 0, 1, 1, 2]
    j = dict(zip(segs, zip(sp.junction_start, sp.junction_end, strict=True), strict=True))
    assert j[(1, 2, 3)] == (False, True)        # node 3 has degree 4
    assert j[(3, 4, 5)] == (True, False)        # node 5: end of A + start of C = degree 2
    assert j[(5, 20)] == (False, False)


def test_split_way_visiting_node_twice():
    sp, flat, _ = O.split_ways([[1, 2, 3, 4, 2, 5]])
    segs = [tuple(flat[s:e + 1]) for s, e in zip(sp.start, sp.end, strict=True)]
    assert segs == [(1, 2), (2, 3, 4, 2), (2, 5)]


def _ways_df(rows):
    return pd.DataFrame([{
        "id": r["id"], "nodes": r["nodes"], "geometry": LineString(r["xy"]),
        "highway": r.get("highway", "residential"), "name": r.get("name"), "name:te": r.get("name_te"),
        "ref": r.get("ref"), "lanes": r.get("lanes"), "oneway": r.get("oneway"), "lit": r.get("lit"),
        "surface": r.get("surface"), "junction": None, "access": r.get("access"), "area": None,
        "service": r.get("service"), "operator": None,
    } for r in rows])


def test_build_segments_lengths_bearings_tags_and_oneway_reverse():
    lat = 17.43
    dx = 0.001  # ~106 m of longitude at 17.43 N
    ways = _ways_df([
        {"id": 1, "nodes": [1, 2, 3], "xy": [(78.45, lat), (78.45 + dx, lat), (78.45 + 2 * dx, lat)],
         "highway": "primary", "name": "Test Road", "lanes": "4", "lit": "yes", "surface": "concrete"},
        {"id": 2, "nodes": [4, 2, 5], "xy": [(78.45 + dx, lat - dx), (78.45 + dx, lat), (78.45 + dx, lat + dx)],
         "highway": "residential", "oneway": "-1"},
        {"id": 3, "nodes": [6, 7], "xy": [(78.40, lat), (78.40, lat + dx)], "highway": "service",
         "service": "parking_aisle"},
        {"id": 4, "nodes": [8, 9], "xy": [(78.41, lat), (78.41, lat + dx)], "highway": "residential",
         "access": "private"},
    ])
    w = O.filter_ways(ways)
    assert sorted(w["id"]) == [1, 2]
    seg = O.build_segments(w, bbox=None)
    assert len(seg) == 4
    a0 = seg[(seg.osm_way_id == 1) & (seg.seg_idx == 0)].iloc[0]
    assert a0.length_m == pytest.approx(106.2, abs=1.0)
    assert a0.bearing_deg == pytest.approx(90.0, abs=0.5)
    assert (a0["surface"], a0["lit"], a0["lanes"], a0["oneway"], a0["name"]) == ("CC", "yes", 4, False, "Test Road")
    assert a0.junction_end and not a0.junction_start
    # oneway=-1: reversed so digitisation == travel direction (north -> south)
    b = seg[seg.osm_way_id == 2].sort_values("seg_idx")
    assert b.oneway.all()
    assert b.iloc[0].bearing_deg == pytest.approx(180.0, abs=0.5)
    assert b.iloc[0].start_node == 5 and b.iloc[-1].end_node == 4
    assert b.iloc[0].surface == "unknown"  # missing tag on residential


@pytest.mark.parametrize("surface,highway,exp", [
    ("asphalt", "residential", "BT"), ("concrete:plates", "primary", "CC"), ("cement_concrete", "tertiary", "CC"),
    (None, "primary", "BT"), (None, "residential", "unknown"), ("unpaved", "primary", "unknown"),
    (float("nan"), "secondary_link", "BT"), ("paving_stones", "service", "unknown"),
])
def test_norm_surface(surface, highway, exp):
    assert O.norm_surface(surface, highway) == exp


def test_norm_misc_tags():
    assert O.norm_lanes("2;3") == 2 and O.norm_lanes(None) is None and O.norm_lanes("x") is None
    assert O.norm_lit("24/7") == "yes" and O.norm_lit("no") == "no" and O.norm_lit(float("nan")) is None
    assert O.norm_oneway("yes", "primary", None) == (True, False)
    assert O.norm_oneway(None, "motorway", None) == (True, False)
    assert O.norm_oneway(None, "primary", "roundabout") == (True, False)
    assert O.norm_oneway("no", "motorway", None) == (False, False)
    assert O.norm_oneway("-1", "residential", None) == (True, True)
    assert O.metro_key("Miyapur Metro Station") == O.metro_key("miyapur") == "miyapur"
    assert O.metro_key("Ameerpet (Red Line)") != O.metro_key("Ameerpet (Blue Line)")


# ----------------------------------------------------------------------------- ownership
RULES = [
    {"if_osm_ref_prefix": "NH", "owner": "NHAI"},
    {"if_osm_ref_prefix": "SH", "owner": "RNB"},
    {"if_highway": "motorway", "if_name_contains": "Outer Ring Road", "owner": "HMDA"},
    {"default_by_admin_unit": True},
]


@pytest.mark.parametrize("hw,ref,name,exp", [
    ("trunk", "NH163;IRR;SH4", None, "NHAI"),
    ("primary", "SH1", "Rajiv Rahadari", "RNB"),
    ("primary", "Old NH7", None, None),
    ("motorway", "ORR", "Outer Ring Road", "HMDA"),
    ("motorway_link", None, "Outer Ring Road", "HMDA"),
    ("primary", None, "Outer Ring Road", None),
    ("residential", float("nan"), float("nan"), None),
])
def test_rule_owner(hw, ref, name, exp):
    assert B.rule_owner(hw, ref, name, RULES) == exp


def test_supplementary_orr_rule():
    rules = RULES + B.SUPPLEMENTARY_OWNER_RULES
    assert B.rule_owner("motorway_link", "ORR", None, rules) == "HMDA"
    assert B.rule_owner("secondary", None, "ORR Service Road", rules) == "HMDA"


def _fc(features):
    return {"type": "FeatureCollection", "features": features}


def test_assign_admin_and_owner_precedence():
    c1 = box(78.0, 17.0, 78.5, 17.5)
    c2 = box(78.5, 17.0, 79.0, 17.5)
    circles = _fc([L.feature(c1, {"code": "GHMC-C01", "agency": "GHMC", "parent_code": "GHMC-Z"}),
                   L.feature(c2, {"code": "SCB-C01", "agency": "SCB", "parent_code": "SCB"})])
    overrides = _fc([L.feature(box(78.1, 17.1, 78.2, 17.2),
                               {"owner_agency": "HMDA", "reason": "x", "name_contains": "Narasimha Rao"})])
    seg = pd.DataFrame({
        "highway": ["primary", "primary", "trunk", "residential", "residential"],
        "ref": [None, None, "NH44", None, None],
        "name": ["PV Narasimha Rao Expressway", "Below Road", None, None, None],
        "geometry": [LineString([(78.12, 17.15), (78.18, 17.15)]), LineString([(78.12, 17.16), (78.18, 17.16)]),
                     LineString([(78.6, 17.1), (78.7, 17.1)]), LineString([(78.6, 17.2), (78.7, 17.2)]),
                     LineString([(79.6, 17.2), (79.7, 17.2)])],
    })
    out = B.assign_admin_and_owner(seg, circles, _fc([]), overrides, RULES)

    def col(c):  # pandas 3 stores missing str values as NaN; the GeoJSON writer emits null
        return [B._clean(v) for v in out[c].tolist()]

    assert col("owner_agency") == ["HMDA", "GHMC", "NHAI", "SCB", None]
    assert col("owner_basis") == ["override", "admin_unit", "rule", "admin_unit", None]
    assert col("admin_unit_code") == ["GHMC-C01", "GHMC-C01", "SCB-C01", "SCB-C01", None]
    assert out.zone_code.tolist()[0] == "GHMC-Z"


# ----------------------------------------------------------------------------- GHMC units partition
def test_build_ghmc_units_is_partition():
    ghmc = box(78.40, 17.35, 78.50, 17.45)
    wards = [{"ward_no": i + 1, "name_en": f"W{i + 1}", "name_te": None, "osm_id": i,
              "geometry": box(78.40 + 0.025 * i, 17.35, 78.425 + 0.025 * i, 17.43)} for i in range(4)]
    seeds = {"ghmc": {"agency": "GHMC", "zones": [
        {"code": "Z1", "name_en": "West", "name_te": "ప", "name_hi": "प",
         "circles": [["C1", "A", "అ", "अ", 17.40, 78.41, "approx"], ["C2", "B", "బ", "ब", 17.40, 78.44, "approx"]]},
        {"code": "Z2", "name_en": "East", "name_te": "తూ", "name_hi": "पू",
         "circles": [["C3", "C", "చ", "च", 17.40, 78.47, "approx"], ["C4", "D", "డ", "ड", 17.40, 78.49, "approx"]]},
    ]}}
    zones, circles, wards_out = L.build_ghmc_units(ghmc, wards, seeds)
    assert [c["code"] for c in circles] == ["C1", "C2", "C3", "C4"]
    tot = sum(c["geometry_m"].area for c in circles)
    gm = L.to_m(ghmc).area
    assert tot == pytest.approx(gm, rel=1e-6)                       # covers GHMC (incl. un-warded strip)
    u = shapely.union_all([c["geometry_m"] for c in circles])
    assert u.area == pytest.approx(tot, rel=1e-6)                   # no overlaps
    assert {w["parent_code"] for w in wards_out} <= {"C1", "C2", "C3", "C4"}
    assert [z["code"] for z in zones] == ["Z1", "Z2"]
    feats = L.unit_features(circles, "2026-11-01")
    p = feats[0]["properties"]
    assert p["official"] is False and p["valid_from"] == "2026-11-01" and p["kind"] == "circle"
    assert p["parent_code"] == "Z1" and p["name_te"] == "అ"


def test_hotspots_and_seeds_file():
    seeds = L.load_seeds()
    feats = L.hotspot_features(seeds)
    assert len(feats) >= 15
    names = {f["properties"]["name"] for f in feats}
    assert {"Masab Tank", "Lakdikapul", "Tolichowki", "Rasoolpura"} <= names
    g = shape(feats[0]["geometry"])
    assert g.geom_type == "Polygon" and L.to_m(g).area == pytest.approx(np.pi * 150 ** 2, rel=0.1)
    n_circles = sum(len(z["circles"]) for z in seeds["ghmc"]["zones"])
    assert n_circles == 30 and len(seeds["ghmc"]["zones"]) == 6


# ----------------------------------------------------------------------------- GTFS
def _write_feed(path: Path, with_calendar_sunday_only_service: bool = True) -> Path:
    files = {
        "agency.txt": "agency_id,agency_name,agency_url,agency_timezone\nT,Test,http://x,Asia/Kolkata\n",
        "stops.txt": "stop_id,stop_name,stop_lat,stop_lon\n"
                     "S1,Alpha,17.4300,78.4500\nS2,Beta,17.4300,78.4560\nS3,Gamma,17.4360,78.4560\n"
                     "SX,Far,18.9000,79.9000\n",
        "routes.txt": "route_id,route_short_name,route_long_name,route_type\nR1,10H,,3\n",
        "trips.txt": "route_id,service_id,trip_id,direction_id,trip_short_name\n"
                     "R1,DAILY,T1,0,R1-A-TO-C\nR1,DAILY,T2,0,R1-A-TO-C\nR1,DAILY,T3,1,R1-C-TO-A\n"
                     "R1,SUN,T4,0,R1-A-TO-C\n",
        "stop_times.txt": "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                          "T1,08:00:00,08:00:00,S1,1\nT1,08:05:00,08:05:00,S2,2\nT1,08:09:00,08:09:00,S3,3\n"
                          "T2,09:00:00,09:00:00,S1,1\nT2,09:05:00,09:05:00,S2,2\nT2,09:09:00,09:09:00,S3,3\n"
                          "T3,10:00:00,10:00:00,S3,1\nT3,10:05:00,10:05:00,S2,2\nT3,10:09:00,10:09:00,S1,3\n"
                          "T4,11:00:00,11:00:00,S1,1\nT4,11:05:00,11:05:00,S2,2\nT4,11:09:00,11:09:00,S3,3\n",
        "calendar.txt": "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
                        "DAILY,1,1,1,1,1,1,1,20260101,20301231\nSUN,0,0,0,0,0,0,1,20260101,20301231\n",
    }
    with zipfile.ZipFile(path, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return path


def _grid_segments():
    """L-shaped main road S1->S2->S3 (primary) with a long detour-free side street and a stub."""
    rows = [
        # main east-west primary, two pieces, nodes 1-2-3
        (1, 2, [(78.4500, 17.4300), (78.4530, 17.4300)], "primary"),
        (2, 3, [(78.4530, 17.4300), (78.4560, 17.4300)], "primary"),
        # main north-south primary 3-4-5
        (3, 4, [(78.4560, 17.4300), (78.4560, 17.4330)], "primary"),
        (4, 5, [(78.4560, 17.4330), (78.4560, 17.4360)], "primary"),
        # 40 m perpendicular residential stub from node 2 (touches the corridor only at its end)
        (2, 6, [(78.4530, 17.4300), (78.4530, 17.43036)], "residential"),
        # parallel residential street 300 m north, never used
        (7, 8, [(78.4500, 17.4330), (78.4530, 17.4330)], "residential"),
    ]
    df = pd.DataFrame({"start_node": [r[0] for r in rows], "end_node": [r[1] for r in rows],
                       "geometry": [LineString(r[2]) for r in rows], "highway": [r[3] for r in rows],
                       "oneway": False})
    tr = Transformer.from_crs(4326, 32644, always_xy=True)
    df["length_m"] = [LineString(np.column_stack(tr.transform(*np.asarray(g.coords).T))).length for g in df.geometry]
    return df


def test_gtfs_feed_patterns_trips_per_day_and_stops(tmp_path):
    feed = G.load_feed(_write_feed(tmp_path / "f.zip"))
    tpd = G.trips_per_day_by_route(feed, "wednesday")
    assert tpd.to_dict() == {"R1": 3.0}                 # SUN service not counted on a Wednesday
    assert G.trips_per_day_by_route(feed, "sunday").to_dict() == {"R1": 4.0}
    pat = G.trip_patterns(feed)
    assert len(pat) == 2
    p0 = pat[pat.direction_id == "0"].iloc[0]
    assert p0.stops == ("S1", "S2", "S3") and p0.trips == 2 and p0.label == "R1-A-TO-C"
    stops = G.bus_stop_pois(feed, bbox=O.HYD_BBOX)
    assert sorted(stops.source_id) == ["S1", "S2", "S3"] and set(stops.kind) == {"bus_stop"}


def test_gtfs_shapes_routed_and_trip_assignment(tmp_path):
    feed = G.load_feed(_write_feed(tmp_path / "f.zip"))
    pat = G.trip_patterns(feed)
    seg = _grid_segments()
    router = G.RoadRouter(seg)
    shapes = G.build_pattern_shapes(feed, pat, router=router, bbox=O.HYD_BBOX)
    s0 = shapes[0]
    assert s0 is not None and len(s0) >= 4
    # routed along the corridor through the corner node (78.456, 17.430)
    assert np.any(np.all(np.isclose(s0, [78.4560, 17.4300], atol=1e-7), axis=1))
    straight = G.build_pattern_shapes(feed, pat, router=None)
    assert len(straight[0]) == 3
    trips = G.assign_trips_to_segments(seg.geometry.values, shapes, pat["trips"].to_numpy())
    assert trips[:4].tolist() == [3.0, 3.0, 3.0, 3.0]   # 2 trips one way + 1 back on every main piece
    assert trips[4] == 0.0                               # stub touching the corridor does not count
    assert trips[5] == 0.0
    routes = G.route_records(feed, pat, shapes)
    assert len(routes) == 1
    r = routes[0]
    assert r["route_id"] == "R1" and r["short_name"] == "10H" and r["long_name"] == "R1-A-TO-C"
    assert r["trips_per_day"] == 3.0 and len(r["shape"]) >= 3


def test_chunk_line_covers_whole_line():
    xy = np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 950.0]])
    parts = G._chunk_line(xy, 400.0)
    assert len(parts) == 5
    tot = sum(LineString(p).length for p in parts)
    assert tot == pytest.approx(1950.0, rel=1e-9)
    for a, b in zip(parts[:-1], parts[1:], strict=True):
        assert np.allclose(a[-1], b[0])


# ----------------------------------------------------------------------------- fixture contract
def _load(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def test_fixture_size_and_files():
    names = ["road_segments.geojson", "pois.geojson", "gtfs_routes.json", "ghmc_zones.geojson", "circles.geojson",
             "crmp_packages.geojson", "waterlogging_hotspots.geojson", "road_owner_overrides.geojson", "manifest.json"]
    for n in names:
        assert (FIX / n).exists(), n
    total = sum(p.stat().st_size for p in FIX.iterdir() if p.is_file())
    assert total < 2_000_000


def test_fixture_road_segments_contract():
    fc = _load(FIX / "road_segments.geojson")
    feats = fc["features"]
    assert len(feats) > 500
    bx = box(*B.FIXTURE_BBOX)
    keys = set()
    for f in feats:
        assert f["geometry"]["type"] == "LineString" and len(f["geometry"]["coordinates"]) >= 2
        assert shape(f["geometry"]).intersects(bx)
        check_segment_props(f["properties"])
        keys.add((f["properties"]["osm_way_id"], f["properties"]["seg_idx"]))
    assert len(keys) == len(feats)                                 # (way, seg_idx) unique
    props = [f["properties"] for f in feats]
    assert all(p["owner_agency"] for p in props)                   # central Hyderabad: every road owned
    assert any(p["owner_agency"] == "NHAI" for p in props)         # NH65 through Punjagutta/Ameerpet
    assert any(p["contract_code"] for p in props)
    assert any(p["gtfs_trips_per_day"] > 100 for p in props)
    assert any(p["junction_start"] for p in props) and any(not p["junction_end"] for p in props)


def test_fixture_pois_routes_layers_contract():
    pois = _load(FIX / "pois.geojson")["features"]
    kinds = {f["properties"]["kind"] for f in pois}
    assert kinds <= set(B.POI_KINDS) and {"bus_stop", "school", "hospital", "metro_station"} <= kinds
    for f in pois:
        assert f["geometry"]["type"] == "Point" and f["properties"]["source"] in {"osm", "gtfs"}
    routes = _load(FIX / "gtfs_routes.json")
    assert routes
    for r in routes:
        assert {"route_id", "short_name", "long_name", "shape", "trips_per_day"} <= set(r)
        assert len(r["shape"]) >= 2 and all(len(c) == 2 for c in r["shape"]) and r["trips_per_day"] > 0
    for name, kind in (("ghmc_zones", "zone"), ("circles", "circle")):
        fs = _load(FIX / f"{name}.geojson")["features"]
        assert fs
        for f in fs:
            p = f["properties"]
            assert {"code", "kind", "name_en", "name_te", "parent_code", "official", "valid_from"} <= set(p)
            assert p["kind"] == kind and p["official"] is False and p["name_te"]
            assert f["geometry"]["type"] in {"Polygon", "MultiPolygon"}
    for f in _load(FIX / "crmp_packages.geojson")["features"]:
        p = f["properties"]
        assert {"code", "agency", "contractor", "dlp_months"} <= set(p) and p["code"].startswith("CRMP-P")
    for f in _load(FIX / "waterlogging_hotspots.geojson")["features"]:
        assert set(f["properties"]) >= {"name", "source"}
    # every segment admin_unit_code refers to a circle present in the fixture
    codes = {f["properties"]["code"] for f in _load(FIX / "circles.geojson")["features"]}
    seg_codes = {f["properties"]["admin_unit_code"] for f in _load(FIX / "road_segments.geojson")["features"]}
    assert seg_codes - {None} <= codes


# ----------------------------------------------------------------------------- real data
@pytest.mark.data
def test_full_build_outputs_contract():
    p = FULL / "road_segments.geojson"
    if not p.exists():
        pytest.skip("run `python -m geo.build_geo` first")
    feats = _load(p)["features"]
    assert len(feats) >= 50_000
    for f in feats[:: max(1, len(feats) // 5000)]:
        check_segment_props(f["properties"])
    owners = pd.Series([f["properties"]["owner_agency"] for f in feats])
    summary = _load(FULL / "build_summary.json")
    assert int(owners.isna().sum()) == summary["segments_without_owner"]
    pois = _load(FULL / "pois.geojson")["features"]
    assert len(pois) >= 500
    assert {f["properties"]["kind"] for f in pois} == set(B.POI_KINDS)
    routes = _load(FULL / "gtfs_routes.json")
    assert len(routes) >= 500 and sum(r["trips_per_day"] for r in routes) > 10_000
    ids = {r["route_id"] for r in routes}
    assert {"10H", "218", "5K"} <= ids
    for name in L.LAYER_FILES:
        assert (ROOT / "gis/layers" / name).exists()
    zones = _load(ROOT / "gis/layers/ghmc_zones.geojson")["features"]
    assert {f["properties"]["code"] for f in zones} >= {"GHMC-LBN", "GHMC-CHR", "GHMC-KHT", "GHMC-SEC",
                                                       "GHMC-SLP", "GHMC-KKP"}
    assert len(_load(ROOT / "gis/layers/crmp_packages.geojson")["features"]) == 7


@pytest.mark.data
def test_build_small_bbox_from_real_pbf(tmp_path):
    if not B.DEFAULT_PBF.exists() or not B.DEFAULT_GTFS.exists():
        pytest.skip("raw OSM/GTFS not downloaded")
    s = B.build(bbox=B.FIXTURE_BBOX, out_dir=tmp_path / "out", layers_dir=tmp_path / "layers")
    assert s["segments"] > 500 and s["segments_without_owner"] == 0
    m = B.make_fixture(tmp_path / "out", tmp_path / "layers", tmp_path / "fx")
    assert m["bytes"] < 2_000_000 and m["counts"]["road_segments"] == s["segments"]


def test_point_helpers():
    p = L.to_ll(L.to_m(Point(78.45, 17.43)))
    assert p.x == pytest.approx(78.45, abs=1e-9) and p.y == pytest.approx(17.43, abs=1e-9)
