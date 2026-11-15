"""Build every geo data file of the WP10 contract.

    python -m geo.build_geo                       # full build on the Hyderabad bbox + fixture clip
    python -m geo.build_geo --bbox 78.40,17.38,78.50,17.46
    python -m geo.build_geo --fixture             # only re-clip tests/fixtures/geo_small from existing outputs

Outputs (WGS84 GeoJSON / JSON):
  data/interim/geo/road_segments.geojson   LineStrings split at intersections (contract properties)
  data/interim/geo/pois.geojson            school|hospital|clinic|bus_stop|metro_station|marketplace
  data/interim/geo/gtfs_routes.json        [{route_id, short_name, long_name, shape, trips_per_day}]
  data/interim/geo/build_summary.json      counts (owner, admin, POI kinds, timings)
  gis/layers/{ghmc_zones,circles,wards,crmp_packages,waterlogging_hotspots,road_owner_overrides}.geojson
  tests/fixtures/geo_small/*               the same files clipped to Ameerpet/Punjagutta (17.42-17.44, 78.44-78.46)

Road ownership (configs/agencies.yaml ``owner_rules``, in order):
  1. ref token NH*  -> NHAI ; ref token SH* -> RNB ; motorway(+_link) named "Outer Ring Road" -> HMDA
  2. built-in supplements: ref token ORR* or name "ORR Service Road" -> HMDA
  3. polygon overrides (gis/layers/road_owner_overrides.geojson, e.g. SCB limits, PVNR Expressway)
  4. default: agency of the admin unit (circle) containing the segment midpoint
  5. otherwise null -> WP4 puts potholes on such roads into the jurisdiction_queue
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import shapely.geometry
import yaml

from geo import gtfs as G
from geo import osm as O

log = logging.getLogger("geo.build_geo")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PBF = ROOT / "data/raw/osm/southern-zone-latest.osm.pbf"
DEFAULT_GTFS = ROOT / "data/raw/gtfs_tgsrtc/tgsrtc_gtfs.zip"
DEFAULT_OUT = ROOT / "data/interim/geo"
DEFAULT_LAYERS = ROOT / "gis/layers"
DEFAULT_FIXTURE = ROOT / "tests/fixtures/geo_small"
DEFAULT_AGENCIES = ROOT / "configs/agencies.yaml"
FIXTURE_BBOX = (78.44, 17.42, 78.46, 17.44)

SEGMENT_PROPS = ("osm_way_id", "seg_idx", "name", "name_te", "highway", "ref", "lanes", "oneway", "lit", "surface",
                 "length_m", "bearing_deg", "junction_start", "junction_end", "gtfs_trips_per_day", "owner_agency",
                 "admin_unit_code", "contract_code")
POI_KINDS = ("school", "hospital", "clinic", "bus_stop", "metro_station", "marketplace")
SUPPLEMENTARY_OWNER_RULES = [
    {"if_osm_ref_prefix": "ORR", "owner": "HMDA"},
    {"if_name_contains": "ORR Service Road", "owner": "HMDA"},
]


def layers_module():
    """``gis.build_layers`` (loaded by path: ``gis`` is not an installed package)."""
    try:
        import gis.build_layers as m  # type: ignore[import-not-found]
        return m
    except ImportError:
        name = "gis.build_layers"
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, ROOT / "gis" / "build_layers.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m


# ----------------------------------------------------------------------------- JSON helpers
def _clean(v):
    """JSON-safe scalar: NaN/NA -> None, numpy -> python."""
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _int_or_none(v):
    v = _clean(v)
    return None if v is None else int(v)


def write_segments_geojson(path: Path, seg: pd.DataFrame, nd: int = 6) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".geojson.tmp")
    geoms = seg.geometry.values
    xy = np.round(shapely.get_coordinates(geoms), nd)
    counts = shapely.get_num_coordinates(geoms)
    offs = np.concatenate([[0], np.cumsum(counts)])
    cols = {c: seg[c].tolist() for c in SEGMENT_PROPS}
    with open(tmp, "w", encoding="utf-8") as f:
        f.write('{"type":"FeatureCollection","features":[\n')
        for i in range(len(seg)):
            props = {c: _clean(cols[c][i]) for c in SEGMENT_PROPS}
            props["osm_way_id"] = int(props["osm_way_id"])
            props["seg_idx"] = int(props["seg_idx"])
            props["lanes"] = _int_or_none(props["lanes"])
            props["oneway"] = bool(props["oneway"])
            props["junction_start"] = bool(props["junction_start"])
            props["junction_end"] = bool(props["junction_end"])
            props["gtfs_trips_per_day"] = float(props["gtfs_trips_per_day"] or 0.0)
            feat = {"type": "Feature", "id": f"{props['osm_way_id']}-{props['seg_idx']}", "properties": props,
                    "geometry": {"type": "LineString", "coordinates": xy[offs[i]:offs[i + 1]].tolist()}}
            f.write(("," if i else "") + json.dumps(feat, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.write("]}\n")
    tmp.replace(path)
    return path


def write_pois_geojson(path: Path, pois: pd.DataFrame) -> Path:
    L = layers_module()
    feats = []
    for r in pois.itertuples(index=False):
        feats.append({"type": "Feature", "properties": {"kind": r.kind, "name": _clean(r.name), "source": r.source,
                                                        "source_id": _clean(r.source_id)},
                      "geometry": {"type": "Point", "coordinates": [round(r.geometry.x, 6), round(r.geometry.y, 6)]}})
    return L.write_fc(path, feats)


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)
    return path


# ----------------------------------------------------------------------------- ownership
def _ref_tokens(ref) -> list[str]:
    ref = _clean(ref)
    if not ref:
        return []
    return [t.strip().upper() for t in str(ref).replace(",", ";").split(";") if t.strip()]


def rule_owner(highway: str, ref, name, rules: Sequence[dict]) -> str | None:
    """First matching ref/name/highway rule (``default_by_admin_unit`` rules are skipped)."""
    toks = _ref_tokens(ref)
    nm = str(_clean(name) or "")
    for r in rules:
        if r.get("default_by_admin_unit"):
            continue
        ok = True
        if "if_osm_ref_prefix" in r:
            ok &= any(t.startswith(str(r["if_osm_ref_prefix"]).upper()) for t in toks)
        if "if_highway" in r:
            ok &= highway in (r["if_highway"], f"{r['if_highway']}_link")
        if "if_name_contains" in r:
            ok &= str(r["if_name_contains"]).lower() in nm.lower()
        if ok and any(k.startswith("if_") for k in r):
            return r["owner"]
    return None


def _midpoints(geoms: np.ndarray) -> np.ndarray:
    return shapely.line_interpolate_point(geoms, 0.5, normalized=True)


def assign_admin_and_owner(seg: pd.DataFrame, circles_fc: dict, zones_fc: dict, overrides_fc: dict,
                           owner_rules: Sequence[dict]) -> pd.DataFrame:
    """Add owner_agency, owner_basis, admin_unit_code (circle), zone_code columns."""
    mids = _midpoints(seg.geometry.values)
    n = len(seg)
    # admin unit (circle) by midpoint
    c_geoms = np.array([shapely.geometry.shape(f["geometry"]) for f in circles_fc["features"]], dtype=object)
    c_props = [f["properties"] for f in circles_fc["features"]]
    admin = np.full(n, None, dtype=object)
    agency = np.full(n, None, dtype=object)
    zone = np.full(n, None, dtype=object)
    if len(c_geoms):
        tree = shapely.STRtree(c_geoms)
        mi, ci = tree.query(mids, predicate="intersects")
        first = np.unique(mi, return_index=True)[1]
        for m, c in zip(mi[first], ci[first], strict=True):
            admin[m] = c_props[c]["code"]
            agency[m] = c_props[c].get("agency")
            zone[m] = c_props[c].get("parent_code")
    # rules
    rules = list(owner_rules) + SUPPLEMENTARY_OWNER_RULES
    hw, refs, names = seg["highway"].tolist(), seg["ref"].tolist(), seg["name"].tolist()
    owner = np.array([rule_owner(h, r, nm, rules) for h, r, nm in zip(hw, refs, names, strict=True)], dtype=object)
    basis = np.where(pd.notna(owner), "rule", None).astype(object)
    # polygon overrides (only where no rule matched)
    o_feats = overrides_fc["features"]
    if o_feats:
        o_geoms = np.array([shapely.geometry.shape(f["geometry"]) for f in o_feats], dtype=object)
        tree = shapely.STRtree(o_geoms)
        mi, oi = tree.query(mids, predicate="intersects")
        lname = pd.Series(names).fillna("").astype(str).str.lower().tolist()
        for m, o in zip(mi, oi, strict=True):
            if owner[m] is not None and basis[m] == "rule":
                continue
            p = o_feats[o]["properties"]
            nc = p.get("name_contains")
            if nc and nc.lower() not in lname[m]:
                continue
            owner[m] = p["owner_agency"]
            basis[m] = "override"
    # default: agency of containing admin unit
    need = np.array([o is None for o in owner])
    owner[need] = agency[need]
    basis[need & pd.notna(agency)] = "admin_unit"
    out = seg.copy()
    out["owner_agency"] = owner
    out["owner_basis"] = basis
    out["admin_unit_code"] = admin
    out["zone_code"] = zone
    return out


def assign_contracts(seg: pd.DataFrame, package_polys_m: dict, highways: Sequence[str],
                     package_agency: dict[str, str]) -> np.ndarray:
    L = layers_module()
    out = np.full(len(seg), None, dtype=object)
    if not package_polys_m:
        return out
    codes = list(package_polys_m)
    polys = np.array([L.to_ll(package_polys_m[c]) for c in codes], dtype=object)
    mids = _midpoints(seg.geometry.values)
    tree = shapely.STRtree(polys)
    mi, pi = tree.query(mids, predicate="intersects")
    hw = seg["highway"].isin(list(highways)).to_numpy()
    own = seg["owner_agency"].to_numpy()
    for m, p in zip(mi, pi, strict=True):
        c = codes[p]
        if out[m] is None and hw[m] and own[m] == package_agency[c]:
            out[m] = c
    return out


# ----------------------------------------------------------------------------- fixture
def _clip_fc(fc: dict, box, clip: bool) -> dict:
    feats = []
    for f in fc["features"]:
        g = shapely.geometry.shape(f["geometry"])
        if not g.intersects(box):
            continue
        if clip and g.geom_type in ("Polygon", "MultiPolygon"):
            g = g.intersection(box)
            if g.is_empty:
                continue
            L = layers_module()
            f = L.feature(L._polys(g), f["properties"])
        feats.append(f)
    out = {"type": "FeatureCollection", "features": feats}
    if "metadata" in fc:
        out["metadata"] = fc["metadata"]
    return out


def make_fixture(out_dir: Path, layers_dir: Path, fixture_dir: Path, bbox: Sequence[float] = FIXTURE_BBOX,
                 route_margin_m: float = 300.0) -> dict:
    """Clip every contract file to ``bbox`` into ``fixture_dir`` (segments/POIs whole, polygons clipped)."""
    L = layers_module()
    box = shapely.box(*bbox)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with open(out_dir / "road_segments.geojson", encoding="utf-8") as f:
        segs = json.load(f)
    segs_c = _clip_fc(segs, box, clip=False)
    L.write_fc(fixture_dir / "road_segments.geojson", segs_c["features"])
    counts["road_segments"] = len(segs_c["features"])
    with open(out_dir / "pois.geojson", encoding="utf-8") as f:
        pois_c = _clip_fc(json.load(f), box, clip=False)
    L.write_fc(fixture_dir / "pois.geojson", pois_c["features"])
    counts["pois"] = len(pois_c["features"])
    with open(out_dir / "gtfs_routes.json", encoding="utf-8") as f:
        routes = json.load(f)
    rbox = L.to_ll(L.to_m(box).buffer(route_margin_m))
    rc = []
    for r in routes:
        line = shapely.linestrings(r["shape"]) if len(r["shape"]) >= 2 else None
        if line is None or not line.intersects(box):
            continue
        part = line.intersection(rbox)
        parts = [p for p in getattr(part, "geoms", [part]) if p.geom_type == "LineString"]
        if not parts:
            continue
        part = max(parts, key=lambda p: p.length)
        rc.append({**r, "shape": np.round(shapely.get_coordinates(part), 6).tolist(), "clipped": True})
    write_json(fixture_dir / "gtfs_routes.json", rc)
    counts["gtfs_routes"] = len(rc)
    for name in L.LAYER_FILES:
        p = layers_dir / name
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            fc = _clip_fc(json.load(f), box, clip=True)
        L.write_fc(fixture_dir / name, fc["features"], fc.get("metadata"))
        counts[name.removesuffix(".geojson")] = len(fc["features"])
    manifest = {"bbox": list(bbox), "description": "Ameerpet/Punjagutta clip of the WP10 geo build (tests + fast seeding)",
                "counts": counts, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "notes": "road segments and POIs are kept whole if they intersect the box; polygons are clipped; "
                         f"route shapes are clipped to the box + {route_margin_m:.0f} m (clipped=true)."}
    write_json(fixture_dir / "manifest.json", manifest)
    total = sum(p.stat().st_size for p in fixture_dir.iterdir() if p.is_file())
    manifest["bytes"] = total
    write_json(fixture_dir / "manifest.json", manifest)
    return manifest


# ----------------------------------------------------------------------------- full build
def build(pbf: Path = DEFAULT_PBF, gtfs_zip: Path = DEFAULT_GTFS, bbox: Sequence[float] = O.HYD_BBOX,
          out_dir: Path = DEFAULT_OUT, layers_dir: Path = DEFAULT_LAYERS, agencies_path: Path = DEFAULT_AGENCIES,
          seeds_path: Path | None = None, route_shapes: bool = True, crop_dir: Path = DEFAULT_OUT) -> dict:
    L = layers_module()
    t0 = time.time()
    timings: dict[str, float] = {}

    def lap(k):
        timings[k] = round(time.time() - t0 - sum(timings.values()), 1)
        log.info("%-12s %6.1f s", k, timings[k])

    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = L.load_seeds(seeds_path or L.SEEDS_PATH)
    with open(agencies_path, encoding="utf-8") as f:
        agencies = yaml.safe_load(f)

    crop_box = O.HYD_BBOX if all(a <= b for a, b in zip(O.HYD_BBOX[:2], bbox[:2], strict=True)) and \
        all(a >= b for a, b in zip(O.HYD_BBOX[2:], bbox[2:], strict=True)) else tuple(bbox)
    crop_name = "hyderabad.osm.pbf" if tuple(crop_box) == O.HYD_BBOX else "crop_{}_{}_{}_{}.osm.pbf".format(*crop_box)
    crop = O.crop_pbf(pbf, crop_dir / crop_name, crop_box)
    lap("crop")

    ways = O.filter_ways(O.read_highway_ways(crop, bbox))
    seg = O.build_segments(ways, bbox)
    log.info("%d ways -> %d segments", len(ways), len(seg))
    lap("segments")

    # ---- admin layers
    bnd = O.read_admin_boundaries(crop, crop_box)
    ghmc_ll = L.ghmc_polygon(bnd)
    wards_ll = L.osm_wards(bnd)
    zones, circles, wards = L.build_ghmc_units(ghmc_ll, wards_ll, seeds)
    scb_m, scb_src = L.scb_polygon_m(ghmc_ll, seeds)
    scb_zone, scb_circle = L.scb_units(scb_m, seeds)
    orr_m = L.orr_ring_polygon_m(seg)
    pz, pc = L.build_periurban(orr_m, ghmc_ll, scb_m, seeds)
    vf = seeds["valid_from"]
    zones_fc = {"type": "FeatureCollection", "features": L.unit_features(zones + [scb_zone] + pz, vf)}
    circles_fc = {"type": "FeatureCollection", "features": L.unit_features(circles + [scb_circle] + pc, vf)}
    wards_fc = {"type": "FeatureCollection", "features": L.unit_features(wards, vf)}
    overrides_fc = {"type": "FeatureCollection", "features": L.override_features(seg, scb_m, scb_src, seeds)}
    lap("admin")

    # ---- ownership / admin unit
    seg = assign_admin_and_owner(seg, circles_fc, zones_fc, overrides_fc, agencies.get("owner_rules", []))
    zones_by_name = {z["name_en"]: z for z in zones}
    crmp_feats, crmp_polys = L.crmp_features(seg, zones_by_name, agencies.get("crmp_packages", []), seeds)
    pkg_agency = {p["code"]: p["agency"] for p in agencies.get("crmp_packages", [])}
    seg["contract_code"] = assign_contracts(seg, crmp_polys, seeds["crmp"]["highways"], pkg_agency)
    lap("ownership")

    # ---- POIs
    pois_osm = O.read_pois(crop, bbox)
    feed = G.load_feed(gtfs_zip)
    pois_bus = G.bus_stop_pois(feed, bbox)
    pois = pd.concat([pois_osm, pois_bus], ignore_index=True)
    lap("pois")

    # ---- GTFS shapes + trips/day per segment
    patterns = G.trip_patterns(feed)
    router = G.RoadRouter(seg) if route_shapes else None
    shapes = G.build_pattern_shapes(feed, patterns, router=router, bbox=bbox)
    seg["gtfs_trips_per_day"] = G.assign_trips_to_segments(seg.geometry.values, shapes, patterns["trips"].to_numpy())
    routes = G.route_records(feed, patterns, shapes)
    lap("gtfs")

    # ---- write
    write_segments_geojson(out_dir / "road_segments.geojson", seg)
    write_pois_geojson(out_dir / "pois.geojson", pois)
    write_json(out_dir / "gtfs_routes.json", routes)
    meta = {"official": False, "built_with": "geo.build_geo", "licence": "derived from OSM (ODbL) + seeds"}
    L.write_fc(layers_dir / "ghmc_zones.geojson", zones_fc["features"], meta)
    L.write_fc(layers_dir / "circles.geojson", circles_fc["features"], meta)
    L.write_fc(layers_dir / "wards.geojson", wards_fc["features"], meta)
    L.write_fc(layers_dir / "crmp_packages.geojson", crmp_feats, meta)
    L.write_fc(layers_dir / "waterlogging_hotspots.geojson", L.hotspot_features(seeds), meta)
    L.write_fc(layers_dir / "road_owner_overrides.geojson", overrides_fc["features"], meta)
    lap("write")

    own = seg["owner_agency"].fillna("NULL(queue)").value_counts().to_dict()
    summary = {
        "bbox": list(bbox),
        "segments": len(seg), "ways": len(ways), "total_km": round(float(seg["length_m"].sum()) / 1000, 1),
        "owner_agency": own,
        "owner_basis": seg["owner_basis"].fillna("none").value_counts().to_dict(),
        "segments_without_owner": int(seg["owner_agency"].isna().sum()),
        "segments_with_admin_unit": int(seg["admin_unit_code"].notna().sum()),
        "segments_with_contract": int(seg["contract_code"].notna().sum()),
        "segments_with_bus_trips": int((seg["gtfs_trips_per_day"] > 0).sum()),
        "highway": seg["highway"].value_counts().to_dict(),
        "surface": seg["surface"].value_counts().to_dict(),
        "pois": pois["kind"].value_counts().to_dict(),
        "gtfs_routes": len(routes), "gtfs_patterns": len(patterns),
        "gtfs_trips_per_day_total": float(patterns["trips"].sum()),
        "admin_units": {"zones": len(zones_fc["features"]), "circles": len(circles_fc["features"]),
                        "wards": len(wards_fc["features"])},
        "crmp_packages": {f["properties"]["code"]: f["properties"]["corridor_km"] for f in crmp_feats},
        "scb_source": scb_src,
        "timings_s": timings,
    }
    write_json(out_dir / "build_summary.json", summary)
    return summary


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    v = [float(x) for x in s.split(",")]
    if len(v) != 4 or v[0] >= v[2] or v[1] >= v[3]:
        raise argparse.ArgumentTypeError("bbox must be minlon,minlat,maxlon,maxlat")
    return tuple(v)  # type: ignore[return-value]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m geo.build_geo", description=__doc__.split("\n")[0])
    ap.add_argument("--pbf", type=Path, default=DEFAULT_PBF)
    ap.add_argument("--gtfs", type=Path, default=DEFAULT_GTFS)
    ap.add_argument("--bbox", type=_parse_bbox, default=O.HYD_BBOX, help="minlon,minlat,maxlon,maxlat")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--layers", type=Path, default=DEFAULT_LAYERS)
    ap.add_argument("--agencies", type=Path, default=DEFAULT_AGENCIES)
    ap.add_argument("--seeds", type=Path, default=None)
    ap.add_argument("--fixture", action="store_true", help="only re-clip the fixture from existing outputs")
    ap.add_argument("--no-fixture", action="store_true", help="skip the fixture clip after the build")
    ap.add_argument("--fixture-out", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--fixture-bbox", type=_parse_bbox, default=FIXTURE_BBOX)
    ap.add_argument("--straight-shapes", action="store_true",
                    help="join GTFS stops with straight lines instead of routing over the road graph")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not a.fixture:
        s = build(a.pbf, a.gtfs, a.bbox, a.out, a.layers, a.agencies, a.seeds, route_shapes=not a.straight_shapes)
        print(json.dumps({k: s[k] for k in ("segments", "total_km", "owner_agency", "segments_without_owner",
                                            "pois", "gtfs_routes", "segments_with_bus_trips", "timings_s")},
                         indent=1, ensure_ascii=False))
    if a.fixture or not a.no_fixture:
        m = make_fixture(a.out, a.layers, a.fixture_out, a.fixture_bbox)
        print(json.dumps(m, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
