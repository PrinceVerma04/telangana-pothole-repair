"""Approximate administrative / contract / hazard layers for Hyderabad (WP10).

All outputs are WGS84 GeoJSON FeatureCollections under ``gis/layers/`` and are marked
``official: false`` - they are demo approximations built from:

* OSM (ODbL): GHMC 2016 limit (relation 7868535 "Hyderabad", admin_level 8) and the
  GHMC ward relations (admin_level 10, 149 of 150 wards present in the 2026-11 extract);
* hand-placed seeds in ``gis/seeds.yaml`` (circle centres, peri-urban corporation seeds,
  waterlogging hotspots reported in news, override corridors);
* the road segments built by ``geo.osm`` (Outer Ring Road ring, arterial corridors).

Construction:
  GHMC circles  = wards assigned to the circle whose Voronoi cell (of the 30 circle seeds,
                  clipped to GHMC) they overlap most + the GHMC area no ward covers, split
                  by the same Voronoi cells. Zones = union of their circles. A partition.
  SCB           = the notch of the GHMC polygon's hull that contains Trimulgherry.
  CMC / MMC     = (ORR ring interior + 1 km) - GHMC - SCB, split by Voronoi of their seeds.
  CRMP packages = 20 m buffers of GHMC-owned arterial segments per hinted zone.

Geometry maths is done in EPSG:32644 (metres) and written back in EPSG:4326.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import yaml
from pyproj import Transformer
from shapely.geometry import MultiPoint, Point, Polygon, mapping

log = logging.getLogger(__name__)

METRIC_EPSG = 32644
_TO_M = Transformer.from_crs(4326, METRIC_EPSG, always_xy=True)
_TO_LL = Transformer.from_crs(METRIC_EPSG, 4326, always_xy=True)
SEEDS_PATH = Path(__file__).with_name("seeds.yaml")
LAYER_FILES = ("ghmc_zones.geojson", "circles.geojson", "wards.geojson", "crmp_packages.geojson",
               "waterlogging_hotspots.geojson", "road_owner_overrides.geojson")


# ----------------------------------------------------------------------------- helpers
def to_m(g):
    return shapely.transform(g, lambda xy: np.column_stack(_TO_M.transform(xy[:, 0], xy[:, 1])))


def to_ll(g):
    return shapely.transform(g, lambda xy: np.column_stack(_TO_LL.transform(xy[:, 0], xy[:, 1])))


def load_seeds(path: str | Path = SEEDS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _polys(g) -> shapely.Geometry:
    """Keep only polygonal parts of ``g`` (drops slivers that became lines/points)."""
    if g is None or g.is_empty:
        return Polygon()
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    parts = [p for p in getattr(g, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.union_all(parts) if parts else Polygon()


def _round_coords(obj, nd: int):
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(float(v), nd) for v in obj]
        return [_round_coords(o, nd) for o in obj]
    return obj


def feature(geom_ll, props: dict, nd: int = 6) -> dict:
    gj = mapping(geom_ll)
    return {"type": "Feature", "properties": props,
            "geometry": {"type": gj["type"], "coordinates": _round_coords(gj["coordinates"], nd)}}


def write_fc(path: str | Path, features: Iterable[dict], meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection", "features": list(features)}
    if meta:
        fc["metadata"] = meta
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)
    return path


def _voronoi_cells(points_m: Sequence[Point], extent) -> list:
    """Voronoi cell per input point (same order), clipped to ``extent``."""
    mp = MultiPoint(list(points_m))
    vor = shapely.voronoi_polygons(mp, extend_to=extent.envelope.buffer(10_000), ordered=True)
    cells = list(vor.geoms)
    out = []
    for p in points_m:  # ``ordered`` keeps order, but verify by containment for safety
        c = next((c for c in cells if c.contains(p)), None)
        if c is None:
            c = min(cells, key=lambda c: c.distance(p))
        out.append(_polys(c.intersection(extent)))
    return out


def _circle_rows(circles: list) -> list[dict]:
    return [{"code": c[0], "name_en": c[1], "name_te": c[2], "name_hi": c[3],
             "lat": float(c[4]), "lon": float(c[5]), "loc_source": c[6]} for c in circles]


# ----------------------------------------------------------------------------- OSM boundaries
def ghmc_polygon(boundaries: pd.DataFrame):
    b = boundaries
    m = b[(b["name"].astype(str) == "Hyderabad") & (b["admin_level"].astype(str) == "8")]
    if len(m) == 0:
        raise ValueError("GHMC boundary (Hyderabad, admin_level 8) not found in OSM boundaries")
    return _polys(shapely.make_valid(m.geometry.iloc[0]))


_WARD_RE = re.compile(r"Ward\s+(\d+)\s*(.*)", re.I)


def osm_wards(boundaries: pd.DataFrame) -> list[dict]:
    """GHMC wards from OSM (admin_level 10, polygons only), WGS84."""
    b = boundaries[boundaries["admin_level"].astype(str) == "10"]
    out, seen = [], set()
    for _, r in b.iterrows():
        g = r.geometry
        if g is None or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        m = _WARD_RE.match(str(r["name"] or ""))
        if not m:
            continue
        no = int(m.group(1))
        if no in seen:
            continue
        seen.add(no)
        te = r.get("name:te")
        te = None if te is None or (isinstance(te, float) and np.isnan(te)) or str(te) in ("", "nan") else str(te)
        out.append({"ward_no": no, "name_en": m.group(2).strip() or f"Ward {no}", "name_te": te,
                    "osm_id": int(r["id"]), "geometry": _polys(shapely.make_valid(g))})
    return sorted(out, key=lambda w: w["ward_no"])


# ----------------------------------------------------------------------------- GHMC zones/circles/wards
def build_ghmc_units(ghmc_ll, wards_ll: list[dict], seeds: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (zones, circles, wards) as dicts with ``geometry_m`` (metric) and properties."""
    g = seeds["ghmc"]
    ghmc_m = to_m(ghmc_ll)
    circ, zone_of = [], {}
    for z in g["zones"]:
        for c in _circle_rows(z["circles"]):
            c["zone_code"] = z["code"]
            circ.append(c)
            zone_of[c["code"]] = z
    seed_pts = [to_m(Point(c["lon"], c["lat"])) for c in circ]
    cells = _voronoi_cells(seed_pts, ghmc_m)

    ward_geoms_m = [_polys(to_m(w["geometry"]).intersection(ghmc_m)) for w in wards_ll]
    ward_circle = []
    for wg in ward_geoms_m:
        ov = [wg.intersection(c).area if not wg.is_empty else 0.0 for c in cells]
        ward_circle.append(int(np.argmax(ov)) if max(ov, default=0) > 0 else -1)

    covered = shapely.union_all([w for w in ward_geoms_m if not w.is_empty]) if ward_geoms_m else Polygon()
    gaps = _polys(ghmc_m.difference(covered))
    circle_geoms = []
    for i, cell in enumerate(cells):
        parts = [ward_geoms_m[j] for j, ci in enumerate(ward_circle) if ci == i]
        parts.append(_polys(gaps.intersection(cell)))
        circle_geoms.append(_polys(shapely.union_all([p for p in parts if not p.is_empty])))
    # enforce a partition (OSM wards overlap slightly in places)
    taken = Polygon()
    for i in range(len(circle_geoms)):
        circle_geoms[i] = _polys(circle_geoms[i].difference(taken).intersection(ghmc_m))
        taken = shapely.union_all([taken, circle_geoms[i]])

    circles = []
    for c, geom in zip(circ, circle_geoms, strict=True):
        if geom.is_empty:
            log.warning("GHMC circle %s ended up empty", c["code"])
            continue
        circles.append({"code": c["code"], "kind": "circle", "agency": g["agency"], "name_en": c["name_en"],
                        "name_te": c["name_te"], "name_hi": c["name_hi"], "parent_code": c["zone_code"],
                        "seed_source": c["loc_source"], "geometry_m": geom})
    zones = []
    for z in g["zones"]:
        geom = _polys(shapely.union_all([c["geometry_m"] for c in circles if c["parent_code"] == z["code"]]))
        zones.append({"code": z["code"], "kind": "zone", "agency": g["agency"], "name_en": z["name_en"],
                      "name_te": z["name_te"], "name_hi": z["name_hi"], "parent_code": None, "geometry_m": geom})
    wards = []
    for w, wg, ci in zip(wards_ll, ward_geoms_m, ward_circle, strict=True):
        if wg.is_empty or ci < 0:
            continue
        wards.append({"code": f"GHMC-W{w['ward_no']:03d}", "kind": "ward", "agency": g["agency"],
                      "name_en": w["name_en"], "name_te": w["name_te"], "name_hi": None,
                      "parent_code": circ[ci]["code"], "osm_id": w["osm_id"], "geometry_m": wg})
    return zones, circles, wards


def scb_polygon_m(ghmc_ll, seeds: dict):
    """Secunderabad Cantonment: the hull notch of GHMC containing the probe point."""
    s = seeds["scb"]
    ghmc_m = to_m(ghmc_ll)
    probe = to_m(Point(s["probe"][1], s["probe"][0]))
    lo, hi = s["expected_km2"]
    notch = _polys(ghmc_m.convex_hull.difference(ghmc_m))
    parts = list(getattr(notch, "geoms", [notch]))
    hit = next((p for p in parts if p.contains(probe)), None)
    if hit is not None:
        # the hull notch may be open to the outside; keep the part near the probe (8 km radius)
        cand = _polys(hit.intersection(probe.buffer(8_000)))
        cand = max(getattr(cand, "geoms", [cand]), key=lambda p: p.area) if not cand.is_empty else cand
        if not cand.is_empty and lo <= cand.area / 1e6 <= hi:
            return cand, "osm GHMC notch"
    ring = [(lon, lat) for lat, lon in s["fallback_polygon"]]
    return _polys(to_m(Polygon(ring)).difference(ghmc_m)), "seeds fallback polygon"


def orr_ring_polygon_m(segments: pd.DataFrame):
    """Polygon enclosed by the Outer Ring Road (ref ORR / name Outer Ring Road motorway)."""
    ref = segments["ref"].fillna("").astype(str)
    name = segments["name"].fillna("").astype(str)
    sel = segments["highway"].isin(["motorway"]) & (ref.str.contains("ORR") | name.str.contains("Outer Ring Road"))
    if sel.sum() == 0:
        return None
    lines = to_m(shapely.union_all(segments.geometry.values[sel.to_numpy()]))
    ring = lines.buffer(150)
    polys = list(getattr(ring, "geoms", [ring]))
    big = max(polys, key=lambda p: p.area)
    interior = Polygon(big.exterior)
    if big.interiors and interior.area > 300e6:
        return interior
    log.warning("ORR ring not closed; using the convex hull of the ORR carriageways")
    return lines.convex_hull


def build_periurban(orr_m, ghmc_ll, scb_m, seeds: dict, margin_m: float = 1000.0) -> tuple[list[dict], list[dict]]:
    if orr_m is None:
        return [], []
    area = _polys(orr_m.buffer(margin_m).difference(to_m(ghmc_ll)).difference(scb_m))
    rows = []
    for z in seeds["periurban"]:
        for c in _circle_rows(z["circles"]):
            c.update(zone_code=z["code"], agency=z["agency"])
            rows.append(c)
    cells = _voronoi_cells([to_m(Point(c["lon"], c["lat"])) for c in rows], area)
    circles = []
    for c, geom in zip(rows, cells, strict=True):
        if geom.is_empty:
            continue
        circles.append({"code": c["code"], "kind": "circle", "agency": c["agency"], "name_en": c["name_en"],
                        "name_te": c["name_te"], "name_hi": c["name_hi"], "parent_code": c["zone_code"],
                        "seed_source": c["loc_source"], "geometry_m": geom})
    zones = []
    for z in seeds["periurban"]:
        parts = [c["geometry_m"] for c in circles if c["parent_code"] == z["code"]]
        if not parts:
            continue
        zones.append({"code": z["code"], "kind": "zone", "agency": z["agency"], "name_en": z["name_en"],
                      "name_te": z["name_te"], "name_hi": z["name_hi"], "parent_code": None,
                      "geometry_m": _polys(shapely.union_all(parts))})
    return zones, circles


def scb_units(scb_m, seeds: dict) -> tuple[dict, dict]:
    s = seeds["scb"]
    base = {"agency": s["agency"], "name_en": s["name_en"], "name_te": s["name_te"], "name_hi": s["name_hi"]}
    zone = {"code": s["code"], "kind": "zone", "parent_code": None, "geometry_m": scb_m, **base}
    circle = {"code": s["circle_code"], "kind": "circle", "parent_code": s["code"], "geometry_m": scb_m, **base}
    return zone, circle


def unit_features(units: list[dict], valid_from: str, simplify_m: float = 2.0) -> list[dict]:
    feats = []
    for u in units:
        g = u["geometry_m"]
        if simplify_m:
            g = _polys(shapely.make_valid(shapely.simplify(g, simplify_m, preserve_topology=True)))
        props = {k: u.get(k) for k in ("code", "kind", "name_en", "name_te", "name_hi", "parent_code", "agency")}
        props.update(official=False, valid_from=valid_from, valid_to=None,
                     source="approx: OSM wards/GHMC limit + seeds (gis/seeds.yaml)",
                     area_km2=round(u["geometry_m"].area / 1e6, 3))
        for k in ("seed_source", "osm_id"):
            if u.get(k) is not None:
                props[k] = u[k]
        feats.append(feature(to_ll(g), props))
    return feats


# ----------------------------------------------------------------------------- overrides / hotspots / CRMP
def override_features(segments: pd.DataFrame, scb_m, scb_source: str, seeds: dict) -> list[dict]:
    feats = [feature(to_ll(scb_m), {"owner_agency": seeds["scb"]["agency"],
                                    "reason": f"Secunderabad Cantonment Board limits (approx.; {scb_source})",
                                    "name_contains": None})]
    names = segments["name"].fillna("").astype(str)
    for o in seeds.get("corridor_overrides", []):
        sel = names.str.contains(o["name_contains"], case=False, regex=False).to_numpy()
        if not sel.any():
            log.warning("override corridor %r: no segment named like that", o["name_contains"])
            continue
        poly = to_m(shapely.union_all(segments.geometry.values[sel])).buffer(o.get("buffer_m", 20), quad_segs=2)
        feats.append(feature(to_ll(poly.simplify(2.0)), {"owner_agency": o["owner_agency"], "reason": o["reason"],
                                                         "name_contains": o["name_contains"]}))
    return feats


def hotspot_features(seeds: dict) -> list[dict]:
    h = seeds["waterlogging_hotspots"]
    feats = []
    for name, lat, lon in h["spots"]:
        poly = to_m(Point(lon, lat)).buffer(h["radius_m"], quad_segs=4)
        feats.append(feature(to_ll(poly), {"name": name, "source": h["source"], "radius_m": h["radius_m"]}))
    return feats


def crmp_features(segments: pd.DataFrame, zones_by_name: dict[str, dict], packages: list[dict],
                  seeds: dict) -> tuple[list[dict], dict[str, object]]:
    """CRMP package corridors. ``segments`` needs owner_agency + zone_code columns.

    Returns (features, {package_code: metric polygon}).
    """
    cfg = seeds["crmp"]
    hw = segments["highway"].isin(cfg["highways"]).to_numpy()
    feats, polys = [], {}
    for p in packages:
        hint = p.get("zone_hint", "")
        base, _, side = hint.partition("-")
        z = zones_by_name.get(base)
        if z is None:
            log.warning("CRMP %s: zone %r not found", p["code"], base)
            continue
        sel = hw & (segments["zone_code"].to_numpy() == z["code"]) & (segments["owner_agency"].to_numpy() == p["agency"])
        if not sel.any():
            continue
        geoms = segments.geometry.values[sel]
        mid_lon = shapely.get_x(shapely.line_interpolate_point(geoms, 0.5, normalized=True))
        has_west = any(q.get("zone_hint", "") == f"{base}-West" for q in packages)
        if side.lower() == "west":
            geoms = geoms[mid_lon < np.median(mid_lon)]
        elif has_west:
            geoms = geoms[mid_lon >= np.median(mid_lon)]
        poly = to_m(shapely.union_all(geoms)).buffer(cfg["buffer_m"], quad_segs=2).simplify(3.0)
        polys[p["code"]] = poly
        km = float(shapely.length(to_m(shapely.union_all(geoms)))) / 1000
        feats.append(feature(to_ll(poly), {
            "code": p["code"], "agency": p["agency"], "contractor": p["contractor"],
            "dlp_months": int(p["dlp_months"]), "zone_hint": hint, "corridor_km": round(km, 1),
            "official": False, "source": "approx: GHMC-owned arterials of the hinted zone (20 m buffer)",
        }))
    return feats, polys
