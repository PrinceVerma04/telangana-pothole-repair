"""OSM PBF -> road segments (split at intersections) and POIs (WP10 / WP5 data prep).

Everything comes from a Geofabrik PBF via pyrosm (no Overpass, no osmium CLI):

1. ``crop_pbf`` cuts the 560 MB southern-zone extract down to the Hyderabad bbox once
   (complete ways, ~38 MB) and caches it under ``data/interim/geo/``.
2. ``read_highway_ways`` reads drivable ways with their ordered node ids.
3. ``split_ways`` splits ways at every node shared by >= 2 ways (or visited twice by one
   way) - the classic "split at intersections" - fully vectorised with numpy.
4. ``build_segments`` adds length (EPSG:32644 metres), bearing, normalised tags and
   junction flags (graph degree >= 3 at the segment end node).
5. ``read_pois`` returns schools/colleges, hospitals/clinics, marketplaces and Hyderabad
   Metro stations as points.

Units: metres, degrees (bearing clockwise from true north, 0-360). Coordinates WGS84.
The PBF is untrusted input: it is only *parsed*, never executed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Geod, Transformer

log = logging.getLogger(__name__)

HYD_BBOX: tuple[float, float, float, float] = (78.1, 17.1, 78.8, 17.7)  # minx, miny, maxx, maxy (lon/lat)
METRIC_EPSG = 32644  # UTM 44N, covers Telangana

ARTERIAL_HIGHWAYS = ("motorway", "trunk", "primary", "secondary")
DRIVABLE_HIGHWAYS: tuple[str, ...] = (
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link",
    "residential", "living_street", "unclassified", "service",
)
# service=* sub-types that are not public carriageways (no pothole dispatch there)
EXCLUDED_SERVICE = {"parking_aisle", "driveway", "drive-through", "emergency_access", "parking"}
EXCLUDED_ACCESS = {"private", "no"}

WAY_TAGS = ["highway", "name", "name:te", "ref", "lanes", "oneway", "lit", "surface",
            "junction", "access", "area", "service", "operator"]

_GEOD = Geod(ellps="WGS84")


# ----------------------------------------------------------------------------- PBF I/O
def crop_pbf(src: str | Path, dst: str | Path, bbox: Sequence[float] = HYD_BBOX, force: bool = False) -> Path:
    """Crop ``src`` to ``bbox`` (complete ways) into ``dst`` unless an up-to-date crop exists."""
    from pyrosm import OSM

    src, dst = Path(src), Path(dst)
    if dst.exists() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    osm = OSM(str(src), bounding_box=list(bbox), keep_metadata=False, progress=False)
    tmp = dst.with_suffix(".tmp.pbf")
    osm.to_pbf(str(tmp), compact=True)
    tmp.replace(dst)
    return dst


def _osm(pbf: str | Path, bbox: Sequence[float] | None, **kw):
    from pyrosm import OSM

    return OSM(str(pbf), bounding_box=list(bbox) if bbox is not None else None,
               keep_metadata=False, progress=False, **kw)


def read_highway_ways(pbf: str | Path, bbox: Sequence[float] | None = HYD_BBOX,
                      highways: Iterable[str] = DRIVABLE_HIGHWAYS) -> pd.DataFrame:
    """Drivable ways with ordered node ids (``nodes``) and line geometry (pyrosm)."""
    osm = _osm(pbf, bbox, keep_node_info=True)
    gdf = osm.get_data_by_custom_criteria(
        custom_filter={"highway": list(highways)}, keep_nodes=False, keep_relations=False,
        tags_as_columns=WAY_TAGS, keep_other_tags=False,
    )
    if gdf is None or len(gdf) == 0:
        return pd.DataFrame(columns=["id", "nodes", "geometry", *WAY_TAGS])
    for c in WAY_TAGS:
        if c not in gdf.columns:
            gdf[c] = None
    return pd.DataFrame(gdf)


def filter_ways(ways: pd.DataFrame, highways: Iterable[str] = DRIVABLE_HIGHWAYS) -> pd.DataFrame:
    """Keep public drivable carriageways (no areas, parking aisles, driveways, private)."""
    hw = set(highways)
    w = ways
    keep = w["highway"].isin(hw)
    keep &= w.geometry.map(lambda g: g is not None and g.geom_type == "LineString")
    keep &= ~w["area"].fillna("").astype(str).str.lower().eq("yes")
    keep &= ~(w["highway"].eq("service") & w["service"].fillna("").astype(str).isin(EXCLUDED_SERVICE))
    keep &= ~w["access"].fillna("").astype(str).str.lower().isin(EXCLUDED_ACCESS)
    return w[keep].reset_index(drop=True)


# ----------------------------------------------------------------------------- tag normalisation
_CC_SURFACES = {"concrete", "concrete:lanes", "concrete:plates", "cement", "cement_concrete", "cc", "rcc"}
_BT_SURFACES = {"asphalt", "bitumen", "tar", "tarmac", "chipseal", "paved", "bituminous", "bt"}
_DEFAULT_BT_HIGHWAYS = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                        "secondary", "secondary_link", "tertiary", "tertiary_link"}


def norm_surface(surface: object, highway: str | None) -> str:
    """OSM ``surface`` -> ``BT`` | ``CC`` | ``unknown``.

    concrete* / cement* -> CC; asphalt/paved/tar -> BT; unpaved/blocks/other -> unknown.
    Missing tag: BT on tertiary-and-above (Hyderabad arterials are bituminous), otherwise
    unknown (many colony/residential lanes are CC roads, so we do not guess).
    """
    if surface is None or (isinstance(surface, float) and np.isnan(surface)) or str(surface).strip() == "":
        return "BT" if highway in _DEFAULT_BT_HIGHWAYS else "unknown"
    s = str(surface).strip().lower()
    if s in _CC_SURFACES or s.startswith("concrete") or "cement" in s:
        return "CC"
    if s in _BT_SURFACES or "asphalt" in s or "tar" in s.split("_"):
        return "BT"
    return "unknown"


def norm_lit(lit: object) -> str | None:
    if lit is None or (isinstance(lit, float) and np.isnan(lit)):
        return None
    s = str(lit).strip().lower()
    if s in {"yes", "24/7", "automatic", "sunset-sunrise", "dusk-dawn", "limited", "interval"}:
        return "yes"
    if s in {"no", "disused"}:
        return "no"
    return None


def norm_lanes(lanes: object) -> int | None:
    if lanes is None or (isinstance(lanes, float) and np.isnan(lanes)):
        return None
    m = re.search(r"\d+", str(lanes))
    if not m:
        return None
    v = int(m.group())
    return v if 0 < v <= 16 else None


def norm_oneway(oneway: object, highway: str | None, junction: object) -> tuple[bool, bool]:
    """Return (oneway, reverse). ``reverse`` means oneway=-1: travel against digitisation."""
    s = "" if oneway is None or (isinstance(oneway, float) and np.isnan(oneway)) else str(oneway).strip().lower()
    if s in {"-1", "reverse"}:
        return True, True
    if s in {"yes", "true", "1"}:
        return True, False
    if s in {"no", "false", "0", "reversible", "alternating"}:
        return False, False
    j = "" if junction is None or (isinstance(junction, float) and np.isnan(junction)) else str(junction).lower()
    if highway in {"motorway", "motorway_link"} or j in {"roundabout", "circular"}:
        return True, False
    return False, False


def _str_or_none(v: object) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s or None


# ----------------------------------------------------------------------------- splitting
@dataclass
class SplitResult:
    way_row: np.ndarray        # index into the input way arrays, per segment
    seg_idx: np.ndarray        # 0-based order of the segment along its way
    start: np.ndarray          # index into the flat vertex arrays
    end: np.ndarray            # inclusive
    start_node: np.ndarray
    end_node: np.ndarray
    junction_start: np.ndarray
    junction_end: np.ndarray


def split_ways(node_lists: Sequence[Sequence[int]]) -> tuple[SplitResult, np.ndarray, np.ndarray]:
    """Split ways at intersections.

    A *split node* is a node that occurs >= 2 times over all ways (shared by two ways, or
    visited twice by one way). Each way is cut at its interior split nodes. A *junction*
    is a node whose graph degree is >= 3 (endpoint occurrences count 1, interior 2), so a
    road that merely continues into another way (degree 2) is split but not a junction.

    Returns (SplitResult, flat_nodes, flat_way_row).
    """
    lengths = np.fromiter((len(n) for n in node_lists), dtype=np.int64, count=len(node_lists))
    if lengths.size == 0 or lengths.sum() == 0:
        empty = np.zeros(0, dtype=np.int64)
        eb = np.zeros(0, dtype=bool)
        return SplitResult(empty, empty, empty, empty, empty, empty, eb, eb), empty, empty
    flat = np.concatenate([np.asarray(n, dtype=np.int64) for n in node_lists if len(n)])
    way_row = np.repeat(np.arange(len(node_lists)), lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    pos = np.arange(flat.size) - np.repeat(offsets, lengths)
    last = np.repeat(lengths - 1, lengths)
    is_end = (pos == 0) | (pos == last)

    uniq, inv, occ = np.unique(flat, return_inverse=True, return_counts=True)
    deg = np.bincount(inv, weights=np.where(is_end, 1, 2), minlength=uniq.size)
    split_flag = occ[inv] >= 2
    junction_flag = deg[inv] >= 3

    cut = is_end | split_flag
    cuts = np.flatnonzero(cut)
    s, e = cuts[:-1], cuts[1:]
    ok = (way_row[s] == way_row[e]) & (e > s)
    s, e = s[ok], e[ok]
    seg_way = way_row[s]
    # order along way: rank within each way
    first_of_way = np.r_[True, seg_way[1:] != seg_way[:-1]]
    grp_start = np.maximum.accumulate(np.where(first_of_way, np.arange(seg_way.size), 0))
    seg_idx = np.arange(seg_way.size) - grp_start
    res = SplitResult(
        way_row=seg_way, seg_idx=seg_idx, start=s, end=e,
        start_node=flat[s], end_node=flat[e],
        junction_start=junction_flag[s], junction_end=junction_flag[e],
    )
    return res, flat, way_row


def _segment_vertex_index(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat vertex index list + per-vertex segment id for ranges [start, end] (inclusive)."""
    n = end - start + 1
    seg_id = np.repeat(np.arange(start.size), n)
    first = np.repeat(np.concatenate([[0], np.cumsum(n)[:-1]]), n)
    idx = np.repeat(start, n) + (np.arange(n.sum()) - first)
    return idx, seg_id


def bearings_deg(lon0, lat0, lon1, lat1) -> np.ndarray:
    az, _, _ = _GEOD.inv(np.asarray(lon0, float), np.asarray(lat0, float),
                         np.asarray(lon1, float), np.asarray(lat1, float))
    return np.mod(np.asarray(az), 360.0)


def build_segments(ways: pd.DataFrame, bbox: Sequence[float] | None = HYD_BBOX,
                   min_length_m: float = 0.5) -> pd.DataFrame:
    """Ways (with ``nodes`` + LineString ``geometry``) -> contract road segments.

    Columns: osm_way_id, seg_idx, name, name_te, highway, ref, lanes, oneway, lit, surface,
    length_m, bearing_deg, junction_start, junction_end, start_node, end_node, geometry.
    """
    ways = ways.reset_index(drop=True)
    geoms = ways.geometry.values
    ncoord = shapely.get_num_coordinates(geoms)
    nnodes = ways["nodes"].map(len).to_numpy()
    good = ncoord == nnodes
    if (~good).any():
        log.warning("dropping %d ways whose node list and geometry disagree", int((~good).sum()))
        ways = ways[good].reset_index(drop=True)
        geoms = ways.geometry.values

    oneway_rev = [norm_oneway(o, h, j) for o, h, j in zip(ways["oneway"], ways["highway"], ways["junction"], strict=True)]
    oneway = np.array([o for o, _ in oneway_rev], dtype=bool)
    reverse = np.array([r for _, r in oneway_rev], dtype=bool)

    node_lists = [list(n)[::-1] if r else list(n) for n, r in zip(ways["nodes"], reverse, strict=True)]
    coords_list = [shapely.get_coordinates(g)[::-1] if r else shapely.get_coordinates(g)
                   for g, r in zip(geoms, reverse, strict=True)]
    xy = np.concatenate(coords_list) if coords_list else np.zeros((0, 2))

    sp, _flat, _wr = split_ways(node_lists)
    vidx, seg_of_v = _segment_vertex_index(sp.start, sp.end)
    seg_xy = xy[vidx]
    lines = shapely.linestrings(seg_xy, indices=seg_of_v)

    # metric length
    tr = Transformer.from_crs(4326, METRIC_EPSG, always_xy=True)
    mx, my = tr.transform(seg_xy[:, 0], seg_xy[:, 1])
    d = np.hypot(np.diff(mx), np.diff(my))
    same = seg_of_v[1:] == seg_of_v[:-1]
    d = np.where(same, d, 0.0)
    length = np.bincount(seg_of_v[1:], weights=d, minlength=sp.start.size)

    # bearing start->end; fall back to first->second vertex for closed segments
    x0, y0 = xy[sp.start, 0], xy[sp.start, 1]
    x1, y1 = xy[sp.end, 0], xy[sp.end, 1]
    closed = (np.abs(x0 - x1) < 1e-9) & (np.abs(y0 - y1) < 1e-9)
    x1 = np.where(closed, xy[sp.start + 1, 0], x1)
    y1 = np.where(closed, xy[sp.start + 1, 1], y1)
    bearing = bearings_deg(x0, y0, x1, y1)

    w = ways.iloc[sp.way_row]
    hw = w["highway"].astype(str).to_numpy()
    seg = pd.DataFrame({
        "osm_way_id": w["id"].astype(np.int64).to_numpy(),
        "seg_idx": sp.seg_idx.astype(np.int64),
        "name": [_str_or_none(v) for v in w["name"]],
        "name_te": [_str_or_none(v) for v in w["name:te"]],
        "highway": hw,
        "ref": [_str_or_none(v) for v in w["ref"]],
        "lanes": [norm_lanes(v) for v in w["lanes"]],
        "oneway": oneway[sp.way_row],
        "lit": [norm_lit(v) for v in w["lit"]],
        "surface": [norm_surface(s, h) for s, h in zip(w["surface"], hw, strict=True)],
        "operator": [_str_or_none(v) for v in w["operator"]],
        "length_m": np.round(length, 2),
        "bearing_deg": np.mod(np.round(bearing, 1), 360.0),
        "junction_start": sp.junction_start,
        "junction_end": sp.junction_end,
        "start_node": sp.start_node,
        "end_node": sp.end_node,
        "geometry": lines,
    })
    keep = seg["length_m"].to_numpy() >= min_length_m
    if bbox is not None:
        keep &= shapely.intersects(lines, shapely.box(*bbox))
    return seg[keep].reset_index(drop=True)


# ----------------------------------------------------------------------------- POIs
POI_AMENITY_KIND = {
    "school": "school", "college": "school", "kindergarten": "school",
    "hospital": "hospital",
    "clinic": "clinic", "doctors": "clinic",
    "marketplace": "marketplace",
}


def metro_key(name: str) -> str:
    """Normalised station name for de-duplication ("Miyapur Metro Station" == "miyapur").

    Line suffixes are kept, so interchange stations ("Ameerpet (Red Line)" / "(Blue Line)")
    stay separate platforms."""
    s = re.sub(r"[^a-z0-9() ]", " ", name.lower())
    s = re.sub(r"\b(metro|station|stn)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_metro(row: pd.Series) -> bool:
    name = str(row.get("name") or "")
    if "(u/c)" in name.lower() or "under construction" in name.lower():
        return False
    txt = " ".join(str(row.get(k) or "") for k in ("network", "operator")).lower()
    station = str(row.get("station") or "").lower()
    subway = str(row.get("subway") or "").lower()
    return "metro" in txt or station in {"subway", "light_rail"} or subway == "yes"


def read_pois(pbf: str | Path, bbox: Sequence[float] | None = HYD_BBOX) -> pd.DataFrame:
    """POIs as points: kind in school|hospital|clinic|marketplace|metro_station; source=osm."""
    osm = _osm(pbf, bbox)
    rows = []
    am = osm.get_data_by_custom_criteria(custom_filter={"amenity": list(POI_AMENITY_KIND)},
                                         tags_as_columns=["name", "amenity"], keep_other_tags=False)
    if am is not None and len(am):
        pts = shapely.point_on_surface(am.geometry.values)
        for (_, r), p in zip(am.iterrows(), pts, strict=True):
            rows.append({"kind": POI_AMENITY_KIND[r["amenity"]], "name": _str_or_none(r.get("name")),
                         "source": "osm", "source_id": f"{r['osm_type'][0]}{int(r['id'])}", "geometry": p})
    metro_cols = ["name", "railway", "station", "public_transport", "subway", "network", "operator"]
    st = osm.get_data_by_custom_criteria(custom_filter={"railway": ["station", "halt"]},
                                         tags_as_columns=metro_cols, keep_other_tags=False)
    seen: dict[str, list[tuple[float, float]]] = {}
    if st is not None and len(st):
        pts = shapely.point_on_surface(st.geometry.values)
        for (_, r), p in zip(st.iterrows(), pts, strict=True):
            if not _is_metro(r):
                continue
            nm = _str_or_none(r.get("name")) or ""
            key = metro_key(nm)
            if key and any(abs(ox - p.x) < 0.005 and abs(oy - p.y) < 0.005 for ox, oy in seen.get(key, [])):
                continue  # same station (another node of it, ~500 m)
            seen.setdefault(key, []).append((p.x, p.y))
            rows.append({"kind": "metro_station", "name": nm or None, "source": "osm",
                         "source_id": f"{r['osm_type'][0]}{int(r['id'])}", "geometry": p})
    df = pd.DataFrame(rows, columns=["kind", "name", "source", "source_id", "geometry"])
    if bbox is not None and len(df):
        df = df[shapely.intersects(df.geometry.values, shapely.box(*bbox))].reset_index(drop=True)
    return df


def read_admin_boundaries(pbf: str | Path, bbox: Sequence[float] | None = HYD_BBOX) -> pd.DataFrame:
    """OSM administrative boundaries (name, name:te, admin_level, geometry)."""
    osm = _osm(pbf, bbox, complete_relations=True)
    b = osm.get_boundaries(boundary_type="administrative")
    if b is None:
        return pd.DataFrame(columns=["id", "name", "name:te", "admin_level", "geometry"])
    for c in ("name", "name:te", "admin_level"):
        if c not in b.columns:
            b[c] = None
    return pd.DataFrame(b[["id", "name", "name:te", "admin_level", "geometry"]])
