"""TGSRTC GTFS -> bus-stop POIs, route shapes, trips/day per route and per road segment.

The feed is untrusted input: it is read with ``zipfile`` + ``pandas.read_csv`` only.

The TGSRTC feed (Open Data Telangana, 2026-02) has no ``shapes.txt``. Shapes are then
built from each trip pattern's stop sequence; when a :class:`RoadRouter` is given, every
pair of consecutive stops is joined by the shortest drivable path over the OSM segment
graph (so shapes follow real roads instead of cutting corners between stops).

``assign_trips_to_segments`` gives each road segment the number of scheduled bus trips per
day whose shape runs along it: shapes are cut into <= 400 m pieces, buffered (~15 m), the
pieces are matched to segments with one vectorised ``STRtree.query`` and a segment counts
for a pattern when >= 90 % of it (or >= 60 m) lies inside the pattern's buffer (a high
fraction so that short side-street stubs touching the corridor at a junction do not count).
"""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

METRIC_EPSG = 32644
_TO_M = Transformer.from_crs(4326, METRIC_EPSG, always_xy=True)
_TO_LL = Transformer.from_crs(METRIC_EPSG, 4326, always_xy=True)

# Routing cost multipliers: buses keep to main roads unless a stop forces otherwise.
CLASS_COST = {
    "motorway": 1.0, "trunk": 1.0, "primary": 1.0, "secondary": 1.0, "tertiary": 1.1,
    "motorway_link": 1.0, "trunk_link": 1.0, "primary_link": 1.0, "secondary_link": 1.0, "tertiary_link": 1.1,
    "unclassified": 1.3, "residential": 1.6, "living_street": 3.0, "service": 3.0,
}
MAJOR_HIGHWAYS = ("motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link",
                  "primary_link", "secondary_link", "tertiary_link")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


# ----------------------------------------------------------------------------- loading
@dataclass
class Feed:
    stops: pd.DataFrame
    routes: pd.DataFrame
    trips: pd.DataFrame
    stop_times: pd.DataFrame
    calendar: pd.DataFrame | None = None
    shapes: pd.DataFrame | None = None
    agency: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)


def _read(z: zipfile.ZipFile, name: str, usecols: Sequence[str] | None = None) -> pd.DataFrame | None:
    names = {Path(n).name: n for n in z.namelist()}
    if name not in names:
        return None
    raw = z.read(names[name])
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if usecols is not None:
        df = df[[c for c in usecols if c in df.columns]]
    return df


def load_feed(path: str | Path) -> Feed:
    """Load a GTFS zip (only the tables we need)."""
    with zipfile.ZipFile(path) as z:
        stops = _read(z, "stops.txt")
        routes = _read(z, "routes.txt")
        trips = _read(z, "trips.txt")
        st = _read(z, "stop_times.txt", ["trip_id", "stop_sequence", "stop_id"])
        cal = _read(z, "calendar.txt")
        shapes = _read(z, "shapes.txt")
        agency = _read(z, "agency.txt")
    if stops is None or routes is None or trips is None or st is None:
        raise ValueError(f"{path}: not a GTFS feed (stops/routes/trips/stop_times required)")
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"]).reset_index(drop=True)
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.dropna(subset=["stop_sequence"])
    if shapes is not None and len(shapes):
        for c in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
            shapes[c] = pd.to_numeric(shapes[c], errors="coerce")
    else:
        shapes = None
    return Feed(stops=stops, routes=routes, trips=trips, stop_times=st, calendar=cal, shapes=shapes, agency=agency)


# ----------------------------------------------------------------------------- service / trips per day
def active_service_ids(feed: Feed, weekday: str = "wednesday") -> set[str] | None:
    """Service ids running on ``weekday`` per calendar.txt (None = no calendar: count all)."""
    cal = feed.calendar
    if cal is None or len(cal) == 0 or weekday not in cal.columns:
        return None
    return set(cal.loc[cal[weekday].astype(str).str.strip() == "1", "service_id"])


def daily_trips(feed: Feed, weekday: str = "wednesday") -> pd.DataFrame:
    """Trips that run on a typical ``weekday``."""
    sids = active_service_ids(feed, weekday)
    t = feed.trips
    if sids is not None and "service_id" in t.columns:
        t = t[t["service_id"].isin(sids)]
    return t.reset_index(drop=True)


def trips_per_day_by_route(feed: Feed, weekday: str = "wednesday") -> pd.Series:
    return daily_trips(feed, weekday).groupby("route_id").size().astype(float)


# ----------------------------------------------------------------------------- stops -> POIs
def bus_stop_pois(feed: Feed, bbox: Sequence[float] | None = None) -> pd.DataFrame:
    s = feed.stops
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        s = s[(s.stop_lon >= x0) & (s.stop_lon <= x1) & (s.stop_lat >= y0) & (s.stop_lat <= y1)]
    return pd.DataFrame({
        "kind": "bus_stop",
        "name": s["stop_name"].astype(str).str.strip().replace("", None).to_numpy(),
        "source": "gtfs",
        "source_id": s["stop_id"].to_numpy(),
        "geometry": shapely.points(s.stop_lon.to_numpy(), s.stop_lat.to_numpy()),
    })


# ----------------------------------------------------------------------------- patterns
def trip_patterns(feed: Feed, weekday: str = "wednesday") -> pd.DataFrame:
    """Distinct stop sequences per (route, direction) with their trips/day.

    Columns: route_id, direction_id, stops (tuple of stop_id), trips, label (most common
    trip_headsign / trip_short_name), shape_id (if the trips carry one).
    """
    trips = daily_trips(feed, weekday)
    st = feed.stop_times[feed.stop_times["trip_id"].isin(trips["trip_id"])]
    st = st.sort_values(["trip_id", "stop_sequence"], kind="stable")
    seq = st.groupby("trip_id", sort=False)["stop_id"].agg(tuple).rename("stops")
    t = trips.merge(seq, left_on="trip_id", right_index=True, how="inner")
    if "direction_id" not in t.columns:
        t["direction_id"] = "0"
    t["direction_id"] = t["direction_id"].replace("", "0")
    label_col = next((c for c in ("trip_headsign", "trip_short_name") if c in t.columns), None)
    t["label"] = t[label_col] if label_col else ""
    if "shape_id" not in t.columns:
        t["shape_id"] = ""
    g = t.groupby(["route_id", "direction_id", "stops"], sort=False)
    pat = g.agg(trips=("trip_id", "size"),
                label=("label", lambda s: s.value_counts().index[0] if len(s) else ""),
                shape_id=("shape_id", lambda s: s.value_counts().index[0] if len(s) else "")).reset_index()
    return pat.sort_values(["route_id", "direction_id", "trips"], ascending=[True, True, False]).reset_index(drop=True)


# ----------------------------------------------------------------------------- road routing
class RoadRouter:
    """Shortest drivable paths over road segments (directed, honours ``oneway``).

    ``segments`` needs columns start_node, end_node, length_m, oneway, highway and WGS84
    LineString ``geometry`` digitised in the direction of travel for one-way segments.
    """

    def __init__(self, segments: pd.DataFrame, max_snap_m: float = 250.0):
        self.max_snap_m = max_snap_m
        geoms = segments.geometry.values
        self.coords = [shapely.get_coordinates(g) for g in geoms]
        nodes = np.unique(np.concatenate([segments["start_node"].to_numpy(), segments["end_node"].to_numpy()]))
        self.node_ids = nodes
        u = np.searchsorted(nodes, segments["start_node"].to_numpy())
        v = np.searchsorted(nodes, segments["end_node"].to_numpy())
        cost = segments["length_m"].to_numpy(float) * segments["highway"].map(CLASS_COST).fillna(2.0).to_numpy()
        cost = np.maximum(cost, 0.01)
        seg = np.arange(len(segments))
        one = segments["oneway"].to_numpy(bool)
        uu = np.concatenate([u, v[~one]])
        vv = np.concatenate([v, u[~one]])
        cc = np.concatenate([cost, cost[~one]])
        ss = np.concatenate([seg, seg[~one]])
        fwd = np.concatenate([np.ones(len(u), bool), np.zeros(int((~one).sum()), bool)])
        keep = uu != vv
        uu, vv, cc, ss, fwd = uu[keep], vv[keep], cc[keep], ss[keep], fwd[keep]
        order = np.lexsort((cc, vv, uu))  # keep the cheapest of parallel edges
        uu, vv, cc, ss, fwd = uu[order], vv[order], cc[order], ss[order], fwd[order]
        first = np.r_[True, (uu[1:] != uu[:-1]) | (vv[1:] != vv[:-1])]
        uu, vv, cc, ss, fwd = uu[first], vv[first], cc[first], ss[first], fwd[first]
        n = nodes.size
        self.graph = sparse.csr_matrix((cc, (uu, vv)), shape=(n, n))
        self.edge = {(int(a), int(b)): (int(s), bool(f)) for a, b, s, f in zip(uu, vv, ss, fwd, strict=True)}
        # node coordinates (metric) for snapping
        xy = np.zeros((n, 2))
        xy[u] = [c[0] for c in self.coords]
        xy[v] = [c[-1] for c in self.coords]
        mx, my = _TO_M.transform(xy[:, 0], xy[:, 1])
        self.node_xy_m = np.column_stack([mx, my])
        self.node_xy = xy
        self.tree = cKDTree(self.node_xy_m)
        # nodes on bus-able roads (tertiary and above): stops snap there first, so routed
        # shapes do not make in-and-out spurs into colony lanes next to a stop
        major = segments["highway"].isin(MAJOR_HIGHWAYS).to_numpy()
        self.major_nodes = np.unique(np.concatenate([u[major], v[major]]))
        self.major_tree = cKDTree(self.node_xy_m[self.major_nodes]) if self.major_nodes.size else None

    def snap(self, lon: np.ndarray, lat: np.ndarray, major_m: float = 120.0) -> np.ndarray:
        """Graph node per point: nearest node on a major road within ``major_m``, else the
        nearest node within ``max_snap_m``; -1 if none."""
        pts = np.column_stack(_TO_M.transform(np.asarray(lon, float), np.asarray(lat, float)))
        d, i = self.tree.query(pts)
        out = np.where(d <= self.max_snap_m, i, -1)
        if self.major_tree is not None:
            dm, im = self.major_tree.query(pts)
            out = np.where(dm <= major_m, self.major_nodes[im], out)
        return out

    def paths_from(self, src: int, targets: Sequence[int], limit_m: float) -> dict[int, list[int] | None]:
        dist, pred = dijkstra(self.graph, directed=True, indices=src, limit=limit_m, return_predecessors=True)
        out: dict[int, list[int] | None] = {}
        for t in targets:
            if t == src:
                out[t] = [src]
                continue
            if not np.isfinite(dist[t]):
                out[t] = None
                continue
            path = [t]
            while path[-1] != src:
                p = pred[path[-1]]
                if p < 0:
                    path = None
                    break
                path.append(int(p))
            out[t] = path[::-1] if path else None
        return out

    def path_coords(self, path: Sequence[int]) -> np.ndarray:
        parts = []
        for a, b in zip(path[:-1], path[1:], strict=True):
            s, fwd = self.edge[(a, b)]
            c = self.coords[s] if fwd else self.coords[s][::-1]
            parts.append(c if not parts else c[1:])
        return np.concatenate(parts) if parts else np.zeros((0, 2))


def _clean_line(xy: np.ndarray) -> np.ndarray:
    if len(xy) < 2:
        return xy
    keep = np.r_[True, np.any(np.abs(np.diff(xy, axis=0)) > 1e-9, axis=1)]
    return xy[keep]


def build_pattern_shapes(feed: Feed, patterns: pd.DataFrame, router: RoadRouter | None = None,
                         bbox: Sequence[float] | None = None, detour_factor: float = 4.0,
                         min_limit_m: float = 3000.0) -> list[np.ndarray | None]:
    """One WGS84 polyline (N x 2, lon/lat) per pattern row (None if < 2 usable stops).

    Uses ``shapes.txt`` when the pattern has a shape_id, otherwise the stop sequence,
    routed over roads when ``router`` is given (straight line fallback per hop).
    """
    stop_xy = feed.stops.set_index("stop_id")[["stop_lon", "stop_lat"]]
    shp = None
    if feed.shapes is not None:
        shp = {k: g.sort_values("shape_pt_sequence")[["shape_pt_lon", "shape_pt_lat"]].to_numpy()
               for k, g in feed.shapes.groupby("shape_id")}

    seqs: list[np.ndarray] = []
    for stops in patterns["stops"]:
        ids = [s for s in stops if s in stop_xy.index]
        xy = stop_xy.loc[ids].to_numpy(float) if ids else np.zeros((0, 2))
        if bbox is not None and len(xy):
            x0, y0, x1, y1 = bbox
            inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
            xy = xy[inside]  # stops outside the study area are dropped
        seqs.append(xy)

    shapes: list[np.ndarray | None] = [None] * len(patterns)
    if router is None:
        for i, (xy, sid) in enumerate(zip(seqs, patterns.get("shape_id", [""] * len(patterns)), strict=True)):
            if shp is not None and sid in shp:
                shapes[i] = shp[sid]
            elif len(xy) >= 2:
                shapes[i] = _clean_line(xy)
        return shapes

    # snap all stops, gather hops, route grouped by source node
    node_seqs = [router.snap(xy[:, 0], xy[:, 1]) if len(xy) else np.zeros(0, int) for xy in seqs]
    hops: dict[int, dict[int, float]] = {}
    for ns in node_seqs:
        for k in range(len(ns) - 1):
            a, b = int(ns[k]), int(ns[k + 1])
            if a < 0 or b < 0 or a == b:
                continue
            ax, ay = router.node_xy_m[a]
            bx, by = router.node_xy_m[b]
            lim = max(min_limit_m, detour_factor * float(np.hypot(ax - bx, ay - by)))
            d = hops.setdefault(a, {})
            d[b] = max(d.get(b, 0.0), lim)
    routed: dict[tuple[int, int], list[int] | None] = {}
    for a, tg in hops.items():
        res = router.paths_from(a, list(tg), max(tg.values()))
        for b, p in res.items():
            routed[(a, b)] = p
    n_fail = sum(p is None for p in routed.values())
    log.info("routed %d stop-to-stop hops (%d fell back to straight lines)", len(routed), n_fail)

    for i, (xy, ns, sid) in enumerate(zip(seqs, node_seqs, patterns.get("shape_id", [""] * len(patterns)), strict=True)):
        if shp is not None and sid in shp:
            shapes[i] = shp[sid]
            continue
        if len(xy) < 2:
            continue
        parts = [xy[:1]]
        for k in range(len(xy) - 1):
            a, b = int(ns[k]), int(ns[k + 1])
            p = routed.get((a, b)) if a >= 0 and b >= 0 and a != b else None
            if p is not None and len(p) >= 2:
                parts.append(router.path_coords(p))
            parts.append(xy[k + 1:k + 2])
        shapes[i] = _clean_line(np.concatenate(parts))
    return shapes


# ----------------------------------------------------------------------------- segment assignment
def _to_metric_lines(lines_ll: Sequence[np.ndarray]) -> list[np.ndarray]:
    out = []
    for xy in lines_ll:
        mx, my = _TO_M.transform(xy[:, 0], xy[:, 1])
        out.append(np.column_stack([mx, my]))
    return out


def _chunk_line(xy: np.ndarray, max_len: float) -> list[np.ndarray]:
    """Cut a metric polyline into consecutive pieces of about ``max_len`` metres.

    The line is densified to <= max_len/8 vertex spacing, then vertices are grouped by
    cumulative distance; neighbouring pieces share their boundary vertex.
    """
    line = shapely.linestrings(xy)
    if line.length <= max_len:
        return [xy]
    d = shapely.get_coordinates(shapely.segmentize(line, max_len / 8.0))
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(d, axis=0).T))]
    grp = np.minimum((cum // max_len).astype(int), int(cum[-1] // max_len))
    starts = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])
    ends = np.r_[starts[1:], len(d) - 1]
    return [d[a:b + 1] for a, b in zip(starts, ends, strict=True) if b > a]


def assign_trips_to_segments(seg_geoms_ll: np.ndarray, shapes_ll: Sequence[np.ndarray | None],
                             trips: Sequence[float], buffer_m: float = 15.0, piece_m: float = 400.0,
                             min_frac: float = 0.9, min_overlap_m: float = 60.0) -> np.ndarray:
    """Sum of trips/day of all shapes whose ``buffer_m`` corridor covers each segment.

    A segment is covered by a shape when the length of the segment inside the shape's
    buffer is >= ``min_frac`` of the segment or >= ``min_overlap_m``.
    Fully vectorised: one STRtree over segments, one bulk query with all buffered pieces.
    """
    nseg = len(seg_geoms_ll)
    out = np.zeros(nseg)
    # identical shapes (patterns that differ only in stops served) are processed once
    uniq: dict[bytes, int] = {}
    shp_u: list[np.ndarray] = []
    trips_u: list[float] = []
    for s, t in zip(shapes_ll, trips, strict=True):
        if s is None or len(s) < 2:
            continue
        k = np.ascontiguousarray(np.round(s, 7)).tobytes()
        if k in uniq:
            trips_u[uniq[k]] += float(t)
        else:
            uniq[k] = len(shp_u)
            shp_u.append(s)
            trips_u.append(float(t))
    shapes_ll, trips = shp_u, trips_u
    valid = list(range(len(shapes_ll)))
    if not valid or nseg == 0:
        return out
    seg_xy = shapely.get_coordinates(seg_geoms_ll)
    counts = shapely.get_num_coordinates(seg_geoms_ll)
    mx, my = _TO_M.transform(seg_xy[:, 0], seg_xy[:, 1])
    seg_m = shapely.linestrings(np.column_stack([mx, my]), indices=np.repeat(np.arange(nseg), counts))
    seg_len = shapely.length(seg_m)

    piece_geoms, piece_pat = [], []
    for i, xy in zip(valid, _to_metric_lines([shapes_ll[i] for i in valid]), strict=True):
        for p in _chunk_line(xy, piece_m):
            if len(p) >= 2:
                piece_geoms.append(shapely.linestrings(p))
                piece_pat.append(i)
    bufs = shapely.buffer(np.array(piece_geoms, dtype=object), buffer_m, quad_segs=4)
    piece_pat = np.asarray(piece_pat)

    tree = shapely.STRtree(seg_m)
    pi, si = tree.query(bufs, predicate="intersects")
    if pi.size == 0:
        return out
    inter = shapely.length(shapely.intersection(seg_m[si], bufs[pi]))
    pat = piece_pat[pi]
    # sum overlaps of all pieces of one pattern on one segment (pieces share only a cap)
    key = pd.DataFrame({"pat": pat, "seg": si, "ov": inter}).groupby(["pat", "seg"], sort=False)["ov"].sum()
    p_idx = key.index.get_level_values(0).to_numpy()
    s_idx = key.index.get_level_values(1).to_numpy()
    ov = np.minimum(key.to_numpy(), seg_len[s_idx])
    covered = (ov >= min_frac * seg_len[s_idx]) | (ov >= min_overlap_m)
    np.add.at(out, s_idx[covered], np.asarray(trips, float)[p_idx[covered]])
    return out


# ----------------------------------------------------------------------------- route records
def route_records(feed: Feed, patterns: pd.DataFrame, shapes: Sequence[np.ndarray | None],
                  simplify_m: float = 3.0, decimals: int = 6) -> list[dict]:
    """Contract ``gtfs_routes.json`` rows: one representative shape per route.

    The shape is the busiest pattern of the busiest direction; ``trips_per_day`` counts all
    trips of the route (both directions, all patterns) on a typical weekday.
    """
    r = feed.routes.set_index("route_id")
    pat = patterns.assign(_shape=list(shapes))
    pat = pat[pat["_shape"].map(lambda s: s is not None and len(s) >= 2)]
    tpd = patterns.groupby("route_id")["trips"].sum()
    out = []
    for rid, g in pat.groupby("route_id", sort=True):
        dir_trips = g.groupby("direction_id")["trips"].sum().sort_values(ascending=False)
        gd = g[g["direction_id"] == dir_trips.index[0]].sort_values("trips", ascending=False)
        best = gd.iloc[0]
        xy = best["_shape"]
        if simplify_m > 0 and len(xy) > 2:
            mx, my = _TO_M.transform(xy[:, 0], xy[:, 1])
            simp = shapely.get_coordinates(shapely.simplify(shapely.linestrings(np.column_stack([mx, my])), simplify_m))
            lon, lat = _TO_LL.transform(simp[:, 0], simp[:, 1])
            xy = np.column_stack([lon, lat])
        row = r.loc[rid] if rid in r.index else pd.Series(dtype=str)
        short = str(row.get("route_short_name", "") or "").strip() or str(rid)
        long_ = str(row.get("route_long_name", "") or "").strip()
        if not long_ or long_ == short:
            long_ = str(best["label"] or short)
        out.append({
            "route_id": str(rid),
            "short_name": short,
            "long_name": long_,
            "shape": np.round(xy, decimals).tolist(),
            "trips_per_day": float(tpd.get(rid, 0.0)),
        })
    return out
