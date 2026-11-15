"""In-memory spatial index over road segments (DECISIONS D1: SQLite + shapely STRtree instead of PostGIS).

``RoadIndex`` keeps the caller's ``SegmentRec`` objects plus one shapely LineString per segment in a
single ``STRtree``. A query takes a lat/lon box of ``radius_m`` around the point from the tree, then
computes exact metric distances in a local tangent plane around the query point.

Build from the DB (WP6): ``RoadIndex(segment_from_wkt(r.id, r.osm_way_id, r.geom_wkt, ...) for r in rows)``.
Build from the WP10 file: ``RoadIndex(iter_segments_geojson(path, agency_ids=..., ...))``. The file is
streamed one feature at a time, so a 226 MB file is never held in memory as one parsed JSON tree.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import shapely
from shapely import STRtree

from geo.locate import bearing_of, m_per_deg, to_local
from geo.types import SegmentRec

DEFAULT_RADIUS_M = 25.0


@dataclass(slots=True)
class Candidate:
    """A segment near a query point, with the projection of the point onto it.

    ``seg`` is set by RoadIndex. The PostGIS adapter (geo.mapmatch_pg) builds candidates from SQL rows and
    sets only the scalar fields. ``start``/``end`` are the (lon, lat) end vertices, used for connectivity.
    ``offset_m`` is the signed lateral offset of the point (+ right of the digitised direction).
    """

    seg_id: int
    way_id: int
    oneway: bool
    dist_m: float
    bearing_deg: float  # segment bearing (digitised direction) at the nearest point
    chainage_m: float  # from the first vertex, digitised order
    length_m: float
    offset_m: float
    start: tuple[float, float]
    end: tuple[float, float]
    seg: Optional[SegmentRec] = None


def project_onto(coords: np.ndarray, lat: float, lon: float) -> tuple[float, float, float, float, float]:
    """Project a point onto a polyline given as an (n, 2) array of (lon, lat).

    Returns ``(dist_m, bearing_deg, chainage_m, length_m, offset_m)``. The bearing is that of the polyline
    piece containing the nearest point. The offset is + when the point is right of the digitised direction.
    """
    ex, ny = to_local(lat, lon, coords[:, 1], coords[:, 0])  # origin = query point
    ax, ay, bx, by = ex[:-1], ny[:-1], ex[1:], ny[1:]
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(seg_len2 > 0, -(ax * dx + ay * dy) / seg_len2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx, qy = ax + t * dx, ay + t * dy
    d = np.hypot(qx, qy)
    k = int(np.argmin(d))
    piece = np.sqrt(seg_len2)
    cum = np.concatenate(([0.0], np.cumsum(piece)))
    # pick a non-degenerate piece for the bearing (zero-length pieces happen with duplicate vertices)
    kb = k
    if piece[kb] == 0:
        nz = np.nonzero(piece)[0]
        kb = int(nz[np.argmin(np.abs(nz - k))]) if len(nz) else k
    bearing = bearing_of(float(dx[kb]), float(dy[kb]))
    # point (origin) relative to the piece start, crossed with the piece direction: + = right of travel
    cross = float(dx[kb] * (0.0 - ay[kb]) - dy[kb] * (0.0 - ax[kb]))
    off = float(d[k]) * (1.0 if cross < 0 else -1.0)
    return float(d[k]), bearing, float(cum[k] + t[k] * piece[k]), float(cum[-1]), off


class RoadIndex:
    """STRtree over road segments with metric nearest-segment queries.

    Memory (measured on the full 454k-segment Hyderabad network, see the WP5 notes in the result): the index
    adds the LineStrings + tree on top of the caller's ``SegmentRec`` list.
    """

    def __init__(self, segments: Iterable[SegmentRec]):
        self.segments: list[SegmentRec] = [s for s in segments if s.coords is not None and len(s.coords) >= 2]
        n = len(self.segments)
        if n:
            counts = np.fromiter((len(s.coords) for s in self.segments), dtype=np.int64, count=n)
            flat = np.empty((int(counts.sum()), 2), dtype=np.float64)
            i = 0
            for s, c in zip(self.segments, counts, strict=True):
                flat[i:i + c] = s.coords
                i += c
            self._offsets = np.concatenate(([0], np.cumsum(counts)))
            self._coords = flat
            self._geoms = shapely.linestrings(flat, indices=np.repeat(np.arange(n), counts))
        else:
            self._offsets = np.zeros(1, dtype=np.int64)
            self._coords = np.zeros((0, 2))
            self._geoms = np.empty(0, dtype=object)
        self._tree = STRtree(self._geoms)
        self._by_id: dict[int, int] = {s.id: i for i, s in enumerate(self.segments)}

    def __len__(self) -> int:
        return len(self.segments)

    def get(self, segment_id: int) -> Optional[SegmentRec]:
        i = self._by_id.get(segment_id)
        return None if i is None else self.segments[i]

    def coords_of(self, i: int) -> np.ndarray:
        return self._coords[self._offsets[i]:self._offsets[i + 1]]

    def _query_box(self, lat: float, lon: float, radius_m: float) -> np.ndarray:
        mlat, mlon = m_per_deg(lat)
        dlat, dlon = radius_m / mlat, radius_m / max(mlon, 1e-9)
        return self._tree.query(shapely.box(lon - dlon, lat - dlat, lon + dlon, lat + dlat))

    def candidates(self, lat: float, lon: float, radius_m: float = DEFAULT_RADIUS_M) -> list[Candidate]:
        """Every segment within ``radius_m`` of the point, nearest first, with the point's projection."""
        out: list[Candidate] = []
        for i in self._query_box(lat, lon, radius_m):
            i = int(i)
            c = self.coords_of(i)
            d, brg, ch, length, off = project_onto(c, lat, lon)
            if d <= radius_m:
                s = self.segments[i]
                out.append(Candidate(s.id, s.way_id, bool(s.oneway), d, brg, ch, length, off,
                                     (float(c[0, 0]), float(c[0, 1])), (float(c[-1, 0]), float(c[-1, 1])), s))
        out.sort(key=lambda k: (k.dist_m, k.seg_id))
        return out

    def nearest(self, lat: float, lon: float, radius_m: float = DEFAULT_RADIUS_M
                ) -> list[tuple[SegmentRec, float, float]]:
        """``[(segment, dist_m, bearing_deg_at_nearest), ...]`` within ``radius_m``, nearest first.

        The bearing is that of the segment's digitised direction at the nearest point.
        """
        return [(c.seg, c.dist_m, c.bearing_deg) for c in self.candidates(lat, lon, radius_m)]  # type: ignore[misc]

    def memory_bytes(self) -> int:
        """Approximate size of the numpy coordinate buffers (shapely/GEOS memory is not included)."""
        return int(self._coords.nbytes + self._offsets.nbytes)


# ----------------------------------------------------------------------------- builders
def coords_from_wkt(wkt: str) -> list[tuple[float, float]]:
    """(lon, lat) vertices of a WKT LineString (``road_segment.geom_wkt``)."""
    g = shapely.from_wkt(wkt)
    if g is None or g.is_empty:
        return []
    if g.geom_type == "MultiLineString":  # defensive: take the longest part
        g = max(g.geoms, key=lambda p: p.length)
    return [(float(x), float(y)) for x, y in shapely.get_coordinates(g)]


def segment_from_wkt(id: int, way_id: int, wkt: str, *, oneway: bool = False, lanes: Optional[int] = None,
                     highway: str = "unclassified", owner_agency_id: Optional[int] = None,
                     contract_id: Optional[int] = None, admin_unit_id: Optional[int] = None) -> SegmentRec:
    """``SegmentRec`` from DB columns (``road_segment.geom_wkt`` etc.)."""
    return SegmentRec(int(id), int(way_id), coords_from_wkt(wkt), bool(oneway), lanes, highway or "unclassified",
                      owner_agency_id, contract_id, admin_unit_id)


def iter_geojson_features(path: str | Path, chunk_chars: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Stream the features of a GeoJSON FeatureCollection without parsing the whole file at once.

    Works for both the one-feature-per-line WP10 output and single-line files. The file is only parsed
    as JSON (it is never executed).
    """
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8") as fh:
        buf = ""
        # find the start of the features array
        while True:
            chunk = fh.read(chunk_chars)
            if not chunk:
                return
            buf += chunk
            k = buf.find('"features"')
            if k >= 0:
                b = buf.find("[", k)
                if b >= 0:
                    buf = buf[b + 1:]
                    break
        pos = 0
        eof = False
        while True:
            # skip whitespace and commas
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf) or eof:
                    break
                chunk = fh.read(chunk_chars)
                eof = not chunk
                buf, pos = buf[pos:] + chunk, 0
            if pos >= len(buf) or buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = fh.read(chunk_chars)
                eof = not chunk
                buf, pos = buf[pos:] + chunk, 0
                continue
            yield obj
            pos = end
            if pos > chunk_chars:  # drop consumed text
                buf, pos = buf[pos:], 0


def _int_or_none(v: Any) -> Optional[int]:
    try:
        return None if v in (None, "") else int(v)
    except (TypeError, ValueError):
        return None


def iter_segments_geojson(path: str | Path, *, agency_ids: Optional[Mapping[str, int]] = None,
                          contract_ids: Optional[Mapping[str, int]] = None,
                          admin_unit_ids: Optional[Mapping[str, int]] = None,
                          first_id: int = 1) -> Iterator[SegmentRec]:
    """``SegmentRec`` per feature of WP10's ``road_segments.geojson``.

    The file has no DB ids, so ``id`` = ``first_id`` + feature index (the seed inserts in file order).
    Owner / contract / admin unit come as codes. Each is mapped to an id through the optional dicts,
    otherwise it is None.
    """
    for n, f in enumerate(iter_geojson_features(path)):
        p = f.get("properties") or {}
        g = f.get("geometry") or {}
        if g.get("type") != "LineString" or len(g.get("coordinates") or []) < 2:
            continue
        coords = [(float(c[0]), float(c[1])) for c in g["coordinates"]]
        owner = p.get("owner_agency")
        contract = p.get("contract_code")
        unit = p.get("admin_unit_code")
        yield SegmentRec(
            id=first_id + n,
            way_id=int(p.get("osm_way_id") or 0),
            coords=coords,
            oneway=bool(p.get("oneway", False)),
            lanes=_int_or_none(p.get("lanes")),
            highway=str(p.get("highway") or "unclassified"),
            owner_agency_id=agency_ids.get(owner) if agency_ids and owner else None,
            contract_id=contract_ids.get(contract) if contract_ids and contract else None,
            admin_unit_id=admin_unit_ids.get(unit) if admin_unit_ids and unit else None,
        )


def endpoints_connected(a_start, a_end, b_start, b_end, tol_m: float = 2.0) -> bool:
    """True if two polylines share an end vertex (within ``tol_m``). Inputs are (lon, lat) tuples."""
    for p in (a_start, a_end):
        for q in (b_start, b_end):
            mlat, mlon = m_per_deg(p[1])
            if math.hypot((p[0] - q[0]) * mlon, (p[1] - q[1]) * mlat) <= tol_m:
                return True
    return False
