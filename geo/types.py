"""Plain data types shared by the geo services (WP5). Pure python: no shapely, no DB, no server imports.

Units: metres, degrees (bearings clockwise from true north, 0-360), WGS84 lat/lon, timestamps UTC.
Coordinates inside ``SegmentRec.coords`` are ``(lon, lat)`` (GeoJSON / shapely order); every function
argument named ``lat``/``lon`` is in that explicit order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

TripId = Union[str, int]


@dataclass(slots=True)
class SegmentRec:
    """One ``road_segment`` row (an OSM way split at intersections, WP10)."""

    id: int  # road_segment.id
    way_id: int  # OSM way id
    coords: list[tuple[float, float]]  # [(lon, lat), ...] in digitised order, >= 2 points
    oneway: bool
    lanes: Optional[int]
    highway: str
    owner_agency_id: Optional[int]
    contract_id: Optional[int]
    admin_unit_id: Optional[int]


@dataclass(slots=True)
class MatchResult:
    """A point matched to a road segment (PLAN 5.2).

    ``direction``   0 = travelling along the digitised direction, 1 = against it.
    ``chainage_m``  metres from the segment's FIRST vertex (digitised order) to the matched point,
                    i.e. ST_LineLocatePoint x ST_Length, independent of ``direction``.
    ``dist_m``      distance from the query point to the segment.
    ``bearing_deg`` bearing of travel at the matched point (segment bearing, +180 when direction = 1).
    ``score``       matching cost, LOWER is better: dist/sigma + 0.5 * heading_diff/45deg.
    """

    road_segment_id: int
    way_id: int
    direction: int
    chainage_m: float
    dist_m: float
    bearing_deg: float
    score: float


@dataclass(slots=True)
class ObsPoint:
    """A verified pothole observation, already located (geo.locate) and matched (geo.mapmatch).

    ``lane_offset_m``: lateral offset of the pothole in metres, + right of travel. It must use the same
    reference for every trip (for example the road-frame x of the camera plus the bus's lane position).
    Clustering uses it to split lanes, and ``geo.cluster.lane_band`` uses it for the dedup key.
    """

    obs_id: int
    trip_id: TripId
    lat: float
    lon: float
    sigma_m: float
    way_id: Optional[int]
    direction: Optional[int]
    lane_offset_m: Optional[float]
    ts: Optional[datetime]


@dataclass(slots=True)
class PotholePoint:
    """An existing pothole entity as seen by clustering / pass counting."""

    pothole_id: int
    lat: float
    lon: float
    way_id: Optional[int]
    direction: Optional[int]
    lane_band: Optional[int]
    status: str


@dataclass(slots=True)
class FixPoint:
    """One ``trip_fix`` row (1 Hz GNSS + camera state) from the edge device."""

    trip_id: TripId
    ts: datetime
    lat: float
    lon: float
    heading_deg: Optional[float]
    speed_mps: Optional[float]
    accuracy_m: Optional[float]
    cam_active: bool


@dataclass(slots=True)
class Resolution:
    """Who owns / maintains the location (geo.jurisdiction.resolve). ``queued`` = needs a human decision."""

    agency_id: Optional[int]
    contract_id: Optional[int]
    admin_unit_id: Optional[int]
    queued: bool
    reason: str
