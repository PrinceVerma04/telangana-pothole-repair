"""Who must fix this pothole: owner agency, maintenance contract, admin unit (PLAN 5 / WP5).

``resolve(segment, lat, lon, admin_units, contracts)`` follows this order:

1. **Owner agency** = ``segment.owner_agency_id`` (WP10 assigns it from OSM ref/highway rules and the
   override layer). If no segment matched, or the segment has no owner, the owner cannot be determined,
   so the result is ``queued=True`` for the jurisdiction queue (``/admin/jurisdiction-queue``). The admin
   unit is still filled in, so the queue item routes to the right zonal admin. With
   ``fallback_admin_agency=True`` the agency of the most specific admin unit is used instead and the
   result is not queued (this is a policy switch; the default is the conservative queue).
2. **Admin unit** = ``segment.admin_unit_id`` if set. Otherwise it is the most specific (smallest-area)
   admin-unit polygon covering the point. Units of the owner agency are preferred.
3. **Contract** = ``segment.contract_id`` if set. Otherwise it is the smallest contract polygon of the
   owner agency covering the point. An unowned or queued location gets no contract.

``admin_units`` and ``contracts`` are sequences of ``(id, agency_id, shapely_geometry)`` built by the caller
(WGS84 lon/lat). Geometries are prepared in place for repeated calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import shapely
from shapely.geometry import Point

from geo.types import Resolution, SegmentRec

Unit = tuple[int, Optional[int], Any]  # (id, agency_id, shapely geometry)

__all__ = ["Resolution", "resolve"]


def _covering(units: Sequence[Unit], pt: Point, agency_id: Optional[int] = None) -> Optional[Unit]:
    hits = []
    for u in units:
        g = u[2]
        if g is None or g.is_empty:
            continue
        shapely.prepare(g)
        if g.covers(pt):
            hits.append(u)
    if agency_id is not None:
        own = [u for u in hits if u[1] == agency_id]
        hits = own or hits
    if not hits:
        return None
    return min(hits, key=lambda u: (u[2].area, u[0]))


def resolve(segment: Optional[SegmentRec], lat: float, lon: float, admin_units: Sequence[Unit],
            contracts: Sequence[Unit], *, fallback_admin_agency: bool = False) -> Resolution:
    pt = Point(float(lon), float(lat))
    owner = segment.owner_agency_id if segment is not None else None

    unit_id: Optional[int] = segment.admin_unit_id if segment is not None else None
    unit_agency: Optional[int] = None
    if unit_id is None:
        u = _covering(admin_units, pt, owner)
        if u is not None:
            unit_id, unit_agency = u[0], u[1]
    else:
        unit_agency = next((u[1] for u in admin_units if u[0] == unit_id), None)

    if owner is None:
        why = "no matched road segment" if segment is None else f"road segment {segment.id} has no owner agency"
        if fallback_admin_agency and unit_agency is not None:
            return Resolution(unit_agency, None, unit_id, False, f"{why}; admin unit {unit_id} agency fallback")
        where = f"; admin unit {unit_id}" if unit_id is not None else "; outside every admin unit"
        return Resolution(None, None, unit_id, True, why + where)

    contract_id: Optional[int] = segment.contract_id if segment is not None else None
    how = "segment contract"
    if contract_id is None:
        c = _covering([c for c in contracts if c[1] == owner], pt)
        contract_id = c[0] if c is not None else None
        how = "contract polygon" if c is not None else "no contract (departmental)"
    reason = f"segment {segment.id} owner agency {owner}; {how}" if segment is not None else ""
    return Resolution(owner, contract_id, unit_id, False, reason)
