"""Exports: Measurement Book CSV, invoice-eligible work orders, potholes GeoJSON.

Only ``verified_closed`` work orders are invoice / Measurement-Book eligible (``states.INVOICE_ELIGIBLE``). Rework
work orders (``is_rework``: QC-fail or DLP rework at the contractor's cost) are never billable and are excluded.
The MB uses the plan each pothole was issued with (``work_order_pothole.repair_plan_id``; current plan as fallback)
and ``repair.mb_export.measurement_book_csv``. Quantities are design quantities from measurements with sigma;
demo calibrations are uncertified, so the CSV says so when any pothole's measurement is not certified.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from repair.mb_export import measurement_book_csv
from server.app.core.util import to_ist
from server.app.db import models as m
from server.app.schemas.states import INVOICE_ELIGIBLE
from server.app.services.runs import current_plan, plan_of

ELIGIBLE_STATUSES = tuple(s.value for s in INVOICE_ELIGIBLE)
GEOJSON_MAX_FEATURES = 50_000


def eligible_stmt(include_rework: bool = False) -> Select:
    stmt = select(m.WorkOrder).where(m.WorkOrder.status.in_(ELIGIBLE_STATUSES))
    if not include_rework:
        stmt = stmt.where(m.WorkOrder.is_rework.is_(False))
    return stmt.order_by(m.WorkOrder.id)


def is_eligible(wo: m.WorkOrder) -> bool:
    return wo.status in ELIGIBLE_STATUSES and not wo.is_rework


def mb_rows(db: Session, wos: Iterable[m.WorkOrder]) -> tuple[list[dict[str, Any]], bool]:
    """MB rows for the eligible work orders among ``wos`` (+ whether every measurement is certified)."""
    rows: list[dict[str, Any]] = []
    all_certified = True
    for wo in wos:
        if not is_eligible(wo):
            continue
        for wop in db.scalars(select(m.WorkOrderPothole).where(m.WorkOrderPothole.work_order_id == wo.id)
                              .order_by(m.WorkOrderPothole.pothole_id)):
            row = db.get(m.RepairPlanRow, wop.repair_plan_id) if wop.repair_plan_id else None
            plan = plan_of(row) or plan_of(current_plan(db, wop.pothole_id))
            if plan is None:
                continue
            ph = db.get(m.Pothole, wop.pothole_id)
            seg = db.get(m.RoadSegment, ph.road_segment_id) if ph is not None and ph.road_segment_id else None
            loc = []
            if seg is not None and seg.name:
                loc.append(seg.name)
            if ph is not None and ph.lat is not None:
                loc.append(f"{ph.lat:.6f},{ph.lon:.6f}")
            all_certified &= bool(ph is not None and ph.certified_measurement)
            when = to_ist(wo.updated_at)
            rows.append({"plan": plan, "work_order_id": wo.id, "pothole_id": wop.pothole_id,
                         "location": " ".join(loc) or None, "date": when.date().isoformat() if when else None})
    return rows, all_certified


def measurement_book(db: Session, wos: Iterable[m.WorkOrder]) -> str:
    rows, certified = mb_rows(db, wos)
    csv_text = measurement_book_csv(rows)
    if rows and not certified:
        csv_text += ("NOTE,Quantities derived from uncertified (demo) camera calibration - not billing-grade; "
                     "verify by site measurement before payment,,,,,,,,\n")
    return csv_text


def pothole_feature(p: m.Pothole, road_name: Optional[str] = None, highway: Optional[str] = None,
                    agency_code: Optional[str] = None, admin_unit: Optional[str] = None) -> dict[str, Any]:
    return {"type": "Feature", "id": p.id,
            "geometry": {"type": "Point", "coordinates": [round(p.lon, 7), round(p.lat, 7)]},
            "properties": {
                "id": p.id, "status": p.status, "severity": p.severity, "priority": p.priority,
                "depth_p95_mm": p.depth_p95_mm, "depth_sigma_mm": p.depth_sigma_mm, "area_m2": p.area_m2,
                "area_sigma": p.area_sigma, "volume_l": p.volume_l, "volume_sigma": p.volume_sigma,
                "water_filled": bool(p.water_filled), "method": p.best_method,
                "certified_measurement": bool(p.certified_measurement), "passes_seen": p.passes_seen,
                "passes_total": p.passes_total, "first_seen": p.first_seen.isoformat() if p.first_seen else None,
                "last_seen": p.last_seen.isoformat() if p.last_seen else None, "way_id": p.way_id,
                "direction": p.direction, "road_name": road_name, "highway": highway, "agency": agency_code,
                "admin_unit": admin_unit, "contract_id": p.contract_id, "contractor_id": p.contractor_id}}


def potholes_geojson(rows: Iterable[tuple[Any, ...]], truncated: bool = False) -> dict[str, Any]:
    """``rows``: (Pothole, road_name, highway, agency_code, admin_unit_name) tuples."""
    feats = [pothole_feature(*r) for r in rows if r[0].lat is not None and r[0].lon is not None]
    return {"type": "FeatureCollection", "features": feats, "truncated": truncated,
            "units": {"depth": "mm", "area": "m2", "volume": "litres"}}


__all__ = ["ELIGIBLE_STATUSES", "eligible_stmt", "is_eligible", "mb_rows", "measurement_book", "pothole_feature",
           "potholes_geojson"]
