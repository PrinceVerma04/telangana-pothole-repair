"""/v1/exports: Measurement Book CSV, invoice-eligible work orders, potholes GeoJSON (WP8b).

Only ``verified_closed`` (non-rework) work orders are invoice / MB eligible; asking for any other work order id
in the MB is silently ignored and reported in the ``X-Excluded-Work-Orders`` header.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy import select

from measure.types import PotholeStatus
from server.app.core.rbac import PublicPrincipalDep, RoleDep, admin_subtree_ids, scope_query
from server.app.core.util import now_utc, parse_bbox, to_ist
from server.app.db import models as m
from server.app.deps import DbDep, StorageDep
from server.app.schemas.api import WorkOrderOut
from server.app.schemas.states import Role
from server.app.services import exports as svc
from server.app.services.state_machine import work_orders_out

router = APIRouter(prefix="/exports", tags=["exports"])

Billing = RoleDep(Role.zonal_admin, Role.field_engineer, Role.contractor)


def _ids(raw: Optional[str]) -> Optional[list[int]]:
    if not raw:
        return None
    try:
        return [int(x) for x in raw.split(",") if x.strip()]
    except ValueError as e:
        raise HTTPException(422, "work_order_ids must be a comma list of integers") from e


@router.get("/invoice-eligible", response_model=list[WorkOrderOut],
            summary="Work orders that may be invoiced / entered in the MB (verified_closed, not rework)")
def invoice_eligible(db: DbDep, storage: StorageDep, p: Billing,
                     contractor: Optional[int] = None) -> list[WorkOrderOut]:
    stmt = scope_query(p, svc.eligible_stmt(), m.WorkOrder)
    if contractor is not None:
        stmt = stmt.where(m.WorkOrder.contractor_id == contractor)
    return work_orders_out(db, list(db.scalars(stmt)), storage)


@router.get("/measurement-book.csv", response_class=Response,
            summary="Measurement Book CSV of invoice-eligible work orders (optionally ?work_order_ids=1,2)")
def measurement_book(db: DbDep, p: Billing, work_order_ids: Optional[str] = None,
                     contractor: Optional[int] = None) -> Response:
    WO = m.WorkOrder
    wanted = _ids(work_order_ids)
    stmt = scope_query(p, select(WO), WO)
    if wanted is not None:
        stmt = stmt.where(WO.id.in_(wanted))
    if contractor is not None:
        stmt = stmt.where(WO.contractor_id == contractor)
    wos = list(db.scalars(stmt.order_by(WO.id)))
    eligible = [w for w in wos if svc.is_eligible(w)]
    excluded = sorted(set(wanted or []) - {w.id for w in eligible})
    body = svc.measurement_book(db, eligible)
    stamp = f"{to_ist(now_utc()):%Y%m%d}"
    headers = {"Content-Disposition": f'attachment; filename="measurement_book_{stamp}.csv"',
               "X-Eligible-Work-Orders": ",".join(str(w.id) for w in eligible)}
    if excluded:
        headers["X-Excluded-Work-Orders"] = ",".join(str(i) for i in excluded)
    return Response(content=body, media_type="text/csv; charset=utf-8", headers=headers)


@router.get("/potholes.geojson", summary="Potholes as GeoJSON (scoped; public callers see public potholes)")
def potholes_geojson(db: DbDep, p: PublicPrincipalDep, response: Response,
                     status_: Optional[str] = Query(None, alias="status", description="comma list"),
                     admin_unit: Optional[int] = None, bbox: Optional[str] = None,
                     limit: int = Query(svc.GEOJSON_MAX_FEATURES, ge=1, le=svc.GEOJSON_MAX_FEATURES)) -> dict[str, Any]:
    P = m.Pothole
    stmt = (select(P, m.RoadSegment.name, m.RoadSegment.highway, m.Agency.code, m.AdminUnit.name_en)
            .outerjoin(m.RoadSegment, m.RoadSegment.id == P.road_segment_id)
            .outerjoin(m.Agency, m.Agency.id == P.agency_id)
            .outerjoin(m.AdminUnit, m.AdminUnit.id == P.admin_unit_id))
    stmt = scope_query(p, stmt, P)
    if status_:
        vals = [v.strip() for v in status_.split(",") if v.strip()]
        bad = [v for v in vals if v not in {s.value for s in PotholeStatus}]
        if bad:
            raise HTTPException(422, f"unknown status {bad}")
        stmt = stmt.where(P.status.in_(vals))
    if admin_unit is not None:
        stmt = stmt.where(P.admin_unit_id.in_(admin_subtree_ids(admin_unit)))
    if bbox:
        try:
            bb = parse_bbox(bbox)
        except ValueError as e:
            raise HTTPException(422, f"bbox: {e}") from e
        if bb:
            stmt = stmt.where(P.lon.between(bb[0], bb[2]), P.lat.between(bb[1], bb[3]))
    rows = db.execute(stmt.order_by(P.id).limit(limit + 1)).all()
    response.headers["Content-Disposition"] = 'attachment; filename="potholes.geojson"'
    return svc.potholes_geojson([tuple(r) for r in rows[:limit]], truncated=len(rows) > limit)
