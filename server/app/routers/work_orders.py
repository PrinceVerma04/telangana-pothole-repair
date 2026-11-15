"""/v1/work-orders: create, list, detail, state transitions, before/after photos (WP8b).

RBAC (``server.app.core.rbac``): staff (zonal_admin / field_engineer, scoped to agency + admin-unit subtree) create
and issue; a contractor sees and moves only its own work orders (others answer 404) and can never verify its own
repair (403). Illegal transitions answer 409, missing evidence (before/after photo, bags_used) 422. Every legal
transition writes audit_log (``server.app.services.state_machine``).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select

from server.app.core import audit
from server.app.core.rbac import PrincipalDep, RoleDep, ensure_visible, scope_query
from server.app.core.util import as_utc, now_utc
from server.app.db import models as m
from server.app.deps import DbDep, SettingsDep, StorageDep
from server.app.schemas.api import WorkOrderCreate, WorkOrderOut, WorkOrderTransition
from server.app.schemas.states import Role, WorkOrderStatus
from server.app.services import exports
from server.app.services.state_machine import (
    IllegalTransition,
    PreconditionFailed,
    TransitionForbidden,
    create_work_order,
    transition_work_order,
    work_order_out,
    work_orders_out,
)

log = logging.getLogger("gunthameter.work_orders")
router = APIRouter(prefix="/work-orders", tags=["work orders"])

Staff = RoleDep(Role.zonal_admin, Role.field_engineer)
Reader = RoleDep(Role.zonal_admin, Role.field_engineer, Role.contractor)
PhotoTaker = RoleDep(Role.zonal_admin, Role.field_engineer, Role.contractor)
W = WorkOrderStatus
PHOTO_STATES = {"before": {W.issued.value, W.acknowledged.value, W.in_progress.value},
                "after": {W.in_progress.value, W.repaired_claimed.value}}
_MAGIC = {b"\xff\xd8\xff": ("image/jpeg", "jpg"), b"\x89PNG\r\n\x1a\n": ("image/png", "png")}


def _http(e: Exception) -> HTTPException:
    if isinstance(e, TransitionForbidden):
        return HTTPException(status.HTTP_403_FORBIDDEN, str(e))
    if isinstance(e, IllegalTransition):
        return HTTPException(status.HTTP_409_CONFLICT, str(e))
    if isinstance(e, LookupError):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(e).strip("'\""))
    return HTTPException(422, str(e))


def _sniff(data: bytes) -> Optional[tuple[str, str]]:
    for magic, ct in _MAGIC.items():
        if data.startswith(magic):
            return ct
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", "webp"
    return None


# =========================================================================== create / list / detail
@router.post("", response_model=WorkOrderOut, status_code=status.HTTP_201_CREATED,
             summary="Create a work order for potholes (plans them if needed). issue: default = issue when a "
                     "contractor is known (from body, contract or the pothole's contract)")
def create(body: WorkOrderCreate, db: DbDep, storage: StorageDep, p: Staff,
           issue: Optional[bool] = Query(None, description="true: issue now; false: keep planned")) -> WorkOrderOut:
    for pid in body.pothole_ids:
        ensure_visible(db, p, m.Pothole, pid)
    if body.contract_id is not None:
        ensure_visible(db, p, m.Contract, body.contract_id)
    try:
        wo = create_work_order(db, body.pothole_ids, actor=p, contract_id=body.contract_id,
                               contractor_id=body.contractor_id, crew=body.crew, sla_due=body.sla_due,
                               run_id=body.run_id, notes=body.notes, issue=False)
        if issue or (issue is None and wo.contractor_id is not None):
            transition_work_order(db, wo, W.issued, actor=p, reason="issued on creation")
    except (TransitionForbidden, IllegalTransition, PreconditionFailed, LookupError, ValueError) as e:
        db.rollback()
        raise _http(e) from e
    db.commit()
    return work_order_out(db, wo, storage)


@router.get("", response_model=list[WorkOrderOut],
            summary="Work orders visible to the caller (status comma list, contractor id, agency code)")
def list_work_orders(
    db: DbDep, storage: StorageDep, p: Reader,
    status_: Optional[str] = Query(None, alias="status", description="comma list of WorkOrderStatus"),
    contractor: Optional[int] = Query(None),
    agency: Optional[str] = Query(None, description="agency code, e.g. GHMC"),
    run: Optional[int] = Query(None, description="repair run id"),
    invoice_eligible: bool = Query(False, description="only verified_closed, non-rework (MB / invoice eligible)"),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> list[WorkOrderOut]:
    WO = m.WorkOrder
    stmt = scope_query(p, select(WO), WO)
    if status_:
        vals = [v.strip() for v in status_.split(",") if v.strip()]
        bad = [v for v in vals if v not in {s.value for s in W}]
        if bad:
            raise HTTPException(422, f"unknown status {bad}")
        stmt = stmt.where(WO.status.in_(vals))
    if contractor is not None:
        stmt = stmt.where(WO.contractor_id == contractor)
    if agency:
        stmt = stmt.where(WO.agency_id.in_(select(m.Agency.id).where(m.Agency.code == agency.upper())))
    if run is not None:
        stmt = stmt.where(WO.run_id == run)
    if invoice_eligible:
        stmt = stmt.where(WO.status.in_(exports.ELIGIBLE_STATUSES), WO.is_rework.is_(False))
    wos = list(db.scalars(stmt.order_by(WO.id.desc()).limit(limit).offset(offset)))
    return work_orders_out(db, wos, storage)


@router.get("/{work_order_id}", response_model=WorkOrderOut, summary="One work order")
def get_work_order(work_order_id: int, db: DbDep, storage: StorageDep, p: Reader) -> WorkOrderOut:
    return work_order_out(db, ensure_visible(db, p, m.WorkOrder, work_order_id), storage)


# =========================================================================== transitions
@router.patch("/{work_order_id}/transition", response_model=WorkOrderOut,
              summary="Move a work order (409 illegal, 403 role/ownership, 422 missing photo/bags)")
def transition(work_order_id: int, body: WorkOrderTransition, db: DbDep, storage: StorageDep,
               p: PrincipalDep) -> WorkOrderOut:
    if p.role not in (Role.zonal_admin, Role.field_engineer, Role.contractor, Role.super_admin):
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"role {p.role.value} cannot move work orders")
    wo: m.WorkOrder = ensure_visible(db, p, m.WorkOrder, work_order_id)
    try:
        transition_work_order(db, wo, body.to, actor=p, reason=body.reason, lat=body.lat, lon=body.lon,
                              bags_used=body.bags_used, bag_lot_no=body.bag_lot_no)
    except (TransitionForbidden, IllegalTransition, PreconditionFailed, LookupError) as e:
        db.rollback()
        raise _http(e) from e
    db.commit()
    return work_order_out(db, wo, storage)


# =========================================================================== photos
@router.post("/{work_order_id}/photos", status_code=status.HTTP_201_CREATED,
             summary="Before/after photo (multipart: kind=before|after, file, optional lat, lon, at ISO time)")
async def upload_photo(
    work_order_id: int, db: DbDep, storage: StorageDep, settings: SettingsDep, p: PhotoTaker,
    kind: Annotated[str, Form(pattern="^(before|after)$")],
    file: Annotated[UploadFile, File(description="JPEG/PNG/WebP")],
    lat: Annotated[Optional[float], Form(ge=6.0, le=38.0)] = None,
    lon: Annotated[Optional[float], Form(ge=68.0, le=98.0)] = None,
    at: Annotated[Optional[datetime], Form(description="capture time (offline queue); default now")] = None,
) -> dict[str, Any]:
    wo: m.WorkOrder = ensure_visible(db, p, m.WorkOrder, work_order_id)
    if p.role == Role.contractor and wo.contractor_id != p.contractor_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"work_order {work_order_id} not found")
    if wo.status not in PHOTO_STATES[kind]:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"{kind} photo not accepted while the work order is {wo.status}")
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"photo larger than {settings.MAX_UPLOAD_MB} MB")
    sniff = _sniff(data)
    if sniff is None:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "photo must be JPEG, PNG or WebP")
    now = now_utc()
    taken = as_utc(at) if at is not None and as_utc(at) <= now else now  # type: ignore[operator]
    key = f"repairs/wo{wo.id}/{kind}-{now:%Y%m%dT%H%M%S%f}.{sniff[1]}"
    storage.put(key, data, sniff[0])
    ev = m.RepairEvent(work_order_id=wo.id, kind=f"{kind}_photo", photo_key=key, lat=lat, lon=lon, at=taken,
                       actor_id=p.user_id, data={"bytes": len(data), "content_type": sniff[0],
                                                 "received_at": now.isoformat(), "uploaded_by": p.label})
    db.add(ev)
    db.flush()
    audit.record(db, "work_order", wo.id, f"photo:{kind}", actor=p, after={"key": key, "repair_event_id": ev.id},
                 lat=lat, lon=lon)
    db.commit()
    try:
        url = storage.url(key)
    except Exception:
        url = None
    return {"id": ev.id, "kind": kind, "url": url, "key": key, "at": taken.isoformat(), "lat": lat, "lon": lon}
