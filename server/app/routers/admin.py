"""/v1/admin: versioned priority config + rate card, admin-unit GeoJSON upload, jurisdiction queue, audit viewer.

zonal_admin acts only on its own agency; super_admin on any (``?agency=CODE``). Every change writes audit_log in
the same transaction (``server.app.configstore`` / ``server.app.core.audit``).
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.app import configstore
from server.app.core import audit
from server.app.core.rbac import Principal, RoleDep, ensure_visible, scope_query
from server.app.core.util import now_utc
from server.app.db import models as m
from server.app.db.convert import model_snapshot
from server.app.deps import DbDep
from server.app.schemas.api import AuditEntry, JurisdictionItem, JurisdictionResolve
from server.app.schemas.states import Role

router = APIRouter(prefix="/admin", tags=["admin"])

Admin = RoleDep(Role.zonal_admin)
Staff = RoleDep(Role.zonal_admin, Role.field_engineer)
ADMIN_UNIT_KINDS = ("zone", "circle", "ward", "division")


def _agency_scope(db: Session, p: Principal, code: Optional[str], *, write: bool) -> Optional[m.Agency]:
    """Agency the call acts on. zonal_admin/field_engineer: always their own (a different code -> 403)."""
    if code:
        ag = db.scalars(select(m.Agency).where(m.Agency.code == code.upper())).first()
        if ag is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown agency {code!r}")
    else:
        ag = db.get(m.Agency, p.agency_id) if p.agency_id is not None else None
    if not p.is_super:
        if p.agency_id is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "user has no agency")
        if ag is None or ag.id != p.agency_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "may only act on your own agency")
    if write and ag is None and not p.is_super:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "only super_admin may edit the global default")
    return ag


# =========================================================================== priority config
def _pc_out(row: Optional[m.PriorityConfig], ag: Optional[m.Agency], cfg: dict[str, Any]) -> dict[str, Any]:
    return {"id": row.id if row else None, "agency": ag.code if ag else None,
            "inherited": bool(row is not None and ag is not None and row.agency_id != ag.id),
            "version": row.version if row else cfg.get("version"), "active": bool(row.active) if row else True,
            "created_at": row.created_at.isoformat() if row and row.created_at else None, "config": cfg}


@router.get("/priority-config", summary="Active priority weights/config for an agency (falls back to global)")
def get_priority_config(db: DbDep, p: Staff, agency: Optional[str] = None) -> dict[str, Any]:
    ag = _agency_scope(db, p, agency, write=False)
    row = configstore.active_priority_row(db, ag.id if ag else None)
    cfg = row.weights if row else configstore.load_yaml(configstore.DEFAULT_PRIORITY_FILE)
    return _pc_out(row, ag, cfg)


@router.get("/priority-config/history", summary="All versions (newest first)")
def priority_config_history(db: DbDep, p: Staff, agency: Optional[str] = None) -> list[dict[str, Any]]:
    ag = _agency_scope(db, p, agency, write=False)
    PC = m.PriorityConfig
    rows = db.scalars(select(PC).where(PC.agency_id == ag.id if ag else PC.agency_id.is_(None))
                      .order_by(PC.version.desc()))
    return [_pc_out(r, ag, r.weights) for r in rows]


@router.put("/priority-config", summary="Save a new version (body = full priority config; weights sum to 1)")
def put_priority_config(db: DbDep, p: Admin,
                        config: Annotated[dict[str, Any], Body(description="configs/priority.default.yaml shape")],
                        agency: Optional[str] = None, reason: Optional[str] = None) -> dict[str, Any]:
    ag = _agency_scope(db, p, agency, write=True)
    try:
        row = configstore.save_priority_config(db, ag.id if ag else None, config, actor=p, reason=reason)
    except configstore.ConfigError as e:
        raise HTTPException(422, str(e)) from e
    db.commit()
    return _pc_out(row, ag, row.weights)


# =========================================================================== rate card
def _rc_out(row: Optional[m.RateCard], ag: Optional[m.Agency], card: dict[str, Any]) -> dict[str, Any]:
    return {"id": row.id if row else None, "agency": ag.code if ag else None,
            "fy": row.fy if row else card.get("fy"), "version": row.version if row else card.get("version"),
            "version_label": configstore.rate_card_version_label(row) if row else "yaml-default",
            "active": bool(row.active) if row else True,
            "created_at": row.created_at.isoformat() if row and row.created_at else None, "card": card}


@router.get("/rate-card", summary="Active rate card (INR) for an agency and FY")
def get_rate_card(db: DbDep, p: Staff, agency: Optional[str] = None, fy: Optional[str] = None) -> dict[str, Any]:
    ag = _agency_scope(db, p, agency, write=False)
    row = configstore.active_rate_card_row(db, ag.id if ag else None, fy)
    card = row.items if row else configstore.load_yaml(configstore.DEFAULT_RATE_CARD_FILE)
    return _rc_out(row, ag, card)


@router.put("/rate-card", summary="Save a new rate-card version (body = full rate card with fy + items)")
def put_rate_card(db: DbDep, p: Admin, card: Annotated[dict[str, Any], Body(description="rate_card yaml shape")],
                  agency: Optional[str] = None, reason: Optional[str] = None) -> dict[str, Any]:
    ag = _agency_scope(db, p, agency, write=True)
    try:
        row = configstore.save_rate_card(db, ag.id if ag else None, card, actor=p, reason=reason)
    except configstore.ConfigError as e:
        raise HTTPException(422, str(e)) from e
    db.commit()
    return _rc_out(row, ag, row.items)


# =========================================================================== admin units
def upsert_admin_units(db: Session, fc: dict[str, Any], *, default_agency: Optional[m.Agency], actor: Any,
                       allowed_agency_id: Optional[int] = None) -> dict[str, Any]:
    """Upsert a GeoJSON FeatureCollection of admin units.

    Feature properties: ``code`` (required), ``kind`` (zone|circle|ward|division), ``name_en`` (required),
    ``name_te``, ``name_hi``, ``parent_code``, ``agency`` (code; default = ``default_agency``), ``official``
    (default false), ``valid_from`` / ``valid_to`` (ISO dates). Geometry: Polygon | MultiPolygon (WGS84).
    Existing rows (same agency, kind, code) are updated in place; before/after go to audit_log.
    Parents are resolved after all rows exist, so a file may list children before parents.
    """
    from shapely.geometry import shape
    from shapely.validation import make_valid

    if not isinstance(fc, dict) or fc.get("type") != "FeatureCollection" or not isinstance(fc.get("features"), list):
        raise ValueError("expected a GeoJSON FeatureCollection")
    agencies = {a.code: a for a in db.scalars(select(m.Agency))}
    created = updated = 0
    errors: list[dict[str, Any]] = []
    pending_parents: list[tuple[m.AdminUnit, str]] = []
    for i, f in enumerate(fc["features"]):
        try:
            props = f.get("properties") or {}
            code, kind, name_en = props.get("code"), props.get("kind"), props.get("name_en") or props.get("name")
            if not code or not name_en:
                raise ValueError("properties.code and properties.name_en are required")
            if kind not in ADMIN_UNIT_KINDS:
                raise ValueError(f"properties.kind must be one of {ADMIN_UNIT_KINDS}")
            ag = agencies.get(str(props["agency"]).upper()) if props.get("agency") else default_agency
            if ag is None:
                raise ValueError("unknown or missing agency")
            if allowed_agency_id is not None and ag.id != allowed_agency_id:
                raise ValueError("may only upload units of your own agency")
            geom = shape(f.get("geometry") or {})
            if geom.geom_type not in ("Polygon", "MultiPolygon"):
                raise ValueError("geometry must be Polygon or MultiPolygon")
            if not geom.is_valid:
                geom = make_valid(geom)
            if geom.is_empty:
                raise ValueError("empty geometry")
            vf = date.fromisoformat(str(props["valid_from"])[:10]) if props.get("valid_from") else None
            vt = date.fromisoformat(str(props["valid_to"])[:10]) if props.get("valid_to") else None
            row = db.scalars(select(m.AdminUnit).where(m.AdminUnit.agency_id == ag.id, m.AdminUnit.kind == kind,
                                                       m.AdminUnit.code == str(code))).first()
            before = model_snapshot(row, exclude=("geom_wkt",)) if row is not None else None
            if row is None:
                row = m.AdminUnit(agency_id=ag.id, kind=kind, code=str(code))
                db.add(row)
                created += 1
            else:
                updated += 1
            row.name_en, row.name_te, row.name_hi = name_en, props.get("name_te"), props.get("name_hi")
            row.geom_wkt = geom.wkt
            row.valid_from, row.valid_to = vf, vt
            row.official = bool(props.get("official", False))
            row.refresh_bbox()
            db.flush()
            if props.get("parent_code"):
                pending_parents.append((row, str(props["parent_code"])))
            audit.record(db, "admin_unit", row.id, "create" if before is None else "update", actor=actor,
                         before=before, after=model_snapshot(row, exclude=("geom_wkt",)))
        except (ValueError, TypeError, KeyError) as e:
            errors.append({"index": i, "code": (f.get("properties") or {}).get("code") if isinstance(f, dict) else None,
                           "error": str(e)})
    for row, pcode in pending_parents:
        parent = db.scalars(select(m.AdminUnit).where(m.AdminUnit.agency_id == row.agency_id,
                                                      m.AdminUnit.code == pcode)
                            .order_by(m.AdminUnit.id.desc())).first()
        if parent is None or parent.id == row.id:
            errors.append({"code": row.code, "error": f"parent_code {pcode!r} not found"})
            continue
        row.parent_id = parent.id
    db.flush()
    return {"created": created, "updated": updated, "errors": errors}


@router.post("/admin-units", summary="Upload admin units (GeoJSON FeatureCollection; pluggable boundaries)")
def post_admin_units(db: DbDep, p: Admin, fc: Annotated[dict[str, Any], Body(description="FeatureCollection")],
                     agency: Optional[str] = None) -> dict[str, Any]:
    ag = _agency_scope(db, p, agency, write=True)
    try:
        res = upsert_admin_units(db, fc, default_agency=ag, actor=p,
                                 allowed_agency_id=None if p.is_super else p.agency_id)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if res["errors"] and not (res["created"] or res["updated"]):
        db.rollback()
        raise HTTPException(422, res)
    db.commit()
    return res


# =========================================================================== jurisdiction queue
def _jq_out(r: m.JurisdictionQueue) -> JurisdictionItem:
    cands = [str(c.get("agency_code", c)) if isinstance(c, dict) else str(c) for c in (r.candidates or [])]
    return JurisdictionItem(id=r.id, pothole_id=r.pothole_id, road_segment_id=r.road_segment_id, reason=r.reason,
                            candidates=cands, created_at=r.created_at, resolved=r.resolved_at is not None)


@router.get("/jurisdiction-queue", response_model=list[JurisdictionItem],
            summary="Potholes / road segments whose owning agency is unknown or disputed")
def jurisdiction_queue(db: DbDep, p: Staff, resolved: bool = False, pothole_only: bool = False,
                       limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)
                       ) -> list[JurisdictionItem]:
    JQ = m.JurisdictionQueue
    stmt = scope_query(p, select(JQ), JQ)
    stmt = stmt.where(JQ.resolved_at.is_not(None) if resolved else JQ.resolved_at.is_(None))
    if pothole_only:
        stmt = stmt.where(JQ.pothole_id.is_not(None))
    rows = db.scalars(stmt.order_by(JQ.pothole_id.is_(None), JQ.created_at.desc(), JQ.id.desc())
                      .limit(limit).offset(offset))
    return [_jq_out(r) for r in rows]


@router.post("/jurisdiction-queue/{item_id}/resolve", response_model=JurisdictionItem,
             summary="Assign the owning agency (and contract) to the pothole and, optionally, its road segment")
def resolve_jurisdiction(item_id: int, body: JurisdictionResolve, db: DbDep, p: Admin) -> JurisdictionItem:
    item: m.JurisdictionQueue = ensure_visible(db, p, m.JurisdictionQueue, item_id)
    if item.resolved_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "already resolved")
    ag = db.scalars(select(m.Agency).where(m.Agency.code == body.agency_code.upper())).first()
    if ag is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown agency {body.agency_code!r}")
    contract = db.get(m.Contract, body.contract_id) if body.contract_id else None
    if body.contract_id and contract is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"contract {body.contract_id} not found")
    if contract is not None and contract.agency_id not in (None, ag.id):
        raise HTTPException(422, "contract belongs to another agency")
    changes: dict[str, Any] = {}
    if item.pothole_id:
        ph = db.get(m.Pothole, item.pothole_id)
        if ph is not None:
            changes["pothole"] = {"id": ph.id, "agency_id_before": ph.agency_id, "agency_id": ag.id}
            ph.agency_id = ag.id
            if contract is not None:
                ph.contract_id, ph.contractor_id = contract.id, contract.contractor_id
    seg_id = item.road_segment_id
    if seg_id is None and item.pothole_id:
        ph = db.get(m.Pothole, item.pothole_id)
        seg_id = ph.road_segment_id if ph else None
    if body.apply_to_segment and seg_id:
        seg = db.get(m.RoadSegment, seg_id)
        if seg is not None:
            changes["road_segment"] = {"id": seg.id, "owner_before": seg.owner_agency_id, "owner": ag.id}
            seg.owner_agency_id = ag.id
            if contract is not None:
                seg.contract_id = contract.id
    item.resolved_at = now_utc()
    item.resolved_by = p.user_id
    item.resolution = {"agency_code": ag.code, "contract_id": body.contract_id,
                       "apply_to_segment": body.apply_to_segment, **changes}
    audit.record(db, "jurisdiction_queue", item.id, "resolve", actor=p, before={"resolved": False},
                 after=item.resolution, reason=f"owner -> {ag.code}")
    db.commit()
    return _jq_out(item)


# =========================================================================== audit
@router.get("/audit", response_model=list[AuditEntry], summary="Audit log (append-only), newest first")
def get_audit(db: DbDep, p: Admin, entity: Optional[str] = None,
              id: Optional[int] = Query(None, description="entity id"),  # noqa: A002 - API contract name
              action: Optional[str] = None, limit: int = Query(100, ge=1, le=1000),
              offset: int = Query(0, ge=0)) -> list[AuditEntry]:
    A = m.AuditLog
    stmt = scope_query(p, select(A), A)
    if entity:
        stmt = stmt.where(A.entity == entity)
    if id is not None:
        stmt = stmt.where(A.entity_id == id)
    if action:
        stmt = stmt.where(A.action == action)
    rows = db.scalars(stmt.order_by(A.at.desc(), A.id.desc()).limit(limit).offset(offset))
    return [AuditEntry(id=r.id, entity=r.entity, entity_id=r.entity_id if r.entity_id is not None else 0,
                       action=r.action, actor=r.actor, before=r.before, after=r.after, reason=r.reason, at=r.at)
            for r in rows]
