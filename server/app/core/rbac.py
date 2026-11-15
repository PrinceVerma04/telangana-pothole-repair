"""Authentication principal, role guards and row-level scoping (WP4).

* ``get_principal`` / ``get_optional_principal``: FastAPI dependencies decoding the bearer JWT (user or device).
* ``require_role(*roles)``: dependency factory; ``super_admin`` always passes.
* ``scope_query(principal, stmt)``: adds WHERE clauses so that
    - contractor   sees only its own work orders (and the potholes / events / photos hanging off them),
    - zonal_admin / field_engineer see only their agency and admin-unit subtree (zone -> circles -> wards),
    - citizen      sees public (non-candidate, non-rejected) potholes and its own complaints,
    - driver_device sees only its own device / trips / observations,
    - super_admin  sees everything.
  Reference tables (agency, admin_unit, road_segment, poi, priority_config, rate_card, model_registry, way_pass)
  are readable by everyone; every other table is DENIED for non-staff roles unless a rule exists (deny-by-default).
* ``ensure_visible(db, principal, Model, id)``: single-row read through the same scope; 404 (not 403) when hidden,
  so object existence is not leaked.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import Select, false, or_, select
from sqlalchemy.orm import Session, aliased

from server.app.core.config import Settings
from server.app.core.security import TokenError, decode_token, sha256_hex
from server.app.db import models as m
from server.app.deps import DbDep, SettingsDep
from server.app.schemas.states import Role

STAFF = frozenset({Role.super_admin, Role.zonal_admin, Role.field_engineer})
REFERENCE_MODELS: frozenset[type] = frozenset({
    m.Agency, m.AdminUnit, m.RoadSegment, m.Poi, m.PriorityConfig, m.RateCard, m.ModelRegistry, m.WayPass,
})
PUBLIC_POTHOLE_HIDDEN = ("candidate", "rejected", "archived")
_CTE_SEQ = itertools.count(1)

_bearer = HTTPBearer(auto_error=False, description="JWT from POST /v1/auth/token or POST /v1/edge/register")


@dataclass(frozen=True)
class Principal:
    kind: str  # "user" | "device"
    role: Role
    user_id: Optional[int] = None
    username: Optional[str] = None
    agency_id: Optional[int] = None
    admin_unit_id: Optional[int] = None
    contractor_id: Optional[int] = None
    device_pk: Optional[int] = None
    device_id: Optional[str] = None

    @property
    def label(self) -> str:
        if self.kind == "anonymous":
            return "anonymous"
        return self.username if self.kind == "user" and self.username else f"device:{self.device_id}"

    @property
    def is_staff(self) -> bool:
        return self.role in STAFF

    @property
    def is_super(self) -> bool:
        return self.role == Role.super_admin


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def principal_from_token(db: Session, settings: Settings, token: str) -> Principal:
    try:
        claims = decode_token(settings, token)
    except TokenError as e:
        raise _unauthorized(str(e)) from e
    typ = claims.get("typ")
    if typ == "device":
        dev = db.get(m.Device, claims.get("did"))
        if dev is None or dev.token_hash != sha256_hex(token):
            raise _unauthorized("device token revoked or unknown device")
        return Principal(kind="device", role=Role.driver_device, device_pk=dev.id, device_id=dev.device_id,
                         agency_id=dev.agency_id)
    if typ == "user":
        try:
            uid = int(claims["sub"])
        except (KeyError, ValueError) as e:
            raise _unauthorized("bad subject") from e
        u = db.get(m.User, uid)
        if u is None or not u.active:
            raise _unauthorized("user inactive or unknown")
        return Principal(kind="user", role=Role(u.role), user_id=u.id, username=u.username, agency_id=u.agency_id,
                         admin_unit_id=u.admin_unit_id, contractor_id=u.contractor_id)
    raise _unauthorized("unknown token type")


def get_optional_principal(
    request: Request,
    creds: Annotated[Optional[HTTPAuthorizationCredentials], Depends(_bearer)],
    db: DbDep,
    settings: SettingsDep,
) -> Optional[Principal]:
    if creds is None or not creds.credentials:
        return None
    p = principal_from_token(db, settings, creds.credentials)
    request.state.principal = p
    return p


OptionalPrincipalDep = Annotated[Optional[Principal], Depends(get_optional_principal)]


def get_principal(p: OptionalPrincipalDep) -> Principal:
    if p is None:
        raise _unauthorized("authentication required")
    return p


PrincipalDep = Annotated[Principal, Depends(get_principal)]
ANONYMOUS = Principal(kind="anonymous", role=Role.citizen)


def get_principal_or_public(p: OptionalPrincipalDep) -> Principal:
    """Bearer principal, or an anonymous principal with the citizen (public) scope."""
    return p if p is not None else ANONYMOUS


PublicPrincipalDep = Annotated[Principal, Depends(get_principal_or_public)]


def require_role(*roles: Role | str) -> Callable[..., Principal]:
    allowed = frozenset(Role(r) for r in roles)

    def _dep(p: PrincipalDep) -> Principal:
        if p.is_super or p.role in allowed:
            return p
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            detail=f"role {p.role.value} not allowed; needs one of {sorted(r.value for r in allowed)}")

    _dep.__name__ = "require_" + "_or_".join(sorted(r.value for r in allowed)) if allowed else "require_any"
    return _dep


def RoleDep(*roles: Role | str) -> Any:  # noqa: N802 - reads like a type in signatures
    """``p: RoleDep(Role.zonal_admin)`` == ``Annotated[Principal, Depends(require_role(Role.zonal_admin))]``."""
    return Annotated[Principal, Depends(require_role(*roles))]


# ============================================================================ scoping
def admin_subtree_ids(root_id: int) -> Select:
    """SELECT id of ``root_id`` and every descendant admin unit (portable recursive CTE)."""
    # unique CTE name per call: a query may embed several subtrees (scope + admin_unit filter)
    name = f"au_tree_{next(_CTE_SEQ)}"
    base = select(m.AdminUnit.id.label("id")).where(m.AdminUnit.id == root_id).cte(name, recursive=True)
    child = aliased(m.AdminUnit)
    tree = base.union_all(select(child.id).where(child.parent_id == base.c.id))
    return select(tree.c.id)


def scoped_pothole_ids(p: Principal) -> Optional[Select]:
    """SELECT of visible pothole ids, or None when unrestricted."""
    cond = _pothole_condition(p)
    if cond is None:
        return None
    return select(m.Pothole.id).where(cond)


def _pothole_condition(p: Principal) -> Any:
    P = m.Pothole
    if p.is_super:
        return None
    if p.role in (Role.zonal_admin, Role.field_engineer):
        if p.agency_id is None and p.admin_unit_id is None:
            return false()
        conds = []
        if p.agency_id is not None:
            conds.append(P.agency_id == p.agency_id)
        if p.admin_unit_id is not None:
            conds.append(or_(P.admin_unit_id.is_(None), P.admin_unit_id.in_(admin_subtree_ids(p.admin_unit_id))))
        return conds[0] if len(conds) == 1 else conds[0] & conds[1]
    if p.role == Role.contractor:
        if p.contractor_id is None:
            return false()
        via_wo = (select(m.WorkOrderPothole.pothole_id)
                  .join(m.WorkOrder, m.WorkOrder.id == m.WorkOrderPothole.work_order_id)
                  .where(m.WorkOrder.contractor_id == p.contractor_id))
        return or_(P.contractor_id == p.contractor_id, P.id.in_(via_wo))
    if p.role == Role.citizen:
        return P.status.notin_(PUBLIC_POTHOLE_HIDDEN)
    return false()


def _work_order_condition(p: Principal) -> Any:
    W = m.WorkOrder
    if p.is_super:
        return None
    if p.role == Role.contractor:
        return W.contractor_id == p.contractor_id if p.contractor_id is not None else false()
    if p.role in (Role.zonal_admin, Role.field_engineer):
        if p.agency_id is None and p.admin_unit_id is None:
            return false()
        conds = []
        if p.agency_id is not None:
            conds.append(W.agency_id == p.agency_id)
        if p.admin_unit_id is not None:
            in_unit = (select(m.WorkOrderPothole.work_order_id)
                       .join(m.Pothole, m.Pothole.id == m.WorkOrderPothole.pothole_id)
                       .where(or_(m.Pothole.admin_unit_id.is_(None),
                                  m.Pothole.admin_unit_id.in_(admin_subtree_ids(p.admin_unit_id)))))
            conds.append(W.id.in_(in_unit))
        return conds[0] if len(conds) == 1 else conds[0] & conds[1]
    return false()


def scoped_work_order_ids(p: Principal) -> Optional[Select]:
    cond = _work_order_condition(p)
    return None if cond is None else select(m.WorkOrder.id).where(cond)


def _condition_for(p: Principal, model: type) -> Any:  # noqa: C901 - one branch per table, kept flat on purpose
    """WHERE clause for ``model`` under principal ``p``; None = unrestricted."""
    if p.is_super or model in REFERENCE_MODELS:
        return None
    if model is m.Pothole:
        return _pothole_condition(p)
    if model is m.WorkOrder:
        return _work_order_condition(p)

    # device-owned telemetry
    if model in (m.Device, m.Observation, m.Trip, m.TripFix, m.CalibCertRow):
        if p.role == Role.driver_device:
            if model is m.Device:
                return m.Device.id == p.device_pk
            if model is m.TripFix:
                return m.TripFix.trip_id.in_(select(m.Trip.id).where(m.Trip.device_id == p.device_pk))
            return model.device_id == p.device_pk  # type: ignore[attr-defined]
        return None if p.is_staff else false()

    if model is m.User:
        if p.role == Role.zonal_admin and p.agency_id is not None:
            return or_(m.User.agency_id == p.agency_id, m.User.id == p.user_id)
        return m.User.id == p.user_id if p.user_id is not None else false()

    if model is m.Contractor:
        if p.role == Role.contractor:
            return m.Contractor.id == p.contractor_id if p.contractor_id else false()
        return None if p.is_staff else false()

    if model is m.Contract:
        if p.role == Role.contractor:
            return m.Contract.contractor_id == p.contractor_id if p.contractor_id else false()
        if p.is_staff:
            return m.Contract.agency_id == p.agency_id if p.agency_id is not None else None
        return false()

    if model is m.RepairRun:
        if p.role == Role.contractor:
            wo = scoped_work_order_ids(p)
            return m.RepairRun.id.in_(select(m.WorkOrder.run_id).where(m.WorkOrder.id.in_(wo))) if wo is not None \
                else None
        if p.is_staff:
            if p.admin_unit_id is not None:
                return m.RepairRun.circle_id.in_(admin_subtree_ids(p.admin_unit_id))
            if p.agency_id is not None:
                return m.RepairRun.circle_id.in_(select(m.AdminUnit.id).where(m.AdminUnit.agency_id == p.agency_id))
            return false()
        return false()

    if model is m.Complaint:
        if p.role == Role.citizen:
            return m.Complaint.user_id == p.user_id if p.user_id is not None else false()
        if p.is_staff:
            return None
        return false()

    if model is m.Notification:
        if p.is_staff and p.role == Role.zonal_admin:
            return None
        conds = [m.Notification.to_role == p.role.value]
        if p.user_id is not None:
            conds.append(m.Notification.to_user_id == p.user_id)
        return or_(*conds)

    if model in (m.AuditLog, m.Job, m.JurisdictionQueue):
        return None if p.role == Role.zonal_admin or (p.is_staff and model is m.JurisdictionQueue) else false()

    # rows hanging off a work order
    if model in (m.RepairEvent, m.DlpEvent, m.WorkOrderPothole):
        wo = scoped_work_order_ids(p)
        if p.role not in (Role.contractor, Role.zonal_admin, Role.field_engineer):
            return false()
        return None if wo is None else model.work_order_id.in_(wo)  # type: ignore[attr-defined]
    if model is m.Verification:
        if p.role not in (Role.contractor, Role.zonal_admin, Role.field_engineer):
            return false()
        wo, ph = scoped_work_order_ids(p), scoped_pothole_ids(p)
        return or_(m.Verification.work_order_id.in_(wo), m.Verification.pothole_id.in_(ph)) \
            if wo is not None and ph is not None else None

    # rows hanging off a pothole
    if model in (m.PotholeEvent, m.PassMeasurementRow, m.RepairPlanRow, m.PotholeObservation):
        if p.role == Role.driver_device:
            return false()
        ph = scoped_pothole_ids(p)
        return None if ph is None else model.pothole_id.in_(ph)  # type: ignore[attr-defined]
    if model in (m.Detection, m.Measurement):
        if p.role == Role.driver_device:
            return false()
        ph = scoped_pothole_ids(p)
        if ph is None:
            return None
        det_ids = select(m.PotholeObservation.detection_id).where(m.PotholeObservation.pothole_id.in_(ph))
        return (m.Detection.id.in_(det_ids) if model is m.Detection else m.Measurement.detection_id.in_(det_ids))

    return None if p.is_staff else false()


def scope_query(p: Principal, stmt: Select, model: Optional[type] = None) -> Select:
    """Apply row-level scope for principal ``p`` to ``stmt`` (entity inferred from the first selected column)."""
    if model is None:
        desc = stmt.column_descriptions
        model = desc[0].get("entity") if desc else None
        if model is None:
            raise ValueError("scope_query: cannot infer the entity; pass model=")
    cond = _condition_for(p, model)
    if cond is None:
        return stmt
    return stmt.where(cond)


def ensure_visible(db: Session, p: Principal, model: type, obj_id: Any, detail: Optional[str] = None) -> Any:
    pk = model.__mapper__.primary_key[0]  # type: ignore[attr-defined]
    obj = db.scalars(scope_query(p, select(model).where(pk == obj_id), model)).first()
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=detail or f"{model.__tablename__} {obj_id} not found")
    return obj
