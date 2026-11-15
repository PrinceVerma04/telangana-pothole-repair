"""Work-order state machine (frozen edges in ``server/app/schemas/states.py``; PLAN 7 lifecycle).

``transition_work_order(db, wo, to, actor=...)``:
1. RBAC: ``WO_TRANSITION_ROLES`` (super_admin may do all); a contractor acts only on its own work orders and a
   contractor can never verify (verification_pending / verified_closed / qc_failed) its own repair
   -> ``TransitionForbidden``. ``actor`` None / str = system (trusted caller, e.g. ``verification.evaluate``).
2. Legality: ``WORK_ORDER_TRANSITIONS`` -> ``server.app.core.transitions.IllegalTransition`` (HTTP 409).
3. Evidence: issued needs a contractor; in_progress needs a before photo; repaired_claimed needs an after photo
   and ``bags_used`` -> ``PreconditionFailed`` (HTTP 422).
4. Side effects: issued_at + sla_due on issue; check-in event (geo + time, bag lot) on in_progress; ``bags`` event
   (stops the SLA clock, bag variance > 25 % flagged) on repaired_claimed; potholes mirrored through
   ``WO_TO_POTHOLE`` with ``transition_pothole`` (shortest legal path, e.g. confirmed -> planned -> work_ordered);
   verification rows on verified_closed / qc_failed; qc_failed -> rework WO (``is_rework``, ``parent_work_order_id``,
   zero pay, issued to the same contractor). Every legal transition writes ``audit_log``.
Nothing here commits.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from measure.types import PotholeStatus as PS
from repair.types import PlanContext
from server.app.core import audit
from server.app.core.transitions import IllegalTransition, transition_pothole
from server.app.core.util import as_utc, now_utc
from server.app.db import models as m
from server.app.schemas.api import WorkOrderOut
from server.app.schemas.states import (
    POTHOLE_TRANSITIONS,
    WO_TO_POTHOLE,
    WO_TRANSITION_ROLES,
    Role,
    WorkOrderStatus,
    work_order_can,
)
from server.app.services import runs, sla

W = WorkOrderStatus
SYSTEM = "system"
TERMINAL = frozenset({W.verified_closed.value, W.qc_failed.value, W.cancelled.value})
VERIFY_TARGETS = frozenset({W.verification_pending, W.verified_closed, W.qc_failed})
WORK_ORDERABLE = frozenset({PS.confirmed.value, PS.planned.value, PS.reopened.value, PS.qc_failed.value})
BAG_VARIANCE_FRAC = 0.25
# intermediate pothole states never used when mirroring a WO move (they mean "not a real pothole")
_NO_PASS_THROUGH = frozenset({PS.rejected, PS.archived, PS.candidate})

__all__ = ["IllegalTransition", "PreconditionFailed", "SYSTEM", "TransitionForbidden", "create_rework",
           "create_work_order", "mirror_pothole", "pothole_ids_of", "transition_work_order", "upsert_verification",
           "work_order_out", "work_orders_out"]


class TransitionForbidden(PermissionError):
    pass


class PreconditionFailed(ValueError):
    pass


# =========================================================================== helpers
def _is_system(actor: Any) -> bool:
    return actor is None or isinstance(actor, (str, int))


def _role(actor: Any) -> Optional[Role]:
    r = getattr(actor, "role", None)
    return Role(r) if r is not None else None


def _actor_user_id(actor: Any) -> Optional[int]:
    return getattr(actor, "user_id", None) if not _is_system(actor) else (actor if isinstance(actor, int) else None)


def pothole_ids_of(db: Session, wo: m.WorkOrder) -> list[int]:
    return list(db.scalars(select(m.WorkOrderPothole.pothole_id).where(m.WorkOrderPothole.work_order_id == wo.id)
                           .order_by(m.WorkOrderPothole.pothole_id)))


def _has_event(db: Session, wo: m.WorkOrder, kind: str) -> bool:
    return db.scalars(select(m.RepairEvent.id).where(m.RepairEvent.work_order_id == wo.id,
                                                     m.RepairEvent.kind == kind).limit(1)).first() is not None


def check_actor(wo: m.WorkOrder, to: WorkOrderStatus, actor: Any) -> None:
    if _is_system(actor):
        return
    role = _role(actor)
    if role is None:
        raise TransitionForbidden("unknown actor role")
    if role != Role.super_admin and role not in WO_TRANSITION_ROLES.get(to, set()):
        allowed = sorted(r.value for r in WO_TRANSITION_ROLES.get(to, set()))
        raise TransitionForbidden(f"role {role.value} may not move a work order to {to.value}; needs one of {allowed}")
    actor_contractor = getattr(actor, "contractor_id", None)
    if role == Role.contractor and (actor_contractor is None or wo.contractor_id != actor_contractor):
        raise TransitionForbidden(f"work order {wo.id} belongs to another contractor")
    if to in VERIFY_TARGETS and actor_contractor is not None and actor_contractor == wo.contractor_id:
        raise TransitionForbidden("a contractor cannot verify its own repair")


def _pothole_path(frm: PS, to: PS, max_len: int = 4) -> Optional[list[PS]]:
    """Shortest legal path frm -> to in POTHOLE_TRANSITIONS (excluding frm), avoiding FP/archive states."""
    if frm == to:
        return []
    prev: dict[PS, PS] = {}
    q: deque[tuple[PS, int]] = deque([(frm, 0)])
    seen = {frm}
    while q:
        cur, d = q.popleft()
        if d >= max_len:
            continue
        for nxt in sorted(POTHOLE_TRANSITIONS.get(cur, set()), key=lambda s: s.value):
            if nxt in seen or (nxt in _NO_PASS_THROUGH and nxt != to):
                continue
            prev[nxt] = cur
            if nxt == to:
                path = [nxt]
                while path[-1] in prev and prev[path[-1]] != frm:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            seen.add(nxt)
            q.append((nxt, d + 1))
    return None


def mirror_pothole(db: Session, ph: m.Pothole, target: PS, *, actor: Any, reason: str,
                   lat: Optional[float] = None, lon: Optional[float] = None) -> None:
    path = _pothole_path(PS(ph.status), target)
    if path is None:
        raise IllegalTransition(f"pothole {ph.id}: cannot follow the work order from {ph.status} to {target.value}")
    for st in path:
        transition_pothole(db, ph, st, actor=actor, reason=reason, lat=lat, lon=lon)


def upsert_verification(db: Session, wo: m.WorkOrder, pothole_id: int, **fields: Any) -> m.Verification:
    V = m.Verification
    row = db.scalars(select(V).where(V.work_order_id == wo.id, V.pothole_id == pothole_id)
                     .order_by(V.id.desc())).first()
    if row is None:
        row = V(work_order_id=wo.id, pothole_id=pothole_id)
        db.add(row)
    for k, v in fields.items():
        setattr(row, k, v)
    return row


# =========================================================================== transition
def transition_work_order(db: Session, wo: m.WorkOrder, to: WorkOrderStatus | str, *, actor: Any,
                          reason: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None,
                          bags_used: Optional[int] = None, bag_lot_no: Optional[str] = None,
                          now: Optional[datetime] = None) -> m.WorkOrder:
    to = W(to)
    frm = W(wo.status)
    now = as_utc(now) or now_utc()
    check_actor(wo, to, actor)
    if not work_order_can(frm, to):
        raise IllegalTransition(f"work order {wo.id}: {frm.value} -> {to.value} not allowed")

    # ---- evidence preconditions (PLAN 7 lifecycle)
    if to == W.issued and wo.contractor_id is None:
        raise PreconditionFailed("assign a contractor (or a contract with a contractor) before issuing")
    if to == W.in_progress and not _has_event(db, wo, "before_photo"):
        raise PreconditionFailed("upload the before photo (POST /work-orders/{id}/photos kind=before) first")
    if to == W.repaired_claimed:
        if bags_used is None or bags_used < 0:
            raise PreconditionFailed("bags_used (>= 0) is required to claim a repair")
        if not _has_event(db, wo, "after_photo"):
            raise PreconditionFailed("upload the after photo (POST /work-orders/{id}/photos kind=after) first")

    before = {"status": frm.value, "bags_used": wo.bags_used, "bag_lot_no": wo.bag_lot_no, "sla_due": wo.sla_due,
              "issued_at": wo.issued_at}
    uid = _actor_user_id(actor)
    wo.status = to.value
    wo.updated_at = now
    if bag_lot_no:
        wo.bag_lot_no = bag_lot_no
    variance = False
    if to == W.issued:
        wo.issued_at = now
        if wo.sla_due is None:
            wo.sla_due = sla.sla_due_for(db, wo, now)
    elif to == W.in_progress:
        db.add(m.RepairEvent(work_order_id=wo.id, kind="check_in", lat=lat, lon=lon, at=now, actor_id=uid,
                             data={"bag_lot_no": bag_lot_no}))
    elif to == W.repaired_claimed:
        wo.bags_used = int(bags_used)  # type: ignore[arg-type]
        variance = bool(wo.est_bags and abs(wo.bags_used - wo.est_bags) > BAG_VARIANCE_FRAC * wo.est_bags)
        db.add(m.RepairEvent(work_order_id=wo.id, kind="bags", lat=lat, lon=lon, at=now, actor_id=uid,
                             data={"bags_used": wo.bags_used, "bag_lot_no": wo.bag_lot_no, "est_bags": wo.est_bags,
                                   "bag_variance_flag": variance}))

    # ---- mirror the potholes
    why = reason or f"work order #{wo.id}: {to.value}"
    target = WO_TO_POTHOLE.get(to)
    potholes = [p for p in (db.get(m.Pothole, pid) for pid in pothole_ids_of(db, wo)) if p is not None]
    for ph in potholes:
        if target is not None:
            mirror_pothole(db, ph, target, actor=actor, reason=why, lat=lat, lon=lon)
        elif to == W.cancelled and ph.status in (PS.work_ordered.value, PS.acknowledged.value, PS.qc_failed.value):
            transition_pothole(db, ph, PS.planned, actor=actor, reason=why, lat=lat, lon=lon)

    # ---- manual verification decisions are recorded like automatic ones
    if to in (W.verified_closed, W.qc_failed) and not _is_system(actor):
        for ph in potholes:
            upsert_verification(db, wo, ph.id, outcome=to.value, qc_sampled=True,
                                 qc_passed=to == W.verified_closed, window_end=now)

    audit.record(db, "work_order", wo.id, f"status:{frm.value}->{to.value}", actor=actor, before=before,
                 after={"status": to.value, "bags_used": wo.bags_used, "bag_lot_no": wo.bag_lot_no,
                        "sla_due": wo.sla_due, "issued_at": wo.issued_at, "bag_variance_flag": variance,
                        "pothole_ids": [p.id for p in potholes]},
                 reason=reason, lat=lat, lon=lon)
    db.flush()
    if to == W.qc_failed:
        create_rework(db, wo, actor=actor, reason=reason or "QC failed", now=now)
    return wo


# =========================================================================== create
def _active_wo_for(db: Session, pothole_id: int) -> Optional[int]:
    WO, WOP = m.WorkOrder, m.WorkOrderPothole
    return db.scalars(select(WO.id).join(WOP, WOP.work_order_id == WO.id)
                      .where(WOP.pothole_id == pothole_id, WO.status.notin_(TERMINAL)).limit(1)).first()


def create_work_order(db: Session, pothole_ids: Iterable[int], *, actor: Any, contract_id: Optional[int] = None,
                      contractor_id: Optional[int] = None, crew: Optional[str] = None,
                      sla_due: Optional[datetime] = None, run_id: Optional[int] = None, notes: Optional[str] = None,
                      issue: bool = False, context: Optional[PlanContext] = None,
                      now: Optional[datetime] = None) -> m.WorkOrder:
    """New WO (status planned; ``issue=True`` also issues it). Plans potholes that have no current plan.

    Raises LookupError (unknown pothole/contract/contractor), IllegalTransition (pothole not work-orderable or
    already on an open WO), PreconditionFailed (contractor does not hold the contract / issue without contractor).
    """
    now = as_utc(now) or now_utc()
    ids = list(dict.fromkeys(int(i) for i in pothole_ids))
    if not ids:
        raise PreconditionFailed("pothole_ids must not be empty")
    potholes = []
    for pid in ids:
        ph = db.get(m.Pothole, pid)
        if ph is None:
            raise LookupError(f"pothole {pid} not found")
        if ph.status not in WORK_ORDERABLE:
            raise IllegalTransition(f"pothole {pid} is {ph.status}; only {sorted(WORK_ORDERABLE)} can be work-ordered")
        if (other := _active_wo_for(db, pid)) is not None:
            raise IllegalTransition(f"pothole {pid} is already on open work order {other}")
        potholes.append(ph)

    contract = None
    cid = contract_id or next((p.contract_id for p in potholes if p.contract_id), None)
    if cid is not None:
        contract = db.get(m.Contract, cid)
        if contract is None:
            raise LookupError(f"contract {cid} not found")
    if contractor_id is not None:
        if db.get(m.Contractor, contractor_id) is None:
            raise LookupError(f"contractor {contractor_id} not found")
        if contract is not None and contract.contractor_id not in (None, contractor_id):
            if contract_id is not None:
                raise PreconditionFailed(f"contractor {contractor_id} does not hold contract {contract.code}")
            contract = None  # the pothole's default contract belongs to someone else; explicit contractor wins
    ctr = contractor_id or (contract.contractor_id if contract else None) or \
        next((p.contractor_id for p in potholes if p.contractor_id), None)

    plan_rows: dict[int, Optional[m.RepairPlanRow]] = {}
    for ph in potholes:
        row = runs.current_plan(db, ph.id)
        if row is None:
            try:
                row, _ = runs.plan_pothole(db, ph, context, actor=actor, now=now)
            except ValueError:
                row = None  # no footprint yet: the WO can still be issued; bags unknown
        elif ph.status in (PS.confirmed.value, PS.reopened.value):
            transition_pothole(db, ph, PS.planned, actor=actor, reason=f"repair plan #{row.id}")
        plan_rows[ph.id] = row
    plans = [runs.plan_of(r) for r in plan_rows.values() if r is not None]
    plans = [p for p in plans if p is not None]

    wo = m.WorkOrder(status=W.planned.value, agency_id=potholes[0].agency_id or getattr(actor, "agency_id", None),
                     contract_id=contract.id if contract else None, contractor_id=ctr, crew=crew, run_id=run_id,
                     sla_due=as_utc(sla_due), est_bags=sum(p.bags for p in plans) if plans else None,
                     est_cost_inr=round(sum(p.est_cost_inr for p in plans), 2) if plans else None, notes=notes,
                     created_at=now, updated_at=now)
    db.add(wo)
    db.flush()
    for ph in potholes:
        r = plan_rows[ph.id]
        db.add(m.WorkOrderPothole(work_order_id=wo.id, pothole_id=ph.id, repair_plan_id=r.id if r else None))
    db.flush()
    audit.record(db, "work_order", wo.id, "create", actor=actor,
                 after={"status": wo.status, "pothole_ids": ids, "contract_id": wo.contract_id,
                        "contractor_id": wo.contractor_id, "est_bags": wo.est_bags,
                        "bags_range": [sum(p.bags_low for p in plans), sum(p.bags_high for p in plans)] if plans
                        else None, "est_cost_inr": wo.est_cost_inr},
                 reason=notes, lat=potholes[0].lat, lon=potholes[0].lon)
    if issue:
        transition_work_order(db, wo, W.issued, actor=actor, reason="issued on creation", now=now)
    return wo


def create_rework(db: Session, parent: m.WorkOrder, *, actor: Any, reason: str, now: Optional[datetime] = None,
                  issue: bool = True, source: str = "qc_failed") -> m.WorkOrder:
    """Rework WO at zero pay (contractor's cost) for the same potholes / contract; issued unless ``issue=False``."""
    now = as_utc(now) or now_utc()
    wo = m.WorkOrder(status=W.planned.value, agency_id=parent.agency_id, contract_id=parent.contract_id,
                     contractor_id=parent.contractor_id, crew=parent.crew, est_bags=parent.est_bags, est_cost_inr=0.0,
                     is_rework=True, parent_work_order_id=parent.id, created_at=now, updated_at=now,
                     notes=f"rework of WO #{parent.id} ({source}): {reason} - at the contractor's cost (zero pay)")
    db.add(wo)
    db.flush()
    for wop in db.scalars(select(m.WorkOrderPothole).where(m.WorkOrderPothole.work_order_id == parent.id)):
        db.add(m.WorkOrderPothole(work_order_id=wo.id, pothole_id=wop.pothole_id, repair_plan_id=wop.repair_plan_id))
    db.flush()
    audit.record(db, "work_order", wo.id, "create_rework", actor=actor,
                 after={"status": wo.status, "parent_work_order_id": parent.id, "source": source,
                        "pothole_ids": pothole_ids_of(db, wo), "est_cost_inr": 0.0}, reason=reason)
    if issue and wo.contractor_id is not None:
        transition_work_order(db, wo, W.issued, actor=f"{SYSTEM}:rework", reason=f"rework of WO #{parent.id}",
                              now=now)
    return wo


# =========================================================================== DTO
def _signed(storage: Any, key: Optional[str]) -> Optional[str]:
    if storage is None or not key:
        return None
    try:
        return storage.url(key)
    except Exception:
        return None


def work_orders_out(db: Session, wos: list[m.WorkOrder], storage: Any = None,
                    now: Optional[datetime] = None) -> list[WorkOrderOut]:
    """Batch DTO builder (one query per related table, not per row)."""
    now = as_utc(now) or now_utc()
    ids = [w.id for w in wos]
    if not ids:
        return []
    pids: dict[int, list[int]] = {i: [] for i in ids}
    for wid, pid in db.execute(select(m.WorkOrderPothole.work_order_id, m.WorkOrderPothole.pothole_id)
                               .where(m.WorkOrderPothole.work_order_id.in_(ids))
                               .order_by(m.WorkOrderPothole.pothole_id)):
        pids[wid].append(pid)
    photos: dict[int, list[dict[str, Any]]] = {i: [] for i in ids}
    for ev in db.scalars(select(m.RepairEvent).where(m.RepairEvent.work_order_id.in_(ids),
                                                     m.RepairEvent.kind.in_(("before_photo", "after_photo")))
                         .order_by(m.RepairEvent.at, m.RepairEvent.id)):
        photos[ev.work_order_id].append({"id": ev.id, "kind": ev.kind.replace("_photo", ""),
                                         "url": _signed(storage, ev.photo_key),
                                         "at": ev.at.isoformat() if ev.at else None, "lat": ev.lat, "lon": ev.lon})
    names = dict(db.execute(select(m.Contractor.id, m.Contractor.name)
                            .where(m.Contractor.id.in_(sorted({w.contractor_id for w in wos if w.contractor_id}))))
                 .all())
    claims = sla.claimed_at_many(db, ids)
    out = []
    for wo in wos:
        variance = bool(wo.bags_used is not None and wo.est_bags and
                        abs(wo.bags_used - wo.est_bags) > BAG_VARIANCE_FRAC * wo.est_bags)
        out.append(WorkOrderOut(
            id=wo.id, status=W(wo.status), pothole_ids=pids[wo.id], agency_id=wo.agency_id,
            contract_id=wo.contract_id, contractor_id=wo.contractor_id, contractor_name=names.get(wo.contractor_id),
            crew=wo.crew, issued_at=wo.issued_at, sla_due=wo.sla_due,
            sla_breached=sla.is_breached(wo, now, claims.get(wo.id)), escalation_level=wo.escalation_level,
            est_bags=wo.est_bags, est_cost_inr=wo.est_cost_inr, bags_used=wo.bags_used, bag_variance_flag=variance,
            is_rework=bool(wo.is_rework), parent_work_order_id=wo.parent_work_order_id, photos=photos[wo.id],
            run_id=wo.run_id))
    return out


def work_order_out(db: Session, wo: m.WorkOrder, storage: Any = None, now: Optional[datetime] = None) -> WorkOrderOut:
    return work_orders_out(db, [wo], storage, now)[0]
