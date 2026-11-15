"""Work-order SLA and the escalation ladder (PLAN 7 "SLA & escalation", PLAN 6 T term).

* SLA hours come from ``priority.sla.sla_hours`` (priority_config ``age_sla`` + ``contract.sla_rules``): arterial
  24 h (GHMC Citizen's Charter 2023), other roads 72 h in the monsoon window / 168 h off-season, R&B 168 h. A work
  order with several potholes takes the strictest (smallest) SLA. The clock starts when the WO is issued.
* The clock stops at the contractor's repair claim (first ``repair_event`` of kind ``bags``, written by the
  ``repaired_claimed`` transition). Breached = (claim time, or now while still open) >= ``sla_due``.
* ``sla_tick(db, now)`` escalates open WOs along ``configs/agencies.yaml`` ``escalation_ladder`` /
  ``escalation_at``: 50 % of SLA -> AE notified; breach -> DEE; breach + 24 h -> EE; breach + 72 h -> Zonal
  Commissioner. Each step (only ever upwards) writes ``work_order.escalation_level``, a ``notification`` row (demo
  adapter: in-app; email/SMS/WhatsApp are stubs) and an ``audit_log`` row. It does not commit.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Iterable, Optional

import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from priority.sla import sla_hours
from server.app import configstore
from server.app.core import audit
from server.app.core.config import CONFIGS_DIR
from server.app.core.util import as_utc, now_utc, to_ist
from server.app.db import models as m
from server.app.schemas.states import WorkOrderStatus as W

AGENCIES_FILE = CONFIGS_DIR / "agencies.yaml"
DEFAULT_LADDER = ("AE", "DEE", "EE", "ZONAL_COMMISSIONER")
DEFAULT_ESCALATION_AT = {"notify_ae_at_sla_frac": 0.5, "dee_at_breach_h": 0.0, "ee_at_breach_h": 24.0,
                         "zc_at_breach_h": 72.0}
# who receives the notification at each ladder step (app roles; the ladder title goes in the subject)
LEVEL_ROLE = {"AE": "field_engineer", "DEE": "zonal_admin", "EE": "zonal_admin", "ZONAL_COMMISSIONER": "zonal_admin"}
CLOCK_RUNNING = (W.issued.value, W.acknowledged.value, W.in_progress.value)
NO_SLA = (W.planned.value, W.cancelled.value)
DEFAULT_SLA_H = 72.0


# =========================================================================== config
@lru_cache(maxsize=4)
def _agencies_yaml(path: str, mtime: float) -> dict[str, Any]:  # mtime keys the cache
    with open(path, encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    return d if isinstance(d, dict) else {}


def escalation_config() -> dict[str, Any]:
    """``{"ladder": [...], <escalation_at keys>}`` from configs/agencies.yaml (defaults if missing)."""
    try:
        d = _agencies_yaml(str(AGENCIES_FILE), AGENCIES_FILE.stat().st_mtime)
    except OSError:
        d = {}
    ladder = [str(x) for x in (d.get("escalation_ladder") or DEFAULT_LADDER)]
    at = {**DEFAULT_ESCALATION_AT, **{k: float(v) for k, v in (d.get("escalation_at") or {}).items()}}
    return {"ladder": ladder, **at}


def level_rank(level: Optional[str], cfg: Optional[dict[str, Any]] = None) -> int:
    ladder = (cfg or escalation_config())["ladder"]
    return ladder.index(level) if level in ladder else -1


def escalation_level(start: datetime, due: datetime, now: datetime,
                     cfg: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Ladder step reached at ``now`` for an SLA running from ``start`` to ``due`` (None = not yet AE)."""
    c = cfg or escalation_config()
    ladder = c["ladder"]
    start, due, now = as_utc(start), as_utc(due), as_utc(now)
    total_h = max(1e-9, (due - start).total_seconds() / 3600.0)
    elapsed_h = (now - start).total_seconds() / 3600.0
    level: Optional[str] = None
    if len(ladder) > 0 and elapsed_h / total_h >= c["notify_ae_at_sla_frac"]:
        level = ladder[0]
    if now >= due:
        over_h = (now - due).total_seconds() / 3600.0
        for idx, key in ((1, "dee_at_breach_h"), (2, "ee_at_breach_h"), (3, "zc_at_breach_h")):
            if len(ladder) > idx and over_h >= c[key]:
                level = ladder[idx]
    return level


# =========================================================================== SLA hours
def _pothole_ids(db: Session, wo: m.WorkOrder) -> list[int]:
    return list(db.scalars(select(m.WorkOrderPothole.pothole_id).where(m.WorkOrderPothole.work_order_id == wo.id)))


def sla_hours_for_work_order(db: Session, wo: m.WorkOrder, at: Optional[datetime] = None) -> float:
    """Strictest SLA (hours) over the WO's potholes, using contract.sla_rules and the agency priority config."""
    at = as_utc(at) or now_utc()
    month = to_ist(at).month  # type: ignore[union-attr]
    contract = db.get(m.Contract, wo.contract_id) if wo.contract_id else None
    rules = (contract.sla_rules or None) if contract is not None else None
    agency = db.get(m.Agency, wo.agency_id) if wo.agency_id else None
    cfg = configstore.active_priority_config(db, wo.agency_id)
    highways: list[Optional[str]] = []
    for pid in _pothole_ids(db, wo):
        ph = db.get(m.Pothole, pid)
        seg = db.get(m.RoadSegment, ph.road_segment_id) if ph is not None and ph.road_segment_id else None
        highways.append(seg.highway if seg is not None else None)
    if not highways:
        highways = [None]
    try:
        return min(sla_hours(h, agency.code if agency else None, month, rules, cfg) for h in highways)
    except (KeyError, TypeError, ValueError):
        return DEFAULT_SLA_H


# =========================================================================== claim time / breach
def claimed_at(db: Session, wo: m.WorkOrder) -> Optional[datetime]:
    """Time the contractor claimed the repair (stops the SLA clock), or None."""
    t = db.scalar(select(func.min(m.RepairEvent.at)).where(m.RepairEvent.work_order_id == wo.id,
                                                          m.RepairEvent.kind == "bags"))
    return as_utc(t)


def claimed_at_many(db: Session, wo_ids: Iterable[int]) -> dict[int, datetime]:
    ids = list(wo_ids)
    if not ids:
        return {}
    rows = db.execute(select(m.RepairEvent.work_order_id, func.min(m.RepairEvent.at))
                      .where(m.RepairEvent.work_order_id.in_(ids), m.RepairEvent.kind == "bags")
                      .group_by(m.RepairEvent.work_order_id)).all()
    return {wid: as_utc(t) for wid, t in rows if t is not None}  # type: ignore[misc]


def is_breached(wo: m.WorkOrder, now: datetime, claimed: Optional[datetime] = None) -> bool:
    """Breached = (claim time, or ``now`` while unclaimed) >= sla_due. Planned/cancelled WOs never breach."""
    if wo.sla_due is None or wo.status in NO_SLA:
        return False
    if claimed is not None:
        end = claimed
    elif wo.status in CLOCK_RUNNING:
        end = now
    else:  # past the claim but no claim event recorded (legacy rows): judge by the last update
        end = as_utc(wo.updated_at) or now
    return as_utc(end) >= as_utc(wo.sla_due)  # type: ignore[operator]


def sla_due_for(db: Session, wo: m.WorkOrder, start: datetime) -> datetime:
    return as_utc(start) + timedelta(hours=sla_hours_for_work_order(db, wo, start))  # type: ignore[operator]


# =========================================================================== tick
def _notify(db: Session, wo: m.WorkOrder, level: str, now: datetime, breached: bool) -> m.Notification:
    due_ist = to_ist(wo.sla_due)
    what = "SLA breached" if breached else "SLA half-time reached"
    n = m.Notification(channel="inapp", to_role=LEVEL_ROLE.get(level, "zonal_admin"),
                       subject=f"[{level}] Work order #{wo.id}: {what}",
                       body=(f"Work order #{wo.id} (status {wo.status}) SLA due {due_ist:%Y-%m-%d %H:%M} IST. "
                             f"Escalated to {level}. Email/SMS/WhatsApp adapters are demo stubs."),
                       entity="work_order", entity_id=wo.id, created_at=now, sent=False)
    db.add(n)
    return n


def sla_tick(db: Session, now: Optional[datetime] = None) -> dict[str, Any]:
    """Escalate open work orders whose SLA clock is running. Idempotent for the same ``now``; never de-escalates."""
    now = as_utc(now) or now_utc()
    cfg = escalation_config()
    W_ = m.WorkOrder
    wos = list(db.scalars(select(W_).where(W_.status.in_(CLOCK_RUNNING)).order_by(W_.id)))
    breached: list[int] = []
    escalated: list[dict[str, Any]] = []
    n_notes = 0
    for wo in wos:
        start = as_utc(wo.issued_at) or as_utc(wo.created_at) or now
        if wo.sla_due is None:
            wo.sla_due = sla_due_for(db, wo, start)
        if as_utc(now) >= as_utc(wo.sla_due):  # type: ignore[operator]
            breached.append(wo.id)
        level = escalation_level(start, wo.sla_due, now, cfg)  # type: ignore[arg-type]
        if level is None or level_rank(level, cfg) <= level_rank(wo.escalation_level, cfg):
            continue
        prev = wo.escalation_level
        wo.escalation_level = level
        _notify(db, wo, level, now, wo.id in breached)
        n_notes += 1
        audit.record(db, "work_order", wo.id, "escalate", actor="system:sla_tick",
                     before={"escalation_level": prev}, after={"escalation_level": level,
                                                               "sla_due": wo.sla_due, "breached": wo.id in breached},
                     reason=f"SLA tick at {now.isoformat()}")
        escalated.append({"work_order_id": wo.id, "from": prev, "to": level})
    db.flush()
    return {"at": now.isoformat(), "checked": len(wos), "sla_breached": breached, "breached": len(breached),
            "escalated": escalated, "notifications": n_notes,
            "escalation_level": {e["work_order_id"]: e["to"] for e in escalated}}


__all__ = ["claimed_at", "claimed_at_many", "escalation_config", "escalation_level", "is_breached", "level_rank",
           "sla_due_for", "sla_hours_for_work_order", "sla_tick"]
