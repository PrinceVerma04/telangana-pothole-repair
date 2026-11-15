"""Defect-liability-period watch (PLAN 7 lifecycle: reopened -> dlp_event if inside DLP).

``watch(db, pothole, now)`` is called when a bus re-detects a pothole that is ``verified_closed`` or
``recurrence_watch`` (WP6 cluster stage / ops tick):

* the pothole moves to ``reopened`` (``reopen_count_12m`` += 1; ``root_cause_flag`` at >= 2 reopenings in 12 months);
* if ``now`` is inside the DLP of the last verified work order - ``contract.dlp_months`` (CRMP packages: 60) or the
  6-month patch warranty default - a ``dlp_event`` is written with an **evidence package** (before/after photos,
  verification record, re-detection observations + frames, latest measurements with sigma, pass log) and a rework
  work order at zero pay is created in status ``planned`` for the engineer to review and issue. This is evidence for
  a debit-note draft, **not** an automatic penalty;
* outside the DLP the pothole is only reopened (returns None).
Does not commit.
"""

from __future__ import annotations

import calendar
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from measure.types import PotholeStatus as PS
from server.app.core import audit
from server.app.core.transitions import transition_pothole
from server.app.core.util import as_utc, now_utc
from server.app.db import models as m
from server.app.schemas.states import WorkOrderStatus as W
from server.app.services.state_machine import create_rework

ACTOR = "system:dlp_watch"
WATCHED = (PS.verified_closed.value, PS.recurrence_watch.value)
PATCH_WARRANTY_MONTHS = 6
ROOT_CAUSE_REOPENINGS = 2
MAX_EVIDENCE_ITEMS = 20


def add_months(dt: datetime, months: int) -> datetime:
    y, mo = divmod(dt.month - 1 + months, 12)
    year, month = dt.year + y, mo + 1
    return dt.replace(year=year, month=month, day=min(dt.day, calendar.monthrange(year, month)[1]))


def last_verified_work_order(db: Session, pothole_id: int) -> Optional[m.WorkOrder]:
    WO, WOP = m.WorkOrder, m.WorkOrderPothole
    return db.scalars(select(WO).join(WOP, WOP.work_order_id == WO.id)
                      .where(WOP.pothole_id == pothole_id, WO.status == W.verified_closed.value)
                      .order_by(WO.updated_at.desc(), WO.id.desc())).first()


def verified_at(db: Session, pothole_id: int) -> Optional[datetime]:
    E = m.PotholeEvent
    return as_utc(db.scalar(select(E.ts).where(E.pothole_id == pothole_id, E.to_status == PS.verified_closed.value)
                            .order_by(E.ts.desc()).limit(1)))


def evidence_package(db: Session, pothole: m.Pothole, wo: Optional[m.WorkOrder], since: Optional[datetime],
                     now: datetime) -> dict[str, Any]:
    ev: dict[str, Any] = {"kind": "dlp_evidence_package", "assembled_at": now.isoformat(), "pothole_id": pothole.id,
                          "lat": pothole.lat, "lon": pothole.lon, "verified_at": since.isoformat() if since else None,
                          "note": "Evidence for a debit-note draft - not an automatic penalty; engineer review required."}
    if wo is not None:
        ev["work_order"] = {"id": wo.id, "contract_id": wo.contract_id, "contractor_id": wo.contractor_id,
                            "bags_used": wo.bags_used, "bag_lot_no": wo.bag_lot_no, "est_bags": wo.est_bags,
                            "issued_at": wo.issued_at.isoformat() if wo.issued_at else None}
        ev["repair_events"] = [{"kind": r.kind, "photo_key": r.photo_key, "at": r.at.isoformat() if r.at else None,
                                "lat": r.lat, "lon": r.lon, "data": r.data}
                               for r in db.scalars(select(m.RepairEvent).where(m.RepairEvent.work_order_id == wo.id)
                                                   .order_by(m.RepairEvent.at))]
        ev["verification"] = [{"clean_passes": v.clean_passes, "distinct_trips": v.distinct_trips,
                               "window_start": v.window_start.isoformat() if v.window_start else None,
                               "window_end": v.window_end.isoformat() if v.window_end else None,
                               "qc_sampled": v.qc_sampled, "qc_passed": v.qc_passed, "outcome": v.outcome}
                              for v in db.scalars(select(m.Verification).where(m.Verification.work_order_id == wo.id,
                                                                               m.Verification.pothole_id == pothole.id))]
    Ob, PO = m.Observation, m.PotholeObservation
    obs_stmt = select(Ob).join(PO, PO.observation_id == Ob.id).where(PO.pothole_id == pothole.id)
    if since is not None:
        obs_stmt = obs_stmt.where(Ob.captured_at >= since)
    ev["redetections"] = [{"observation_id": o.id, "captured_at": o.captured_at.isoformat() if o.captured_at else None,
                           "trip_id": o.trip_id, "frame_key": o.frame_key, "source": o.source}
                          for o in db.scalars(obs_stmt.order_by(Ob.captured_at.desc()).limit(MAX_EVIDENCE_ITEMS))]
    det_ids = select(PO.detection_id).where(PO.pothole_id == pothole.id)
    ev["measurements"] = [{"id": r.id, "depth_p95_mm": r.depth_p95_mm, "depth_sigma_mm": r.depth_sigma_mm,
                           "area_m2": r.area_m2, "area_sigma": r.area_sigma, "volume_l": r.volume_l,
                           "volume_sigma": r.volume_sigma, "method": r.method, "quality": r.quality,
                           "certified": r.certified}
                          for r in db.scalars(select(m.Measurement).where(m.Measurement.detection_id.in_(det_ids))
                                              .order_by(m.Measurement.id.desc()).limit(MAX_EVIDENCE_ITEMS))]
    ev["pass_log"] = {"passes_seen": pothole.passes_seen, "passes_total": pothole.passes_total,
                      "trips_seen": pothole.trips_seen, "last_seen": pothole.last_seen.isoformat()
                      if pothole.last_seen else None}
    return ev


def watch(db: Session, pothole: m.Pothole, now: Optional[datetime] = None) -> Optional[m.DlpEvent]:
    now = as_utc(now) or now_utc()
    if pothole.status not in WATCHED:
        return None
    since = verified_at(db, pothole.id)
    wo = last_verified_work_order(db, pothole.id)
    transition_pothole(db, pothole, PS.reopened, actor=ACTOR, reason="re-detected by bus after a verified repair")
    pothole.reopen_count_12m = (pothole.reopen_count_12m or 0) + 1
    if pothole.reopen_count_12m >= ROOT_CAUSE_REOPENINGS:
        pothole.root_cause_flag = True
    contract_id = (wo.contract_id if wo is not None else None) or pothole.contract_id
    contract = db.get(m.Contract, contract_id) if contract_id else None
    months = int(contract.dlp_months) if contract is not None and contract.dlp_months else PATCH_WARRANTY_MONTHS
    start = since or (as_utc(wo.updated_at) if wo is not None else None)
    if start is None or now > add_months(start, months):
        db.flush()
        return None
    dlp_until = add_months(start, months)
    ev = m.DlpEvent(pothole_id=pothole.id, work_order_id=wo.id if wo else None, contract_id=contract_id,
                    detected_at=now, dlp_until=dlp_until, evidence=evidence_package(db, pothole, wo, start, now))
    db.add(ev)
    db.flush()
    if wo is not None:
        rework = create_rework(db, wo, actor=ACTOR, now=now, issue=False, source="dlp",
                               reason=f"re-detected inside the DLP (until {dlp_until.date().isoformat()})")
        ev.rework_work_order_id = rework.id
    db.add(m.Notification(channel="inapp", to_role="zonal_admin", entity="dlp_event", entity_id=ev.id,
                          subject=f"DLP re-detection: pothole #{pothole.id}",
                          body=(f"Pothole #{pothole.id} re-detected on {now.date().isoformat()} inside the DLP "
                                f"({months} months, until {dlp_until.date().isoformat()}). Evidence package ready; "
                                "rework WO drafted at zero pay - review before issuing."),
                          created_at=now))
    audit.record(db, "dlp_event", ev.id, "create", actor=ACTOR,
                 after={"pothole_id": pothole.id, "work_order_id": ev.work_order_id, "contract_id": contract_id,
                        "dlp_until": dlp_until, "rework_work_order_id": ev.rework_work_order_id},
                 reason="re-detection inside DLP", lat=pothole.lat, lon=pothole.lon)
    db.flush()
    return ev


__all__ = ["add_months", "evidence_package", "last_verified_work_order", "verified_at", "watch"]
