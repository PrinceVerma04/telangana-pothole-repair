"""KPIs per admin unit (``KpiOut``) and the contractor scorecard (``ContractorScorecard``).

``kpi(db, admin_unit_id, window_days, now)`` - ``admin_unit_id=None`` means the whole network. Potholes belong to a
unit through ``pothole.admin_unit_id`` (unit + all its descendants). Window counts use ``pothole_event`` times.
``bus_km_scanned`` sums the hops between consecutive camera-active trip fixes inside the window (hops > 500 m are
GPS gaps and skipped), restricted to the unit's bounding box when it has a geometry.
``baseline_cost_inr`` = open potholes x the rate card's ``baseline.cost_per_pothole_inr`` (GHMC 2021: INR 7,396).

``scorecard(db, contractor_id, now)``: SLA met % (claim before due, over WOs whose SLA outcome is known), verified /
QC-failed / DLP counts, reopen rate (reopened potholes / verified potholes), median days issue -> claim, and a
Kaplan-Meier survival curve of verified patches (event = reopened; censored at ``now``).
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from measure.types import PotholeStatus as PS
from server.app import configstore
from server.app.core.rbac import admin_subtree_ids
from server.app.core.util import as_utc, haversine_m, now_utc
from server.app.db import models as m
from server.app.schemas.api import ContractorScorecard, KpiOut
from server.app.schemas.states import WorkOrderStatus as W
from server.app.services import sla
from server.app.services.runs import current_plan, plan_of

OPEN_STATUSES = (PS.confirmed.value, PS.planned.value, PS.work_ordered.value, PS.acknowledged.value,
                 PS.in_progress.value, PS.repaired_claimed.value, PS.verification_pending.value, PS.qc_failed.value,
                 PS.reopened.value)
MAX_HOP_M = 500.0
MAX_FIX_ROWS = 1_000_000
SURVIVAL_DAYS = (0, 15, 30, 45, 60, 90, 120, 150, 180, 270, 365)
DEFAULT_BASELINE_INR = 7396.0


def _median(xs: list[float]) -> Optional[float]:
    return round(statistics.median(xs), 2) if xs else None


def _bus_km(db: Session, since: datetime, until: datetime, bbox: Optional[tuple[float, float, float, float]]) -> float:
    F = m.TripFix
    stmt = select(F.trip_id, F.ts, F.lat, F.lon).where(F.ts >= since, F.ts <= until, F.cam_active.is_(True))
    if bbox is not None:
        stmt = stmt.where(F.lon.between(bbox[0], bbox[2]), F.lat.between(bbox[1], bbox[3]))
    total, last = 0.0, None
    for trip_id, _ts, lat, lon in db.execute(stmt.order_by(F.trip_id, F.ts).limit(MAX_FIX_ROWS)):
        if last is not None and last[0] == trip_id:
            d = haversine_m(last[1], last[2], lat, lon)
            if d <= MAX_HOP_M:
                total += d
        last = (trip_id, lat, lon)
    return round(total / 1000.0, 2)


def _event_times(db: Session, pothole_ids: Any, status: str, since: datetime) -> dict[int, datetime]:
    E = m.PotholeEvent
    rows = db.execute(select(E.pothole_id, func.min(E.ts)).where(E.pothole_id.in_(pothole_ids), E.to_status == status,
                                                                 E.ts >= since).group_by(E.pothole_id)).all()
    return {pid: as_utc(t) for pid, t in rows}  # type: ignore[misc]


def kpi(db: Session, admin_unit_id: Optional[int], window_days: int, now: Optional[datetime] = None) -> dict[str, Any]:
    now = as_utc(now) or now_utc()
    since = now - timedelta(days=int(window_days))
    P = m.Pothole
    unit = db.get(m.AdminUnit, admin_unit_id) if admin_unit_id is not None else None
    if admin_unit_id is not None and unit is None:
        raise LookupError(f"admin unit {admin_unit_id} not found")
    in_unit = select(P.id)
    if unit is not None:
        in_unit = in_unit.where(P.admin_unit_id.in_(admin_subtree_ids(unit.id)))
    rows = db.execute(select(P.id, P.status, P.severity, P.confirmed_at, P.agency_id).where(P.id.in_(in_unit))).all()
    by_status: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    open_ids: list[int] = []
    for pid, status, severity, _c, _a in rows:
        by_status[status] = by_status.get(status, 0) + 1
        if status in OPEN_STATUSES:
            open_ids.append(pid)
            by_severity[severity or "unknown"] = by_severity.get(severity or "unknown", 0) + 1
    confirmed_at = {pid: as_utc(c) for pid, _s, _v, c, _a in rows}

    confirmed_new = _event_times(db, in_unit, PS.confirmed.value, since)
    verified = _event_times(db, in_unit, PS.verified_closed.value, since)
    reopened = _event_times(db, in_unit, PS.reopened.value, since)
    claimed = _event_times(db, in_unit, PS.repaired_claimed.value, since)
    hours = [(t - confirmed_at[pid]).total_seconds() / 3600.0 for pid, t in claimed.items()
             if confirmed_at.get(pid) is not None and t >= confirmed_at[pid]]  # type: ignore[operator]

    WO, WOP = m.WorkOrder, m.WorkOrderPothole
    wos = list(db.scalars(select(WO).where(WO.id.in_(select(WOP.work_order_id).where(WOP.pothole_id.in_(in_unit))),
                                           WO.sla_due.is_not(None), WO.sla_due >= since, WO.sla_due <= now)))
    claims = sla.claimed_at_many(db, [w.id for w in wos])
    breaches = sum(1 for w in wos if sla.is_breached(w, now, claims.get(w.id)))

    est = 0.0
    for pid in open_ids:
        plan = plan_of(current_plan(db, pid))
        est += plan.est_cost_inr if plan else 0.0
    card = configstore.active_rate_card(db, unit.agency_id if unit is not None else None)
    baseline = float((card.get("baseline") or {}).get("cost_per_pothole_inr", DEFAULT_BASELINE_INR))
    bbox = None
    if unit is not None and unit.min_lon is not None:
        bbox = (unit.min_lon, unit.min_lat, unit.max_lon, unit.max_lat)
    out = KpiOut(admin_unit_id=unit.id if unit else None, admin_unit=unit.name_en if unit else "All",
                 window_days=int(window_days), open_potholes=len(open_ids), confirmed_new=len(confirmed_new),
                 repaired_verified=len(verified), median_hours_to_repair=_median(hours), sla_breaches=breaches,
                 reopened=len(reopened), bus_km_scanned=_bus_km(db, since, now, bbox), est_cost_inr=round(est, 2),
                 baseline_cost_inr=round(baseline * len(open_ids), 2), by_severity=by_severity, by_status=by_status)
    return out.model_dump(mode="json")


# =========================================================================== scorecard
def kaplan_meier(durations: list[tuple[float, bool]], at_days: tuple[int, ...] = SURVIVAL_DAYS) -> list[dict[str, float]]:
    """``durations``: (days, event_observed). Returns S(t) at ``at_days`` (product-limit estimator)."""
    if not durations:
        return []
    times = sorted({d for d, e in durations if e})
    s, curve, steps = 1.0, [], []
    for t in times:
        at_risk = sum(1 for d, _ in durations if d >= t)
        events = sum(1 for d, e in durations if e and d == t)
        if at_risk:
            s *= 1.0 - events / at_risk
        steps.append((t, s))
    for day in at_days:
        val = 1.0
        for t, sv in steps:
            if t <= day:
                val = sv
        curve.append({"days": float(day), "survival": round(val, 4)})
    return curve


def scorecard(db: Session, contractor_id: int, now: Optional[datetime] = None) -> dict[str, Any]:
    now = as_utc(now) or now_utc()
    ctr = db.get(m.Contractor, contractor_id)
    if ctr is None:
        raise LookupError(f"contractor {contractor_id} not found")
    WO, WOP = m.WorkOrder, m.WorkOrderPothole
    wos = list(db.scalars(select(WO).where(WO.contractor_id == contractor_id, WO.status != W.cancelled.value)))
    claims = sla.claimed_at_many(db, [w.id for w in wos])
    judged = [w for w in wos if w.sla_due is not None and (w.id in claims or as_utc(now) >= as_utc(w.sla_due))]
    met = sum(1 for w in judged if not sla.is_breached(w, now, claims.get(w.id)))
    fix_days = [(claims[w.id] - as_utc(w.issued_at)).total_seconds() / 86400.0  # type: ignore[operator]
                for w in wos if w.id in claims and w.issued_at is not None]
    n_dlp = db.scalar(select(func.count(m.DlpEvent.id)).join(WO, WO.id == m.DlpEvent.work_order_id)
                      .where(WO.contractor_id == contractor_id)) or 0

    # patch survival: each verified (WO, pothole) from the verification time until reopened / now
    E = m.PotholeEvent
    verified_pairs = db.execute(select(WOP.pothole_id).join(WO, WO.id == WOP.work_order_id)
                                .where(WO.contractor_id == contractor_id, WO.status == W.verified_closed.value)).all()
    durations: list[tuple[float, bool]] = []
    reopened_n = 0
    for (pid,) in verified_pairs:
        t0 = as_utc(db.scalar(select(func.min(E.ts)).where(E.pothole_id == pid, E.to_status == PS.verified_closed.value)))
        if t0 is None:
            continue
        t1 = as_utc(db.scalar(select(func.min(E.ts)).where(E.pothole_id == pid, E.to_status == PS.reopened.value,
                                                           E.ts >= t0)))
        if t1 is not None:
            reopened_n += 1
        end = t1 or now
        durations.append((max(0.0, (end - t0).total_seconds() / 86400.0), t1 is not None))
    out = ContractorScorecard(
        contractor_id=ctr.id, name=ctr.name, work_orders=len(wos),
        sla_met_pct=round(100.0 * met / len(judged), 1) if judged else None,
        verified_closed=sum(1 for w in wos if w.status == W.verified_closed.value),
        qc_failed=sum(1 for w in wos if w.status == W.qc_failed.value), dlp_events=int(n_dlp),
        reopen_rate=round(reopened_n / len(durations), 4) if durations else None,
        median_days_to_fix=_median(fix_days), survival=kaplan_meier(durations))
    return out.model_dump(mode="json")


__all__ = ["OPEN_STATUSES", "kaplan_meier", "kpi", "scorecard"]
