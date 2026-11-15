"""Ops ticks (WP6 ``tasks_ops`` without Celery, DECISIONS D2). Each tick calls a WP8b service in ``server/app/services``.

    sla_tick           services.sla.sla_tick(db, now)                         sla_breached + escalation ladder
    verify_repairs     services.verification.evaluate(db, wo, now, clean_passes) for WOs in repaired_claimed /
                       verification_pending (PLAN 5.7: >= 3 clean passes, >= 2 trips, <= 14 d, QC rule)
    dlp_watch          services.dlp.watch(db, pothole, now) for closed potholes re-detected after their last event
    daily_indent       services.runs.generate(db, circle_id, run_date, actor=...) per circle with issued WOs not in a run
    kpi_rollup         services.kpi.kpi(...) per zone + services.kpi.scorecard(...) per contractor -> JSON snapshot
                       in storage (``kpi/rollup/<IST date>.json``)
    archive_candidates candidate with no 2nd pass in 30 d and >= 5 clean passes -> archived (PLAN 5.4); no service

``run_ticks(db, now, names)`` runs them in order, each in its own SAVEPOINT; an exception (including a missing
service module -- there is no fallback) is reported as ``{"error": ...}``, a tick with nothing to do as a normal
result (``{"evaluated": 0}`` etc.).
``clean_passes(db, pothole, since)`` is the callable handed to ``services.verification.evaluate``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from measure.types import PotholeStatus
from server.app.core.transitions import transition_pothole
from server.app.core.util import as_utc, haversine_m, now_utc, radius_box
from server.app.db import models as m
from server.workers import adapters
from server.workers.stages import (
    ACTOR,
    EPS_M,
    PASS_EPS_M,
    POTHOLE_CLASSES,
    _geo,
    _ist_day,
    _storage,
    service,
)

log = logging.getLogger("gunthameter.workers.ops")

OPS_ACTOR = "system:ops"
TICKS: tuple[str, ...] = ("sla_tick", "verify_repairs", "dlp_watch", "daily_indent", "kpi_rollup",
                          "archive_candidates")
VERIFY_WO_STATUSES = ("repaired_claimed", "verification_pending")
ARCHIVE_AFTER_D = 30
ARCHIVE_MIN_CLEAN = 5


class TickSkipped(Exception):
    """Raised by a tick that decides not to run (reported as ``{"skipped": ...}``)."""


# =========================================================================== clean passes (verification input)
def redetections(db: Session, p: m.Pothole, since: datetime,
                 trips: Optional[Iterable[Any]] = None) -> list[tuple[int, datetime]]:
    """Verified pothole detections within eps of ``p`` after ``since`` as ``[(trip_id, captured_at)]`` (one per trip).

    Also the ``redetections`` provider of ``services.verification.evaluate`` (re-detected -> qc_failed)."""
    if p.lat is None or p.lon is None:
        return []
    Ob, D = m.Observation, m.Detection
    x1, y1, x2, y2 = radius_box(p.lat, p.lon, PASS_EPS_M)
    lat_c = func.coalesce(Ob.lat_ground, Ob.lat_raw)
    lon_c = func.coalesce(Ob.lon_ground, Ob.lon_raw)
    q = (select(Ob.trip_id, Ob.captured_at, lat_c, lon_c).join(D, D.observation_id == Ob.id)
         .where(Ob.captured_at >= as_utc(since), Ob.status == "verified", Ob.trip_id.is_not(None),
                D.verified.is_(True), D.cls.in_(POTHOLE_CLASSES), lat_c.between(y1, y2), lon_c.between(x1, x2))
         .order_by(Ob.captured_at))
    if trips is not None:
        ids = [int(t) for t in trips if str(t).isdigit()]
        if not ids:
            return []
        q = q.where(Ob.trip_id.in_(ids))
    out: dict[int, datetime] = {}
    for trip_id, ts, lat, lon in db.execute(q).all():
        if haversine_m(p.lat, p.lon, lat, lon) <= max(EPS_M, PASS_EPS_M):
            out.setdefault(int(trip_id), as_utc(ts))
    return sorted(out.items(), key=lambda x: x[1])


def clean_passes(db: Session, p: m.Pothole, since: datetime) -> list[tuple[Any, datetime]]:
    """Camera-on passes over ``p`` since ``since`` with no re-detection (PLAN 5.7). ``[(trip_id, ts)]``, one entry per
    pass: a trip that drives past twice (a looping route, > 2 min apart) contributes two passes, so ">= 3 clean passes
    from >= 2 trips" can be met by two trips. A trip that re-detected the pothole contributes none.

    Uses the trip fixes' stored way/direction (written by the trip-close hook) with ``geo.passes.count_passes``.
    """
    if p.lat is None or p.lon is None:
        return []
    since = as_utc(since)
    x1, y1, x2, y2 = radius_box(p.lat, p.lon, PASS_EPS_M * 2)
    F = m.TripFix
    fixes = list(db.scalars(select(F).where(F.ts >= since, F.lat.between(y1, y2), F.lon.between(x1, x2))
                            .order_by(F.trip_id, F.ts)))
    if not fixes:
        return []
    passes_mod = _geo("geo.passes")
    pts = [adapters.fix_point(f) for f in fixes]
    matched = [adapters.match_result(f) for f in fixes]
    got = [(t, as_utc(ts)) for t, ts in passes_mod.count_passes(adapters.pothole_point(p), pts, matched,
                                                                 since, eps_m=PASS_EPS_M)]
    dirty = {str(t) for t, _ in redetections(db, p, since, {t for t, _ in got})}
    return [(t, ts) for t, ts in got if str(t) not in dirty]


def clean_passes_fn(db: Session) -> Callable[[m.Pothole, datetime], list[tuple[Any, datetime]]]:
    return lambda p, since: clean_passes(db, p, since)


# =========================================================================== ticks
def sla_tick(db: Session, now: datetime) -> Any:
    return service("sla").sla_tick(db, now)


def verify_repairs(db: Session, now: datetime, pothole_ids: Optional[Iterable[int]] = None) -> dict[str, Any]:
    WO, WOP = m.WorkOrder, m.WorkOrderPothole
    q = select(WO).where(WO.status.in_(VERIFY_WO_STATUSES)).order_by(WO.id)
    if pothole_ids is not None:
        ids = list(pothole_ids)
        if not ids:
            return {"evaluated": 0}
        q = q.where(WO.id.in_(select(WOP.work_order_id).where(WOP.pothole_id.in_(ids))))
    wos = list(db.scalars(q))
    if not wos:
        return {"evaluated": 0}
    svc = service("verification")
    fn = clean_passes_fn(db)
    redet = (lambda p, since: redetections(db, p, since))  # noqa: E731 - re-detected after the claim -> qc_failed
    outcomes: Counter[str] = Counter()
    errors: list[str] = []
    for wo in wos:
        sp = db.begin_nested()
        try:
            res = svc.evaluate(db, wo, now, clean_passes=fn, redetections=redet)
            sp.commit()
            outcome = res.get("outcome") if isinstance(res, dict) else None
            outcomes[str(outcome or wo.status)] += 1
        except Exception as e:
            sp.rollback()
            log.exception("verification of work order %s failed", wo.id)
            errors.append(f"wo {wo.id}: {type(e).__name__}: {e}")
    return {"evaluated": len(wos), "outcomes": dict(outcomes), "errors": errors}


def dlp_watch(db: Session, now: datetime, limit: int = 500) -> dict[str, Any]:
    P, E = m.Pothole, m.PotholeEvent
    last_ev = select(E.pothole_id, func.max(E.ts).label("ts")).group_by(E.pothole_id).subquery()
    rows = db.execute(select(P).join(last_ev, last_ev.c.pothole_id == P.id)
                      .where(P.status.in_(("verified_closed", "recurrence_watch")), P.last_seen.is_not(None),
                             P.last_seen > last_ev.c.ts).order_by(P.id).limit(limit)).scalars().all()
    if not rows:
        return {"checked": 0, "dlp_events": 0}
    svc = service("dlp")
    n_ev = 0
    for p in rows:
        sp = db.begin_nested()
        try:
            ev = svc.watch(db, p, now)
            sp.commit()
            n_ev += ev is not None
        except Exception:
            sp.rollback()
            log.exception("dlp.watch failed for pothole %s", p.id)
    return {"checked": len(rows), "dlp_events": n_ev}


def circle_of(db: Session, admin_unit_id: Optional[int], cache: Optional[dict[int, Optional[int]]] = None) -> \
        Optional[int]:
    """Walk admin_unit.parent_id up to the unit of kind 'circle' (None if the chain has no circle)."""
    if admin_unit_id is None:
        return None
    cache = cache if cache is not None else {}
    if admin_unit_id in cache:
        return cache[admin_unit_id]
    cur, hops, res = db.get(m.AdminUnit, admin_unit_id), 0, None
    while cur is not None and hops < 6:
        if cur.kind == "circle":
            res = cur.id
            break
        cur = db.get(m.AdminUnit, cur.parent_id) if cur.parent_id else None
        hops += 1
    cache[admin_unit_id] = res
    return res


def daily_indent(db: Session, now: datetime, run_date: Any = None) -> dict[str, Any]:
    WO, WOP, P = m.WorkOrder, m.WorkOrderPothole, m.Pothole
    rows = db.execute(select(P.admin_unit_id).join(WOP, WOP.pothole_id == P.id).join(WO, WO.id == WOP.work_order_id)
                      .where(WO.status == "issued", WO.run_id.is_(None)).distinct()).all()
    cache: dict[int, Optional[int]] = {}
    circles = sorted({c for (au,) in rows if (c := circle_of(db, au, cache)) is not None})
    if not circles:
        return {"circles": 0, "runs": []}
    svc = service("runs")
    day = run_date or _ist_day(now)
    runs = []
    for cid in circles:
        if db.scalars(select(m.RepairRun.id).where(m.RepairRun.circle_id == cid, m.RepairRun.run_date == day)).first():
            continue  # one run per circle per day (unique); already generated
        sp = db.begin_nested()
        try:
            run = svc.generate(db, cid, day, actor=OPS_ACTOR)
            sp.commit()
            runs.append(getattr(run, "id", None))
        except LookupError as e:  # services.runs.NothingToSchedule: e.g. every WO deferred by the crew cap
            sp.rollback()
            log.info("daily indent for circle %s: %s", cid, e)
        except Exception:
            sp.rollback()
            log.exception("daily indent for circle %s failed", cid)
    return {"circles": len(circles), "runs": runs, "run_date": str(day)}


def kpi_rollup(db: Session, now: datetime, window_days: int = 30, storage: Any = None) -> dict[str, Any]:
    svc = service("kpi")
    zones = list(db.scalars(select(m.AdminUnit).where(m.AdminUnit.kind == "zone", m.AdminUnit.valid_to.is_(None))
                            .order_by(m.AdminUnit.id)))
    snap: dict[str, Any] = {"at": now.isoformat(), "window_days": window_days, "kpi": {}, "scorecards": {}}
    for z in zones:
        snap["kpi"][z.code] = svc.kpi(db, z.id, window_days, now)
    for c in db.scalars(select(m.Contractor).order_by(m.Contractor.id)):
        snap["scorecards"][str(c.id)] = svc.scorecard(db, c.id, now)
    key = None
    try:
        st = _storage(storage, None)
        key = f"kpi/rollup/{_ist_day(now).isoformat()}.json"
        st.put(key, json.dumps(snap, default=str, ensure_ascii=False).encode("utf-8"), "application/json")
    except Exception:
        log.exception("kpi snapshot not stored")
        key = None
    return {"zones": len(zones), "contractors": len(snap["scorecards"]), "snapshot_key": key}


def archive_candidates(db: Session, now: datetime) -> dict[str, Any]:
    P = m.Pothole
    cutoff = now - timedelta(days=ARCHIVE_AFTER_D)
    rows = db.scalars(select(P).where(P.status == "candidate", P.first_seen <= cutoff, P.trips_seen <= 1)).all()
    n = 0
    for p in rows:
        clean = int(p.passes_total or 0) - int(p.passes_seen or 0)
        if clean >= ARCHIVE_MIN_CLEAN:
            transition_pothole(db, p, PotholeStatus.archived, actor=OPS_ACTOR,
                               reason=f"no second pass in {ARCHIVE_AFTER_D} d and {clean} clean passes")
            n += 1
    return {"checked": len(rows), "archived": n}


TICK_FNS: dict[str, Callable[[Session, datetime], Any]] = {
    "sla_tick": sla_tick,
    "verify_repairs": lambda db, now: verify_repairs(db, now),
    "dlp_watch": lambda db, now: dlp_watch(db, now),
    "daily_indent": lambda db, now: daily_indent(db, now),
    "kpi_rollup": lambda db, now: kpi_rollup(db, now),
    "archive_candidates": archive_candidates,
}


def run_ticks(db: Session, now: Optional[datetime] = None, names: Optional[Iterable[str]] = None,
              commit: bool = True) -> dict[str, Any]:
    """Run the ops ticks once; every tick is isolated in its own SAVEPOINT. Never raises for a tick failure."""
    now = as_utc(now) or now_utc()
    out: dict[str, Any] = {}
    for name in names or TICKS:
        fn = TICK_FNS.get(name)
        if fn is None:
            out[name] = {"error": f"unknown tick {name!r}; known: {list(TICKS)}"}
            continue
        sp = db.begin_nested()
        try:
            res = fn(db, now)
            sp.commit()
            out[name] = res if isinstance(res, dict) else {"result": res}
        except TickSkipped as s:
            sp.rollback()
            out[name] = {"skipped": str(s)}
        except Exception as e:
            sp.rollback()
            log.exception("ops tick %s failed", name)
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    if commit:
        db.commit()
    return out


def tick_errors(result: dict[str, Any]) -> list[str]:
    return [f"{k}: {v['error']}" for k, v in result.items() if isinstance(v, dict) and "error" in v]


__all__ = ["TICKS", "archive_candidates", "circle_of", "clean_passes", "daily_indent", "dlp_watch", "kpi_rollup",
           "redetections", "run_ticks", "sla_tick", "tick_errors", "verify_repairs", "ACTOR"]
