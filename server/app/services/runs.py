"""Repair planning of a pothole and the daily repair run / indent (PLAN 7 "Daily repair run").

* ``plan_pothole(db, pothole, context)`` runs ``repair.planner.plan`` with the agency's active rate card
  (``configstore``) and ``configs/irc_rules.yaml``, supersedes older plans, stores a ``repair_plan`` row, audits it
  and moves ``confirmed``/``reopened`` potholes to ``planned``. Without an explicit context it uses
  ``plan_context`` (monsoon window + confirmed potholes within 300 m for the jet-patcher rule, PLAN 7.9). This is the
  single planning path: the API (``POST /potholes/{id}/plan``) and the WP6 pipeline ``plan`` stage both call it.
* ``generate(db, circle_id, run_date, actor=...)`` groups the circle's ``issued``/``acknowledged`` work orders
  (issued on or before the run date and not already in a later run), orders the stops by greedy nearest-neighbour
  from the circle office (centroid of the circle polygon; centroid of the stops if the circle has no geometry),
  caps the run by crew-hours (``crew_hours_cap`` per crew x number of distinct contractor crews; work orders that do
  not fit are deferred to the next run, at least one is always scheduled) and stores ``repair_run`` (one per circle
  and date; regenerating updates it). Totals = ``repair.mb_export.indent_totals`` over the stops' plans plus the
  bag range sums ``bags_low`` / ``bags_high`` (never "exact" bags).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from measure.types import PotholeStatus
from priority.weather import in_monsoon_window
from repair import planner
from repair.loaders import load_rules
from repair.mb_export import indent_csv, indent_totals
from repair.types import PlanContext, RepairPlan
from server.app import configstore
from server.app.core import audit
from server.app.core.rbac import admin_subtree_ids
from server.app.core.transitions import transition_pothole
from server.app.core.util import as_utc, haversine_m, now_utc, radius_box, to_ist
from server.app.db import models as m
from server.app.db.convert import row_to_pothole_fused
from server.app.schemas.api import RepairRunOut
from server.app.schemas.states import WorkOrderStatus as W

RUN_WO_STATUSES = (W.issued.value, W.acknowledged.value)
DEFAULT_CREW_HOURS_CAP = 8.0
DEFAULT_LABOUR_H = 2.0  # PLAN 7.7 base labour per pothole when a stop has no plan


class NothingToSchedule(LookupError):
    pass


# =========================================================================== planning
def current_plan(db: Session, pothole_id: int) -> Optional[m.RepairPlanRow]:
    R = m.RepairPlanRow
    return db.scalars(select(R).where(R.pothole_id == pothole_id, R.superseded.is_(False))
                      .order_by(R.id.desc())).first()


def plan_of(row: Optional[m.RepairPlanRow]) -> Optional[RepairPlan]:
    if row is None:
        return None
    try:
        return RepairPlan.model_validate(row.json)
    except ValueError:
        return None


def default_context(now: Optional[datetime] = None, cfg: Optional[dict[str, Any]] = None) -> PlanContext:
    month = to_ist(as_utc(now) or now_utc()).month  # type: ignore[union-attr]
    return PlanContext(in_monsoon_window=in_monsoon_window(month, cfg))


CLUSTER_RADIUS_M = 300.0  # PLAN 7.9 jet-patcher cluster rule
CLUSTER_STATUSES = (PotholeStatus.confirmed.value, PotholeStatus.planned.value, PotholeStatus.work_ordered.value,
                    PotholeStatus.reopened.value)


def cluster_count(db: Session, pothole: m.Pothole, radius_m: float = CLUSTER_RADIUS_M) -> int:
    """Other confirmed (not yet repaired) potholes within ``radius_m`` of ``pothole``."""
    if pothole.lat is None or pothole.lon is None:
        return 0
    x1, y1, x2, y2 = radius_box(pothole.lat, pothole.lon, radius_m)
    P = m.Pothole
    rows = db.execute(select(P.id, P.lat, P.lon).where(P.lat.between(y1, y2), P.lon.between(x1, x2),
                                                       P.status.in_(CLUSTER_STATUSES), P.id != pothole.id)).all()
    return sum(1 for _i, la, lo in rows if haversine_m(pothole.lat, pothole.lon, la, lo) <= radius_m)


def plan_context(db: Session, pothole: m.Pothole, now: Optional[datetime] = None) -> PlanContext:
    """Default situational inputs: monsoon window (agency priority config) + jet-patcher cluster count."""
    ctx = default_context(now, configstore.active_priority_config(db, pothole.agency_id))
    return ctx.model_copy(update={"cluster_n_within_300m": cluster_count(db, pothole)})


def plan_pothole(db: Session, pothole: m.Pothole, context: Optional[PlanContext] = None, *, actor: Any = None,
                 now: Optional[datetime] = None, transition: bool = True,
                 extras: Optional[dict[str, Any]] = None) -> tuple[m.RepairPlanRow, RepairPlan]:
    """Plan one pothole and store it. Raises ``ValueError`` when the pothole cannot be planned (no footprint).
    ``extras`` are merged into the stored plan's ``extras`` (e.g. the pipeline's input signature)."""
    now = as_utc(now) or now_utc()
    ctx = context or plan_context(db, pothole, now)
    fused = row_to_pothole_fused(pothole)
    plan = planner.plan(fused, load_rules(), configstore.active_rate_card(db, pothole.agency_id), ctx)
    if extras:
        plan = plan.model_copy(update={"extras": {**plan.extras, **extras}})
    R = m.RepairPlanRow
    prev = current_plan(db, pothole.id)
    db.execute(update(R).where(R.pothole_id == pothole.id, R.superseded.is_(False)).values(superseded=True))
    row = R(pothole_id=pothole.id, json=plan.model_dump(mode="json"), spec_code=plan.spec_code.value,
            est_cost_inr=float(plan.est_cost_inr), bags=int(plan.bags), rate_card_version=plan.rate_card_version,
            created_at=now, superseded=False)
    db.add(row)
    db.flush()
    audit.record(db, "repair_plan", row.id, "create", actor=actor,
                 before={"superseded_plan_id": prev.id} if prev else None,
                 after={"pothole_id": pothole.id, "spec_code": plan.spec_code.value, "bags": plan.bags,
                        "bags_low": plan.bags_low, "bags_high": plan.bags_high, "bag_kg": plan.bag_kg,
                        "est_cost_inr": plan.est_cost_inr, "pending_remeasure": plan.pending_remeasure,
                        "rate_card_version": plan.rate_card_version},
                 lat=pothole.lat, lon=pothole.lon)
    if transition and pothole.status in (PotholeStatus.confirmed.value, PotholeStatus.reopened.value):
        transition_pothole(db, pothole, PotholeStatus.planned, actor=actor, reason=f"repair plan #{row.id}")
    return row, plan


# =========================================================================== runs
def _circle_office(circle: m.AdminUnit) -> Optional[tuple[float, float]]:
    if not circle.geom_wkt:
        return None
    try:
        from shapely import wkt as swkt

        c = swkt.loads(circle.geom_wkt).centroid
        return (float(c.y), float(c.x)) if not c.is_empty else None
    except Exception:
        return None


def _candidate_work_orders(db: Session, circle_id: int, run_date: date, run_id: Optional[int]) -> list[m.WorkOrder]:
    WO, WOP, P = m.WorkOrder, m.WorkOrderPothole, m.Pothole
    in_circle = (select(WOP.work_order_id).join(P, P.id == WOP.pothole_id)
                 .where(P.admin_unit_id.in_(admin_subtree_ids(circle_id))))
    earlier_runs = select(m.RepairRun.id).where(m.RepairRun.run_date < run_date)
    stmt = select(WO).where(WO.status.in_(RUN_WO_STATUSES), WO.id.in_(in_circle),
                            WO.run_id.is_(None) | WO.run_id.in_(earlier_runs)
                            | (WO.run_id == run_id if run_id is not None else WO.run_id.is_(None)))
    out = []
    for wo in db.scalars(stmt.order_by(WO.id)):
        issued = to_ist(wo.issued_at) if wo.issued_at else None
        if issued is None or issued.date() <= run_date:
            out.append(wo)
    return out


def _stop_plan(db: Session, wop: m.WorkOrderPothole, ph: m.Pothole) -> tuple[Optional[int], Optional[RepairPlan]]:
    row = db.get(m.RepairPlanRow, wop.repair_plan_id) if wop.repair_plan_id else None
    if row is None:
        row = current_plan(db, ph.id)
    if row is None:
        try:
            row, _ = plan_pothole(db, ph, transition=False)
        except ValueError:
            return None, None
    return row.id, plan_of(row)


def generate(db: Session, circle_id: int, run_date: date, *, actor: Any = None,
             crew_hours_cap: float = DEFAULT_CREW_HOURS_CAP, now: Optional[datetime] = None) -> m.RepairRun:
    """Build (or rebuild) the repair run of ``circle_id`` for ``run_date``. Does not commit."""
    now = as_utc(now) or now_utc()
    circle = db.get(m.AdminUnit, circle_id)
    if circle is None:
        raise LookupError(f"admin unit {circle_id} not found")
    run = db.scalars(select(m.RepairRun).where(m.RepairRun.circle_id == circle_id,
                                               m.RepairRun.run_date == run_date)).first()
    wos = _candidate_work_orders(db, circle_id, run_date, run.id if run else None)
    if not wos and run is None:
        raise NothingToSchedule(f"no issued/acknowledged work orders in circle {circle.code} for {run_date}")

    # ---- stops (one per pothole) with their plans
    stops: list[dict[str, Any]] = []
    wo_hours: dict[int, float] = {}
    for wo in wos:
        hours = 0.0
        for wop in db.scalars(select(m.WorkOrderPothole).where(m.WorkOrderPothole.work_order_id == wo.id)
                              .order_by(m.WorkOrderPothole.pothole_id)):
            ph = db.get(m.Pothole, wop.pothole_id)
            if ph is None or ph.lat is None or ph.lon is None:
                continue
            plan_id, plan = _stop_plan(db, wop, ph)
            hours += plan.labour_h if plan else DEFAULT_LABOUR_H
            seg = db.get(m.RoadSegment, ph.road_segment_id) if ph.road_segment_id else None
            stops.append({"pothole_id": ph.id, "work_order_id": wo.id, "lat": ph.lat, "lon": ph.lon,
                          "repair_plan_id": plan_id, "plan": plan, "severity": ph.severity,
                          "road_name": seg.name if seg else None, "contractor_id": wo.contractor_id})
        wo_hours[wo.id] = hours

    # ---- greedy nearest-neighbour from the circle office, capped by crew-hours (whole work orders)
    crews = max(1, len({(wo.contractor_id, wo.crew) for wo in wos}))
    cap_total = float(crew_hours_cap) * crews
    office = _circle_office(circle)
    if office is None and stops:
        office = (sum(s["lat"] for s in stops) / len(stops), sum(s["lon"] for s in stops) / len(stops))
    cur = office
    remaining = list(stops)
    route: list[dict[str, Any]] = []
    included: list[int] = []
    deferred: list[int] = []
    used = 0.0
    while remaining:
        nxt = min(remaining, key=lambda s: haversine_m(cur[0], cur[1], s["lat"], s["lon"]))  # type: ignore[index]
        wid = nxt["work_order_id"]
        if wid not in included:
            if included and used + wo_hours[wid] > cap_total + 1e-9:
                deferred.append(wid)
                remaining = [s for s in remaining if s["work_order_id"] != wid]
                continue
            included.append(wid)
            used += wo_hours[wid]
        hop = haversine_m(cur[0], cur[1], nxt["lat"], nxt["lon"])  # type: ignore[index]
        route.append({**nxt, "hop_m": round(hop, 1)})
        cur = (nxt["lat"], nxt["lon"])
        remaining.remove(nxt)

    items = [{"plan": s["plan"], "work_order_id": s["work_order_id"], "pothole_id": s["pothole_id"],
              "lat": s["lat"], "lon": s["lon"], "hop_m": s["hop_m"]} for s in route if s["plan"] is not None]
    totals: dict[str, float] = {k: float(v) for k, v in indent_totals(items).items()}
    totals.update({
        "bags_low": float(sum(i["plan"].bags_low for i in items)),
        "bags_high": float(sum(i["plan"].bags_high for i in items)),
        "n_stops": float(len(route)), "n_work_orders": float(len(included)), "deferred_work_orders": float(len(deferred)),
        "stops_without_plan": float(sum(1 for s in route if s["plan"] is None)),
        "route_km": round(sum(s["hop_m"] for s in route) / 1000.0, 3), "crews": float(crews),
        "crew_hours_cap_total": cap_total,
    })
    stored_route = []
    for seq, s in enumerate(route, start=1):
        p = s["plan"]
        stored_route.append({
            "seq": seq, "pothole_id": s["pothole_id"], "work_order_id": s["work_order_id"], "lat": s["lat"],
            "lon": s["lon"], "hop_m": s["hop_m"], "repair_plan_id": s["repair_plan_id"], "severity": s["severity"],
            "road_name": s["road_name"], "bags": p.bags if p else None, "bags_low": p.bags_low if p else None,
            "bags_high": p.bags_high if p else None, "bag_kg": p.bag_kg if p else None,
            "labour_h": p.labour_h if p else None, "est_cost_inr": p.est_cost_inr if p else None,
            "spec_code": p.spec_code.value if p else None, "pending_remeasure": p.pending_remeasure if p else None})

    before = {"route_len": len(run.route or []), "totals": run.totals} if run is not None else None
    if run is None:
        run = m.RepairRun(circle_id=circle_id, run_date=run_date)
        db.add(run)
    run.totals, run.route, run.crew_hours_cap, run.generated_at = totals, stored_route, float(crew_hours_cap), now
    db.flush()
    WO_ = m.WorkOrder
    db.execute(update(WO_).where(WO_.run_id == run.id, WO_.id.notin_(included or [-1])).values(run_id=None))
    if included:
        db.execute(update(WO_).where(WO_.id.in_(included)).values(run_id=run.id))
    db.flush()
    for wo in wos:  # keep the identity map in sync with the bulk UPDATEs
        db.refresh(wo, ["run_id"])
    audit.record(db, "repair_run", run.id, "generate" if before is None else "regenerate", actor=actor,
                 before=before, after={"circle_id": circle_id, "run_date": run_date, "work_order_ids": included,
                                       "deferred_work_order_ids": deferred, "totals": totals})
    return run


def run_work_order_ids(db: Session, run: m.RepairRun) -> list[int]:
    return list(db.scalars(select(m.WorkOrder.id).where(m.WorkOrder.run_id == run.id).order_by(m.WorkOrder.id)))


def run_out(db: Session, run: m.RepairRun) -> RepairRunOut:
    circle = db.get(m.AdminUnit, run.circle_id)
    totals = {k: float(v) for k, v in (run.totals or {}).items() if isinstance(v, (int, float))}
    return RepairRunOut(id=run.id, circle_id=run.circle_id, circle_name=circle.name_en if circle else None,
                        run_date=run.run_date, work_order_ids=run_work_order_ids(db, run), route=list(run.route or []),
                        totals=totals, crew_hours_cap=run.crew_hours_cap, generated_at=run.generated_at)


def run_indent_csv(db: Session, run: m.RepairRun) -> str:
    """Indent CSV in route order from the plans the run was generated with (TOTAL row == run.totals)."""
    circle = db.get(m.AdminUnit, run.circle_id)
    items = []
    for s in run.route or []:
        plan = plan_of(db.get(m.RepairPlanRow, s.get("repair_plan_id"))) if s.get("repair_plan_id") else None
        if plan is None:
            continue
        items.append({"plan": plan, "work_order_id": s.get("work_order_id"), "pothole_id": s.get("pothole_id"),
                      "lat": s.get("lat"), "lon": s.get("lon"), "hop_m": s.get("hop_m")})
    return indent_csv({"id": run.id, "circle_id": run.circle_id, "circle_name": circle.name_en if circle else None,
                       "run_date": run.run_date.isoformat(), "items": items})


__all__ = ["NothingToSchedule", "current_plan", "default_context", "generate", "plan_of", "plan_pothole",
           "run_indent_csv", "run_out", "run_work_order_ids"]
