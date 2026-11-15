"""Bus-verified repair (PLAN 5.7, CONTRACTS pothole state machine).

``evaluate(db, wo, now, clean_passes)``: for a work order in ``repaired_claimed`` / ``verification_pending``:

* ``repaired_claimed`` is first moved to ``verification_pending`` (system actor; it is "normally automatic").
* For every pothole of the WO, ``clean_passes(pothole, since=claim time)`` returns ``[(trip_id, ts), ...]`` - bus
  passes over the same way + direction + lane band with the camera active and no verified detection within eps
  (built by WP5/WP6 from ``geo.passes.count_passes``). Only passes in ``[claim, claim + window_days]`` and not after
  ``now`` count.
* Closed when every pothole has >= ``clean_passes`` (3) passes from >= ``min_distinct_trips`` (2) trips within
  ``window_days`` (14) AND the deterministic ``qc_sample_frac`` (10 %) engineer QC sample is passed or not drawn.
  A drawn sample keeps the WO pending until an engineer moves it to verified_closed (QC passed) or qc_failed.
* ``redetections`` (optional provider, same signature) returning anything -> ``qc_failed`` + rework WO.
* Otherwise the WO stays ``verification_pending``; after the window the result says ``window_expired`` so an
  engineer can visit (no automatic failure: absence of passes is not evidence of a bad repair).
Thresholds come from the agency priority config ``verification`` block. Writes/updates ``verification`` rows.
Does not commit.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Sequence

from sqlalchemy.orm import Session

from server.app import configstore
from server.app.core.util import as_utc, now_utc
from server.app.db import models as m
from server.app.schemas.states import WorkOrderStatus as W
from server.app.services import sla
from server.app.services.state_machine import pothole_ids_of, transition_work_order, upsert_verification

PassProvider = Callable[[m.Pothole, datetime], Sequence[tuple[Any, datetime]]]
ACTOR = "system:verification"
DEFAULTS = {"clean_passes": 3, "window_days": 14, "min_distinct_trips": 2, "qc_sample_frac": 0.10}


def thresholds(db: Session, wo: m.WorkOrder) -> dict[str, float]:
    cfg = configstore.active_priority_config(db, wo.agency_id).get("verification") or {}
    return {k: float(cfg.get(k, v)) for k, v in DEFAULTS.items()}


def qc_sampled(wo_id: int, frac: float) -> bool:
    """Deterministic ~``frac`` random sample (stable across ticks / restarts; not guessable from the id order)."""
    h = int(hashlib.sha256(f"gunthameter-qc:{wo_id}".encode()).hexdigest()[:8], 16)
    return (h % 10_000) < frac * 10_000


def evaluate(db: Session, wo: m.WorkOrder, now: Optional[datetime], clean_passes: PassProvider, *,
             redetections: Optional[PassProvider] = None,
             qc_sampler: Optional[Callable[[m.WorkOrder, float], bool]] = None) -> dict[str, Any]:
    now = as_utc(now) or now_utc()
    out: dict[str, Any] = {"work_order_id": wo.id, "status_before": wo.status, "potholes": []}
    if wo.status not in (W.repaired_claimed.value, W.verification_pending.value):
        return {**out, "outcome": "not_applicable", "status": wo.status,
                "reason": "only repaired_claimed / verification_pending work orders are evaluated"}
    th = thresholds(db, wo)
    if wo.status == W.repaired_claimed.value:
        transition_work_order(db, wo, W.verification_pending, actor=ACTOR, reason="repair claimed; bus verification",
                              now=now)
    start = sla.claimed_at(db, wo) or as_utc(wo.updated_at) or now
    end = start + timedelta(days=th["window_days"])
    sampled = (qc_sampler or (lambda w, f: qc_sampled(w.id, f)))(wo, th["qc_sample_frac"])
    out.update({"window_start": start.isoformat(), "window_end": end.isoformat(), "qc_sampled": sampled,
                "thresholds": th})

    potholes = [p for p in (db.get(m.Pothole, pid) for pid in pothole_ids_of(db, wo)) if p is not None]
    redetected: list[int] = []
    all_ok = bool(potholes)
    for ph in potholes:
        passes = [(t, as_utc(ts)) for t, ts in clean_passes(ph, start)]
        passes = [(t, ts) for t, ts in passes if ts is not None and start <= ts <= min(end, now)]
        trips = {str(t) for t, _ in passes}
        ok = len(passes) >= th["clean_passes"] and len(trips) >= th["min_distinct_trips"]
        all_ok &= ok
        if redetections is not None and list(redetections(ph, start)):
            redetected.append(ph.id)
        upsert_verification(db, wo, ph.id, clean_passes=len(passes), distinct_trips=len(trips), window_start=start,
                             window_end=end, qc_sampled=sampled)
        out["potholes"].append({"pothole_id": ph.id, "clean_passes": len(passes), "distinct_trips": len(trips),
                                "ok": ok})

    if redetected:
        transition_work_order(db, wo, W.qc_failed, actor=ACTOR, now=now,
                              reason=f"pothole(s) {redetected} re-detected by bus after the repair claim")
        for ph in potholes:
            upsert_verification(db, wo, ph.id, outcome="qc_failed", qc_passed=False)
        outcome, reason = "qc_failed", "re-detected"
    elif all_ok and not sampled:
        transition_work_order(db, wo, W.verified_closed, actor=ACTOR, now=now,
                              reason=(f">= {int(th['clean_passes'])} clean bus passes from >= "
                                      f"{int(th['min_distinct_trips'])} trips within {int(th['window_days'])} d"))
        for ph in potholes:
            upsert_verification(db, wo, ph.id, outcome="verified_closed")
        outcome, reason = "verified_closed", "clean passes"
    elif all_ok:
        outcome, reason = "pending", "clean passes met; awaiting the engineer QC sample"
    elif now > end:
        outcome, reason = "pending", "window_expired"
    else:
        outcome, reason = "pending", "waiting for clean passes"
    db.flush()
    return {**out, "outcome": outcome, "status": wo.status, "reason": reason}


__all__ = ["ACTOR", "PassProvider", "evaluate", "qc_sampled", "thresholds"]
