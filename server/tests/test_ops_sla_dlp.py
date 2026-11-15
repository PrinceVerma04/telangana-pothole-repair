"""WP8b acceptance: SLA tick breaches a 24 h arterial work order and escalates AE -> DEE -> EE -> Zonal
Commissioner (configs/agencies.yaml); DLP watch reopens a verified pothole and writes a dlp_event + evidence package
inside the DLP (none outside it)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from server.app.core.util import as_utc
from server.app.db import models as m
from server.app.services import dlp, sla, verification
from server.app.services import state_machine as sm
from server.tests.test_ops_support import C17_LAT, audit_actions, drive_to_claimed, make_pothole, passes_provider

T0 = datetime(2026, 11, 10, 4, 0, tzinfo=timezone.utc)  # 09:30 IST, NE-monsoon window (Oct-Nov)


def test_escalation_ladder_pure() -> None:
    due = T0 + timedelta(hours=24)
    assert sla.escalation_level(T0, due, T0 + timedelta(hours=11)) is None
    assert sla.escalation_level(T0, due, T0 + timedelta(hours=12)) == "AE"
    assert sla.escalation_level(T0, due, due) == "DEE"
    assert sla.escalation_level(T0, due, due + timedelta(hours=24)) == "EE"
    assert sla.escalation_level(T0, due, due + timedelta(hours=72)) == "ZONAL_COMMISSIONER"


def test_sla_tick_breaches_arterial_and_escalates(client: TestClient, app: Any, login: Any) -> None:
    arterial = make_pothole(app, highway="primary")
    local = make_pothole(app, highway="residential", lat=C17_LAT + 0.002)
    with app.state.database.session() as s:
        wa = sm.create_work_order(s, [arterial], actor="test", issue=True, now=T0)
        wl = sm.create_work_order(s, [local], actor="test", issue=True, now=T0)
        assert as_utc(wa.sla_due) == T0 + timedelta(hours=24)  # arterial: 24 h
        assert as_utc(wl.sla_due) == T0 + timedelta(hours=72)  # other road, monsoon window: 72 h
        s.commit()
        wa_id, wl_id = wa.id, wl.id

        seen = []
        for h, expect in ((6, None), (13, "AE"), (25, "DEE"), (49, "EE"), (97, "ZONAL_COMMISSIONER"),
                          (98, "ZONAL_COMMISSIONER")):
            res = sla.sla_tick(s, T0 + timedelta(hours=h))
            s.commit()
            assert s.get(m.WorkOrder, wa_id).escalation_level == expect, h
            seen.append(res)
        assert seen[2]["sla_breached"] == [wa_id] and seen[2]["breached"] == 1  # local WO not breached at 25 h
        assert [e["to"] for r in seen for e in r["escalated"] if e["work_order_id"] == wa_id] == \
            ["AE", "DEE", "EE", "ZONAL_COMMISSIONER"]  # each step once, never repeated
        assert s.get(m.WorkOrder, wl_id).escalation_level == "EE"  # 72 h SLA: 25 h past breach at the 97 h tick
        notes = list(s.scalars(select(m.Notification).where(m.Notification.entity == "work_order",
                                                            m.Notification.entity_id == wa_id)
                               .order_by(m.Notification.id)))
        assert [n.subject.split("]")[0][1:] for n in notes] == ["AE", "DEE", "EE", "ZONAL_COMMISSIONER"]
        assert notes[0].to_role == "field_engineer" and notes[1].to_role == "zonal_admin"
        assert audit_actions(s, "work_order", wa_id).count("escalate") == 4

    out = client.get(f"/v1/work-orders/{wa_id}", headers=login("engineer1")).json()
    assert out["sla_breached"] is True and out["escalation_level"] == "ZONAL_COMMISSIONER"


def test_claim_stops_the_sla_clock(app: Any, seeded: Any) -> None:
    pid = make_pothole(app)
    with app.state.database.session() as s:
        wo = sm.create_work_order(s, [pid], actor="test", issue=True, now=T0)
        drive_to_claimed(s, wo, T0 + timedelta(hours=10))  # claimed at T0+11 h, inside the 24 h SLA
        s.commit()
        assert not sla.is_breached(wo, T0 + timedelta(days=5), sla.claimed_at(s, wo))
        assert sla.sla_tick(s, T0 + timedelta(days=5))["checked"] == 0  # clock stopped: not ticked any more


def _verified(app: Any, pid: int, at: datetime) -> int:
    with app.state.database.session() as s:
        wo = sm.create_work_order(s, [pid], actor="test", issue=True, now=at)
        drive_to_claimed(s, wo, at)
        passes = passes_provider([("t1", at + timedelta(days=1)), ("t2", at + timedelta(days=2)),
                                  ("t2", at + timedelta(days=3))])
        res = verification.evaluate(s, wo, at + timedelta(days=4), passes, qc_sampler=lambda _w, _f: False)
        assert res["outcome"] == "verified_closed"
        s.commit()
        return wo.id


def test_dlp_watch_inside_dlp_creates_event_and_reopens(client: TestClient, app: Any, login: Any) -> None:
    pid = make_pothole(app, contract_code="CRMP-P2")  # CRMP package: 60-month DLP
    wid = _verified(app, pid, T0)
    with app.state.database.session() as s:
        ph = s.get(m.Pothole, pid)
        assert ph.status == "verified_closed"
        ev = dlp.watch(s, ph, T0 + timedelta(days=40))
        s.commit()
        assert ev is not None and ph.status == "reopened" and ph.reopen_count_12m == 1
        assert ev.work_order_id == wid and as_utc(ev.dlp_until) > T0 + timedelta(days=365 * 4)
        evidence = ev.evidence
        assert evidence["kind"] == "dlp_evidence_package" and "not an automatic penalty" in evidence["note"]
        kinds = [r["kind"] for r in evidence["repair_events"]]
        assert "before_photo" in kinds and "after_photo" in kinds and "bags" in kinds
        assert evidence["verification"][0]["outcome"] == "verified_closed"
        rework = s.get(m.WorkOrder, ev.rework_work_order_id)
        assert rework.is_rework and rework.parent_work_order_id == wid and rework.status == "planned"
        assert rework.est_cost_inr == 0.0
        assert s.scalar(select(func.count(m.Notification.id)).where(m.Notification.entity == "dlp_event")) == 1
        assert audit_actions(s, "dlp_event", ev.id) == ["create"]
        assert dlp.watch(s, ph, T0 + timedelta(days=41)) is None  # already reopened: no duplicate event
        rework_id = ev.rework_work_order_id

    # the engineer reviews the evidence and issues the zero-pay rework: pothole reopened -> planned -> work_ordered
    r = client.patch(f"/v1/work-orders/{rework_id}/transition", json={"to": "issued"}, headers=login("engineer1"))
    assert r.status_code == 200, r.text
    detail = client.get(f"/v1/potholes/{pid}", headers=login("engineer1")).json()
    assert [e["to_status"] for e in detail["events"]][-3:] == ["reopened", "planned", "work_ordered"]
    card = client.get(f"/v1/contractors/{r.json()['contractor_id']}/scorecard", headers=login("zonal_khairatabad"))
    assert card.status_code == 200 and card.json()["dlp_events"] == 1 and card.json()["reopen_rate"] == 1.0


def test_dlp_watch_outside_warranty_only_reopens(app: Any, seeded: Any) -> None:
    with app.state.database.session() as s:
        ctr = s.scalars(select(m.Contract).where(m.Contract.code == "CRMP-P7")).one().contractor_id
    pid = make_pothole(app, contract_code=None, contractor_id=ctr)  # no contract: 6-month patch warranty
    _verified(app, pid, T0)
    other_id = make_pothole(app, lat=C17_LAT + 0.004)
    with app.state.database.session() as s:
        ph = s.get(m.Pothole, pid)
        assert dlp.watch(s, ph, T0 + timedelta(days=250)) is None
        s.commit()
        assert ph.status == "reopened"
        assert s.scalar(select(func.count(m.DlpEvent.id))) == 0
        other = s.get(m.Pothole, other_id)
        assert dlp.watch(s, other, T0) is None and other.status == "confirmed"  # not watched: untouched
