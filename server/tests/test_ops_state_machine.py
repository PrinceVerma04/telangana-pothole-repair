"""WP8b acceptance: work-order state machine over the API - 409 on illegal moves, audit on every legal one, full
lifecycle with before/after photos and bus verification (fake clean-pass provider), qc_failed -> rework, RBAC."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select

from server.app.core.util import as_utc, now_utc
from server.app.db import models as m
from server.app.services import verification
from server.tests.test_ops_support import audit_actions, make_pothole, move, passes_provider, photo

NO_QC = {"qc_sampler": lambda _wo, _frac: False}


def test_full_lifecycle_with_bus_verification(client: TestClient, app: Any, login: Any) -> None:
    eng, c1, c2 = login("engineer1"), login("contractor1"), login("contractor2")
    pid = make_pothole(app)

    r = client.post("/v1/work-orders?issue=false", json={"pothole_ids": [pid], "crew": "crew A"}, headers=eng)
    assert r.status_code == 201, r.text
    wo = r.json()
    wid = wo["id"]
    assert wo["status"] == "planned" and wo["pothole_ids"] == [pid] and wo["contractor_name"]
    detail = client.get(f"/v1/potholes/{pid}", headers=eng).json()
    plan = detail["plan"]
    assert detail["pothole"]["status"] == "planned"
    assert plan["bags_low"] <= plan["bags"] <= plan["bags_high"] and wo["est_bags"] == plan["bags"]
    assert abs(wo["est_cost_inr"] - plan["est_cost_inr"]) < 0.01

    # contractors cannot create work orders; a contractor cannot touch another contractor's WO (hidden -> 404)
    assert client.post("/v1/work-orders", json={"pothole_ids": [pid]}, headers=c1).status_code == 403
    r = move(client, wid, "issued", eng)
    assert r.status_code == 200, r.text
    issued = r.json()
    assert issued["status"] == "issued" and issued["sla_due"]
    sla_h = (as_utc(_dt(issued["sla_due"])) - as_utc(_dt(issued["issued_at"]))).total_seconds() / 3600
    assert abs(sla_h - 24.0) < 1e-6  # primary road = arterial: 24 h (GHMC Citizen's Charter)
    assert move(client, wid, "acknowledged", c2).status_code == 404
    assert photo(client, wid, "before", c2).status_code == 404
    assert move(client, wid, "in_progress", c1).status_code == 409  # issued -> in_progress skips acknowledged
    assert move(client, wid, "acknowledged", c1).status_code == 200
    assert move(client, wid, "in_progress", c1).status_code == 422  # before photo missing
    assert photo(client, wid, "after", c1).status_code == 409  # after photo only once in progress
    assert photo(client, wid, "before", c1).status_code == 201
    r = move(client, wid, "in_progress", c1, lat=17.4225, lon=78.455, bag_lot_no="LOT-2026-11-A")
    assert r.status_code == 200 and r.json()["photos"][0]["kind"] == "before"
    assert move(client, wid, "repaired_claimed", c1, bags_used=3).status_code == 422  # after photo missing
    assert photo(client, wid, "after", c1).status_code == 201
    assert move(client, wid, "repaired_claimed", c1).status_code == 422  # bags_used missing
    r = move(client, wid, "repaired_claimed", c1, bags_used=3)
    assert r.status_code == 200 and r.json()["bags_used"] == 3 and len(r.json()["photos"]) == 2
    assert client.get(f"/v1/potholes/{pid}", headers=eng).json()["pothole"]["status"] == "repaired_claimed"

    # the contractor can never verify its own repair
    assert move(client, wid, "verification_pending", c1).status_code == 403

    # bus verification: not enough evidence yet (1 trip), then >= 3 clean passes from >= 2 trips within 14 d
    with app.state.database.session() as s:
        w = s.get(m.WorkOrder, wid)
        t0 = now_utc()
        weak = passes_provider([("trip-1", t0 + timedelta(days=1)), ("trip-1", t0 + timedelta(days=2)),
                                ("trip-1", t0 + timedelta(days=3))])
        res = verification.evaluate(s, w, t0 + timedelta(days=4), weak, **NO_QC)
        assert res["outcome"] == "pending" and w.status == "verification_pending"
        late = passes_provider([("trip-1", t0 + timedelta(days=1)), ("trip-2", t0 + timedelta(days=2)),
                                ("trip-3", t0 + timedelta(days=20))])  # third pass outside the 14-day window
        assert verification.evaluate(s, w, t0 + timedelta(days=21), late, **NO_QC)["outcome"] == "pending"
        good = passes_provider([("trip-1", t0 + timedelta(days=1)), ("trip-2", t0 + timedelta(days=2)),
                                ("trip-2", t0 + timedelta(days=3))])
        res = verification.evaluate(s, w, t0 + timedelta(days=5), good, **NO_QC)
        assert res["outcome"] == "verified_closed" and w.status == "verified_closed", res
        s.commit()
        v = s.scalars(select(m.Verification).where(m.Verification.work_order_id == wid)).one()
        assert (v.clean_passes, v.distinct_trips, v.outcome) == (3, 2, "verified_closed")
        actions = audit_actions(s, "work_order", wid)
    status_moves = [a for a in actions if a.startswith("status:")]
    assert status_moves == ["status:planned->issued", "status:issued->acknowledged",
                            "status:acknowledged->in_progress", "status:in_progress->repaired_claimed",
                            "status:repaired_claimed->verification_pending",
                            "status:verification_pending->verified_closed"]  # every legal move, no illegal ones
    assert actions[0] == "create" and actions.count("photo:before") == 1 and actions.count("photo:after") == 1
    assert client.get(f"/v1/potholes/{pid}", headers=eng).json()["pothole"]["status"] == "verified_closed"
    # terminal: nothing moves a verified WO any more
    assert move(client, wid, "qc_failed", eng, reason="late QC").status_code == 409
    # invoice / MB eligible now
    assert [w["id"] for w in client.get("/v1/exports/invoice-eligible", headers=eng).json()] == [wid]
    assert [w["id"] for w in client.get("/v1/work-orders?invoice_eligible=true", headers=c1).json()] == [wid]


def _dt(s: str) -> Any:
    from datetime import datetime

    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def test_qc_failed_creates_issued_rework(client: TestClient, app: Any, login: Any) -> None:
    eng, zonal, c1 = login("engineer1"), login("zonal_khairatabad"), login("contractor1")
    pid = make_pothole(app)
    r = client.post("/v1/work-orders", json={"pothole_ids": [pid]}, headers=eng)  # contractor known -> issued
    assert r.status_code == 201 and r.json()["status"] == "issued", r.text
    wid = r.json()["id"]
    assert move(client, wid, "acknowledged", c1).status_code == 200
    assert photo(client, wid, "before", c1).status_code == 201
    assert move(client, wid, "in_progress", c1).status_code == 200
    assert photo(client, wid, "after", c1).status_code == 201
    assert move(client, wid, "repaired_claimed", c1, bags_used=9).json()["bag_variance_flag"] is True
    assert move(client, wid, "verified_closed", eng).status_code == 409  # must pass verification_pending first
    assert move(client, wid, "verification_pending", eng).status_code == 200
    r = move(client, wid, "qc_failed", zonal, reason="patch ravelled at edges; QC sample failed")
    assert r.status_code == 200 and r.json()["status"] == "qc_failed"
    assert move(client, wid, "issued", zonal).status_code == 409  # qc_failed is terminal

    wos = client.get("/v1/work-orders", headers=c1).json()
    rework = [w for w in wos if w["parent_work_order_id"] == wid]
    assert len(rework) == 1
    rw = rework[0]
    assert rw["is_rework"] and rw["status"] == "issued" and rw["pothole_ids"] == [pid]
    assert rw["est_cost_inr"] == 0.0 and rw["contractor_id"] == r.json()["contractor_id"]
    detail = client.get(f"/v1/potholes/{pid}", headers=eng).json()
    assert [e["to_status"] for e in detail["events"]][-2:] == ["qc_failed", "work_ordered"]
    with app.state.database.session() as s:
        v = s.scalars(select(m.Verification).where(m.Verification.work_order_id == wid)).one()
        assert v.outcome == "qc_failed" and v.qc_passed is False
    # neither the failed WO nor the (unfinished) rework is invoice eligible
    assert client.get("/v1/exports/invoice-eligible", headers=eng).json() == []
    mb = client.get("/v1/exports/measurement-book.csv", headers=eng)
    assert mb.status_code == 200 and f"WO {wid}" not in mb.text


def test_illegal_transitions_answer_409_and_are_not_audited(client: TestClient, app: Any, login: Any) -> None:
    eng, zonal = login("engineer1"), login("zonal_khairatabad")
    pid = make_pothole(app)
    wid = client.post("/v1/work-orders?issue=false", json={"pothole_ids": [pid]}, headers=eng).json()["id"]
    for to in ("verified_closed", "acknowledged", "repaired_claimed", "qc_failed", "planned"):
        assert move(client, wid, to, eng).status_code in (403, 409), to
    assert move(client, wid, "verified_closed", zonal).status_code == 409
    assert move(client, wid, "cancelled", eng).status_code == 403  # only zonal_admin cancels
    assert move(client, wid, "cancelled", zonal, reason="duplicate").status_code == 200
    assert move(client, wid, "issued", zonal).status_code == 409
    assert move(client, wid, "bogus", zonal).status_code == 422
    # second open WO for the same pothole is refused while one is live; a cancelled one does not block
    pid2 = make_pothole(app)
    assert client.post("/v1/work-orders?issue=false", json={"pothole_ids": [pid2]}, headers=eng).status_code == 201
    assert client.post("/v1/work-orders?issue=false", json={"pothole_ids": [pid2]}, headers=eng).status_code == 409
    assert client.post("/v1/work-orders", json={"pothole_ids": [pid]}, headers=eng).status_code == 201
    with app.state.database.session() as s:
        assert audit_actions(s, "work_order", wid) == ["create", "status:planned->cancelled"]
    audit = client.get(f"/v1/admin/audit?entity=work_order&id={wid}", headers=zonal).json()
    assert {a["action"] for a in audit} == {"create", "status:planned->cancelled"}


def test_contractor_isolation_on_work_order_endpoints(client: TestClient, app: Any, login: Any) -> None:
    eng, c1, c2 = login("engineer1"), login("contractor1"), login("contractor2")
    p1 = make_pothole(app, contract_code="CRMP-P2")
    p2 = make_pothole(app, contract_code="CRMP-P7", lat=17.43, lon=78.46)
    w1 = client.post("/v1/work-orders", json={"pothole_ids": [p1]}, headers=eng).json()["id"]
    w2 = client.post("/v1/work-orders", json={"pothole_ids": [p2]}, headers=eng).json()["id"]
    assert [w["id"] for w in client.get("/v1/work-orders", headers=c1).json()] == [w1]
    assert [w["id"] for w in client.get("/v1/work-orders", headers=c2).json()] == [w2]
    assert {w["id"] for w in client.get("/v1/work-orders", headers=eng).json()} == {w1, w2}
    assert client.get(f"/v1/work-orders/{w2}", headers=c1).status_code == 404
    assert move(client, w2, "acknowledged", c1).status_code == 404
    assert client.get("/v1/work-orders", headers=login("citizen1")).status_code == 403
    assert client.get("/v1/work-orders").status_code == 401
    # an R&B admin cannot see GHMC work orders
    assert client.get("/v1/work-orders", headers=login("rnb_admin")).json() == []
    # contractor1 cannot issue / cancel either (roles)
    assert move(client, w1, "cancelled", c1).status_code == 403
