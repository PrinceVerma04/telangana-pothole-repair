"""WP8b: KPI endpoint (admin unit by id / code / name / all), contractor list + scorecard (scoped), Kaplan-Meier,
truth read endpoints, plan endpoints."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi.testclient import TestClient

from server.app.core.util import now_utc
from server.app.services import kpi as kpi_svc
from server.app.services import state_machine as sm
from server.app.services import verification
from server.tests.test_ops_support import C17_LAT, drive_to_claimed, make_pothole, passes_provider


def test_kaplan_meier() -> None:
    assert kpi_svc.kaplan_meier([]) == []
    curve = kpi_svc.kaplan_meier([(10.0, True), (20.0, False), (30.0, True), (40.0, False)], at_days=(0, 15, 35, 50))
    # S(10) = 3/4; at 30 two at risk, one event -> 3/4 * 1/2
    assert [c["survival"] for c in curve] == [1.0, 0.75, 0.375, 0.375]


def test_plan_endpoints(client: TestClient, app: Any, login: Any) -> None:
    eng = login("engineer1")
    pid = make_pothole(app)
    assert client.post(f"/v1/potholes/{pid}/plan", json={"context": None}, headers=login("contractor2")).status_code \
        in (403, 404)
    r = client.post(f"/v1/potholes/{pid}/plan", json={"context": None}, headers=eng)
    assert r.status_code == 200, r.text
    plan = r.json()
    # PLAN 7.10 worked example geometry: 0.5 x 0.4 m, p95 60 mm -> A_cut 0.4875 m2, single 60 mm layer
    assert abs(plan["a_cut_m2"] - 0.4875) < 1e-3 and plan["d_design_mm"] == 60 and not plan["layered"]
    assert plan["bags_low"] <= plan["bags"] <= plan["bags_high"]
    pid_plan = plan["extras"]["repair_plan_id"]
    again = client.post(f"/v1/potholes/{pid}/plan", headers=eng).json()  # no body: default context
    assert again["extras"]["repair_plan_id"] != pid_plan
    old = client.get(f"/v1/plans/{pid_plan}", headers=eng).json()
    assert old["extras"]["superseded"] is True and old["pothole_id"] == pid
    assert client.get(f"/v1/potholes/{pid}", headers=eng).json()["pothole"]["status"] == "planned"
    flat = make_pothole(app, length_m=0.0, width_m=0.0, lat=C17_LAT + 0.004)
    with app.state.database.session() as s:
        from server.app.db import models as m

        s.get(m.Pothole, flat).area_m2 = 0.0
        s.commit()
    assert client.post(f"/v1/potholes/{flat}/plan", headers=eng).status_code == 422


def test_kpi_scorecard_contractors(client: TestClient, app: Any, login: Any) -> None:
    zonal, c1 = login("zonal_khairatabad"), login("contractor1")
    p_open = make_pothole(app)
    p_fixed = make_pothole(app, lat=C17_LAT + 0.003)
    t0 = now_utc() - timedelta(days=6)
    with app.state.database.session() as s:
        wo = sm.create_work_order(s, [p_fixed], actor="test", issue=True, now=t0)
        drive_to_claimed(s, wo, t0 + timedelta(hours=3))
        res = verification.evaluate(s, wo, t0 + timedelta(days=5), passes_provider(
            [("a", t0 + timedelta(days=1)), ("b", t0 + timedelta(days=2)), ("b", t0 + timedelta(days=3))]),
            qc_sampler=lambda _w, _f: False)
        assert res["outcome"] == "verified_closed"
        contractor_id = wo.contractor_id
        s.commit()

    for unit in ("all", "Khairatabad", "GHMC-C17"):
        r = client.get(f"/v1/kpi/{unit}?window=30", headers=zonal)
        assert r.status_code == 200, (unit, r.text)
        k = r.json()
        assert k["window_days"] == 30 and k["open_potholes"] >= 1 and k["repaired_verified"] >= 1
        assert k["by_status"].get("verified_closed") == 1 and k["by_status"].get("confirmed") == 1
        assert k["est_cost_inr"] == 0.0 or k["est_cost_inr"] > 0
        assert k["baseline_cost_inr"] == 7396.0 * k["open_potholes"]
    assert client.get("/v1/kpi/Khairatabad").json()["admin_unit"] == "Khairatabad"  # zone wins over circle
    assert client.get("/v1/kpi/nowhere").status_code == 404
    assert p_open

    card = client.get(f"/v1/contractors/{contractor_id}/scorecard", headers=c1)
    assert card.status_code == 200, card.text
    sc = card.json()
    assert sc["work_orders"] == 1 and sc["verified_closed"] == 1 and sc["sla_met_pct"] == 100.0
    assert sc["median_days_to_fix"] is not None and sc["median_days_to_fix"] < 1
    assert sc["survival"] and sc["survival"][0]["survival"] == 1.0 and sc["reopen_rate"] == 0.0
    assert client.get(f"/v1/contractors/{contractor_id}/scorecard", headers=login("contractor2")).status_code == 404
    mine = client.get("/v1/contractors", headers=c1).json()
    assert [c["id"] for c in mine] == [contractor_id] and "CRMP-P2" in mine[0]["package"]
    assert len(client.get("/v1/contractors", headers=zonal).json()) >= 7


def test_truth_read_endpoints(client: TestClient, app: Any, login: Any) -> None:
    eng = login("engineer1")
    pid = make_pothole(app)
    body = {"pothole_id": pid, "method": "manual_gauge", "depth_p95_mm": 55.0, "area_m2": 0.16, "width_m": 0.4,
            "measured_at": "2026-11-12T05:00:00Z", "measured_by": "AE Khairatabad"}
    assert client.post("/v1/truth/measurements", json=body, headers=eng).status_code == 201  # core measure router
    rows = client.get("/v1/truth/measurements", headers=eng).json()
    assert len(rows) == 1 and rows[0]["pothole_id"] == pid and rows[0]["method"] == "manual_gauge"
    pairs = client.get("/v1/truth/pairs", headers=eng).json()
    assert pairs["n"] == 1 and pairs["pairs"][0]["measured"]["depth_p95_mm"] == 60.0
    assert abs(pairs["summary"]["all"]["depth_bias_mm"] - 5.0) < 1e-9
    assert client.get("/v1/truth/measurements", headers=login("contractor1")).status_code == 403
