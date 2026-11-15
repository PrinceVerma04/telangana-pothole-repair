"""WP8b acceptance: daily repair run (nearest-neighbour route, crew cap) and indent totals == sum of plans;
Measurement Book / invoice-eligible exports contain only verified_closed work orders; GeoJSON export."""

from __future__ import annotations

import csv
import io
from datetime import timedelta
from typing import Any

from fastapi.testclient import TestClient

from server.app.core.util import haversine_m, now_utc, to_ist
from server.app.db import models as m
from server.app.services import state_machine as sm
from server.app.services import verification
from server.tests.test_ops_support import C17_LAT, C17_LON, drive_to_claimed, make_pothole, passes_provider


def _plan(client: TestClient, pid: int, h: dict[str, str]) -> dict[str, Any]:
    return client.get(f"/v1/potholes/{pid}", headers=h).json()["plan"]


def test_run_generate_route_and_indent_totals(client: TestClient, app: Any, login: Any) -> None:
    eng, c1 = login("engineer1"), login("contractor1")
    # three potholes at increasing distance from the circle office (centroid of the GHMC-C17 box), varied sizes
    pa = make_pothole(app, lat=C17_LAT + 0.010, lon=C17_LON, length_m=0.9, width_m=0.7, depth_p95_mm=90)
    pb = make_pothole(app, lat=C17_LAT + 0.001, lon=C17_LON, contract_code="CRMP-P7")
    pc = make_pothole(app, lat=C17_LAT + 0.004, lon=C17_LON + 0.002, length_m=0.3, width_m=0.25, depth_p95_mm=35)
    w1 = client.post("/v1/work-orders", json={"pothole_ids": [pa, pc]}, headers=eng).json()
    w2 = client.post("/v1/work-orders", json={"pothole_ids": [pb]}, headers=eng).json()
    assert w1["status"] == w2["status"] == "issued"
    planned_only = make_pothole(app, lat=C17_LAT, lon=C17_LON + 0.003)
    w3 = client.post("/v1/work-orders?issue=false", json={"pothole_ids": [planned_only]}, headers=eng).json()

    today = to_ist(now_utc()).date().isoformat()
    assert client.post(f"/v1/runs/generate?circle=GHMC-C17&date={today}", headers=c1).status_code == 403
    r = client.post(f"/v1/runs/generate?circle=GHMC-C17&date={today}", headers=eng)
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["circle_name"] == "Khairatabad" and run["run_date"] == today
    assert sorted(run["work_order_ids"]) == sorted([w1["id"], w2["id"]])  # planned WO w3 not scheduled
    assert w3["id"] not in run["work_order_ids"]
    stops = run["route"]
    assert [s["pothole_id"] for s in stops] == [pb, pc, pa]  # greedy nearest neighbour from the office
    assert abs(stops[0]["hop_m"] - haversine_m(C17_LAT, C17_LON, C17_LAT + 0.001, C17_LON)) < 2.0

    plans = [_plan(client, pid, eng) for pid in (pa, pb, pc)]
    t = run["totals"]
    assert t["bags"] == sum(p["bags"] for p in plans)
    assert t["bags_low"] == sum(p["bags_low"] for p in plans) and t["bags_high"] == sum(p["bags_high"] for p in plans)
    assert abs(t["cost_inr"] - sum(p["est_cost_inr"] for p in plans)) < 0.05
    assert abs(t["crew_h"] - sum(p["labour_h"] for p in plans)) < 1e-3
    assert abs(t["crm_t"] - sum(p["crm_t"] for p in plans)) < 1e-3 and t["crm_t"] > 0  # 90 mm hole is layered

    csv_r = client.get(f"/v1/runs/{run['id']}/indent.csv", headers=eng)
    assert csv_r.status_code == 200 and csv_r.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(csv_r.text)))
    body = [x for x in rows[2:] if x[0] != "TOTAL"]
    total = next(x for x in rows if x[0] == "TOTAL")
    assert [int(x[2]) for x in body] == [pb, pc, pa]
    assert int(total[10]) == sum(p["bags"] for p in plans) == sum(int(x[10]) for x in body)
    assert abs(float(total[17]) - t["cost_inr"]) < 0.05

    # WOs carry the run; contractor sees the run of its own WOs; regenerating keeps the same run
    assert client.get(f"/v1/work-orders/{w1['id']}", headers=eng).json()["run_id"] == run["id"]
    assert client.get(f"/v1/runs/{run['id']}", headers=c1).status_code == 200
    again = client.post(f"/v1/runs/generate?circle=GHMC-C17&date={today}", headers=eng).json()
    assert again["id"] == run["id"] and again["totals"]["bags"] == t["bags"]
    # tight crew cap: one WO fits, the other is deferred to a later run
    capped = client.post(f"/v1/runs/generate?circle=GHMC-C17&date={today}&crew_hours_cap=0.5", headers=eng).json()
    assert len(capped["work_order_ids"]) == 1 and capped["totals"]["deferred_work_orders"] == 1
    assert client.post("/v1/runs/generate?circle=GHMC-KHT-NOPE", headers=eng).status_code == 404
    assert client.get("/v1/runs/99999", headers=eng).status_code == 404


def test_measurement_book_only_verified_closed(client: TestClient, app: Any, login: Any) -> None:
    eng, c1, c2 = login("engineer1"), login("contractor1"), login("contractor2")
    p_done = make_pothole(app)
    p_claimed = make_pothole(app, lat=C17_LAT + 0.002)
    p_other = make_pothole(app, lat=C17_LAT + 0.003, contract_code="CRMP-P7")
    t0 = now_utc()
    with app.state.database.session() as s:
        ids = {}
        for name, pid in (("done", p_done), ("claimed", p_claimed), ("other", p_other)):
            wo = sm.create_work_order(s, [pid], actor="test", issue=True, now=t0)
            drive_to_claimed(s, wo, t0)
            ids[name] = wo.id
        good = passes_provider([("t1", t0 + timedelta(days=1)), ("t2", t0 + timedelta(days=2)),
                                ("t3", t0 + timedelta(days=2, hours=3))])
        for name in ("done", "other"):
            res = verification.evaluate(s, s.get(m.WorkOrder, ids[name]), t0 + timedelta(days=3), good,
                                        qc_sampler=lambda _w, _f: False)
            assert res["outcome"] == "verified_closed"
        s.commit()

    mb = client.get("/v1/exports/measurement-book.csv", headers=eng)
    assert mb.status_code == 200 and mb.headers["content-type"].startswith("text/csv")
    assert f"WO {ids['done']}" in mb.text and f"WO {ids['other']}" in mb.text
    assert f"WO {ids['claimed']}" not in mb.text
    rows = list(csv.reader(io.StringIO(mb.text)))
    assert rows[0][0] == "Item No" and any(r[0] == "TOTAL" for r in rows)
    assert "uncertified" in mb.text  # demo calibration -> not billing-grade
    sel = client.get(f"/v1/exports/measurement-book.csv?work_order_ids={ids['done']},{ids['claimed']}", headers=eng)
    assert sel.headers["x-excluded-work-orders"] == str(ids["claimed"])
    assert sel.headers["x-eligible-work-orders"] == str(ids["done"])
    # contractor-scoped: each contractor's MB holds only its own verified work
    own = client.get("/v1/exports/measurement-book.csv", headers=c1).text
    assert f"WO {ids['done']}" in own and f"WO {ids['other']}" not in own
    assert [w["id"] for w in client.get("/v1/exports/invoice-eligible", headers=c2).json()] == [ids["other"]]
    assert sorted(w["id"] for w in client.get("/v1/exports/invoice-eligible", headers=eng).json()) == \
        sorted([ids["done"], ids["other"]])
    assert client.get("/v1/exports/measurement-book.csv", headers=login("citizen1")).status_code == 403


def test_potholes_geojson_export(client: TestClient, app: Any, login: Any) -> None:
    pid = make_pothole(app)
    cand = make_pothole(app, status="candidate", lat=C17_LAT + 0.005)
    fc = client.get("/v1/exports/potholes.geojson", headers=login("engineer1")).json()
    assert fc["type"] == "FeatureCollection"
    by_id = {f["id"]: f for f in fc["features"]}
    assert {pid, cand} <= set(by_id)
    props = by_id[pid]["properties"]
    assert props["depth_p95_mm"] == 60.0 and props["depth_sigma_mm"] == 8.0 and props["area_sigma"] == 0.02
    public = client.get("/v1/exports/potholes.geojson").json()  # anonymous: candidates hidden
    assert pid in {f["id"] for f in public["features"]} and cand not in {f["id"] for f in public["features"]}
    only = client.get("/v1/exports/potholes.geojson?status=candidate", headers=login("engineer1")).json()
    assert [f["id"] for f in only["features"]] == [cand]
