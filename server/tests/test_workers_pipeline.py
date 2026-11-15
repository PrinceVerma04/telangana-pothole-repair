"""WP6 acceptance (DECISIONS D2, in-process): 20 simulated observations from 4 trips over the same potholes posted
through POST /v1/ingest/observations in DEMO_MODE -> measurement rows (with sigma), potholes, confirmed where the
CONTRACTS rule says so; a forced depth failure leaves the observation 'verified' with measurement.reject_reason;
trip-close hooks run; ops ticks run on an empty and on a populated DB; the job poller drains queued work.

Geo: the real ``geo`` modules (WP5; ``GM_TEST_GEO=fake`` swaps in the test fakes). Services: the real WP8b
``server.app.services`` unless a test installs the fakes explicitly. Depth: the synthetic
backend on a rendered frame (no torch model is loaded). DB: tmp SQLite seeded from tests/fixtures/geo_small.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from server.app.core.util import now_utc
from server.app.db import models as m
from server.tests import test_workers_support as S


@pytest.fixture(scope="module")
def frame() -> dict[str, Any]:
    return S.synthetic_frame()


@pytest.fixture
def wenv(app: Any, settings: Any, frame: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    from measure.depth_backends.synthetic import SyntheticBackend
    from server.app.seed import seed_all
    from server.workers import adapters, install, stages, uninstall

    geo_mode = S.install_geo(monkeypatch)
    adapters.clear_caches()
    t = time.perf_counter()
    with app.state.database.session() as s:
        summary = seed_all(s, fixture=True)
    seed_s = time.perf_counter() - t
    install(settings, app.state.storage)
    stages.set_depth_backend(SyntheticBackend(depth=frame["depth"], tag="wp6-test"))
    yield {"geo": geo_mode, "seed": summary, "seed_s": seed_s}
    uninstall()
    adapters.clear_caches()


def _count(db: Any, model: Any, *where: Any) -> int:
    return int(db.scalar(select(func.count()).select_from(model).where(*where)) or 0)


def test_install_is_idempotent_and_reversible(app: Any, settings: Any) -> None:
    from server.app import pipeline
    from server.workers import install, installed, uninstall

    uninstall()  # create_app() installed the workers already: start from the pipeline defaults
    before = {n: pipeline.get_stage(n) for n in pipeline.STAGE_ORDER}
    assert all(f.__module__ == "server.app.pipeline" for f in before.values())
    try:
        s1 = install(settings, app.state.storage)
        s2 = install()
        assert s1 == s2 and installed()
        assert all(mod == "server.workers.stages" for mod in s1["stages"].values())
        assert s1["hooks"]["trip_close"].count("trip_close_hook") == 1
        assert "default_complaint_hook" not in s1["hooks"]["complaint"]
    finally:
        uninstall()
    assert {n: pipeline.get_stage(n) for n in pipeline.STAGE_ORDER} == before
    assert pipeline.default_complaint_hook in pipeline._HOOKS["complaint"]


def test_wp6_acceptance_20_observations(client: TestClient, app: Any, wenv: dict[str, Any],
                                        frame: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    t_all = time.perf_counter()
    sc = S.run_scenario(client, app, frame)
    with app.state.database.session() as db:
        obs = list(db.scalars(select(m.Observation).where(m.Observation.id.in_(sc["obs_ids"]))))
        assert len(obs) == 20
        assert all(o.status == "verified" for o in obs), [o.status for o in obs]
        assert all(o.processed_at is not None for o in obs)
        # map matching: every observation on the chosen road, direction 0 (along the digitised way)
        assert {(o.way_id, o.direction) for o in obs} == {(sc["way_id"], 0)}

        # measurement rows: one per detection; 19 measured + 1 forced depth failure
        meas = list(db.scalars(select(m.Measurement)))
        assert len(meas) == 20
        ok = [r for r in meas if r.reject_reason is None]
        bad = [r for r in meas if r.reject_reason is not None]
        assert len(ok) == 19 and len(bad) == 1
        assert bad[0].reject_reason.startswith("depth_failed: ValueError") and bad[0].depth_quality == "rejected"
        bad_obs = db.get(m.Observation, db.get(m.Detection, bad[0].detection_id).observation_id)
        assert bad_obs.status == "verified"  # WP6 acceptance: depth failure != rejected observation
        for r in ok:
            assert r.depth_sigma_mm is not None and r.depth_sigma_mm > 0
            assert r.area_sigma > 0 and r.volume_sigma is not None and r.width_sigma_m is not None
            assert r.method == "synthetic" and r.calib_id == "sim_bus" and r.certified is False
            assert 0.2 < r.quality <= 1.0 and 45.0 < r.depth_p95_mm < 70.0  # rendered cap: p95 ~57 mm
            assert r.depth_map_key and app.state.storage.exists(r.depth_map_key)

        # potholes: P1..P3 seen on 4 trips -> confirmed; P4 seen once -> candidate
        ps = list(db.scalars(select(m.Pothole).order_by(m.Pothole.chainage_m)))
        assert len(ps) == 4, [(p.id, p.lat, p.lon, p.status) for p in ps]
        for p, truth in zip(ps, sc["potholes"], strict=True):
            assert S.hav(p.lat, p.lon, truth["lat"], truth["lon"]) < 4.0
            assert p.way_id == sc["way_id"] and p.direction == 0 and p.lane_band == 1
        assert [p.status for p in ps] == ["confirmed", "confirmed", "confirmed", "candidate"]
        assert [p.trips_seen for p in ps] == [4, 4, 4, 1]
        for p in ps:
            assert p.depth_p95_mm is not None and 45.0 < p.depth_p95_mm < 70.0
            assert p.depth_sigma_mm is not None and p.volume_sigma is not None
            assert p.severity == "large"  # IRC:82: depth_p95 > 50 mm
            assert p.priority is not None and 0 < p.priority <= 100 and set("SEVTM") <= set(p.priority_breakdown)
            assert p.agency_id is not None and p.admin_unit_id is not None
            assert p.passes_total >= p.passes_seen  # trip close counted the camera-on passes
        # one pass_measurement per (trip, pothole): 3 x 4 + P4 x 1; trip 4 / P2 holds only the failed frame
        assert _count(db, m.PassMeasurementRow) == 13
        rej = list(db.scalars(select(m.PassMeasurementRow).where(m.PassMeasurementRow.reject_reason.is_not(None))))
        assert len(rej) == 1 and rej[0].pothole_id == ps[1].id

        # plans for confirmed potholes (estimate only; pothole stays 'confirmed' unless GM_AUTO_PLAN_STATUS=1)
        for p in ps[:3]:
            rows = list(db.scalars(select(m.RepairPlanRow).where(m.RepairPlanRow.pothole_id == p.id,
                                                                 m.RepairPlanRow.superseded.is_(False))))
            assert len(rows) == 1
            plan = rows[0].json
            assert plan["bags_low"] <= plan["bags"] <= plan["bags_high"] and rows[0].est_cost_inr > 0
        assert _count(db, m.RepairPlanRow, m.RepairPlanRow.pothole_id == ps[3].id) == 0

        # trip-close hook effects: fixes map-matched, one way pass per trip and day
        assert _count(db, m.TripFix, m.TripFix.way_id == sc["way_id"]) > 0
        wp = list(db.scalars(select(m.WayPass).where(m.WayPass.way_id == sc["way_id"], m.WayPass.direction == 0)))
        assert sum(w.n_passes for w in wp) == 4 and sum(w.n_cam_active for w in wp) == 4
        p4_id = ps[3].id

    # complaint hook: a citizen report within 25 m of the single-pass P4 confirms it (CONTRACTS rule)
    truth = sc["potholes"][3]
    r = client.post("/v1/citizen/reports", data={"lat": str(truth["lat"] + 3e-5), "lon": str(truth["lon"])})
    assert r.status_code == 201, r.text
    assert r.json()["pothole_id"] == p4_id and r.json()["status"] == "confirmed"

    # API view of the result
    admin = S.login(client, "admin")  # engineer1 is scoped to circle C17; the fixture road may lie elsewhere
    page = client.get("/v1/potholes?status=confirmed", headers=admin).json()
    assert page["total"] == 4
    d = client.get(f"/v1/potholes/{ps[0].id}", headers=admin).json()
    assert len(d["measurements"]) == 8 and len(d["passes"]) == 4 and d["plan"]["bags"] >= 1
    total_s = time.perf_counter() - t_all
    print("\nWP6 timing " + json.dumps({"geo": wenv["geo"], "seed_fixture_s": round(wenv["seed_s"], 2),
                                        **{k: round(v, 2) for k, v in sc["timing"].items()},
                                        "scenario_total_s": round(total_s, 2)}))


def test_auto_plan_status_flag(client: TestClient, app: Any, wenv: dict[str, Any], frame: dict[str, Any],
                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GM_AUTO_PLAN_STATUS", "1")
    S.run_scenario(client, app, frame, close_trips=False)
    with app.state.database.session() as db:
        st = sorted(db.scalars(select(m.Pothole.status)))
    assert st == ["candidate", "planned", "planned", "planned"]


def test_ops_ticks_empty_and_populated(client: TestClient, app: Any, settings: Any, wenv: dict[str, Any],
                                       frame: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from server.workers import ops, run

    # empty DB through the CLI (real services, nothing to do, exit 0)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "cli_store"))
    empty = f"sqlite:///{tmp_path}/empty.db"
    assert run.main(["--tick", "--database-url", empty]) == 0
    from server.workers import install

    install(settings, app.state.storage)  # the CLI re-pointed the runtime at its own settings

    # populated DB, real services
    sc = S.run_scenario(client, app, frame)
    with app.state.database.session() as db:
        res = ops.run_ticks(db)
    assert not ops.tick_errors(res), res
    assert all("skipped" not in v for v in res.values()), res
    assert res["archive_candidates"]["archived"] == 0

    # a broken deployment (service modules missing) is reported as tick errors, never silently skipped
    with monkeypatch.context() as mp:
        S.hide_services(mp)
        with app.state.database.session() as db:
            broken = ops.run_ticks(db, names=["sla_tick", "kpi_rollup"], commit=False)
    assert {e.split(":")[0] for e in ops.tick_errors(broken)} == {"sla_tick", "kpi_rollup"}, broken

    # populated DB with (fake) WP8b services: each tick calls its service with the pinned arguments
    calls = S.install_fake_services(monkeypatch)
    with app.state.database.session() as db:
        p1, p2 = list(db.scalars(select(m.Pothole).order_by(m.Pothole.chainage_m)))[:2]
        wo_v = m.WorkOrder(status="repaired_claimed", agency_id=p1.agency_id)
        wo_i = m.WorkOrder(status="issued", agency_id=p2.agency_id)
        db.add_all([wo_v, wo_i])
        db.flush()
        db.add_all([m.WorkOrderPothole(work_order_id=wo_v.id, pothole_id=p1.id),
                    m.WorkOrderPothole(work_order_id=wo_i.id, pothole_id=p2.id)])
        db.commit()
        p1_id, wo_v_id = p1.id, wo_v.id
    # two later camera-on trips with no detection = clean passes; trip close evaluates the WO it drove past
    from datetime import timedelta

    for k, dev in enumerate(("sim-bus-02", "sim-bus-03")):
        S.drive_trip(client, sc["headers"], sc["coords"], dev, f"cccccccc-0000-4000-8000-00000000000{k}",
                     sc["t0"] + timedelta(days=10 + k))
    ev = [c for c in calls if c[0] == "evaluate"]
    assert ev and all(c[1] == wo_v_id for c in ev)
    clean = ev[-1][2][p1_id]  # the 4 scenario trips re-detected P1 -> not clean; the 2 new trips are clean
    assert len(clean) == 2, clean
    with app.state.database.session() as db:
        p1 = db.get(m.Pothole, p1_id)
        p1.status = "verified_closed"  # + a sighting after its last event -> dlp_watch must look at it
        closed_at = now_utc()
        db.add(m.PotholeEvent(pothole_id=p1.id, ts=closed_at, from_status="verification_pending",
                              to_status="verified_closed", reason="test"))
        p1.last_seen = closed_at + timedelta(minutes=5)
        db.commit()
        calls.clear()
        res = ops.run_ticks(db, sc["t0"] + timedelta(days=12))
    assert not ops.tick_errors(res), res
    names = calls.names()
    assert {"sla_tick", "evaluate", "dlp_watch", "generate", "kpi", "scorecard"} <= set(names), (names, res)
    assert res["daily_indent"]["circles"] == 1 and res["dlp_watch"]["checked"] == 1
    assert res["kpi_rollup"]["snapshot_key"] and app.state.storage.exists(res["kpi_rollup"]["snapshot_key"])


def test_poller_drains_jobs_and_pending(client: TestClient, app: Any, settings: Any, wenv: dict[str, Any],
                                        frame: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    from server.workers import run

    sc = S.run_scenario(client, app, frame, close_trips=False)
    with app.state.database.session() as db:
        src = db.get(m.Observation, sc["obs_ids"][0])
        src_det = db.scalars(select(m.Detection).where(m.Detection.observation_id == src.id)).first()
        clone = m.Observation(client_uuid="bbbbbbbb-0000-4000-8000-000000000001", device_id=src.device_id,
                              trip_id=src.trip_id, seq=99, captured_at=src.captured_at, source="sim",
                              calib_id=src.calib_id, model_version="x", lat_raw=src.lat_raw, lon_raw=src.lon_raw,
                              gps_accuracy_m=src.gps_accuracy_m, speed_mps=src.speed_mps,
                              heading_deg=src.heading_deg, frame_key=src.frame_key, status="pending",
                              payload=src.payload)
        db.add(clone)
        db.flush()
        db.add(m.Detection(observation_id=clone.id, track_id=None, cls="pothole", conf=0.9, bbox=src_det.bbox,
                           polygon=src_det.polygon))
        job = run.enqueue(db, "process_observation", {"observation_id": clone.id})
        bad = run.enqueue(db, "process_observation", {"observation_id": 999999})
        tick = run.enqueue(db, "ops_tick", {"ticks": ["archive_candidates"]})
        db.commit()
        counts = run.drain(db, settings, app.state.storage, max_jobs=10)
        assert counts["done"] == 2 and counts["requeued"] == 1, counts
        db.expire_all()
        assert db.get(m.Job, job.id).status == "done" and db.get(m.Job, tick.id).status == "done"
        assert db.get(m.Job, bad.id).status == "queued" and "LookupError" in db.get(m.Job, bad.id).error
        assert db.get(m.Observation, clone.id).status == "verified"
        assert _count(db, m.Measurement, m.Measurement.detection_id.in_(
            select(m.Detection.id).where(m.Detection.observation_id == clone.id))) == 1
        for _ in range(3):
            run.drain(db, settings, app.state.storage, max_jobs=10)
        db.expire_all()
        assert db.get(m.Job, bad.id).status == "failed" and db.get(m.Job, bad.id).attempts == run.MAX_ATTEMPTS


def test_repair_loop_with_real_services(client: TestClient, app: Any, settings: Any, wenv: dict[str, Any],
                                        frame: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """claim -> 3 clean bus passes (trip close) -> verified_closed -> re-detection -> services.dlp -> reopened."""
    from datetime import timedelta

    from server.app.services.verification import qc_sampled, thresholds
    from server.workers import ops

    sc = S.run_scenario(client, app, frame)
    t0 = sc["t0"]
    with app.state.database.session() as db:
        p1 = db.scalars(select(m.Pothole).order_by(m.Pothole.chainage_m)).first()
        db.add(m.WorkOrder(status="cancelled", agency_id=p1.agency_id))  # id 1 falls in the 10 % QC sample
        db.flush()
        wo = m.WorkOrder(status="repaired_claimed", agency_id=p1.agency_id, contract_id=p1.contract_id,
                         contractor_id=p1.contractor_id)
        db.add(wo)
        db.flush()
        db.add(m.WorkOrderPothole(work_order_id=wo.id, pothole_id=p1.id))
        db.add(m.RepairEvent(work_order_id=wo.id, kind="bags", at=t0 + timedelta(days=9), data={"bags": 3}))
        p1.status = "repaired_claimed"
        db.commit()
        p1_id, wo_id = p1.id, wo.id
        sampled = qc_sampled(wo.id, thresholds(db, wo)["qc_sample_frac"])
        assert not sampled, "pick a work order id outside the QC sample so the DLP branch is exercised"
    for k, dev in enumerate(("sim-bus-02", "sim-bus-03", "sim-bus-01")):
        S.drive_trip(client, sc["headers"], sc["coords"], dev, f"dddddddd-0000-4000-8000-00000000000{k}",
                     t0 + timedelta(days=10 + k))
    with app.state.database.session() as db:
        wo, p1 = db.get(m.WorkOrder, wo_id), db.get(m.Pothole, p1_id)
        expect = "verification_pending" if sampled else "verified_closed"
        assert (wo.status, p1.status) == (expect, expect)
        ver = list(db.scalars(select(m.Verification).where(m.Verification.work_order_id == wo_id)))
        assert ver and ver[-1].clean_passes >= 3 and ver[-1].distinct_trips >= 2
    if not sampled:  # re-detection after the verified repair -> reopened (+ DLP evidence package)
        h = sc["headers"]
        la, lo, brg = S.along(sc["coords"], sc["potholes"][0]["chainage"] - S.FIX_BACK_M)
        o = S.observation(device_id="sim-bus-02", trip_uuid="eeeeeeee-0000-4000-8000-000000000001", seq=0,
                          ts=t0 + timedelta(days=14), lat=la, lon=lo, heading=brg, frame_key="frames/test/good.png",
                          frame=frame, track_id=1)
        r = client.post("/v1/ingest/observations", json={"observations": [o]}, headers=h)
        assert r.status_code == 200 and r.json()["accepted"] == 1, r.text
        with app.state.database.session() as db:
            assert db.get(m.Pothole, p1_id).status == "reopened"
            assert _count(db, m.DlpEvent, m.DlpEvent.pothole_id == p1_id) == 1
    with app.state.database.session() as db:  # every ops tick with the real services on the populated DB
        res = ops.run_ticks(db, t0 + timedelta(days=15))
    assert not ops.tick_errors(res), res
    assert all("skipped" not in v for v in res.values()), res
    # the CLI on the same populated database (python -m server.workers.run --tick)
    from server.workers import install, run

    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "cli_store"))
    assert run.main(["--tick", "--database-url", settings.DATABASE_URL, "--now", (t0 + timedelta(days=16)).isoformat()]) == 0
    install(settings, app.state.storage)
