"""Wave 2 batch A end-to-end (WP5 geo + WP6 workers + WP8b ops) through the HTTP API, in-process (DECISIONS D2).

Fixture geo DB (``tests/fixtures/geo_small``), synthetic depth backend on a rendered frame (no torch model), real
``geo.*`` and ``server.app.services.*``, workers installed by ``create_app``. One flow:

    2 trips of observations over 2 potholes -> confirmed (>= 2 trips) -> scored (IRC:82 severity + priority) and
    planned (pipeline plan row; POST /potholes/{id}/plan -> planned) -> work order created + issued ->
    acknowledged -> in_progress (before photo) -> repaired_claimed (after photo, bags used) ->
    trip close of 2 camera-on trips giving 3 clean passes (one trip loops twice) -> verified_closed ->
    a later re-detection -> reopened + dlp_event (evidence package, zero-pay rework drafted) ;
    an observation on an unowned road segment -> jurisdiction_queue item (provisional agency for visibility).

The 10 % engineer QC sample is switched off here (it is covered by the WP8b tests) so the bus path decides.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update

from server.app.core.config import Settings
from server.app.core.util import now_utc
from server.app.db import models as m
from server.tests import test_workers_support as S

LATERAL_M = S.LATERAL_M
CHAINAGES = (40.0, 100.0)  # P1, P2 along the chosen road (m)
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 64 + b"\xff\xd9"


# =========================================================================== fixtures
@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    from measure.depth_backends.synthetic import SyntheticBackend
    from server.app.main import create_app
    from server.app.seed import seed_all
    from server.app.services import verification
    from server.workers import adapters, installed, stages, uninstall

    monkeypatch.delenv("GM_JURISDICTION_POLICY", raising=False)
    monkeypatch.delenv("GM_JURISDICTION_ADMIN_FALLBACK", raising=False)
    monkeypatch.delenv("GM_AUTO_PLAN_STATUS", raising=False)
    monkeypatch.setattr(verification, "qc_sampled", lambda _wo_id, _frac: False)
    settings = Settings(DATABASE_URL=f"sqlite:///{tmp_path}/e2e.db", STORAGE_ROOT=str(tmp_path / "store"),
                        DEMO_MODE=True, JWT_SECRET="e2e-secret-not-for-production-0123456789",
                        API_BASE_URL="http://testserver", _env_file=None)
    a = create_app(settings)
    assert installed(), "create_app must install the WP6 workers"
    adapters.clear_caches()
    with a.state.database.session() as s:
        seed_all(s, fixture=True)
    frame = S.synthetic_frame()
    stages.set_depth_backend(SyntheticBackend(depth=frame["depth"], tag="e2e"))
    a.state.e2e_frame = frame
    yield a
    uninstall()
    adapters.clear_caches()
    a.state.database.dispose()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


# =========================================================================== helpers
def _isolated_road(db: Any, *, exclude_way: int | None = None, chainages: tuple[float, ...] = CHAINAGES) -> \
        tuple[m.RoadSegment, list[tuple[float, float]]]:
    """A two-way road with no other way within 25 m of the given chainages (and far from ``exclude_way``)."""
    from server.workers import adapters

    min_len = max(chainages) + 20.0
    rows = db.scalars(select(m.RoadSegment).where(m.RoadSegment.oneway.is_(False), m.RoadSegment.length_m >= min_len)
                      .order_by(m.RoadSegment.length_m.desc(), m.RoadSegment.id)).all()
    for r in rows:
        if r.osm_way_id == exclude_way:
            continue
        coords = adapters.parse_linestring(r.geom_wkt)
        if len(coords) < 2 or S.polyline_length(coords) < min_len:
            continue
        pts = [S.along(coords, c) for c in (5.0, *chainages)]
        idx = adapters.road_index_for(db, [(la, lo) for la, lo, _ in pts])
        if idx is None:
            continue
        if all(seg.way_id == r.osm_way_id for la, lo, _ in pts for seg, _d, _b in idx.nearest(la, lo, 60.0)):
            return r, coords
    raise AssertionError("no isolated two-way road in the geo_small fixture")


def _observations(coords: list[tuple[float, float]], chainages: tuple[float, ...], device: str, trip: str,
                  t0: datetime, frame: dict[str, Any], frames_per_pothole: int = 2) -> list[dict[str, Any]]:
    out = []
    seq = 0
    for k, c in enumerate(chainages):
        for j in range(frames_per_pothole):
            la, lo, brg = S.along(coords, c - S.FIX_BACK_M + 0.5 * j)
            out.append(S.observation(device_id=device, trip_uuid=trip, seq=seq, ts=t0 + timedelta(seconds=3 * seq),
                                     lat=la, lon=lo, heading=brg, frame_key="frames/e2e/good.png", frame=frame,
                                     track_id=k + 1))
            seq += 1
    return out


def _drive(client: TestClient, headers: dict[str, str], coords: list[tuple[float, float]], device: str, trip: str,
           start: datetime, laps: int = 1, speed: float = 9.0) -> dict[str, Any]:
    """1 Hz camera-on fixes along the road (``laps`` times, 10 min apart, same direction), then trip close."""
    fixes = []
    L = S.polyline_length(coords)
    t = start
    for _lap in range(laps):
        c = 0.0
        while c <= L:
            la, lo, brg = S.along(coords, c)
            fixes.append({"ts": t.isoformat(), "lat": la, "lon": lo, "accuracy_m": 4.0, "speed_mps": speed,
                          "heading_deg": brg % 360.0, "az_rms": 0.05, "cam_active": True})
            c += speed
            t += timedelta(seconds=1)
        t += timedelta(minutes=10)
    r = client.post("/v1/ingest/trip-fixes", json={"device_id": device, "trip_id": trip, "fixes": fixes},
                    headers=headers)
    assert r.status_code == 200, r.text
    r = client.post("/v1/ingest/trip-close", headers=headers,
                    json={"device_id": device, "trip_id": trip, "ended_at": (t + timedelta(seconds=5)).isoformat()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert all(h == "ok" for h in body["hooks"]), body
    return body


def _move(client: TestClient, wid: int, to: str, headers: dict[str, str], **extra: Any) -> dict[str, Any]:
    r = client.patch(f"/v1/work-orders/{wid}/transition", headers=headers, json={"to": to, **extra})
    assert r.status_code == 200, (to, r.text)
    return r.json()


def _photo(client: TestClient, wid: int, kind: str, headers: dict[str, str], lat: float, lon: float) -> None:
    r = client.post(f"/v1/work-orders/{wid}/photos", headers=headers, data={"kind": kind, "lat": str(lat),
                                                                          "lon": str(lon)},
                    files={"file": (f"{kind}.jpg", JPEG, "image/jpeg")})
    assert r.status_code == 201, r.text


def _trip() -> str:
    return str(uuid.uuid4())


# =========================================================================== the flow
def test_ops_flow_end_to_end(client: TestClient, app: FastAPI) -> None:
    t_start = time.perf_counter()
    frame = app.state.e2e_frame
    st = app.state.storage
    st.put("frames/e2e/good.png", frame["png"], "image/png")
    with app.state.database.session() as db:
        road, coords = _isolated_road(db)
        way_id = road.osm_way_id
        p2 = db.scalars(select(m.Contract).where(m.Contract.code == "CRMP-P2")).one()
        contract_id, contractor_id = p2.id, p2.contractor_id
    truth = []
    for c in CHAINAGES:
        la, lo, brg = S.along(coords, c)
        truth.append(S.offset(la, lo, brg, 0.0, LATERAL_M))

    dev_h = S.login(client, "admin")  # ingest accepts staff tokens in DEMO_MODE (sim devices are auto-created)
    admin = dev_h
    c1 = S.login(client, "contractor1")
    now = now_utc()
    t0 = now - timedelta(days=2)

    # ---- 1. two trips of observations over two potholes -> confirmed ------------------------------------------
    trips = [("sim-bus-01", _trip(), t0), ("sim-bus-02", _trip(), t0 + timedelta(hours=3))]
    for dev, trip, ts in trips:
        batch = _observations(coords, CHAINAGES, dev, trip, ts, frame)
        r = client.post("/v1/ingest/observations", json={"observations": batch}, headers=dev_h)
        assert r.status_code == 200 and r.json()["accepted"] == len(batch) and not r.json()["rejected"], r.text
        _drive(client, dev_h, coords, dev, trip, ts - timedelta(seconds=5))
    with app.state.database.session() as db:
        ps = list(db.scalars(select(m.Pothole).order_by(m.Pothole.chainage_m)))
        assert len(ps) == 2, [(p.id, p.lat, p.lon, p.status) for p in ps]
        for p, (tla, tlo) in zip(ps, truth, strict=True):
            assert S.hav(p.lat, p.lon, tla, tlo) < 4.0
            assert (p.way_id, p.direction) == (way_id, 0)
            assert p.status == "confirmed" and p.trips_seen == 2
            # ---- 2. scored + planned by the pipeline ---------------------------------------------------------
            assert p.depth_p95_mm is not None and p.depth_sigma_mm is not None and p.volume_sigma is not None
            assert p.severity == "large" and p.priority is not None and set("SEVTM") <= set(p.priority_breakdown)
            assert p.agency_id is not None and p.admin_unit_id is not None
            plan_row = db.scalars(select(m.RepairPlanRow).where(m.RepairPlanRow.pothole_id == p.id,
                                                                m.RepairPlanRow.superseded.is_(False))).one()
            assert plan_row.json["bags_low"] <= plan_row.json["bags"] <= plan_row.json["bags_high"]
            assert plan_row.json["extras"]["input_sig"]  # pipeline plan via services.runs.plan_pothole
        pids = [p.id for p in ps]
        assert db.scalar(select(func.count()).select_from(m.JurisdictionQueue)) == 0  # owned road

    # explicit plan through the API: confirmed -> planned (same service, new plan supersedes the pipeline's)
    r = client.post(f"/v1/potholes/{pids[0]}/plan", headers=admin)
    assert r.status_code == 200, r.text
    plan = r.json()
    assert plan["bags_low"] <= plan["bags"] <= plan["bags_high"] and plan["est_cost_inr"] > 0
    assert client.get(f"/v1/potholes/{pids[0]}", headers=admin).json()["pothole"]["status"] == "planned"

    # ---- 3. work order created + issued ---------------------------------------------------------------------------
    r = client.post("/v1/work-orders", json={"pothole_ids": pids, "contract_id": contract_id, "crew": "crew A"},
                    headers=admin)
    assert r.status_code == 201, r.text
    wo = r.json()
    wid = wo["id"]
    assert wo["status"] == "issued" and wo["contractor_id"] == contractor_id and wo["sla_due"]
    assert sorted(wo["pothole_ids"]) == sorted(pids)
    for pid in pids:
        assert client.get(f"/v1/potholes/{pid}", headers=admin).json()["pothole"]["status"] == "work_ordered"

    # ---- 4. contractor: acknowledged -> in_progress (before photo) -> repaired_claimed (after photo, bags) ----
    _move(client, wid, "acknowledged", c1)
    _photo(client, wid, "before", c1, *truth[0])
    _move(client, wid, "in_progress", c1, bag_lot_no="LOT-E2E-1")
    _photo(client, wid, "after", c1, *truth[0])
    claimed = _move(client, wid, "repaired_claimed", c1, bags_used=max(1, int(wo["est_bags"] or 1)))
    assert claimed["status"] == "repaired_claimed" and len(claimed["photos"]) == 2

    # ---- 5. bus verification on trip close: 3 clean passes from 2 trips (trip C loops twice) -----------------
    claim = now_utc()
    c_trip, d_trip = _trip(), _trip()
    _drive(client, dev_h, coords, "sim-bus-03", c_trip, claim + timedelta(days=1), laps=2)
    with app.state.database.session() as db:
        w = db.get(m.WorkOrder, wid)
        assert w.status == "verification_pending"  # 2 passes, 1 trip: not enough yet
        v = db.scalars(select(m.Verification).where(m.Verification.work_order_id == wid)).all()
        assert v and all((x.clean_passes, x.distinct_trips) == (2, 1) for x in v), [(x.clean_passes, x.distinct_trips)
                                                                                     for x in v]
    _drive(client, dev_h, coords, "sim-bus-01", d_trip, claim + timedelta(days=2))
    with app.state.database.session() as db:
        w = db.get(m.WorkOrder, wid)
        assert w.status == "verified_closed"
        assert [db.get(m.Pothole, pid).status for pid in pids] == ["verified_closed", "verified_closed"]
        v = db.scalars(select(m.Verification).where(m.Verification.work_order_id == wid)).all()
        assert len(v) == 2 and all(x.clean_passes >= 3 and x.distinct_trips >= 2 and x.outcome == "verified_closed"
                                   for x in v)
    eligible = client.get("/v1/exports/invoice-eligible", headers=admin).json()
    assert [w["id"] for w in eligible] == [wid]

    # ---- 6. later re-detection of P1 -> reopened + dlp_event (evidence package, not an automatic penalty) -----
    e_trip = _trip()
    obs = _observations(coords, CHAINAGES[:1], "sim-bus-02", e_trip, claim + timedelta(days=5), frame, 1)
    r = client.post("/v1/ingest/observations", json={"observations": obs}, headers=dev_h)
    assert r.status_code == 200 and r.json()["accepted"] == 1, r.text
    with app.state.database.session() as db:
        p1 = db.get(m.Pothole, pids[0])
        assert p1.status == "reopened"
        assert db.get(m.Pothole, pids[1]).status == "verified_closed"
        ev = db.scalars(select(m.DlpEvent).where(m.DlpEvent.pothole_id == pids[0])).one()
        assert ev.work_order_id == wid and ev.contract_id == contract_id and ev.evidence
        assert "penalty" in str(ev.evidence).lower()
        rework = db.get(m.WorkOrder, ev.rework_work_order_id)
        assert rework.is_rework and rework.parent_work_order_id == wid and rework.status == "planned"
        assert db.scalar(select(func.count()).select_from(m.PotholeObservation)
                         .where(m.PotholeObservation.pothole_id == pids[0])) == 5  # 4 + the re-detection

    # ---- 7. unowned road segment -> jurisdiction queue item ------------------------------------------------------
    with app.state.database.session() as db:
        other, ocoords = _isolated_road(db, exclude_way=way_id, chainages=(40.0,))
        db.execute(update(m.RoadSegment).where(m.RoadSegment.osm_way_id == other.osm_way_id)
                   .values(owner_agency_id=None, contract_id=None))
        db.commit()
        other_way = other.osm_way_id
    from server.workers import adapters

    adapters.clear_caches()  # the road index caches SegmentRec owners
    obs = _observations(ocoords, (40.0,), "sim-bus-03", _trip(), claim + timedelta(days=6), frame, 1)
    r = client.post("/v1/ingest/observations", json={"observations": obs}, headers=dev_h)
    assert r.status_code == 200 and r.json()["accepted"] == 1, r.text
    with app.state.database.session() as db:
        q = db.scalars(select(m.Pothole).where(m.Pothole.way_id == other_way)).one()
        assert q.status == "candidate" and q.admin_unit_id is not None
        item = db.scalars(select(m.JurisdictionQueue).where(m.JurisdictionQueue.pothole_id == q.id)).one()
        assert "no owner agency" in item.reason and "provisionally" in item.reason
        unit_agency = db.get(m.AdminUnit, q.admin_unit_id).agency_id
        assert q.agency_id == unit_agency  # provisional owner: visible to that agency's staff meanwhile
        qid = q.id
    queue = client.get("/v1/admin/jurisdiction-queue?pothole_only=true", headers=admin).json()
    assert qid in [x["pothole_id"] for x in queue]

    print(f"\nE2E ops flow: {time.perf_counter() - t_start:.1f} s")
