"""WP6 unit-level checks: adapters (WKT parsing, bbox-tiled road index), no silent fallback when ``geo`` is
missing (the stage errors are recorded, the observation stays verified), jurisdiction policy, confirmation rules."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from server.app.db import models as m
from server.tests import test_workers_support as S


def test_parse_linestring() -> None:
    from server.workers.adapters import parse_linestring

    assert parse_linestring("LINESTRING (78.1 17.2, 78.3 17.4)") == [(78.1, 17.2), (78.3, 17.4)]
    assert parse_linestring("LINESTRING(78.1 17.2,78.3 17.4)") == [(78.1, 17.2), (78.3, 17.4)]
    assert parse_linestring("MULTILINESTRING ((1 2, 3 4), (5 6, 7 8))") == [(1, 2), (3, 4), (5, 6), (7, 8)]
    assert parse_linestring(None) == [] and parse_linestring("garbage") == []


def test_lane_band_uses_geo_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    from server.workers.adapters import lane_band

    assert lane_band(None) is None and lane_band(0.9) == 1 and lane_band(-2.0) == -1 and lane_band(0.5) == 0
    monkeypatch.setitem(sys.modules, "geo.cluster", None)  # no fallback: a missing WP5 module is an error
    with pytest.raises(ImportError):
        lane_band(0.9)


@pytest.fixture
def seeded_geo(app: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    from server.app.seed import seed_all
    from server.workers import adapters

    adapters.clear_caches()
    with app.state.database.session() as s:
        seed_all(s, fixture=True)
    yield app
    adapters.clear_caches()


def test_road_index_loads_only_nearby_tiles(seeded_geo: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from server.workers import adapters

    S.install_geo(monkeypatch, "fake")
    with seeded_geo.state.database.session() as db:
        total = db.scalar(select(func.count()).select_from(m.RoadSegment))
        road, coords = S.pick_road(db)
        la, lo, _ = S.along(coords, 50.0)
        adapters.clear_caches()
        idx = adapters.road_index_for(db, [(la, lo)], margin_m=50.0)
        assert idx is not None and 0 < len(idx.segments) < total
        assert any(seg.way_id == road.osm_way_id for seg, _d, _b in idx.nearest(la, lo, 25.0))
        n1 = len(idx.segments)
        assert adapters.road_index_for(db, [(la, lo)], margin_m=50.0) is idx  # cached, not rebuilt
        assert len(adapters.road_index_for(db, [(17.425, 78.455)], margin_m=50.0).segments) >= n1  # grows by tile
        assert adapters.road_index_for(db, [(12.97, 77.59)]) is not None  # far away: no new segments, same index
        units, contracts = adapters.jurisdiction_inputs(db)
        assert units and all(len(u) == 3 for u in units)


def test_missing_geo_is_an_error_not_a_fallback(client: TestClient, app: Any, settings: Any, seeded_geo: Any,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """WP5 modules missing: locate/match/cluster fail loudly (stage errors), depth + measurement still run and the
    observation stays 'verified'; no pothole is invented by a fallback clusterer."""
    from measure.depth_backends.synthetic import SyntheticBackend
    from server.app import pipeline
    from server.workers import adapters, install, stages, uninstall

    frame = S.synthetic_frame()
    with app.state.database.session() as db:
        road, coords = S.pick_road(db)
    install(settings, app.state.storage)
    stages.set_depth_backend(SyntheticBackend(depth=frame["depth"]))
    headers = S.login(client)
    app.state.storage.put("frames/test/good.png", frame["png"], "image/png")
    la, lo, brg = S.along(coords, 40.0)
    o = S.observation(device_id="sim-bus-01", trip_uuid="ffffffff-0000-4000-8000-000000000001", seq=0,
                      ts=datetime(2026, 11, 10, 3, 30, tzinfo=timezone.utc), lat=la, lon=lo, heading=brg,
                      frame_key="frames/test/good.png", frame=frame, track_id=1)
    for name in S.GEO_API:
        monkeypatch.setitem(sys.modules, name, None)
    adapters.clear_caches()
    try:
        with app.state.database.session() as db:
            obs_id = client.post("/v1/ingest/observations", json={"observations": [o]},
                                 headers=headers).json()["observation_ids"][0]
            # inline run already happened at ingest (DEMO_MODE); run it again to read the per-stage results
            res = pipeline.process_observation(obs_id, db, settings=settings, storage=app.state.storage)
    finally:
        uninstall()
    assert res.status == "verified"
    for name in ("locate", "match", "cluster"):
        assert res.stages[name].startswith("error: ModuleNotFoundError"), res.stages
    assert res.stages["depth"] == "ok" and res.stages["measure"] == "ok", res.stages
    with app.state.database.session() as db:
        assert db.scalar(select(func.count()).select_from(m.Measurement)) == 1
        assert db.scalar(select(func.count()).select_from(m.Pothole)) == 0


def test_jurisdiction_policy(app: Any, seeded_geo: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unowned location: 'provisional' (default) = admin-unit agency + queue item; 'queue' = no agency + queue item;
    'admin_fallback' = admin-unit agency, no queue item."""
    from server.workers.stages import apply_jurisdiction, jurisdiction_policy

    with app.state.database.session() as db:
        au = db.scalars(select(m.AdminUnit).where(m.AdminUnit.kind == "ward", m.AdminUnit.agency_id.is_not(None),
                                                  m.AdminUnit.geom_wkt.is_not(None))).first()
        from shapely import wkt as swkt

        pt = swkt.loads(au.geom_wkt).representative_point()
        out = {}
        for policy in ("provisional", "queue", "admin_fallback"):
            monkeypatch.setenv("GM_JURISDICTION_POLICY", policy)
            assert jurisdiction_policy() == policy
            p = m.Pothole(lat=pt.y, lon=pt.x, status="candidate")
            reason = apply_jurisdiction(db, p, None)  # no matched road segment -> owner unknown
            out[policy] = (p.agency_id, reason is not None, p.admin_unit_id is not None)
        db.rollback()
    agency = out["admin_fallback"][0]
    assert agency is not None
    assert out == {"provisional": (agency, True, True), "queue": (None, True, True),
                   "admin_fallback": (agency, False, True)}


def test_confirm_rules(app: Any) -> None:
    from server.app.core.util import now_utc
    from server.workers.stages import maybe_confirm

    with app.state.database.session() as db:
        def mk(**kw: Any) -> m.Pothole:
            p = m.Pothole(**{"lat": 17.43, "lon": 78.45, "status": "candidate", "first_seen": now_utc(), **kw})
            db.add(p)
            db.flush()
            return p

        assert maybe_confirm(db, mk(trips_seen=1, passes_seen=1)) is None
        assert "distinct trips" in maybe_confirm(db, mk(trips_seen=2, passes_seen=2))
        assert "IMU" in maybe_confirm(db, mk(trips_seen=1, passes_seen=1, imu_corroborations=1, quality=0.7))
        assert maybe_confirm(db, mk(trips_seen=1, passes_seen=1, imu_corroborations=1, quality=0.5)) is None
        p = mk(trips_seen=1, passes_seen=1, lat=17.44, lon=78.46)
        db.add(m.Complaint(ticket="GM-T-1", lat=17.44005, lon=78.46, status="received"))
        db.flush()
        assert "citizen" in maybe_confirm(db, p) and p.status == "confirmed" and p.complaint_count == 1
        db.rollback()


def test_flat_road_ground_xz_matches_rendered_geometry() -> None:
    """Depth failed -> the polygon centroid projected on a flat road lands near the true (0.9 m, 8 m) offset."""
    from measure.types import CalibCert
    from server.app.core.config import CONFIGS_DIR
    from server.workers.stages import flat_road_ground_xz

    fr = S.synthetic_frame()
    cert = CalibCert.load(CONFIGS_DIR / "cameras" / "sim_bus.json")
    det = m.Detection(cls="pothole", conf=0.8, bbox=fr["bbox"], polygon=fr["polygon"])
    obs = m.Observation(payload={"frame": {"w": fr["W"], "h": fr["H"]}})
    x, z = flat_road_ground_xz(det, obs, cert)
    assert abs(x - 0.9) < 0.15 and abs(z - 8.0) < 0.8
    det_box = m.Detection(cls="pothole", conf=0.8, bbox=fr["bbox"], polygon=None)
    xb, zb = flat_road_ground_xz(det_box, obs, cert)
    assert abs(zb - z) < 0.5
    sky = m.Detection(cls="pothole", conf=0.8, bbox=[300, 10, 340, 30], polygon=None)  # above the horizon
    assert flat_road_ground_xz(sky, obs, cert) is None
