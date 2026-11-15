"""WP7 unit tests: fleet scheduling / matching / acceptance, Jetson trigger + GPS track (no API, no datasets)."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from edge.jetson.runner import Fix, GpsTrack, KeyframeTrigger
from edge.sim import fleet as F
from edge.sim.geodesy import haversine_m
from edge.sim.route import Route
from edge.sim.scene import SimPothole


def _route(key: str, lat0: float = 17.43, lon0: float = 78.45, km: float = 3.0) -> Route:
    # straight north-going line, ~111 m per 0.001 deg latitude
    n = 30
    pts = [(lon0, lat0 + i * (km / 111.0) / n) for i in range(n + 1)]
    return Route(key, f"route {key}", pts, 100.0)


def _pothole(route: Route, sid: int, c: float, lateral: float) -> SimPothole:
    la, lo = route.latlon_at(c)
    x, y = route.xy_at(c)
    return SimPothole(sid=sid, route_key=route.route_id, chainage_m=c, lateral_m=lateral, diameter_m=0.5,
                      label=None, x=x, y=y, lat=la, lon=lo, heading_deg=0.0)  # type: ignore[arg-type]


def _worlds() -> list[F.RouteWorld]:
    out = []
    for i, key in enumerate(("A", "B")):
        r = _route(key, lon0=78.45 + 0.01 * i)
        pots = [_pothole(r, 100 * i + k, 200.0 + 300.0 * k, 1.0 if k % 2 else 0.2) for k in range(6)]
        out.append(F.RouteWorld(key=key, gtfs_route_id=key, route=r, potholes=pots, buses=[f"sim-bus-0{i + 1}"]))
    ur = _route("U", lat0=17.30)
    out.append(F.RouteWorld(key="unowned:U", gtfs_route_id="U", route=ur, potholes=[_pothole(ur, 900, 300.0, 0.0)],
                            buses=["sim-bus-01"], unowned=True))
    return out


def _args(**kw: object) -> argparse.Namespace:
    base = dict(days=2, repair_day=2, trips_per_day=3, headway_min=75.0, jolts=2, speed_kmh=35.0)
    base.update(kw)
    return argparse.Namespace(**base)


def test_schedule_orders_survey_before_claim_and_verification_after() -> None:
    worlds = _worlds()
    claim = datetime(2026, 11, 14, 10, 0, tzinfo=timezone.utc)
    survey, verify = F.schedule(_args(), worlds, claim, np.random.default_rng(1))
    # survey: 3 trips x 2 owned routes + 2 trips on the unowned stretch; verification: 3 x 2 (unowned skipped)
    assert len(survey) == 3 * 2 + 2 and len(verify) == 3 * 2
    assert all(j.t_start + timedelta(seconds=F.trip_seconds(j.world, 35.0)) < claim for j in survey)
    assert all(j.t_start > claim for j in verify)
    assert [j.t_start for j in survey] == sorted(j.t_start for j in survey)
    assert not any(j.world.unowned for j in verify)
    assert all(j.phase == "survey" for j in survey) and all(j.phase == "verify" for j in verify)
    jolted = [sid for j in survey for sid in j.jolts]
    assert len(jolted) == 2 and all(j.day == 1 and j.k == 0 for j in survey if j.jolts)
    # every jolt target ends up in a wheel path so the simulated wheel hits it
    pots = {p.sid: p for w in worlds for p in w.potholes}
    assert all(pots[s].in_wheel_path() for s in jolted)


def test_schedule_three_days_repair_on_day_three() -> None:
    claim = datetime(2026, 11, 14, 10, 0, tzinfo=timezone.utc)
    survey, verify = F.schedule(_args(days=3, repair_day=3), _worlds(), claim, np.random.default_rng(2))
    assert sorted({j.day for j in survey}) == [1, 2] and {j.day for j in verify} == {3}
    assert sum(1 for j in survey if j.world.unowned) == 4  # two confirming trips on each survey day


def test_schedule_rejects_overlap_with_claim() -> None:
    claim = datetime(2026, 11, 14, 10, 0, tzinfo=timezone.utc)
    with pytest.raises(SystemExit):
        # survey day 1 starts 38 h before the claim; 30 trips x 90 min = 45 h overruns it
        F.schedule(_args(trips_per_day=30, headway_min=90.0), _worlds(), claim, np.random.default_rng(3))


def test_match_is_one_to_one_nearest_within_radius() -> None:
    worlds = _worlds()[:1]
    p0, p1 = worlds[0].potholes[:2]
    items = [{"id": 1, "lat": p0.lat + 3e-5, "lon": p0.lon},  # ~3.3 m from p0
             {"id": 2, "lat": p0.lat + 6e-5, "lon": p0.lon},  # ~6.7 m from p0 (second choice)
             {"id": 3, "lat": p1.lat + 5e-4, "lon": p1.lon}]  # ~55 m from p1: too far
    m = F.Fleet.match(worlds, items)
    assert m[("A", p0.sid)]["id"] == 1
    assert ("A", p1.sid) not in m
    assert len({int(v["id"]) for v in m.values()}) == len(m)


def test_acceptance_thresholds() -> None:
    s = {"potholes_by_status": {"confirmed": 25, "verified_closed": 1, "reopened": 1}, "ever_confirmed": 31,
         "work_orders_by_status": {"verified_closed": 1}, "rework_work_orders": 1, "jurisdiction_queue_items": 2}
    assert all(F.acceptance(s, 30).values())
    s2 = dict(s, ever_confirmed=12, jurisdiction_queue_items=0, rework_work_orders=0,
              potholes_by_status={"confirmed": 12})
    acc = F.acceptance(s2, 30)
    assert not acc[">= 30 confirmed (ever)"] and not acc[">= 1 jurisdiction-queue item"]
    assert not acc[">= 1 reopened / dlp_event"]


def test_keyframe_trigger_distance_and_speed() -> None:
    trig = KeyframeTrigger(4.0, 5.0)
    lat, lon = 17.43, 78.45
    assert not trig(Fix(0, lat, lon, speed_mps=1.0))  # too slow
    assert trig(Fix(0, lat, lon, speed_mps=10.0))  # first fix at speed
    step = 1e-5  # ~1.1 m north
    fired = [trig(Fix(i, lat + i * step, lon, speed_mps=10.0)) for i in range(1, 13)]
    # fires every 4 fixes (4.4 m) after the first keyframe
    assert fired.count(True) == 3 and fired[3] and fired[7] and fired[11]


def test_gps_track_interpolates_and_derives_kinematics(tmp_path: Path) -> None:
    csv = tmp_path / "gps.csv"
    csv.write_text("t_s,lat,lon\n0,17.4300,78.4500\n10,17.4310,78.4500\n20,17.4320,78.4500\n", encoding="utf-8")
    g = GpsTrack.from_csv(csv)
    f = g.at(5.0)
    assert f.lat == pytest.approx(17.4305) and f.lon == pytest.approx(78.45)
    assert f.speed_mps == pytest.approx(haversine_m(17.43, 78.45, 17.431, 78.45) / 10.0, rel=1e-3)
    assert f.heading_deg == pytest.approx(0.0, abs=0.5)  # due north
    assert g.at(-3).lat == 17.43 and g.at(99).lat == pytest.approx(17.432)
    iso = tmp_path / "iso.csv"
    iso.write_text("ts,lat,lon,speed_mps\n2026-11-14T10:00:00+05:30,17.43,78.45,8\n"
                   "2026-11-14T10:00:04+05:30,17.4303,78.45,8\n", encoding="utf-8")
    g2 = GpsTrack.from_csv(iso)
    assert g2.ts == [0.0, 4.0] and g2.at(2.0).speed_mps == pytest.approx(8.0)
