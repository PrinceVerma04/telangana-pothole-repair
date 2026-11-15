"""PLAN 10 WP5 acceptance: dedup/clustering over synthetic bus trips.

Every observation goes through the production chain: GPS fix (lagged by the GNSS latency, biased per trip,
jittered per fix) -> geo.locate.ground_point -> geo.mapmatch.match_point -> geo.cluster.assign.
Road: way 100, two-way 2-lane, along y = 0. India drives on the left: eastbound buses (heading 90) run at
y = +1.75, westbound ones at y = -1.75. ``lane_offset_m`` = camera x + the bus's lateral position, in the
travel frame (+ right of travel), so it uses the same reference for every trip in a direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from geo.cluster import assign, lane_band
from geo.locate import ground_point
from geo.mapmatch import match_point
from geo.tests.synth import T0, ll, two_lane_network
from geo.types import ObsPoint, PotholePoint

LATENCY = 0.3
SPEED = 10.0
GPS_SIGMA = 8.0  # per axis (pessimistic reading of "sigma 8 m")


def _observe(idx, rng, trip, potholes, eastbound=True, obs_id0=0, gps_sigma=GPS_SIGMA):
    """One pass of one bus. ``potholes`` = [(x, y)] in local metres. 3 frames per pothole at z = 12, 9, 6 m."""
    heading = 90.0 if eastbound else 270.0
    bus_y = 1.75 if eastbound else -1.75
    sgn = 1.0 if eastbound else -1.0  # travel axis along +x or -x
    bias = rng.normal(0.0, gps_sigma, 2)
    out = []
    oid = obs_id0
    for px, py in potholes:
        for z in (12.0, 9.0, 6.0):
            bus_x = px - sgn * z
            # camera x (right of travel): eastbound right = -y, westbound right = +y
            cam_x = (bus_y - py) * sgn + rng.normal(0, 0.08)
            fix_x = bus_x - sgn * LATENCY * SPEED + bias[0] + rng.normal(0, 0.5)
            fix_y = bus_y + bias[1] + rng.normal(0, 0.5)
            lat, lon = ll(fix_x, fix_y)
            glat, glon, sig = ground_point(lat, lon, heading, SPEED, (cam_x, z + rng.normal(0, 0.3)), LATENCY, 8.0)
            m = match_point(idx, glat, glon, heading, sigma_m=sig)
            lane_off = cam_x + (-bus_y * sgn)  # bus is 1.75 m left of the centreline in its travel frame
            out.append(ObsPoint(oid, trip, glat, glon, sig, m.way_id if m else None, m.direction if m else None,
                                lane_off, T0))
            oid += 1
    return out


def _clusters(res):
    """{cluster label: set(obs_id)} for new keys and attached potholes alike."""
    out: dict = {}
    for oid, pid, key in res:
        out.setdefault(pid if pid is not None else key, set()).add(oid)
    return out


def _trips_per_cluster(obs, res):
    trip = {o.obs_id: o.trip_id for o in obs}
    return {k: {trip[i] for i in v} for k, v in _clusters(res).items()}


def test_lane_band():
    assert lane_band(None) is None
    assert lane_band(0.0) == 0
    assert lane_band(0.87) == 0 and lane_band(0.88) == 1
    assert lane_band(-1.0) == -1 and lane_band(-2.7) == -2
    assert lane_band(1.75 / 2) == 1 and lane_band(-1.75 / 2) == -1  # half away from zero, symmetric


def test_three_trips_gps_sigma8_same_pothole_one_cluster():
    idx = two_lane_network()
    rng = np.random.default_rng(42)
    obs = []
    for t in range(3):
        obs += _observe(idx, rng, f"trip{t}", [(150.0, 1.0)], obs_id0=100 * t)
    assert all(o.way_id == 100 and o.direction == 0 for o in obs)
    res = assign(obs, [])
    tpc = _trips_per_cluster(obs, res)
    assert len(tpc) == 1, tpc
    (key, trips), = tpc.items()
    assert trips == {"trip0", "trip1", "trip2"}  # >= 2 distinct trips -> WP6 confirms it
    assert key.startswith("100:0:") and key.endswith(":-1")  # way:direction:h3:lane_band


def test_three_trips_gps_sigma8_robust_over_seeds():
    """The same scenario over 200 random seeds, to report a rate rather than one lucky seed. The network is
    only the 2-lane way and its crossing road. With a parallel road 40 m away, a trip whose constant GPS
    bias is > 20 m toward it (2.5 sigma) map-matches to that road. That is a map-matching limit, not a
    clustering one. It is measured separately below."""
    idx = two_lane_network(parallel=False)
    ok = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        obs = []
        for t in range(3):
            obs += _observe(idx, rng, f"trip{t}", [(150.0, 1.0)], obs_id0=100 * t)
        ok += len(_clusters(assign(obs, []))) == 1
    # measured 2026-11-13: 194/200 with per-axis sigma 8 m (radial RMS 11.3 m). In the misses the trips
    # lie 30-44 m apart.
    assert ok / 200 >= 0.95, ok
    # with accuracy_m read as the usual 68 % horizontal radius (per-axis sigma 8/1.51 = 5.3 m): 200/200
    ok = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        obs = []
        for t in range(3):
            obs += _observe(idx, rng, f"trip{t}", [(150.0, 1.0)], obs_id0=100 * t, gps_sigma=GPS_SIGMA / 1.51)
        ok += len(_clusters(assign(obs, []))) == 1
    assert ok / 200 >= 0.99, ok


def test_parallel_road_mismatch_rate_is_bounded():
    idx = two_lane_network(parallel=True)
    ok = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        obs = []
        for t in range(3):
            obs += _observe(idx, rng, f"trip{t}", [(150.0, 1.0)], obs_id0=100 * t)
        ok += len(_clusters(assign(obs, []))) == 1
    # measured 189/200: 7 seeds lose one trip to the parallel road (map-matching), 4 fragment (clustering)
    assert ok / 200 >= 0.93, ok


def test_opposite_directions_stay_separate():
    """Pothole A in the eastbound lane and B in the westbound lane at the same chainage."""
    idx = two_lane_network()
    rng = np.random.default_rng(7)
    obs = []
    for t in range(3):
        obs += _observe(idx, rng, f"east{t}", [(150.0, 1.0)], eastbound=True, obs_id0=100 * t)
        obs += _observe(idx, rng, f"west{t}", [(150.0, -1.0)], eastbound=False, obs_id0=1000 + 100 * t)
    tpc = _trips_per_cluster(obs, assign(obs, []))
    assert len(tpc) == 2
    keys = sorted(tpc)
    assert keys[0].startswith("100:0:") and keys[1].startswith("100:1:")
    assert tpc[keys[0]] == {"east0", "east1", "east2"} and tpc[keys[1]] == {"west0", "west1", "west2"}


def test_two_potholes_2m_apart_in_different_lanes_stay_separate():
    """Eastbound buses see pothole A in their lane (y=+1) and B in the oncoming lane (y=-1), 2 m apart."""
    idx = two_lane_network()
    rng = np.random.default_rng(11)
    obs = []
    for t in range(3):
        obs += _observe(idx, rng, f"trip{t}", [(150.0, 1.0), (150.0, -1.0)], obs_id0=100 * t)
    res = assign(obs, [])
    cl = _clusters(res)
    assert len(cl) == 2
    by_id = {o.obs_id: o for o in obs}
    bands = sorted({lane_band(float(np.median([by_id[i].lane_offset_m for i in v]))) for v in cl.values()})
    assert bands == [-1, 1]
    for members in cl.values():
        offs = [by_id[i].lane_offset_m for i in members]
        assert max(offs) - min(offs) < 0.6  # each cluster is one lane only


def test_one_trip_seeing_two_potholes_20m_apart_keeps_them_apart():
    """The sigma-scaled gate may chain them through a noisy trip. The trip-anchor split must undo that."""
    idx = two_lane_network()
    rng = np.random.default_rng(5)
    obs = []
    for t in range(3):
        obs += _observe(idx, rng, f"trip{t}", [(150.0, 1.0), (170.0, 1.0)], obs_id0=100 * t, gps_sigma=3.0)
    cl = _clusters(assign(obs, []))
    assert len(cl) == 2


def _pothole(pid, x, y, status="candidate", way=100, direction=0, band=-1):
    lat, lon = ll(x, y)
    return PotholePoint(pid, lat, lon, way, direction, band, status)


def test_attach_to_existing_live_pothole_of_same_key_only():
    idx = two_lane_network()
    rng = np.random.default_rng(42)
    obs = []
    for t in range(3):
        obs += _observe(idx, rng, f"trip{t}", [(150.0, 1.0)], obs_id0=100 * t)
    existing = [
        _pothole(7, 152.0, 1.0, direction=1),  # same place, other direction -> not this key
        _pothole(8, 151.0, 1.0, status="rejected"),  # engineer FP -> never attached
        _pothole(9, 153.0, 1.0, status="confirmed"),
        _pothole(10, 150.0, 1.0, band=2),  # incompatible lane band (|-1 - 2| > 1)
    ]
    res = assign(obs, existing)
    assert {pid for _, pid, _ in res} == {9}
    assert all(key is None for _, _, key in res)
    # a verified_closed pothole is attachable (DLP re-detection), an archived one is not
    res2 = assign(obs, [_pothole(11, 150.0, 1.0, status="verified_closed")])
    assert {pid for _, pid, _ in res2} == {11}
    res3 = assign(obs, [_pothole(12, 150.0, 1.0, status="archived")])
    assert {pid for _, pid, _ in res3} == {None}


def test_far_existing_pothole_not_attached_and_keys_unique():
    idx = two_lane_network()
    rng = np.random.default_rng(3)
    obs = _observe(idx, rng, "tripA", [(-150.0, 1.0)], gps_sigma=2.0)
    obs += _observe(idx, rng, "tripA", [(-250.0, 1.0)], obs_id0=50, gps_sigma=2.0)
    res = assign(obs, [_pothole(1, 150.0, 1.0)])
    keys = {k for _, _, k in res}
    assert None not in keys and len(keys) == 2
    assert [r[0] for r in res] == [o.obs_id for o in obs]  # output order follows input


def test_unmatched_observations_still_cluster():
    obs = [ObsPoint(i, "t", *ll(0.5 * i, 0.0), 3.0, None, None, None, T0) for i in range(3)]
    res = assign(obs, [])
    assert len({k for _, _, k in res}) == 1 and res[0][2].startswith("None:None:")


@pytest.mark.parametrize("n", [0, 1])
def test_small_inputs(n):
    obs = [ObsPoint(1, "t", *ll(0, 0), 5.0, 100, 0, 0.2, T0)][:n]
    assert len(assign(obs, [])) == n
