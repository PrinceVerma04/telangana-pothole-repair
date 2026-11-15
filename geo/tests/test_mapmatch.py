import numpy as np
import pytest

from geo import mapmatch
from geo.mapmatch import match_point, match_trip
from geo.tests.synth import drive, dual_carriageway, ll, two_lane_network


def test_match_point_uses_heading_to_pick_the_right_carriageway():
    idx = dual_carriageway()
    lat, lon = ll(0.0, -2.0)  # 4 m from the westbound carriageway, 8 m from the eastbound one
    east = match_point(idx, lat, lon, 90.0, sigma_m=8.0)
    assert east is not None and east.way_id == 11 and east.direction == 0
    assert east.dist_m == pytest.approx(8.0, abs=0.05)
    west = match_point(idx, lat, lon, 270.0, sigma_m=8.0)
    assert west is not None and west.way_id == 12 and west.direction == 0
    # without a heading the nearest carriageway wins
    assert match_point(idx, lat, lon, None).way_id == 12
    # a heading that fits neither one-way carriageway (northbound) -> no match
    assert match_point(idx, lat, lon, 0.0) is None


def test_two_way_direction_from_heading():
    idx = two_lane_network()
    lat, lon = ll(150.0, 2.0)
    m0 = match_point(idx, lat, lon, 85.0)
    m1 = match_point(idx, lat, lon, 268.0)
    assert (m0.road_segment_id, m0.direction) == (2, 0)
    assert (m1.road_segment_id, m1.direction) == (2, 1)
    assert m1.bearing_deg == pytest.approx(270.0, abs=0.1)
    assert m0.chainage_m == pytest.approx(150.0, abs=0.05) == m1.chainage_m


def test_match_trip_viterbi_avoids_spurious_switch_to_crossing_road():
    """Eastbound bus stops at the junction (speed 0 -> heading ignored). One stopped fix lands 1 m from the
    crossing road and 5 m from the main road; the point matcher flips, the trip matcher must not."""
    idx = two_lane_network()
    xs = [-60.0, -50.0, -40.0, -30.0, -20.0, -10.0, -4.0, -1.0, -4.0, -4.0, 6.0, 16.0, 26.0, 36.0]
    ys = [1.5] * len(xs)
    ys[7] = 5.0  # the GNSS jump while stopped
    fixes = drive("T1", xs, ys, 90.0, 10.0)
    for k in (6, 7, 8, 9):
        fixes[k].speed_mps = 0.0
    alone = match_point(idx, fixes[7].lat, fixes[7].lon, None)
    assert alone.way_id == 300  # the point matcher alone is fooled
    res = match_trip(idx, fixes)
    assert all(r is not None and r.way_id == 100 for r in res)
    assert all(r.direction == 0 for r in res if r.road_segment_id in (1, 2) and r is not res[7])


def test_match_trip_follows_a_real_turn():
    idx = two_lane_network()
    xs = [-50.0, -40.0, -30.0, -20.0, -10.0, -2.0, 1.5, 1.5, 1.5, 1.5, 1.5]
    ys = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 8.0, 18.0, 28.0, 38.0, 48.0]
    heads = [90.0] * 6 + [0.0] * 5
    fixes = drive("T2", xs, ys, 90.0, 8.0)
    for f, h in zip(fixes, heads, strict=True):
        f.heading_deg = h
    res = match_trip(idx, fixes)
    assert [r.way_id for r in res] == [100] * 6 + [300] * 5
    assert res[-1].road_segment_id == 4 and res[-1].direction == 0


def test_match_trip_gap_and_unmatched_fix():
    idx = two_lane_network()
    fixes = drive("T3", [10.0, 20.0, 30.0], [1.0, 120.0, 1.0], 90.0, 10.0)
    res = match_trip(idx, fixes)
    assert res[0].way_id == 100 and res[1] is None and res[2].way_id == 100


def test_match_trip_returns_input_order_when_fixes_unsorted():
    idx = two_lane_network()
    fixes = drive("T4", [-30.0, -20.0, -10.0], [1.0, 1.0, 1.0], 90.0, 10.0)
    res = match_trip(idx, list(reversed(fixes)))
    assert [round(r.chainage_m) for r in res] == [390, 380, 370]


def test_viterbi_never_worse_than_point_matching_on_noisy_straight_drive():
    idx = two_lane_network()
    rng = np.random.default_rng(3)
    xs = np.arange(-300.0, -20.0, 10.0)
    fixes = drive("T5", xs, np.full_like(xs, 1.75), 90.0, 10.0, jitter=6.0, rng=rng)
    res = match_trip(idx, fixes)
    assert all(r is not None and (r.way_id, r.direction) == (100, 0) for r in res)
    assert mapmatch.UNCONNECTED_COST == 1.0  # PLAN 5.2
