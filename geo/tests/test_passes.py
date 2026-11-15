from datetime import timedelta

import numpy as np
import pytest

from geo.mapmatch import match_trip
from geo.passes import count_passes
from geo.tests.synth import T0, drive, ll, two_lane_network
from geo.types import PotholePoint


def _pothole():
    lat, lon = ll(150.0, 1.0)
    return PotholePoint(1, lat, lon, 100, 0, -1, "repaired_claimed")


def _trip(idx, trip, xs, y, heading, speed, t0, cam=True):
    fixes = drive(trip, list(xs), [y] * len(xs), heading, speed, t0, cam_active=cam)
    return fixes, match_trip(idx, fixes)


def test_count_passes_only_cam_active_same_way_direction_within_eps():
    idx = two_lane_network()
    p = _pothole()
    since = T0 + timedelta(hours=1)
    fixes, matched = [], []

    def add(*a, **k):
        f, m = _trip(idx, *a, **k)
        fixes.extend(f)
        matched.extend(m)

    h = timedelta(hours=1)
    add("A", np.arange(100, 200, 5.0), 1.75, 90.0, 5.0, T0 + 2 * h)  # slow: many fixes in range -> ONE pass
    add("A", np.arange(100, 200, 10.0), 1.75, 90.0, 10.0, T0 + 5 * h)  # same trip id, 3 h later -> 2nd pass
    add("B", np.arange(100, 200, 10.0), 1.75, 90.0, 10.0, T0 + 2 * h, cam=False)  # camera off
    add("C", np.arange(200, 100, -10.0), -1.75, 270.0, 10.0, T0 + 2 * h)  # opposite direction
    add("D", [130.0, 150.0 - 10.0, 150.0 + 10.0, 170.0], 1.75, 90.0, 20.0, T0 + 3 * h)  # fixes straddle it
    add("E", np.arange(100, 200, 10.0), -40.0, 90.0, 10.0, T0 + 2 * h)  # parallel road 41 m away
    add("F", np.arange(100, 200, 10.0), 1.75, 90.0, 10.0, T0)  # before `since`
    add("G", np.arange(100, 200, 10.0), 12.0, 90.0, 10.0, T0 + 2 * h)  # same way+dir but 11 m off -> > 8 m

    res = count_passes(p, fixes, matched, since)
    assert [t for t, _ in res] == ["A", "D", "A"]
    assert [ts for _, ts in res] == sorted(ts for _, ts in res)
    assert res[0][1] == T0 + 2 * h + timedelta(seconds=10)  # closest approach of trip A (x = 150)
    # D: no fix within 8 m (both 10 m away); the interpolated closest point between them is
    d_ts = res[1][1]
    assert d_ts == pytest.approx(T0 + 3 * h + timedelta(seconds=1.5), abs=timedelta(milliseconds=1))
    # without `since` trip F counts too
    assert len(count_passes(p, fixes, matched, None)) == 4


def test_count_passes_requires_aligned_inputs_and_handles_unmatched_pothole():
    idx = two_lane_network()
    fixes, matched = _trip(idx, "A", np.arange(100, 200, 10.0), 1.75, 90.0, 10.0, T0)
    with pytest.raises(ValueError):
        count_passes(_pothole(), fixes, matched[:-1], None)
    lat, lon = ll(150.0, 1.0)
    off_net = PotholePoint(2, lat, lon, None, None, None, "candidate")
    assert [t for t, _ in count_passes(off_net, fixes, [None] * len(fixes), None)] == ["A"]
