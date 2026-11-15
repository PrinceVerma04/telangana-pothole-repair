import math

import pytest

from geo.locate import angle_diff, ground_point, haversine_m, to_local
from geo.tests.synth import LAT0, LON0, ll


def _en(lat, lon):
    e, n = to_local(LAT0, LON0, lat, lon)
    return float(e), float(n)


def test_ground_point_moves_forward_by_latency_speed_plus_z_and_right_by_x_heading_north():
    lat, lon, sigma = ground_point(LAT0, LON0, 0.0, 10.0, (2.0, 10.0), gnss_latency_s=0.3, gps_accuracy_m=8.0)
    e, n = _en(lat, lon)
    assert n == pytest.approx(0.3 * 10.0 + 10.0, abs=0.01)  # forward = north
    assert e == pytest.approx(2.0, abs=0.01)  # right of travel = east
    assert sigma == pytest.approx(math.sqrt(8.0**2 + (0.05 * math.hypot(2.0, 10.0)) ** 2))


def test_ground_point_heading_east_right_is_south():
    lat, lon, _ = ground_point(LAT0, LON0, 90.0, 5.0, (-1.5, 8.0), gnss_latency_s=0.4)
    e, n = _en(lat, lon)
    assert e == pytest.approx(0.4 * 5.0 + 8.0, abs=0.01)
    assert n == pytest.approx(1.5, abs=0.01)  # x = -1.5 (left of travel) -> north when heading east


def test_ground_point_without_offset_and_without_heading():
    lat, lon, sigma = ground_point(LAT0, LON0, 45.0, 0.0, None, gps_accuracy_m=None)
    assert (lat, lon) == pytest.approx((LAT0, LON0))
    assert sigma == pytest.approx(8.0)
    lat, lon, sigma = ground_point(LAT0, LON0, None, 10.0, (0.0, 10.0), gps_accuracy_m=5.0)
    assert (lat, lon) == pytest.approx((LAT0, LON0))
    assert sigma > 10.0  # unknown direction: the pothole is somewhere within the range


def test_local_projection_error_below_1cm_over_30m():
    from pyproj import Geod

    la, lo = ll(21.0, -21.0)
    e, n = _en(la, lo)
    assert e == pytest.approx(21.0, abs=0.005) and n == pytest.approx(-21.0, abs=0.005)
    _, _, geodesic = Geod(ellps="WGS84").inv(LON0, LAT0, lo, la)
    assert math.hypot(e, n) == pytest.approx(geodesic, abs=0.01)
    # spherical haversine (used for clustering distances) is within 0.5 % of the ellipsoid
    assert haversine_m(LAT0, LON0, la, lo) == pytest.approx(geodesic, rel=5e-3)


def test_angle_diff():
    assert angle_diff(350, 10) == pytest.approx(20)
    assert angle_diff(90, 270) == pytest.approx(180)
