"""Priority score (PLAN 6): each term, SLA breach at 24 h on an arterial, weights capped at 100."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from measure.types import PotholeFused, Severity
from priority.config import load_priority_config, normalise_highway
from priority.context import Poi, PriorityContext
from priority.exposure import class_weight, exposure
from priority.score import compute, severity_term, weights_of
from priority.severity import classify_dry, severity_of, water_lower_bound
from priority.sla import age_term, is_breached, sla_due, sla_hours
from priority.vulnerability import vulnerability
from priority.weather import ManualImdAdapter, NullImdAdapter, in_monsoon_window, monsoon_term, month_ist

CFG = load_priority_config()


def ph(**kw) -> PotholeFused:
    base = dict(id=1, length_m=0.5, width_m=0.4, depth_p95_mm=60.0, depth_sigma_mm=8.0, area_m2=0.16)
    base.update(kw)
    return PotholeFused(**base)


def S(p, **ctx):
    return severity_term(p, PriorityContext(**ctx), CFG)[0]


# ------------------------------------------------------------------------- weights / P
def test_default_weights_sum_to_one():
    w, norm = weights_of(CFG)
    assert sum(w.values()) == pytest.approx(1.0) and not norm
    assert w == {"S": 0.30, "E": 0.25, "V": 0.20, "T": 0.15, "M": 0.10}


def _max_ctx():
    return PriorityContext(road_class="primary", bus_passes_per_day=200, pois=[Poi("school", 50), Poi("hospital", 80), Poi("metro_station", 150)],
                           lit=False, hours_since_confirmed=100, month=7, rain_72h_mm=80, in_hotspot=True)


def test_all_terms_max_gives_100():
    P, b = compute(ph(depth_p95_mm=90, lane_offset_m=0.9, imu_corroborated=True), _max_ctx(), CFG)
    assert P == 100.0
    assert sum(b[k]["points"] for k in "SEVTM") == pytest.approx(100.0)
    for k in "SEVTM":
        assert b[k]["value"] == pytest.approx(1.0)


def test_weights_not_summing_to_one_are_normalised_and_capped():
    cfg = json.loads(json.dumps(CFG))
    cfg["weights"] = {"S": 0.6, "E": 0.5, "V": 0.4, "T": 0.3, "M": 0.2}  # sums to 2.0
    P, b = compute(ph(depth_p95_mm=90, lane_offset_m=0.9, imu_corroborated=True), _max_ctx(), cfg)
    assert P == 100.0 and b["weights_normalised"]
    assert sum(b["weights"].values()) == pytest.approx(1.0)
    P2, _ = compute(ph(), PriorityContext(), cfg)
    assert 0.0 <= P2 <= 100.0


def test_zero_weights_rejected():
    cfg = json.loads(json.dumps(CFG))
    cfg["weights"] = dict.fromkeys("SEVTM", 0)
    with pytest.raises(ValueError):
        compute(ph(), PriorityContext(), cfg)


def test_low_priority_pothole_and_breakdown_serialisable():
    ctx = PriorityContext(road_class="service", lit=True, hours_since_confirmed=0, month=2)
    P, b = compute(ph(depth_p95_mm=20, width_m=0.15, length_m=0.15, area_m2=0.02), ctx, CFG)
    # S 0.3, E 0.1, V 0, T 0, M 0.5 -> 100*(0.09 + 0.025 + 0 + 0 + 0.05) = 16.5
    assert P == pytest.approx(16.5)
    json.dumps(b)
    assert b["P"] == P and b["config_version"] == 1


def test_ctx_from_dict_and_mapping_pothole():
    ctx = {"road_class": "primary", "pois": [{"kind": "school", "distance_m": 10}], "hours_since_confirmed": 24}
    P1, _ = compute(ph().to_dict(), ctx, CFG)
    P2, _ = compute(ph(), PriorityContext.from_dict(ctx), CFG)
    assert P1 == P2
    with pytest.raises(ValueError):
        PriorityContext.from_dict({"bogus": 1})


# ------------------------------------------------------------------------- S
def test_severity_classes():
    assert S(ph(depth_p95_mm=60)) == pytest.approx(1.0)
    assert S(ph(depth_p95_mm=40, width_m=0.15)) == pytest.approx(0.6)
    assert S(ph(depth_p95_mm=20, width_m=0.15)) == pytest.approx(0.3)
    assert S(ph(depth_p95_mm=20, width_m=0.6)) == pytest.approx(1.0)  # W > 500 mm -> large
    assert S(ph(depth_p95_mm=20, width_m=0.3)) == pytest.approx(0.6)  # W 200-500 mm -> medium


def test_stored_severity_wins():
    assert severity_of(ph(depth_p95_mm=20, width_m=0.1, severity=Severity.large))[0] == Severity.large


def test_water_filled_severity():
    wf = dict(water_filled=True, depth_p95_mm=None)
    assert S(ph(width_m=0.15, area_m2=0.02, **wf)) == pytest.approx(0.6)          # floor 0.6
    assert S(ph(width_m=0.3, area_m2=0.31, **wf)) == pytest.approx(0.8)           # A >= 0.3 m2
    assert S(ph(width_m=0.6, area_m2=0.2, **wf)) == pytest.approx(1.0)            # lower bound large
    assert S(ph(width_m=0.3, area_m2=0.1, complaint_count=2, **wf)) == pytest.approx(1.0)


def test_severity_bonuses_wheel_path_junction_imu():
    small = dict(depth_p95_mm=20, width_m=0.15)
    assert S(ph(lane_offset_m=0.8, **small)) == pytest.approx(0.5)                        # |0.8-0.9| < 0.35
    assert S(ph(lane_offset_m=0.0, **small)) == pytest.approx(0.3)                        # lane centre
    assert S(ph(lane_offset_m=0.0, junction_dist_m=25, **small)) == pytest.approx(0.5)
    assert S(ph(lane_offset_m=0.0, **small), junction_dist_m=31) == pytest.approx(0.3)
    assert S(ph(lane_offset_m=0.8, junction_dist_m=10, **small)) == pytest.approx(0.5)    # not stacked
    assert S(ph(imu_corroborated=True, **small)) == pytest.approx(0.4)
    assert S(ph(imu_corroborations=2, **small)) == pytest.approx(0.4)
    assert S(ph(lane_offset_m=-0.9, imu_corroborated=True, depth_p95_mm=60)) == pytest.approx(1.0)  # clip
    assert S(ph(lane_offset_m=0.0, **small), wheel_tracks_m=(-0.1, 1.7)) == pytest.approx(0.5)


def test_irc_helpers():
    assert classify_dry(25, 200) == Severity.small
    assert classify_dry(25.1, 0) == Severity.medium
    assert classify_dry(50.1, 0) == Severity.large
    assert water_lower_bound(150) == Severity.small
    assert water_lower_bound(200) == Severity.medium
    assert water_lower_bound(250, imu_peak_g=0.6) == Severity.large


# ------------------------------------------------------------------------- E
def test_exposure_term():
    assert class_weight("primary_link", CFG) == 1.0 and normalise_highway("Primary_Link") == "primary"
    assert class_weight(None, CFG) == 0.4 and class_weight("weird", CFG) == 0.4
    e, parts = exposure("primary", 80, None, CFG)
    assert e == pytest.approx(1.0) and parts["bus_source"] == "observed_trip_fix"
    e, parts = exposure("residential", None, 25, CFG)
    assert e == pytest.approx(0.5 * 0.4 + 0.5 * 0.5) and parts["bus_source"] == "gtfs"
    e, _ = exposure("secondary", 10, 100, CFG)  # observed wins over GTFS
    assert e == pytest.approx(0.5 * 0.8 + 0.5 * 0.2)
    e, parts = exposure("service", None, None, CFG)
    assert e == pytest.approx(0.1) and parts["bus_source"] == "none"


# ------------------------------------------------------------------------- V
def test_vulnerability_term():
    v, _ = vulnerability([Poi("school", 120), Poi("bus_stop", 50)], lit=True, highway="residential", cfg=CFG)
    assert v == pytest.approx(1 - 0.5 * 0.7)
    v, parts = vulnerability([{"kind": "hospital", "distance_m": 250}], lit=True, cfg=CFG)  # beyond 200 m
    assert v == 0.0 and parts["pois"] == []
    v, _ = vulnerability([("market", 100)], lit=False, highway="primary", cfg=CFG)          # alias + unlit
    assert v == pytest.approx(0.2 + 0.2)
    v, _ = vulnerability([], lit=None, highway="residential", cfg=CFG)                        # no data, non-arterial
    assert v == pytest.approx(0.2)
    v, _ = vulnerability([], lit=None, highway="primary", cfg=CFG)                            # no data, arterial
    assert v == 0.0
    many = [Poi("school", 10), Poi("hospital", 10), Poi("metro_station", 10), Poi("clinic", 10)]
    v, _ = vulnerability(many, lit=False, cfg=CFG)
    assert v == pytest.approx(1.0)


# ------------------------------------------------------------------------- T / SLA
def test_sla_hours_rules():
    assert sla_hours("primary", "GHMC", 2, cfg=CFG) == 24
    assert sla_hours("secondary_link", "GHMC", 2, cfg=CFG) == 24
    assert sla_hours("residential", "GHMC", 7, cfg=CFG) == 72
    assert sla_hours("residential", "GHMC", 2, cfg=CFG) == 168
    assert sla_hours("primary", "RNB", 7, cfg=CFG) == 168
    assert sla_hours("residential", "GHMC", 2, cfg=CFG, manual_monsoon=True) == 72
    assert sla_hours("primary", "GHMC", 2, {"arterial_h": 12}, CFG) == 12
    assert sla_hours("residential", "GHMC", 2, {"all_h": 48}, CFG) == 48


def test_T_breach_at_24h_for_arterial():
    base = dict(road_class="primary", agency_code="GHMC", month=2)
    _, b23 = compute(ph(), PriorityContext(hours_since_confirmed=23, **base), CFG)
    _, b24 = compute(ph(), PriorityContext(hours_since_confirmed=24, **base), CFG)
    assert b23["T"]["parts"]["sla_hours"] == 24
    assert not b23["T"]["parts"]["breached"] and b23["T"]["value"] == pytest.approx(23 / 24, abs=1e-4)
    assert b24["T"]["parts"]["breached"] and b24["T"]["value"] == pytest.approx(1.0)
    _, b24r = compute(ph(), PriorityContext(hours_since_confirmed=24, road_class="residential", month=2), CFG)
    assert not b24r["T"]["parts"]["breached"]  # 168 h off-season


def test_age_term_components():
    t, p = age_term(12, 72, complaints=0, cfg=CFG)
    assert t == pytest.approx(12 / 72) and not p["breached"]
    t, p = age_term(12, 72, complaints=2, cfg=CFG)
    assert t == pytest.approx(12 / 72 + 0.2)
    t, _ = age_term(12, 72, complaints=9, cfg=CFG)
    assert t == pytest.approx(12 / 72 + 0.3)  # complaint bonus capped
    t, p = age_term(6, 168, breached=True, cfg=CFG)  # explicit breach flag (e.g. contract SLA)
    assert t == pytest.approx(6 / 168 + 0.3) and p["breached"]
    t, _ = age_term(None, 24, cfg=CFG)  # candidate (never confirmed)
    assert t == 0.0
    _, b = compute(ph(complaint_count=3), PriorityContext(hours_since_confirmed=0, road_class="primary"), CFG)
    assert b["T"]["value"] == pytest.approx(0.3)


def test_sla_helpers():
    assert is_breached(24, 24) and not is_breached(23.9, 24) and not is_breached(None, 24)
    t0 = datetime(2026, 11, 13, 4, 30, tzinfo=timezone.utc)
    assert sla_due(t0, 24) == datetime(2026, 11, 14, 4, 30, tzinfo=timezone.utc)


# ------------------------------------------------------------------------- M / weather
def test_monsoon_term():
    assert monsoon_term(2, cfg=CFG)[0] == pytest.approx(0.5)
    assert monsoon_term(7, cfg=CFG)[0] == pytest.approx(0.8)
    m, parts = monsoon_term(10, cfg=CFG)
    assert m == pytest.approx(0.8) and parts["window"] == "NE"
    assert monsoon_term(2, rain_72h_mm=31, cfg=CFG)[0] == pytest.approx(1.0)
    assert monsoon_term(2, rain_72h_mm=30, cfg=CFG)[0] == pytest.approx(0.5)   # > 30 mm strictly
    assert monsoon_term(2, in_hotspot=True, cfg=CFG)[0] == pytest.approx(0.7)
    assert monsoon_term(2, water_filled=True, cfg=CFG)[0] == pytest.approx(0.6)
    assert monsoon_term(7, in_hotspot=True, water_filled=True, cfg=CFG)[0] == pytest.approx(1.0)  # clip
    assert monsoon_term(2, cfg=CFG, manual_monsoon=True)[0] == pytest.approx(0.8)
    assert monsoon_term(7, cfg=CFG, manual_monsoon=False)[0] == pytest.approx(0.5)


def test_M_uses_pothole_water_flag():
    _, b = compute(ph(water_filled=True, depth_p95_mm=None), PriorityContext(month=2), CFG)
    assert b["M"]["value"] == pytest.approx(0.6)


def test_weather_helpers():
    assert in_monsoon_window(6, CFG) and in_monsoon_window(11, CFG) and not in_monsoon_window(12, CFG)
    assert not in_monsoon_window(None, CFG) and in_monsoon_window(None, CFG, manual=True)
    assert month_ist(datetime(2026, 5, 31, 19, 0, tzinfo=timezone.utc)) == 6  # 00:30 IST on 1 June
    imd = ManualImdAdapter({"Hyderabad": 42.0}, {"Hyderabad": True})
    assert imd.rain_72h_mm("hyderabad") == 42.0 and imd.raining_now("HYDERABAD") is True
    assert imd.rain_72h_mm("Warangal") is None
    imd.set_rain("Warangal", 5)
    assert imd.rain_72h_mm("warangal") == 5.0
    assert NullImdAdapter().rain_72h_mm("Hyderabad") is None
