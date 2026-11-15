"""WP8a acceptance: repair planner (PLAN 7) + MB / indent exports."""

from __future__ import annotations

import csv
import io
import json

import pytest

from measure.types import PotholeFused, RoadSurface
from repair.loaders import load_rate_card, load_rules, rate_card_version
from repair.mb_export import INDENT_COLUMNS, MB_COLUMNS, indent_csv, indent_totals, measurement_book_csv
from repair.planner import footprint_gap_m, group_for_merge, merge_cut, plan, plan_many
from repair.types import PlanContext, RepairPlan, SpecCode

RULES = load_rules()
CARD = load_rate_card()


def pothole(**kw) -> PotholeFused:
    base = dict(id=1, length_m=0.50, width_m=0.40, depth_p95_mm=60.0, depth_sigma_mm=8.0, depth_max_mm=60.0,
                depth_mean_mm=35.0, area_m2=0.16, area_sigma=0.008, volume_l=6.0, quality=0.8)
    base.update(kw)
    return PotholeFused(**base)


def _sum_lines(p: RepairPlan) -> float:
    return round(sum(cl.amount_inr for cl in p.cost_breakdown), 2)


# ------------------------------------------------------------------------- worked example (PLAN 7.10)
def test_worked_example_plan_7_10():
    p = plan(pothole(), RULES, CARD, PlanContext())
    assert p.l_cut_m == pytest.approx(0.75)
    assert p.w_cut_m == pytest.approx(0.65)
    assert p.a_cut_m2 == pytest.approx(0.4875)
    assert p.d_design_mm == 60 and not p.layered and p.d_patch_mm == 60 and p.d_crm_mm == 0
    assert p.v_patch_l == pytest.approx(29.25)
    assert p.mix_kg == pytest.approx(72.07, abs=0.05)
    assert p.bag_kg == 25 and p.bags == 3 and (p.bags_low, p.bags_high) == (3, 4)
    assert p.emulsion_kg == pytest.approx(0.146, abs=1e-3)
    assert p.sand_kg == pytest.approx(0.49, abs=0.01)
    assert p.labour_h == pytest.approx(2.2)
    assert p.est_cost_inr == pytest.approx(1406, abs=2.0)
    assert p.baseline_cost_inr == 7396
    assert p.spec_code == SpecCode.IRC116_COLDMIX
    assert p.v_cuboid_l == pytest.approx(12.0)
    assert p.v_cavity_l == pytest.approx(6.0)
    assert p.v_cavity_l < p.v_cuboid_l < p.v_patch_l  # all four volumes shown, cuboid over-estimates cavity
    assert p.rate_card_version == rate_card_version(CARD) == "GHMC-2026-27-v1"
    assert any("IRC:82-2015" in r for r in p.refs) and any("IRC:116-2014" in r for r in p.refs)
    assert not p.pending_remeasure and not p.weather_blocked


def test_worked_example_cost_lines_have_provenance_and_sum():
    p = plan(pothole(), RULES, CARD)
    assert _sum_lines(p) == pytest.approx(p.est_cost_inr, abs=0.005)
    items = {cl.item.split(",")[0]: cl for cl in p.cost_breakdown}
    assert items["Cold mix"].qty == 3 and items["Cold mix"].rate_inr == 350
    for cl in p.cost_breakdown:
        assert cl.provenance, cl.item
        assert cl.amount_inr == pytest.approx(cl.qty * cl.rate_inr, abs=0.01)


def test_50kg_bags():
    p = plan(pothole(), RULES, CARD, PlanContext(bag_kg=50))
    assert p.bag_kg == 50 and p.bags == 2
    assert p.cost_breakdown[0].rate_inr == 680


def test_no_exact_bags_language():
    for kw in ({}, {"water_filled": True, "depth_p95_mm": None, "width_m": 0.6}):
        txt = plan(pothole(**kw), RULES, CARD).to_json().lower()
        assert "exact" not in txt
    p = plan(pothole(), RULES, CARD)
    assert any("range 3-4" in n for n in p.notes)


# ------------------------------------------------------------------------- design depth / layering
@pytest.mark.parametrize("p95,expected", [(30.0, 50.0), (50.0, 50.0), (60.0, 60.0), (61.2, 65.0), (75.0, 75.0)])
def test_design_depth_ceil5_min50(p95, expected):
    p = plan(pothole(depth_p95_mm=p95, depth_max_mm=p95), RULES, CARD)
    assert p.d_design_mm == expected
    assert not p.layered


def test_layered_90mm():
    p = plan(pothole(depth_p95_mm=90.0, depth_max_mm=95.0), RULES, CARD)
    assert p.layered and p.d_design_mm == 90
    assert p.d_crm_mm == 40 and p.d_patch_mm == 50
    assert p.v_crm_l == pytest.approx(0.4875 * 40, abs=0.01)
    assert p.v_patch_l == pytest.approx(0.4875 * 50, abs=0.01)
    assert p.crm_t == pytest.approx(0.4875 * 0.040 * 2.1, abs=1e-4)
    assert p.extras["crm_layers"] == 1
    assert any("Crusher run" in cl.item for cl in p.cost_breakdown)
    assert _sum_lines(p) == pytest.approx(p.est_cost_inr, abs=0.005)


def test_deep_pothole_splits_crm_layers():
    p = plan(pothole(depth_p95_mm=160.0, depth_max_mm=170.0), RULES, CARD)
    assert p.d_crm_mm == 110 and p.extras["crm_layers"] == 2  # each CRM layer <= 75 mm


def test_labour_steps_per_additional_0025_m3():
    # 0.6 x 0.6 m, 70 mm -> A_cut 0.7225 m2 (below the 1.0 m2 jet threshold)
    p = plan(pothole(length_m=0.6, width_m=0.6, area_m2=0.28, depth_p95_mm=70.0, depth_max_mm=72.0), RULES, CARD)
    v = 0.85 * 0.85 * 0.070  # 0.0506 m3 -> one complete additional 0.025 m3 step
    assert p.v_patch_l == pytest.approx(v * 1000, abs=0.01)
    assert p.labour_h == pytest.approx((2.0 + 1.0) * 1.1)


def test_sigma_range_widens_with_depth_sigma():
    tight = plan(pothole(depth_sigma_mm=3.0), RULES, CARD)
    loose = plan(pothole(depth_sigma_mm=25.0), RULES, CARD)
    assert loose.mix_kg_sigma > tight.mix_kg_sigma
    assert (loose.bags_high - loose.bags_low) >= (tight.bags_high - tight.bags_low)


@pytest.mark.parametrize("L,W,d,s", [(0.1, 0.1, 10, 3), (0.3, 0.2, 40, 5), (0.9, 0.6, 120, 20), (0.5, 0.5, 74, 30)])
def test_bag_range_brackets_estimate(L, W, d, s):
    p = plan(pothole(length_m=L, width_m=W, area_m2=L * W * 0.8, depth_p95_mm=d, depth_sigma_mm=s,
                     depth_max_mm=d * 1.1), RULES, CARD)
    assert 1 <= p.bags_low <= p.bags <= p.bags_high
    assert p.bags * p.bag_kg >= p.mix_kg


# ------------------------------------------------------------------------- water-filled
def test_water_filled_pending_remeasure_lower_bound():
    wf = pothole(water_filled=True, depth_p95_mm=None, depth_max_mm=None, depth_sigma_mm=None, volume_l=None,
                 width_m=0.6, length_m=0.7, area_m2=0.35)
    p = plan(wf, RULES, CARD)
    assert p.pending_remeasure
    assert p.extras["lower_bound_severity"] == "large"  # width > 500 mm
    assert p.d_design_mm == 50 and not p.layered     # lower bound: class floor 50 mm, min patch 50 mm
    assert p.v_cavity_l is None and p.v_cuboid_l is None
    assert p.bags >= 1 and p.bags_low <= p.bags
    assert any("LOWER BOUND" in n for n in p.notes)
    assert any("water" in r.lower() for r in p.refs)
    assert any("Cl. 7.5.3.4" in r for r in p.refs)


@pytest.mark.parametrize("width,complaints,expected", [(0.15, 0, "small"), (0.3, 0, "medium"), (0.3, 2, "large"),
                                                       (0.55, 0, "large")])
def test_water_lower_bound_severity(width, complaints, expected):
    wf = pothole(water_filled=True, depth_p95_mm=None, depth_max_mm=None, volume_l=None, width_m=width,
                 complaint_count=complaints)
    p = plan(wf, RULES, CARD)
    assert p.pending_remeasure and p.extras["lower_bound_severity"] == expected


def test_missing_depth_not_water_is_pending():
    p = plan(pothole(depth_p95_mm=None, depth_max_mm=None), RULES, CARD)
    assert p.pending_remeasure and p.d_design_mm == 50


# ------------------------------------------------------------------------- method selector / weather
def test_jet_patcher_when_cluster_ge_8():
    p8 = plan(pothole(), RULES, CARD, PlanContext(cluster_n_within_300m=8))
    p7 = plan(pothole(), RULES, CARD, PlanContext(cluster_n_within_300m=7))
    assert p8.spec_code == SpecCode.JET_PATCHER
    assert p7.spec_code == SpecCode.IRC116_COLDMIX
    assert len(p8.cost_breakdown) == 1
    jet = p8.cost_breakdown[0]
    assert jet.rate_inr == 1942 and "BBMP" in jet.provenance
    assert p8.est_cost_inr == pytest.approx(0.4875 * 1942, abs=0.01)
    assert p8.extras["coldmix_alternative_cost_inr"] == pytest.approx(p7.est_cost_inr, abs=0.01)


def test_jet_patcher_when_a_cut_gt_1m2():
    p = plan(pothole(length_m=0.9, width_m=0.8, area_m2=0.6), RULES, CARD)
    assert p.a_cut_m2 > 1.0 and p.spec_code == SpecCode.JET_PATCHER


def test_cc_surface_cc_mortar():
    p = plan(pothole(surface=RoadSurface.CC), RULES, CARD, PlanContext(cluster_n_within_300m=10))
    assert p.spec_code == SpecCode.CC_MORTAR
    assert any("cold-mix basis" in n for n in p.notes)
    assert _sum_lines(p) == pytest.approx(p.est_cost_inr, abs=0.005)


def test_infrared_rule():
    ctx = PlanContext(infrared_unit_in_zone=True, dry_forecast_48h=True)
    assert plan(pothole(depth_p95_mm=35.0), RULES, CARD, ctx).spec_code == SpecCode.INFRARED
    assert plan(pothole(depth_p95_mm=45.0), RULES, CARD, ctx).spec_code == SpecCode.IRC116_COLDMIX
    wet = PlanContext(infrared_unit_in_zone=True, dry_forecast_48h=False)
    assert plan(pothole(depth_p95_mm=35.0), RULES, CARD, wet).spec_code == SpecCode.IRC116_COLDMIX


def test_hot_mix_only_off_season_near_plant():
    assert plan(pothole(), RULES, CARD, PlanContext(hot_mix_plant_km=6)).spec_code == SpecCode.HOT_MIX
    monsoon = PlanContext(hot_mix_plant_km=6, in_monsoon_window=True)
    assert plan(pothole(), RULES, CARD, monsoon).spec_code == SpecCode.IRC116_COLDMIX
    assert plan(pothole(), RULES, CARD, PlanContext(hot_mix_plant_km=25)).spec_code == SpecCode.IRC116_COLDMIX


def test_agency_pref_emulsion():
    p = plan(pothole(), RULES, CARD, PlanContext(agency_pref=SpecCode.IRCSP100_EMULSION))
    assert p.spec_code == SpecCode.IRCSP100_EMULSION
    assert "IRC:SP:100" in p.method


def test_weather_gate_blocks_in_rain():
    p = plan(pothole(), RULES, CARD, PlanContext(raining_now=True))
    assert p.weather_blocked
    assert any("rain" in n.lower() for n in p.notes)


@pytest.mark.parametrize("kw,ctx", [
    ({}, PlanContext()),
    ({"depth_p95_mm": 90.0}, PlanContext()),
    ({"depth_p95_mm": 133.0, "length_m": 0.7}, PlanContext(bag_kg=50)),
    ({"water_filled": True, "depth_p95_mm": None, "volume_l": None}, PlanContext()),
    ({}, PlanContext(cluster_n_within_300m=12)),
    ({"surface": RoadSurface.CC}, PlanContext()),
    ({}, PlanContext(hot_mix_plant_km=3)),
])
def test_cost_lines_sum_to_estimate(kw, ctx):
    p = plan(pothole(**kw), RULES, CARD, ctx)
    assert _sum_lines(p) == pytest.approx(p.est_cost_inr, abs=0.005)
    RepairPlan.from_json(p.to_json())  # frozen contract round-trip


def test_rate_card_edit_changes_cost():
    card = load_rate_card()
    card["items"]["coldmix_bag_25kg"]["rate"] = 400
    card["version"] = 2
    p = plan(pothole(), RULES, card)
    assert p.cost_breakdown[0].rate_inr == 400
    assert p.rate_card_version.endswith("-v2")
    assert p.est_cost_inr > plan(pothole(), RULES, CARD).est_cost_inr


def test_accepts_mapping_input():
    d = pothole().to_dict()
    d["unrelated_db_column"] = 1
    assert plan(d, RULES, CARD).bags == 3


def test_zero_footprint_raises():
    with pytest.raises(ValueError):
        plan(pothole(length_m=0, width_m=0, area_m2=0), RULES, CARD)


# ------------------------------------------------------------------------- merge (PLAN 7.1)
def _ph(i, chain, off, **kw):
    base = dict(id=i, way_id=100, direction=0, chainage_m=chain, lane_offset_m=off, length_m=0.4, width_m=0.3,
                area_m2=0.1, volume_l=3.0)
    base.update(kw)
    return pothole(**base)


def test_merge_within_half_metre_same_lane_band():
    a, b = _ph(1, 10.0, 0.0), _ph(2, 10.8, 0.1)  # along gap 0.8 - 0.4 = 0.4 m
    assert footprint_gap_m(a, b) == pytest.approx(0.4, abs=1e-6)
    cuts = merge_cut([a, b], RULES)
    assert len(cuts) == 1 and cuts[0].member_ids == [1, 2]
    m = cuts[0].pothole
    assert m.length_m == pytest.approx(1.2)
    assert m.width_m == pytest.approx(0.4)
    assert m.area_m2 == pytest.approx(0.2) and m.volume_l == pytest.approx(6.0)


def test_no_merge_beyond_half_metre_or_other_band_or_direction():
    a = _ph(1, 10.0, 0.0)
    assert len(group_for_merge([a, _ph(2, 11.0, 0.0)], RULES)) == 2          # gap 0.6 m
    assert len(group_for_merge([a, _ph(2, 10.5, 1.9)], RULES)) == 2          # lane band 1 vs 0
    other_dir = _ph(2, 10.5, 0.0).model_copy(update={"direction": 1})
    assert len(group_for_merge([a, other_dir], RULES)) == 2


def test_merge_is_transitive_and_plan_many_records_ids():
    ps = [_ph(1, 10.0, 0.0), _ph(2, 10.7, 0.0), _ph(3, 11.4, 0.0), _ph(4, 30.0, 0.0)]
    plans = plan_many(ps, RULES, CARD)
    assert len(plans) == 2
    assert plans[0].merged_pothole_ids == [1, 2, 3] and plans[0].pothole_id == 1
    assert plans[1].merged_pothole_ids == [] and plans[1].pothole_id == 4
    assert plans[0].l_cut_m == pytest.approx(1.8 + 0.25)


def test_merge_deepest_governs_and_water_member_flags_pending():
    a = _ph(1, 10.0, 0.0, depth_p95_mm=40.0)
    b = _ph(2, 10.6, 0.0, depth_p95_mm=95.0, depth_max_mm=100.0)
    p = plan_many([a, b], RULES, CARD)[0]
    assert p.d_design_mm == 95 and p.layered
    w = _ph(3, 10.6, 0.0, water_filled=True, depth_p95_mm=None, volume_l=None)
    q = plan_many([a, w], RULES, CARD)[0]
    assert q.pending_remeasure and q.d_design_mm == 50


def test_merge_latlon_fallback():
    lat0, lon0 = 17.3850, 78.4867  # Hyderabad
    a = pothole(id=1, lat=lat0, lon=lon0, length_m=0.4, width_m=0.4)
    b = pothole(id=2, lat=lat0 + 0.6 / 111_195, lon=lon0, length_m=0.4, width_m=0.4)  # ~0.6 m north
    gap = footprint_gap_m(a, b)
    assert gap == pytest.approx(0.2, abs=0.02)
    cuts = merge_cut([a, b], RULES)
    assert len(cuts) == 1 and cuts[0].pothole.length_m == pytest.approx(1.0, abs=0.02)
    far = pothole(id=3, lat=lat0 + 5 / 111_195, lon=lon0)
    assert len(merge_cut([a, far], RULES)) == 2


# ------------------------------------------------------------------------- exports
def _plans():
    return [plan(pothole(id=1), RULES, CARD), plan(pothole(id=2, depth_p95_mm=90.0), RULES, CARD),
            plan(pothole(id=3), RULES, CARD, PlanContext(bag_kg=50))]


def test_measurement_book_csv():
    plans = _plans()
    rows = [{"plan": plans[0], "work_order_id": 11, "location": "Ameerpet Rd", "date": "2026-11-13"},
            plans[1].to_dict(), plans[2]]
    text = measurement_book_csv(rows)
    data = list(csv.reader(io.StringIO(text)))
    assert data[0] == MB_COLUMNS
    body = data[1:-1]
    assert len(body) == sum(len(p.cost_breakdown) for p in plans)
    assert body[0][0] == "1.1" and "WO 11" in body[0][1] and "Pothole #1" in body[0][1]
    mix = body[0]
    assert mix[3] == "0.750" and mix[4] == "0.650" and mix[5] == "0.060" and mix[6] == "3.000" and mix[7] == "bag"
    assert any(r[5] == "0.040" for r in body if "Crusher" in r[1])  # CRM depth recorded
    total = float(data[-1][-1])
    assert total == pytest.approx(sum(p.est_cost_inr for p in plans), abs=0.01)
    for r in body:
        if r[7] != "LS":
            assert float(r[9]) == pytest.approx(float(r[6]) * float(r[8]), abs=0.02)


def test_indent_csv_totals_equal_sum_of_plans():
    plans = _plans()
    run = {"id": 7, "circle_name": "Khairatabad", "run_date": "2026-11-14",
           "items": [{"plan": p, "work_order_id": 100 + i, "lat": 17.4 + i * 1e-3, "lon": 78.45, "hop_m": 120.0 * i}
                     for i, p in enumerate(plans)]}
    t = indent_totals(run)
    assert t["bags"] == sum(p.bags for p in plans)
    assert t["bags_25kg"] == plans[0].bags + plans[1].bags and t["bags_50kg"] == plans[2].bags
    assert t["cost_inr"] == pytest.approx(sum(p.est_cost_inr for p in plans))
    assert t["crm_t"] == pytest.approx(sum(p.crm_t for p in plans))
    assert t["crew_h"] == pytest.approx(sum(p.labour_h for p in plans))
    data = list(csv.reader(io.StringIO(indent_csv(run))))
    assert "Khairatabad" in data[0][0]
    assert data[1] == INDENT_COLUMNS
    assert len(data) == 2 + len(plans) + 1
    tot = data[-1]
    assert tot[0] == "TOTAL" and int(tot[10]) == t["bags"]
    assert float(tot[INDENT_COLUMNS.index("Est. cost (INR)")]) == pytest.approx(t["cost_inr"], abs=0.01)
    # bare list of plans also accepted
    assert indent_totals(plans)["n_plans"] == 3


def test_loaders():
    assert RULES["repair"]["cut_margin_m"] == 0.125
    assert CARD["items"]["coldmix_bag_25kg"]["rate"] == 350
    json.dumps(RULES)
