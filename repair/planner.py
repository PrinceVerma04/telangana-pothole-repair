"""Repair planner (PLAN section 7): IRC square cut -> design depth -> layers -> volumes ->
cold-mix mass and bags (+/- sigma) -> ancillaries -> labour -> rate-card cost -> method.

References: IRC:82-2015 Sec 10 & Cl. 11.4 (cut to sound material), IRC:116-2014 (readymade cold
mix, min. 50 mm patching layer), IRC:SP:100-2014 Sec 9 (cold maintenance mixes).

Pure library: inputs are ``measure.types.PotholeFused`` rows plus config dicts
(``configs/irc_rules.yaml``, ``configs/rate_card.*.yaml``) and a ``repair.types.PlanContext``.
No DB, no server imports. Units: m, mm (depths), m2, litres, kg, t, h, INR.

Bag counts are always an estimate with a +/-1 sigma range ("3 bags, range 3-4"); never
presented as an exact number.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple, Optional

from measure.types import PotholeFused, RoadSurface, Severity
from priority.severity import CLASS_DEPTH_FLOOR_MM, severity_of
from repair.loaders import rate_card_version
from repair.types import CostLine, PlanContext, RepairPlan, SpecCode

LANE_BAND_M = 1.75  # PLAN 5.3: lane band = round(lane_offset / 1.75 m)
_EPS = 1e-9

SPEC_NAMES_FALLBACK = {
    SpecCode.IRC116_COLDMIX: "Readymade cold mix (IRC:116-2014)",
    SpecCode.IRCSP100_EMULSION: "Emulsion cold mix (IRC:SP:100-2014)",
    SpecCode.JET_PATCHER: "Spray-injection jet patcher",
    SpecCode.INFRARED: "Infrared recycling patch",
    SpecCode.HOT_MIX: "Hot mix patch",
    SpecCode.CC_MORTAR: "Cement concrete surface: engineer item",
}
_COLDMIX_FAMILY = (SpecCode.IRC116_COLDMIX, SpecCode.IRCSP100_EMULSION)


# =============================================================================== helpers
def _ceil_step(x: float, step: float) -> float:
    """Ceil ``x`` to a multiple of ``step`` robustly against float noise (0.060/0.005 = 11.9999...)."""
    return math.ceil(x / step - 1e-6) * step


def _ceil_div(a: float, b: float) -> int:
    return int(math.ceil(a / b - 1e-9)) if a > 0 else 0


def _r(x: float, nd: int = 3) -> float:
    return float(round(x + 0.0, nd))


def _lane_band(p: PotholeFused) -> Optional[int]:
    if p.lane_band is not None:
        return p.lane_band
    if p.lane_offset_m is not None:
        return int(round(p.lane_offset_m / LANE_BAND_M))
    return None


def _coerce(p: Any) -> PotholeFused:
    if isinstance(p, PotholeFused):
        return p
    if isinstance(p, Mapping):
        known = set(PotholeFused.model_fields)
        return PotholeFused.model_validate({k: v for k, v in p.items() if k in known})
    raise TypeError(f"expected PotholeFused or mapping, got {type(p).__name__}")


def _rate(card: Mapping[str, Any], key: str) -> tuple[float, str, str]:
    item = card["items"][key]
    return float(item["rate"]), str(item.get("unit", "")), str(item.get("provenance", "rate card"))


def _bag_rate(card: Mapping[str, Any], bag_kg: int) -> tuple[float, str]:
    items = card["items"]
    key = f"coldmix_bag_{bag_kg}kg"
    if key in items:
        r, _, prov = _rate(card, key)
        return r, prov
    r25, _, prov25 = _rate(card, "coldmix_bag_25kg")
    return r25 / 25.0 * bag_kg, f"pro-rata from coldmix_bag_25kg ({prov25})"


# =============================================================================== merge
class MergedCut(NamedTuple):
    """One rectangular cut covering one or more potholes (PLAN 7.1)."""

    pothole: PotholeFused
    member_ids: list[int]


def _same_band(a: PotholeFused, b: PotholeFused) -> bool:
    if a.way_id is not None and b.way_id is not None and a.way_id != b.way_id:
        return False
    if a.direction is not None and b.direction is not None and a.direction != b.direction:
        return False
    la, lb = _lane_band(a), _lane_band(b)
    return la is None or lb is None or la == lb


def _local_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Equirectangular metres (east, north) around (lat0, lon0); plenty for sub-50 m gaps."""
    r = 6_371_008.8
    x = math.radians(lon - lon0) * r * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * r
    return x, y


def _use_chainage(ps: Sequence[PotholeFused]) -> bool:
    return all(p.chainage_m is not None and p.lane_offset_m is not None for p in ps) and \
        len({p.way_id for p in ps}) == 1


def footprint_gap_m(a: PotholeFused, b: PotholeFused) -> Optional[float]:
    """Edge-to-edge gap between two pothole footprints in metres (None if positions unknown).

    Preferred frame: (chainage along the way, lane offset across) with L along / W across.
    Fallback: lat/lon with footprints approximated by discs of radius (L+W)/4.
    """
    if _use_chainage([a, b]):
        ga = max(0.0, abs(a.chainage_m - b.chainage_m) - (a.length_m + b.length_m) / 2.0)
        gw = max(0.0, abs(a.lane_offset_m - b.lane_offset_m) - (a.width_m + b.width_m) / 2.0)
        return math.hypot(ga, gw)
    if None not in (a.lat, a.lon, b.lat, b.lon):
        x, y = _local_xy(b.lat, b.lon, a.lat, a.lon)
        ra = (a.length_m + a.width_m) / 4.0
        rb = (b.length_m + b.width_m) / 4.0
        return max(0.0, math.hypot(x, y) - ra - rb)
    return None


def group_for_merge(potholes: Iterable[Any], rules: Optional[Mapping[str, Any]] = None) -> list[list[PotholeFused]]:
    """Transitively group potholes within ``merge_within_m`` (0.5 m) on the same way/direction/lane band."""
    ps = [_coerce(p) for p in potholes]
    within = float(((rules or {}).get("repair") or {}).get("merge_within_m", 0.5))
    parent = list(range(len(ps)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if not _same_band(ps[i], ps[j]):
                continue
            gap = footprint_gap_m(ps[i], ps[j])
            if gap is not None and gap <= within + _EPS:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[PotholeFused]] = {}
    for i, p in enumerate(ps):
        groups.setdefault(find(i), []).append(p)
    return [groups[k] for k in sorted(groups)]


def _merge_group(group: Sequence[PotholeFused]) -> PotholeFused:
    if len(group) == 1:
        return group[0]
    first = group[0]
    if _use_chainage(group):
        lo_a = min(p.chainage_m - p.length_m / 2 for p in group)
        hi_a = max(p.chainage_m + p.length_m / 2 for p in group)
        lo_c = min(p.lane_offset_m - p.width_m / 2 for p in group)
        hi_c = max(p.lane_offset_m + p.width_m / 2 for p in group)
        length, width = hi_a - lo_a, hi_c - lo_c
        chain, off = (lo_a + hi_a) / 2, (lo_c + hi_c) / 2
    else:
        # lat/lon fallback: principal axis of the centroids = "along"; squares of side (L+W)/2.
        pts = [(p, *_local_xy(p.lat, p.lon, first.lat, first.lon)) for p in group]
        far = max(pts, key=lambda t: math.hypot(t[1], t[2]))
        n = math.hypot(far[1], far[2]) or 1.0
        ux, uy = far[1] / n, far[2] / n
        along = [(x * ux + y * uy, -x * uy + y * ux, (p.length_m + p.width_m) / 2) for p, x, y in pts]
        length = max(a + s / 2 for a, _, s in along) - min(a - s / 2 for a, _, s in along)
        width = max(c + s / 2 for _, c, s in along) - min(c - s / 2 for _, c, s in along)
        chain, off = first.chainage_m, first.lane_offset_m

    dry = [p for p in group if p.depth_p95_mm is not None]
    deepest = max(dry, key=lambda p: p.depth_p95_mm) if dry else None
    vols = [p.volume_l for p in group if p.volume_l is not None]
    vsig = [p.volume_sigma for p in group if p.volume_sigma is not None]
    area = sum(p.area_m2 for p in group)
    dmax = [p.depth_max_mm for p in group if p.depth_max_mm is not None]
    lats = [p.lat for p in group if p.lat is not None]
    lons = [p.lon for p in group if p.lon is not None]
    upd: dict[str, Any] = {
        "length_m": length,
        "width_m": width,
        "length_sigma_m": max((p.length_sigma_m or 0.0) for p in group) or None,
        "width_sigma_m": max((p.width_sigma_m or 0.0) for p in group) or None,
        "area_m2": area,
        "area_sigma": math.sqrt(sum(p.area_sigma ** 2 for p in group)),
        "depth_p95_mm": deepest.depth_p95_mm if deepest else None,
        "depth_sigma_mm": deepest.depth_sigma_mm if deepest else None,
        "depth_max_mm": max(dmax) if dmax else None,
        "volume_l": sum(vols) if vols else None,
        "volume_sigma": math.sqrt(sum(v ** 2 for v in vsig)) if vsig else None,
        "depth_mean_mm": (sum(vols) / area) if (vols and area > 0) else None,  # L / m2 == mm
        "water_filled": any(p.water_filled for p in group),
        "quality": min(p.quality for p in group),
        "complaint_count": sum(p.complaint_count for p in group),
        "imu_corroborated": any(bool(p.imu_corroborated) for p in group) or None,
        "surface": RoadSurface.CC if any(p.surface == RoadSurface.CC for p in group) else first.surface,
        "chainage_m": chain,
        "lane_offset_m": off,
        "lat": sum(lats) / len(lats) if lats else None,
        "lon": sum(lons) / len(lons) if lons else None,
        "severity": None,
    }
    d = first.model_dump()
    d.update(upd)
    return PotholeFused.model_validate(d)


def merge_cut(potholes: Iterable[Any], rules: Optional[Mapping[str, Any]] = None) -> list[MergedCut]:
    """PLAN 7.1: potholes within 0.5 m on the same lane band -> one cut.

    Returns one ``MergedCut`` per cut; ``member_ids`` lists the pothole ids it covers (input
    order). A merged cut's footprint is the bounding rectangle of its members, area/volume are
    summed and the design depth is governed by the deepest member.
    """
    out: list[MergedCut] = []
    for g in group_for_merge(potholes, rules):
        out.append(MergedCut(_merge_group(g), [p.id for p in g if p.id is not None]))
    return out


# =============================================================================== method
def select_method(p: PotholeFused, a_cut_m2: float, rules: Mapping[str, Any],
                  ctx: PlanContext) -> tuple[SpecCode, list[str]]:
    """PLAN 7.9 method selector. Returns (spec_code, reasons)."""
    ms = rules["repair"].get("method_selector", {})
    if p.surface == RoadSurface.CC:
        return SpecCode.CC_MORTAR, ["cement concrete surface -> engineer item (no bituminous patch)"]
    n_cluster = int(ms.get("jet_patcher_if_cluster_n_ge", 8))
    a_jet = float(ms.get("jet_patcher_if_a_cut_m2_gt", 1.0))
    radius = ms.get("jet_patcher_cluster_radius_m", 300)
    if ctx.cluster_n_within_300m >= n_cluster:
        return SpecCode.JET_PATCHER, [f"{ctx.cluster_n_within_300m} confirmed potholes within {radius} m (>= {n_cluster})"]
    if a_cut_m2 > a_jet:
        return SpecCode.JET_PATCHER, [f"A_cut {a_cut_m2:.2f} m2 > {a_jet} m2"]
    ir_depth = float(ms.get("infrared_if_depth_mm_le", 40))
    if (p.surface == RoadSurface.BT and not p.water_filled and p.depth_p95_mm is not None
            and p.depth_p95_mm <= ir_depth and ctx.dry_forecast_48h and not ctx.raining_now
            and ctx.infrared_unit_in_zone):
        return SpecCode.INFRARED, [f"BT, depth_p95 {p.depth_p95_mm:.0f} mm <= {ir_depth:.0f} mm, dry forecast, unit in zone"]
    plant_km = float(ms.get("hot_mix_if_plant_km_le", 10))
    if ctx.hot_mix_plant_km is not None and ctx.hot_mix_plant_km <= plant_km and not ctx.in_monsoon_window:
        return SpecCode.HOT_MIX, [f"hot-mix plant {ctx.hot_mix_plant_km:.1f} km <= {plant_km:.0f} km, off-season"]
    pref = ctx.agency_pref
    if pref in _COLDMIX_FAMILY:
        return pref, [f"agency preference {pref.value}"]
    return SpecCode.IRC116_COLDMIX, [f"agency preference {pref.value} not applicable by rule; default cold mix"]


# =============================================================================== plan
def plan(pothole: Any, rules: Mapping[str, Any], rate_card: Mapping[str, Any],
         context: Optional[PlanContext] = None) -> RepairPlan:
    """PLAN 7: square-cut cold-mix repair plan for one pothole (or one merged cut).

    Water-filled (or depth-unknown) potholes get ``pending_remeasure=True`` and LOWER-BOUND
    quantities from the IRC water lower-bound severity (PLAN 4.10): design depth = max(class floor,
    any known depth, 50 mm).
    """
    p = _coerce(pothole)
    ctx = context or PlanContext()
    rep = rules["repair"]
    notes: list[str] = []
    refs: list[str] = [str(rep.get("ref_cut", "IRC:82-2015 Sec 10, Cl. 11.4"))]
    extras: dict[str, Any] = {}

    # ---- 7.2 square cut ---------------------------------------------------------------
    length, width = float(p.length_m or 0.0), float(p.width_m or 0.0)
    if (length <= 0 or width <= 0) and p.area_m2 > 0:
        side = math.sqrt(p.area_m2)
        length, width = length or side, width or side
        notes.append("L/W missing: square footprint assumed from area")
    if length <= 0 or width <= 0:
        raise ValueError("pothole has no footprint (length/width/area all zero)")
    margin = float(rep["cut_margin_m"])
    l_cut, w_cut = length + 2 * margin, width + 2 * margin
    a_cut = l_cut * w_cut
    # sigma of A_cut: margins are exact, so only the L/W uncertainty propagates.
    sl, sw = p.length_sigma_m, p.width_sigma_m
    if (sl is None or sw is None) and p.area_m2 > 0 and p.area_sigma > 0:
        rel_dim = (p.area_sigma / p.area_m2) / math.sqrt(2.0)  # equal relative error on L and W
        sl = sl if sl is not None else length * rel_dim
        sw = sw if sw is not None else width * rel_dim
    sigma_a_cut = math.hypot(w_cut * (sl or 0.0), l_cut * (sw or 0.0))

    # ---- 7.3 design depth (water -> lower bound) ----------------------------------------
    sev, lower = severity_of(p, rules)
    pending = False
    min_patch_mm = float(rep["min_patch_depth_m"]) * 1000.0
    max_layer_mm = float(rep["max_layer_m"]) * 1000.0
    step_mm = float(rep.get("round_depth_to_m", 0.005)) * 1000.0
    if p.water_filled or p.depth_p95_mm is None:
        pending = True
        lb_sev = lower or Severity.medium
        floor_mm = CLASS_DEPTH_FLOOR_MM.get(lb_sev, 0.0)
        basis_mm = max(floor_mm, p.depth_p95_mm or 0.0)
        sigma_d_mm = 0.0  # lower bound: depth spread unknown, only area spread is propagated
        if p.water_filled:
            notes.append(f"Water-filled: depth unknown. Quantities are LOWER BOUNDS from IRC lower-bound "
                         f"severity '{lb_sev.value}'; re-measure after water recedes (rain + 48 h) or by inspection van.")
            notes.append("Bail out standing water before patching; damp cavity acceptable, active rain is not.")
            refs.append(str(rep.get("ref_wet", "IRC:82-2015 Sec 10")))
            refs.append(str(rules.get("severity", {}).get("ref", "IRC:82-2015 Cl. 7.5.3.4")) + " (water-filled lower bound, PLAN 4.10)")
        elif not (rep.get("depth_unknown_assumed_max_mm") or {}):
            notes.append("Depth unavailable: quantities are lower bounds at the minimum patch depth; re-measure.")
        extras["lower_bound_severity"] = lb_sev.value
    else:
        basis_mm = float(p.depth_p95_mm)
        if p.depth_sigma_mm is None:
            sigma_d_mm = max(3.0, 0.15 * basis_mm)
            notes.append("depth sigma missing: 15 % of depth_p95 assumed")
        else:
            sigma_d_mm = float(p.depth_sigma_mm)
    d_design_mm = _ceil_step(max(basis_mm, min_patch_mm), step_mm)
    refs.append("IRC:116-2014 Cl. 5.2 (patching mix layer >= 50 mm)")
    layered = d_design_mm > max_layer_mm + _EPS
    if layered:
        d_patch_mm = min_patch_mm
        d_crm_mm = d_design_mm - min_patch_mm
        crm_layers = _ceil_div(d_crm_mm, max_layer_mm)
        extras["crm_layers"] = crm_layers
        notes.append(f"Layered: {d_crm_mm:.0f} mm crusher-run macadam / 25 mm angular aggregate with MC-30 prime "
                     f"({crm_layers} layer(s) <= {max_layer_mm:.0f} mm) + {d_patch_mm:.0f} mm patching mix on top.")
    else:
        d_patch_mm, d_crm_mm = d_design_mm, 0.0

    # ---- 7.4 volumes (m3 internally, litres out) ------------------------------------------
    v_patch_m3 = a_cut * d_patch_mm / 1000.0
    v_crm_m3 = a_cut * d_crm_mm / 1000.0
    v_cavity_l = None if p.water_filled else p.volume_l
    v_cuboid_l = None if (p.water_filled or p.depth_max_mm is None) else length * width * p.depth_max_mm  # m2*mm = L

    # ---- 7.5 mass and bags +/- sigma -----------------------------------------------------
    rho = float(rep["mix_density_kg_m3"])
    wst = float(rep["wastage"])
    mix_kg = v_patch_m3 * rho * (1 + wst)
    # In the layered case the mix layer is fixed at 50 mm: depth uncertainty goes to the CRM layer.
    sd_patch_m = 0.0 if layered else sigma_d_mm / 1000.0
    mix_sigma = rho * (1 + wst) * math.hypot(a_cut * sd_patch_m, (d_patch_mm / 1000.0) * sigma_a_cut)
    bag_kg = int(ctx.bag_kg or rep["bag_kg_default"])
    if bag_kg <= 0:
        raise ValueError("bag_kg must be positive")
    if bag_kg not in rep.get("bag_kg_options", [bag_kg]):
        notes.append(f"bag size {bag_kg} kg is not in irc_rules bag_kg_options")
    bags = max(1, _ceil_div(mix_kg, bag_kg))
    bags_low = min(bags, max(1, _ceil_div(mix_kg - mix_sigma, bag_kg)))
    bags_high = max(bags, _ceil_div(mix_kg + mix_sigma, bag_kg))
    assumed = (rep.get("depth_unknown_assumed_max_mm") or {}).get(sev.value) if not p.water_filled else None
    if pending and p.depth_p95_mm is None and assumed is not None:
        # D27 size-only: depth not measured. Low = the depth basis above (IRC:116 minimum patch / class floor);
        # high = an assumed maximum depth for the size class, through the same layering rule.
        d_hi_mm = _ceil_step(max(float(assumed), d_design_mm), step_mm)
        patch_hi_mm = min_patch_mm if d_hi_mm > max_layer_mm + _EPS else d_hi_mm
        crm_hi_mm = d_hi_mm - patch_hi_mm
        mix_hi = a_cut * patch_hi_mm / 1000.0 * rho * (1 + wst)
        mix_hi_sigma = rho * (1 + wst) * (patch_hi_mm / 1000.0) * sigma_a_cut
        bags_high = max(bags_high, _ceil_div(mix_hi + mix_hi_sigma, bag_kg))
        extras["depth_assumed_range_mm"] = [_r(d_design_mm, 1), _r(d_hi_mm, 1)]
        extras["crm_t_high"] = _r(a_cut * crm_hi_mm / 1000.0 * float(rep.get("crm_density_t_m3", 2.1)), 4)
        rng = f"{bags_low}" if bags_low == bags_high else f"{bags_low}-{bags_high}"
        notes.append(f"Depth not measured (size-only, D27): {rng} bag(s) of {bag_kg} kg for a "
                     f"patch depth of {d_design_mm:.0f}-{d_hi_mm:.0f} mm (upper value assumed for a "
                     f"'{sev.value}' pothole)" + (f"; deeper than {max_layer_mm:.0f} mm the bottom is crusher-run, up to "
                                                     f"{extras['crm_t_high']:g} t" if crm_hi_mm > 0 else "")
                     + "; the crew confirms the depth on site.")
    elif pending:
        notes.append(f"Estimate: at least {bags} bag(s) of {bag_kg} kg; upper bound unknown until re-measured.")
    else:
        notes.append(f"Estimate: {bags} bag(s) of {bag_kg} kg (range {bags_low}-{bags_high} at +/-1 sigma).")

    # ---- 7.6 ancillaries ------------------------------------------------------------------
    emulsion_kg = float(rep.get("tack_emulsion_kg_m2", 0.30)) * a_cut
    sand_kg = float(rep.get("sand_blinding_kg_m2", 1.0)) * a_cut
    crm_t = v_crm_m3 * float(rep.get("crm_density_t_m3", 2.1))

    # ---- 7.7 labour ---------------------------------------------------------------------
    lab = rep.get("labour", {})
    base_h = float(lab.get("base_h_per_pothole", 2.0))
    per_m3 = float(lab.get("extra_h_per_m3", 40.0))
    vstep = 1.0 / per_m3 if per_m3 > 0 else math.inf  # 0.025 m3 -> 1.0 h
    extra_h = math.floor(max(0.0, v_patch_m3 - vstep) / vstep + 1e-6) if per_m3 > 0 else 0
    labour_h = (base_h + extra_h) * (1 + float(lab.get("supervision_frac", 0.10)))

    # ---- 7.9 method + weather gate ------------------------------------------------------------
    spec, reasons = select_method(p, a_cut, rules, ctx)
    spec_names = rep.get("spec_codes", {})
    method = str(spec_names.get(spec.value, SPEC_NAMES_FALLBACK[spec]))
    refs.append(str(rep.get("ref_mix", "IRC:116-2014; IRC:SP:100-2014 Sec 9")))
    weather_blocked = bool(ctx.raining_now)
    if weather_blocked:
        notes.append("Weather gate: active rain - patching blocked; surface-course cold mix is never laid in rain.")
    elif ctx.in_monsoon_window:
        notes.append("Monsoon window: schedule in a dry spell; damp holes may be patched after bailing.")
    if spec == SpecCode.CC_MORTAR:
        notes.append("CC surface: polymer-modified mortar / engineer item; bituminous quantities are indicative only.")

    # ---- 7.8 cost lines -------------------------------------------------------------------
    lines = _cost_lines(spec, rate_card, a_cut=a_cut, bags=bags, bag_kg=bag_kg, emulsion_kg=emulsion_kg,
                        sand_kg=sand_kg, crm_t=crm_t, labour_h=labour_h, notes=notes, extras=extras)
    est_cost = _r(sum(cl.amount_inr for cl in lines), 2)
    baseline = (rate_card.get("baseline") or {}).get("cost_per_pothole_inr")
    if baseline is not None:
        extras["baseline_provenance"] = rate_card["baseline"].get("provenance")
        extras["saving_vs_baseline_inr"] = _r(float(baseline) - est_cost, 2)

    extras.update({
        "severity": sev.value,
        "depth_basis_mm": _r(basis_mm, 1),
        "depth_sigma_mm": _r(sigma_d_mm, 2),
        "sigma_a_cut_m2": _r(sigma_a_cut, 5),
        "method_reasons": reasons,
        "labour_extra_steps": int(extra_h),
    })
    if v_cuboid_l is not None and v_patch_m3 > 0:
        extras["cuboid_l_vs_cavity_l"] = [_r(v_cuboid_l, 2), None if v_cavity_l is None else _r(v_cavity_l, 2)]

    return RepairPlan(
        pothole_id=p.id,
        merged_pothole_ids=list(ctx.merged_pothole_ids),
        l_cut_m=_r(l_cut, 4),
        w_cut_m=_r(w_cut, 4),
        a_cut_m2=_r(a_cut, 5),
        d_design_mm=_r(d_design_mm, 1),
        layered=layered,
        d_patch_mm=_r(d_patch_mm, 1),
        d_crm_mm=_r(d_crm_mm, 1),
        v_patch_l=_r(v_patch_m3 * 1000.0, 2),
        v_crm_l=_r(v_crm_m3 * 1000.0, 2),
        v_cavity_l=None if v_cavity_l is None else _r(v_cavity_l, 2),
        v_cuboid_l=None if v_cuboid_l is None else _r(v_cuboid_l, 2),
        mix_kg=_r(mix_kg, 2),
        mix_kg_sigma=_r(mix_sigma, 2),
        bag_kg=bag_kg,
        bags=bags,
        bags_low=bags_low,
        bags_high=bags_high,
        emulsion_kg=_r(emulsion_kg, 3),
        sand_kg=_r(sand_kg, 3),
        crm_t=_r(crm_t, 4),
        labour_h=_r(labour_h, 2),
        method=method,
        spec_code=spec,
        est_cost_inr=est_cost,
        cost_breakdown=lines,
        baseline_cost_inr=None if baseline is None else float(baseline),
        rate_card_version=rate_card_version(rate_card),
        pending_remeasure=pending,
        weather_blocked=weather_blocked,
        notes=notes,
        refs=list(dict.fromkeys(refs)),
        extras=extras,
    )


def _line(item: str, qty: float, unit: str, rate: float, provenance: str) -> CostLine:
    return CostLine(item=item, qty=qty, unit=unit, rate_inr=_r(rate, 2), amount_inr=_r(qty * rate, 2),
                    provenance=provenance)


def _coldmix_lines(card: Mapping[str, Any], *, bags: int, bag_kg: int, emulsion_kg: float, sand_kg: float,
                   crm_t: float, labour_h: float) -> list[CostLine]:
    lines: list[CostLine] = []
    rate, prov = _bag_rate(card, bag_kg)
    lines.append(_line(f"Cold mix, {bag_kg} kg bag", float(bags), "bag", rate, prov))
    r, u, prov = _rate(card, "tack_emulsion_kg")
    lines.append(_line("Tack/prime emulsion SS-1/CSS-1", _r(emulsion_kg, 3), u or "kg", r, prov))
    if crm_t > 0:
        r, u, prov = _rate(card, "crm_tonne")
        lines.append(_line("Crusher run macadam (bottom layer)", _r(crm_t, 4), u or "t", r, prov))
    r, u, prov = _rate(card, "sand_kg")
    lines.append(_line("Sand blinding", _r(sand_kg, 3), u or "kg", r, prov))
    r, u, prov = _rate(card, "labour_hour")
    lines.append(_line("Labour (mazdoor incl. 10 % mate/supervision)", _r(labour_h, 2), u or "h", r, prov))
    return lines


def _overhead(card: Mapping[str, Any], lines: list[CostLine]) -> Optional[CostLine]:
    if "overhead_frac" not in card["items"]:
        return None
    frac, _, prov = _rate(card, "overhead_frac")
    sub = _r(sum(cl.amount_inr for cl in lines), 2)
    return CostLine(item=f"T&P + transport ({frac * 100:.0f} %)", qty=sub, unit="INR", rate_inr=frac,
                    amount_inr=_r(sub * frac, 2), provenance=prov)


_ENGINEER_RATE_KEYS = {SpecCode.INFRARED: "infrared_m2", SpecCode.HOT_MIX: "hot_mix_m2", SpecCode.CC_MORTAR: "cc_mortar_m2"}


def _cost_lines(spec: SpecCode, card: Mapping[str, Any], *, a_cut: float, bags: int, bag_kg: int, emulsion_kg: float,
                sand_kg: float, crm_t: float, labour_h: float, notes: list[str], extras: dict[str, Any]) -> list[CostLine]:
    items = card["items"]
    coldmix = _coldmix_lines(card, bags=bags, bag_kg=bag_kg, emulsion_kg=emulsion_kg, sand_kg=sand_kg,
                             crm_t=crm_t, labour_h=labour_h)
    oh = _overhead(card, coldmix)
    coldmix_all = coldmix + ([oh] if oh else [])
    coldmix_total = _r(sum(cl.amount_inr for cl in coldmix_all), 2)

    area_key = "jet_patcher_m2" if spec == SpecCode.JET_PATCHER else _ENGINEER_RATE_KEYS.get(spec)
    if area_key and area_key in items:
        r, u, prov = _rate(card, area_key)
        extras["material_basis"] = "cold-mix equivalent (bags/emulsion shown for indent comparison)"
        extras["coldmix_alternative_cost_inr"] = coldmix_total
        notes.append(f"{spec.value} priced per m2 of cut from the rate card ({prov}); "
                     f"cold-mix alternative Rs {coldmix_total:,.0f}.")
        return [_line(f"{spec.value} patching (all-inclusive rate)", _r(a_cut, 4), u or "m2", r, prov)]
    if area_key:
        notes.append(f"No rate-card item '{area_key}' for {spec.value}: cost shown on cold-mix basis; "
                     f"engineer to price from the circle SOR.")
        extras["material_basis"] = "cold-mix basis (engineer item not in rate card)"
    return coldmix_all


def plan_many(potholes: Iterable[Any], rules: Mapping[str, Any], rate_card: Mapping[str, Any],
              context: Optional[PlanContext] = None) -> list[RepairPlan]:
    """Merge nearby potholes (PLAN 7.1) and plan each resulting cut."""
    base = context or PlanContext()
    plans: list[RepairPlan] = []
    for cut in merge_cut(potholes, rules):
        ctx = base.model_copy(update={"merged_pothole_ids": cut.member_ids if len(cut.member_ids) > 1 else []})
        plans.append(plan(cut.pothole, rules, rate_card, ctx))
    return plans


__all__ = ["MergedCut", "footprint_gap_m", "group_for_merge", "merge_cut", "plan", "plan_many", "select_method"]
