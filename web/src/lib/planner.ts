/**
 * Client-side port of the PLAN 7 repair planner, used ONLY by the mock API and by what-if previews.
 * The server (repair/planner.py, WP8) is authoritative; this port follows it step by step (square cut,
 * design depth, layering, mass and bags +/- sigma, ancillaries, stepped labour, method selector, cost
 * lines) and `planner.test.ts` checks it against the Python worked example (repair/tests/test_planner.py).
 * Constants mirror configs/irc_rules.yaml and configs/rate_card.ghmc_fy2026.yaml.
 *
 * Differences from the server, by design: cost-line `item` is the rate-card key (translated in the UI via
 * `rateItem.*`), and merging of nearby potholes (PLAN 7.1) is not ported.
 * Bag counts are an estimate with a +/-1 sigma range, never an exact number.
 */
import type { CostLine, DepthQuality, PlanContext, RateCard, RepairPlan, RoadSurface, SpecCode } from './types';

export const IRC = {
  ref_cut: 'IRC:82-2015 Sec 10, Cl. 11.4 (cut to sound material, square/rectangular)',
  ref_mix: 'IRC:116-2014 (readymade cold mix); IRC:SP:100-2014 Sec 9 (cold maintenance mixes)',
  ref_wet: 'IRC:82-2015 Sec 10: pothole need not be dry; excess water shall be removed',
  ref_severity: 'IRC:82-2015 Cl. 7.5.3.4',
  cut_margin_m: 0.125,
  min_patch_depth_m: 0.05,
  max_layer_m: 0.075,
  round_depth_to_m: 0.005,
  mix_density_kg_m3: 2200,
  wastage: 0.12,
  bag_kg_default: 25,
  tack_emulsion_kg_m2: 0.3,
  sand_blinding_kg_m2: 1.0,
  crm_density_t_m3: 2.1,
  labour_base_h: 2.0,
  labour_extra_h_per_m3: 40,
  supervision_frac: 0.1,
  jet_patcher_if_cluster_n_ge: 8,
  jet_patcher_cluster_radius_m: 300,
  jet_patcher_if_a_cut_m2_gt: 1.0,
  infrared_if_depth_mm_le: 40,
  hot_mix_if_plant_km_le: 10,
  // PLAN 4.10 water-filled lower bound (irc_rules.severity.water_filled_lower_bound)
  water_medium_if_width_mm_ge: 200,
  water_large_if_width_mm_gt: 500,
  water_large_if_complaints_ge: 2,
};

/** Depth lower bound (mm) implied by each IRC class (bins are inclusive upper bounds). */
export const CLASS_DEPTH_FLOOR_MM = { small: 0, medium: 25, large: 50 } as const;

export const DEFAULT_RATE_CARD: RateCard = {
  agency: 'GHMC',
  fy: '2026-27',
  version: 1,
  currency: 'INR',
  items: {
    coldmix_bag_25kg: { rate: 350, unit: 'bag', provenance: 'IndiaMART / HINCOL retail listings 2025-26 (demo)' },
    coldmix_bag_50kg: { rate: 680, unit: 'bag', provenance: 'IndiaMART bulk listings 2025-26 (demo)' },
    tack_emulsion_kg: { rate: 45, unit: 'kg', provenance: 'SS-1 emulsion trade price (demo)' },
    crm_tonne: { rate: 1100, unit: 't', provenance: 'Crusher run aggregate, Hyderabad quarry (demo)' },
    sand_kg: { rate: 2, unit: 'kg', provenance: 'demo' },
    labour_hour: { rate: 100, unit: 'h', provenance: 'Unskilled mazdoor ~Rs 800/day, Telangana min. wage band (demo)' },
    jet_patcher_m2: { rate: 1942, unit: 'm2', provenance: 'BBMP Bengaluru 2025 reference - replace with GHMC circle tender rate' },
    overhead_frac: { rate: 0.1, unit: 'frac', provenance: 'T&P + transport, demo' },
  },
  baseline: { cost_per_pothole_inr: 7396, provenance: 'GHMC reported spend per pothole, Deccan Chronicle 2021 (vs Rs 1,600 budgeted)' },
};

/** configs/irc_rules.yaml repair.spec_codes */
export const SPEC_LABEL: Record<SpecCode, string> = {
  IRC116_COLDMIX: 'Storable cut-back (MC-800) readymade cold mix, 4.5-5.5 % residual bitumen (IRC:116-2014)',
  IRCSP100_EMULSION: 'Emulsion (SS-2/MS) cold mix, 7-8 % by weight of aggregate (IRC:SP:100-2014)',
  JET_PATCHER: 'Spray-injection jet patcher (velocity patching)',
  INFRARED: 'Infrared recycling patch',
  HOT_MIX: 'Hot mix BC/SDBC patch (plant <= 10 km, off-season only)',
  CC_MORTAR: 'Cement concrete surface: polymer-modified mortar / engineer item',
};

/** Specs priced per m2 of cut when the rate card has the item (otherwise shown on a cold-mix basis). */
const AREA_RATE_KEY: Partial<Record<SpecCode, string>> = {
  JET_PATCHER: 'jet_patcher_m2', INFRARED: 'infrared_m2', HOT_MIX: 'hot_mix_m2', CC_MORTAR: 'cc_mortar_m2',
};
const COLDMIX_FAMILY: SpecCode[] = ['IRC116_COLDMIX', 'IRCSP100_EMULSION'];

export interface PlannerInput {
  pothole_id?: number | null;
  length_m: number;
  width_m: number;
  length_sigma_m?: number | null;
  width_sigma_m?: number | null;
  area_m2: number;
  area_sigma: number;
  depth_p95_mm?: number | null;
  depth_sigma_mm?: number | null;
  depth_max_mm?: number | null;
  depth_quality?: DepthQuality;
  volume_l?: number | null;
  water_filled: boolean;
  surface?: RoadSurface;
  complaint_count?: number;
}

const EPS = 1e-9;
/** Ceil to a multiple of `step`, robust against float noise (0.060/0.005 = 11.999...). */
const ceilStep = (x: number, step: number) => Math.ceil(x / step - 1e-6) * step;
const ceilDiv = (a: number, b: number) => (a > 0 ? Math.ceil(a / b - EPS) : 0);
const r = (x: number, d = 3) => Math.round(x * 10 ** d) / 10 ** d;

/** PLAN 4.10: lower-bound IRC class of a water-filled pothole from its width and complaint count. */
export function waterLowerBound(width_mm: number, complaintCount = 0): 'small' | 'medium' | 'large' {
  if (width_mm > IRC.water_large_if_width_mm_gt || complaintCount >= IRC.water_large_if_complaints_ge) return 'large';
  if (width_mm >= IRC.water_medium_if_width_mm_ge) return 'medium';
  return 'small';
}

/** PLAN 7.9 method selector (repair/planner.py select_method). */
export function selectMethod(p: PlannerInput, a_cut_m2: number, ctx: PlanContext = {}): { spec: SpecCode; reasons: string[] } {
  const surface = p.surface ?? 'BT';
  if (surface === 'CC') return { spec: 'CC_MORTAR', reasons: ['cement concrete surface -> engineer item (no bituminous patch)'] };
  const cluster = ctx.cluster_n_within_300m ?? 0;
  if (cluster >= IRC.jet_patcher_if_cluster_n_ge) {
    return { spec: 'JET_PATCHER', reasons: [`${cluster} confirmed potholes within ${IRC.jet_patcher_cluster_radius_m} m (>= ${IRC.jet_patcher_if_cluster_n_ge})`] };
  }
  if (a_cut_m2 > IRC.jet_patcher_if_a_cut_m2_gt) {
    return { spec: 'JET_PATCHER', reasons: [`A_cut ${a_cut_m2.toFixed(2)} m2 > ${IRC.jet_patcher_if_a_cut_m2_gt} m2`] };
  }
  const dry = ctx.dry_forecast_48h ?? true;
  if (surface === 'BT' && !p.water_filled && p.depth_p95_mm != null && p.depth_p95_mm <= IRC.infrared_if_depth_mm_le
    && dry && !ctx.raining_now && ctx.infrared_unit_in_zone) {
    return { spec: 'INFRARED', reasons: [`BT, depth_p95 ${Math.round(p.depth_p95_mm)} mm <= ${IRC.infrared_if_depth_mm_le} mm, dry forecast, unit in zone`] };
  }
  if (ctx.hot_mix_plant_km != null && ctx.hot_mix_plant_km <= IRC.hot_mix_if_plant_km_le && !ctx.in_monsoon_window) {
    return { spec: 'HOT_MIX', reasons: [`hot-mix plant ${ctx.hot_mix_plant_km.toFixed(1)} km <= ${IRC.hot_mix_if_plant_km_le} km, off-season`] };
  }
  const pref = ctx.agency_pref ?? 'IRC116_COLDMIX';
  if (COLDMIX_FAMILY.includes(pref)) return { spec: pref, reasons: [`agency preference ${pref}`] };
  return { spec: 'IRC116_COLDMIX', reasons: [`agency preference ${pref} not applicable by rule; default cold mix`] };
}

export function planRepair(p: PlannerInput, ctx: PlanContext = {}, card: RateCard = DEFAULT_RATE_CARD): RepairPlan {
  const notes: string[] = [];
  const refs: string[] = [IRC.ref_cut];
  const extras: Record<string, unknown> = {};

  // ---- 7.2 square cut
  let L = p.length_m > 0 ? p.length_m : 0;
  let W = p.width_m > 0 ? p.width_m : 0;
  if ((L <= 0 || W <= 0) && p.area_m2 > 0) {
    const side = Math.sqrt(p.area_m2);
    L = L || side;
    W = W || side;
    notes.push('L/W missing: square footprint assumed from area');
  }
  if (L <= 0 || W <= 0) throw new Error('pothole has no footprint (length/width/area all zero)');
  const m = IRC.cut_margin_m;
  const l_cut = L + 2 * m;
  const w_cut = W + 2 * m;
  const a_cut = l_cut * w_cut;
  // sigma of A_cut: the margins are exact, only the L/W uncertainty propagates
  let sl = p.length_sigma_m ?? null;
  let sw = p.width_sigma_m ?? null;
  if ((sl == null || sw == null) && p.area_m2 > 0 && p.area_sigma > 0) {
    const relDim = p.area_sigma / p.area_m2 / Math.SQRT2; // equal relative error on L and W
    sl ??= L * relDim;
    sw ??= W * relDim;
  }
  const sigma_a_cut = Math.hypot(w_cut * (sl ?? 0), l_cut * (sw ?? 0));

  // ---- 7.3 design depth (water-filled / unknown depth -> lower bound, PLAN 4.10)
  const water = p.water_filled || (p.depth_p95_mm == null && p.depth_quality === 'unknown_water');
  const pending = p.water_filled || p.depth_p95_mm == null;
  const minPatchMm = IRC.min_patch_depth_m * 1000;
  const maxLayerMm = IRC.max_layer_m * 1000;
  let basisMm: number;
  let sigmaDMm: number;
  if (pending) {
    const lb = water ? waterLowerBound(W * 1000, p.complaint_count ?? 0) : 'medium';
    basisMm = Math.max(CLASS_DEPTH_FLOOR_MM[lb], p.depth_p95_mm ?? 0);
    sigmaDMm = 0; // lower bound: depth spread unknown, only the area spread is propagated
    if (p.water_filled) {
      notes.push(`Water-filled: depth unknown. Quantities are LOWER BOUNDS from IRC lower-bound severity '${lb}'; re-measure after water recedes (rain + 48 h) or by inspection van.`);
      notes.push('Bail out standing water before patching; damp cavity acceptable, active rain is not.');
      refs.push(IRC.ref_wet, `${IRC.ref_severity} (water-filled lower bound, PLAN 4.10)`);
    } else {
      notes.push('Depth unavailable: quantities are lower bounds at the minimum patch depth; re-measure.');
    }
    extras.lower_bound_severity = lb;
  } else {
    basisMm = p.depth_p95_mm as number;
    if (p.depth_sigma_mm == null) {
      sigmaDMm = Math.max(3, 0.15 * basisMm);
      notes.push('depth sigma missing: 15 % of depth_p95 assumed');
    } else sigmaDMm = p.depth_sigma_mm;
  }
  const dDesignMm = ceilStep(Math.max(basisMm, minPatchMm), IRC.round_depth_to_m * 1000);
  refs.push('IRC:116-2014 Cl. 5.2 (patching mix layer >= 50 mm)');
  const layered = dDesignMm > maxLayerMm + EPS;
  const dPatchMm = layered ? minPatchMm : dDesignMm;
  const dCrmMm = layered ? dDesignMm - minPatchMm : 0;
  if (layered) {
    const crmLayers = ceilDiv(dCrmMm, maxLayerMm);
    extras.crm_layers = crmLayers;
    notes.push(`Layered: ${Math.round(dCrmMm)} mm crusher-run macadam / 25 mm angular aggregate with MC-30 prime (${crmLayers} layer(s) <= ${maxLayerMm} mm) + ${Math.round(dPatchMm)} mm patching mix on top.`);
  }

  // ---- 7.4 volumes (m3 internally, litres out)
  const v_patch = (a_cut * dPatchMm) / 1000;
  const v_crm = (a_cut * dCrmMm) / 1000;
  const v_cavity_l = p.water_filled ? null : p.volume_l ?? null;
  const v_cuboid_l = p.water_filled || p.depth_max_mm == null ? null : L * W * p.depth_max_mm; // m2 * mm = L

  // ---- 7.5 mass and bags +/- sigma
  const rho = IRC.mix_density_kg_m3 * (1 + IRC.wastage);
  const mix_kg = v_patch * rho;
  // layered: the mix layer is fixed at 50 mm, so the depth uncertainty goes to the CRM layer
  const sdPatchM = layered ? 0 : sigmaDMm / 1000;
  const mix_sigma = rho * Math.hypot(a_cut * sdPatchM, (dPatchMm / 1000) * sigma_a_cut);
  const bag_kg = ctx.bag_kg || IRC.bag_kg_default;
  if (!(bag_kg > 0)) throw new Error('bag_kg must be positive');
  const bags = Math.max(1, ceilDiv(mix_kg, bag_kg));
  const bags_low = Math.min(bags, Math.max(1, ceilDiv(mix_kg - mix_sigma, bag_kg)));
  const bags_high = Math.max(bags, ceilDiv(mix_kg + mix_sigma, bag_kg));
  notes.push(pending
    ? `Estimate: at least ${bags} bag(s) of ${bag_kg} kg; upper bound unknown until re-measured.`
    : `Estimate: ${bags} bag(s) of ${bag_kg} kg (range ${bags_low}-${bags_high} at +/-1 sigma).`);

  // ---- 7.6 ancillaries, 7.7 labour (1.0 h per complete additional 0.025 m3 of patch volume)
  const emulsion = IRC.tack_emulsion_kg_m2 * a_cut;
  const sand = IRC.sand_blinding_kg_m2 * a_cut;
  const crm_t = v_crm * IRC.crm_density_t_m3;
  const vStep = 1 / IRC.labour_extra_h_per_m3;
  const extraH = Math.floor(Math.max(0, v_patch - vStep) / vStep + 1e-6);
  const labour = (IRC.labour_base_h + extraH) * (1 + IRC.supervision_frac);

  // ---- 7.9 method + weather gate
  const { spec, reasons } = selectMethod(p, a_cut, ctx);
  refs.push(IRC.ref_mix);
  const weather_blocked = !!ctx.raining_now;
  if (weather_blocked) notes.push('Weather gate: active rain - patching blocked; surface-course cold mix is never laid in rain.');
  else if (ctx.in_monsoon_window) notes.push('Monsoon window: schedule in a dry spell; damp holes may be patched after bailing.');
  if (spec === 'CC_MORTAR') notes.push('CC surface: polymer-modified mortar / engineer item; bituminous quantities are indicative only.');

  // ---- 7.8 cost lines
  const it = card.items;
  const line = (item: string, qty: number, unit: string, rate: number, provenance?: string): CostLine =>
    ({ item, qty, unit, rate_inr: r(rate, 2), amount_inr: r(qty * rate, 2), provenance: provenance ?? 'rate card' });
  const bagKey = `coldmix_bag_${bag_kg}kg`;
  const bagLine = it[bagKey]
    ? line(bagKey, bags, 'bag', it[bagKey].rate, it[bagKey].provenance)
    : line('coldmix_bag_25kg', bags, 'bag', (it.coldmix_bag_25kg.rate / 25) * bag_kg, `pro-rata from coldmix_bag_25kg (${it.coldmix_bag_25kg.provenance ?? 'rate card'})`);
  const coldmix: CostLine[] = [bagLine, line('tack_emulsion_kg', r(emulsion, 3), it.tack_emulsion_kg.unit || 'kg', it.tack_emulsion_kg.rate, it.tack_emulsion_kg.provenance)];
  if (crm_t > 0) coldmix.push(line('crm_tonne', r(crm_t, 4), it.crm_tonne.unit || 't', it.crm_tonne.rate, it.crm_tonne.provenance));
  coldmix.push(line('sand_kg', r(sand, 3), it.sand_kg.unit || 'kg', it.sand_kg.rate, it.sand_kg.provenance));
  coldmix.push(line('labour_hour', r(labour, 2), it.labour_hour.unit || 'h', it.labour_hour.rate, it.labour_hour.provenance));
  const ovh = it.overhead_frac;
  if (ovh) {
    const sub = r(coldmix.reduce((s, l) => s + l.amount_inr, 0), 2);
    // qty = fraction, rate = subtotal (the UI shows the rate column as money)
    coldmix.push({ item: 'overhead_frac', qty: ovh.rate, unit: 'frac', rate_inr: sub, amount_inr: r(sub * ovh.rate, 2), provenance: ovh.provenance ?? 'rate card' });
  }
  const coldmixTotal = r(coldmix.reduce((s, l) => s + l.amount_inr, 0), 2);
  let lines = coldmix;
  const areaKey = AREA_RATE_KEY[spec];
  if (areaKey && it[areaKey]) {
    const a = it[areaKey];
    extras.material_basis = 'cold-mix equivalent (bags/emulsion shown for indent comparison)';
    extras.coldmix_alternative_cost_inr = coldmixTotal;
    notes.push(`${spec} priced per m2 of cut from the rate card (${a.provenance ?? 'rate card'}); cold-mix alternative Rs ${Math.round(coldmixTotal)}.`);
    lines = [line(areaKey, r(a_cut, 4), a.unit || 'm2', a.rate, a.provenance)];
  } else if (areaKey) {
    notes.push(`No rate-card item '${areaKey}' for ${spec}: cost shown on cold-mix basis; engineer to price from the circle SOR.`);
    extras.material_basis = 'cold-mix basis (engineer item not in rate card)';
  }
  const est = r(lines.reduce((s, l) => s + l.amount_inr, 0), 2);
  const baseline = card.baseline?.cost_per_pothole_inr ?? null;
  if (baseline != null) extras.saving_vs_baseline_inr = r(baseline - est, 2);
  Object.assign(extras, {
    depth_basis_mm: r(basisMm, 1), depth_sigma_mm: r(sigmaDMm, 2), sigma_a_cut_m2: r(sigma_a_cut, 5),
    method_reasons: reasons, labour_extra_steps: extraH,
  });

  return {
    pothole_id: p.pothole_id ?? null,
    merged_pothole_ids: ctx.merged_pothole_ids ?? [],
    l_cut_m: r(l_cut, 4), w_cut_m: r(w_cut, 4), a_cut_m2: r(a_cut, 5),
    d_design_mm: r(dDesignMm, 1), layered, d_patch_mm: r(dPatchMm, 1), d_crm_mm: r(dCrmMm, 1),
    v_patch_l: r(v_patch * 1000, 2), v_crm_l: r(v_crm * 1000, 2),
    v_cavity_l: v_cavity_l == null ? null : r(v_cavity_l, 2),
    v_cuboid_l: v_cuboid_l == null ? null : r(v_cuboid_l, 2),
    mix_kg: r(mix_kg, 2), mix_kg_sigma: r(mix_sigma, 2), bag_kg, bags, bags_low, bags_high,
    emulsion_kg: r(emulsion, 3), sand_kg: r(sand, 3), crm_t: r(crm_t, 4), labour_h: r(labour, 2),
    method: SPEC_LABEL[spec], spec_code: spec, est_cost_inr: est, cost_breakdown: lines,
    baseline_cost_inr: baseline, rate_card_version: `${card.agency}-${card.fy}-v${card.version}`,
    pending_remeasure: pending, weather_blocked, notes, refs: [...new Set(refs)], extras,
  };
}

/** IRC:82-2015 Cl. 7.5.3.4 severity (measure/irc.py is authoritative). */
export function ircSeverity(depth_p95_mm: number | null | undefined, width_m: number, water_filled: boolean): 'small' | 'medium' | 'large' | 'unknown_water' {
  // Same rule as measure/irc.py: water -> unknown_water (the planner uses waterLowerBound separately);
  // unknown depth without water -> classified on width alone (a lower bound).
  if (water_filled) return 'unknown_water';
  const wmm = width_m * 1000;
  const d = depth_p95_mm ?? 0;
  if (d > 50 || wmm > 500) return 'large';
  if (d > 25 || wmm > 200) return 'medium';
  return 'small';
}
