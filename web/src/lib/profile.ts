/**
 * Depth cross-section for the pothole detail chart.
 *
 * When the server provides a depth map (MeasurementOut.depth_map_url) the profile would be sampled from it.
 * Without one (mock / no stored depth PNG) we draw a MODELLED profile that is consistent with the fused
 * statistics: a symmetric bowl z(u) = d_max * (1 - |u|^k), u in [-1, 1] across the width, whose mean depth
 * equals depth_mean (mean of 1-|u|^k is k/(k+1)), with a +/- sigma band. It is labelled "modelled" in the UI.
 */
export interface ProfilePoint { x_m: number; depth_mm: number; lo_mm: number; hi_mm: number; band: [number, number] }

export interface ProfileInput {
  width_m: number;
  depth_max_mm?: number | null;
  depth_p95_mm?: number | null;
  depth_mean_mm?: number | null;
  depth_sigma_mm?: number | null;
}

export function bowlExponent(meanMm: number | null | undefined, maxMm: number): number {
  if (meanMm == null || !(maxMm > 0)) return 2;
  const r = Math.max(0.3, Math.min(0.86, meanMm / maxMm));
  return Math.max(0.6, Math.min(6, r / (1 - r)));
}

/** Depths are positive-down (mm below the fitted road surface). Includes a 0.15 m road margin each side. */
export function depthProfile(p: ProfileInput, n = 41, margin_m = 0.15): ProfilePoint[] | null {
  const dmax = p.depth_max_mm ?? (p.depth_p95_mm != null ? p.depth_p95_mm * 1.15 : null);
  if (dmax == null || !(dmax > 0) || !(p.width_m > 0)) return null;
  const k = bowlExponent(p.depth_mean_mm, dmax);
  const sigma = p.depth_sigma_mm ?? 0;
  const half = p.width_m / 2;
  const total = p.width_m + 2 * margin_m;
  const out: ProfilePoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = -half - margin_m + (total * i) / (n - 1);
    const u = Math.abs(x) / half;
    const d = u >= 1 ? 0 : dmax * (1 - u ** k);
    const s = u >= 1 ? 0 : sigma * Math.sqrt(1 - u * u * 0.5);
    const lo = Math.max(0, d - s);
    const hi = d + s;
    out.push({ x_m: Math.round(x * 1000) / 1000, depth_mm: Math.round(d * 10) / 10, lo_mm: Math.round(lo * 10) / 10, hi_mm: Math.round(hi * 10) / 10, band: [Math.round(lo * 10) / 10, Math.round(hi * 10) / 10] });
  }
  return out;
}

/** Mean of the modelled profile inside the cavity (for tests / sanity display). */
export function profileMean(pts: ProfilePoint[], width_m: number): number {
  const inside = pts.filter((q) => Math.abs(q.x_m) < width_m / 2);
  return inside.length ? inside.reduce((s, q) => s + q.depth_mm, 0) / inside.length : 0;
}
