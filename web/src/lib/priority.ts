/**
 * Priority breakdown helpers (PLAN 6). The server stores `priority_breakdown` as free JSON (WP5 decides the
 * exact layout), so parsing here is tolerant: components may be numbers or objects with score/value/weight.
 */
export const COMPONENTS = ['S', 'E', 'V', 'T', 'M'] as const;
export type Component = (typeof COMPONENTS)[number];

export const DEFAULT_WEIGHTS: Record<Component, number> = { S: 0.3, E: 0.25, V: 0.2, T: 0.15, M: 0.1 };

export interface ComponentRow {
  key: Component;
  score: number; // 0..1
  weight: number; // 0..1
  points: number; // contribution to the 0..100 priority
  why: string[];
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

export function parseBreakdown(bd: Record<string, unknown> | null | undefined, fallbackWeights = DEFAULT_WEIGHTS): ComponentRow[] {
  if (!bd) return [];
  const container = (bd.components ?? bd.scores ?? bd) as Record<string, unknown>;
  const weights = (bd.weights ?? {}) as Record<string, unknown>;
  const rows: ComponentRow[] = [];
  for (const key of COMPONENTS) {
    const raw = container[key] ?? container[key.toLowerCase()];
    if (raw === undefined) continue;
    let score: number | null = null;
    let weight: number | null = num(weights[key]);
    let why: string[] = [];
    if (typeof raw === 'number') score = raw;
    else if (raw && typeof raw === 'object') {
      const o = raw as Record<string, unknown>;
      score = num(o.score) ?? num(o.value) ?? num(o.s);
      weight = num(o.weight) ?? num(o.w) ?? weight;
      const w = o.why ?? o.reasons ?? o.notes;
      if (Array.isArray(w)) why = w.map(String);
      else if (typeof w === 'string') why = [w];
    }
    if (score === null) continue;
    const wt = weight ?? fallbackWeights[key];
    rows.push({ key, score: Math.max(0, Math.min(1, score)), weight: wt, points: 100 * wt * Math.max(0, Math.min(1, score)), why });
  }
  return rows;
}

/** What-if: priority if the monsoon component were `mNew` (PLAN 6 M: base 0.5, window 0.8, heavy rain 1.0). */
export function priorityWithMonsoon(priority: number | null | undefined, bd: Record<string, unknown> | null | undefined, mNew: number): number | null {
  if (priority == null) return null;
  const rows = parseBreakdown(bd);
  const m = rows.find((r) => r.key === 'M');
  if (!m) return priority;
  return Math.max(0, Math.min(100, priority + 100 * m.weight * (mNew - m.score)));
}

export function weightsSumOk(w: Record<string, number>): boolean {
  const s = COMPONENTS.reduce((acc, k) => acc + (w[k] ?? 0), 0);
  return Math.abs(s - 1) < 0.005;
}
