/**
 * Client logic: the TS planner must agree with the authoritative Python planner (repair/tests/test_planner.py
 * worked example, PLAN 7.10); priority weights; WO/pothole state machines; mock API flows; offline outbox.
 */
import { describe, expect, it } from 'vitest';
import { planRepair, ircSeverity, waterLowerBound, type PlannerInput } from './planner';
import { DEFAULT_WEIGHTS, parseBreakdown, weightsSumOk } from './priority';
import { allowedWorkOrderActions, potholeCan, workOrderCan, INVOICE_ELIGIBLE } from './states';
import { MockApi } from './mock/mockApi';
import { FieldDb } from './offline/db';
import { OfflineQueue, type SyncApi } from './offline/queue';
import { ApiError, NetworkError } from './api';

const WORKED: PlannerInput = {
  length_m: 0.5, width_m: 0.4, area_m2: 0.16, area_sigma: 0.008, depth_p95_mm: 60, depth_sigma_mm: 8,
  depth_max_mm: 60, volume_l: 6, water_filled: false,
};

describe('planner (port of repair/planner.py)', () => {
  it('matches the PLAN 7.10 worked example', () => {
    const p = planRepair(WORKED);
    expect(p.l_cut_m).toBeCloseTo(0.75, 6);
    expect(p.w_cut_m).toBeCloseTo(0.65, 6);
    expect(p.a_cut_m2).toBeCloseTo(0.4875, 6);
    expect([p.d_design_mm, p.layered, p.d_patch_mm, p.d_crm_mm]).toEqual([60, false, 60, 0]);
    expect(p.v_patch_l).toBeCloseTo(29.25, 2);
    expect(Math.abs(p.mix_kg - 72.07)).toBeLessThan(0.05);
    expect([p.bag_kg, p.bags, p.bags_low, p.bags_high]).toEqual([25, 3, 3, 4]);
    expect(p.labour_h).toBeCloseTo(2.2, 6);
    expect(Math.abs(p.est_cost_inr - 1406)).toBeLessThanOrEqual(2);
    expect(p.spec_code).toBe('IRC116_COLDMIX');
    const sum = p.cost_breakdown.reduce((a, c) => a + c.amount_inr, 0);
    expect(sum).toBeCloseTo(p.est_cost_inr, 1);
  });

  it('layers a 90 mm pothole: 40 mm CRM + 50 mm mix', () => {
    const p = planRepair({ ...WORKED, depth_p95_mm: 90, depth_max_mm: 95 });
    expect([p.layered, p.d_design_mm, p.d_crm_mm, p.d_patch_mm]).toEqual([true, 90, 40, 50]);
    expect(p.crm_t).toBeCloseTo(0.4875 * 0.04 * 2.1, 3); // TS rounds tonnes to 3 dp
  });

  it('water-filled: pending re-measure, lower-bound class and a bag range, never exact', () => {
    const p = planRepair({ ...WORKED, water_filled: true, depth_p95_mm: null, depth_max_mm: null, depth_sigma_mm: null,
      volume_l: null, width_m: 0.6, length_m: 0.7, area_m2: 0.35 });
    expect(p.pending_remeasure).toBe(true);
    expect(p.d_design_mm).toBe(50);
    expect(p.bags_low).toBeLessThanOrEqual(p.bags);
    expect(p.bags).toBeLessThanOrEqual(p.bags_high);
  });

  it('IRC severity and water lower bound mirror configs/irc_rules.yaml', () => {
    expect(ircSeverity(20, 0.15, false)).toBe('small');
    expect(ircSeverity(30, 0.15, false)).toBe('medium');
    expect(ircSeverity(20, 0.9, false)).toBe('large');
    expect(ircSeverity(80, 0.9, true)).toBe('unknown_water');
    expect(ircSeverity(null, 0.3, false)).toBe('medium'); // unknown depth -> width-only lower bound
    expect(waterLowerBound(600)).toBe('large');
    expect(waterLowerBound(250)).toBe('medium');
    expect(waterLowerBound(100, 2)).toBe('large');
  });
});

describe('priority', () => {
  it('default weights sum to 1 and bad weights are flagged', () => {
    expect(weightsSumOk(DEFAULT_WEIGHTS)).toBe(true);
    expect(weightsSumOk({ ...DEFAULT_WEIGHTS, S: 0.5 })).toBe(false);
  });
  it('parses a breakdown into one row per component', () => {
    const rows = parseBreakdown({ S: 0.8, E: 0.5, V: 0.2, T: 1, M: 0 });
    expect(rows.map((r) => r.key)).toEqual(['S', 'E', 'V', 'T', 'M']);
  });
});

describe('state machines (server/app/schemas/states.py)', () => {
  it('rejects illegal work-order moves and allows the happy path', () => {
    const path = ['planned', 'issued', 'acknowledged', 'in_progress', 'repaired_claimed', 'verification_pending', 'verified_closed'] as const;
    for (let i = 0; i + 1 < path.length; i++) expect(workOrderCan(path[i], path[i + 1])).toBe(true);
    expect(workOrderCan('planned', 'verified_closed')).toBe(false);
    expect(workOrderCan('qc_failed', 'in_progress')).toBe(false); // qc_failed is terminal: a rework WO is created
    expect(INVOICE_ELIGIBLE).toEqual(['verified_closed']);
  });
  it('rejects illegal pothole moves', () => {
    expect(potholeCan('candidate', 'confirmed')).toBe(true);
    expect(potholeCan('candidate', 'verified_closed')).toBe(false);
    expect(potholeCan('verified_closed', 'reopened')).toBe(true);
  });
  it('a contractor cannot verify its own repair', () => {
    expect(allowedWorkOrderActions('verification_pending', 'contractor')).not.toContain('verified_closed');
    expect(allowedWorkOrderActions('verification_pending', null)).toEqual([]);
  });
});

describe('mock API', () => {
  const tok = { v: null as string | null };
  const api = new MockApi({ latencyMs: 0, now: Date.UTC(2026, 10, 13, 6), seed: 1, getToken: () => tok.v });

  it('lists seeded potholes with severity and priority', async () => {
    const page = await api.listPotholes({ limit: 20 });
    expect(page.total).toBeGreaterThan(0);
    expect(page.items.length).toBeGreaterThan(0);
    for (const p of page.items) expect(p.priority ?? 0).toBeGreaterThanOrEqual(0);
  });

  it('admin creates a work order and moves it one legal step; an illegal move is refused', async () => {
    tok.v = (await api.login({ username: 'admin', password: 'demo1234' })).access_token;
    const page = await api.listPotholes({ status: ['confirmed'], limit: 5 } as never);
    const target = page.items.find((p) => p.status === 'confirmed') ?? page.items[0];
    const wo = await api.createWorkOrder({ pothole_ids: [target.id] });
    // auto-issued when the CRMP contract names a contractor, else planned
    expect(['planned', 'issued']).toContain(wo.status);
    const next = wo.status === 'planned' ? 'issued' : 'acknowledged';
    const moved = await api.transitionWorkOrder(wo.id, { to: next });
    expect(moved.status).toBe(next);
    await expect(api.transitionWorkOrder(wo.id, { to: 'verified_closed' })).rejects.toBeTruthy();
  });
});

describe('offline outbox', () => {
  it('keeps items while offline and drains them in order when back online', async () => {
    const db = new FieldDb(`test-${Math.random()}`);
    await db.workOrders.put({ id: 7, status: 'acknowledged' } as never);
    let t = 1_000_000;
    const q = new OfflineQueue(db, { now: () => t, backoffBaseMs: 1000 });
    await q.enqueueTransition(7, { to: 'in_progress' });
    await q.enqueuePhoto(7, { kind: 'before', blob: new Blob([new Uint8Array(10)], { type: 'image/jpeg' }) });

    const sent: string[] = [];
    let online = false;
    const api: SyncApi = {
      transitionWorkOrder: async (_id, body) => {
        if (!online) throw new NetworkError('offline');
        sent.push(`t:${body.to}`);
        return {} as never;
      },
      uploadWorkOrderPhoto: async (_id, p) => {
        if (!online) throw new NetworkError('offline');
        sent.push(`p:${p.kind}`);
        return {};
      },
    };

    const off = await q.flush(api);
    expect(off.offline).toBe(true);
    expect(sent).toEqual([]);
    expect((await q.counts()).pending).toBe(2);

    online = true;
    t += 60_000; // past the back-off
    const on = await q.flush(api);
    expect(on.sent).toBe(2);
    expect(sent).toEqual(['t:in_progress', 'p:before']);
    expect((await q.counts()).pending).toBe(0);
  });

  it('a 4xx rejection marks the item failed instead of retrying forever', async () => {
    const db = new FieldDb(`test-${Math.random()}`);
    await db.workOrders.put({ id: 9, status: 'issued' } as never);
    const q = new OfflineQueue(db);
    await q.enqueueTransition(9, { to: 'verified_closed' });
    const api: SyncApi = {
      transitionWorkOrder: async () => { throw new ApiError(409, 'illegal transition'); },
      uploadWorkOrderPhoto: async () => ({}),
    };
    const r = await q.flush(api);
    expect(r.failed).toBe(1);
    expect((await q.counts()).failed).toBe(1);
  });
});
