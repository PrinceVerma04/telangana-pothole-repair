/**
 * HTTP client vs the live server (WP9 leftovers, checked against /v1/openapi.json of the API on a copy of the demo
 * DB on 2026-11-13): response adapters, the /potholes page cap, the health guard, admin units for the circle and
 * zone pickers, and the mock's parity with the server (circle codes, offline writes, thumbnail size).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  HttpApi, NetworkError, POTHOLE_PAGE_MAX, isHealth, unwrapEnvelope, wardStatsFromPublic, zoneOf,
} from './api';
import { CIRCLES, circleName, sortUnits, unitName } from './circles';
import { MockApi } from './mock/mockApi';
import type { AdminUnitBrief, PublicWardsResponse } from './types';

type Call = { method: string; url: URL; body?: unknown };

/** fetch stub: routes by pathname; records calls. */
function stubFetch(routes: Record<string, (u: URL, init: RequestInit) => unknown>, calls: Call[]) {
  return (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(String(input), 'http://api.test');
    calls.push({ method: init.method ?? 'GET', url, body: init.body });
    const h = routes[url.pathname];
    if (!h) return new Response(JSON.stringify({ detail: 'Not Found' }), { status: 404, headers: { 'content-type': 'application/json' } });
    const out = h(url, init);
    if (out instanceof Response) return out;
    return new Response(JSON.stringify(out), { status: 200, headers: { 'content-type': 'application/json' } });
  }) as typeof fetch;
}

// Trimmed copies of real responses from the live API (app_data/gunthameter.db seed).
const ZONES = {
  kind: 'zone', window_days: 30, generated_at: '2026-11-13T17:58:07Z', items: [
    { id: 3, code: 'GHMC-KHT', kind: 'zone', name_en: 'Khairatabad', name_te: 'ఖైరతాబాద్', name_hi: 'खैरताबाद', parent_id: null, official: false, open: 0, open_large: 0, repaired_30d: 0, repair_ratio: null, rank: 1 },
    { id: 1, code: 'GHMC-LBN', kind: 'zone', name_en: 'LB Nagar', name_te: 'ఎల్బీ నగర్', name_hi: 'एलबी नगर', parent_id: null, official: false, open: 0, open_large: 0, repaired_30d: 0, repair_ratio: null, rank: 2 },
  ],
} satisfies PublicWardsResponse;
const CIRCLE_ROWS = {
  kind: 'circle', window_days: 30, generated_at: '2026-11-13T17:58:07Z', items: [
    { id: 26, code: 'GHMC-C17', kind: 'circle', name_en: 'Khairatabad', name_te: 'ఖైరతాబాద్', name_hi: 'खैरताबाद', parent_id: 3, official: false, open: 4, open_large: 1, repaired_30d: 2, repair_ratio: 0.333, rank: 1 },
    { id: 15, code: 'GHMC-C04', kind: 'circle', name_en: 'LB Nagar', name_te: 'ఎల్బీ నగర్', name_hi: 'एलबी नगर', parent_id: 1, official: false, open: 0, open_large: 0, repaired_30d: 0, repair_ratio: null, rank: 2 },
  ],
} satisfies PublicWardsResponse;
const WARD_ROWS = {
  kind: 'ward', window_days: 30, generated_at: '2026-11-13T17:58:07Z', items: [
    { id: 152, code: 'GHMC-W085', kind: 'ward', name_en: 'Adikmet', name_te: null, name_hi: null, parent_id: 26, official: false, open: 3, open_large: 1, repaired_30d: 2, repair_ratio: 0.4, rank: 1 },
    { id: 999, code: 'X-W1', kind: 'ward', name_en: 'Orphan', name_te: null, name_hi: null, parent_id: null, official: false, open: 0, open_large: 0, repaired_30d: 0, repair_ratio: null, rank: 2 },
  ],
} satisfies PublicWardsResponse;

function wardsRoute(u: URL) {
  const kind = u.searchParams.get('kind');
  return kind === 'zone' ? ZONES : kind === 'circle' ? CIRCLE_ROWS : WARD_ROWS;
}

describe('live response adapters', () => {
  it('maps /public/wards rows to the UI WardStat shape, zone via the parent chain', () => {
    const units: AdminUnitBrief[] = [...ZONES.items, ...CIRCLE_ROWS.items].map((r) => ({ ...r }));
    const ws = wardStatsFromPublic(WARD_ROWS, units);
    expect(ws).toHaveLength(2);
    expect(ws[0]).toMatchObject({
      admin_unit_id: 152, code: 'GHMC-W085', name_en: 'Adikmet', zone: 'Khairatabad', open_potholes: 3, large_open: 1,
      repaired_verified_30d: 2, repair_ratio: 0.4, rank: 1, potholes_per_km: null, sla_met_pct: null,
    });
    expect(ws[1].zone).toBeNull();
  });

  it('reads repaired_<window>d for other windows and prefers a zone sent by the server', () => {
    const res = { ...WARD_ROWS, window_days: 90, items: [{ ...WARD_ROWS.items[0], repaired_30d: undefined, repaired_90d: 7, zone: 'Secunderabad' }] };
    const [w] = wardStatsFromPublic(res as unknown as PublicWardsResponse);
    expect(w.repaired_verified_30d).toBe(7);
    expect(w.zone).toBe('Secunderabad');
  });

  it('zoneOf survives parent cycles', () => {
    const a: AdminUnitBrief = { id: 1, kind: 'ward', parent_id: 2, name_en: 'a' };
    const b: AdminUnitBrief = { id: 2, kind: 'circle', parent_id: 1, name_en: 'b' };
    expect(zoneOf(a, new Map<number, AdminUnitBrief>([[1, a], [2, b]]))).toBeNull();
  });

  it('unwraps the admin config envelopes and leaves bare documents alone', () => {
    const cfg = { version: 1, weights: { S: 0.3, E: 0.25, V: 0.2, T: 0.15, M: 0.1 } };
    expect(unwrapEnvelope({ id: 2, agency: 'GHMC', inherited: false, config: cfg }, 'config')).toEqual(cfg);
    expect(unwrapEnvelope(cfg, 'config')).toEqual(cfg);
    const card = { agency: 'GHMC', fy: '2026-27', version: 1, currency: 'INR', items: {}, baseline: { cost_per_pothole_inr: 7396 } };
    expect(unwrapEnvelope({ id: 2, version_label: 'GHMC/2026-27/v1', card }, 'card')).toEqual(card);
  });

  it('isHealth rejects an SPA fallback page', () => {
    expect(isHealth({ status: 'ok', db: true, storage: true, demo_mode: true, version: '0.1.0' })).toBe(true);
    expect(isHealth('<!doctype html><html></html>')).toBe(false);
    expect(isHealth({ detail: 'Not Found' })).toBe(false);
  });
});

describe('HttpApi against the live endpoint shapes', () => {
  it('publicWards fetches wards + circles + zones and returns WardStat[]', async () => {
    const calls: Call[] = [];
    const api = new HttpApi({ baseUrl: 'http://api.test', fetchImpl: stubFetch({ '/v1/public/wards': wardsRoute }, calls) });
    const ws = await api.publicWards();
    expect(calls.map((c) => c.url.searchParams.get('kind')).sort()).toEqual(['circle', 'zone', null].sort());
    expect(ws[0]).toMatchObject({ admin_unit_id: 152, zone: 'Khairatabad', open_potholes: 3 });
  });

  it('adminUnits(kind) lists circles with their real ids and codes', async () => {
    const calls: Call[] = [];
    const api = new HttpApi({ baseUrl: 'http://api.test', fetchImpl: stubFetch({ '/v1/public/wards': wardsRoute }, calls) });
    const cs = await api.adminUnits('circle');
    expect(cs.map((c) => [c.id, c.code, c.kind, c.parent_id])).toEqual([[26, 'GHMC-C17', 'circle', 3], [15, 'GHMC-C04', 'circle', 1]]);
    expect(calls[0].url.search).toBe('?kind=circle');
  });

  it('generateRun sends the admin-unit id or code as ?circle=', async () => {
    const calls: Call[] = [];
    const run = { id: 9, circle_id: 26, run_date: '2026-11-13', work_order_ids: [], route: [], totals: {}, crew_hours_cap: 16, generated_at: '2026-11-13T00:00:00Z' };
    const api = new HttpApi({ baseUrl: 'http://api.test', fetchImpl: stubFetch({ '/v1/runs/generate': () => run }, calls) });
    await api.generateRun(26, '2026-11-13');
    await api.generateRun('GHMC-C17', '2026-11-14');
    expect(calls.map((c) => [c.method, c.url.searchParams.get('circle'), c.url.searchParams.get('date')]))
      .toEqual([['POST', '26', '2026-11-13'], ['POST', 'GHMC-C17', '2026-11-14']]);
  });

  it('listPotholes pages limits above the server cap (limit <= 1000) and merges', async () => {
    const calls: Call[] = [];
    const TOTAL = 2300;
    const api = new HttpApi({
      baseUrl: 'http://api.test',
      fetchImpl: stubFetch({
        '/v1/potholes': (u) => {
          const limit = Number(u.searchParams.get('limit'));
          const offset = Number(u.searchParams.get('offset') ?? 0);
          if (limit > POTHOLE_PAGE_MAX) return new Response('{"detail":"le 1000"}', { status: 422, headers: { 'content-type': 'application/json' } });
          const n = Math.max(0, Math.min(limit, TOTAL - offset));
          return { items: Array.from({ length: n }, (_, i) => ({ id: offset + i + 1 })), total: TOTAL, limit, offset };
        },
      }, calls),
    });
    const page = await api.listPotholes({ limit: 2000, status: ['confirmed', 'planned'] });
    expect(page.items).toHaveLength(2000);
    expect(page.total).toBe(TOTAL);
    expect(page.items[1999].id).toBe(2000);
    expect(calls.map((c) => [c.url.searchParams.get('limit'), c.url.searchParams.get('offset')])).toEqual([['1000', '0'], ['1000', '1000']]);
    expect(calls.every((c) => c.url.searchParams.get('status') === 'confirmed,planned')).toBe(true);

    calls.length = 0;
    const small = await api.listPotholes({ limit: 500 });
    expect(small.items).toHaveLength(500);
    expect(calls).toHaveLength(1);

    calls.length = 0;
    const short = await api.listPotholes({ limit: 5000 }); // stops when the server runs out
    expect(short.items).toHaveLength(TOTAL);
    expect(calls).toHaveLength(3);
  });

  it('priority config and rate card are unwrapped from the live envelope', async () => {
    const cfg = { version: 1, weights: { S: 0.3, E: 0.25, V: 0.2, T: 0.15, M: 0.1 } };
    const card = { agency: 'GHMC', fy: '2026-27', version: 1, currency: 'INR', items: {}, baseline: { cost_per_pothole_inr: 7396 } };
    const api = new HttpApi({
      baseUrl: 'http://api.test',
      fetchImpl: stubFetch({
        '/v1/admin/priority-config': (_u, init) => ({ id: 3, agency: 'GHMC', config: init.body ? JSON.parse(String(init.body)) : cfg }),
        '/v1/admin/rate-card': () => ({ id: 2, agency: 'GHMC', version_label: 'GHMC/2026-27/v1', card }),
      }, []),
    });
    expect((await api.getPriorityConfig()).weights.S).toBe(0.3);
    expect((await api.putPriorityConfig({ ...cfg, version: 2 })).version).toBe(2);
    expect((await api.getRateCard()).baseline.cost_per_pothole_inr).toBe(7396);
  });

  it('health() refuses a static host that answers every path with index.html', async () => {
    const html = () => new Response('<!doctype html><title>RoadFix</title>', { status: 200, headers: { 'content-type': 'text/html' } });
    const api = new HttpApi({ baseUrl: 'http://api.test', fetchImpl: stubFetch({ '/v1/healthz': html, '/healthz': html }, []) });
    await expect(api.health()).rejects.toBeInstanceOf(NetworkError);
    const ok = new HttpApi({
      baseUrl: 'http://api.test',
      fetchImpl: stubFetch({ '/healthz': () => ({ status: 'ok', db: true, storage: true, demo_mode: true, version: '0.1.0' }) }, []),
    });
    expect((await ok.health()).status).toBe('ok'); // /v1/healthz 404 -> /healthz
  });
});

describe('mock parity with the live server', () => {
  it('adminUnits: zones and circles carry the live codes', async () => {
    const api = new MockApi({ latencyMs: 0 });
    const zones = await api.adminUnits('zone');
    expect(zones.map((z) => z.code).sort()).toEqual(['GHMC-KHT', 'GHMC-LBN']);
    const circles = await api.adminUnits('circle');
    expect(circles.map((c) => c.code)).toEqual(CIRCLES.map((c) => c.code));
    expect(circles.every((c) => zones.some((z) => z.id === c.parent_id))).toBe(true);
  });

  it('kpi and potholes accept zone and circle codes like the server', async () => {
    const api = new MockApi({ latencyMs: 0 });
    const byName = await api.kpi('Khairatabad');
    const byCode = await api.kpi('GHMC-KHT');
    expect(byCode.open_potholes).toBe(byName.open_potholes);
    const all = await api.listPotholes({ limit: 1000 });
    const kht = await api.listPotholes({ admin_unit: 'GHMC-KHT', limit: 1000 });
    const c17 = await api.listPotholes({ admin_unit: 'GHMC-C17', limit: 1000 });
    expect(kht.total).toBeGreaterThan(0);
    expect(kht.total).toBeLessThan(all.total);
    expect(c17.total).toBeGreaterThan(0);
    expect(c17.total).toBeLessThanOrEqual(kht.total);
  });

  it('generateRun accepts a circle code and rejects unknown circles with 404', async () => {
    const api = new MockApi({ latencyMs: 0, getToken: () => 'mock.zonal_admin.zonal_khairatabad.-' });
    const run = await api.generateRun('GHMC-C17', '2026-11-13');
    expect(run.circle_id).toBe(1);
    await expect(api.generateRun('NOPE-1', '2026-11-13')).rejects.toMatchObject({ status: 404 });
  });

  describe('offline-aware writes', () => {
    afterEach(() => vi.unstubAllGlobals());
    it('outbox writes fail with NetworkError while navigator.onLine is false, reads still work', async () => {
      const api = new MockApi({ latencyMs: 0, getToken: () => 'mock.contractor.contractor1.1' });
      const wo = (await api.listWorkOrders({ status: 'issued' }))[0];
      expect(wo).toBeTruthy();
      vi.stubGlobal('navigator', { onLine: false });
      await expect(api.transitionWorkOrder(wo.id, { to: 'acknowledged' })).rejects.toBeInstanceOf(NetworkError);
      await expect(api.uploadWorkOrderPhoto(wo.id, { kind: 'before', photo: new Blob(['x'], { type: 'image/jpeg' }) })).rejects.toBeInstanceOf(NetworkError);
      expect((await api.listWorkOrders()).length).toBeGreaterThan(0);
      vi.stubGlobal('navigator', { onLine: true });
      expect((await api.transitionWorkOrder(wo.id, { to: 'acknowledged' })).status).toBe('acknowledged');
    });
  });

  it('list thumbnails stay <= 20 KB (PLAN WP9)', async () => {
    const api = new MockApi({ latencyMs: 0 });
    const page = await api.listPotholes({ limit: 1000 });
    const sizes = page.items.map((p) => (p.thumb_url ? new TextEncoder().encode(p.thumb_url).length : 0));
    expect(sizes.filter((n) => n > 0).length).toBeGreaterThan(0);
    expect(Math.max(...sizes)).toBeLessThanOrEqual(20 * 1024);
  });
});

describe('circle labels', () => {
  const units: AdminUnitBrief[] = CIRCLE_ROWS.items.map((r) => ({ ...r }));
  it('uses the API list first, then the run name, then the mock list', () => {
    expect(circleName(26, 'te', 'Khairatabad', units)).toBe('ఖైరతాబాద్');
    expect(circleName(26, 'en', null, units)).toBe('Khairatabad');
    expect(circleName(41, 'en', 'Alwal', units)).toBe('Alwal');
    expect(circleName(2, 'hi')).toBe('मेहदीपट्टनम');
    expect(circleName(77, 'en')).toBe('#77');
  });
  it('falls back to English when a translation is missing and sorts by the shown name', () => {
    expect(unitName({ name_en: 'Adikmet', name_te: null, name_hi: null }, 'te')).toBe('Adikmet');
    expect(sortUnits(units, 'en').map((u) => u.code)).toEqual(['GHMC-C17', 'GHMC-C04']);
  });
});
