/**
 * Typed API client for GunthaMeter /v1 (docs/CONTRACTS.md, server/app/schemas/api.py).
 *
 * Two implementations behind one interface:
 *   HttpApi  - real backend (JWT bearer, JSON + multipart)
 *   MockApi  - in-browser seeded Hyderabad data (src/lib/mock/), used when VITE_USE_MOCK=1 or the API is unreachable.
 */
import type {
  AdminUnitBrief, AdminUnitKind, AuditEntry, CitizenReportIn, CitizenReportOut, ContractorBrief, ContractorScorecard,
  Health, JurisdictionItem, JurisdictionResolve, KpiOut, PhotoUpload, PlanContext, PotholeDetail, PotholePage,
  PotholeQuery, PriorityConfig, PublicWardsResponse, PublicWardsRow, RateCard, RejectRequest, RepairPlan, RepairRunOut,
  SegmentRating, TokenRequest, TokenResponse, WardStat, WorkOrderCreate, WorkOrderOut, WorkOrderQuery,
  WorkOrderTransition, CalibCert,
} from './types';

/** Server caps (server/app/routers): GET /potholes limit <= 1000. Larger client limits are paged. */
export const POTHOLE_PAGE_MAX = 1000;

export type ApiMode = 'http' | 'mock';
export type ExportKind = 'measurement-book.csv' | 'potholes.geojson';
export interface GeoJsonFeatureCollection { type: 'FeatureCollection'; features: { type: 'Feature'; geometry: unknown; properties?: Record<string, unknown> | null }[] }

/** HTTP error with the server's `detail` (FastAPI) parsed out. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;
  constructor(status: number, detail: unknown, message?: string) {
    super(message ?? `HTTP ${status}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
  /** Retrying later may succeed (server busy / timeouts / rate limited). */
  get retryable(): boolean {
    return this.status >= 500 || this.status === 408 || this.status === 429;
  }
}

/** The request never got an HTTP response (offline, DNS, CORS, timeout). Always retryable. */
export class NetworkError extends Error {
  readonly cause?: unknown;
  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = 'NetworkError';
    this.cause = cause;
  }
}

export function isRetryable(e: unknown): boolean {
  return e instanceof NetworkError || (e instanceof ApiError && e.retryable);
}

export interface Api {
  readonly mode: ApiMode;
  health(): Promise<Health>;
  // auth
  requestOtp(phone: string): Promise<void>;
  login(req: TokenRequest): Promise<TokenResponse>;
  // potholes
  listPotholes(q?: PotholeQuery): Promise<PotholePage>;
  getPothole(id: number): Promise<PotholeDetail>;
  rejectPothole(id: number, body: RejectRequest): Promise<unknown>;
  planPothole(id: number, context?: PlanContext | null): Promise<RepairPlan>;
  segmentRatings(bbox: [number, number, number, number]): Promise<SegmentRating[]>;
  // work orders / runs
  listWorkOrders(q?: WorkOrderQuery): Promise<WorkOrderOut[]>;
  createWorkOrder(body: WorkOrderCreate): Promise<WorkOrderOut>;
  transitionWorkOrder(id: number, body: WorkOrderTransition): Promise<WorkOrderOut>;
  uploadWorkOrderPhoto(id: number, p: PhotoUpload): Promise<unknown>;
  /** circle: admin-unit id or code (e.g. GHMC-C17), as POST /runs/generate?circle= accepts. */
  generateRun(circle: number | string, date: string): Promise<RepairRunOut>;
  getRun(id: number): Promise<RepairRunOut>;
  indentCsvUrl(id: number): string;
  // reporting
  contractorScorecard(id: number): Promise<ContractorScorecard>;
  listContractors(): Promise<ContractorBrief[]>;
  kpi(adminUnit: string | number, windowDays?: number): Promise<KpiOut>;
  publicWards(): Promise<WardStat[]>;
  /** Zones / circles / wards (GET /public/wards?kind=...; there is no dedicated admin-unit list endpoint). */
  adminUnits(kind: AdminUnitKind): Promise<AdminUnitBrief[]>;
  // admin
  getPriorityConfig(): Promise<PriorityConfig>;
  putPriorityConfig(cfg: PriorityConfig): Promise<PriorityConfig>;
  getRateCard(): Promise<RateCard>;
  putRateCard(card: RateCard): Promise<RateCard>;
  jurisdictionQueue(): Promise<JurisdictionItem[]>;
  resolveJurisdiction(id: number, body: JurisdictionResolve): Promise<unknown>;
  audit(entity?: string, id?: number): Promise<AuditEntry[]>;
  deviceCert(deviceId: string): Promise<CalibCert>;
  /** POST /admin/admin-units with a GeoJSON FeatureCollection (wards/circles/zones with valid_from). */
  uploadAdminUnits(fc: GeoJsonFeatureCollection): Promise<unknown>;
  /** GET /exports/measurement-book.csv | /exports/potholes.geojson (link target). */
  exportUrl(kind: ExportKind): string;
  // citizen
  createCitizenReport(r: CitizenReportIn): Promise<CitizenReportOut>;
  getCitizenReport(ticket: string): Promise<CitizenReportOut>;
}

type QueryValue = string | number | boolean | null | undefined | (string | number)[];

/** `{a: 1, b: undefined, c: [1,2]}` -> `?a=1&c=1,2` (FastAPI comma lists; bbox as minLon,minLat,maxLon,maxLat). */
export function buildQuery(q: Record<string, QueryValue> | undefined): string {
  if (!q) return '';
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(q)) {
    if (v === undefined || v === null || v === '') continue;
    if (Array.isArray(v)) {
      if (v.length === 0) continue;
      p.set(k, v.join(','));
    } else p.set(k, String(v));
  }
  const s = p.toString();
  return s ? `?${s}` : '';
}

// ------------------------------------------------------------------ live-response adapters (pure, tested)
const isObj = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null && !Array.isArray(x);

/** GET /healthz answered by a GunthaMeter API (not an SPA fallback page or a foreign server). */
export function isHealth(x: unknown): x is Health {
  return isObj(x) && (x.status === 'ok' || x.status === 'degraded') && typeof x.db === 'boolean';
}

/** The live admin config endpoints return `{id, agency, version, ..., config|card: {...}}`; the mock returns the bare doc. */
export function unwrapEnvelope<T>(x: unknown, field: 'config' | 'card'): T {
  return (isObj(x) && isObj(x[field]) ? x[field] : x) as T;
}

export function adminUnitFromRow(r: Record<string, unknown>): AdminUnitBrief {
  return {
    id: Number(r.id), code: (r.code as string | null | undefined) ?? null, kind: r.kind as AdminUnitKind,
    name_en: String(r.name_en ?? r.code ?? r.id), name_te: (r.name_te as string | null | undefined) ?? null,
    name_hi: (r.name_hi as string | null | undefined) ?? null, parent_id: (r.parent_id as number | null | undefined) ?? null,
    official: Boolean(r.official),
  };
}

type UnitLink = Pick<AdminUnitBrief, 'id' | 'kind' | 'parent_id' | 'name_en'>;

/** Walk parent links up to the zone (English name) of an admin unit. */
export function zoneOf(unit: UnitLink, byId: Map<number, UnitLink>): string | null {
  let u: UnitLink | undefined = unit;
  const seen = new Set<number>();
  while (u && !seen.has(u.id)) {
    if (u.kind === 'zone') return u.name_en;
    seen.add(u.id);
    u = u.parent_id != null ? byId.get(u.parent_id) : undefined;
  }
  return null;
}

/**
 * Live GET /v1/public/wards -> WardStat[]. Field map: id -> admin_unit_id, open -> open_potholes,
 * open_large -> large_open, repaired_<window>d -> repaired_verified_30d. The zone comes from the row (when the
 * server sends it) or from the parent chain in `parents` (circles + zones). Metrics the endpoint does not compute
 * stay null and render as "-".
 */
export function wardStatsFromPublic(res: PublicWardsResponse | PublicWardsRow[], parents: AdminUnitBrief[] = []): WardStat[] {
  const rows: PublicWardsRow[] = Array.isArray(res) ? res : res.items ?? [];
  const win = Array.isArray(res) ? 30 : res.window_days ?? 30;
  const byId = new Map<number, UnitLink>(parents.map((u) => [u.id, u]));
  for (const r of rows) if (!byId.has(Number(r.id))) byId.set(Number(r.id), adminUnitFromRow(r));
  return rows.map((r) => {
    const u = adminUnitFromRow(r);
    const repaired = r[`repaired_${win}d`] ?? Object.entries(r).find(([k]) => /^repaired_\d+d$/.test(k))?.[1];
    return {
      admin_unit_id: u.id, code: u.code, name_en: u.name_en, name_te: u.name_te, name_hi: u.name_hi, kind: u.kind,
      parent_id: u.parent_id, official: u.official, zone: (r.zone as string | null | undefined) ?? zoneOf(u, byId),
      open_potholes: Number(r.open ?? 0), large_open: Number(r.open_large ?? 0), repaired_verified_30d: Number(repaired ?? 0),
      repair_ratio: r.repair_ratio ?? null, rank: r.rank ?? null,
      median_hours_to_repair: null, sla_met_pct: null, km_scanned: null, road_km: null, potholes_per_km: null, est_cost_inr: null,
    };
  });
}

export interface HttpApiOptions {
  baseUrl?: string; // '' = same origin
  getToken?: () => string | null | undefined;
  onUnauthorized?: () => void;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}

export class HttpApi implements Api {
  readonly mode = 'http' as const;
  private readonly base: string;
  private readonly opts: HttpApiOptions;

  constructor(opts: HttpApiOptions = {}) {
    this.opts = opts;
    this.base = (opts.baseUrl ?? '').replace(/\/+$/, '');
  }

  url(path: string, q?: Record<string, QueryValue>): string {
    return `${this.base}/v1${path}${buildQuery(q)}`;
  }

  async request<T>(method: string, path: string, init: { query?: Record<string, QueryValue>; json?: unknown; form?: FormData; raw?: boolean } = {}): Promise<T> {
    const headers: Record<string, string> = { Accept: 'application/json' };
    const token = this.opts.getToken?.();
    if (token) headers.Authorization = `Bearer ${token}`;
    let body: BodyInit | undefined;
    if (init.json !== undefined) {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(init.json);
    } else if (init.form) body = init.form; // browser sets the multipart boundary
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), this.opts.timeoutMs ?? 20000);
    const f = this.opts.fetchImpl ?? globalThis.fetch.bind(globalThis);
    let res: Response;
    try {
      res = await f(this.url(path, init.query), { method, headers, body, signal: ctrl.signal });
    } catch (e) {
      throw new NetworkError(ctrl.signal.aborted ? `timeout: ${method} ${path}` : `network: ${method} ${path}`, e);
    } finally {
      clearTimeout(timer);
    }
    if (!res.ok) {
      let detail: unknown = res.statusText;
      try {
        const txt = await res.text();
        try {
          const j = JSON.parse(txt) as { detail?: unknown };
          detail = j?.detail ?? j;
        } catch {
          if (txt) detail = txt;
        }
      } catch { /* body unreadable */ }
      if (res.status === 401) this.opts.onUnauthorized?.();
      throw new ApiError(res.status, detail);
    }
    if (res.status === 204) return undefined as T;
    const ct = res.headers.get('content-type') ?? '';
    if (init.raw || !ct.includes('json')) return (await res.text()) as unknown as T;
    return (await res.json()) as T;
  }

  private get<T>(path: string, query?: Record<string, QueryValue>) { return this.request<T>('GET', path, { query }); }

  // ---------------------------------------------------------------- endpoints
  async health(): Promise<Health> {
    let h: unknown;
    try {
      h = await this.get<unknown>('/healthz');
    } catch (e) {
      if (!(e instanceof ApiError && e.status === 404)) throw e;
      // CONTRACTS lists GET /healthz among public endpoints; accept it unprefixed too.
      h = await this.rawGetAbsolute<unknown>(`${this.base}/healthz`);
    }
    // A static host (vite preview, CDN) answers unknown paths with index.html (200): that is not an API.
    if (!isHealth(h)) throw new NetworkError('health: no GunthaMeter API at this address');
    return h;
  }

  private async rawGetAbsolute<T>(url: string): Promise<T> {
    const f = this.opts.fetchImpl ?? globalThis.fetch.bind(globalThis);
    let res: Response;
    try { res = await f(url, { headers: { Accept: 'application/json' } }); } catch (e) { throw new NetworkError(`network: GET ${url}`, e); }
    if (!res.ok) throw new ApiError(res.status, res.statusText);
    try { return (await res.json()) as T; } catch (e) { throw new NetworkError(`not JSON: GET ${url}`, e); }
  }

  async requestOtp(phone: string): Promise<void> { await this.request('POST', '/auth/otp', { json: { phone } }); }
  login(req: TokenRequest) { return this.request<TokenResponse>('POST', '/auth/token', { json: req }); }

  /** Limits above the server cap (POTHOLE_PAGE_MAX) are fetched page by page and merged. */
  async listPotholes(q: PotholeQuery = {}): Promise<PotholePage> {
    const base = {
      bbox: q.bbox, status: q.status as QueryValue, severity: q.severity as QueryValue, admin_unit: q.admin_unit,
      agency: q.agency, min_priority: q.min_priority,
    };
    const want = q.limit;
    const offset0 = q.offset ?? 0;
    if (want == null || want <= POTHOLE_PAGE_MAX) return this.get<PotholePage>('/potholes', { ...base, limit: want, offset: q.offset });
    const items: PotholePage['items'] = [];
    let total = 0;
    for (let off = offset0; items.length < want;) {
      const n = Math.min(POTHOLE_PAGE_MAX, want - items.length);
      const page = await this.get<PotholePage>('/potholes', { ...base, limit: n, offset: off });
      total = page.total;
      items.push(...page.items);
      off += page.items.length;
      if (page.items.length < n || off >= total) break;
    }
    return { items, total, limit: want, offset: offset0 };
  }
  getPothole(id: number) { return this.get<PotholeDetail>(`/potholes/${id}`); }
  rejectPothole(id: number, body: RejectRequest) { return this.request<unknown>('POST', `/potholes/${id}/reject`, { json: body }); }
  planPothole(id: number, context?: PlanContext | null) {
    return this.request<RepairPlan>('POST', `/potholes/${id}/plan`, { json: { context: context ?? null } });
  }
  segmentRatings(bbox: [number, number, number, number]) { return this.get<SegmentRating[]>('/segments/rating', { bbox }); }

  listWorkOrders(q: WorkOrderQuery = {}) {
    return this.get<WorkOrderOut[]>('/work-orders', { status: q.status as QueryValue, contractor: q.contractor, agency: q.agency });
  }
  createWorkOrder(body: WorkOrderCreate) { return this.request<WorkOrderOut>('POST', '/work-orders', { json: body }); }
  transitionWorkOrder(id: number, body: WorkOrderTransition) {
    return this.request<WorkOrderOut>('PATCH', `/work-orders/${id}/transition`, { json: body });
  }
  uploadWorkOrderPhoto(id: number, p: PhotoUpload) {
    const fd = new FormData();
    fd.set('kind', p.kind);
    fd.set('file', p.photo, p.name ?? `${p.kind}.jpg`);
    if (p.lat != null) fd.set('lat', String(p.lat));
    if (p.lon != null) fd.set('lon', String(p.lon));
    if (p.at) fd.set('at', p.at);
    return this.request<unknown>('POST', `/work-orders/${id}/photos`, { form: fd });
  }
  generateRun(circle: number | string, date: string) {
    return this.request<RepairRunOut>('POST', '/runs/generate', { query: { circle, date } });
  }
  getRun(id: number) { return this.get<RepairRunOut>(`/runs/${id}`); }
  indentCsvUrl(id: number) { return this.url(`/runs/${id}/indent.csv`); }

  contractorScorecard(id: number) { return this.get<ContractorScorecard>(`/contractors/${id}/scorecard`); }
  async listContractors(): Promise<ContractorBrief[]> {
    // Not in the frozen contract; tolerate its absence (contract change request filed).
    try { return await this.get<ContractorBrief[]>('/contractors'); } catch (e) {
      if (e instanceof ApiError && (e.status === 404 || e.status === 405)) return [];
      throw e;
    }
  }
  kpi(adminUnit: string | number, windowDays = 30) {
    return this.get<KpiOut>(`/kpi/${encodeURIComponent(String(adminUnit))}`, { window: windowDays });
  }
  async publicWards(): Promise<WardStat[]> {
    // Default kind (ward if any exist, else circle) plus the circle and zone lists for the zone column.
    const [res, circles, zones] = await Promise.all([
      this.get<PublicWardsResponse | WardStat[]>('/public/wards'),
      this.adminUnits('circle').catch(() => [] as AdminUnitBrief[]),
      this.adminUnits('zone').catch(() => [] as AdminUnitBrief[]),
    ]);
    // Already in the UI shape (mock-compatible server): pass through.
    if (Array.isArray(res) && (res.length === 0 || 'admin_unit_id' in res[0])) return res as WardStat[];
    return wardStatsFromPublic(res as PublicWardsResponse | PublicWardsRow[], [...circles, ...zones]);
  }
  async adminUnits(kind: AdminUnitKind): Promise<AdminUnitBrief[]> {
    const res = await this.get<PublicWardsResponse>('/public/wards', { kind });
    return (res.items ?? []).filter((r) => r.kind === kind).map(adminUnitFromRow);
  }

  async getPriorityConfig() { return unwrapEnvelope<PriorityConfig>(await this.get<unknown>('/admin/priority-config'), 'config'); }
  async putPriorityConfig(cfg: PriorityConfig) {
    return unwrapEnvelope<PriorityConfig>(await this.request<unknown>('PUT', '/admin/priority-config', { json: cfg }), 'config');
  }
  async getRateCard() { return unwrapEnvelope<RateCard>(await this.get<unknown>('/admin/rate-card'), 'card'); }
  async putRateCard(card: RateCard) {
    return unwrapEnvelope<RateCard>(await this.request<unknown>('PUT', '/admin/rate-card', { json: card }), 'card');
  }
  jurisdictionQueue() { return this.get<JurisdictionItem[]>('/admin/jurisdiction-queue'); }
  resolveJurisdiction(id: number, body: JurisdictionResolve) {
    return this.request<unknown>('POST', `/admin/jurisdiction-queue/${id}/resolve`, { json: body });
  }
  audit(entity?: string, id?: number) { return this.get<AuditEntry[]>('/admin/audit', { entity, id }); }
  deviceCert(deviceId: string) { return this.get<CalibCert>(`/devices/${encodeURIComponent(deviceId)}/cert`); }
  uploadAdminUnits(fc: GeoJsonFeatureCollection) { return this.request<unknown>('POST', '/admin/admin-units', { json: fc }); }
  exportUrl(kind: ExportKind) { return this.url(`/exports/${kind}`); }

  createCitizenReport(r: CitizenReportIn) {
    const fd = new FormData();
    fd.set('lat', String(r.lat));
    fd.set('lon', String(r.lon));
    if (r.text) fd.set('text', r.text);
    if (r.lang) fd.set('lang', r.lang);
    if (r.phone) fd.set('phone', r.phone);
    if (r.photo) fd.set('photo', r.photo, r.photo_name ?? 'report.jpg');
    return this.request<CitizenReportOut>('POST', '/citizen/reports', { form: fd });
  }
  getCitizenReport(ticket: string) { return this.get<CitizenReportOut>(`/citizen/reports/${encodeURIComponent(ticket)}`); }
}

// ------------------------------------------------------------------ mode resolution
export const MODE_OVERRIDE_KEY = 'gm.apiMode';

export interface ResolveOptions {
  envUseMock?: string | undefined; // import.meta.env.VITE_USE_MOCK
  override?: string | null; // localStorage gm.apiMode
  probe: () => Promise<unknown>; // resolves when the API answers
  probeTimeoutMs?: number;
}

/**
 * Order: explicit localStorage override > VITE_USE_MOCK (1 -> mock, 0 -> http) > health probe
 * (reachable -> http, otherwise mock).
 */
export async function resolveApiMode(o: ResolveOptions): Promise<{ mode: ApiMode; reason: string }> {
  if (o.override === 'mock' || o.override === 'http') return { mode: o.override, reason: 'override' };
  if (o.envUseMock === '1' || o.envUseMock === 'true') return { mode: 'mock', reason: 'env' };
  if (o.envUseMock === '0' || o.envUseMock === 'false') return { mode: 'http', reason: 'env' };
  const timeout = new Promise<never>((_, rej) => setTimeout(() => rej(new NetworkError('probe timeout')), o.probeTimeoutMs ?? 1500));
  try {
    await Promise.race([o.probe(), timeout]);
    return { mode: 'http', reason: 'probe-ok' };
  } catch {
    return { mode: 'mock', reason: 'probe-failed' };
  }
}
