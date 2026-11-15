/**
 * In-browser mock of the /v1 API. Same `Api` interface as HttpApi; state lives in memory (per tab) and is
 * mutated by work-order transitions, run generation, citizen reports and admin edits so the demo flows work.
 * Enforces the frozen state machines and RBAC (src/lib/states.ts mirrors server/app/schemas/states.py).
 */
import { ApiError, NetworkError, type Api, type ExportKind, type GeoJsonFeatureCollection } from '../api';
import { DEMO_USERS } from '../demoUsers';
import { OPEN_STATUSES, WO_TO_POTHOLE, WO_TRANSITION_ROLES, workOrderCan } from '../states';
import { planRepair } from '../planner';
import type {
  AdminUnitBrief, AdminUnitKind, AuditEntry, CalibCert, CitizenReportIn, CitizenReportOut, ContractorBrief, ContractorScorecard, Health,
  JurisdictionItem, JurisdictionResolve, KpiOut, PhotoUpload, PlanContext, PotholeDetail, PotholeFused,
  PotholeListItem, PotholePage, PotholeQuery, PotholeStatus, PriorityConfig, RateCard, RejectRequest, RepairPlan,
  RepairRunOut, Role, SegmentRating, TokenRequest, TokenResponse, WardStat, WorkOrderCreate, WorkOrderOut,
  WorkOrderQuery, WorkOrderTransition,
} from '../types';
import { audit, buildRun, CIRCLES, createSeed, inMonsoonWindow, mockAdminUnits, slaHours, wardAliases, type MockDb } from './seed';
import { haversineM, inBbox, lineLengthM, median, pointAlong, sliceLine } from './geo';
import { thumbSvg } from './svg';

const HOUR = 3600_000;
const DAY = 24 * HOUR;
const iso = (t: number) => new Date(t).toISOString();
const clone = <T,>(x: T): T => structuredClone(x);
const asArr = <T,>(v: T | T[] | undefined): T[] | undefined => (v === undefined ? undefined : Array.isArray(v) ? v : [v]);

export interface MockApiOptions {
  latencyMs?: number;
  now?: number;
  seed?: number;
  getToken?: () => string | null | undefined;
  /**
   * Behave like a remote API when the browser is offline: writes that the field outbox sends fail with a
   * NetworkError while `navigator.onLine === false` (default true). Reads keep working (offline demo).
   */
  offlineAware?: boolean;
}

/** Mock token format: `mock.<role>.<username>.<contractor_id|->` (not a JWT; never sent anywhere). */
export function parseMockToken(tok: string | null | undefined): { role: Role; username: string; contractor_id: number | null } | null {
  if (!tok || !tok.startsWith('mock.')) return null;
  const [, role, username, cid] = tok.split('.');
  return { role: role as Role, username, contractor_id: cid && cid !== '-' ? Number(cid) : null };
}

export class MockApi implements Api {
  readonly mode = 'mock' as const;
  readonly db: MockDb;
  private readonly latency: number;
  private readonly getToken?: () => string | null | undefined;
  private readonly offlineAware: boolean;

  constructor(opts: MockApiOptions = {}) {
    this.db = createSeed({ now: opts.now, seed: opts.seed });
    this.latency = opts.latencyMs ?? 120;
    this.getToken = opts.getToken;
    this.offlineAware = opts.offlineAware ?? true;
  }

  /** Simulated network for outbox writes (see MockApiOptions.offlineAware). */
  private network(what: string) {
    if (this.offlineAware && typeof navigator !== 'undefined' && navigator.onLine === false) {
      throw new NetworkError(`network: ${what} (offline)`);
    }
  }

  private async delay<T>(v: T): Promise<T> {
    if (this.latency > 0) await new Promise((r) => setTimeout(r, this.latency));
    return clone(v);
  }

  private session() { return parseMockToken(this.getToken?.()); }

  private requireRole(...roles: Role[]) {
    const s = this.session();
    if (!s) throw new ApiError(401, 'not authenticated');
    if (s.role !== 'super_admin' && !roles.includes(s.role)) throw new ApiError(403, `role ${s.role} not allowed`);
    return s;
  }

  private pothole(id: number): PotholeFused {
    const p = this.db.potholes.find((x) => x.id === id);
    if (!p) throw new ApiError(404, `pothole ${id} not found`);
    return p;
  }

  private setPotholeStatus(p: PotholeFused, to: PotholeStatus, reason: string, actor: string) {
    if (p.status === to) return;
    const ev = { ts: iso(Date.now()), from_status: p.status, to_status: to, reason, actor };
    (this.db.events.get(p.id!) ?? this.db.events.set(p.id!, []).get(p.id!)!).push(ev);
    audit(this.db, 'pothole', p.id!, 'transition', actor, { status: p.status }, { status: to }, reason, Date.now());
    p.status = to;
  }

  listItem(p: PotholeFused): PotholeListItem {
    const m = this.db.meta.get(p.id!)!;
    const ward = this.db.wards.find((w) => w.id === m.ward_id);
    return {
      id: p.id!, lat: p.lat!, lon: p.lon!, status: p.status, severity: p.severity ?? null, priority: p.priority ?? null,
      depth_p95_mm: p.depth_p95_mm ?? null, depth_sigma_mm: p.depth_sigma_mm ?? null, area_m2: p.area_m2, volume_l: p.volume_l ?? null,
      water_filled: p.water_filled, road_name: m.road_name, highway: m.highway, agency_code: m.agency_code, admin_unit: ward?.name_en ?? null,
      passes_seen: p.passes_seen, passes_total: p.passes_total, last_seen: p.last_seen ?? null, thumb_url: m.thumb_url,
      best_method: p.best_method ?? null, certified_measurement: p.certified_measurement,
    };
  }

  // ---------------------------------------------------------------- endpoints
  health(): Promise<Health> {
    return this.delay({ status: 'ok', db: true, storage: true, gpu: null, seg_weights: 'mock', depth_backend: 'mock', demo_mode: true, version: 'mock-0.1' });
  }

  async requestOtp(phone: string): Promise<void> {
    if (!/^\+?91?[6-9]\d{9}$/.test(phone)) throw new ApiError(422, 'Enter a valid Indian mobile number');
    await this.delay(null);
  }

  async login(req: TokenRequest): Promise<TokenResponse> {
    const exp = iso(Date.now() + 12 * HOUR);
    if (req.phone) {
      if (req.otp !== '123456') throw new ApiError(401, 'invalid OTP (demo OTP is 123456)');
      return this.delay({ access_token: `mock.citizen.${req.phone.replace(/\D/g, '')}.-`, token_type: 'bearer', role: 'citizen', agency_id: null, contractor_id: null, expires_at: exp });
    }
    const u = DEMO_USERS.find((d) => d.username === req.username);
    if (!u || req.password !== u.password) throw new ApiError(401, 'invalid username or password');
    return this.delay({
      access_token: `mock.${u.role}.${u.username}.${u.contractor_id ?? '-'}`, token_type: 'bearer', role: u.role,
      agency_id: u.agency_id ?? 1, contractor_id: u.contractor_id ?? null, expires_at: exp,
    });
  }

  listPotholes(q: PotholeQuery = {}): Promise<PotholePage> {
    const st = asArr(q.status);
    const sv = asArr(q.severity);
    let items = this.db.potholes.filter((p) => {
      const m = this.db.meta.get(p.id!)!;
      if (q.bbox && !inBbox([p.lon!, p.lat!], q.bbox)) return false;
      if (st && !st.includes(p.status)) return false;
      if (sv && (!p.severity || !sv.includes(p.severity))) return false;
      if (q.min_priority != null && (p.priority ?? 0) < q.min_priority) return false;
      if (q.agency && m.agency_code !== q.agency) return false;
      if (q.admin_unit) {
        const w = this.db.wards.find((x) => x.id === m.ward_id)!;
        if (!wardAliases(w).includes(q.admin_unit)) return false;
      }
      return true;
    });
    const s = this.session();
    if (s?.role === 'contractor') items = items.filter((p) => p.contractor_id === s.contractor_id);
    items.sort((a, b) => (b.priority ?? -1) - (a.priority ?? -1));
    const limit = q.limit ?? 500;
    const offset = q.offset ?? 0;
    return this.delay({ items: items.slice(offset, offset + limit).map((p) => this.listItem(p)), total: items.length, limit, offset });
  }

  getPothole(id: number): Promise<PotholeDetail> {
    const p = this.pothole(id);
    const m = this.db.meta.get(id)!;
    return this.delay({
      pothole: p,
      road: { osm_way_id: m.way_id, name: m.road_name, highway: m.highway, lanes: m.lanes, surface: p.surface, owner: m.agency_code },
      measurements: this.db.measurements.get(id) ?? [],
      passes: this.db.passes.get(id) ?? [],
      plan: this.db.plans.get(id) ?? null,
      events: this.db.events.get(id) ?? [],
      work_orders: this.db.workOrders.filter((w) => w.pothole_ids.includes(id)),
      nearby_pois: m.nearby_pois,
    });
  }

  async rejectPothole(id: number, body: RejectRequest): Promise<unknown> {
    const s = this.requireRole('field_engineer', 'zonal_admin');
    const p = this.pothole(id);
    if (['verified_closed', 'rejected', 'archived'].includes(p.status)) throw new ApiError(409, `cannot reject from ${p.status}`);
    if (body.cause) p.defect_cause = body.cause;
    this.setPotholeStatus(p, 'rejected', body.reason, s.username);
    return this.delay({ ok: true, status: p.status });
  }

  async planPothole(id: number, context?: PlanContext | null): Promise<RepairPlan> {
    const p = this.pothole(id);
    const plan = planRepair({ ...p, pothole_id: id }, { in_monsoon_window: inMonsoonWindow(Date.now()), ...(context ?? {}) }, this.db.rateCard);
    this.db.plans.set(id, plan);
    if (p.status === 'confirmed' || p.status === 'reopened') this.setPotholeStatus(p, 'planned', 'repair plan computed', 'planner');
    return this.delay(plan);
  }

  segmentRatings(bbox: [number, number, number, number]): Promise<SegmentRating[]> {
    const out: SegmentRating[] = [];
    const byWay = new Map<number, PotholeFused[]>();
    for (const p of this.db.potholes) {
      if (!OPEN_STATUSES.includes(p.status) && p.status !== 'candidate') continue;
      const w = this.db.meta.get(p.id!)!.way_id;
      (byWay.get(w) ?? byWay.set(w, []).get(w)!).push(p);
    }
    let segId = 1;
    for (const way of this.db.ways) {
      if (!way.coords.some((c) => inBbox(c, bbox))) continue;
      const L = lineLengthM(way.coords);
      const lanes = way.lanes ?? (way.highway === 'trunk' ? 3 : 2);
      const ps = byWay.get(way.way_id) ?? [];
      for (let d0 = 0; d0 < L; d0 += 100) {
        const d1 = Math.min(L, d0 + 100);
        if (d1 - d0 < 20) break;
        const area = ps.filter((p) => {
          const c = this.db.meta.get(p.id!)!.chainage_m;
          return c >= d0 && c < d1;
        }).reduce((s, p) => s + p.area_m2, 0);
        const pct = (100 * area) / ((d1 - d0) * lanes * 3.5);
        out.push({
          road_segment_id: segId++, osm_way_id: way.way_id, bin_start_m: Math.round(d0), bin_end_m: Math.round(d1),
          pothole_area_pct: Math.round(pct * 1000) / 1000, rating: pct > 0.5 ? 'Poor' : pct > 0 ? 'Fair' : 'Good',
          geometry: { type: 'LineString', coordinates: sliceLine(way.coords, d0, d1) },
        });
      }
    }
    return this.delay(out);
  }

  listWorkOrders(q: WorkOrderQuery = {}): Promise<WorkOrderOut[]> {
    const st = asArr(q.status);
    const s = this.session();
    const items = this.db.workOrders.filter((w) => {
      if (st && !st.includes(w.status)) return false;
      if (q.contractor != null && w.contractor_id !== q.contractor) return false;
      if (s?.role === 'contractor' && w.contractor_id !== s.contractor_id) return false; // RBAC row scope
      return true;
    });
    return this.delay([...items].sort((a, b) => b.id - a.id));
  }

  async createWorkOrder(body: WorkOrderCreate): Promise<WorkOrderOut> {
    const s = this.requireRole('zonal_admin', 'field_engineer');
    if (!body.pothole_ids.length) throw new ApiError(422, 'pothole_ids must not be empty');
    const ps = body.pothole_ids.map((id) => this.pothole(id));
    for (const p of ps) {
      if (!['confirmed', 'planned', 'reopened', 'qc_failed'].includes(p.status)) throw new ApiError(409, `pothole ${p.id} is ${p.status}; only confirmed/planned/reopened/qc_failed can be work-ordered`);
    }
    const now = Date.now();
    const plans: RepairPlan[] = [];
    for (const p of ps) {
      let pl = this.db.plans.get(p.id!);
      if (!pl) {
        pl = planRepair({ ...p, pothole_id: p.id! }, { in_monsoon_window: inMonsoonWindow(now) }, this.db.rateCard);
        this.db.plans.set(p.id!, pl);
      }
      plans.push(pl);
    }
    const contractorId = body.contractor_id ?? ps[0].contractor_id ?? null;
    const m = this.db.meta.get(ps[0].id!)!;
    const sla = slaHours(m.highway, m.agency_code, inMonsoonWindow(now), this.db.priorityConfig);
    const status = contractorId ? 'issued' : 'planned';
    const parent = this.db.workOrders.find((w) => w.status === 'qc_failed' && w.pothole_ids.some((id) => body.pothole_ids.includes(id)));
    const wo: WorkOrderOut = {
      id: this.db.seq.wo++, status, pothole_ids: [...body.pothole_ids], agency_id: ps[0].agency_id ?? 1, contract_id: body.contract_id ?? ps[0].contract_id ?? null,
      contractor_id: contractorId, contractor_name: this.db.contractors.find((c) => c.id === contractorId)?.name ?? null,
      crew: body.crew ?? null, issued_at: status === 'issued' ? iso(now) : null, sla_due: body.sla_due ?? iso(now + sla * HOUR),
      sla_breached: false, escalation_level: null, est_bags: plans.reduce((a, b) => a + b.bags, 0),
      est_cost_inr: Math.round(plans.reduce((a, b) => a + b.est_cost_inr, 0)), bags_used: null, bag_variance_flag: false,
      is_rework: !!parent, parent_work_order_id: parent?.id ?? null, photos: [], run_id: body.run_id ?? null,
    };
    this.db.workOrders.push(wo);
    audit(this.db, 'work_order', wo.id, 'create', s.username, null, { status, pothole_ids: wo.pothole_ids }, body.notes ?? null, now);
    for (const p of ps) {
      if (p.status === 'confirmed' || p.status === 'reopened') this.setPotholeStatus(p, 'planned', 'repair plan computed', 'planner');
      if (status === 'issued') this.setPotholeStatus(p, 'work_ordered', `work order #${wo.id} issued${parent ? ' (rework)' : ''}`, s.username);
    }
    return this.delay(wo);
  }

  async transitionWorkOrder(id: number, body: WorkOrderTransition): Promise<WorkOrderOut> {
    this.network(`PATCH /work-orders/${id}/transition`);
    const s = this.session();
    if (!s) throw new ApiError(401, 'not authenticated');
    const wo = this.db.workOrders.find((w) => w.id === id);
    if (!wo) throw new ApiError(404, `work order ${id} not found`);
    if (s.role === 'contractor' && wo.contractor_id !== s.contractor_id) throw new ApiError(403, 'not your work order');
    if (!workOrderCan(wo.status, body.to)) throw new ApiError(409, `illegal transition ${wo.status} -> ${body.to}`);
    const roles = WO_TRANSITION_ROLES[body.to] ?? [];
    if (s.role !== 'super_admin' && !roles.includes(s.role)) throw new ApiError(403, `role ${s.role} may not move a work order to ${body.to}`);
    if (body.to === 'repaired_claimed' && body.bags_used == null) throw new ApiError(422, 'bags_used is required when claiming a repair');
    const before = wo.status;
    wo.status = body.to;
    if (body.to === 'issued') wo.issued_at = iso(Date.now());
    if (body.bags_used != null) {
      wo.bags_used = body.bags_used;
      wo.bag_variance_flag = !!wo.est_bags && Math.abs(body.bags_used - wo.est_bags) / wo.est_bags > 0.25;
    }
    if (body.to === 'repaired_claimed' && wo.sla_due) wo.sla_breached = Date.now() > Date.parse(wo.sla_due);
    audit(this.db, 'work_order', id, 'transition', s.username, { status: before }, { status: body.to, bags_used: body.bags_used ?? null, bag_lot_no: body.bag_lot_no ?? null, lat: body.lat ?? null, lon: body.lon ?? null }, body.reason ?? null, Date.now());
    const pst = WO_TO_POTHOLE[body.to];
    for (const pid of wo.pothole_ids) {
      const p = this.pothole(pid);
      if (pst) this.setPotholeStatus(p, pst, body.reason ?? `work order #${id}: ${body.to}`, s.username);
      else if (body.to === 'cancelled' && ['work_ordered', 'acknowledged'].includes(p.status)) this.setPotholeStatus(p, 'planned', `work order #${id} cancelled`, s.username);
    }
    return this.delay(wo);
  }

  async uploadWorkOrderPhoto(id: number, p: PhotoUpload): Promise<unknown> {
    this.network(`POST /work-orders/${id}/photos`);
    const wo = this.db.workOrders.find((w) => w.id === id);
    if (!wo) throw new ApiError(404, `work order ${id} not found`);
    let url: string;
    try { url = URL.createObjectURL(p.photo); } catch { url = thumbSvg(id, 'small', false); }
    const rec = { kind: p.kind, url, at: p.at ?? iso(Date.now()), lat: p.lat ?? null, lon: p.lon ?? null, bytes: p.photo.size };
    (wo.photos as Record<string, unknown>[]).push(rec);
    return this.delay(rec);
  }

  async generateRun(circleRef: number | string, date: string): Promise<RepairRunOut> {
    this.requireRole('zonal_admin', 'field_engineer');
    // like the server: an admin-unit id or a circle code
    const ref = String(circleRef);
    const circle = CIRCLES.find((c) => String(c.id) === ref || c.code === ref)?.id;
    if (circle == null) throw new ApiError(404, `circle '${ref}' not found`);
    // issue WOs for the top confirmed/planned potholes in the circle until the crew-hour cap is used (PLAN 7 indent)
    const crewCap = 8 * 2;
    let used = 0;
    const cands = this.db.potholes
      .filter((p) => ['confirmed', 'planned'].includes(p.status) && this.db.meta.get(p.id!)!.circle_id === circle && this.db.meta.get(p.id!)!.agency_code === 'GHMC')
      .sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
    for (const p of cands) {
      const pl = this.db.plans.get(p.id!) ?? planRepair({ ...p, pothole_id: p.id! }, { in_monsoon_window: inMonsoonWindow(Date.now()) }, this.db.rateCard);
      if (used + pl.labour_h > crewCap) break;
      used += pl.labour_h;
      await this.createWorkOrder({ pothole_ids: [p.id!], contractor_id: p.contractor_id ?? undefined, notes: `repair run ${date}` });
    }
    this.db.now = Date.now();
    const run = buildRun(this.db, circle, date);
    if (!run) throw new ApiError(404, `nothing to schedule in circle ${circle}`);
    return this.delay(run);
  }

  getRun(id: number): Promise<RepairRunOut> {
    const r = this.db.runs.find((x) => x.id === id);
    if (!r) return Promise.reject(new ApiError(404, `run ${id} not found`));
    return this.delay(r);
  }

  indentCsvUrl(id: number): string {
    const r = this.db.runs.find((x) => x.id === id);
    if (!r) return '#';
    const rows = [['stop', 'pothole_id', 'work_order_id', 'road', 'lat', 'lon', 'hop_m', 'severity', 'bags', 'bags_low', 'bags_high', 'bag_kg', 'est_cost_inr']];
    r.route.forEach((s, i) => rows.push([String(i + 1), String(s.pothole_id), String(s.work_order_id ?? ''), String(s.road_name ?? ''), String(s.lat), String(s.lon), String(s.hop_m), String(s.severity ?? ''), String(s.bags ?? ''), String(s.bags_low ?? ''), String(s.bags_high ?? ''), String(s.bag_kg ?? ''), String(s.est_cost_inr ?? '')]));
    const csv = rows.map((row) => row.map((c) => (/[",\n]/.test(c) ? `"${c.replace(/"/g, '""')}"` : c)).join(',')).join('\n');
    return `data:text/csv;charset=utf-8,${encodeURIComponent(csv)}`;
  }

  contractorScorecard(id: number): Promise<ContractorScorecard> {
    const c = this.db.contractors.find((x) => x.id === id);
    if (!c) return Promise.reject(new ApiError(404, `contractor ${id} not found`));
    const wos = this.db.workOrders.filter((w) => w.contractor_id === id);
    const closedish = wos.filter((w) => ['repaired_claimed', 'verification_pending', 'verified_closed', 'qc_failed'].includes(w.status));
    const slaMet = closedish.length ? (100 * closedish.filter((w) => !w.sla_breached).length) / closedish.length : null;
    const reopened = wos.filter((w) => w.pothole_ids.some((pid) => ['reopened'].includes(this.pothole(pid).status))).length;
    const fixDays = closedish.map((w) => {
      const p = this.pothole(w.pothole_ids[0]);
      const ev = (this.db.events.get(p.id!) ?? []).find((e) => e.to_status === 'repaired_claimed');
      return ev && w.issued_at ? (Date.parse(ev.ts) - Date.parse(w.issued_at)) / DAY : null;
    }).filter((x): x is number => x != null && x >= 0);
    // Kaplan-Meier-style survival of patches (demo curve, parameterised by the contractor's QC/reopen record)
    const hazard = 0.0025 + 0.004 * ((wos.filter((w) => w.status === 'qc_failed').length + reopened) / Math.max(1, wos.length)) + 0.0007 * (id % 3);
    const survival = [0, 15, 30, 45, 60, 90, 120, 150, 180].map((d) => ({ days: d, survival: Math.round(Math.exp(-hazard * d) * 1000) / 1000 }));
    return this.delay({
      contractor_id: id, name: c.name, work_orders: wos.length, sla_met_pct: slaMet == null ? null : Math.round(slaMet * 10) / 10,
      verified_closed: wos.filter((w) => w.status === 'verified_closed').length, qc_failed: wos.filter((w) => w.status === 'qc_failed').length,
      dlp_events: reopened, reopen_rate: wos.length ? Math.round((reopened / wos.length) * 1000) / 1000 : null,
      median_days_to_fix: fixDays.length ? Math.round(median(fixDays)! * 10) / 10 : null, survival,
    });
  }

  listContractors(): Promise<ContractorBrief[]> { return this.delay(this.db.contractors); }

  private wardIdsFor(unit: string | number): number[] {
    const u = String(unit);
    if (['all', 'GHMC', '0', ''].includes(u)) return this.db.wards.map((w) => w.id);
    return this.db.wards.filter((w) => wardAliases(w).includes(u)).map((w) => w.id);
  }

  adminUnits(kind: AdminUnitKind): Promise<AdminUnitBrief[]> { return this.delay(mockAdminUnits(kind)); }

  kpi(adminUnit: string | number, windowDays = 30): Promise<KpiOut> {
    const wardIds = this.wardIdsFor(adminUnit);
    if (!wardIds.length) return Promise.reject(new ApiError(404, `admin unit ${adminUnit} not found`));
    const now = Date.now();
    const since = now - windowDays * DAY;
    const ps = this.db.potholes.filter((p) => wardIds.includes(this.db.meta.get(p.id!)!.ward_id));
    const evAt = (p: PotholeFused, s: PotholeStatus) => (this.db.events.get(p.id!) ?? []).filter((e) => e.to_status === s).map((e) => Date.parse(e.ts));
    const open = ps.filter((p) => OPEN_STATUSES.includes(p.status));
    const repairHours = ps.map((p) => {
      const c = evAt(p, 'confirmed')[0];
      const r = evAt(p, 'repaired_claimed')[0];
      return c && r && r >= since ? (r - c) / HOUR : null;
    }).filter((x): x is number => x != null);
    const wos = this.db.workOrders.filter((w) => w.pothole_ids.some((pid) => ps.some((p) => p.id === pid)));
    const by_severity: Record<string, number> = {};
    const by_status: Record<string, number> = {};
    for (const p of open) by_severity[p.severity ?? 'unknown'] = (by_severity[p.severity ?? 'unknown'] ?? 0) + 1;
    for (const p of ps) by_status[p.status] = (by_status[p.status] ?? 0) + 1;
    const estCost = open.reduce((s, p) => s + (this.db.plans.get(p.id!)?.est_cost_inr ?? planRepair({ ...p }, {}, this.db.rateCard).est_cost_inr), 0);
    const share = wardIds.length / this.db.wards.length;
    const single = wardIds.length === 1 ? this.db.wards.find((w) => w.id === wardIds[0]) : undefined;
    return this.delay({
      admin_unit_id: single?.id ?? null, admin_unit: single?.name_en ?? String(adminUnit), window_days: windowDays,
      open_potholes: open.length,
      confirmed_new: ps.filter((p) => evAt(p, 'confirmed').some((t) => t >= since)).length,
      repaired_verified: ps.filter((p) => evAt(p, 'verified_closed').some((t) => t >= since)).length,
      median_hours_to_repair: repairHours.length ? Math.round(median(repairHours)! * 10) / 10 : null,
      sla_breaches: wos.filter((w) => w.sla_breached).length,
      reopened: ps.filter((p) => evAt(p, 'reopened').some((t) => t >= since)).length,
      bus_km_scanned: Math.round(windowDays * 2480 * share),
      est_cost_inr: Math.round(estCost),
      baseline_cost_inr: open.length * this.db.rateCard.baseline.cost_per_pothole_inr,
      by_severity, by_status,
    });
  }

  publicWards(): Promise<WardStat[]> {
    const now = Date.now();
    const out: WardStat[] = this.db.wards.map((w) => {
      const ps = this.db.potholes.filter((p) => this.db.meta.get(p.id!)!.ward_id === w.id);
      const open = ps.filter((p) => OPEN_STATUSES.includes(p.status));
      const wos = this.db.workOrders.filter((x) => x.pothole_ids.some((pid) => ps.some((p) => p.id === pid)));
      const done = wos.filter((x) => ['repaired_claimed', 'verification_pending', 'verified_closed', 'qc_failed'].includes(x.status));
      const hrs = ps.map((p) => {
        const ev = this.db.events.get(p.id!) ?? [];
        const c = ev.find((e) => e.to_status === 'confirmed');
        const r = ev.find((e) => e.to_status === 'repaired_claimed');
        return c && r ? (Date.parse(r.ts) - Date.parse(c.ts)) / HOUR : null;
      }).filter((x): x is number => x != null);
      return {
        admin_unit_id: w.id, name_en: w.name_en, name_te: w.name_te, name_hi: w.name_hi, kind: 'ward', zone: w.zone,
        open_potholes: open.length, large_open: open.filter((p) => p.severity === 'large').length,
        repaired_verified_30d: ps.filter((p) => (this.db.events.get(p.id!) ?? []).some((e) => e.to_status === 'verified_closed' && Date.parse(e.ts) > now - 30 * DAY)).length,
        median_hours_to_repair: hrs.length ? Math.round(median(hrs)! * 10) / 10 : null,
        sla_met_pct: done.length ? Math.round((1000 * done.filter((x) => !x.sla_breached).length) / done.length) / 10 : null,
        km_scanned: Math.round(30 * 2480 / this.db.wards.length), road_km: w.road_km,
        potholes_per_km: Math.round((open.length / w.road_km) * 100) / 100,
        est_cost_inr: Math.round(open.reduce((s, p) => s + (this.db.plans.get(p.id!)?.est_cost_inr ?? 0), 0)),
      };
    });
    return this.delay(out);
  }

  getPriorityConfig(): Promise<PriorityConfig> { return this.delay(this.db.priorityConfig); }
  async putPriorityConfig(cfg: PriorityConfig): Promise<PriorityConfig> {
    const s = this.requireRole('zonal_admin');
    const sum = Object.values(cfg.weights).reduce((a, b) => a + b, 0);
    if (Math.abs(sum - 1) > 0.005) throw new ApiError(422, `weights must sum to 1.0 (got ${sum.toFixed(3)})`);
    const before = this.db.priorityConfig;
    this.db.priorityConfig = { ...clone(cfg), version: before.version + 1 };
    audit(this.db, 'priority_config', 1, 'update', s.username, { weights: before.weights, version: before.version }, { weights: cfg.weights, version: before.version + 1 }, 'edited in admin', Date.now());
    return this.delay(this.db.priorityConfig);
  }
  getRateCard(): Promise<RateCard> { return this.delay(this.db.rateCard); }
  async putRateCard(card: RateCard): Promise<RateCard> {
    const s = this.requireRole('zonal_admin');
    for (const [k, v] of Object.entries(card.items)) if (!(v.rate >= 0)) throw new ApiError(422, `rate for ${k} must be >= 0`);
    const before = this.db.rateCard;
    this.db.rateCard = { ...clone(card), version: before.version + 1 };
    audit(this.db, 'rate_card', 1, 'update', s.username, { version: before.version }, { version: before.version + 1 }, 'edited in admin', Date.now());
    return this.delay(this.db.rateCard);
  }
  jurisdictionQueue(): Promise<JurisdictionItem[]> { return this.delay(this.db.jurisdiction); }
  async resolveJurisdiction(id: number, body: JurisdictionResolve): Promise<unknown> {
    const s = this.requireRole('zonal_admin');
    const j = this.db.jurisdiction.find((x) => x.id === id);
    if (!j) throw new ApiError(404, `queue item ${id} not found`);
    j.resolved = true;
    if (j.pothole_id) {
      const m = this.db.meta.get(j.pothole_id);
      if (m) m.agency_code = body.agency_code;
    }
    audit(this.db, 'jurisdiction_queue', id, 'resolve', s.username, { candidates: j.candidates }, { agency_code: body.agency_code, apply_to_segment: body.apply_to_segment ?? true }, null, Date.now());
    return this.delay({ ok: true });
  }
  audit(entity?: string, id?: number): Promise<AuditEntry[]> {
    const rows = this.db.audit.filter((a) => (!entity || a.entity === entity) && (id == null || a.entity_id === id));
    return this.delay([...rows].sort((a, b) => Date.parse(b.at) - Date.parse(a.at)).slice(0, 200));
  }
  async uploadAdminUnits(fc: GeoJsonFeatureCollection): Promise<unknown> {
    const s = this.requireRole('zonal_admin');
    if (fc?.type !== 'FeatureCollection' || !Array.isArray(fc.features)) throw new ApiError(422, 'expected a GeoJSON FeatureCollection');
    const bad = fc.features.findIndex((f) => !f.geometry || !['Polygon', 'MultiPolygon'].includes((f.geometry as { type?: string }).type ?? ''));
    if (bad >= 0) throw new ApiError(422, `feature ${bad}: geometry must be Polygon or MultiPolygon`);
    const names = fc.features.map((f) => String(f.properties?.name_en ?? f.properties?.name ?? '?'));
    audit(this.db, 'admin_unit', 0, 'upload', s.username, null, { count: fc.features.length, names: names.slice(0, 20) }, 'GeoJSON upload (mock: validated, not applied)', Date.now());
    return this.delay({ ok: true, accepted: fc.features.length, names });
  }

  exportUrl(kind: ExportKind): string {
    if (kind === 'potholes.geojson') {
      const fc = {
        type: 'FeatureCollection',
        features: this.db.potholes.map((p) => ({ type: 'Feature', geometry: { type: 'Point', coordinates: [p.lon, p.lat] }, properties: this.listItem(p) })),
      };
      return `data:application/geo+json;charset=utf-8,${encodeURIComponent(JSON.stringify(fc))}`;
    }
    // Measurement Book: only verified_closed work orders are invoice/MB eligible (states.INVOICE_ELIGIBLE)
    const rows = [['work_order_id', 'pothole_id', 'road', 'l_cut_m', 'w_cut_m', 'd_design_mm', 'bags_est', 'bags_used', 'est_cost_inr', 'certified_measurement']];
    for (const w of this.db.workOrders.filter((x) => x.status === 'verified_closed')) {
      for (const pid of w.pothole_ids) {
        const pl = this.db.plans.get(pid);
        const p = this.db.potholes.find((x) => x.id === pid);
        rows.push([w.id, pid, this.db.meta.get(pid)?.road_name ?? '', pl?.l_cut_m ?? '', pl?.w_cut_m ?? '', pl?.d_design_mm ?? '', pl?.bags ?? '', w.bags_used ?? '', pl?.est_cost_inr ?? '', p?.certified_measurement ?? false].map(String));
      }
    }
    const csv = rows.map((r) => r.map((c) => (/[",\n]/.test(c) ? `"${c.replace(/"/g, '""')}"` : c)).join(',')).join('\n');
    return `data:text/csv;charset=utf-8,${encodeURIComponent(csv)}`;
  }

  deviceCert(deviceId: string): Promise<CalibCert> {
    const c = this.db.certs.find((x) => x.device_id === deviceId || x.calib_id === deviceId);
    if (!c) return Promise.reject(new ApiError(404, `device ${deviceId} not found`));
    return this.delay(c);
  }

  async createCitizenReport(r: CitizenReportIn): Promise<CitizenReportOut> {
    if (!(r.lat >= 6 && r.lat <= 38 && r.lon >= 68 && r.lon <= 98)) throw new ApiError(422, 'location outside India');
    const n = this.db.seq.ticket++;
    const ticket = `GM-${String(n).padStart(6, '0')}`;
    // link to an existing pothole within 25 m (PLAN 5.5 citizen confirmation radius)
    const near = this.db.potholes
      .map((p) => ({ p, d: haversineM([p.lon!, p.lat!], [r.lon, r.lat]) }))
      .filter((x) => x.d <= 25 && !['rejected', 'archived'].includes(x.p.status))
      .sort((a, b) => a.d - b.d)[0]?.p;
    let pid: number | null = null;
    if (near) {
      near.complaint_count += 1;
      pid = near.id!;
      if (near.status === 'candidate' && near.passes_seen >= 1) this.setPotholeStatus(near, 'confirmed', '1 pass + citizen report within 25 m', 'citizen');
    }
    const rep: CitizenReportOut = { ticket, status: near ? 'linked' : 'received', pothole_id: pid, created_at: iso(Date.now()), lat: r.lat, lon: r.lon };
    this.db.citizenReports.push(rep);
    return this.delay(rep);
  }

  getCitizenReport(ticket: string): Promise<CitizenReportOut> {
    const t = this.db.citizenReports.find((x) => x.ticket.toUpperCase() === ticket.trim().toUpperCase());
    if (!t) return Promise.reject(new ApiError(404, `ticket ${ticket} not found`));
    const p = t.pothole_id ? this.db.potholes.find((x) => x.id === t.pothole_id) : undefined;
    const status = !p ? t.status : p.status === 'verified_closed' ? 'resolved' : ['work_ordered', 'acknowledged', 'in_progress'].includes(p.status) ? 'repair_scheduled' : p.status === 'rejected' ? 'closed_not_pothole' : t.status;
    return this.delay({ ...t, status });
  }
}

/** Used by tests and the run page fallback: circle list for the mock. */
export const MOCK_CIRCLES = CIRCLES;
export { pointAlong };
