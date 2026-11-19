/**
 * Deterministic Hyderabad demo seed for the mock API.
 *
 * Road geometry is REAL (OpenStreetMap, extracted by web/scripts/extract_roads.py from
 * datasets/extracted/geo/hyderabad.osm.pbf). Potholes, measurements, work orders, contractors and admin-unit
 * stats are SYNTHETIC but internally consistent: severities follow IRC:82-2015 Cl. 7.5.3.4, priorities
 * follow PLAN 6 with the default weights, plans follow PLAN 7 (src/lib/planner.ts). Contractor names are
 * the CRMP package placeholders from configs/agencies.yaml, marked "(demo)". Ward boundaries are illustrative.
 */
import roadsJson from './roads.json';
import { haversineM, lineLengthM, median, pointAlong, type LonLat } from './geo';
import { blobPath, thumbSvg } from './svg';
import { DEFAULT_RATE_CARD, ircSeverity, planRepair } from '../planner';
import { DEFAULT_WEIGHTS } from '../priority';
import type {
  AdminUnitBrief, AuditEntry, CalibCert, CitizenReportOut, ContractorBrief, JurisdictionItem, MeasurementOut, Method,
  PassMeasurement, PotholeEventOut, PotholeFused, PotholeStatus, PriorityConfig, RateCard, RepairPlan, RepairRunOut,
  Severity, WorkOrderOut, WorkOrderStatus,
} from '../types';

export interface Way { way_id: number; name: string; highway: string; lanes: number | null; area: string; coords: LonLat[] }

export interface Ward {
  id: number; name_en: string; name_te: string; name_hi: string; zone: string; zone_te: string; zone_hi: string;
  circle_id: number; circle_name: string; road_km: number;
}

export interface PotholeMeta {
  road_name: string; highway: string; lanes: number; agency_code: string; ward_id: number; circle_id: number;
  way_id: number; chainage_m: number; bearing: number; thumb_url: string; nearby_pois: Record<string, unknown>[];
  mask_norm: [number, number][]; created_by_citizen?: boolean;
}

export interface MockDb {
  now: number;
  potholes: PotholeFused[];
  meta: Map<number, PotholeMeta>;
  measurements: Map<number, MeasurementOut[]>;
  passes: Map<number, PassMeasurement[]>;
  plans: Map<number, RepairPlan>;
  events: Map<number, PotholeEventOut[]>;
  workOrders: WorkOrderOut[];
  runs: RepairRunOut[];
  contractors: ContractorBrief[];
  wards: Ward[];
  citizenReports: CitizenReportOut[];
  jurisdiction: JurisdictionItem[];
  audit: AuditEntry[];
  priorityConfig: PriorityConfig;
  rateCard: RateCard;
  certs: CalibCert[];
  ways: Way[];
  seq: { pothole: number; wo: number; run: number; audit: number; measurement: number; ticket: number; jur: number };
}

// ------------------------------------------------------------------ PRNG
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    uniform: (lo: number, hi: number) => lo + (hi - lo) * next(),
    int: (lo: number, hi: number) => Math.floor(lo + (hi - lo + 1) * next()),
    chance: (p: number) => next() < p,
    pick: <T,>(xs: readonly T[]): T => xs[Math.floor(next() * xs.length)],
    normal: (mu = 0, sd = 1) => {
      const u = Math.max(1e-12, next());
      const v = next();
      return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    },
  };
}
type Rng = ReturnType<typeof mulberry32>;

// ------------------------------------------------------------------ admin units
export const ZONES = {
  Khairatabad: { te: 'ఖైరతాబాద్ జోన్', hi: 'खैरताबाद ज़ोन' },
  'LB Nagar': { te: 'ఎల్‌బీ నగర్ జోన్', hi: 'एलबी नगर ज़ोन' },
} as const;

/** Zone ids / codes as seeded by server/app/seed.py (gis layers), so the mock and the live API use the same codes. */
export const ZONE_UNITS: AdminUnitBrief[] = [
  { id: 3, code: 'GHMC-KHT', kind: 'zone', name_en: 'Khairatabad', name_te: ZONES.Khairatabad.te, name_hi: ZONES.Khairatabad.hi, parent_id: null, official: false },
  { id: 1, code: 'GHMC-LBN', kind: 'zone', name_en: 'LB Nagar', name_te: ZONES['LB Nagar'].te, name_hi: ZONES['LB Nagar'].hi, parent_id: null, official: false },
];

export const WARDS: Ward[] = [
  { id: 101, name_en: 'Ameerpet', name_te: 'అమీర్‌పేట్', name_hi: 'अमीरपेट', zone: 'Khairatabad', zone_te: ZONES.Khairatabad.te, zone_hi: ZONES.Khairatabad.hi, circle_id: 1, circle_name: 'Khairatabad', road_km: 38 },
  { id: 102, name_en: 'Punjagutta', name_te: 'పంజాగుట్ట', name_hi: 'पंजागुट्टा', zone: 'Khairatabad', zone_te: ZONES.Khairatabad.te, zone_hi: ZONES.Khairatabad.hi, circle_id: 1, circle_name: 'Khairatabad', road_km: 24 },
  { id: 103, name_en: 'Khairatabad', name_te: 'ఖైరతాబాద్', name_hi: 'खैरताबाद', zone: 'Khairatabad', zone_te: ZONES.Khairatabad.te, zone_hi: ZONES.Khairatabad.hi, circle_id: 1, circle_name: 'Khairatabad', road_km: 31 },
  { id: 104, name_en: 'Mehdipatnam', name_te: 'మెహదీపట్నం', name_hi: 'मेहदीपट्टनम', zone: 'Khairatabad', zone_te: ZONES.Khairatabad.te, zone_hi: ZONES.Khairatabad.hi, circle_id: 2, circle_name: 'Mehdipatnam', road_km: 35 },
  { id: 105, name_en: 'LB Nagar', name_te: 'ఎల్‌బీ నగర్', name_hi: 'एलबी नगर', zone: 'LB Nagar', zone_te: ZONES['LB Nagar'].te, zone_hi: ZONES['LB Nagar'].hi, circle_id: 3, circle_name: 'LB Nagar', road_km: 42 },
];
/** Mock circles (ids 1..3 are mock-only; codes match the live seed: GHMC-C17 / C12 / C04). */
export const CIRCLES = [
  { id: 1, name: 'Khairatabad', code: 'GHMC-C17', zone: 'Khairatabad', te: 'ఖైరతాబాద్', hi: 'खैरताबाद' },
  { id: 2, name: 'Mehdipatnam', code: 'GHMC-C12', zone: 'Khairatabad', te: 'మెహదీపట్నం', hi: 'मेहदीपट्टनम' },
  { id: 3, name: 'LB Nagar', code: 'GHMC-C04', zone: 'LB Nagar', te: 'ఎల్‌బీ నగర్', hi: 'एलबी नगर' },
];

export function mockAdminUnits(kind: string): AdminUnitBrief[] {
  if (kind === 'zone') return ZONE_UNITS.map((z) => ({ ...z }));
  if (kind === 'circle') {
    return CIRCLES.map((c) => ({
      id: c.id, code: c.code, kind: 'circle' as const, name_en: c.name, name_te: c.te, name_hi: c.hi,
      parent_id: ZONE_UNITS.find((z) => z.name_en === c.zone)?.id ?? null, official: false,
    }));
  }
  if (kind === 'ward') {
    return WARDS.map((w) => ({
      id: w.id, code: `MOCK-W${w.id}`, kind: 'ward' as const, name_en: w.name_en, name_te: w.name_te, name_hi: w.name_hi,
      parent_id: w.circle_id, official: false,
    }));
  }
  return [];
}

/** Names / ids / codes by which a mock ward can be addressed (ward, its circle, its zone). */
export function wardAliases(w: Ward): string[] {
  const c = CIRCLES.find((x) => x.id === w.circle_id);
  const z = ZONE_UNITS.find((x) => x.name_en === w.zone);
  return [w.name_en, String(w.id), `MOCK-W${w.id}`, w.zone, w.circle_name, `circle-${w.circle_id}`, c?.code ?? '', z?.code ?? '']
    .filter((x) => x !== '');
}

export const CONTRACTORS: ContractorBrief[] = [
  { id: 1, name: 'CRMP Package 2 contractor (demo)', package: 'CRMP-P2', agency_code: 'GHMC' },
  { id: 2, name: 'CRMP Package 7 contractor (demo)', package: 'CRMP-P7', agency_code: 'GHMC' },
  { id: 3, name: 'CRMP Package 6 contractor (demo)', package: 'CRMP-P6', agency_code: 'GHMC' },
  { id: 4, name: 'Circle maintenance contractor, Khairatabad (demo)', package: 'KHB-AMC-2026', agency_code: 'GHMC' },
  { id: 5, name: 'Circle maintenance contractor, LB Nagar (demo)', package: 'LBN-AMC-2026', agency_code: 'GHMC' },
];

const AREA_PLAN: { area: string; ward: number; n: number }[] = [
  { area: 'Ameerpet', ward: 101, n: 13 },
  { area: 'Punjagutta', ward: 102, n: 9 },
  { area: 'Khairatabad', ward: 103, n: 11 },
  { area: 'Mehdipatnam', ward: 104, n: 14 },
  { area: 'LB Nagar', ward: 105, n: 13 },
];

const ARTERIAL = new Set(['motorway', 'trunk', 'primary', 'secondary']);
const CLASS_W: Record<string, number> = { trunk: 1.0, primary: 1.0, secondary: 0.8, tertiary: 0.6 };
const POI_KINDS: { kind: string; v: number; names: string[] }[] = [
  { kind: 'school', v: 0.5, names: ['Govt. High School', 'ZPHS (demo)', 'Little Flower School (demo)', 'Govt. Girls High School'] },
  { kind: 'hospital', v: 0.5, names: ['Area Hospital (demo)', 'Nursing Home (demo)', 'Basti Dawakhana'] },
  { kind: 'bus_stop', v: 0.3, names: ['TGSRTC bus stop'] },
  { kind: 'metro_station', v: 0.3, names: ['Metro station'] },
  { kind: 'marketplace', v: 0.2, names: ['Rythu Bazar', 'Vegetable market'] },
];

export const DEMO_CERTS: CalibCert[] = [
  certOf('sim_bus', 'sim-bus', 'bus', 2.4, 18, false, 'Simulator bus (DEMO fallback calibration, uncertified).'),
  certOf('bus_default', 'bus-default', 'bus', 2.4, 18, false, 'DEMO fallback calibration (PLAN 4.1 step 4). Prioritisation only.'),
  certOf('phone_demo', 'phone-demo', 'phone', 1.4, 12, false, 'DEMO fallback calibration (PLAN 4.1 step 4). Prioritisation only.'),
  {
    ...certOf('van_oakd_01', 'van-oakd-01', 'van', 1.62, 14, true, 'Tier B stereo van (demo record): depot checkerboard + tape height; truth fit on staged targets.'),
    reproj_rms_px: 0.31, calibrated_by: 'AE (Q&C), Khairatabad (demo)', known_object_err_pct: 1.8,
    truth_fit: { k_sigma: 0.00031, g_imu: null, n_truth: 24, depth_mae_mm: 4.1, volume_mape: 0.14, fitted_at: '2026-10-28T06:30:00Z' },
    drift_history: [{ trip_id: 'van-2026-11-10-a', scale_drift: 0.012, pitch_drift: 0.2, at: '2026-11-10T05:10:00Z' }],
    calibrated_at: '2026-10-27T05:00:00Z',
  },
];

function certOf(calib_id: string, device_id: string, kind: string, h: number, pitch: number, certified: boolean, notes: string): CalibCert {
  return {
    calib_id, device_id, vehicle_kind: kind, fx: 1280, fy: 1280, cx: 640, cy: 360, dist: [0, 0, 0, 0, 0], img_w: 1280, img_h: 720,
    reproj_rms_px: null, h_true_m: h, pitch_deg: pitch, roll_deg: 0, yaw_deg: 0, lateral_offset_m: 0,
    wheel_track_left_m: kind === 'bus' ? -1.0 : -0.9, wheel_track_right_m: kind === 'bus' ? 1.0 : 0.9, gnss_latency_s: 0.3,
    known_object_err_pct: null, calibrated_at: null, calibrated_by: null, certified, drift_history: [],
    truth_fit: { k_sigma: 0.0004, g_imu: null, n_truth: 0, depth_mae_mm: null, volume_mape: null, fitted_at: null }, notes,
  };
}

export const DEFAULT_PRIORITY_CONFIG: PriorityConfig = {
  version: 1,
  weights: { ...DEFAULT_WEIGHTS },
  age_sla: { sla_hours: { arterial: 24, other_monsoon: 72, other_offseason: 168, rnb_state_road: 168 }, breached_bonus: 0.3 },
  monsoon: { base: 0.5, in_window: 0.8, heavy_rain: 1.0, hotspot_bonus: 0.2, water_filled_bonus: 0.1, manual_heavy_rain: false },
};

// ------------------------------------------------------------------ helpers
const HOUR = 3600_000;
const DAY = 24 * HOUR;
const iso = (t: number) => new Date(t).toISOString();
const round = (x: number, d = 1) => Math.round(x * 10 ** d) / 10 ** d;

export function inMonsoonWindow(t: number): boolean {
  const m = new Date(t + 5.5 * HOUR).getUTCMonth() + 1; // IST month
  return (m >= 6 && m <= 9) || (m >= 10 && m <= 11);
}

export function slaHours(highway: string, agency: string, monsoon: boolean, cfg: PriorityConfig = DEFAULT_PRIORITY_CONFIG): number {
  const h = cfg.age_sla?.sla_hours ?? { arterial: 24, other_monsoon: 72, other_offseason: 168, rnb_state_road: 168 };
  if (agency === 'RNB') return h.rnb_state_road ?? 168;
  if (ARTERIAL.has(highway)) return h.arterial ?? 24;
  return monsoon ? h.other_monsoon ?? 72 : h.other_offseason ?? 168;
}

const STATUS_PLAN: [PotholeStatus, number][] = [
  ['candidate', 7], ['confirmed', 17], ['planned', 9], ['work_ordered', 6], ['acknowledged', 2], ['in_progress', 3],
  ['repaired_claimed', 2], ['verification_pending', 3], ['verified_closed', 5], ['qc_failed', 1], ['reopened', 2],
  ['recurrence_watch', 1], ['rejected', 1],
];
// lifecycle order used to build event timelines
const LIFECYCLE: PotholeStatus[] = [
  'candidate', 'confirmed', 'planned', 'work_ordered', 'acknowledged', 'in_progress', 'repaired_claimed',
  'verification_pending', 'verified_closed',
];
const POTHOLE_TO_WO: Partial<Record<PotholeStatus, WorkOrderStatus>> = {
  work_ordered: 'issued', acknowledged: 'acknowledged', in_progress: 'in_progress', repaired_claimed: 'repaired_claimed',
  verification_pending: 'verification_pending', verified_closed: 'verified_closed', qc_failed: 'qc_failed',
  recurrence_watch: 'verified_closed', reopened: 'verified_closed',
};

function contractorFor(area: string, highway: string, agency: string): number | null {
  if (agency !== 'GHMC') return null;
  const arterial = ARTERIAL.has(highway);
  if (area === 'LB Nagar') return arterial ? 3 : 5;
  if (area === 'Mehdipatnam') return arterial ? 2 : 4;
  return arterial ? 1 : 4;
}

/** Raw way as stored in roads.json (JSON arrays are untyped `number[]`, not [lon, lat] tuples). */
export interface RawWay {
  way_id: number; name: string; highway: string; lanes: number | null; area: string;
  coords: readonly (readonly number[])[];
}

/** One JSON coordinate -> [lon, lat], or null when it is malformed (short, non-finite, out of WGS84 range). */
export function toLonLat(c: readonly number[]): LonLat | null {
  if (c.length < 2) return null;
  const lon = c[0];
  const lat = c[1];
  if (!Number.isFinite(lon) || !Number.isFinite(lat) || Math.abs(lon) > 180 || Math.abs(lat) > 90) return null;
  return [lon, lat];
}

/** Validate the OSM extract: malformed points are dropped, ways left with fewer than 2 points are dropped. */
export function normaliseWays(raw: readonly RawWay[]): Way[] {
  const out: Way[] = [];
  for (const w of raw) {
    const coords = w.coords.map(toLonLat).filter((c): c is LonLat => c !== null);
    if (coords.length >= 2) out.push({ way_id: w.way_id, name: w.name, highway: w.highway, lanes: w.lanes, area: w.area, coords });
  }
  return out;
}

function agencyFor(way: Way): string {
  if (/vija[yv]a?wada highway/i.test(way.name)) return 'NHAI'; // NH65 through LB Nagar
  return 'GHMC';
}

// ------------------------------------------------------------------ seed
export function createSeed(opts: { now?: number; seed?: number } = {}): MockDb {
  const now = opts.now ?? Date.now();
  const rng = mulberry32(opts.seed ?? 20261113);
  const monsoon = inMonsoonWindow(now);
  const ways: Way[] = normaliseWays(roadsJson.ways);
  const db: MockDb = {
    now, potholes: [], meta: new Map(), measurements: new Map(), passes: new Map(), plans: new Map(), events: new Map(),
    workOrders: [], runs: [], contractors: CONTRACTORS.map((c) => ({ ...c })), wards: WARDS.map((w) => ({ ...w })),
    citizenReports: [], jurisdiction: [], audit: [], priorityConfig: structuredClone(DEFAULT_PRIORITY_CONFIG),
    rateCard: structuredClone(DEFAULT_RATE_CARD), certs: structuredClone(DEMO_CERTS), ways,
    seq: { pothole: 1, wo: 1, run: 1, audit: 1, measurement: 1, ticket: 1, jur: 1 },
  };

  // statuses, shuffled deterministically
  const statuses: PotholeStatus[] = STATUS_PLAN.flatMap(([s, n]) => Array<PotholeStatus>(n).fill(s));
  for (let i = statuses.length - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    [statuses[i], statuses[j]] = [statuses[j], statuses[i]];
  }

  let k = 0;
  for (const ap of AREA_PLAN) {
    const ward = WARDS.find((w) => w.id === ap.ward)!;
    const pool = ways.filter((w) => w.area === ap.area && !/flyover|underpass/i.test(w.name));
    const lens = pool.map((w) => lineLengthM(w.coords) * (CLASS_W[w.highway] ?? 0.5));
    const total = lens.reduce((a, b) => a + b, 0);
    const placed: LonLat[] = db.potholes.map((p) => [p.lon!, p.lat!]);
    // Mehdipatnam: a deliberate cluster on Golkonda Road (exercises the jet-patcher rule and a "Poor" stretch)
    const clusterWay = ap.area === 'Mehdipatnam' ? pool.filter((w) => w.name === 'Golkonda Road').sort((a, b) => lineLengthM(b.coords) - lineLengthM(a.coords))[0] : undefined;
    const clusterN = clusterWay ? 8 : 0;
    for (let i = 0; i < ap.n; i++) {
      let way: Way;
      let chain: number;
      let pos: { p: LonLat; bearing: number };
      let tries = 0;
      for (;;) {
        if (i < clusterN && clusterWay) {
          way = clusterWay;
          const L = lineLengthM(way.coords);
          chain = Math.min(L - 5, Math.max(5, L * 0.4 + i * 11 + rng.uniform(-3, 3)));
        } else {
          let x = rng.next() * total;
          let wi = 0;
          while (wi < lens.length - 1 && x > lens[wi]) { x -= lens[wi]; wi++; }
          way = pool[wi];
          chain = rng.uniform(5, Math.max(6, lineLengthM(way.coords) - 5));
        }
        pos = pointAlong(way.coords, chain);
        const minGap = i < clusterN ? 6 : 45;
        if (placed.every((q) => haversineM(q, pos.p) > minGap) || ++tries > 40) break;
      }
      placed.push(pos.p);
      const status = statuses[k++];
      makePothole(db, rng, { way, chain, pos, ward, status, monsoon, inCluster: i < clusterN });
    }
  }

  buildWorkOrders(db, rng);
  buildRun(db, 1);
  buildCitizenAndQueues(db, rng);
  return db;
}

interface MakeArgs { way: Way; chain: number; pos: { p: LonLat; bearing: number }; ward: Ward; status: PotholeStatus; monsoon: boolean; inCluster: boolean }

function makePothole(db: MockDb, rng: Rng, a: MakeArgs): PotholeFused {
  const id = db.seq.pothole++;
  const now = db.now;
  const water = !a.inCluster && rng.chance(0.09);
  // geometry & depth (log-normal-ish depth, larger on arterials under heavy axle loads)
  const heavy = a.way.highway === 'trunk' || a.way.highway === 'primary';
  const depth = water ? null : Math.max(12, Math.min(140, Math.exp(rng.normal(Math.log(heavy ? 48 : 36), 0.45))));
  const length = round(rng.uniform(0.28, a.inCluster ? 1.3 : 1.15), 2);
  const width = round(Math.min(length, rng.uniform(0.2, a.inCluster ? 1.0 : 0.85)), 2);
  const area = round((Math.PI / 4) * length * width * rng.uniform(0.82, 0.95), 3);
  const certifiedVan = !water && rng.chance(0.08);
  const method: Method = certifiedVan ? 'stereo' : rng.chance(0.35) ? 'mono+imu' : 'mono';
  const sigmaFloor = certifiedVan ? 3 : 4;
  const depthSigma = depth == null ? null : round(Math.max(sigmaFloor, (certifiedVan ? 0.05 : 0.11) * depth + rng.uniform(1, 4)), 1);
  const depthMean = depth == null ? null : round(depth * rng.uniform(0.5, 0.62), 1);
  const depthMax = depth == null ? null : round(depth * rng.uniform(1.12, 1.3), 1);
  const volume = depthMean == null ? null : round(area * depthMean, 1); // m2 * mm = litres
  const severity: Severity = ircSeverity(depth, width, water);
  const highway = a.way.highway;
  const agency = agencyFor(a.way);
  const lanes = a.way.lanes ?? (highway === 'trunk' ? 3 : highway === 'tertiary' ? 1 : 2);
  const ageDays = a.status === 'candidate' ? rng.uniform(0.2, 6) : a.status === 'confirmed' ? rng.uniform(0.3, 4) : rng.uniform(3, 40);
  const firstSeen = now - ageDays * DAY;
  const passesTotal = rng.int(3, 14);
  const passesSeen = a.status === 'candidate' ? 1 : Math.min(passesTotal, rng.int(2, 9));
  const imu = method === 'mono+imu';
  const laneOffset = round(rng.uniform(-3.2, 3.2), 2);
  const inWheel = Math.abs(Math.abs(laneOffset) - 0.9) < 0.35;
  const complaints = rng.chance(0.25) ? rng.int(1, 3) : 0;

  // POIs within 200 m (synthetic, names generic)
  const pois: Record<string, unknown>[] = [];
  for (const pk of POI_KINDS) {
    if (rng.chance(pk.kind === 'bus_stop' ? 0.55 : pk.kind === 'metro_station' ? (['Ameerpet', 'Punjagutta', 'Khairatabad', 'LB Nagar'].includes(a.way.area) ? 0.35 : 0.05) : 0.22)) {
      pois.push({ kind: pk.kind, name: rng.pick(pk.names), dist_m: Math.round(rng.uniform(25, 195)), v: pk.v });
    }
  }

  // priority (PLAN 6, default weights)
  const sevScore = severity === 'large' ? 1 : severity === 'medium' ? 0.6 : severity === 'small' ? 0.3 : area >= 0.3 ? 0.8 : 0.6;
  const sWhy = [`IRC severity ${severity}`];
  let S = sevScore;
  if (inWheel) { S += 0.2; sWhy.push('in wheel path'); }
  if (imu) { S += 0.1; sWhy.push('IMU corroborated'); }
  S = Math.min(1, S);
  const busPerDay = highway === 'trunk' || highway === 'primary' ? rng.int(55, 190) : highway === 'secondary' ? rng.int(15, 60) : rng.int(0, 12);
  const C = CLASS_W[highway] ?? 0.4;
  const E = 0.5 * C + 0.5 * Math.min(1, busPerDay / 50);
  const V = Math.min(1, 1 - pois.reduce((acc, p) => acc * (1 - (p.v as number)), 1));
  const sla = slaHours(highway, agency, a.monsoon);
  const confirmedAt = a.status === 'candidate' ? null : firstSeen + rng.uniform(2, 20) * HOUR;
  const hrsSince = confirmedAt ? (now - confirmedAt) / HOUR : 0;
  const open = !['verified_closed', 'recurrence_watch', 'rejected', 'archived'].includes(a.status);
  const breachedT = open && confirmedAt != null && hrsSince > sla;
  const T = confirmedAt == null ? 0 : Math.min(1, (open ? Math.min(1, hrsSince / sla) : 0) + (breachedT ? 0.3 : 0) + Math.min(0.3, 0.1 * complaints));
  const hotspot = (a.way.area === 'Mehdipatnam' || a.way.area === 'Khairatabad') && rng.chance(0.3);
  const M = Math.min(1, (a.monsoon ? 0.8 : 0.5) + (hotspot ? 0.2 : 0) + (water ? 0.1 : 0));
  const w = DEFAULT_WEIGHTS;
  const priority = round(100 * Math.min(1, w.S * S + w.E * E + w.V * V + w.T * T + w.M * M), 1);
  const breakdown = {
    config_version: 1,
    weights: { ...w },
    components: {
      S: { score: round(S, 3), why: sWhy },
      E: { score: round(E, 3), why: [`OSM ${highway} (C=${C})`, `${busPerDay} bus passes/day`] },
      V: { score: round(V, 3), why: pois.length ? pois.map((p) => `${p.kind} ${p.dist_m} m`) : ['no POI within 200 m'] },
      T: { score: round(T, 3), why: confirmedAt ? [`${Math.round(hrsSince)} h since confirmed / SLA ${sla} h`, ...(breachedT ? ['SLA breached'] : []), ...(complaints ? [`${complaints} complaint(s)`] : [])] : ['not confirmed yet'] },
      M: { score: round(M, 3), why: [a.monsoon ? 'NE/SW monsoon window' : 'off-season', ...(hotspot ? ['waterlogging hotspot'] : []), ...(water ? ['water-filled'] : [])] },
    },
  };

  const lastSeen = a.status === 'candidate' ? firstSeen : now - rng.uniform(0.1, Math.min(ageDays, 3)) * DAY;
  const p: PotholeFused = {
    id, agency_id: agency === 'GHMC' ? 1 : agency === 'NHAI' ? 6 : 4, admin_unit_id: a.ward.id, road_segment_id: a.way.way_id % 100000,
    way_id: a.way.way_id, direction: rng.chance(0.5) ? 0 : 1, lane_band: Math.round(laneOffset / 1.75), lane_offset_m: laneOffset,
    chainage_m: round(a.chain, 1), lat: round(a.pos.p[1], 6), lon: round(a.pos.p[0], 6), h3_r12: null,
    status: a.status, severity, surface: 'BT', water_filled_ratio: water ? round(rng.uniform(0.6, 1), 2) : 0,
    first_seen: iso(firstSeen), last_seen: iso(lastSeen), passes_seen: passesSeen, passes_total: passesTotal,
    trips_seen: Math.max(1, passesSeen - rng.int(0, 1)), complaint_count: complaints, imu_corroborations: imu ? rng.int(1, passesSeen) : 0,
    priority, priority_breakdown: breakdown, contract_id: null, contractor_id: contractorFor(a.way.area, highway, agency),
    best_method: method, certified_measurement: certifiedVan, defect_cause: water ? 'waterlogging' : rng.pick(['unknown', 'unknown', 'base_failure', 'utility_cut', 'drainage', 'heavy_axle'] as const),
    reopen_count_12m: a.status === 'reopened' ? rng.int(1, 2) : 0, root_cause_flag: false,
    remeasure_after: water ? iso(now + 2 * DAY) : null, junction_dist_m: round(rng.uniform(8, 220), 0), lit: rng.chance(0.85),
    depth_p95_mm: depth == null ? null : round(depth, 1), depth_max_mm: depthMax, depth_mean_mm: depthMean, depth_sigma_mm: depthSigma,
    area_m2: area, area_sigma: round(area * (certifiedVan ? 0.05 : 0.11), 3), volume_l: volume,
    volume_sigma: volume == null ? null : round(volume * (certifiedVan ? 0.12 : 0.24), 1), length_m: length, width_m: width,
    length_sigma_m: round(length * 0.06, 3), width_sigma_m: round(width * 0.07, 3), water_filled: water,
    depth_quality: water ? 'unknown_water' : 'ok', quality: round(rng.uniform(0.45, 0.92), 2), method,
    imu_corroborated: imu, imu_depth_mm: null,
  };
  p.root_cause_flag = p.reopen_count_12m >= 2;
  if (a.way.area === 'LB Nagar') p.contract_id = agency === 'GHMC' ? 6 : null;
  else p.contract_id = agency === 'GHMC' ? (ARTERIAL.has(highway) ? (a.way.area === 'Mehdipatnam' ? 7 : 2) : null) : null;
  db.potholes.push(p);

  const mask = blobPath(0.5 + rng.uniform(-0.12, 0.12), 0.72 + rng.uniform(-0.06, 0.06), 0.06 + 0.1 * width, 0.03 + 0.05 * length, id);
  db.meta.set(id, {
    road_name: a.way.name, highway, lanes, agency_code: agency, ward_id: a.ward.id, circle_id: a.ward.circle_id, way_id: a.way.way_id,
    chainage_m: a.chain, bearing: a.pos.bearing, thumb_url: thumbSvg(id, severity, water), nearby_pois: pois,
    mask_norm: mask.map(([x, y]) => [round(x, 4), round(y, 4)] as [number, number]),
  });

  // measurements & passes (synthetic; consistent with fused values)
  const ms: MeasurementOut[] = [];
  const passes: PassMeasurement[] = [];
  const nPasses = Math.min(passesSeen, 4);
  for (let pi = 0; pi < nPasses; pi++) {
    const tripId = `trip-${(id * 7 + pi * 13) % 997}`;
    const frames: number[] = [];
    const nF = rng.int(2, 3);
    for (let fi = 0; fi < nF; fi++) {
      const range = round(rng.uniform(5.5, 13.5), 2);
      const fs = depth == null ? null : round(Math.max(3, (certifiedVan ? 0.0003 : 0.0004) * range * range * 1000 * 0.35 + 3), 1);
      const fd = depth == null ? null : round(depth + rng.normal(0, fs ?? 5), 1);
      const seqNo = 1000 + pi * 10 + fi;
      frames.push(seqNo);
      ms.push({
        id: db.seq.measurement++, detection_id: null, observation_id: null, frame_url: null, mask_url: null, depth_map_url: null,
        depth_p95_mm: fd, depth_max_mm: fd == null ? null : round(fd * 1.2, 1), depth_mean_mm: fd == null ? null : round(fd * 0.56, 1),
        depth_sigma_mm: fs, area_m2: round(area * rng.uniform(0.9, 1.1), 3), area_sigma: round(area * 0.15, 3),
        volume_l: fd == null ? null : round(area * fd * 0.56, 1), volume_sigma: fd == null ? null : round(area * fd * 0.56 * 0.3, 1),
        length_m: round(length * rng.uniform(0.93, 1.07), 2), width_m: round(width * rng.uniform(0.93, 1.07), 2),
        water_filled: water, depth_quality: water ? 'unknown_water' : 'ok', quality: round(rng.uniform(0.3, 0.9), 2),
        method: pi === 0 && certifiedVan ? 'stereo' : imu ? 'mono+imu' : 'mono', imu_corroborated: imu, imu_depth_mm: null,
        range_m: range, lane_offset_m: round(laneOffset + rng.normal(0, 0.2), 2), ground_xz: [round(laneOffset, 2), range],
        surface_model: rng.chance(0.4) ? 'quadratic' : 'plane', plane_rms_mm: round(rng.uniform(3, 9), 1), ring_rms_mm: round(rng.uniform(2, 7), 1),
        plane_inlier_ratio: round(rng.uniform(0.55, 0.9), 2), scale_factor: round(rng.uniform(0.86, 1.14), 3), reject_reason: null,
        depth_model_version: pi === 0 && certifiedVan ? 'stereo-oakd-sgbm' : 'da2-metric-outdoor-base', calib_id: pi === 0 && certifiedVan ? 'van_oakd_01' : 'sim_bus',
        certified: pi === 0 && certifiedVan, mask_px: rng.int(2500, 26000), seq: seqNo, t_frame_ms: seqNo * 33,
        extras: { mask_polygon_norm: db.meta.get(id)!.mask_norm },
      });
    }
    passes.push({
      depth_p95_mm: depth == null ? null : round(depth + rng.normal(0, (depthSigma ?? 5) * 0.6), 1),
      depth_max_mm: depthMax, depth_mean_mm: depthMean, depth_sigma_mm: depthSigma == null ? null : round(depthSigma * 1.3, 1),
      area_m2: area, area_sigma: round(area * 0.12, 3), volume_l: volume, volume_sigma: volume == null ? null : round(volume * 0.28, 1),
      length_m: length, width_m: width, water_filled: water, depth_quality: water ? 'unknown_water' : 'ok', quality: round(rng.uniform(0.4, 0.9), 2),
      method: pi === 0 && certifiedVan ? 'stereo' : imu ? 'mono+imu' : 'mono', imu_corroborated: imu, imu_depth_mm: null,
      n_frames: nF, n_frames_total: nF, trip_id: tripId, pothole_id: id, range_m: null, lane_offset_m: laneOffset, ground_xz: null,
      scale_factor: null, depth_model_version: pi === 0 && certifiedVan ? 'stereo-oakd-sgbm' : 'da2-metric-outdoor-base',
      certified: pi === 0 && certifiedVan, frames_used: frames, reject_reason: null,
    });
  }
  db.measurements.set(id, ms);
  db.passes.set(id, passes);

  // plan for anything past confirmation
  if (!['candidate', 'confirmed', 'rejected'].includes(a.status)) {
    db.plans.set(id, planRepair({ ...p, pothole_id: id }, { in_monsoon_window: a.monsoon, cluster_n_within_300m: a.inCluster ? 8 : 0 }, db.rateCard));
  }

  // event timeline
  const events: PotholeEventOut[] = [{ ts: iso(firstSeen), from_status: null, to_status: 'candidate', reason: 'first detection (bus pass)', actor: 'pipeline' }];
  let t = confirmedAt ?? firstSeen;
  const target = a.status;
  const pathTo = (s: PotholeStatus): PotholeStatus[] => {
    if (s === 'rejected') return ['rejected'];
    if (s === 'qc_failed') return [...LIFECYCLE.slice(1, LIFECYCLE.indexOf('verification_pending') + 1), 'qc_failed'];
    if (s === 'recurrence_watch') return [...LIFECYCLE.slice(1), 'recurrence_watch'];
    if (s === 'reopened') return [...LIFECYCLE.slice(1), 'recurrence_watch', 'reopened'];
    const ix = LIFECYCLE.indexOf(s);
    return ix > 0 ? LIFECYCLE.slice(1, ix + 1) : [];
  };
  const steps = pathTo(target);
  const span = Math.max(1, (lastSeen - t) / HOUR);
  let prev: PotholeStatus = 'candidate';
  steps.forEach((s, i) => {
    t = i === 0 ? (confirmedAt ?? firstSeen + HOUR) : t + (span / (steps.length + 1)) * HOUR * rng.uniform(0.6, 1.2);
    t = Math.min(t, now - 10 * 60_000);
    events.push({ ts: iso(t), from_status: prev, to_status: s, reason: REASONS[s] ?? null, actor: ACTORS[s] ?? 'pipeline' });
    prev = s;
  });
  db.events.set(id, events);
  return p;
}

const REASONS: Partial<Record<PotholeStatus, string>> = {
  confirmed: '2 independent passes (2 trips)',
  planned: 'repair plan computed (IRC:116 cold mix)',
  work_ordered: 'work order issued',
  acknowledged: 'contractor acknowledged',
  in_progress: 'crew check-in, before photo',
  repaired_claimed: 'after photo, bags used recorded',
  verification_pending: 'awaiting 3 clean passes within 14 days',
  verified_closed: '3 clean passes on 2 trips; QC sample ok',
  qc_failed: 'engineer QC: patch edge failure',
  recurrence_watch: '12-month recurrence watch',
  reopened: 're-detected on 2 passes inside DLP',
  rejected: 'engineer: manhole cover (false positive)',
};
const ACTORS: Partial<Record<PotholeStatus, string>> = {
  planned: 'planner', work_ordered: 'ae.khairatabad', acknowledged: 'contractor', in_progress: 'contractor', repaired_claimed: 'contractor',
  qc_failed: 'ae.khairatabad', rejected: 'ae.khairatabad',
};

function buildWorkOrders(db: MockDb, rng: Rng): void {
  const groups = new Map<string, PotholeFused[]>();
  for (const p of db.potholes) {
    const ws = POTHOLE_TO_WO[p.status];
    if (!ws) continue;
    const m = db.meta.get(p.id!)!;
    const key = `${ws}|${p.contractor_id ?? 'none'}|${m.ward_id}`;
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(p);
  }
  for (const [key, ps] of groups) {
    const ws = key.split('|')[0] as WorkOrderStatus;
    // 500 m same-owner batching, max 3 potholes per WO
    const remaining = [...ps];
    while (remaining.length) {
      const head = remaining.shift()!;
      const batch = [head];
      for (let i = 0; i < remaining.length && batch.length < 3; ) {
        if (haversineM([head.lon!, head.lat!], [remaining[i].lon!, remaining[i].lat!]) <= 500) batch.push(...remaining.splice(i, 1));
        else i++;
      }
      const id = db.seq.wo++;
      const evs = db.events.get(head.id!)!;
      const issuedEv = evs.find((e) => e.to_status === 'work_ordered');
      const issuedAt = issuedEv ? Date.parse(issuedEv.ts) : db.now - 2 * DAY;
      const m = db.meta.get(head.id!)!;
      const sla = slaHours(m.highway, m.agency_code, inMonsoonWindow(db.now), db.priorityConfig);
      const slaDue = issuedAt + sla * HOUR;
      const plans = batch.map((p) => db.plans.get(p.id!)).filter(Boolean) as RepairPlan[];
      const estBags = plans.reduce((s, pl) => s + pl.bags, 0);
      const done = ['repaired_claimed', 'verification_pending', 'verified_closed', 'qc_failed'].includes(ws);
      const repairedEv = evs.find((e) => e.to_status === 'repaired_claimed');
      const repairedAt = repairedEv ? Date.parse(repairedEv.ts) : null;
      const breached = repairedAt ? repairedAt > slaDue : !['cancelled'].includes(ws) && db.now > slaDue && !done;
      const overH = breached ? ((repairedAt ?? db.now) - slaDue) / HOUR : 0;
      const bagsUsed = done ? Math.max(1, Math.round(estBags * rng.uniform(0.85, rng.chance(0.25) ? 1.5 : 1.15))) : null;
      const photos: Record<string, unknown>[] = [];
      if (['in_progress', ...(done ? [ws] : [])].includes(ws)) photos.push({ kind: 'before', url: thumbSvg(head.id!, head.severity, head.water_filled), at: iso(issuedAt + 4 * HOUR), lat: head.lat, lon: head.lon });
      if (done) photos.push({ kind: 'after', url: thumbSvg(head.id! + 1000, 'small', false), at: iso((repairedAt ?? issuedAt) + HOUR), lat: head.lat, lon: head.lon });
      db.workOrders.push({
        id, status: ws, pothole_ids: batch.map((p) => p.id!), agency_id: head.agency_id ?? 1, contract_id: head.contract_id ?? null,
        contractor_id: head.contractor_id ?? null, contractor_name: db.contractors.find((c) => c.id === head.contractor_id)?.name ?? null,
        crew: `Crew ${String.fromCharCode(65 + (id % 4))}`, issued_at: iso(issuedAt), sla_due: iso(slaDue), sla_breached: breached,
        escalation_level: !breached ? null : overH > 72 ? 'ZONAL_COMMISSIONER' : overH > 24 ? 'EE' : 'DEE',
        est_bags: estBags, est_cost_inr: Math.round(plans.reduce((s, pl) => s + pl.est_cost_inr, 0)), bags_used: bagsUsed,
        bag_variance_flag: bagsUsed != null && estBags > 0 && Math.abs(bagsUsed - estBags) / estBags > 0.25,
        is_rework: false, parent_work_order_id: null, photos, run_id: null,
      });
      for (const p of batch) {
        audit(db, 'work_order', id, 'create', 'ae.khairatabad', null, { status: 'issued', pothole_id: p.id }, null, issuedAt);
      }
    }
  }
}

/** Daily per-circle repair run: open WOs (issued/acknowledged) in the circle, greedy nearest-neighbour route. */
export function buildRun(db: MockDb, circleId: number, date?: string): RepairRunOut | null {
  const wos = db.workOrders.filter((w) => ['issued', 'acknowledged'].includes(w.status) && w.pothole_ids.some((pid) => db.meta.get(pid)?.circle_id === circleId));
  const pids = wos.flatMap((w) => w.pothole_ids).filter((pid) => db.meta.get(pid)?.circle_id === circleId);
  if (!pids.length) return null;
  const id = db.seq.run++;
  const pts = pids.map((pid) => db.potholes.find((p) => p.id === pid)!);
  const office: LonLat = circleId === 1 ? [78.4625, 17.4105] : circleId === 2 ? [78.4398, 17.3936] : [78.5518, 17.3477];
  const route: RepairRunOut['route'] = [];
  let cur = office;
  const left = [...pts];
  const tot = { bags: 0, crm_t: 0, emulsion_kg: 0, sand_kg: 0, crew_h: 0, cost_inr: 0 };
  while (left.length) {
    left.sort((a, b) => haversineM(cur, [a.lon!, a.lat!]) - haversineM(cur, [b.lon!, b.lat!]));
    const p = left.shift()!;
    const hop = haversineM(cur, [p.lon!, p.lat!]);
    const pl = db.plans.get(p.id!);
    const m = db.meta.get(p.id!)!;
    route.push({
      pothole_id: p.id!, lat: p.lat!, lon: p.lon!, hop_m: Math.round(hop), road_name: m.road_name, severity: p.severity ?? null,
      bags: pl?.bags ?? null, bags_low: pl?.bags_low ?? null, bags_high: pl?.bags_high ?? null, bag_kg: pl?.bag_kg ?? 25,
      est_cost_inr: pl?.est_cost_inr ?? null, spec_code: pl?.spec_code ?? null, work_order_id: wos.find((w) => w.pothole_ids.includes(p.id!))?.id ?? null,
      a_cut_m2: pl?.a_cut_m2 ?? null, d_design_mm: pl?.d_design_mm ?? null,
    });
    if (pl) {
      tot.bags += pl.bags; tot.crm_t += pl.crm_t; tot.emulsion_kg += pl.emulsion_kg; tot.sand_kg += pl.sand_kg;
      tot.crew_h += pl.labour_h; tot.cost_inr += pl.est_cost_inr;
    }
    cur = [p.lon!, p.lat!];
  }
  const run: RepairRunOut = {
    id, circle_id: circleId, circle_name: CIRCLES.find((c) => c.id === circleId)?.name ?? null,
    run_date: date ?? new Date(db.now + 5.5 * HOUR).toISOString().slice(0, 10), work_order_ids: wos.map((w) => w.id), route,
    totals: Object.fromEntries(Object.entries(tot).map(([k, v]) => [k, Math.round(v * 1000) / 1000])) as RepairRunOut['totals'],
    crew_hours_cap: 8, generated_at: iso(db.now - 3 * HOUR),
  };
  for (const w of wos) w.run_id = id;
  db.runs.push(run);
  return run;
}

function buildCitizenAndQueues(db: MockDb, rng: Rng): void {
  const withComplaints = db.potholes.filter((p) => p.complaint_count > 0).slice(0, 4);
  for (const p of withComplaints) {
    const n = db.seq.ticket++;
    db.citizenReports.push({
      ticket: `GM-${String(n).padStart(6, '0')}`, status: p.status === 'verified_closed' ? 'resolved' : 'linked',
      pothole_id: p.id!, created_at: iso(Date.parse(p.first_seen!) + rng.uniform(1, 30) * HOUR), lat: p.lat!, lon: p.lon!,
    });
  }
  const nh = db.potholes.find((p) => db.meta.get(p.id!)!.agency_code === 'NHAI');
  if (nh) {
    db.jurisdiction.push({ id: db.seq.jur++, pothole_id: nh.id!, road_segment_id: nh.road_segment_id ?? null, reason: 'OSM name suggests NH65 (NHAI) but segment lies inside GHMC limits; no ref tag', candidates: ['NHAI', 'GHMC'], created_at: iso(db.now - 30 * HOUR), resolved: false });
  }
  const irr = db.potholes.find((p) => db.meta.get(p.id!)!.road_name === 'Inner Ring Road');
  if (irr) {
    db.jurisdiction.push({ id: db.seq.jur++, pothole_id: irr.id!, road_segment_id: irr.road_segment_id ?? null, reason: 'Inner Ring Road: R&B vs GHMC ownership not recorded for this stretch', candidates: ['GHMC', 'RNB'], created_at: iso(db.now - 52 * HOUR), resolved: false });
  }
  for (const p of db.potholes) {
    for (const e of db.events.get(p.id!) ?? []) {
      if (e.from_status) audit(db, 'pothole', p.id!, 'transition', e.actor ?? 'pipeline', { status: e.from_status }, { status: e.to_status }, e.reason ?? null, Date.parse(e.ts));
    }
  }
  audit(db, 'priority_config', 1, 'update', 'zonal.khairatabad', { weights: { S: 0.3, E: 0.25, V: 0.2, T: 0.15, M: 0.1 }, version: 0 }, { weights: db.priorityConfig.weights, version: 1 }, 'adopt PLAN 6 defaults', db.now - 20 * DAY);
  db.audit.sort((a, b) => Date.parse(b.at) - Date.parse(a.at));
}

export function audit(db: MockDb, entity: string, entity_id: number, action: string, actor: string | null, before: Record<string, unknown> | null, after: Record<string, unknown> | null, reason: string | null, at = db.now): void {
  db.audit.push({ id: db.seq.audit++, entity, entity_id, action, actor, before, after, reason, at: iso(at) });
}

export { median };
