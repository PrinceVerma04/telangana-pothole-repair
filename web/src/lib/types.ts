/**
 * TypeScript mirror of the frozen pydantic contracts:
 *   measure/types.py, repair/types.py, server/app/schemas/{api,states}.py
 * Field names are identical to the pydantic models (JSON mode: datetimes are ISO strings,
 * enums are their string values, tuples are arrays). Units: depth mm, area m2, volume litres,
 * lengths m, money INR, timestamps UTC (displayed in IST).
 *
 * Kept in sync by hand. Checked against the live server's /v1/openapi.json on 2026-11-13 (WP9 leftovers):
 * every DTO field matches; the differences were in envelopes and limits, handled in src/lib/api.ts.
 */

export type ISODateTime = string;
export type ISODate = string;
type Dict = Record<string, unknown>;

// ------------------------------------------------------------------ enums (measure/types.py)
export const SEVERITIES = ['small', 'medium', 'large', 'unknown_water'] as const;
export type Severity = (typeof SEVERITIES)[number];

export const METHODS = ['mono', 'mono+imu', 'stereo', 'lidar_phone', 'manual_gauge', 'citizen', 'synthetic', 'mono_ground'] as const;
export type Method = (typeof METHODS)[number];

export type DepthQuality = 'ok' | 'unknown_water' | 'rejected';
export type SurfaceModel = 'plane' | 'quadratic';
export type RoadSurface = 'BT' | 'CC' | 'unknown';

export const POTHOLE_STATUSES = [
  'candidate', 'confirmed', 'planned', 'work_ordered', 'acknowledged', 'in_progress', 'repaired_claimed',
  'verification_pending', 'verified_closed', 'qc_failed', 'reopened', 'recurrence_watch', 'rejected', 'archived',
] as const;
export type PotholeStatus = (typeof POTHOLE_STATUSES)[number];

export const DEFECT_CAUSES = [
  'unknown', 'waterlogging', 'utility_cut', 'base_failure', 'edge_drop', 'drainage', 'poor_patch', 'heavy_axle',
] as const;
export type DefectCause = (typeof DEFECT_CAUSES)[number];

// ------------------------------------------------------------------ enums (states.py)
export const WORK_ORDER_STATUSES = [
  'planned', 'issued', 'acknowledged', 'in_progress', 'repaired_claimed', 'verification_pending',
  'verified_closed', 'qc_failed', 'cancelled',
] as const;
export type WorkOrderStatus = (typeof WORK_ORDER_STATUSES)[number];

export const ROLES = ['citizen', 'driver_device', 'field_engineer', 'contractor', 'zonal_admin', 'super_admin'] as const;
export type Role = (typeof ROLES)[number];

// ------------------------------------------------------------------ enums (repair/types.py)
export const SPEC_CODES = ['IRC116_COLDMIX', 'IRCSP100_EMULSION', 'JET_PATCHER', 'INFRARED', 'HOT_MIX', 'CC_MORTAR'] as const;
export type SpecCode = (typeof SPEC_CODES)[number];

// ------------------------------------------------------------------ calibration
export interface DriftRecord { trip_id: string; scale_drift: number; pitch_drift?: number | null; at?: ISODateTime | null }
export interface TruthFit {
  k_sigma: number; g_imu?: number | null; n_truth: number; depth_mae_mm?: number | null;
  volume_mape?: number | null; fitted_at?: ISODateTime | null;
}
export interface CalibCert {
  calib_id: string; device_id: string; vehicle_kind: string;
  fx: number; fy: number; cx: number; cy: number; dist: number[]; img_w: number; img_h: number;
  reproj_rms_px?: number | null; h_true_m: number; pitch_deg: number; roll_deg: number; yaw_deg: number;
  lateral_offset_m: number; wheel_track_left_m: number; wheel_track_right_m: number; gnss_latency_s: number;
  known_object_err_pct?: number | null; calibrated_at?: ISODateTime | null; calibrated_by?: string | null;
  certified: boolean; drift_history: DriftRecord[]; truth_fit: TruthFit; notes?: string | null;
}

// ------------------------------------------------------------------ measurements
export interface MeasurementCore {
  depth_p95_mm?: number | null;
  depth_max_mm?: number | null;
  depth_mean_mm?: number | null;
  depth_sigma_mm?: number | null;
  area_m2: number;
  area_sigma: number;
  volume_l?: number | null;
  volume_sigma?: number | null;
  length_m: number;
  width_m: number;
  length_sigma_m?: number | null;
  width_sigma_m?: number | null;
  water_filled: boolean;
  depth_quality: DepthQuality;
  quality: number;
  method: Method;
  imu_corroborated?: boolean | null;
  imu_depth_mm?: number | null;
}

export interface FrameMeasurement extends MeasurementCore {
  range_m?: number | null;
  lane_offset_m?: number | null;
  ground_xz?: [number, number] | null;
  surface_model: SurfaceModel;
  plane_rms_mm?: number | null;
  ring_rms_mm?: number | null;
  plane_inlier_ratio?: number | null;
  scale_factor?: number | null;
  reject_reason?: string | null;
  depth_model_version: string;
  calib_id?: string | null;
  certified: boolean;
  mask_px: number;
  seq?: number | null;
  t_frame_ms?: number | null;
  extras: Dict;
}

export interface PassMeasurement extends MeasurementCore {
  n_frames: number;
  n_frames_total: number;
  trip_id?: string | null;
  pothole_id?: number | null;
  range_m?: number | null;
  lane_offset_m?: number | null;
  ground_xz?: [number, number] | null;
  scale_factor?: number | null;
  depth_model_version: string;
  certified: boolean;
  frames_used: number[];
  reject_reason?: string | null;
}

export interface PotholeFused extends MeasurementCore {
  id?: number | null;
  agency_id?: number | null;
  admin_unit_id?: number | null;
  road_segment_id?: number | null;
  way_id?: number | null;
  direction?: number | null;
  lane_band?: number | null;
  lane_offset_m?: number | null;
  chainage_m?: number | null;
  lat?: number | null;
  lon?: number | null;
  h3_r12?: string | null;
  status: PotholeStatus;
  severity?: Severity | null;
  surface: RoadSurface;
  water_filled_ratio: number;
  first_seen?: ISODateTime | null;
  last_seen?: ISODateTime | null;
  passes_seen: number;
  passes_total: number;
  trips_seen: number;
  complaint_count: number;
  imu_corroborations: number;
  priority?: number | null;
  priority_breakdown: Dict;
  contract_id?: number | null;
  contractor_id?: number | null;
  best_method?: Method | null;
  certified_measurement: boolean;
  defect_cause: DefectCause;
  reopen_count_12m: number;
  root_cause_flag: boolean;
  remeasure_after?: ISODateTime | null;
  junction_dist_m?: number | null;
  lit?: boolean | null;
}

// ------------------------------------------------------------------ repair plan
export interface PlanContext {
  cluster_n_within_300m?: number;
  raining_now?: boolean;
  dry_forecast_48h?: boolean;
  infrared_unit_in_zone?: boolean;
  hot_mix_plant_km?: number | null;
  in_monsoon_window?: boolean;
  agency_pref?: SpecCode;
  bag_kg?: number | null;
  merged_pothole_ids?: number[];
}

export interface CostLine { item: string; qty: number; unit: string; rate_inr: number; amount_inr: number; provenance?: string | null }

export interface RepairPlan {
  pothole_id?: number | null;
  merged_pothole_ids: number[];
  l_cut_m: number; w_cut_m: number; a_cut_m2: number;
  d_design_mm: number; layered: boolean; d_patch_mm: number; d_crm_mm: number;
  v_patch_l: number; v_crm_l: number; v_cavity_l?: number | null; v_cuboid_l?: number | null;
  mix_kg: number; mix_kg_sigma: number; bag_kg: number; bags: number; bags_low: number; bags_high: number;
  emulsion_kg: number; sand_kg: number; crm_t: number; labour_h: number;
  method: string; spec_code: SpecCode; est_cost_inr: number; cost_breakdown: CostLine[];
  baseline_cost_inr?: number | null; rate_card_version: string;
  pending_remeasure: boolean; weather_blocked: boolean; notes: string[]; refs: string[]; extras: Dict;
}

// ------------------------------------------------------------------ auth
export interface TokenRequest { phone?: string | null; otp?: string | null; username?: string | null; password?: string | null }
export interface OtpRequest { phone: string }
export interface TokenResponse {
  access_token: string; token_type: 'bearer'; role: Role; agency_id?: number | null;
  contractor_id?: number | null; expires_at: ISODateTime;
}

// ------------------------------------------------------------------ potholes
export interface MeasurementOut extends FrameMeasurement {
  id: number; detection_id?: number | null; observation_id?: number | null;
  frame_url?: string | null; mask_url?: string | null; depth_map_url?: string | null;
}

export interface PotholeEventOut {
  ts: ISODateTime; from_status?: PotholeStatus | null; to_status: PotholeStatus; reason?: string | null; actor?: string | null;
}

export interface PotholeListItem {
  id: number; lat: number; lon: number; status: PotholeStatus; severity?: Severity | null; priority?: number | null;
  depth_p95_mm?: number | null; depth_sigma_mm?: number | null; area_m2: number; volume_l?: number | null;
  water_filled: boolean; road_name?: string | null; highway?: string | null; agency_code?: string | null;
  admin_unit?: string | null; passes_seen: number; passes_total: number; last_seen?: ISODateTime | null;
  thumb_url?: string | null; best_method?: Method | null; certified_measurement: boolean;
}

export interface PotholePage { items: PotholeListItem[]; total: number; limit: number; offset: number }

export interface PotholeDetail {
  pothole: PotholeFused;
  road?: { osm_way_id?: number; name?: string; highway?: string; lanes?: number | null; surface?: string; owner?: string } | null;
  measurements: MeasurementOut[];
  passes: PassMeasurement[];
  plan?: RepairPlan | null;
  events: PotholeEventOut[];
  work_orders: WorkOrderOut[];
  nearby_pois: Dict[];
}

export interface RejectRequest { reason: string; cause?: DefectCause | null }

export interface SegmentRating {
  road_segment_id: number; osm_way_id: number; bin_start_m: number; bin_end_m: number;
  pothole_area_pct: number; rating: 'Good' | 'Fair' | 'Poor';
  geometry: { type: 'LineString'; coordinates: [number, number][] } | Dict;
}

// ------------------------------------------------------------------ plans / WOs / runs
export interface PlanRequest { context?: PlanContext | null }

export interface WorkOrderCreate {
  pothole_ids: number[]; contract_id?: number | null; contractor_id?: number | null; crew?: string | null;
  sla_due?: ISODateTime | null; run_id?: number | null; notes?: string | null;
}

export interface WorkOrderTransition {
  to: WorkOrderStatus; reason?: string | null; lat?: number | null; lon?: number | null;
  bags_used?: number | null; bag_lot_no?: string | null;
}

export interface WorkOrderPhoto { kind: 'before' | 'after'; url: string; at?: ISODateTime; lat?: number | null; lon?: number | null }

export interface WorkOrderOut {
  id: number; status: WorkOrderStatus; pothole_ids: number[]; agency_id?: number | null; contract_id?: number | null;
  contractor_id?: number | null; contractor_name?: string | null; crew?: string | null; issued_at?: ISODateTime | null;
  sla_due?: ISODateTime | null; sla_breached: boolean; escalation_level?: string | null; est_bags?: number | null;
  est_cost_inr?: number | null; bags_used?: number | null; bag_variance_flag: boolean; is_rework: boolean;
  parent_work_order_id?: number | null; photos: WorkOrderPhoto[] | Dict[]; run_id?: number | null;
}

export interface RunStop { pothole_id: number; lat: number; lon: number; hop_m: number; [k: string]: unknown }

export interface RepairRunOut {
  id: number; circle_id: number; circle_name?: string | null; run_date: ISODate; work_order_ids: number[];
  route: RunStop[];
  totals: Partial<Record<'bags' | 'crm_t' | 'emulsion_kg' | 'sand_kg' | 'crew_h' | 'cost_inr', number>> & Record<string, number>;
  crew_hours_cap: number; generated_at: ISODateTime;
}

export interface ContractorScorecard {
  contractor_id: number; name: string; work_orders: number; sla_met_pct?: number | null; verified_closed: number;
  qc_failed: number; dlp_events: number; reopen_rate?: number | null; median_days_to_fix?: number | null;
  survival: { days: number; survival: number }[];
}

export interface KpiOut {
  admin_unit_id?: number | null; admin_unit?: string | null; window_days: number; open_potholes: number;
  confirmed_new: number; repaired_verified: number; median_hours_to_repair?: number | null; sla_breaches: number;
  reopened: number; bus_km_scanned: number; est_cost_inr: number; baseline_cost_inr: number;
  by_severity: Record<string, number>; by_status: Record<string, number>;
}

// ------------------------------------------------------------------ admin / citizen
export interface JurisdictionItem {
  id: number; pothole_id?: number | null; road_segment_id?: number | null; reason: string; candidates: string[];
  created_at: ISODateTime; resolved: boolean;
}
export interface JurisdictionResolve { agency_code: string; contract_id?: number | null; apply_to_segment?: boolean }

export interface AuditEntry {
  id: number; entity: string; entity_id: number; action: string; actor?: string | null;
  before?: Dict | null; after?: Dict | null; reason?: string | null; at: ISODateTime;
}

export interface CitizenReportOut {
  ticket: string; status: string; pothole_id?: number | null; created_at: ISODateTime; lat: number; lon: number;
}

export interface Health {
  status: 'ok' | 'degraded'; db: boolean; storage: boolean; gpu?: string | null; seg_weights?: string | null;
  depth_backend?: string | null; demo_mode: boolean; version: string;
}

// ------------------------------------------------------------------ NOT in api.py (shapes proposed to WP4/WP8)
/**
 * Ward league row as the UI uses it. The live server's GET /v1/public/wards returns a different envelope
 * (PublicWardsResponse below); HttpApi.publicWards maps it with `wardStatsFromPublic` (src/lib/api.ts).
 * Metrics the live endpoint does not compute (per-km, SLA %, median hours, km scanned, cost) stay null.
 */
export interface WardStat {
  admin_unit_id: number;
  code?: string | null;
  name_en: string; name_te?: string | null; name_hi?: string | null;
  kind?: string; zone?: string | null; parent_id?: number | null; official?: boolean;
  open_potholes: number; large_open?: number; repaired_verified_30d: number;
  median_hours_to_repair?: number | null; sla_met_pct?: number | null; km_scanned?: number | null;
  road_km?: number | null; potholes_per_km?: number | null; est_cost_inr?: number | null;
  repair_ratio?: number | null; rank?: number | null;
}

export const ADMIN_UNIT_KINDS = ['zone', 'circle', 'ward', 'division'] as const;
export type AdminUnitKind = (typeof ADMIN_UNIT_KINDS)[number];

/** An admin unit (zone / circle / ward) as listed by GET /v1/public/wards?kind=<kind> (no DTO in api.py). */
export interface AdminUnitBrief {
  id: number; code?: string | null; kind: AdminUnitKind;
  name_en: string; name_te?: string | null; name_hi?: string | null;
  parent_id?: number | null; official?: boolean;
}

/** One row of the live GET /v1/public/wards (server/app/routers/public.py). `repaired_<window>d` is dynamic. */
export interface PublicWardsRow extends AdminUnitBrief {
  open: number; open_large: number; repair_ratio: number | null; rank: number;
  zone?: string | null; // not sent today (shared-file request filed); used when present
  [k: string]: unknown;
}

/** Live GET /v1/public/wards (without geometry). */
export interface PublicWardsResponse {
  kind: AdminUnitKind; window_days: number; generated_at: ISODateTime; note?: string; items: PublicWardsRow[];
}

/** Live GET/PUT /v1/admin/priority-config and /rate-card wrap the document in a version envelope. */
export interface PriorityConfigEnvelope {
  id?: number | null; agency?: string | null; inherited?: boolean; version?: number; active?: boolean;
  created_at?: ISODateTime | null; config: PriorityConfig;
}
export interface RateCardEnvelope {
  id?: number | null; agency?: string | null; fy?: string; version?: number; version_label?: string; active?: boolean;
  created_at?: ISODateTime | null; card: RateCard;
}

/** configs/priority.default.yaml shape (GET/PUT /v1/admin/priority-config). */
export interface PriorityConfig {
  version: number;
  weights: { S: number; E: number; V: number; T: number; M: number };
  age_sla?: { sla_hours: Record<string, number>; [k: string]: unknown };
  monsoon?: { base: number; in_window: number; heavy_rain: number; [k: string]: unknown };
  [k: string]: unknown;
}

export interface RateItem { rate: number; unit: string; provenance?: string }
/** configs/rate_card.*.yaml shape (GET/PUT /v1/admin/rate-card). */
export interface RateCard {
  agency: string; fy: string; version: number; currency: string;
  items: Record<string, RateItem>;
  baseline: { cost_per_pothole_inr: number; provenance?: string };
}

/** Minimal contractor record; no list endpoint in the contract yet (contract change request). */
export interface ContractorBrief { id: number; name: string; package?: string | null; agency_code?: string | null }

export interface PotholeQuery {
  bbox?: [number, number, number, number];
  status?: PotholeStatus | PotholeStatus[];
  severity?: Severity | Severity[];
  admin_unit?: string;
  agency?: string;
  min_priority?: number;
  limit?: number;
  offset?: number;
}

export interface WorkOrderQuery { status?: WorkOrderStatus | WorkOrderStatus[]; contractor?: number; agency?: string }

export interface CitizenReportIn {
  lat: number; lon: number; text?: string; lang?: 'te' | 'hi' | 'en' | 'ur'; phone?: string;
  photo?: Blob | null; photo_name?: string;
}

export interface PhotoUpload { kind: 'before' | 'after'; photo: Blob; name?: string; lat?: number | null; lon?: number | null; at?: ISODateTime }
