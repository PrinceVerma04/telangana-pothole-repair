# Frozen table contract (WP0) — implemented by `server/app/db/models.py` (WP4)

Names are fixed (PLAN 10 WP0). Types are written portably for **SQLite (demo) and PostgreSQL/PostGIS (production)**,
per DECISIONS D1: points are `lat`, `lon` (WGS84 float) + `h3_r12` (text); lines/polygons are `geom_wkt` (WKT text,
WGS84). On PostGIS a generated `geog` column + GIST index is added by the production migration only.
Spatial queries go through `geo/index.py` (shapely STRtree) so business code never writes spatial SQL.

Conventions: every table has `id INTEGER PK` unless noted; timestamps are timezone-aware UTC (`*_at`), displayed in IST.
`json` = SQLite JSON text / Postgres JSONB. `enum(...)` = string column with CHECK. FK names `<table>_id`.

| table | key columns |
|---|---|
| `agency` | code UNIQUE, name_en, name_te, name_hi, kind enum(ulb,state,authority,central,cantonment) |
| `admin_unit` | agency_id, kind enum(zone,circle,ward,division), code, name_en, name_te, name_hi, parent_id, geom_wkt, valid_from, valid_to, official bool |
| `user` | username UNIQUE, phone, password_hash (argon2), role enum(Role), agency_id, admin_unit_id, contractor_id, active |
| `device` | device_id UNIQUE (text), agency_id, kind enum(phone,jetson,oakd,zed,sim), vehicle_reg, calib_cert_id, token_hash, last_seen_at |
| `calib_cert` | calib_id UNIQUE, device_id, json (measure.types.CalibCert), certified bool, issued_at |
| `trip` | trip_uuid UNIQUE, device_id, gtfs_route_id, started_at, ended_at, raw_wkt, matched_wkt, km, closed bool |
| `trip_fix` | trip_id, ts, lat, lon, accuracy_m, speed_mps, heading_deg, az_rms, cam_active, way_id, direction (filled by passes job). Index (trip_id, ts). Postgres: monthly partitions, BRIN(ts) |
| `observation` | client_uuid UNIQUE, device_id, trip_id, seq, captured_at, mono_ms, source enum(Source), calib_id, model_version, lat_raw, lon_raw, gps_accuracy_m, speed_mps, heading_deg, lat_ground, lon_ground, pos_sigma_m, way_id, direction, chainage_m, lane_offset_m, frame_key, imu_key, az_peak_g, az_rms_1s, status enum(pending,verified,rejected), reject_reason, payload json. Postgres: monthly partitions |
| `detection` | observation_id, track_id, cls, conf, bbox json[4], polygon json, verifier_cls, verifier_conf, verified bool, mask_key |
| `measurement` | detection_id, all `measure.types.FrameMeasurement` fields as columns (extras json), depth_map_key, imu_corroborated, imu_depth_mm |
| `pass_measurement` | trip_id, pothole_id, all `PassMeasurement` fields, frames_used json |
| `pothole` | every `measure.types.PotholeFused` field (priority_breakdown json); `lat`,`lon`,`h3_r12`; index (way_id, direction, lane_band), (status), (priority) |
| `pothole_observation` | pothole_id, observation_id, detection_id — PK (pothole_id, detection_id) |
| `pothole_event` | pothole_id, ts, from_status, to_status, reason, actor_id |
| `road_segment` | osm_way_id, seg_idx, name, name_te, highway, ref, lanes, oneway, lit, surface enum(BT,CC,unknown), geom_wkt, length_m, bearing_deg, owner_agency_id, contract_id, admin_unit_id, gtfs_trips_per_day, junction_start bool, junction_end bool |
| `way_pass` | way_id, direction, day, n_passes, n_cam_active — UNIQUE (way_id, direction, day) |
| `poi` | kind enum(school,hospital,clinic,bus_stop,metro_station,marketplace), name, lat, lon, source enum(osm,gtfs,manual) |
| `contractor` | name, gstin, phone, agency_id |
| `contract` | code UNIQUE, package, agency_id, contractor_id, geom_wkt, start_date, end_date, dlp_months, sla_rules json |
| `priority_config` | agency_id, version, weights json (full configs/priority.default.yaml), active bool |
| `rate_card` | agency_id, fy, version, items json (configs/rate_card.*.yaml) |
| `repair_plan` | pothole_id, json (repair.types.RepairPlan), spec_code, est_cost_inr, bags, rate_card_version, created_at, superseded bool |
| `work_order` | status enum(WorkOrderStatus), agency_id, contract_id, contractor_id, crew, run_id, issued_at, sla_due, escalation_level, est_bags, est_cost_inr, bags_used, bag_lot_no, is_rework, parent_work_order_id, invoice_ref |
| `work_order_pothole` | work_order_id, pothole_id, repair_plan_id |
| `repair_run` | circle_id, run_date, totals json, route json, crew_hours_cap, generated_at — UNIQUE (circle_id, run_date) |
| `repair_event` | work_order_id, kind enum(before_photo,after_photo,check_in,bags,note), photo_key, lat, lon, at, actor_id, data json |
| `verification` | work_order_id, pothole_id, clean_passes, distinct_trips, window_start, window_end, qc_sampled, qc_passed, outcome enum(pending,verified_closed,qc_failed) |
| `dlp_event` | pothole_id, work_order_id, contract_id, detected_at, dlp_until, evidence json (before/after/redetect keys, measurements, pass log), rework_work_order_id |
| `complaint` | ticket UNIQUE, source enum(web,whatsapp,mycure_csv), lat, lon, text, lang, photo_key, pothole_id, status, created_at |
| `jurisdiction_queue` | pothole_id, road_segment_id, reason, candidates json, created_at, resolved_at, resolved_by, resolution json |
| `notification` | channel enum(email,sms,whatsapp,inapp), to_role, to_user_id, subject, body, entity, entity_id, created_at, sent bool |
| `model_registry` | name, version, task, format, sha256, metrics json, signature, path, created_at, active bool |
| `audit_log` | entity, entity_id, action, actor_id, before json, after json, reason, lat, lon, at — append-only |
| `job` | kind, payload json, status enum(queued,running,done,failed), attempts, error, created_at, started_at, finished_at (DECISIONS D2 production poller) |
