# GunthaMeter edge protocol (bus device ↔ API)

This describes how a vehicle device (Android phone on a TGSRTC bus, Jetson box, or the WP7 simulator) talks to the
GunthaMeter API. The wire format is frozen in `edge/schema/observation.schema.json` and
`edge/schema/trip_fix.schema.json` (examples in `edge/schema/examples/`). See `docs/CONTRACTS.md`; any change to a
schema goes together with that document. The Python reference client is `edge/sim/client.py`; the simulator
(`edge/sim/replay.py`, `edge/sim/fleet.py`) uses exactly this protocol.

All endpoints are under `/v1`. Bodies are JSON unless stated. Times on the wire are ISO-8601 **with an offset**
(devices send IST `+05:30`; the server stores UTC). Units: metres, m/s, degrees, g, pixels of the raw frame.

## 1. Device lifecycle

| Step | Call | Notes |
|---|---|---|
| Register | `POST /edge/register` `{device_id, kind, vehicle_reg?, agency_code, calib_id?}` → `{device_id, token}` | `kind` ∈ phone, jetson, oakd, zed, sim; `device_id` matches `[A-Za-z0-9_.-]{1,64}`. Open in DEMO_MODE (`OPEN_DEVICE_REGISTRATION`); otherwise a `zonal_admin`/`super_admin` JWT is required. Re-registering issues a new token (the old one stops working) and is audited. |
| Config | `GET /edge/config` with `If-None-Match` → 200 + `ETag` or 304 | Signed OTA config (`EdgeConfig`): model URL + sha256, `trigger_min_distance_m` (4 m), `trigger_min_speed_kmh` (5), `conf_threshold` (0.35), `max_keyframes_per_track` (3), `trip_fix_hz` (1), `upload_wifi_only`. The device verifies `signature_ed25519` over `json.dumps(config_without_signature, sort_keys=True, separators=(',',':'))` with the key from `GET /edge/config/pubkey` (pin it at install time) and refuses an unsigned or badly signed config. |
| Calibration | `calib_id` on every observation | Refers to a `CalibCert` (`configs/cameras/*.json` / `calib_cert` table). Demo certificates are `certified=false`: their measurements are prioritisation-grade, never Measurement-Book (billing) grade. |

The device token is a bearer token: `Authorization: Bearer <token>` on every call below.

## 2. On the bus (no network needed)

1. **Keyframe trigger.** The camera runs continuously; a keyframe is taken when the bus has moved
   `trigger_min_distance_m` since the last one **and** drives at ≥ `trigger_min_speed_kmh`. At most
   `max_keyframes_per_track` keyframes per tracked pothole.
2. **Detection.** The on-device segmenter (YOLO26n-seg LiteRT/TensorRT, `model_version` string) gives
   `{track_id, cls, conf, bbox, polygon}` per detection in raw frame pixels. Classes come from `configs/classes.yaml`
   (`pothole`, `pothole_water`, `alligator_crack`, `crack`, `manhole`, `speed_breaker`, `utility_cut`).
3. **Privacy.** Faces and number plates are blurred on the device before a frame is stored or uploaded
   (`edge/sim/frames.py:Redactor` is the reference hook).
4. **Observation.** One JSON object per keyframe with detections (`source: edge`; `sim` from the simulator, `van`
   from a survey van). `client_uuid` (UUID4) is fixed when the observation is created, so retries are idempotent.
   `seq` increases per trip; `mono_ms` is the device monotonic clock shared with the IMU windows.
5. **IMU jolt.** When the vertical acceleration spikes (wheel over a pothole) the device emits an observation with
   `source: jolt`, `frame: null`, `detections: []` and `imu {az_rms_1s, az_peak_g, window_key}` (the ±1 s window is
   uploaded as `.npz`). A jolt alone never creates a measured pothole: it only corroborates existence
   (CONTRACTS: 1 pass + IMU & quality ≥ 0.6 can confirm).
6. **Trip fixes.** 1 Hz `{ts, lat, lon, accuracy_m, speed_mps, heading_deg, az_rms, cam_active}`. `cam_active`
   matters: only passes with the camera on count as clean passes for repair verification.
7. **Outbox.** Observations and fixes wait in a durable queue (Room on Android, `Outbox` in the simulator) until
   the server has answered. Nothing is dropped on a network failure.

## 3. Upload (en route over LTE, or at the depot over Wi-Fi when `upload_wifi_only`)

Order matters: frames before the observations that reference them, fixes before the trip close.

| # | Call | Idempotency / limits |
|---|---|---|
| 1 | `POST /edge/presign` `{keys: [...≤50], content_type}` → `{urls: {key: url}}` | Keys: `frames/<device_id>/<trip_id>/<seq>.jpg`, `imu/<device_id>/<trip_id>/<seq>.npz` |
| 2 | `PUT <url>` with the bytes | Re-PUT of the same key is harmless. `MAX_UPLOAD_MB` per object (413 above). In the demo the URL points at `PUT /edge/upload/{key}` (filesystem storage); in production it is an S3 presigned URL. |
| 3 | `POST /ingest/observations` `{observations: [...≤200]}` → `{accepted, duplicates, observation_ids, rejected: [{client_uuid, error}]}` | Idempotent by `client_uuid` (a resend counts as `duplicates`). Schema errors are per item in `rejected`; the rest of the batch is accepted. 413 above 200 items. |
| 4 | `POST /ingest/trip-fixes` `{device_id, trip_id, gtfs_route_id?, fixes: [...≤600]}` | Idempotent by `(trip_id, ts)`. |
| 5 | `POST /ingest/trip-close` `{device_id, trip_id, ended_at, km?}` → `{hooks, already_closed, fixes}` | Builds the trip geometry and runs the trip-close hooks: bus passes over known potholes, repair verification (≥ 3 clean passes from ≥ 2 trips after the repair claim), deferred depth for frames that arrived late. Closing twice is harmless (`already_closed: true`). |

If an observation arrives before its frame, the server keeps it and measures it at trip close (`depth_deferred`).

### Retries
Retry on transport errors, HTTP 429 and 5xx with exponential backoff (0.5 s, 1 s, 2 s, 4 s), resending the **same**
payload: every call above is idempotent. Do not retry 4xx other than 429; log the body and keep the item for an
operator. State-changing staff calls (work-order transitions) are never retried blindly.

## 4. What the server does with it (for device developers)

Per observation (inline in DEMO_MODE, or via the DB job queue + `python -m server.workers.run`):
verify (second model, optional) → depth (DA-V2 metric; the simulator's `sim_server.py` swaps in a synthetic road
render) → measure (depth/area/volume with σ, method, quality) → locate (ground point ahead of the camera, GNSS
latency) → map-match (OSM segment, direction, lane band) → fuse → cluster (pothole entity) → IRC:82 severity +
priority → repair plan (bags as a low–high range). Confirmation rules and the full state machine are in
`docs/CONTRACTS.md`.

## 5. Citizen reports (not a device call, same pipeline)

`POST /citizen/reports` multipart `{lat, lon, text, lang (te|hi|en), photo?}` → `{ticket, status, pothole_id?}`.
A report within 25 m of a single-pass candidate confirms it. Tracking: `GET /citizen/reports/{ticket}`.

## 6. Reference implementations

| Target | Where | State |
|---|---|---|
| Python simulator | `edge/sim/` (`replay.py` one bus, `fleet.py` buses × routes × days) | Working; used by `make demo` / `tasks.ps1 demo` |
| Android (Kotlin, CameraX + LiteRT + Room + WorkManager) | `edge/android/` | Skeleton: builds with the Android SDK (`./gradlew assembleDebug`); camera, model and upload paths are stubs with the protocol wired |
| Jetson (Python, TensorRT/ONNX Runtime + GStreamer) | `edge/jetson/` | Stub: the same loop as the simulator with a camera source; needs JetPack |
