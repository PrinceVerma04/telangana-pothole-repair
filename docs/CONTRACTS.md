# Frozen contracts (WP0)

Any work package that needs to change one of these files updates this document in the same change.

| Contract | File | Consumers |
|---|---|---|
| Measurement types, units, frames of reference | `measure/types.py` | WP2, WP4, WP5, WP6, WP8 |
| Repair plan | `repair/types.py` | WP8, WP4, WP9 |
| API DTOs | `server/app/schemas/api.py` | WP4, WP6, WP7, WP8, WP9 |
| State machines + RBAC on WO transitions | `server/app/schemas/states.py` | WP4, WP5, WP8 |
| Edge wire protocol | `edge/schema/{observation,trip_fix}.schema.json` (+ `examples/`) | WP7, WP4, Android/Jetson |
| Tables | `server/app/db/tables.md` | WP4 (+ all readers) |
| Classes | `configs/classes.yaml` (vocabulary, 7 classes). Training datasets and models may use an **ordered subset**, renumbered 0..k-1 (D15: `configs/model/india_pothole_seg.yaml` = pothole, alligator_crack, crack, manhole); detections are mapped to the vocabulary **by name** (`ml.common.canonical_class`), never by index. `source_maps` (how each dataset's labels map into the vocabulary) may gain entries for new sources without changing the vocabulary (D16 added `pothole_seg_web`) | WP1, WP3, WP7 |
| IRC constants | `configs/irc_rules.yaml` | WP2 (`measure/irc.py`), WP8 |
| Priority weights | `configs/priority.default.yaml` | WP5 |
| Rate card | `configs/rate_card.ghmc_fy2026.yaml` | WP8 |
| Measurement parameters | `configs/measure.yaml` | WP2, WP6 |
| Camera certificates | `configs/cameras/*.json` (`CalibCert`) | WP2, WP6, WP7 |

## Units (never mixed)
depth **mm** · area **m²** · volume **litres** · length/width/range **m** · money **INR** · time **UTC in storage, IST in UI**.

## Pothole state machine (`POTHOLE_TRANSITIONS`)

```
candidate ──(≥2 passes / ≥2 trips | 1 pass + citizen ≤25 m | 1 pass + IMU & q≥0.6)──> confirmed
confirmed ──plan──> planned ──WO issued──> work_ordered ──> acknowledged ──> in_progress
in_progress ──> repaired_claimed ──> verification_pending ──(≥3 clean passes, ≥2 trips, ≤14 d, QC ok)──> verified_closed
verification_pending ──QC fail / re-detected──> qc_failed ──rework──> work_ordered
verified_closed ──> recurrence_watch (12 months) ──re-detect──> reopened ──> confirmed|planned  (+ dlp_event if in DLP)
any live state ──engineer FP──> rejected ;  candidate ──30 d no 2nd pass & ≥5 clean passes──> archived
```

## Work order state machine (`WORK_ORDER_TRANSITIONS`)
`planned → issued → acknowledged → in_progress → repaired_claimed → verification_pending → verified_closed | qc_failed`;
`cancelled` from planned/issued/acknowledged. `qc_failed` is terminal — a **rework** WO (`is_rework`, `parent_work_order_id`)
is created. Only `verified_closed` WOs are invoice/MB eligible. WO moves mirror the pothole status (`WO_TO_POTHOLE`).

## API (`/v1`, JWT bearer)
Roles: citizen, driver_device, field_engineer, contractor, zonal_admin, super_admin.

| Area | Endpoints |
|---|---|
| auth | `POST /auth/otp`, `POST /auth/token` |
| edge | `POST /edge/register`, `GET /edge/config` (ETag, signed), `POST /edge/presign`, `PUT /edge/upload/{key}` (FsStorage demo upload) |
| ingest | `POST /ingest/observations` (batch, idempotent by client_uuid), `POST /ingest/trip-fixes`, `POST /ingest/trip-close` |
| citizen | `POST /citizen/reports` (multipart), `GET /citizen/reports/{ticket}` |
| potholes | `GET /potholes?bbox&status&severity&admin_unit&agency&min_priority&limit&offset`, `GET /potholes/{id}`, `POST /potholes/{id}/reject`, `GET /tiles/potholes/{z}/{x}/{y}.mvt` (or `.geojson` in demo), `GET /segments/rating?bbox` |
| measure | `POST /measure` (image + calib → FrameMeasurement), `POST /truth/measurements`, `GET /devices/{id}/cert` |
| plans | `POST /potholes/{id}/plan`, `GET /plans/{id}` |
| work orders | `POST /work-orders`, `GET /work-orders?status&contractor&agency`, `PATCH /work-orders/{id}/transition`, `POST /work-orders/{id}/photos` |
| runs | `POST /runs/generate?circle&date`, `GET /runs/{id}`, `GET /runs/{id}/indent.csv` |
| reporting | `GET /contractors/{id}/scorecard`, `GET /kpi/{admin_unit}?window`, `GET /exports/measurement-book.csv`, `GET /exports/potholes.geojson` |
| read-only additions (wave 2 batch A, no DTO change) | `GET /contractors` (brief list `{id, name, package, agency_code}`, used by the web client), `GET /work-orders/{id}`, `GET /runs`, `GET /exports/invoice-eligible`, `GET /truth/measurements`, `GET /truth/pairs` |
| admin | `GET/PUT /admin/priority-config`, `GET/PUT /admin/rate-card`, `POST /admin/admin-units`, `GET /admin/jurisdiction-queue`, `POST /admin/jurisdiction-queue/{id}/resolve`, `GET /admin/audit?entity&id` |
| public | `GET /public/wards`, `GET /healthz` |

## Pipeline function boundaries (DECISIONS D2)
`server/app/pipeline.py:process_observation(obs_id)` = verify → depth → measure → locate → match → fuse → cluster → score → plan.
Pure-library signatures (WP2/WP5/WP8) are listed in PLAN §10 and must not import from `server/`.
