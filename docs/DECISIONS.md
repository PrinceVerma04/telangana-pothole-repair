# Engineering decisions (deviations from docs/PLAN.md), 2026-11-12

Verified on the dev box before deciding:
- `.venv` = Python 3.11.18, torch 2.11.0+cu128 on RTX 5080 (16 GB), ultralytics 8.4.211, transformers 5.22.2, opencv 5.0.0, fastapi 0.162.5, onnxruntime-gpu 1.31.1 (CUDA + TensorRT providers), albumentations 2.0.8, osmnx 2.1.1, pyrosm 0.29.0, h3 4.5.0, shapely 2.2.0.
- **No sudo** on the dev box -> PostgreSQL/PostGIS/Redis/MinIO cannot be apt-installed.
- Kaggle credentials present (`~/.kaggle/kaggle.json`); no Roboflow key.
- HF checkpoints verified: `depth-anything/Depth-Anything-V2-Metric-Outdoor-{Small,Base}-hf` (works via `transformers.pipeline`, 0.23 s / 1280x720 frame on the 5080), `depth-anything/DA3METRIC-LARGE` (exists, needs the DA3 git package - optional backend), `facebook/sam2.1-hiera-small` (loads via `transformers.Sam2Model`), `Ultralytics/YOLO26` seg weights (`weights/yolo26{n,s}-seg.pt`), bootstrap detectors `weights/bootstrap_pothole_yolo26m.pt` (1 class `Pothole`) and `weights/bootstrap_pothole_yolov8s.pt` (RDD 5 classes, Apache-2.0).
- RDD2022 per-country S3 link returns 403; full 13 GB release downloading from Figshare into `data/raw/rdd2022/`. BBBike Hyderabad PBF is gone; Geofabrik `southern-zone-latest.osm.pbf` (560 MB) downloading into `data/raw/osm/`. TGSRTC GTFS (10.5 MB) downloading into `data/raw/gtfs_tgsrtc/`. Overpass API refuses plain curl (406) -> POIs come from the PBF via pyrosm, not Overpass.

## D1. Database: SQLite for demo, PostgreSQL/PostGIS for production, same SQLAlchemy models
Spatial columns are stored as `lat`, `lon` (WGS84 floats), `h3_r12` (string) and WKT/GeoJSON text for lines/polygons. Spatial queries (nearest-way, within-radius, point-in-polygon) are done in Python with `shapely.STRtree` indexes built from the DB at startup (`geo/index.py`). On PostGIS the same functions can be swapped for SQL (`geo/mapmatch_pg.py` kept as the production adapter). Rationale: the demo must run with zero system services; 50k road segments fit in memory trivially.

## D2. Job queue: in-process pipeline, optional DB-backed worker. No Celery/Redis.
`server/app/pipeline.py:process_observation(obs_id)` runs verify -> depth -> measure -> locate -> match -> fuse -> cluster -> score -> plan synchronously. In `DEMO_MODE` the ingest endpoint runs it inline (or in a FastAPI BackgroundTask). For production a `job` table + `python -m server.workers.run` poller executes the same function; Celery can be added behind the same function later.

## D3. Object storage: filesystem (`data/store/`) by default, S3 adapter behind the same interface.

## D4. Depth backend default = Depth-Anything-V2-Metric-Outdoor-Base (transformers). DA3 and Metric3D-ONNX are optional backends. Height scale-lock (camera height) makes the absolute-scale convention non-critical.

## D5. Verifier = second detector (bootstrap YOLOv8s RDD model, Apache-2.0) instead of RF-DETR-Seg (rfdetr pins its own torch; not installed). Interface identical, so RF-DETR can be dropped in.

## D6. Datasets actually used for training: BharatPotHole (Indian dashcam boxes), eYantra Indian Roads (masks), RDD2022-India (boxes) when the Figshare download finishes. Boxes -> masks via SAM2.1 box prompts. No Roboflow (no key).

## D7. Basemap: online OpenFreeMap "liberty" style (no PMTiles tooling without sudo); roads/POIs/zones overlaid from our own GeoJSON/MVT endpoints.

## D8. Edge: Python simulator is fully working; Android (Kotlin) and Jetson runners are skeletons with documented build steps.

## D9. Training is never a demo gate: bootstrap weights power the pipeline until `make train` finishes; `SEG_WEIGHTS` env selects.

## D10. GPU is shared (found 2026-11-13)
Another local account (`ml`) runs two Streamlit apps holding ~12 GB of the 16 GB RTX 5080. We never touch them.
Consequences: depth inference (DA-V2 Base, ~1.5 GB) and SAM2 pseudo-labelling fit in the ~3 GB left; YOLO26 training
uses `batch` sized to free VRAM (`ml/train_seg.py` picks batch from `torch.cuda.mem_get_info`), or waits until the GPU
is free. Training remains non-blocking for the demo (D9).

## D11. WP0 status (2026-11-13)
The WP0 workflow from 2026-11-12 died with the session; WP0 was completed by hand: `pyproject.toml`, `Makefile`,
`.env.example`, `LOCKFILE.md`, `configs/*`, `configs/cameras/*.json`, `edge/schema/*` + examples, `repair/types.py`,
`server/app/schemas/{api,states}.py`, `server/app/db/tables.md`, `docs/CONTRACTS.md`, `tests/test_schema.py` (14 pass).
All datasets are downloaded intact in `data/raw/` (RDD2022 full 9 GB zip, eYantra, BharatPotHole, TGSRTC GTFS, OSM southern zone).

## D12. Windows laptop is the dev machine from 2026-11-13; its GPU is NOT usable by torch -> CPU
Facts (checked 2026-11-13, see `LOCKFILE.md`): Windows 11 Pro build 10.0.26300 (native, PowerShell 5.1, no WSL),
Intel Core i7-10510U (4 cores / 8 threads, 1.8 GHz), 15.8 GB RAM, ~326 GB free on C:, NVIDIA GeForce **MX330, 2 GB
VRAM, compute capability 6.1 (Pascal)**, driver 573.91 (CUDA 12.8). Python 3.11.9 venv, torch 2.11.0+cu128, Node
24.16 / npm 11.13 (PLAN asked Node 22; the web build works on 24). Project at `C:\gunthameter` (no spaces).
- The cu128 torch wheel ships kernels for sm_75+ only, so `torch.cuda.is_available()` is True but every kernel fails
  ("no kernel image is available"). Old `pick_device()` only checked free VRAM (1.7 GB free > 1.0 GB asked by the
  detector) and would have picked CUDA and crashed. Fix: `measure.depth_backends.base.cuda_status()/cuda_usable()`
  checks the wheel's arch list against the device capability and runs a smoke kernel (cached); every device
  picker (`pick_device`, SAM2 masker, detector cache release, `/healthz`, lockfile, GPU test skip) uses it.
  `GM_DEVICE=cpu` forces CPU. Tests: `tests/test_device.py`.
- Consequence: everything runs on CPU on this laptop. Depth (DA-V2 Base) + roadseg ~5-20 s per image; `measure-demo`
  ~5 min; the API and tests are unaffected. torch was NOT reinstalled (CLAUDE.md). Even a cu126 wheel (which still
  has sm_61 kernels) would give only 2 GB VRAM: fine for inference, unusable for YOLO26-seg training -> WP3 training
  decision is part of the wave-2 plan.
- D10 (shared GPU) is Linux-box-only and no longer applies here.
- PowerShell 5.1 turns any stderr line of a native exe into a red `NativeCommandError` when redirected (`2>&1`);
  harmless warnings were removed (pytest-asyncio loop-scope option set in `pyproject.toml`). `tasks.ps1` swallows
  `-p`, `-ea`, `-ov` etc. as PowerShell common parameters: pass such pytest flags via `python -m pytest` directly.
- Machine-specific generated files rewritten for this machine: `LOCKFILE.md` (now records OS, CPU, RAM, GPU
  capability/VRAM and CUDA usability), `data/datasets/india_pothole_seg/data.yaml` `path:` (was the Linux box's
  absolute path; ultralytics `check_det_dataset` now validates 4,597/1,305/514 images, 7 classes).

## D13. eYantra "shallow" -> manhole is confirmed (WP1 open question, 2026-11-13)
Visual check on the laptop (12-crop montages per label): eYantra `shallow` polygons are manhole / drain covers and
concrete slabs; eYantra `pothole` polygons are mostly kerb / wall-base edges with only a few real potholes. The builder's
defaults stay: `--eyantra-shallow manhole`, `--eyantra-pothole exclude_image`. Consequence: class 1 `pothole_water`
has **0 training instances**; water-filled potholes are handled by the measurement flags (PLAN 4.10), not the detector.
No dataset rebuild needed.

## Wave 1 status (2026-11-13, Windows laptop)
Verified by running, on this machine:
- **Python tests**: `.\scripts\windows\tasks.ps1 test` -> **179 passed**, 4 deselected (gpu/data). The 146 from the Linux box,
  plus new: `tests/test_device.py` (4), `tests/test_measure_core.py` (10, the WP2 acceptance tests PLAN asked for, which were
  never written), and `priority/tests` (19), which existed but were not in `testpaths`. Fixed an order-dependent failure in
  `server/tests/test_api_smoke.py`: the `/measure` router now loads the core with `importlib.import_module`.
- **Web (WP9)**: `npm run build` OK, `npm test` **20 passed** (were: build failing, no tests). Files the stopped builder never wrote
  and that are now written: `src/i18n/{en,te,hi}.json` (≈500 keys each; Telugu default), `src/styles.css` (the whole
  stylesheet), `src/test/setup.ts`; new tests `src/i18n/i18n.test.ts` (every key used by the UI resolves in te/hi/en, placeholder
  parity, real translation) and `src/lib/logic.test.ts` (TS planner == Python worked example PLAN 7.10, priority, state machines,
  mock API, offline outbox off/on). Fixed: TS type errors (map style null, roads.json LonLat), and `ircSeverity` in the TS port
  diverging from `measure/irc.py` (water / unknown-depth cases). Tests live in `tsconfig.test.json` (node types); the app
  tsconfig excludes them. Not done: `npm run smoke` points at `scripts/smoke.mjs`, which does not exist; no Playwright offline
  test, no Lighthouse run.
- **measure-demo (WP2/WP12)**: `measure/demo.py` did not exist (so `make measure-demo` had never worked); now written. Run
  on CPU in 328 s -> `reports/measure_demo/` (README.md, summary.json, overlays, per-image JSON):
  - synthetic accuracy (noise-free, production code path): **16/16 cases pass**; max |volume error| 2.9 %, max |p95 error|
    0.6 mm; e.g. car, plane, 0.8 m x 50 mm cap @ 7 m: 12.63 L truth -> 12.73 L; bus, crown, 1.0 m x 70 mm @ 8 m:
    27.72 -> 27.69 L.
  - σ coverage, 200 noisy trials: 2σ coverage **1.00** for depth p95, volume, area, width (target ≥ 0.90). σ is
    **conservative**: z_rms 0.37 (depth), 0.25 (volume), meaning reported σ ≈ 3x the actual error under this noise model. To be
    recalibrated with the staged truth campaign (WP11).
  - k recovery: injected 6.0e-4, fitted 5.2e-4 (13 % error).
  - real images (5 picked: labelled potholes first; uncertified `phone_demo` calibration): 6 measurements, depth p95
    0-68 mm, quality 0.08-0.30 (all below the 0.6 dispatch threshold). Several areas/volumes are **physically implausible**
    (6.7 / 9.9 / 27.3 m²) because the dashcam/phone geometry does not match the demo calibration. The pipeline flags it
    itself (`scale_factor 0.26 outside [0.5, 2.0] -> wrong intrinsics/height`). These numbers show the pipeline runs end to
    end; they are not measurements. A waterlogged edge was not flagged `water_filled` (follow-up for WP2/WP11).
  - bootstrap detector (`bootstrap_pothole_yolo26m.pt`) is weak on Indian frames: most confidences < 0.25 on the 21 demo
    images; WP3 training matters.
- **DB + seed (WP4/WP10)**: `tasks.ps1 db` + `seed` on SQLite with the real geo files in 89 s: 454,450 road segments,
  87,616 unowned -> jurisdiction queue, 6,725 POIs, 216 admin units (11 zones / 56 circles / 149 wards), 7 CRMP contracts,
  7 agencies, 7 users, 6 devices, 4 calibration certs, 2 model-registry rows; DB file 137 MB. GTFS routes are used for
  `gtfs_trips_per_day` on segments only (no route table seeded).
- **API**: uvicorn on :8000. `/healthz` -> `{"status":"ok","db":true,"storage":true,"gpu":null,...}`;
  `POST /v1/auth/token` (admin/demo1234) OK; `/v1/potholes` -> `{"items":[],"total":0}` (correct: potholes come from the
  simulator, WP7, or `seed --sample-potholes N`); `/v1/public/wards` and `/v1/admin/jurisdiction-queue` return seeded data;
  33 OpenAPI paths. Anonymous reads of `/v1/potholes` are allowed with citizen scope (by design). The optional routers
  plans/work_orders/runs/contractors/kpi/exports/truth do not exist yet (WP8b). The server was later stopped by Claude Code
  under memory pressure (torch demo + agents running at the same time), not by an API error.
- Laptop fixes: D12 (GPU check), Nirmala UI fallback font for Telugu/Hindi in overlays, `data.yaml` path, `tasks.ps1 web-test`.
- Still open from wave 1: 5 pre-existing ruff B905 warnings in `measure/`; `scripts/workflows/wave1.js` still has Linux paths
  (superseded by the wave-2 workflow).

## D14. Laptop GPU enabled: torch 2.11.0+cu126 (user decision, 2026-11-13). Supersedes the "CPU only" part of D12.
The user chose to train/infer on the laptop's MX330 instead of CPU-only. The cu128 wheel has no sm_61 kernels, so torch and
torchvision were reinstalled as the **same versions, CUDA 12.6 build** (`pip install --no-deps --force-reinstall
torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126`); no other package changed.
`scripts/windows/setup.ps1` now defaults to `-Cuda cu126` (`-Cuda cu128` for Turing+ machines; `-Cpu` still works).
Verified: wheel archs include sm_61, `cuda_status()` usable, `pick_device` selects CUDA. Measured on this laptop:
**fp32 ≈ 0.12 TFLOPS, fp16 ≈ 0.10 TFLOPS, CPU ≈ 0.07 TFLOPS** (2048² matmul). The MX330 is power-limited here, so it is
only ~1.7x the CPU, and half precision gives no gain on Pascal. Consequences:
- `cuda_half_ok()` (compute ≥ 7.0) gates fp16/bf16: DA-V2 and SAM2 run fp32 on this GPU.
- 1.6 GB free VRAM: the detector (needs 1.0 GB) runs on the GPU; DA-V2 Base asks for 2.0 GB -> stays on CPU unless
  `min_free_gpu_gb` is lowered.
- WP3 training on this GPU: only yolo26n-seg, small batch, fp32 (`amp=False`), expect on the order of an hour per epoch on
  the 4.6k-image train split. A real-quality model still wants the RTX 5080 box; WP3 code must run on both.
- Rollback: the same pip command with `.../whl/cu128` (or `/cpu`).

## Wave 2 batch A status (2026-11-13, Windows laptop; integrator)
Built by three parallel builders (workflow `wf_71aab87a-18c`), then integrated. Everything below was verified by
running it on this laptop with `GM_DEVICE=cpu`.

**What was built**
- **WP5 geo** (`geo/types.py, locate.py, index.py, mapmatch.py, mapmatch_pg.py, cluster.py, passes.py,
  jurisdiction.py`, 43 tests in `geo/tests`). Pure library, no `server/` imports. Pinned interfaces implemented. Additive
  extras: `cluster.assign(sigma_gate=2.5, existing_sigma_m=4.0)` (cross-trip pairs join when d <= max(6 m, 2.5·σ_pair),
  because 6 m DBSCAN alone cannot merge 3 trips at GPS σ 8 m), `jurisdiction.resolve(fallback_admin_agency=False)`.
  Cluster robustness (builder, 200 seeds, 2-lane way): 194/200 single clusters at per-axis σ 8 m, 200/200 at σ 5.3 m.
- **WP6 workers** (`server/workers/{__init__,adapters,stages,ops,run}.py`). `install()` registers all 9 stages plus the
  trip-close and complaint hooks; ops ticks are sla_tick, verify_repairs, dlp_watch, daily_indent, kpi_rollup and
  archive_candidates; job poller `python -m server.workers.run [--once|--tick]`.
- **WP8b ops** (`server/app/services/{state_machine,sla,verification,dlp,runs,kpi,exports}.py`, routers plans,
  work_orders, runs, contractors, kpi, exports, truth). Services never commit (the caller commits, so audit rows commit
  with the change). OpenAPI now has 50 paths (was 33).

**Integration changes (this step)**
- `server/app/main.py`: `create_app()` calls `server.workers.install(settings, storage)`, so the DEMO_MODE inline ingest,
  trip close and complaints run the real stages. A failure in the worker layer is logged; the API still starts.
- **No silent fallbacks in production.** `server/workers` now always calls the real `geo.*` and
  `server.app.services.*`. They are looked up with `importlib` at call time so tests can still put fakes in
  `sys.modules`. A missing module is a stage error or a tick error, never a quiet downgrade to raw-GPS or
  nearest-pothole clustering. Removed: `optional_module`, the `pipeline.stage_cluster` fallback, the distance-only pass
  counting, the `TypeError` retry in jurisdiction, the `inspect.signature` probe for `redetections`, and the
  `lane_band` fallback. The fake geo and fake services live only in `server/tests/test_workers_support.py`. Their
  default is now the real geo (`GM_TEST_GEO=fake` opts in to the fakes).
- **One planning path.** The pipeline `plan` stage now calls `services.runs.plan_pothole`, the same function as
  `POST /potholes/{id}/plan`, and keeps its input-signature dedupe through a new `extras=` argument. New
  `services.runs.plan_context()` gives the API's default context the jet-patcher cluster count (confirmed potholes
  within 300 m, PLAN 7.9) as well. Before this change, API plans always used `cluster_n_within_300m=0`.
- **Clean passes counted per pass.** `ops.clean_passes` used to keep only one pass per trip, so ">= 3 clean passes from
  >= 2 trips" (CONTRACTS) actually needed 3 trips. It now keeps every `geo.passes` pass; a trip that loops past the
  pothole twice counts twice. The e2e test exercises exactly this case (2 trips, 3 passes).
- **Jurisdiction policy** (the open question from WP6; 87,616 of the 454,450 demo segments are unowned):
  `GM_JURISDICTION_POLICY=provisional` is the default. For an unowned road or an unmatched location, the pothole gets the
  agency of the admin unit it lies in, so that agency's staff see it on dashboards and queues, AND a
  `jurisdiction_queue` item is created ("... provisionally assigned to admin unit X's agency") for an admin to confirm or
  reassign. `queue` keeps the agency None, which is WP5's conservative default: only super_admin and the queue see the
  pothole. `admin_fallback` uses the admin-unit agency and creates no queue item. The legacy
  `GM_JURISDICTION_ADMIN_FALLBACK=1` still means `admin_fallback`. Reason: PLAN 4 wants unknown owners in the queue, but
  hiding 19 % of potholes from every zonal dashboard until someone resolves them would break the demo and the SLA clock.
- `/potholes/{id}` work orders now use `services.state_machine.work_order_out`, so `sla_breached` follows the same rule as
  `/work-orders` (the clock stops at the repair claim).
- `ops.daily_indent` treats `runs.NothingToSchedule` as info, not an error with a traceback.
- Shared files: `pyproject.toml` testpaths += `geo/tests`; `tasks.ps1` targets `worker`, `ops-tick`, `tick`; Makefile
  `worker`, `ops-tick`, `tick`. Small cleanups: `seed.py` no longer calls the deprecated `Result.tuples()`, and the 5
  wave-1 ruff B905 warnings in `measure/{cli,overlay,pothole_geom,truth}.py` are fixed with `strict=False`, which keeps
  the old behaviour.
- `docs/CONTRACTS.md`: the API table lists the read-only endpoints WP8b added (`GET /contractors`,
  `/work-orders/{id}`, `/runs`, `/exports/invoice-eligible`, `/truth/measurements`, `/truth/pairs`). No DTO or other
  contract file changed.

**Tests changed (the test was wrong, not the code)**
- `test_api_smoke::test_register_ingest_duplicate_list`: the pothole is no longer placed on the raw fix. The real locate
  stage puts it a few metres ahead (flat-road projection plus GNSS latency; measured about 6.5 m), so the assertion is now
  1-20 m from the fix instead of 1e-6 deg.
- `test_workers_pipeline::test_install_is_idempotent_and_reversible`: it now starts from `uninstall()`, because
  `create_app` installs the workers. The ops/acceptance tests use the real services instead of `hide_services` ("skipped").
  A new assertion checks that missing services are reported as tick errors.
- `test_workers_units`: `test_pipeline_falls_back_without_geo` is replaced by `test_missing_geo_is_an_error_not_a_fallback`
  (locate/match/cluster report `ModuleNotFoundError`, depth and measurement still run, the observation stays verified, no
  pothole is invented). New `test_jurisdiction_policy` (3 policies).

**Verified numbers**
- Full suite: `pytest -m "not gpu and not data"` → **253 passed, 4 deselected in 215 s**. Before batch A: 180; after the
  builders, before integration: 208 (without geo/tests). +43 geo, +1 policy test, +1 e2e.
- `ruff check geo server priority repair measure tests` → All checks passed.
- Web: `npm test` 20 passed; `npm run build` OK. No web change was needed.
- **E2E** `tests/e2e/test_ops_flow.py` (fixture geo DB, synthetic depth, TestClient, real geo and services) passes in
  **10.0 s**. Flow: 2 trips × 2 potholes → confirmed, scored (severity large, S/E/V/T/M breakdown) and planned → WO issued
  to the CRMP-P2 contractor → acknowledged → in_progress (before photo) → repaired_claimed (after photo, bags) → trip
  close: trip C, which loops twice, gives (2 passes, 1 trip) = still pending, then trip D gives 3 passes from 2 trips =
  verified_closed, invoice-eligible → re-detection → reopened + dlp_event (evidence package, rework WO drafted as planned)
  → observation on an unowned segment → queue item with a provisional agency. The 10 % engineer QC sample is switched off
  in this test only (WP8b tests cover it).
- **Live smoke** on `data/gunthameter.db` (uvicorn 127.0.0.1:8001, CPU; server stopped afterwards; only GET requests,
  nothing written): `/healthz` ok (db, storage, depth_backend da2_metric); all 7 optional routers mounted; admin login ok;
  `GET /v1/work-orders` → `[]`; `GET /v1/kpi/1?window=30` (zone GHMC-LBN "LB Nagar") → 200 with every KpiOut field, all
  zero; `GET /v1/exports/potholes.geojson` → empty FeatureCollection. These are correct: the demo DB has no potholes until
  WP7 runs. `/v1/contractors` lists 7. Response times were 15-270 ms.
- **RoadIndex on the real data**, Hyderabad-central bbox 78.44-78.52 E, 17.36-17.44 N (about 8.5 × 8.9 km), read-only:
  33,806 of 454,450 segments. SQL bbox query 5.3 s (cold cache, ORM rows with WKT; `ix_road_segment_bbox` exists) +
  SegmentRec 0.5 s + STRtree build **0.12 s**; RSS 72 → 169 MiB; `match_point` 0.28 ms/query. The worker tile cache
  (0.01° tiles + 300 m margin) cold-loads 1,704 segments for one observation in 0.24 s; warm 0.25 ms. WP5's whole-network
  measurement: 454k segments, build 1.8 s, 568 MiB RSS.

**Open issues**
- The `mapmatch_pg` SQL has never run against PostGIS; only scoring parity with a fake connection was checked.
- Map-matching and clustering are tested only on synthetic trips. Real-trip accuracy waits for WP7/pilot data. A constant
  GPS bias of more than 20 m for a whole trip next to a parallel road can still mis-match that trip (WP5: 7/200 seeds).
- The second-model verifier and the DA-V2 depth path inside the worker never ran in tests (that would load torch).
  Production depth on CPU is about 5-20 s per frame (D12/D14), so DEMO_MODE inline ingest of real frames is slow. Use
  `PIPELINE_INLINE=0` + `tasks.ps1 worker` for the WP7 fleet run.
- The demo DB has no potholes yet, so the live smoke only proves the endpoints and shapes. The scaled run (seed → API →
  `fleet.py` → state counts) is batch B's integrator job.
- The pipeline driver still logs a full traceback for an expected depth failure (unreadable frame); this is noisy but
  harmless.
- web `CIRCLES` uses mock ids 1..3. Live `POST /runs/generate?circle=` needs a real admin-unit id or code (WP8b note).

## Wave 2 batch B status (2026-11-14, Windows laptop; finished by hand after workflow `wf_6045d889-f84` was interrupted)
The workflow stopped mid-run on 2026-11-13. It had already written `ml/{segmenter,train_seg,eval,export,metrics,
common}.py`, `ml/augment/india.py`, the baseline `reports/model_eval.json`, `edge/sim/*` except `fleet.py` (incl.
`sim_server.py`), `scripts/{staged_pothole_truth,make_report}.py`, `data/truth/template.csv`, `web/scripts/smoke.mjs`
and `web/e2e/field-offline.spec.ts`. No smoke training weights were left (`runs/seg/_data` holds only the data yaml).
The rest was finished sequentially in one session (user choice: no parallel builders). Verified on this laptop, CPU.

**Built in this step**
- **WP7** `edge/sim/fleet.py`: buses × routes × days. Survey days (in the past) → repair day at the real server clock
  (plan + WO + contractor acknowledge / before photo / in_progress / after photo / repaired_claimed) → verification
  days (timestamped after the claim; repaired potholes suppressed) → DLP check trip (re-detects one verified repair).
  Also 2 IMU jolts, 1 Telugu citizen report, and a 1.5 km stretch of a busy GTFS route outside every admin circle
  driven twice (jurisdiction queue). `--in-process` runs `sim_server.build_app` through `TestClient`; `--strict` turns
  the acceptance check into the exit code.
  Two behaviours found in the first full run and fixed in the fleet, not the server:
  (a) roads with no CRMP contract leave the WO `planned` (no contractor) - correct server behaviour; the fleet (and
  `scripts/demo.py`) now assign a contractor round-robin, as an engineer would; (b) one of 3 verification passes of a
  repair was not counted (GNSS noise put the pass on a parallel way - the known WP5 limitation), so no repair closed;
  the fleet now runs up to `--extra-verify-rounds 2` more trips on routes whose repairs are still pending.
  Also `edge/protocol.md`, `edge/jetson/runner.py` (+ README; video + GPS CSV → keyframes → segmenter → upload, `--dry-run`),
  `edge/android/` (Kotlin skeleton: CameraX, fused location, KeyframeTrigger, JSON-lines outbox, OkHttp client with
  retries, WorkManager upload, StubSegmenter; **not compiled here - no Android SDK**), `tests/test_edge_sim.py` (7).
- **WP3** `ml/pseudo_label.py` (segmenter [+ verifier] → accepted YOLO-seg labels + review overlays/CSV + manifest with
  `status: needs_human_review`; never writes into `data/datasets`), `tests/test_ml_pseudo_label.py` (3). Not built:
  `train_rfdetr_seg.py` (D5: rfdetr not installed). **The long training was not run** (user decision: leave the
  command). Targets `train` / `train-resume` / `eval` / `export` / `pseudo-label` fixed in Makefile + tasks.ps1 (the
  old `train` passed arguments `ml.train_seg` no longer accepts).
- **WP11** docs `DEPTH_METHOD.md` (§4 formulas + the synthetic table, σ coverage, staged *rehearsal* table, real-image
  flags), `CALIBRATION_SOP.md` (te/en; Telugu marked as a working translation to be reviewed), `DEPLOYMENT_TSRTC.md`
  (20 buses / 3 depots, BOM with indicative prices, 24 V DC-DC, heat, SIM, depot Wi-Fi, server, schedule),
  `LICENCES.md` (with open decisions: Ultralytics AGPL, DA-V2 Metric Base card licence, eYantra GPL-2, BharatPotHole),
  `DEMO_SCRIPT.md` (7 min). `make_report.py`: the WP3 baseline context is printed with 4 decimals (0.0885 was printed
  as 0.088 by float rounding; its test expected 0.089) and `reports/REPORT.md` is regenerated - the old copy showed
  `targets.mask_iou.floor` 0.650 as a measured "meets", which the current code already refuses.
- **WP9** leftovers: `field-offline.spec.ts` picked "the first card with a file input"; in the mock data that is an
  in_progress WO (after-photo step), so the test claimed a repair instead of starting one. It now selects the card by
  its "Before photo" step and pins it by WO number (the label becomes "Retake" after the photo). App code unchanged.

**Verified numbers**
- `pytest -m "not gpu and not data"`: **276 passed**, 0 failed (253 after batch A; + WP11 script tests, + 7 edge,
  + 3 pseudo-label, + 1 WP12 e2e). `ruff check .` clean.
- Web: vitest 38 passed; `npm run build` OK; `npm run smoke` 15 ok, 1 skipped (live-API thumbnails); Playwright 3/3
  passed with `PW_CHANNEL=msedge`. The default channel (installed Google Chrome) hung for > 10 min on this laptop, and
  the bundled Chromium here is build 1243 while Playwright 1.64.1 wants 1248 (`npx playwright install chromium`).
  Lighthouse was not run.
- **Scaled fleet run** (`scripts/demo.py --exit`, API on a fresh copy of `data/gunthameter.db`, 454k segments,
  synthetic road-render depth, 3 buses, 10H/218/5K × 4.5 km + the unowned stretch, 2 days, 21 trips, 478 requests):
  candidate 9, confirmed 37, verification_pending 3, verified_closed 2, reopened 1 → 43 ever confirmed; WOs:
  verified_closed 3, verification_pending 3 (one is the deterministic 10 % QC sample), planned 1, 1 zero-pay rework;
  4 jurisdiction-queue items. **WP7 acceptance 4/4 PASS; fleet 714 s (< 15 min).** Records: `reports/fleet_run.json`,
  `reports/demo_summary.json`. The first full run (before the two fleet fixes) reached only 2/4 (no verified repair).

## WP12 status (2026-11-14)
- `scripts/demo.py` (one Python orchestrator for both OSes) + `scripts/run_demo.sh` (`make demo`) + `tasks.ps1 demo`:
  seeds `data/gunthameter.db` if missing → copies it to `data/demo/demo.db` (the seeded DB is never modified) → API via
  `sim_server.py` with a per-run random JWT secret (the `.env` placeholder is 23 bytes) → Vite dev server if
  `web/node_modules` exists → fleet → tomorrow's work (top-8 confirmed potholes get issued WOs; the `daily_indent` ops
  tick builds one run per circle, so the admin hierarchy is walked server-side) → per-zone KPIs → summary
  (`data/demo/summary.json`). `--exit` stops after the summary; otherwise it keeps API + web up until Ctrl-C.
- Measured: **demo 732 s from a clean copy (< 20 min)**; tomorrow's indent 8 WOs, bags 49-60 (range), runs Mehdipatnam
  45 bags / 6 stops, Khairatabad 7, RC Puram & Patancheru 3; 30-day KPIs: est. ₹63,922 vs baseline ₹2,81,048 over 2
  zones with activity.
- `tests/e2e/test_demo_pipeline.py` (marked slow, ~1-2.5 min): fixture roads + a generated 9-frame label bank, one bus
  on the fixture route window best inside the box; asserts confirmed → repaired → verified_closed → DLP reopen + rework.
- README quickstart rewritten (Windows and Linux columns). `.gitignore` += `data/demo/`, `data/tmp_wp7/`, `data/tmp_wp9/`.
- **Not verified:** the README quickstart by a fresh agent/person; `make demo` under bash/WSL (only the Windows path ran);
  the demo with `--depth settings` (real DA-V2; slow on CPU); the web UI against the live demo API by hand.

**Open after this step (not code)**
- Train the segmenter (RTX 5080 box: `make train PROFILE=rtx5080_s NAME=y26s_seg_india`; laptop: `tasks.ps1 train`,
  then `train-resume`), then `eval` → `reports/model_eval.json` → `report`. Acceptance (box mAP50 ≥ 0.70) is unmet.
- Staged foam-board session with a calibrated camera (`scripts/staged_pothole_truth.py`), certified calibration.
- Android build with the SDK; Lighthouse PWA audit; licence decisions in `docs/LICENCES.md`.

## D15. 4-class training dataset; manholes re-split by drive sequence (user decision, 2026-11-14)
The WP1 build declares the 7 vocabulary classes, but `pothole_water`, `speed_breaker` and `utility_cut` have **no
labelled examples** (eYantra "shallow" became manhole in D13; no source has speed breakers or utility cuts), and all
manholes sat in train (792) and val (622) with **none in test**, because the test split was RDD-India only and the
manholes come from eYantra.
- **Classes:** `data/derive_dataset.py` derives `data/datasets/india_pothole_seg4` from the build (hard links, the
  build is never modified) with only the classes that have train labels, renumbered in canonical order:
  0 pothole, 1 alligator_crack, 2 crack, 3 manhole. `configs/model/india_pothole_seg.yaml` now points at it; the
  7-class config is kept as `india_pothole_seg7.yaml` (old evals). **The vocabulary is unchanged** (`configs/
  classes.yaml`, edge schema, server water-filled logic, web): a 4-class model maps to it by name, and a future model
  with water/speed-breaker data plugs in without a contract change. Removing the classes system-wide was rejected: it
  would break the frozen edge protocol and the water-filled design for no training benefit.
- **Manholes:** they come from only 6 eYantra drive sequences (ey_1_007 228 frames, ey_1_005 226, ey_1_018 73,
  ey_2_006 32, ey_1_016 20, ey_1_012 11). Frames of one sequence are near-duplicates, so sequences move **whole**
  (no leakage). With 6 groups every assignment is tried and the one closest to 70/15/15 of the instances is kept:
  train ey_1_005 + ey_1_007 = 1,269 (90 %), val ey_1_018 = 76 (5 %), test ey_1_012 + ey_1_016 + ey_2_006 = 69 (5 %).
  70/15/15 is not reachable without cutting a sequence; the first greedy attempt (6 groups, largest-first) put
  685 of 1,414 into test and left 145 for training, and was replaced by the exact search.
- **Result:** train 4,465 images, val 933, test 1,018 (514 RDD-India + 504 eYantra frames of the 3 test sequences).
- **Evaluation:** label ids are mapped by the dataset's names (`ml.common.dataset_class_ids`); the box GT uses the
  original RDD labels per image where they exist, dataset labels otherwise; a new `rdd_india_subset` block scores the
  RDD-India images alone, and the WP3 acceptance uses it (PLAN: "box mAP50 on the RDD-India D40 test"). Verified with
  the bootstrap detector (`baseline_bootstrap_seg4`): RDD-India subset pothole AP50 **0.0885 = the old 514-image
  baseline**; over the whole new test split 0.0478 (its false positives on the eYantra frames).
- Training: `--batch` also takes a GPU-memory fraction (Ultralytics AutoBatch); profile `laptop_mx330_max` (batch
  0.90 → 3 on the MX330, 4 workers, RAM cache); `scripts/windows/train_full_gpu.ps1` / `tasks.ps1 train-max`
  (foreground, resumable); `tasks.ps1 classes` prints the class table.
