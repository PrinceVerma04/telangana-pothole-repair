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

## D16. v2 training data: BharatPotHole by drive, human-drawn web masks, test near-duplicate guard (2026-11-16)
Linux RTX 5080 box again (the project moved back from the laptop; `.venv` rebuilt: Python 3.11.18, torch
2.11.0+cu128, ultralytics 8.4.211). The 0.221 laptop model had only ~705 Indian pothole frames in train.
- **BharatPotHole licence verified:** Kaggle's dataset metadata (`api/v1/datasets/metadata`) lists **CC BY-SA 4.0**
  (the bundled Roboflow yaml says CC BY 4.0) - the same terms as RDD2022. It is now used for training.
- **Its own splits leak:** 111 of its 162 drives (source videos) have frames in more than one of its train/valid/test
  splits. `build_dataset.py --bharat-split drive` (default) moves whole drives: 80/10/10 of frames -> 5,597 / 742 / 735.
- **Human-drawn pothole masks from the web:** Kaggle `farzadnekouei/pothole-image-segmentation-dataset` (Apache-2.0;
  Roboflow origin CC BY 4.0), 720 + 60 images, 2,047 polygons (`data/download.py --webseg`, `--include-webseg`).
  train -> train; valid -> test (`ws_` subset): the first pothole masks scored against people, not SAM2.
- **Test near-duplicate guard:** any train/val image within 24/256 bits (dHash) of a test image is dropped, across
  sources (web pothole sets are often RDD remixes). The v2 build dropped 2 (bharat->bharat, webseg->webseg).
- **Comparability:** the 514 RDD-India test images are locked in `data/splits/rdd_india_test.lock.txt` and checked by
  `ml/pipeline.py`; the v2 build's RDD-India test equals the lock, and the laptop model scores exactly 0.2207 on it.
- **Result** `data/datasets/india_pothole_seg4_v2` (from `india_pothole_seg_v2`, splits in `data/splits/v2/`):
  train 8,911 images / 14,564 potholes (v1: 3,926), val 1,722 / 879 potholes (v1: 174), test 1,843 / 3,428
  (RDD-India 514 + BharatPotHole drives + `ws_` + eYantra manhole sequences). SAM2.1 outlined 4,043 new images at
  52 img/s (2.7 GB GPU). `configs/model/india_pothole_seg.yaml` points at v2; v1 is `india_pothole_seg4_v1.yaml`.
- `ml/eval.py` reports `source_subsets` (bharat, webseg_human_masks, eyantra) next to the unchanged `rdd_india_subset`
  acceptance block; `--limit` samples evenly over sources. `ml/export.py` parity uses the dataset's class mapping and
  the registry records the RDD-India acceptance AP.
- `configs/classes.yaml` gained the `pothole_seg_web` source map (vocabulary unchanged; CONTRACTS.md updated).

## D17. Training pipeline with the GPU used to ~95 % (user request 2026-11-16)
`ml/pipeline.py` (`make retrain`, `make retrain-smoke`): check -> data -> baseline -> train -> eval -> export -> report,
all in the foreground. Profile `rtx5080_gpu95`:
- **"95 % GPU" = 95 % of the memory that is free at start.** Ultralytics 8.4.211 AutoBatch measures
  `torch.cuda.mem_get_info`, so `batch: 0.95` already means that; 95 % of the whole card would not fit (the desktop
  holds 1.2-2 GB).
- **OOM:** Ultralytics halves the batch on CUDA OOM (first epoch only) and crashes later. `oom_policy: restart` sets
  its retry counter to the maximum at every epoch start, so an OOM exits with code 3 and the pipeline resumes from
  `last.pt` with a 10 % smaller batch (fresh process). Tested: forced batch 44 -> OOM -> restarted at 39 -> finished.
- **Model choice by measurement** (one full epoch each, v2 data, AMP, RAM cache, 16 workers):
  yolo26s-seg batch 33, peak 91.7 % of the card's memory, load while training median 87 % / mean 75 %, 0.9 min/epoch;
  **yolo26m-seg batch 18, peak 96.5 %, median 91 % / mean 86 %, 1.7 min/epoch.** More loader workers did not help
  (16 vs 28: 0.159 s/step both); the gap with s is Ultralytics' fixed per-step CPU work. cuDNN autotuner: 4 % faster
  steps but slow first epoch/validation -> off (option `--set cudnn_benchmark=true`). Default = m (more accurate, meets
  the 95 % goal); `rtx5080_gpu95_s` kept. Running at the absolute memory limit is slower per image (batch 39 on s:
  6.9 ms/img vs 4.8 ms/img at the 95 % batch 33).
- **Best epoch by the acceptance metric:** `weights/best_pothole.pt` = the epoch with the best validation pothole box
  AP50 (Ultralytics' best.pt follows its all-class box+mask fitness); eval and export use it.
- **Pretrained pothole models from the web** (stage baseline, v2 test): laptop y26n RDD-India 0.2207 / BharatPotHole
  0.035; bootstrap YOLO26m detector 0.088 / 0.021; Hugging Face `keremberke/yolov8s-pothole-segmentation` 0.001 /
  0.000. None transfers to Indian dashcam frames, so we train our own (from COCO-pretrained YOLO26 weights).
- **Third-party checkpoints are pickles:** before loading, `ml/pipeline.py` lists every import in the pickle
  (pickletools, nothing executed) against an allow-list (torch, ultralytics, numpy, collections, ...); `os.system`,
  `eval` and unresolvable imports are refused. `getattr` is allowed only in the exact
  `getattr(<allowed class>, "forward")` form old YOLOv8 heads contain (keremberke). Tests: tests/test_training_pipeline.py.

## D18. Clean-up and Linux-only (user decision 2026-11-16)
Deleted with the user's approval: root logs and the duplicate `pull.ps1`, `data/*.log`, `data/tmp_wp7` (300 MB
simulator DBs, recreated on demand) and `data/tmp_wp9`, `reports/*.log`, smoke-run configs, the stale Windows-path dHash
cache, **the raw download archives** (RDD2022 full release 8.4 GB, eYantra 2.7 GB, BharatPotHole 0.8 GB, OSM southern
zone 0.5 GB; the TGSRTC GTFS zip is kept) and **`scripts/windows` + `scripts/transfer`** (the project runs on Linux).
`data/download.py` treats a source whose raw archive is gone but whose extracted copy is complete as OK (`--fetch`
downloads it again); `geo/osm.py` uses the existing Hyderabad crop when the 560 MB extract is gone.
Approved for later: the v1 datasets (`india_pothole_seg`, `india_pothole_seg4`), the v1 split files and the laptop run
`runs/seg/y26n_seg_india_laptop`, once a v2 model beats RDD-India pothole AP50 0.2207 and `.env` `SEG_WEIGHTS` no
longer points at the laptop weights: `make prune-v1` checks both and refuses otherwise.

### D17 addendum: launcher and further efficiency tests (2026-11-16)
`scripts/train_gpu.sh` (`make train-gpu`) starts or resumes the pipeline in the foreground: it lists the programs that
hold GPU memory (asks before starting when one holds > 300 MB, because every freed GiB goes to the batch), sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and logs to `runs/logs/`. One-epoch YOLO26m tests: expandable
segments changed nothing measurable (batch 18, 0.173 s/step, median load 91 %) and is kept against fragmentation near the
95 % limit (ml/pipeline.py sets it for every training step too); `torch.compile` could not run because Triton needs the
Python 3.11 headers (`python3.11-dev`, sudo needed, not installed). The CPU governor was already `performance` and the
GPU at its 360 W power limit, so the remaining lever is free GPU memory at start.

## D19. First full v2 run failed the gate; fine-tuning profile `rtx5080_gpu95_ft` (user decision 2026-11-17)
`make train-gpu` (profile `rtx5080_gpu95`, run `y26m_seg_india_v2`) ran 2026-11-16 17:29-19:22 and early-stopped at
epoch 67/150. Test (v2 split): **RDD-India pothole box AP50 0.1602** (laptop y26n 0.2207, floor 0.70), BharatPotHole
0.070 (laptop 0.035), human-mask `ws_` AP50 0.637 (0.338), `ws_` mask IoU 0.77 (0.72). Better on the new sources, worse
on the acceptance basis, so `SEG_WEIGHTS` stays on the laptop model and `make prune-v1` refuses.
- **Cause:** the profile had `optimizer: auto`; Ultralytics 8.4.211 then ignores `lr0` and picks MuSGD lr 0.01 when a
  run has more than 10,000 steps (74k here), AdamW at `0.002*5/(4+nc)` = 0.00125 otherwise (the laptop run). Validation
  pothole AP50 peaked at epoch 9 (0.164, = `best_pothole.pt`), all-class mAP50 jumped 0.13-0.19 between epochs and val
  cls loss stayed 4-8 against ~2.7 in training. Early stopping (patience 40 on Ultralytics' all-class fitness) ended the
  run while the linear LR was still ~55 % of lr0 and before mosaic closes (last 15 epochs), the phases that usually
  give the largest gain.
- **New profile `rtx5080_gpu95_ft`** (run `y26m_seg_india_v2ft`): same model, data, batch rule and augmentation;
  `optimizer: AdamW`, `lr0: 0.00125`, `cos_lr: true` (to lr0*0.01), 80 epochs, `close_mosaic: 15`, `patience: 0`
  (Ultralytics: 0 = no early stop). Estimate ~1.6 min/epoch at batch 18 -> ~2.2 h. `rtx5080_gpu95` is kept unchanged
  for comparability. Command: `make train-gpu PROFILE=rtx5080_gpu95_ft NAME=y26m_seg_india_v2ft`.

## D20. Display reserve: training no longer freezes the desktop (2026-11-17)
Run `y26m_seg_india_v2ft` (D19) reached epoch 16/80 (best validation pothole AP50 0.1719, epoch 15) when the whole
desktop froze at 12:40 IST and the machine was reset. RAM was not the cause (59 GiB, swap unused). Kernel log: the
NVIDIA driver returned `NV_ERR_NO_MEMORY` and gnome-shell could not get video memory (`Failed to allocate NVKMS memory
for GEM object`, `Failed to lock front buffer`). The monitor is on the RTX 5080 (HDMI, card1), the same GPU the D17 batch
fills to ~96.5 %, so any new desktop allocation (window, VS Code/Chrome GPU process, lock or login screen) failed. Same
cause as the 2026-11-16 VS Code freeze.
- **Fix in code** (`ml/train_seg.py`, `apply_display_reserve`): when a *connected* monitor hangs off the training GPU
  (Linux DRM sysfs, matched by PCI address), 3 GiB (`DISPLAY_RESERVE_GIB`) stay free: PyTorch is hard-capped with
  `torch.cuda.set_per_process_memory_fraction` and a fractional batch is scaled to the capped budget (0.95 of 14.3 GiB
  free -> 0.75, i.e. ~95 % of the 11.3 GiB cap). An overshoot is a normal CUDA OOM, answered by the D17 restart with a
  10 % smaller batch. Applied on resume too (Ultralytics allows `batch` on resume). Override:
  `GM_DISPLAY_RESERVE_GB` or `ml/pipeline.py --display-reserve-gb` (0 = off). Fewer than 2 GiB left -> refuses to start.
- **Cost:** batch ~18 -> ~14 on YOLO26m-seg, so somewhat slower epochs. D17's "95 %" now means 95 % of what is left
  after the reserve.
- **Recommended hardware fix (zero cost):** plug the monitor into the motherboard port (Ryzen 9950X3D Radeon iGPU,
  `0000:71:00.0`, already enabled; `prime-select` is `on-demand`). The reserve then switches itself off and training gets
  the whole card again.

## D21. Bus camera geometry augmentation: profile `rtx5080_gpu95_ft_geo` (user request 2026-11-17)
The phone is mounted at a different height and pitch on every bus (windscreen, ~2-3 m) while most training footage is
car/dashcam (~1.2-1.5 m). Up to D19 the geometry was scale 0.4, translate 0.1 (Ultralytics default), degrees 3,
perspective 0. New profile = `rtx5080_gpu95_ft` (D19 recipe) with scale 0.5, translate 0.15, degrees 5,
perspective 0.0005 (shear 0, flipud 0: not physical for a fixed forward camera). Checked on a preview of real v2 training
images: masks follow the warp, keystone and zoom-out look like plausible mount changes. Ultralytics does not let these
keys change on resume, so the user chose a **new run** `y26m_seg_india_v2ft_geo` over patching `y26m_seg_india_v2ft`
(stopped at epoch 16/80, kept for comparison). Augmentation only makes the detector robust; metric depth/area still uses
each bus's tape-measured height/pitch from its calibration certificate. Best evidence later: real frames from
bus-mounted cameras in the training and test sets.

## D22. Multi-user dataset review before the next training (user request 2026-11-17)
`scripts/dataset_review.py` (`make review`): FastAPI app on the office LAN (port 8770) over the training dataset
`india_pothole_seg4_v2` (12,476 images), decisions in SQLite `data/review/review.sqlite` (append-only history). User
choices: LAN access with name + shared access code (signed HttpOnly cookie; images served only to signed-in reviewers,
nothing leaves the PC); **split work** (batches of 20 held per reviewer, 30 min) with a **second look** for every reject
by a different reviewer (two rejects remove; a keep overrules, so no single person removes data); decisions Keep / Reject
/ Needs fix with reason tags and a note. **Test split stays locked:** test images can be reviewed, but their rejects go to
`exports/rejected_test_pending.txt` and are applied only with a deliberate new test lock (scores stay comparable until
then). Train/val rejects -> `exports/rejected_trainval.txt` -> `make derive-dataset-reviewed`
(`data/derive_dataset.py --exclude`, applied after the manhole re-split so no other image changes split). Needs-fix images
stay in the dataset and are listed in `exports/needs_fix.txt`. Rejected: a published claude.ai page (would upload the
licensed images), Label Studio/CVAT (heavy, results need conversion).

## D23. Model test page for own photos and videos (user request 2026-11-17)
`scripts/model_playground.py` (`make playground`, LAN port 8780, same access code and sign-in as D22). User choices:
own photos/videos, a web page, detection + measurement. Uploads are decoded before they are stored
(`data/playground/`, gitignored); videos are sampled (1 frame/s default, <= 150 frames) and the video file is not kept.
Detection runs once per frame and model at conf 0.05 and the page filters with its slider, so threshold changes and a
two-model comparison cost no GPU time. Models: `runs/seg/*/weights/best_pothole.pt`, the `.env` model and pothole
weights in `weights/` (COCO bases hidden), each through the D17 pickle scan; the list shows the RDD-India test pothole
AP50 (`metrics.rdd_india_subset`, the acceptance basis). "Measure" runs `measure.cli.run_measure` on the exact pothole
masks at or above the slider with a camera preset and optional per-photo height/pitch (written as an uncertified
`+user` cert); refusals (e.g. `plane_inlier_ratio<0.5` when the road plane does not fit the assumed camera) are shown
with their reason, and every result is labelled certified=false.

## D24. Keep only the best model; new model in use, verifier off (user request 2026-11-17)
User asked to keep only models with good accuracy, in the dashboards and in the project. Kept:
`runs/seg/y26m_seg_india_v2ft_geo` (RDD-India pothole box AP50 0.2494) with `best_pothole.pt`, `best.pt`, `last.pt`
(epoch0-70 snapshots deleted). Deleted with the user's approval (4.5 GB): runs `y26m_seg_india_v2` (0.1602),
`y26m_seg_india_v2ft` (unfinished), `y26n_seg_india_laptop` (0.2207), `weights/bootstrap_pothole_yolo26m.pt` (0.088) and
`weights/bootstrap_pothole_yolov8s.pt` (0.004, + its yaml). Their scores stay in `reports/model_eval.json`; LAPTOP_AP50
stays the bar a new model must beat. Consequences: `.env` / `server/app/core/config.py` / `ml.common.DEFAULT_SEG_WEIGHTS`
/ measure demo / registry seed point at the new model; `VERIFIER=off` (the verifier weights are gone; HANDOFF already
listed "verifier off" as a fix for the gate that would have dropped most detections); the pipeline's local baselines are
empty. COCO bases `weights/yolo26*-seg.pt` stay (training starts from them) and `weights/pretrained/` (web baseline) is
unchanged. The model test page lists only models scored on RDD-India and at least as good as the model in use.
Note: a shell that loaded the old `.env` into its environment (e.g. a VS Code terminal) still has the old values, which
override `.env`; open a new terminal.

## D25. Project renamed to RoadFix; plain-English summary on the model test page (user request 2026-11-17)
**Name:** "RoadFix — Pothole Detection & Maintenance" replaces GunthaMeter in everything people see: page titles, web
app (PWA name, header, i18n `app.name`; the Telugu brand name is dropped, user chose English only; Telugu UI text
stays), API title, Android app label, overlays, docs, reports, LaTeX sources, and the package names (`pyproject`
`roadfix`, re-registered editable with `--no-deps`; web `roadfix-web`). **Kept on purpose (legacy IDs)**: the folder
`~/gunthameter` (the `.venv` has it baked in), `data/gunthameter.db`, Android `applicationId`/package
`app.gunthameter.edge` (a new id would be a different app), the OTA config signing seed `gunthameter-edge-config-v1`
(deployed devices verify it), schema ids `gunthameter.*` and `https://gunthameter.local/...`, IndexedDB
`gunthameter-field` (offline data), dataset markers `.gunthameter_dataset`, `GUNTHAMETER_CONFIG_DIR`, and the file names
of the delivered report archives `reports/GunthaMeter_*.zip`.
**Summary (model test page, D23):** a short plain-English card: what was found (photo, or which frames of a video),
then per measured pothole depth / area / volume / IRC:82 severity and a repair estimate from `repair.planner.plan`
(cut size, cold-mix bags as a low-high range, labour hours, cost from the rate card). Sizes and repair estimates are
given only when the measurement passes the fusion gates of `configs/measure.yaml` (not rejected, quality >=
`min_quality_for_fusion` 0.2, scale lock inside `clip` [0.5, 2]); otherwise the summary says why it could not be
measured. Found while testing: on the demo dashcam frames with demo calibrations the scale lock was 0.26 and quality
0.12, which without the gate turned into a "14 m pothole" and 184-250 bags.

## D26. Project control dashboard: what is left to finish end to end (user request 2026-11-18)
`scripts/project_control.py` (`make control`, http://127.0.0.1:8790) shows every open step to finish RoadFix: 39 items
in 8 areas (detection model, export and bus model, real-camera measurement, the 2026-11-16 code-review issues,
production platform, bus and web apps, licences, demo and pilot). The list, gates and commands are in
`configs/roadmap.yaml`. Every refresh checks the repo: `reports/model_eval.json` metrics (AP50 floor 0.70, mask IoU,
ONNX delta), review progress read-only from `data/review/review.sqlite`, exported files, staged-truth and demo summaries,
file ages (dataset rebuilt after the review export, report after the eval), the last pytest run
(`.pytest_cache/.../lastfailed`), training in progress (`gm_live.json` < 120 s old), tools up (8765/8770/8780/8000/5173),
GPU and git. An automatically checked item whose dependencies are not done counts as **blocked** even when its own check
passes. Example: the 2026-11-17 eval is newer than the current dataset, but "retrain on reviewed data" stays blocked
until the review and rebuild are done. Blocking chains through dependencies. Items that cannot be checked automatically
(licences, code fixes, PostGIS, pilot) are ticked by a named person on the page. Ticks and notes go to
`docs/roadmap_state.json` (committed, so they travel with the repo) and are accepted only from loopback; automatic items
take notes only. User choices: a **local live** page (rejected: a hosted claude.ai snapshot, which goes stale) and
**read-only** (rejected: buttons that start jobs; training stays in the foreground, D17), so commands are shown with a
copy button. Standard library + PyYAML, no CDN. `--check` prints the same status in the terminal.

## D27. Depth model off: size-only severity from the pothole outline (user decision 2026-11-18)
User view: monocular depth on bus frames and detection are both hard; drop the depth pipeline and grade potholes
another way. The user first suggested **detector confidence as severity**. It was tested on the v2 test split (current
model, 3,428 labelled potholes) and rejected: Spearman(confidence, pothole area) = **+0.02 on RDD-India** (+0.16
BharatPotHole, +0.03 web close-ups). Confidence measures how clear the image is, not how big the pothole is, and its
scale shifts with every retrain. The model keeps using confidence only as the trust gate (conf < 0.5 dropped).
Chosen instead (user approved):
- **`depth.backend: "off"`** (`configs/measure.yaml`, `.env`/server default `DEPTH_BACKEND=off`): `measure_frames` takes
  the size-only path (`measure/ground_geom.py`). Each mask pixel ray meets the road plane known from the calibration
  (height, pitch, roll, yaw). Output: area, length, width (m), range and lane offset, with sigma from re-projecting at
  height +-`ground.height_sigma_m` (0.02 m), pitch +-`ground.pitch_sigma_deg` (0.75 deg) and the +-1 px outline.
  Depth, volume and depth-based water detection are None. Contract additions (CONTRACTS.md): `Method.mono_ground`,
  `DepthQuality.not_measured`. No depth model is loaded (server `depth_backend()` returns None; the CLI skips the
  SegFormer road mask). `/measure` now follows `DEPTH_BACKEND` like the workers (before, it only read measure.yaml).
- **Severity**: IRC:82 width class (Large > 500 mm, Medium 200-500 mm, else Small; `measure.irc.severity_size_only`),
  raised one level when a bus felt the jolt (`pothole.imu_corroborations >= 1`; `server/workers/stages.update_severity`).
  With measured depth, plain IRC:82 as before.
- **Repair**: depth unknown and not water -> bags low = the IRC:116 50 mm minimum patch, high = an assumed maximum per
  width class (`irc_rules.yaml repair.depth_unknown_assumed_max_mm`: small 50, medium 75, large 100 mm; NOT IRC values,
  engineer-editable), through the same layering rule (above 75 mm the extra depth is crusher-run, `extras.crm_t_high`).
  The note says the crew confirms depth on site. Water-filled potholes keep the PLAN 4.10 lower-bound rule.
- Model test page and web app show width / length / area and "Size only (no depth)"; the depth chart shows "Depth not
  measured".
- **Accuracy** (tests/test_size_only.py, rectangle drawn with an independent pinhole projection, demo phone 1.4 m /
  12 deg): width within 1-3 % at 4-10 m. Length and area read 3-15 % high at 6-10 m (the outline's edge pixels), inside
  the stated sigma.
  Real accuracy depends on the bus calibration certificate and a flat road; the foam-board session now checks width and
  area instead of depth.
- **Kept, not deleted**: all depth code, backends and tests. `DEPTH_BACKEND=da2_metric` brings depth back. Future
  option: severity classes trained into the detector, once reviewers have labelled severity.

## D28. Phone as the bus camera for the demo; SQLite CHECK-constraint refresh (user decision 2026-11-18)
The user asked what to provide for a demo, or for a simple Android app that does the bus camera's job. Chosen: a
**phone web page** (rejected: finishing the native `edge/android` skeleton, which has no JDK/SDK on this PC, a stub
detector and needs a trained and exported YOLO26n; rejected: video + GPS-logger import, which has error-prone time
sync and no IMU).
- `make capture` (`scripts/phone_capture.py` + `.html`): HTTPS on the LAN :8443 with a self-signed certificate for the
  PC's IP (openssl, `data/capture/tls/`; phones need HTTPS for camera, GPS and motion). Same access code and signed
  cookie as the review tool. The page: **Set up** (rear camera, tape height, tilt read from the gravity sensor with a
  level indicator, field of view default 67 deg) -> `configs/cameras/phonecap_<hash of values>.json` (certified=false;
  the id changes when any value changes, so the API never reuses a stale cached certificate). **Drive** (no network):
  keyframe every 4 m at >= 5 km/h with the odometer dead-reckoned between 1 Hz GPS fixes (50 ms tick), 1 Hz trip
  fixes with `cam_active`, jolts at a vertical peak >= 0.35 g and >= 15 km/h, screen wake lock, all in IndexedDB.
  **Upload** on office Wi-Fi, resumable, every call idempotent (frames by seq, fix and jolt batches by content).
  Trip ids are UUIDs (edge schema). Frames stay in `data/capture/` (git-ignored; faces and number plates, DPDP).
- `make capture-replay TRIP=<id> [API=...]` (`scripts/capture_replay.py`): the PC acts as the bus box. It runs
  SEG_WEIGHTS at the edge confidence 0.35 on every keyframe, links detections across consecutive keyframes (overlap,
  or a box moving down the image), keeps <= 3 keyframes per track (first, middle, nearest), and builds edge
  observations and jolt observations (schema-validated). It registers the device and uploads through
  `edge/sim/client.py` (frames, observations, fixes, trip close), then writes `replay.json`. `--dry-run` only detects.
- Tested end to end (2026-11-18): a phone-like trip from 60 consecutive BharatPotHole frames along GTFS 10H was
  replayed into a size-only API on a copy of `data/gunthameter.db`. Result: 1 track, 2 keyframes, size-only
  measurements 0.37 +- 0.02 m and 0.20 +- 0.01 m wide, a candidate pothole on OSM way 313353060, fused width 0.275 m,
  Medium, method `mono_ground`. The page itself was driven in Chrome with fake sensors (canvas camera, synthetic GPS at
  36 km/h, gravity at 9 deg): tilt read 9.0 deg, 24 keyframes over 0.11 km, 5/5 bumps as jolts, upload complete, and
  the replay was accepted with 0 schema rejections.
- **Found by that test, fixed:** (1) `EnumStr` columns carry CHECK constraints written at table creation, so every
  existing SQLite DB refused D27's `mono_ground` / `not_measured`. `Database.create_all` now calls
  `server/app/db/migrate.refresh_sqlite_checks`: tables whose CHECK constraints differ from the models are rebuilt with
  the SQLite 12-step procedure (foreign keys off, copy, rename, indexes, `foreign_key_check`) in one transaction,
  after a copy of the file to `<db dir>/backups/`. On the 137 MB DB copy it rebuilt pothole, measurement and
  pass_measurement, integrity_check ok. PostgreSQL still needs real migrations (D1-D3). (2) `edge/sim/sim_server.py
  --depth settings` hard-coded `da2_metric` when `DEPTH_BACKEND` was not in the environment; it now follows .env / the
  Settings default.
- Note: a shell that ran `set -a; source .env` before D27 still exports `DEPTH_BACKEND=da2_metric`, and an environment
  variable overrides .env. Re-source .env or `unset DEPTH_BACKEND` before starting the API.
- What the demo needs from the team: phone in a fixed landscape mount; tape height; a route with 10-20 potholes driven
  twice; for about 10 of them tape width/length + photo + location (size accuracy and IRC class check); optionally the
  same road after one repair (bus verification). Every keyframe is also bus-view training data (roadmap
  `model.bus_frames`).

## D29. Phone capture from any network (user request 2026-11-18)
The phone page (D28) only reached the PC on the office LAN. `make capture-public` (`phone_capture.py --public`) serves
on 127.0.0.1 only and starts the ngrok agent already installed and signed in on this PC (`~/bin/ngrok`, v3). The
phone then uses a public `https://...ngrok...` URL with a valid certificate (no warning page), on mobile data or any
Wi-Fi. `--ngrok-url` / `GM_NGROK_URL` pins a fixed domain (e.g. the account's free static dev domain). Rejected: router
port forwarding (needs office IT); Tailscale (private and the most secure, but every phone needs the app and an account,
and it is not installed); Cloudflare quick tunnel (not installed, URL changes on every start). Trade-offs accepted
by the user's request: frames pass through ngrok's servers (TLS ends there; DPDP: share the link only with drivers),
and the sign-in page is public. Hardening, on in both modes:
- the capture page has its **own access code** (4 words + 4 digits) and signing key in `data/capture/` (0600), no
  longer the review tool's (`GM_CAPTURE_CODE` overrides);
- **sign-in lockout**: 5 wrong codes in 10 min lock that client for 15 min; 50 wrong codes overall lock all sign-ins.
  The client is the tunnel's `X-Forwarded-For`, trusted only from loopback;
- uploads refused (HTTP 507) below 5 GB free disk; at most 20,000 keyframes per trip;
- headers: nosniff, `X-Frame-Options: DENY`, no-referrer, no-store, Permissions-Policy camera/geolocation/motion self.
Phone page: every call sends `ngrok-skip-browser-warning`. New options in 3 Upload: **upload while driving** (every
20 s when connected; only what is not yet sent, so it resumes after dead zones; after Stop the final upload waits for
a running one, then closes the trip) and **Wi-Fi only** (Android Chrome reports Wi-Fi vs mobile data; other phones
report "unknown" and upload). The list shows MB left (~25-50 MB per km, since every keyframe goes to the PC).
Found and fixed while testing in Chrome with fake sensors: (1) Stop during a keyframe save read the cleared trip and
lost that frame; the frame count then disagreed with the saved frames, so the PC showed the trip "incomplete". Now the
save keeps its own trip reference and the counts come from storage. (2) In a browser tab the page did not see,
Chrome ran the trigger timer once a second, giving 10 m spacing. The trigger now follows the camera's own frame
callback (`requestVideoFrameCallback`, ~30 Hz) with the 50 ms timer as fallback. Time in the background is reported
to the driver and the PC (`hidden_s`), with `cam_active=false` in those fixes. Result of the last browser run: 36
keyframes, 30 fixes and 6 jolts uploaded while driving, trip closed, PC summary "complete". The run was in a background
automation tab, so the spacing still reflects 1 Hz timers. A real phone run is still to do.

## D30. No setup for the driver: any orientation, tilt measured during the drive (user request 2026-11-18)
User: stop asking to mount sideways and hold still 2 s for the tilt. Now:
- **Orientation free**: portrait or landscape. The certificate follows the real frame size, and the 67 deg field of
  view is defined along the long side, so fx is the same either way. Checked with the independent ground-rectangle test
  on a 720x1280 portrait frame: width 0.611 / 0.613 m for 0.60 m at 4 / 6 m. Only a picture rolled more than 45 deg is
  refused (auto-rotate off).
- **Tilt without a button**: the page logs the 1 s mean of `accelerationIncludingGravity` plus the screen orientation
  angle with every 1 Hz fix and keyframe. `phone_capture.tilt_from_gravity` maps device axes to camera axes for the
  four orientations and uses `measure.surface.pitch_roll_from_normal`, so the sign conventions are the measurement
  code's own. Tested round-trip against `measure.camera.plane_from_extrinsics` in 0/90/180/270 deg, within 0.05 deg.
  Readings outside 0.93-1.07 g are dropped. `capture_replay.py` uses the median over +-30 s around each keyframe:
  braking tips the reading (0.1 g ~ 6 deg) only for seconds and is outvoted (tested: 6 s of 15 deg in 40 s of 9 deg
  gives 9.0). It writes one certificate per 0.5 deg pitch / 1 deg roll step (`phone_capture.tilted`) and gives every
  observation its own calib_id. The gravity fields are stripped before the API (frozen trip_fix schema). Without
  gravity data (motion permission refused) the start tilt or 10 deg is used, and the replay prints that.
- **Height from a vehicle preset** (car dashboard 115 cm, car windscreen 135, SUV 155, auto-rickshaw 130,
  two-wheeler 105, bus 230), remembered on the phone; a tape-measured value is optional and better (size scales with
  height: 10 cm off at 1.15 m is ~9 % on width). Calibration is made automatically when Start is pressed; the
  "Set up" tab is optional and the page opens on Drive.
- Browser check with fake sensors (phone sideways, 9 deg down): sign in -> Drive -> Start with no setup, 13 keyframes,
  upload complete. The replay found "phone gravity sensor during the drive, 9.0 deg down, roll 0.0 deg" from 24
  readings.

## D31. Phone held sideways with auto-rotate off: frames turned upright on the PC (found in the first real-phone test, 2026-11-18)
First real test (Samsung S24, public ngrok link): sign-in, recording, upload and trip close worked. It was a
12 s / 16 m standing or walking test (0-3 km/h, GPS +-14-25 m), so 0 keyframes, which is correct below the 5 km/h
trigger. The gravity log showed roll ~ +88-90 deg with the screen in portrait (orient 0): the phone was physically
sideways with auto-rotate off, so every frame would hold the road turned 90 deg. The detector is trained on upright
roads, and the D30 tilt step dropped all readings as rolled > 45 deg. Fix without asking the driver to change
anything: `capture_replay.upright_turn` lets every gravity reading vote for the screen angle that would have been
level (`tilt_from_gravity` at 0/90/180/270, smallest roll), minus the recorded angle. Frames are turned by that before
detection (`turn_image`: 90 = counter-clockwise, as the screen would have), the turned JPEG is what is uploaded, the
calibration gets the swapped frame size, and the tilt series is computed for the turned view. Tested for 90/180/270:
the side gravity calls 'up' ends at the top of the turned picture, and tilt is recovered within 0.05 deg. On the S24
data: turn 90 deg, tilt 16.7 deg down (10-90 %: 9.9-19.7, hand-held), roll -1.1 deg. The page no longer tells anyone
to switch on auto-rotate.

## D32. Automatic processing and a Drives dashboard (user request 2026-11-18)
The user wanted the phone data processed automatically, with potholes and their GPS positions on a dashboard, and asked
whether the phone sends data by itself. Done:
- **Phone**: "Upload automatically while driving" is now ON by default (every 20 s with signal; it finishes after
  Stop). A 4 Results tab opens the dashboard.
- **Capture server worker** (`scripts/capture_worker.py`, started by `make capture` / `make capture-public`;
  `--no-process` turns it off, `--api` / `GM_CAPTURE_API` picks the API): one thread with the model loaded once. Every
  keyframe is processed as it arrives:
  - upright turn (D31) and tilt from gravity so far (D30);
  - detection;
  - size-only measurement (D27);
  - ground position: the GPS fix advanced by GNSS latency x speed and by the measured distance along the heading
    (`geo.locate.ground_point`), +- sqrt(GPS accuracy^2 + (5 % range)^2). Without a heading (< 1 m/s) the pothole is
    placed at the phone, with the range added to the +-.

  Output: `data/capture/<trip>/results/<seq>.json` plus a crop. When the phone closes a trip, the worker re-checks
  the turn, then `capture_replay.replay_trip` sends it to the RoadFix API. If the API is down, `results/api_error.json`
  is kept and the send is retried every 5 min. A trip with nothing to send (no GPS) is marked skipped. On restart,
  `scan()` resumes unprocessed frames and unsent trips. `capture_replay.main` is now a thin CLI over `replay_trip`.
- **Dashboard** `/dashboard` (`scripts/capture_dashboard.html`, same sign-in, works on the phone and the PC):
  - drives with live status (recording / processing n/m / done, sent to RoadFix / API error);
  - Leaflet 1.9.4 (cdnjs) + OSM tiles, with the route from the 1 Hz fixes and pothole markers coloured by IRC class
    with the +- ring;
  - cards with crop, confidence, width +- / area, lat/lon +- m and a Google Maps link.

  A pothole = detections linked across consecutive keyframes. Its position is the precision-weighted mean, but the +-
  shown is the best single radius: frames 4 m apart share most of the GPS error, so averaging them must not shrink it.
  Media are served only to signed-in users, by strict name pattern.
- **"Exact location"**: not promised. Phone GPS is about +-5-15 m on open roads; the dashboard shows that radius, and
  a second pass lets the API cluster and average.
- Tests (`tests/test_capture_worker.py`): a ground rectangle 0.6 m wide, 5 m ahead and 0.3 m right, projected
  independently, while the phone heads east at 8 m/s. Measured width 0.6 +- 5 %, Large; placed 7.4 +- 0.4 m east
  (5 m + 0.3 s x 8 m/s) and 0.3 m south. Also tested: track linking, API down -> error and no re-queue within 5 min,
  no GPS -> skipped, and the server routes (queueing, auth, media path guard).
- Live end to end (separate test store + throwaway API on a DB copy): 60 real BharatPotHole frames uploaded over HTTP
  like a phone. The worker processed all 60 and read the tilt as 10.0 deg (gravity). It found 1 pothole: 0.37 +- 0.02 m
  wide, Medium, at 17.4571395, 78.3761092 +- 5 m on the HITEC City-Kondapur road, and sent it to the API with 0
  rejections, with no manual step. In Chrome the map, popup and card zoom worked. Found and fixed there: the map fit
  used the circle layers' bounds before the map had a view (Leaflet error, blank map); it now fits the coordinates.
  The crop of that detection looks like a rectangular drain grating: a likely false positive, the weak detector again.

## D33. Black keyframes from the phone; keyframe gallery; testing on foot (second real-phone test, 2026-11-18)
Second S24 test (public link): 53 s, 79 m, GPS settled at +-5 m, phone sideways with auto-rotate off (turn 90 handled,
tilt 20 deg). Only 2 keyframes, because the user was walking at ~4.5 km/h, under the 5 km/h trigger. Both keyframes
were **pure black** (every pixel 0). Cause: switching tabs re-assigned `srcObject` on the video player, which resets
it and leaves it paused (no autoplay, no play()), and keyframes are copied from that player. Desktop tests went
straight to Drive, so they never hit it. Fixes:
- The camera is attached once; tab switches only resume play (`keepPlaying`); both players have `autoplay`.
- `grab()` refuses a paused / not-ready / black picture: a 32x18 probe whose maximum is < 3. It restarts the player,
  retries on the next tick and warns the driver after 5 tries.
- The worker flags black frames (`blank`, max < 8) without running the model; the dashboard warns "N of M keyframes
  are completely black".
- The Drive-tab link to the results opens a new tab (it used to navigate away from a recording).
- New **"Testing on foot"** option (Set up): keyframes from 1.5 km/h instead of 5 km/h, still every 4 m.
- **Recordings on the dashboard**: a keyframe gallery per drive, each frame turned upright as the processing saw it
  (`/media/<trip>/view/<seq>?w=`, signed-in only), with time, speed, potholes found and black marks; a click opens
  the full frame.
Check in Chrome, the user's flow (Preview camera -> Drive -> Start, sideways, on foot at 4.5 km/h): both players
playing, 8 keyframes over 31 m with mean brightness 100-126, upload complete.

## D34. One RoadFix device per tester phone (second tester, 2026-11-18)
A second tester (Sathwik) joined. His first two "drives" were 4 s and 1 s long (Start then Stop): no GPS fix yet, so
0 frames, 0 fixes, marked skipped (correct). It showed a multi-user flaw: the device id came from the phone name
field, and an empty name fell back to the vehicle label, so every tester who leaves it empty would be one device
("phone-car-phone-on-the-dashboard") in the API, with tokens replacing each other on registration. Now the page keeps
a random key per phone (`rf.devKey`, 6 chars) and the server makes `phone-<tester name>-<key>`
(`phone_capture.phone_device_id`). The device id is part of the calibration hash, so two phones never share a
certificate. Old pages without a key fall back to the previous id.

## D35. Pretrained pothole detector replaces the trained model (user request 2026-11-18)
The user saw the model miss most potholes in Sathwik's walk (39 frames of a broken Hyderabad street) and asked for a
pretrained pothole model. Checked: the current model found 0 of the 13 frames that clearly show a pothole (judged by
eye), even at confidence 0.05. It called "manhole" in almost every frame, so this was not a threshold problem.
- **Candidates**: 15 ultralytics checkpoints from 11 Hugging Face repos (`hf models list --search pothole`).
  Downloaded to `data/raw/hf_pothole/` and pickle-scanned with `ml.pipeline.checkpoint_globals`: all safe. Copied to
  `weights/pretrained/hf_*.pt`; metadata in `reports/pretrained_scan_D35.json`. cazzz307 failed to download;
  RF-DETR (infra-intelligence) skipped (needs `rfdetr`).
- **Locked RDD-India test (514 images, AP50)**: harisanth 0.812, logesshhh rdd2022_multiclass 0.725, **logesshhh
  pothole_yolov8 0.598**, engjameso yolov9c 0.322, samdutse 0.320, current 0.249, others <= 0.27 (keremberke m 0.003).
  Caveat: the test images are public RDD2022 India, and rdd2022_multiclass was trained on "rdd_JapanIndia", so it has
  probably seen them. Any score here can be inflated by such leakage. mfranzon/pothole-yolo26 scores exactly like our
  old "bootstrap" model (the same model).
- **Sathwik's frames** (never public; 13 pothole frames and 20 clean frames by eye; 6 unclear frames left out), found
  / false alarms at conf 0.25:
  - **logesshhh pothole_yolov8 9/13 and 3/20** (boxes on the broken surface);
  - engjameso yolov9c 12/11; keremberke 12/13; spatel02 11/11; harisanth 7/11 (boxes plants, a mat, a pigeon,
    image edges);
  - current model 0/3.
- **Chosen: `Logesshhh/road-anomaly-pothole-yolov8m` `pothole_yolov8.pt`** (YOLOv8m, 7 classes: vehicles, pedestrian,
  crack, severe crack, pothole, speed bump; Apache-2.0 card; sha256 0ba816b66b588b6e...). Its negative classes keep
  it from boxing street objects. `SEG_WEIGHTS` in .env points to `weights/pretrained/hf_logesshhh_pothole_yolov8.pt`.
  The old model stays on disk (`runs/seg/y26m_seg_india_v2ft_geo`); switch back by editing .env.
- **Box model -> outline**: `capture_replay.default_detector` keeps segmentation models as they were. For a box model
  it keeps only the pothole class and gets the outline from SAM2.1-small (box prompt, `measure.detect.box_to_mask`:
  sam2 -> grabcut -> ellipse), because the size measurement needs an outline. Speed: ~50-90 ms per frame after
  loading. Edge confidence 0.35 -> 0.25.
- **Re-run of Sathwik's walk**: 12 potholes (was 3 false alarms). Still wrong: a dark repaired asphalt patch (frames
  15-16) and a pale patch (frame 38). Repaired patches are the known confusion; fix with labelled phone frames +
  fine-tuning this model later.
- **Stale environment trap**: shells that sourced the old .env export SEG_WEIGHTS / DEPTH_BACKEND, which override
  .env. `make api / capture / capture-public / capture-replay` now run with `env -u SEG_WEIGHTS -u DEPTH_BACKEND`, and
  the capture server prints the model it loaded.
- Not deleted (ask first): the other 14 candidate checkpoints (data/raw/hf_pothole 648 MB + copies in
  weights/pretrained), kept for comparison.

## D36. New pothole dataset + user-supplied model: analysis (user request 2026-11-18; no switch yet)
Supplied: `new dataset for pathole/pothole_dataset_trained_20260907`, the pothole subset of the "Nagar Netra"
16-class set (710 images / 1,482 boxes; 568/71/71 split; YOLO boxes, class id 1). Also
`new trained model/weights/best.pt|last.pt`: the YOLO11n 16-class `civic_services_oversampled` model trained on it
(2026-09-07, 100 epochs; all-class mAP50 0.603). Pickle scan clean.
- **Dataset** (`reports/new_dataset_analysis_D36.json`):
  - mostly Roboflow 640x640 exports;
  - test = "pothole_intel" (public Intel set) 30, "PotHoles" 16, RDD2022 "India" 13, "ashrith" 11, "AN" 1;
  - train dominated by "AN" 288 and "PotHoles" 226;
  - **leakage**: 15 test images have a near-copy (256-bit dHash <= 24; checked by eye: same scene, consecutive
    shots) in the new train/val, although the README calls the split leakage-free. 4 test images are near-copies of
    our own training data. "Clean" test = without those: 78 boxes.
- **New test, box AP50 (full / clean)** (`reports/new_dataset_eval_test_D36.json`):
  - engjameso yolov11 0.569 / 0.705, samdutse 0.548 / 0.672, engjameso yolov9c 0.530 / 0.647, harisanth 0.520 /
    0.631, potbot 0.509 / 0.624;
  - **new civic best.pt 0.483 / 0.505**, last.pt 0.458 / 0.471;
  - in use (logesshhh, D35) 0.385 / 0.443; old trained y26m 0.298 / 0.387.

  By source, the HF models win only on the public sets they were probably trained on (pothole_intel 0.86, ashrith
  0.86 vs new 0.57 / 0.45). **On RDD India dashcam images the new model is about twice as good** (0.23-0.27 vs
  <= 0.11; in use 0.03), and also on "PotHoles" (0.48-0.53 vs <= 0.36).
- **Sathwik's real street frames** (seen by no model; 13 pothole and 20 clean frames by eye), conf 0.25:
  - **new best.pt 11/13 found, 2/20 false alarms** (best of all);
  - last.pt 11/3; in use 9/3; engjameso 9/10; samdutse 9/8; potbot 8/8.
- **Locked RDD-India test (514)**: new best.pt AP50 0.056 (R@0.25 = 5 %). Looked at: RDD's pothole labels are
  mostly tiny, faint, distant marks, and on 6 random samples neither the new nor the in-use model boxes them at 0.25.
  The test rewards models that learned RDD's labelling style; in-use 0.598 and harisanth 0.812 are probably helped by
  RDD exposure. It stays our historical benchmark, but it is not a good judge of street use for these models.
- **Conclusion**: the highest number on the new test (engjameso yolov11) comes from probably-seen public images. On
  the independent evidence (unseen phone frames, India dashcam subset) the **user-supplied civic YOLO11n best.pt** is
  the best and smallest (5.5 MB, a few ms per image). Switching `SEG_WEIGHTS` is the user's call (asked 2026-11-18).

## D37. Switched to the user-supplied civic YOLO11n (user decision 2026-11-18)
After D36 the user chose to switch. `new trained model/weights/best.pt` (sha256 1b542fe9d0fadc0f...) was copied to
`weights/pretrained/civic_services_yolo11n_best.pt`; the original folder is untouched. `.env SEG_WEIGHTS` now points
to the copy. The detector is unchanged: box model, pothole class only (id 1), SAM2 outline, conf 0.25. Re-run of
Sathwik's walk: 13 potholes, confidences 0.62-0.84 on the clear ones; the D35 model found 12 at 0.26-0.72. Previous
models stay on disk (D35 logesshhh; trained y26m). Known gaps: tiny distant potholes (RDD-India locked test AP50
0.056) and repaired patches. Both need labelled phone/bus frames and fine-tuning of this model.

## D38. Re-organisation around the supplied base model and base dataset (user request 2026-11-18)
User: use the supplied YOLO11n (`new trained model/`) as the BASE model and the supplied dataset
(`new dataset for pathole/`) as the BASE dataset; every earlier dataset becomes a TEST set; rebuild and organise the
project for an end-to-end test. User choices: fine-tune now and adopt only if better on all test sets; fix the leaky
split; keep old material as an archive (delete nothing).
- **Base dataset** `data/datasets/nagar_netra_pothole` (`data/build_base_dataset.py`, `make base-dataset`). The 710
  supplied images and labels are copied unchanged, keeping the 16 names with pothole = 1, so the base model's head
  lines up. Images are grouped by near-duplicate (256-bit dHash <= 24, union-find) and each group goes to the split
  most of its members had (ties -> train). Result: 597 / 57 / 56 images (1,263 / 131 / 88 potholes); 15 test->train
  and 14 val->train moved; 0 near-copies across splits afterwards. `split_manifest.json` keeps the supplied split.
- **Base model** `weights/base/civic_yolo11n.pt` (+ `_last.pt`, `.onnx`, `_int8.tflite`) = the supplied files; the
  `.env SEG_WEIGHTS` points there. The source folders stay untouched (git-ignored, with `weights/base/` and
  `data/testsets/`).
- **Registries**: `configs/datasets.yaml` holds the base + 6 test sets:
  - Nagar Netra test (leak-free);
  - RDD-India locked 514, scored on the original RDD D40 boxes;
  - BharatPotHole bh_ and web ws_ (from the v2 test);
  - eYantra ey_ (534 images without potholes -> false-alarm rate);
  - `data/testsets/phone_street_20261118`: Sathwik's 39 frames, upright, with the frame-level labels by eye.

  `configs/models.yaml` holds the base, adopted fine-tunes and the archive.
- **`ml/benchmark.py`** (`make benchmark`): every model x every test set, with the same COCO matching / 101-point AP
  as ml.eval and only the model's pothole class (pothole / potholes / D40). Rates at conf 0.25. Output:
  reports/benchmark.json and reports/BENCHMARK.md. **`ml/train_det.py`** (`make train`):
  - fine-tunes the base on the base dataset: AdamW lr0 1e-3 cosine, AutoBatch 80 % of free GPU memory,
    `runs/detect/<name>`, live files for `make dashboard`;
  - benchmarks the candidate against the model in use;
  - adopts it (copy to weights/candidates, .env, registry) only if no test set is worse (AP50 -0.01, false-alarm rate
    +0.01, phone frames +-0) and the mean box AP50 is better (`benchmark.no_test_worse`).
- **First fine-tune** `det_yolo11n_ft_v1` (80 epochs): base test 0.507 (base 0.471) and web 0.411 (0.378), but
  eYantra false alarms 24.9 % (2.8 %), phone 11/13 with 8/20 false (2/20), RDD 0.042 (0.055). **Not adopted.** The
  pothole-only base data (no clean roads, the other classes absent) teaches it to call more things potholes. Archived
  in the registry. Next lever: real phone/bus frames with potholes and clean-road negatives in the base dataset.
- **Tools on the new base**:
  - dataset review opens the base dataset: box labels drawn as rectangles, list-style names, its own DB
    `review_nagar_netra_pothole.sqlite`; the v2 decisions stay in review.sqlite;
  - model test page: box models get SAM2 outlines, scores = base-test AP50 from the benchmark, the model in use
    first;
  - training dashboard: reads runs/detect and runs/seg;
  - project control: new check kind `bench`, and the model / export areas rewritten around the base and the
    benchmark.
- **Makefile** in sections (`make help`): setup, model (base-dataset / benchmark / train / dashboard), services, demo,
  and archive (`seg-*`: the old YOLO26-seg pipeline and the v2 data builds, still runnable). README, HANDOFF section 0,
  WORKFLOW and CLAUDE.md describe the new layout.

## D39. Combined fine-tuning set: base + the earlier dataset's train/val (user request 2026-11-18)
The pothole-only fine-tune (D38) raised false alarms. The user asked to use the earlier dataset to improve it.
`data/build_combined_dataset.py` (`make combined-dataset`) builds `data/datasets/combined_ft` (symlinks + box labels):
- base train repeated x5: 2,985 images;
- earlier `india_pothole_seg4_v2` train + val: 10,329 images. Pothole polygons become boxes with id 1 (the base
  model's pothole). Crack, alligator-crack and manhole labels are dropped, so those regions are background.
- train 13,314 images / 21,593 potholes / 3,819 images without potholes. Val: base val 57 + 300 earlier val (seed 0).
- no test part is read for training, and 4 earlier train images were dropped as near-copies (256-bit dHash <= 24) of a
  test image (base test, earlier test incl. the locked RDD-India 514, the phone frames).

`make train DATA=combined [EPOCHS=60]` fine-tunes from `weights/base/civic_yolo11n.pt` with the same benchmark +
adoption rule (D38). `SMOKE=1`: 1 epoch on 10 %, 2 test sets, never adopts. The trial ran end to end: AutoBatch batch
73 at 80 %, ~10 it/s, so ~30 s per full epoch and ~30 min for 60 epochs plus ~4 min benchmark; rejected as expected.
`train_sets` in configs/datasets.yaml names the training sets (base, combined).

## D40. On-the-fly augmentation and a responsive system during fine-tuning (user request 2026-11-18)
The user asked whether `make train` augments on the fly, and asked that training not stall the PC. Before: Ultralytics
defaults only (mosaic, HSV jitter, left-right flip, scale 0.5, translate 0.1). The project's own augmentations and
the D20 display reserve were only wired into the archived segmentation trainer. Now `ml/train_det.py`:
- **Geometry** (D21 bus mount, every image as it loads): scale 0.5, translate 0.15, degrees 5, perspective 0.0005;
  no up-down flip, no shear (fixed forward camera); left-right flip 0.5, mosaic 1.0 (off for the last 10 epochs),
  HSV defaults. Boxes follow the warp.
- **India road conditions** (`ml/augment/india.py`, pixel-only, so boxes stay valid; `AUG=off|light|default|strong`,
  default `default`): rain / wet asphalt, sun glare, hard shadows, dusk and haze, motion blur <= 7 px, downscale,
  JPEG 60-90.
- **Gentle mode**:
  - process at nice 10 with the lowest best-effort I/O priority (dataloader workers inherit both);
  - workers = a quarter of the CPU threads (max 8); 4 PyTorch CPU threads; no RAM image cache;
  - D20 display reserve: with the monitor on the RTX 5080, PyTorch is capped at 10.7 of 15.5 GiB and the 80 %
    batch fraction is scaled to that cap.
- **Smoke check**:
  - batch 57; peak GPU 7.0 GB; load average below 2 of 32 threads; >= 32 GB RAM free while training;
  - albumentations transforms logged; `train_batch*.jpg` shows shadows, perspective, zoom-out and darkening with the
    boxes still on the potholes.

## D41. Annotation dashboard: the user draws the potholes (user request 2026-11-18)
The combined fine-tune (D39/D40, `det_yolo11n_ft_20261118_1717`) beat the base model on every public test set and on
false alarms (base test 0.604 vs 0.471, RDD-India 0.368 vs 0.055, eYantra 1.5 % vs 2.8 %) but found only 5 / 13
phone-walk pothole frames (base: 11 / 13). It was not adopted. The phone-walk labels are frame-level and by eye, so it
was unclear which model was right. The user asked for a page to outline the potholes themselves.
- `scripts/annotate.py` + `annotate.html`: `make annotate` (LAN :8775) and `<capture link>/annotate` (mounted on the
  phone capture server, same access code, own cookie).
- **Photo sets**: the phone-walk test set (39 frames), plus every phone drive with usable photos (turned upright as
  D31; black frames hidden). The test walk's own drive is never offered, to avoid leaking the test set.
- **Tools**: outline (click points or press-and-drag freehand) and box. Each photo is marked "potholes", "no pothole"
  or "not sure". The guesses of the `suggest: true` models in `configs/models.yaml` (base + combined fine-tune) are
  shown as dashed boxes, and one click accepts a guess.
- **Storage**: `data/annotate/annotations.sqlite`. Every save is kept (who, when); the newest mark per photo counts.
- **`make annotate-export`**:
  - test walk:
    - `labels/<stem>.txt` YOLO class 1 (outline = polygon, box = cx cy w h); "no pothole" = empty file; "not sure"
      = no file;
    - `labels.json` frame lists use the hand marks per photo; unmarked photos keep the by-eye label, which is
      kept as `eyeball_v1`;
    - new test set `phone_street_marked` (kind yolo, `require_label`) scores box AP50 on hand-marked photos only;
      the benchmark skips a test set with no images yet.
  - drives: `data/datasets/phone_marked` (upright JPEG copies + labels), train set `phone`; near-copies (dHash ≤ 24)
    of test photos are left out. No run uses it until the user chooses to.
- The model choice (base vs combined vs a retrain with more phone data) is made after the user's marks exist, with
  `make benchmark` on `phone_street_marked`.

## D42. Clean-roads fine-tune and the two-phone keyframe-vs-video test (user request 2026-11-18)
D41 marks (25 pothole photos, 41 outlines, 14 clean): the base-only fine-tune `det_yolo11n_ft_v1` found 25 / 41
outlines, the base model 14 / 41, the combined fine-tune 7 / 41. All three flagged 3–4 of the 14 clean photos.
- **Training set `base_neg`** (`make negatives-dataset`, `data/build_combined_dataset.py --earlier negatives`):
  - base train ×5 (2,985) plus at most 1,250 earlier images with NO pothole label: all 747 eYantra clean roads first,
    then cracks / manholes at random (seed 0); about 30 % background, because more risks recall;
  - validation = base val + 300 earlier pothole-free val images;
  - the earlier pothole labels are not used (they pulled close-range phone recall down);
  - test-set near-copies are guarded as in D39;
  - `make train DATA=base_neg`; adoption rule unchanged (D38).
- **Phone side or model side?** The user suspects the frames the phone page sends, not the model.
  - The page sends 1280×720 frames from the browser camera stream, JPEG quality 0.75, one keyframe every 4 m.
  - Evidence so far points partly at the model: on the SAME 39 keyframes another model finds 25 / 41 vs 14 / 41.
    The comparison measures the rest.
  - `scripts/phone_vs_video.py` (`make compare-video TRIP= VIDEO= [OFFSET=] [MODELS=]`):
    - phone B records a normal video next to phone A running the capture page. One Android camera cannot serve
      two apps at once, so the test needs two phones.
    - Alignment by picture similarity (MP4 `mvhd` creation time narrows it; `OFFSET` = video time of keyframe 1
      overrides it).
    - Per keyframe, each model runs on the keyframe and on the ONE video frame at that moment. A ±0.5 s window
      would favour the video, so it is reported only as extra information.
    - Outcomes: both / video only / keyframe only / neither, plus sharpness (Laplacian variance at 640 px),
      brightness, size and JPEG kB.
    - Pothole runs in the 2 fps video scan with no keyframe during them = misses due to keyframe spacing.
  - The matched video frames become annotation photo set `video:<name>`; marks there and on the drive are scored
    on the next run. Video sets are never exported to training.
  - Report: `reports/phone_vs_video/<name>/index.html` + `summary.json`.
