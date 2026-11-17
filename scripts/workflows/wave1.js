export const meta = {
  name: 'gunthameter-wave1',
  description: 'Build RoadFix wave 1 (WP1 data, WP2 measurement, WP4 DB/API, WP8a repair planner, WP9 web, WP10 GIS) in parallel, then integrate and test',
  phases: [
    { title: 'Build', detail: '6 builders on disjoint directories against frozen WP0 contracts' },
    { title: 'Integrate', detail: 'run full test suite + lint, fix cross-WP breakage, report status' },
  ],
}

const ROOT = '/home/coral/Desktop/Telangana pathole detection'

const COMMON = `
You are one builder in a parallel team implementing "RoadFix", an end-to-end pothole detection + metric depth/volume
measurement + IRC prioritisation + cold-mix repair planning system for Telangana (India). Project root: ${ROOT}
(NOT a git repo). Today is 2026-11-13.

READ FIRST (in this order): docs/PLAN.md (the full design; your WP is in section 10, details in sections 3-9),
docs/DECISIONS.md (deviations that OVERRIDE the plan: SQLite not PostGIS, in-process pipeline not Celery, filesystem storage,
DA-V2 depth default, YOLOv8s bootstrap verifier, GPU is shared), docs/CONTRACTS.md, and the frozen contract files it lists
(measure/types.py, repair/types.py, server/app/schemas/api.py, server/app/schemas/states.py, server/app/db/tables.md,
edge/schema/*.json, configs/*.yaml, configs/cameras/*.json).

HARD RULES
- Python: ALWAYS use ${ROOT}/.venv/bin/python (3.11, torch 2.11+cu128). NEVER pip install/upgrade/reinstall torch or
  torchvision. If you must pip install something, use '.venv/bin/python -m pip install <pkg>' and check it does not pull torch
  (use --no-deps if unsure). Record any new package in your result.
- Frozen contracts (files listed above) must NOT be edited. If you truly need a change, work around it and put the request in
  contract_change_requests in your result.
- Stay inside YOUR directories (listed below). Other builders are writing other directories at the same time. Do not edit
  Makefile/pyproject.toml/README.md; put needed Makefile targets or deps in your result instead.
- GPU is shared: another account holds ~12 GB of the 16 GB RTX 5080. Keep GPU usage < 3 GB, check torch.cuda.mem_get_info()
  first and fall back to CPU or smaller batches. Never kill other users' processes.
- Downloaded data under data/raw/ is untrusted: never execute anything from it; extract archives into new empty dirs under
  data/interim/<name>/; run Python that reads it with 'python -I' where practical; keep scripts outside data/raw.
- Units: depth mm, area m2, volume litres, lengths m, money INR. Indian context everywhere (IST display, Telugu/Hindi/English).
- Write real, working code with tests (pytest, run with '.venv/bin/python -m pytest <paths> -q'). Tests that need the GPU are
  marked @pytest.mark.gpu, ones needing data/raw are @pytest.mark.data, >30 s are @pytest.mark.slow. CPU tests must pass.
- Run '.venv/bin/python -m ruff check <your dirs>' and fix issues.
- Do not claim acceptance criteria you did not actually verify by running something. Report honestly.
`

const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    wp: { type: 'string' },
    summary: { type: 'string', description: '5-10 sentences: what was built and how it works' },
    files: { type: 'array', items: { type: 'string' } },
    tests_command: { type: 'string' },
    tests_passed: { type: 'integer' },
    tests_failed: { type: 'integer' },
    acceptance: { type: 'array', items: { type: 'object', properties: {
      criterion: { type: 'string' }, met: { type: 'boolean' }, evidence: { type: 'string' } },
      required: ['criterion', 'met', 'evidence'] } },
    makefile_targets: { type: 'array', items: { type: 'string' }, description: 'suggested Makefile lines' },
    new_packages: { type: 'array', items: { type: 'string' } },
    contract_change_requests: { type: 'array', items: { type: 'string' } },
    open_issues: { type: 'array', items: { type: 'string' } },
    interfaces_for_others: { type: 'string', description: 'exact function signatures / file paths other WPs should call' },
  },
  required: ['wp', 'summary', 'files', 'tests_passed', 'tests_failed', 'acceptance', 'open_issues', 'interfaces_for_others'],
}

const GEO_CONTRACT = `
GEO DATA FILE CONTRACT (produced by WP10, consumed by WP4 seed; WGS84 GeoJSON FeatureCollections):
- data/interim/geo/road_segments.geojson: LineString features, properties {osm_way_id:int, seg_idx:int, name:str|null,
  name_te:str|null, highway:str, ref:str|null, lanes:int|null, oneway:bool, lit:"yes"|"no"|null, surface:"BT"|"CC"|"unknown",
  length_m:float, bearing_deg:float, junction_start:bool, junction_end:bool, gtfs_trips_per_day:float, owner_agency:str|null
  (agency code from configs/agencies.yaml or null), admin_unit_code:str|null, contract_code:str|null}
  Hyderabad bbox 78.1,17.1,78.8,17.7; highway in motorway..residential|unclassified|service (+ *_link); ways split at intersections.
- data/interim/geo/pois.geojson: Point features {kind: school|hospital|clinic|bus_stop|metro_station|marketplace, name, source: osm|gtfs}
- data/interim/geo/gtfs_routes.json: [{route_id, short_name, long_name, shape: [[lon,lat],...], trips_per_day}]
- gis/layers/ghmc_zones.geojson, gis/layers/circles.geojson (admin_unit: properties {code, kind, name_en, name_te, parent_code,
  official:false, valid_from}), gis/layers/crmp_packages.geojson ({code, agency, contractor, dlp_months}),
  gis/layers/waterlogging_hotspots.geojson ({name, source}), gis/layers/road_owner_overrides.geojson ({owner_agency, reason}).
- tests/fixtures/geo_small/ : the SAME files clipped to a ~2x2 km box around Ameerpet/Punjagutta (17.42-17.44, 78.44-78.46),
  small enough (<2 MB total) for fast tests and seeding without the big extracts.
`

const BUILDERS = [
  {
    wp: 'WP2', label: 'WP2 measurement core',
    prompt: `${COMMON}
YOUR PACKAGE: WP2 — Measurement core (the project's differentiator; the user's hard requirement is DEPTH measurement).
Your directories: measure/ (except measure/types.py which is frozen), tests/test_volume_synthetic.py, tests/test_surface.py,
tests/test_fuse.py, tests/test_imu.py, tests/test_irc.py, tests/test_measure_*.py, reports/measure_demo/ (outputs).
Implement PLAN section 4 fully and section 10 WP2 interfaces exactly:
- measure/depth_backends/{base,da2_metric,da3_metric,synthetic,stereo_sgbm,metric3d_onnx}.py with
  DepthBackend.predict(img_bgr, fx_px) -> (depth_m[H,W], conf|None). da2_metric = transformers
  depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf (already in HF cache; ~0.23 s/frame); handle focal-length rescaling
  (DA-V2 metric outdoor was trained on vKITTI; document how fx is used). da3/metric3d may be stubs that raise a clear
  NotImplementedError-with-instructions if the package is missing. Backend factory from configs/measure.yaml.
- measure/calib.py (cert I/O, undistort maps, checkerboard intrinsics, PnP extrinsics, drift check), measure/surface.py
  (numpy RANSAC plane, height scale-lock, road frame, local quadratic ring fit), measure/pothole_geom.py (mask integral:
  depth_p95/max/mean, area with per-pixel footprint, volume, minAreaRect L/W, range, lane offset, ground_xz, full sigma model
  of PLAN 4.7, quality gates, water flag PLAN 4.10), measure/fuse.py (PLAN 4.8 + the MAD rule in PassMeasurement docstring),
  measure/imu.py (PLAN 4.9), measure/irc.py (severity + Table 5.3 stretch rating, reading configs/irc_rules.yaml),
  measure/synthetic.py (render planar + crowned road with spherical-cap cavity, exact volume), measure/truth.py (truth ingest,
  MAE/MAPE per range bin, fit k_sigma and g_imu), measure/pipeline.py measure_frame(img, mask, cert, cfg) -> FrameMeasurement,
  measure/cli.py: 'python -m measure.cli run --image X --calib C [--mask m.png | --detect weights.pt] --json out.json' and
  'python -m measure.cli demo --out reports/measure_demo' which (a) runs the synthetic bowl validation and writes a table, and
  (b) runs real depth + detection (weights/bootstrap_pothole_yolo26m.pt via ultralytics; it is a box detector -> derive a mask
  e.g. SAM2 facebook/sam2.1-hiera-small box prompt if it fits in GPU, else GrabCut/box ellipse, record which) on ~5 real Indian
  images (unzip a handful of India images from data/raw/rdd2022/road_damage_dataset.zip or bharatpothole.zip into
  data/interim/measure_demo_imgs/ — read zip listings with python -I), writing JSON + an overlay PNG (mask, depth colormap,
  depth/area/volume ± sigma, IRC class) per image and a summary.md.
Acceptance (PLAN WP2 + 4.12 level 1): synthetic bowl tests pass (volume within 3 % noise-free, depth_p95 within 2 mm, crowned
road within 5 %, sigma coverage >= 90 % over 200 noisy trials — if coverage fails, fix the sigma model, don't loosen the test);
measure.cli run on a real image with configs/cameras/phone_demo.json returns JSON with all FrameMeasurement fields and
certified=false; depth forward pass timing recorded; truth.py fits k on synthetic noisy runs.
Note real-image photos are not taken with phone_demo intrinsics: scale-lock to h_true makes depth relative-to-road robust;
state limitations honestly in reports/measure_demo/summary.md.`,
  },
  {
    wp: 'WP1', label: 'WP1 data + dataset build',
    prompt: `${COMMON}
YOUR PACKAGE: WP1 — Data acquisition & unified dataset build.
Your directories: data/ (download.py, build_dataset.py, splits/, interim/, datasets/), configs/model/ is frozen (it already holds
india_pothole_seg.yaml — your builder writes the REAL yaml next to the dataset at data/datasets/india_pothole_seg/data.yaml),
tests/test_dataset_*.py, ml/augment/india.py (India augmentations, PLAN 3.3).
State of data/raw (ALL already downloaded, verified readable): data/raw/rdd2022/road_damage_dataset.zip (9 GB, full RDD2022
via HF mirror akiii1234/RoadDamageDetection — inspect its listing to find the India subset), data/raw/eyantra/
semantic-segmentation-datasets-of-indian-roads.zip (2.9 GB, masks), data/raw/bharatpothole/bharatpothole.zip (800 MB, YOLO boxes,
licence UNVERIFIED -> include only behind --include-bharat flag, default off, but DO unpack it so the simulator can use frames),
data/raw/gtfs_tgsrtc/tgsrtc_gtfs.zip, data/raw/osm/southern-zone-latest.osm.pbf (OSM/GTFS belong to WP10, not you).
Implement: data/download.py (idempotent; '--all|--rdd|--eyantra|--bharat|--gtfs|--osm|--weights' — since downloads are done it
mainly verifies + extracts into data/interim/<name>/; keep the original download commands as fallback incl. the HF mirror),
data/build_dataset.py per PLAN 3.3 with the class map in configs/classes.yaml: RDD India D40->pothole, D20->alligator_crack,
D00/D10->crack, boxes->masks via SAM2.1 (facebook/sam2.1-hiera-small, transformers Sam2Model, box prompts, keep if IoU(mask
bbox, box) >= 0.6 else box polygon; batch on GPU within ~2.5 GB, else CPU), eYantra masks->polygons (shallow->pothole_water
heuristic), negatives (non-D40 RDD frames) >= 20 % of train; split: val = eYantra test masks + 10 % RDD train held out,
test = up to 500 RDD-India D40 held-out images. Output YOLO-seg layout at data/datasets/india_pothole_seg/ + data.yaml +
stats.json (counts per class/split, mask source counts). Make it resumable (cache SAM2 results). ACTUALLY RUN the build to
completion (it may take a while; run long steps in the background and poll). Also --limit N for quick tests.
Acceptance: ultralytics.data.utils.check_det_dataset on data.yaml succeeds; stats.json shows >= 3000 pothole instances total and
>= 500 val mask images (if the real data cannot reach this, report the actual numbers honestly); nothing executed from data/raw;
a quick test builds a --limit 20 dataset into a tmp dir.`,
  },
  {
    wp: 'WP4', label: 'WP4 DB + storage + auth + core API',
    prompt: `${COMMON}
${GEO_CONTRACT}
YOUR PACKAGE: WP4 — Database, storage, auth, core API (SQLite demo per DECISIONS D1; same models must work on PostgreSQL).
Your directories: server/app/ (except server/app/schemas/* which are frozen), server/tests/, scripts/seed_demo.py.
Implement: server/app/core/{config.py (pydantic-settings or dotenv reading .env.example keys), security.py (argon2 passwords,
JWT HS256, demo OTP), rbac.py (require_role(*roles) dependency, scope_query(user, stmt): contractor sees only own WOs, zonal_admin
only own agency/admin unit), audit.py (append-only audit_log writer)}; server/app/db/{session.py, models.py (SQLAlchemy 2.x
models for EVERY table in server/app/db/tables.md, using lat/lon/h3/WKT columns, JSON columns), init.py ('python -m
server.app.db.init' creates tables)}; server/app/storage.py (FsStorage + S3Storage behind put/get/presign/url; FsStorage under
data/store/ with a PUT upload route for presigned demo uploads); geo/index.py is NOT yours (WP5) — for bbox/radius queries use
simple lat/lon range filters in SQL; server/app/main.py (FastAPI factory, CORS for http://localhost:5173, GZip, /healthz,
/v1 prefix, OpenAPI); routers: auth, edge (register/config/presign/upload), ingest (observations batch idempotent by client_uuid,
JSON-schema validated with edge/schema files AND the pydantic DTOs; trip-fixes; trip-close), citizen (multipart report + ticket
lookup), potholes (list with filters/bbox/pagination, detail, reject, GET /tiles/potholes/{z}/{x}/{y}.geojson per-tile GeoJSON
for the demo, /segments/rating stub reading measure.irc if importable), measure (POST /measure: image upload + calib_id -> calls
measure.pipeline.measure_frame if importable, else 503 with a clear message — WP2 is being written concurrently), admin
(priority-config, rate-card get/put versioned, admin-units upload, jurisdiction queue list/resolve, audit query), public/wards.
server/app/pipeline.py: process_observation(obs_id) skeleton with the stage order from DECISIONS D2, each stage a function that
later WPs (WP5/WP6) fill in; for now it marks observations verified and is called inline when DEMO_MODE=1.
scripts/seed_demo.py: loads configs/agencies.yaml, rate card, priority config, demo users per role (password 'demo1234',
documented), devices with configs/cameras certs, and the GEO files above if present (fall back to tests/fixtures/geo_small/ which
WP10 is creating concurrently; if neither exists yet, seed without roads and say so). '--fixture' flag uses the small fixture.
Acceptance: fresh SQLite DB create + seed works; pytest server/tests/test_api_smoke.py (register device -> post the
edge/schema/examples observations -> list potholes endpoint responds; duplicate client_uuid counted as duplicate) passes;
RBAC test: contractor cannot read another contractor's work orders; audit_log written on admin config changes;
uvicorn starts and /healthz returns ok. Use FastAPI TestClient for tests with a tmp SQLite file.`,
  },
  {
    wp: 'WP8a', label: 'WP8a repair planner (pure lib)',
    prompt: `${COMMON}
YOUR PACKAGE: WP8a — Repair planning pure library (no DB, no server imports).
Your directories: repair/ (except repair/types.py which is frozen), repair/tests/, priority/ (pure scoring library).
Implement PLAN section 7 exactly in repair/planner.py: plan(pothole: measure.types.PotholeFused, rules: dict (configs/irc_rules.yaml),
rate_card: dict (configs/rate_card.ghmc_fy2026.yaml), context: repair.types.PlanContext) -> repair.types.RepairPlan — merge
handling (merge_cut(potholes) for potholes within 0.5 m on the same lane band), square cut with margin, design depth from
depth_p95 (ceil 5 mm, min 50 mm), layering (> 75 mm -> CRM bottom + 50 mm mix), the four volumes, mix mass with wastage, bags with
sigma range (NO 'exact bags' language), ancillaries, labour, cost lines with provenance from the rate card, baseline Rs 7,396,
method selector (spec codes, weather gate, CC surface -> CC_MORTAR), water-filled -> pending_remeasure with lower-bound quantities
using the IRC lower-bound severity, refs list of IRC clauses. repair/mb_export.py: measurement_book_csv(rows) and indent_csv(run)
(Measurement Book columns used by Telangana PWD/GHMC: item no, description, nos, L, B, D, quantity, unit, rate, amount).
repair/loaders.py: load_rules(), load_rate_card() from configs.
Also priority/: score.py compute(pothole, ctx) -> (P 0-100, breakdown dict) implementing PLAN section 6 S/E/V/T/M terms with
configs/priority.default.yaml (pure: ctx carries road class, bus passes/day, nearby POIs list with distances, lit, hours since
confirmed, sla hours, breached, complaints, month, rain_72h_mm, in_hotspot, junction distance, wheel tracks); exposure.py,
vulnerability.py, weather.py (monsoon window + manual flag + IMD adapter stub), sla.py sla_hours(highway, agency_code, month,
contract_rules) per PLAN 6 T term; stretch rating lives in measure/irc.py (WP2) — don't duplicate it.
Acceptance: repair/tests/test_planner.py worked example from PLAN 7.10 (0.50x0.40 m, p95 60 mm sigma 8 -> A_cut 0.4875, single
layer 60 mm, ~72 kg -> 3 bags of 25 kg range 3-4, cost ~Rs 1,406); layered 90 mm -> CRM 40 mm + 50 mm mix; water-filled ->
pending_remeasure; jet patcher when cluster >= 8; cost lines sum to est_cost_inr; tests/test_priority.py (put it at
priority/tests/test_priority.py) covers each term incl. T breach at 24 h for an arterial and weights summing to 100 max.`,
  },
  {
    wp: 'WP10', label: 'WP10 GIS + OSM + GTFS',
    prompt: `${COMMON}
${GEO_CONTRACT}
YOUR PACKAGE: WP10 (+ the data-prep half of WP5) — GIS layers, OSM roads/POIs, TGSRTC GTFS.
Your directories: gis/, geo/osm.py, geo/gtfs.py, geo/build_geo.py, data/interim/geo/, tests/fixtures/geo_small/, tests/test_geo_build.py.
Inputs: data/raw/osm/southern-zone-latest.osm.pbf (560 MB Geofabrik), data/raw/gtfs_tgsrtc/tgsrtc_gtfs.zip. pyrosm 0.29, osmnx 2.1,
shapely 2.2, geopandas, pyproj, h3 are installed. No osmium CLI, no sudo; Overpass refuses curl -> everything from the PBF.
Implement geo/osm.py (pyrosm with bounding_box 78.1,17.1,78.8,17.7: drivable ways, split at intersections (nodes shared by >=2
ways), segment bearing, length in EPSG:32644, lit/surface/lanes/ref/name/name:te tags (surface concrete->CC, asphalt/missing->BT
or unknown), junction flags; POIs: amenity school/college/hospital/clinic/marketplace, railway=station+station=subway or
public_transport for Hyderabad Metro), geo/gtfs.py (stops->bus_stop POIs, routes+shapes (build shapes from stop sequences if
shapes.txt missing), trips/day per route; assign gtfs_trips_per_day to road segments by buffering route shapes ~15 m and
intersecting — vectorised with STRtree), geo/build_geo.py ('python -m geo.build_geo [--bbox ...] [--fixture]' writes all
contract files, then the fixture clip), owner assignment from configs/agencies.yaml owner_rules (NH ref -> NHAI, SH -> RNB,
ORR -> HMDA, else ULB by zone polygon; unknown -> null so WP4 puts it in jurisdiction_queue).
gis/layers: GHMC zones and circles — try a public source (e.g. git clone --depth 1 https://github.com/gggodhwani/telangana_boundaries
into data/raw/tg_boundaries — read-only data, check licence), otherwise construct approximate zone polygons (Charminar, Khairatabad,
Secunderabad, Serilingampally, Kukatpally, LB Nagar) from well-known locality centroids with Voronoi clipped to the GHMC
envelope, clearly marked official:false; Cyberabad/Malkajgiri corporations per configs/agencies.yaml similarly approximated;
crmp_packages (7, approximated along arterial corridors in each zone), waterlogging_hotspots (well-known Hyderabad spots reported
in news e.g. Masab Tank, Lakdikapul, Khairatabad, Tolichowki, Malakpet, Rasoolpura... as small polygons, source='news (approx.)'),
road_owner_overrides. gis/build_tiles.sh: pmtiles extract command documented (optional; online OpenFreeMap fallback per D7).
ACTUALLY RUN the build on the real PBF (expect >= 50k segments in the bbox; report the count), and generate the fixture.
Acceptance: road_segments.geojson has >= 50k segments with all contract properties; every segment has owner_agency or will be
queued (report counts); pois >= several hundred with kinds; gtfs routes loaded with trips/day; fixture < 2 MB; tests pass.`,
  },
  {
    wp: 'WP9', label: 'WP9 web PWA',
    prompt: `${COMMON}
YOUR PACKAGE: WP9 — Web PWA (React + Vite + TypeScript + MapLibre), Telugu-first i18n (te/hi/en).
Your directory: web/ only. Node v22 and npm 9 are installed; npm registry access works. Use current stable versions that resolve
(PLAN 8 lists target versions; if one does not exist, use the latest that does and note it). The backend (WP4) is being written
concurrently: code strictly against docs/CONTRACTS.md + server/app/schemas/api.py (DTO field names) and build a mock layer
(web/src/lib/mock/ with realistic Hyderabad seed data: ~60 potholes on real roads around Ameerpet/Punjagutta/Khairatabad/
Mehdipatnam/LB Nagar with depths, sigma, severities, priorities, plans with bags low-high and Rs cost, work orders, a repair run,
contractors, KPIs) toggled by VITE_USE_MOCK=1 (default when the API at VITE_API_BASE is unreachable).
Routes per PLAN WP9: / zonal dashboard (KPI tiles, map, ward/zone league table, monsoon toggle, cost vs GHMC Rs 7,396 baseline),
/map (MapLibre with OpenFreeMap liberty style https://tiles.openfreemap.org/styles/liberty, potholes coloured by severity/priority,
filters, stretch rating layer when available), /potholes/:id (frame with mask overlay placeholder, depth cross-section chart,
measurement ± sigma, method + 'uncertified' badge, priority breakdown bars, repair plan with bags low-high, timeline), /queue
(priority list, select -> create work order), /work-orders, /runs/:id (printable indent via print CSS), /field (offline list,
before/after photo capture resized to 1280 px, bags used, Dexie queue + background sync on reconnect), /contractors/:id
(scorecard + survival curve), /citizen (report with photo + location, track by ticket), /admin (priority weights, rate card,
SLA, jurisdiction queue, audit viewer, calibration status), /public/wards. Login screen with role picker for the demo users.
Typed API client in src/lib/api.ts mirroring the pydantic DTOs. i18n JSON files with real Telugu and Hindi translations of UI
strings (Noto Sans Telugu/Devanagari fonts via @fontsource). PWA manifest + service worker (vite-plugin-pwa). Clean, serious
government-dashboard design, mobile-first for field mode.
Acceptance: 'npm run build' passes with no TS errors; 'npm run dev' serves and the main routes render with mock data (verify by
fetching the dev server HTML and, if possible, a quick headless check via node — do not open the user's browser); language
switch works; vitest unit tests for the offline queue and API client pass ('npm test').`,
  },
]

phase('Build')
const results = await parallel(BUILDERS.map(b => () =>
  agent(b.prompt, { label: b.label, phase: 'Build', schema: RESULT_SCHEMA })
    .then(r => r ? { ...r, wp: b.wp } : { wp: b.wp, failed: true })))

const ok = results.filter(Boolean)
for (const r of ok) log(`${r.wp}: ${r.failed ? 'AGENT FAILED' : `${r.tests_passed} passed / ${r.tests_failed} failed, ${(r.acceptance || []).filter(a => !a.met).length} acceptance gaps`}`)

phase('Integrate')
const integration = await agent(`${COMMON}
YOUR ROLE: wave-1 INTEGRATOR. Six builders just finished in parallel (WP1 data, WP2 measurement, WP4 DB/API, WP8a repair+priority,
WP9 web, WP10 GIS). Their self-reports (JSON) are below. Do NOT trust them blindly — verify by running things.
1. Run the full CPU test suite: '.venv/bin/python -m pytest -m "not gpu and not data" -q' from the root, plus each WP's own tests,
   plus 'cd web && npm run build && npm test' if web exists. Run ruff over measure repair priority geo server data ml scripts tests.
2. Fix cross-package breakage (import errors, mismatched signatures between WP4 server and WP2 measure / WP8a repair / WP10 geo
   files, seed_demo reading the geo files, pytest collisions like duplicate test module basenames). You MAY edit any non-frozen
   file now. Frozen contracts stay frozen.
3. Wire the server to the real libraries where WP4 left stubs: POST /measure -> measure.pipeline.measure_frame; plan endpoint
   is wave 2 so leave it. Run 'python -m server.app.db.init' + 'python scripts/seed_demo.py' on a fresh DB (with real geo files
   if WP10 produced them) and report counts. Start uvicorn briefly on a free port and hit /healthz and /v1/potholes.
4. Update the Makefile with the targets the builders suggested (keep existing ones working), add any new deps to pyproject.toml,
   and append a short 'Wave 1 status' section to docs/DECISIONS.md (what exists, verified numbers, known gaps).
5. Run 'make measure-demo' if WP2 provided it and report the key numbers.
Report honestly: what passes, what fails, and the precise gaps for wave 2 (WP3 training, WP5 geo services, WP6 pipeline,
WP7 simulator, WP8b ops services/routers, WP11 docs, WP12 e2e).

BUILDER REPORTS:
${JSON.stringify(ok, null, 1)}`, {
  label: 'Wave-1 integrator', phase: 'Integrate',
  schema: {
    type: 'object',
    properties: {
      tests_passed: { type: 'integer' }, tests_failed: { type: 'integer' },
      web_build_ok: { type: 'boolean' },
      seed_counts: { type: 'string' },
      measure_demo: { type: 'string' },
      fixes_made: { type: 'array', items: { type: 'string' } },
      remaining_gaps: { type: 'array', items: { type: 'string' } },
      wave2_notes: { type: 'string' },
    },
    required: ['tests_passed', 'tests_failed', 'fixes_made', 'remaining_gaps', 'wave2_notes'],
  },
})

return { builders: ok, integration }
