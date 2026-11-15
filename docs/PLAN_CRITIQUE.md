## Completeness critique of the GunthaMeter plan

### (a) Missing vs the brief
- **"Fixed for good" root-cause loop is absent.** `recurrence_watch` only re-opens a patch. Add: ≥2 reopenings in 12 months or Poor stretch rating → escalate from patching to a stretch-level item (`defect_cause` enum: waterlogging, utility_cut, base_failure, edge_drop; drainage/resurfacing recommendation; separate work-order type). Otherwise the system optimises re-patching, not permanence.
- **Cold-mix material QA**: no shelf-life/lot expiry, batch sample test (IRC:116 Cl. 6), or storage record. `lot_no` is captured but never validated.
- **Night operation**: TGSRTC runs 05:00–23:00; the luma gate just drops night frames. State that night passes count for `cam_active` but not for clean-pass verification, or repairs get "verified" by dark frames.
- **Cracks** (D00/D20) are trained but never used; drop them or feed the stretch rating.
- **Complaints** are today's main source (brief); MyGHMC/040-21111111 import is a "stub" with no CSV columns defined.
- **DPDP Act 2023** / face-plate blur deferred to production; citizen-facing frames in the demo PWA need blur or placeholders now.

### (b) Technical errors / unrealistic claims
- `yolo export format=litert quantize=16/8` — Ultralytics uses `format=tflite half=True` / `int8=True data=...`; `quantize` is not an argument. Same for `format=engine quantize=16` → `half=True`.
- HF ids `Ultralytics/YOLO26`, `mfranzon/pothole-yolo26`, `vinothvikas1987/pothole-detection-yolov8`, `depth-anything/DA3METRIC-LARGE` are unverified; WP0 must check each with `hf repo info`, record in LOCKFILE.md, and name fallbacks (`yolo11n-seg.pt` ultralytics assets, DA-V2-Metric-Outdoor-Base-hf).
- Pins (`ultralytics 8.4.211`, `celery 5.7.0`, `transformers 5.22.2`, `fastapi 0.162.5`, `vite 8.3.8`, `maplibre-gl 6.18.0`, `react-i18next 18`) are unverifiable; require `pip index versions` / `npm view` before writing pyproject/package.json.
- Celery has no native batching; "batch 16" needs an explicit chunking collector in `depth_measure`.
- `DBSCAN(metric='haversine', eps=6 m)` needs radians: `eps = 6/6_371_000`, inputs `[lat, lon]` in radians.
- `osmium extract` is the osmium-tool C++ binary; `pip install osmium` (pyosmium) does not provide it. Use a pyosmium bbox handler or a BBBike/Protomaps Hyderabad extract.
- `pmtiles extract` is the Go CLI; the python `pmtiles` package lacks it. Download the binary or use only the online style in demo.
- GTFS URL hard-codes a dated build (`mdb-3361-202607201528`); query the Mobility Database API for the latest.
- σ_d95 double-counts `surface_rms` (already inside σ_δ,i); drop the outer term.
- Plane rejection at >4° pitch delta conflicts with bus pitch under braking; use ~6° and log rather than reject.
- `kaggle` needs `~/.kaggle/kaggle.json`; not mentioned.

### (c) Ambiguities a coding agent will trip on
- Observation `polygon`: raw or undistorted pixels? Specify raw, remapped server-side after `cv2.undistort`.
- `imu.window_key` npz contents (`t_mono_ms[]`, `az[]`, rate) and `mono_ms`↔`trip_fix.ts` alignment undefined.
- `POST /edge/presign` under `DEMO_LITE`/`FsStorage`: define a `PUT /v1/ingest/frames/{key}` fallback that `presign()` returns.
- `PothFused` (planner input) undefined; define as the `pothole` row DTO.
- "Inverse-variance weighted median" is undefined; use weighted mean with MAD outlier rejection.
- `direction_bucket` for `oneway=yes` ways and heading noise at <5 km/h (hold previous heading).
- `ground_xz` frame unspecified: road frame, metres.
- Labour "1.0 h per additional 0.025 m³": define as `max(0, V_patch−0.025)/0.025`.
- Repair-run greedy route when no pothole within 500 m — unspecified.
- Water test "sky luma": define as mean luma of top 15 % rows.
- Monthly partitions: no pg_partman; add `partitions.ensure(ts)` called on ingest.
- `way_pass` population is missing from the WP6 chain; add `count_passes` on `trip-close`.
- Seed users/passwords and DEMO OTP (`123456`) unspecified.
- Bootstrap weights have different class names; `Segmenter` needs a `class_map`.
- WP9 "needs only WP4's OpenAPI spec" but WP4 generates it; commit a hand-written `server/openapi.json` in WP0.
- MVT endpoint: list `ST_AsMVT` properties and zoom thresholds for H3 heat vs points.

### (d) Needed for an end-to-end demo on the dev box
- **PostGIS is not installed; sudo unknown.** `setup_dev.sh` must test `sudo -n true`, else bootstrap micromamba (`curl micro.mamba.pm`), create `.mamba`, `initdb` on 5433, `CREATE EXTENSION postgis`; confirm conda-forge postgis availability. Provide a last-resort non-PostGIS path or the demo dies at step 1.
- **No-credential data path**: Kaggle/Roboflow/IDD need keys; `build_dataset.py` must work with RDD2022-India alone; eYantra val absent → 10 % RDD hold-out.
- **Do not gate the demo on training**: 100 epochs is hours; `make demo` uses bootstrap weights or `--quick` 5 epochs, with `fleet.py --use-labels` as default.
- **Deterministic simulated potholes**: `fleet.py` must place K virtual potholes at fixed chainages per route (seeded RNG) and serve pothole-labelled frames only there; otherwise "≥30 confirmed" is irreproducible.
- Staged foam-board validation needs a human; mark optional, not a demo gate.
- Commit cached Overpass POIs and GHMC zones as GeoJSON; use a direct Hyderabad PBF (BBBike) instead of the ~1 GB southern-zone file.
- Eager Celery loads DA3 inside uvicorn; specify a lazy singleton and `CUDA_VISIBLE_DEVICES`.
- Make `demo-lite` the default `make demo`; build `make measure-demo` (no services) first in WP2 as the guaranteed jury fallback.
- e2e test must reset the DB (`alembic downgrade base && upgrade head`) and fix ports 8000/5173/5433/6379.