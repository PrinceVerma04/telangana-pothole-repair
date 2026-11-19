# RoadFix — FINAL IMPLEMENTATION PLAN
## "Potholes, Fixed for Good" — Telangana R&B / Municipal Administration challenge

Base: DESIGN 2 (RoadFix, measurement-first; chosen by 2 of 3 judges). Grafted: DESIGN 3's operations layer (audit log, AE→DEE→EE→Zonal Commissioner escalation, daily per-circle repair-run indent, jurisdiction queue, contract/DLP evidence packages, pluggable admin units, PostGIS nearest-way map-matching) and DESIGN 1's fleet layer (distance-triggered sampling, 1 Hz `trip_fix` telemetry with `cam_active`, jolt-only observations, lateral-offset cluster split, 500 m same-owner work-order batching, signed OTA config, cavity-vs-cut volume both stored). Every judge critique is resolved; the resolution is marked **[FIX]** where it occurs.

Verified on the dev box (2026-11-12): `.venv` = Python 3.11.18, torch 2.11.0+cu128, `torch.cuda.is_available()==True` on NVIDIA GeForce RTX 5080 (16,303 MiB, driver 595.99.02). System Python 3.14.4, Node v22.22.1. PostgreSQL/Redis/MinIO/OSRM are NOT installed. Project directory contains only `.venv`.

---

## 1. Executive summary (for the Telangana brief)

RoadFix converts the TGSRTC Greater-Hyderabad fleet (~2,800 buses, ~270 km/day each) and GHMC/R&B inspection vehicles into a continuous, calibrated road-measuring instrument, and closes the loop from detection to a verified, warranty-tracked repair.

1. **Locate.** A ₹10–12k Android phone (pilot) or a Jetson/Pi edge box (production) behind the windscreen runs a YOLO26-seg pothole segmenter on-vehicle, triggered every ≥4 m of travel, and uploads only pothole keyframes + masks + GPS/IMU (<1 GB/bus/month) over depot Wi-Fi or a ₹100/month IoT SIM. 1 Hz trip telemetry with a `cam_active` flag makes "this bus passed here with a working camera and saw nothing" provable.
2. **Measure (the user's hard requirement).** On the server GPU a metric monocular depth model (Depth Anything 3 METRIC-LARGE, Apache-2.0; DA-V2-Metric fallback) fed the calibrated focal length is back-projected to a point cloud; the road surface is fitted by RANSAC plane + local quadratic (handles crowned/rutted Indian roads) and scale-locked to the tape-measured camera height; per-pixel depth below the surface is integrated over the mask to give depth_p95, depth_max, area, volume, length/width — each with a propagated σ, a quality score and a `method` tag (mono / mono+imu / stereo / phone-LiDAR / manual gauge / citizen). Frames from the approach are fused inverse-variance; the wheel IMU jolt corroborates existence; water-filled holes are flagged `depth=unknown`, never reported as a floor. Stereo (OAK-D Pro/ZED 2i) on 5–10 inspection vans and iPhone-LiDAR + straightedge ground truth provide billing-grade numbers and per-device calibration certificates.
3. **Prioritise.** Detections are map-matched (PostGIS nearest-way + heading; OSRM optional), clustered per way+direction+lane, confirmed on ≥2 independent passes, graded by IRC:82-2015 Cl. 7.5.3.4 (Small ≤25 mm/200 mm, Medium 25–50 mm/500 mm, Large >50 mm/>500 mm) and scored 0–100 with per-ULB editable weights: severity, exposure (OSM class × real bus pass counts), vulnerability (schools/hospitals/bus stops/metro within 200 m), age vs the GHMC 24-h charter, monsoon/IMD rain and waterlogging hotspots.
4. **Repair permanently and economically.** Volume → IRC square-cut design (margin 125 mm, d_design = max(depth_p95, 50 mm), 75 mm layering with CRM below) → IRC:116 / IRC:SP:100 cold-mix bags (±σ), tack emulsion, labour, ₹ cost from an editable rate card; jet patcher / infrared / hot mix recommended by rules. Work orders are batched into a daily per-circle "repair run" indent, routed to the owning agency (GHMC / Cyberabad MC / Malkajgiri MC / R&B / HMDA / NHAI / Secunderabad Cantonment) and CRMP contractor with SLA, escalation ladder and audit log on every transition.
5. **Verify and hold to account.** The same buses verify the repair (≥3 clean passes within 14 days + 10 % engineer QC → `verified_closed`); invoices can only reference verified-closed orders; re-detection inside the defect-liability period creates an evidence package (before/after/re-detection) and a contractor-liable rework order. Survival curves per contractor/material/method show what "fixed for good" actually means.

Expected impact: faster triage (hours not complaint cycles), objective material quantities (vs GHMC's reported ₹7,396/pothole against ₹1,600 budgeted), lower recurrence through DLP enforcement, and a Telugu/Hindi/English offline-first PWA for ward engineers. Everything runs end-to-end on the dev machine today in demo mode with public Indian data (RDD2022-India, eYantra Indian-road masks, BharatPotHole, TGSRTC GTFS, OSM Hyderabad) and a simulated bus fleet.

---

## 2. System architecture

### 2.1 Principles
- **Measurement-first**: every number carries value + σ + method + quality; billing is gated on certified devices and minimum quality.
- **Edge detects, server measures**: a phone cannot run a 0.35 B depth model; the server can re-measure archived frames when models improve.
- **Bandwidth, not compute, is scarce**: never upload video; ≤3 keyframes per pothole track.
- **Immutable observations, derived entities**: potholes, severities, priorities, plans are recomputable.
- **Operations-grade from day one**: audit log, escalation, indent, jurisdiction queue, multi-tenant (`agency`).

### 2.2 ASCII diagram

```
 ┌──────────────────────────── VEHICLE (bus / van) ────────────────────────────┐
 │  Camera 1280x720@30fps ──> YOLO26n-seg (LiteRT FP16 / TensorRT)             │
 │  GNSS 1Hz + IMU 100Hz ───> distance trigger (>=4 m, >=5 km/h)               │
 │  ├─ trip_fix 1Hz {lat,lon,acc,speed,heading,az_rms,cam_active}  (~60 B)      │
 │  ├─ observation {frame.jpg 1280x720 q75, masks, bbox, conf, gps, imu}        │
 │  └─ jolt observation (IMU peak, no detection)                                 │
 │  SQLite/Room store-and-forward ──> depot Wi-Fi / 4G IoT SIM (resumable S3)   │
 └───────────────────────────────┬──────────────────────────────────────────────┘
                                 │ HTTPS  POST /v1/ingest/*   (demo: edge/sim)
 ┌───────────────────────────────▼──────────────────────────────────────────────┐
 │ SERVER (dev RTX 5080 today; NIC/MeghRaj GPU VM later)                         │
 │  FastAPI ingest ──> MinIO/S3 (frames, masks, depth PNG16)  + PostgreSQL/PostGIS│
 │        │                                                                       │
 │        ▼ Celery queue "gpu" (pool=solo, batch 16)                              │
 │  [verify: RF-DETR-Seg] -> [DA3METRIC depth] -> [plane+quadratic, height lock] │
 │        -> [mask integral: depth_p95/max, area, volume, σ, quality, water flag] │
 │        -> measurement rows (method=mono)                                       │
 │        ▼ Celery queue "cpu"                                                    │
 │  [ground point = GPS + IPM offset] -> [map-match: PostGIS nearest-way+heading] │
 │        -> [track fusion (inverse-variance)] -> [DBSCAN per way+dir, lane split]│
 │        -> pothole entity + state machine -> [IRC severity] -> [priority score] │
 │        -> [repair plan: cut, layers, bags±σ, ₹] -> [jurisdiction + contract]   │
 │        ▼ Celery queue "ops"                                                    │
 │  [SLA timers, escalation AE->DEE->EE->ZC] [clean-pass verifier] [DLP watcher]  │
 │  [daily repair-run indent per circle] [notify stubs] [audit_log everything]    │
 └───────────────────────────────┬──────────────────────────────────────────────┘
                                 │ JSON + MVT tiles (gzip/brotli)
 ┌───────────────────────────────▼──────────────────────────────────────────────┐
 │ WEB (React 19 + Vite 8 + MapLibre 6 PWA, te/hi/en, offline Dexie queue)       │
 │  Zonal dashboard | Pothole detail (depth profile, σ) | Priority queue          │
 │  Work orders + Repair runs (indent print) | Field mode (before/after photos)   │
 │  Contractor portal + scorecards | Citizen report/track | Admin | Public ward   │
 └──────────────────────────────────────────────────────────────────────────────┘
   Citizen intake: web form / WhatsApp webhook stub / MyCURE CSV ──> same cluster logic
```

### 2.3 Deployment topology
- **Tier A (bus)**: Android phone ₹8–12k (Snapdragon 6-series+, 4 GB) in shaded, fan-cooled windscreen mount, 24 V→5 V isolated DC-DC with ignition-off delay (₹600–800), IoT SIM ₹100/mo. Production alternative: Raspberry Pi 5 8 GB + Hailo-8L (₹16–19k) or Jetson Orin Nano Super on industrial carrier (RDP CARINA / Seeed J3011, −10..60 °C; ₹55–75k) + IMX296 global-shutter cam (₹4–8k) + u-blox GNSS with NavIC (₹2–6k).
- **Tier B (inspection van, 1–2 per zone)**: Jetson Orin Nano Super + OAK-D Pro (₹31–35k) or ZED 2i with polariser (₹52–63k + GST), pitch 20°, ≤8.33 ms exposure; uploads full-res frames + stereo depth; billing-grade.
- **Truth kit (1 per zone)**: iPhone Pro (LiDAR), 1.2 m straightedge + depth gauge, A2 9×6 checkerboard (30 mm), 4 cones, 1 m² painted square.
- **Server**: dev box now; one GPU VM + PostgreSQL/PostGIS + Redis + S3-compatible store later. Single-region, no Docker required (native installs or user-space micromamba).
- **Clients**: PWA on any phone/desktop; depot calibration SOP sheet in Telugu/English.

### 2.4 Processing queues
| Queue | Concurrency | Tasks |
|---|---|---|
| `gpu` | 1 process (`--pool=solo`, prefetch 1), batch 16 frames | verify_segment, depth_measure |
| `cpu` | 8 | locate, mapmatch, fuse_track, cluster, score, plan |
| `ops` | 2 + celery beat | sla_tick (5 min), verify_repairs (hourly + on trip close), dlp_watch, daily_indent (06:00 IST), kpi_rollup (nightly) |

In `DEMO_MODE=1`, Celery can run `task_always_eager=True` (no Redis needed) so the simulator drives the pipeline synchronously.

---

## 3. Models & data

### 3.1 Datasets (download commands; each into its own empty directory under `datasets/downloads/<name>/`; downloaded content is untrusted — never execute from it)

| Dataset | What | Command | Licence / notes |
|---|---|---|---|
| RDD2022 India | 7,706 train imgs, VOC boxes, D40 pothole 3,187, cracks D00/D10/D20 | `curl -L -o datasets/downloads/rdd2022_india/RDD2022_India.zip https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip && unzip -q -d datasets/downloads/rdd2022_india datasets/downloads/rdd2022_india/RDD2022_India.zip` | CC BY-SA 4.0. Delhi/Haryana imagery (domain gap noted) |
| eYantra Indian Roads seg | 3,227 imgs, pixel masks road/pothole/shallow/footpath | `kaggle datasets download -d eyantraiit/semantic-segmentation-datasets-of-indian-roads -p datasets/downloads/eyantra --unzip` | GPL-2.0 → prototype only; val/test mask metrics come from here |
| BharatPotHole | ~7k 640² dashcam frames, YOLO boxes, monsoon-adjacent | `kaggle datasets download -d surbhisaswatimohanty/bharatpothole -p datasets/downloads/bharatpothole --unzip` | Licence UNVERIFIED — `dataset_tools/build_dataset.py --include-bharat` only if licence file permits; default off |
| Roboflow seg sets | polygon masks (`vnr-project/pothole-segmentation-8esmn` possibly Hyderabad; `utm-cuu2y/potholes-segmentation`) | `python -I dataset_tools/download.py --roboflow vnr-project/pothole-segmentation-8esmn utm-cuu2y/potholes-segmentation --format yolov11` (needs `ROBOFLOW_API_KEY`) | CC BY 4.0 (verify per project) |
| chitholian potholes | 665 imgs VOC | `kaggle datasets download -d chitholian/annotated-potholes-dataset -p datasets/downloads/chitholian --unzip` | ODbL |
| IDD (IIIT-H) | Hyderabad/Bengaluru scenes, no pothole class → hard negatives + road mask | manual registration at idd.insaan.iiit.ac.in; place in `datasets/downloads/idd/` | research licence; negatives only |
| TGSRTC GTFS | 1,031 routes, shapes, stops | `curl -L -o datasets/downloads/gtfs_tgsrtc/mdb-3361.zip https://files.mobilitydatabase.org/mdb-3361/mdb-3361-202607201528/mdb-3361-202607201528.zip` | open data |
| OSM Hyderabad | roads, POIs, lit tag | `curl -L -o datasets/downloads/osm/southern-zone-latest.osm.pbf https://download.geofabrik.de/asia/india/southern-zone-latest.osm.pbf && osmium extract -b 78.1,17.1,78.8,17.7 -o datasets/downloads/osm/hyd.osm.pbf datasets/downloads/osm/southern-zone-latest.osm.pbf` (osmium via `pip install osmium` fallback: `pyosmium` filter) | ODbL |
| Telangana boundaries | districts/mandals GeoJSON | `git clone --depth 1 https://github.com/gggodhwani/telangana_boundaries datasets/downloads/tg_boundaries` | OGD/community; GHMC wards = community GeoJSON, marked unofficial, swappable |
| Overpass POIs | schools, hospitals, markets, metro stations in bbox | `python -I dataset_tools/download.py --overpass` (query `amenity~"school|hospital|clinic|marketplace"`, `railway=station`+`station=subway`) | ODbL |
| Basemap | Hyderabad PMTiles | `pmtiles extract https://build.protomaps.com/$(date +%Y%m%d).pmtiles app_data/tiles/hyd.pmtiles --bbox=78.1,17.1,78.8,17.7` (fallback online style `https://tiles.openfreemap.org/styles/liberty`) | ODbL |

### 3.2 Checkpoints (Hugging Face repo ids; `hf download <repo>` or `hf_hub_download`)
- Segmentation base: `Ultralytics/YOLO26` → `yolo26n-seg.pt`, `yolo26s-seg.pt` (AGPL-3.0; Enterprise licence flagged for production; `Segmenter` interface allows RF-DETR-Seg Apache swap).
- Bootstrap (day-0 demo before fine-tune finishes): `mfranzon/pothole-yolo26` (YOLO26m, dashcam); `vinothvikas1987/pothole-detection-yolov8` (Apache 2.0, `best.pt`, pothole mAP50 0.782).
- Verifier/teacher: RF-DETR-Seg-Small via `pip install rfdetr` (weights auto-download, Apache-2.0).
- Pseudo-mask generator: `facebook/sam2.1-hiera-small` (transformers `Sam2Model`, box prompts) **[FIX: SAM2 instead of eYantra-trained seg model for RDD box→mask]**.
- Depth primary: `depth-anything/DA3METRIC-LARGE` (Apache-2.0; `pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3`). Scale convention is **configurable** (`scale_convention: fx_over_300 | fx | relative`) and empirically validated on the staged target; the height lock makes it non-critical.
- Depth fallback: `depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf` (transformers) else repo code + `depth-anything/Depth-Anything-V2-Metric-VKITTI-Base`.
- Edge/ONNX depth option (Tier B Jetson): `onnx-community/metric3d-vit-small` (BSD-2).
- Avoid: UniDepth, Depth Pro (non-commercial).

### 3.3 Unified dataset build (`dataset_tools/build_dataset.py`)
Classes (`configs/classes.yaml`): `0 pothole, 1 pothole_water, 2 alligator_crack, 3 crack, 4 manhole, 5 speed_breaker, 6 utility_cut` (4–6 exist to absorb false positives and are dropped from reporting).
- RDD: D40→pothole, D20→alligator_crack, D00+D10→crack **(D10 merged: only 68 labels)**; boxes→SAM2 masks (box prompt, keep mask if IoU(mask bbox, box) ≥ 0.6 else fall back to box polygon); 2 % random sample exported to Label Studio for QC.
- eYantra: masks→polygons; `shallow` class → `pothole_water` when mean mask brightness/specularity high else `pothole`; road mask kept as auxiliary `road.png` for plane-fit road prior training (optional).
- Negatives: IDD frames and RDD frames without D40 as empty-label images (≥20 % of train); manhole/speed-breaker/utility-cut polygons labelled on ~300 frames via Label Studio (WP-time-boxed; if not done, classes stay but have few samples).
- Split: train = RDD-train + eYantra-train + Roboflow + negatives; **val = eYantra test masks (honest mask metrics) + 10 % RDD-train held out (box metrics)**; test = RDD-India D40 held-out 500 imgs.
- India augmentations (`ml/augment/india.py`, albumentations): RandomRain, RandomSunFlare, RandomShadow, MotionBlur (≤7 px ≈ 2.5 cm at 40 km/h), RandomBrightnessContrast (dusk), Downscale, ImageCompression(60–90), wet-asphalt specular overlay.

### 3.4 Training recipe (ultralytics 8.4.211)
```bash
# edge model
yolo segment train model=yolo26n-seg.pt data=configs/model/india_pothole_seg.yaml imgsz=640 epochs=100 batch=32 \
  mosaic=1.0 close_mosaic=10 hsv_v=0.5 degrees=3 fliplr=0.5 scale=0.4 optimizer=auto patience=25 device=0 project=runs name=y26n_seg_india
yolo segment train model=yolo26s-seg.pt data=... imgsz=960 epochs=80 batch=16 name=y26s_seg_india   # server/van model
# export
yolo export model=runs/y26n_seg_india/weights/best.pt format=litert quantize=16 imgsz=640
yolo export model=runs/y26n_seg_india/weights/best.pt format=litert quantize=8 imgsz=640 data=configs/model/india_pothole_seg.yaml
yolo export model=runs/y26s_seg_india/weights/best.pt format=onnx dynamic=False imgsz=960
yolo export model=runs/y26s_seg_india/weights/best.pt format=engine quantize=16 imgsz=960   # dev 5080 only; engines are device-specific
```
RF-DETR-Seg-Small verifier: `ml/train_rfdetr_seg.py` (COCO-format export of the same set; 30 epochs; `rfdetr` API `RFDETRSegSmall().train(dataset_dir=..., epochs=30, batch_size=8)`). Verification rule: a detection survives if verifier class ∈ {pothole, pothole_water} with mask IoU ≥ 0.4 against the edge mask, or verifier conf ≥ 0.6 anywhere inside the dilated edge bbox; verifier mask replaces edge mask when IoU ≥ 0.4.

Eval metrics (`ml/eval.py`, written to `reports/model_eval.json`):
- Box mAP50 / mAP50-95 on RDD-India D40 test; target ≥ 0.75 / ≥ 0.45.
- Mask mAP50 and mean mask IoU on eYantra test; target mask IoU ≥ 0.70.
- FP rate per 1,000 frames on IDD negatives and on a manhole/speed-breaker hard-negative set; target ≤ 5 FP/1k frames after verifier.
- Latency: ms/frame FP16 on RTX 5080 (ONNX/TRT), and LiteRT on one Android phone if available.
- Edge/verifier agreement rate (feeds active-learning queue).

---

## 4. Depth & volume measurement

### 4.1 Camera model and calibration
Pinhole with distortion: K = [[fx,0,cx],[0,fy,cy],[0,0,1]], distortion (k1,k2,p1,p2,k3). All frames and masks are undistorted (`cv2.undistort` / `remap`) before anything else; K becomes the undistorted K′.

**Calibration procedure (per device, produces `calib_cert.json`):**
1. Intrinsics (once per camera model): 9×6 checkerboard, 30–40 mm squares printed on A2 foam board, 20–25 poses covering the frame, fixed focus/zoom locked. `cv2.calibrateCamera`; accept if RMS reprojection < 0.3 px. Store fx, fy, cx, cy, dist, image size.
2. Extrinsics (per vehicle, ~10 min at depot): tape-measure camera height `h_true` (bus dash ≈ 2.2–2.5 m, car ≈ 1.3–1.5 m; record ±1 cm). Place 4 cones at 3/5/8/12 m along the lane centre, measure lateral offsets; `cv2.solvePnP` on cone-base pixels → pitch, roll, yaw. Accept if the 1 m² painted square (or 750 mm IRC cone) at 5 m reconstructs within 5 %.
3. Online per-trip self-checks: (a) rolling median of RANSAC-estimated camera height ĥ vs h_true → `scale_drift = |ĥ_median·s_nominal − h_true|/h_true`, alert > 5 %; (b) vanishing point from lane markings → pitch/yaw drift > 3° alert; (c) lane-width prior 3.5 ± 0.25 m only on marked arterials (IRC:86) → roll/height sanity.
4. Demo fallback calibration (`configs/cameras/phone_demo.json`, `car_default.json`, `bus_default.json`): fx = fy = image_width × 1.0 (≈ 60° HFOV phone), cx,cy = centre, dist = 0, h = 1.4 m (car) / 2.4 m (bus), pitch 12°/18°; flagged `certified=false` so its measurements are usable for prioritisation, never for Measurement Book billing.

Certificate JSON: `{device_id, fx, fy, cx, cy, dist[5], img_w, img_h, reproj_rms_px, h_true_m, pitch_deg, roll_deg, yaw_deg, known_object_err_pct, calibrated_at, calibrated_by, certified: bool, drift_history: [{trip_id, scale_drift, pitch_drift}], truth_fit: {k_sigma, g_imu, n_truth, depth_mae_mm, volume_mape}}`.

### 4.2 Metric depth
Z_raw(u,v) = DepthBackend.predict(I_undist, fx′). For DA3METRIC: `Z_raw = fx′ · net_out / 300` (convention `fx_over_300`; alternatives `fx`, `relative`, chosen by `measure/depth_backends/da3_metric.py` config, validated once on the staged target by minimising |ĥ − h_true| before scale lock). Inference at long side 1036 px (DA3 patch multiple) on the full 1280×720 frame; the depth map is bilinearly resized back to full resolution. Backend interface: `predict(img_bgr: np.ndarray, fx_px: float) -> (depth_m: np.ndarray[H,W], conf: np.ndarray[H,W] | None)`.

### 4.3 Back-projection
For pixel (u,v): X = (u − cx′)·Z/fx′, Y = (v − cy′)·Z/fy′, p = (X, Y, Z) in camera frame (x right, y down, z forward).

### 4.4 Road surface: RANSAC plane + local quadratic + scale lock
1. Road candidate pixels R₀: lower 65 % of the image, minus every detection mask dilated by 20 px, minus pixels with Z_raw outside [2.5, 14] m, minus (optional) road-seg "not road" pixels; subsample to ≤ 20,000 points.
2. RANSAC plane n·p + d = 0 (‖n‖=1): numpy implementation (3-point sampling, 1,000 iterations, inlier threshold 15 mm, early exit at 99 % confidence), least-squares refit on inliers. `plane_inlier_ratio`, `plane_rms_mm` recorded. Reject frame if inlier ratio < 0.5 or RMS > 25 mm, or if plane pitch differs from calibrated pitch by > 4° (frame `quality=0`, reason stored). open3d `segment_plane` optional (`MEASURE_USE_OPEN3D=1`).
3. Scale lock: ĥ = |d| (point-to-plane distance from camera origin). `s = h_true / ĥ`, clipped to [0.5, 2.0] else reject. Multiply all Z (and hence p) by s. Record `scale_factor=s`. This cancels the 5–30 % absolute-scale error of monocular models on an unseen camera (Gamma-from-Mono insight).
4. Road frame: rotate so the plane normal is +Y_up, forward +Z along the vehicle axis projected onto the plane, lateral +X. Coordinates (x, y, z) per pixel.
5. **Local quadratic surface** (crown/rutting fix): take ring region Rring = pixels whose road-frame (x,z) lie within 0.3–1.2 m of the mask footprint boundary and are not in any mask; fit `y_s(x,z) = a + b·x + c·z + d·x² + e·z² + f·x·z` by least squares (if < 300 ring pixels or condition number > 1e6, fall back to the global plane, `surface_model='plane'`). Record `ring_rms_mm`.

### 4.5 Depth map within the mask
For each mask pixel i (mask M = verifier mask, eroded 1 px to avoid edge bleed, with depth confidence above threshold): δ_i = y_s(x_i, z_i) − y_i (positive = below the road surface), clipped at ≥ 0.

Per-pixel ground footprint (IPM-equivalent): a_i = (Z_i / fx′)·(Z_i / fy′) / cos θ_i, where θ_i = angle between the viewing ray r_i and the surface normal n (cos θ_i = |r̂_i · n̂|).

### 4.6 Per-frame metrics
- A = Σ_i a_i (m²)
- V = Σ_i δ_i · a_i (m³; reported in litres)
- depth_mean = V / A
- depth_p95 = 95th percentile of δ_i weighted by a_i (**the design metric**)
- depth_max = 99th percentile of δ_i weighted by a_i (robust max) **[FIX: single consistent definition]**
- Length L, width W: minimum-area rectangle of the footprint (x_i, z_i) via `cv2.minAreaRect` on the projected points; L = extent along travel.
- Range D = median Z_i; lane_offset = median x_i (metres from camera axis; converted to lane-centre offset using lane width 3.5 m and the camera's lateral mounting offset from the certificate).
- Ground point of the pothole = median (x_i, z_i) projected through heading onto the map (§5).

### 4.7 Uncertainty model (stored with every measurement)
- Per-pixel depth noise along the ray: σ_Z,i = k · Z_i², k initial 4×10⁻⁴ m⁻¹ (σ_Z = 4 cm at 10 m); re-fitted per device from the truth set.
- Cavity-depth error: σ_δ,i = sqrt( (σ_Z,i · cos θ_i)² + surface_rms² + (δ_i · σ_s)² ), where surface_rms = ring_rms (or plane_rms) and σ_s = relative scale uncertainty = sqrt((0.01/h_true)² + (plane_rms/ĥ)²) (tape ±1 cm plus plane-fit height error).
- depth_p95 σ: σ_d95 = sqrt( mean over the top-5 % pixels of σ_δ,i² + surface_rms² ).
- Area: σ_A = sqrt( (2·A·σ_s)² + (P · ℓ_px)² ) where P = footprint perimeter (m) and ℓ_px = mean footprint edge length at the boundary (±1 px mask boundary error) **[FIX: boundary term added]**.
- Volume (conservative, correlated depth errors): σ_V = sqrt( (Σ_i a_i σ_δ,i)² + (depth_mean · σ_A)² ).
- Quality score q ∈ [0,1] = product of gates: range gate (1 at 3–7 m, linear to 0.3 at 12 m, 0 beyond 12 m), blur (Laplacian variance normalised), illumination (mean luma in [40,220]), mask confidence, plane inlier ratio, mask area ≥ 400 px. **[FIX: 3–12 m gate, best weight 3–7 m; 15 m dropped.]**

### 4.8 Multi-frame fusion (track → pass)
Edge assigns BoT-SORT `track_id`s (ultralytics tracker) and uploads ≤ 3 keyframes per track (first sight ≥ 10 m, mid, nearest ≥ 3 m). Server re-associates by predicted ground point ± 1.5 m if ids are missing (simulator/citizen).
For each metric m ∈ {depth_p95, depth_max, A, V, L, W}: reject frames with q < 0.2; reject outliers |m_j − median| > 3σ_j; fused m̂ = Σ w_j m_j / Σ w_j with w_j = q_j / σ_j²; fused σ̂ = sqrt(1/Σ(1/σ_j²)) floored at 3 mm (depth), 5 % (A, V). Result: one `pass_measurement` per (trip, pothole).

### 4.9 IMU corroboration **[FIX: existence corroboration, depth only after per-vehicle fit]**
Edge records az (vertical accel) at 100 Hz, uploads `az_rms_1s`, `az_peak_g` and a ±1 s window per observation. Predicted pass time t_pass = t_frame + D / v. If |lane_offset − wheel_track| < 0.35 m (left or right wheel track from certificate) and max |az − g| in [t_pass − 0.5 s, t_pass + 0.5 s] > 0.25 g (speed-normalised: threshold × (v/8 m/s)^0.5), set `imu_corroborated=true` (raises confirmation confidence; filters shadow/wet-patch FPs). Depth hint: `imu_depth_mm = g_v · peak_g` only when the certificate has `g_imu` fitted on ≥ 20 truth potholes; otherwise `imu_depth_mm=null`. In fusion the IMU hint enters with σ = 25 mm (low weight); it is never the sole source for a severity class. Jolt-only observations (no camera detection, peak > 0.35 g, speed > 15 km/h) create `source='jolt'` observations that count as passes for candidate creation but never for `confirmed` on their own.

### 4.10 Water-filled potholes
`water_filled = true` if any of: verifier class `pothole_water`; depth_p95 < 8 mm AND specular score high (mask mean luma > 0.6·sky luma AND texture (local std) < threshold) AND area > 0.05 m²; Tier B polarised-stereo disagreement > 30 %. Then: `depth_p95 = null`, `depth_quality='unknown_water'`, area kept; severity lower bound = Medium if width ≥ 200 mm, Large if width > 500 mm or imu peak > 0.5 g or citizen complaints ≥ 2; schedule `remeasure_after` = last rain + 48 h or inspection-van visit. The water surface is never reported as the floor.

### 4.11 Tier B stereo and truth tiers
Stereo backends (`stereo_oakd.py`, `stereo_sgbm.py`) output metric Z directly; steps 4.4–4.8 are still applied (tilt removal, scale sanity: s must be within [0.9, 1.1] else flag). `method='stereo'`, k = 1×10⁻⁴ (fitted). Phone-LiDAR `.ply`/`.obj` meshes and manual straightedge+gauge readings are ingested by `measure/truth.py` as `method='lidar_phone'` (σ ≈ 5 % volume) and `method='manual_gauge'` (σ = 3 mm depth).

### 4.12 Validation plan (three levels, all with pass criteria)
1. **Synthetic analytic bowl (unit test, runs in CI, no model)**: `tests/test_volume_synthetic.py` renders a planar road at known K, h, pitch with a spherical-cap cavity (R, depth d) at 4/7/10 m and also a crowned road (parabolic crown 2 %); the "depth model" is the exact rendered Z (+ optional Gaussian noise k·Z²). Pass: volume within 3 % (noise-free), depth_p95 within 2 mm; with noise, true value inside ±2σ in ≥ 90 % of 200 trials (calibrates the σ formulas).
2. **Staged known-depth target (dev box, one afternoon)**: foam-board/plywood "potholes" of 20/40/70 mm depth and known area, filmed with a checkerboard-calibrated phone at h = 1.4 m from 3–12 m while walking/driving. `tools/staged_pothole_truth.py` runs the full chain; report depth MAE, volume MAPE per range bin, and σ-coverage. Pass for demo: fused depth MAE ≤ 10 mm, IRC class accuracy ≥ 90 %, volume MAPE ≤ 25 %. This is the jury credibility artefact.
3. **Field truth campaign (pilot)**: 50–100 potholes per zone measured by inspector (straightedge + gauge) and iPhone LiDAR; `measure/truth.py` fits k and g_v per device, issues certificate; publishes MAE/MAPE per range bin and per tier. Expected (uncertain, no published bus-mounted study): mono single-frame ±10–25 mm / ±20–40 % volume; fused ±5–15 mm / ±15–25 %; stereo ±12 % depth within 8 m.

---

## 5. Geolocation, dedup, map-matching

1. **Ground point**: pothole position = GPS fix at frame time advanced by the GNSS latency estimate (default 0.3 s × speed) then offset by (z_med forward, x_med lateral) rotated by heading: lat/lon via `pyproj` local azimuthal projection (EPSG:32644 for metric maths). Accuracy: σ_pos = sqrt(gps_accuracy_m² + (0.05·D)²).
2. **Map-matching (shipped default)**: PostGIS `road_segment` table from OSM ways (split at intersections, `highway` ∈ motorway…residential|unclassified|service). Query: nearest segments within 25 m (`<->` KNN on geography), keep those whose bearing at the nearest point is within ±45° of the vehicle heading (both directions for two-way ways), choose minimum of (distance/σ_pos + 0.5·heading_diff/45°). Output `way_id`, `direction_bucket` (0/1 = along/against the way's digitised direction), `chainage_m` = `ST_LineLocatePoint × ST_Length`. Trip-level smoothing: a Viterbi pass over consecutive fixes penalising way switches (transition cost 1.0 unless connected). OSRM `/match` is a drop-in (`geo/mapmatch_osrm.py`) when `OSRM_URL` is set; Valhalla Meili documented for production.
3. **Dedup key**: (`way_id`, `direction_bucket`, H3 res-12 cell of the ground point, lane band = round(lane_offset / 1.75 m)).
4. **Clustering** **[FIX: candidate reachable]**: DBSCAN(metric='haversine', eps = 6 m, **min_samples = 1**) over verified observations in the same (`way_id`, `direction_bucket`) within a rolling 30-day window; then split clusters whose lane_offset spread > 1.5 m (k-means k=2 on lane_offset). min_samples=1 means every verified observation yields/attaches to an entity; FP flooding is controlled by (a) the verifier gate, (b) `candidate` entities hidden from dispatch views, (c) auto-archive of candidates with no second pass in 30 days and ≥ 5 clean passes.
5. **Pothole state machine**: `candidate` (1 pass) → `confirmed` (≥ 2 passes from ≥ 2 distinct trips/devices, or 1 pass + citizen report within 25 m, or 1 pass with `imu_corroborated` and q ≥ 0.6) → `planned` → `work_ordered` → `acknowledged` → `in_progress` → `repaired_claimed` → `verification_pending` → `verified_closed` | `qc_failed` → (re-detect) `reopened` → `recurrence_watch` (12 months) ; any state → `rejected` (engineer marks FP; feeds retraining) ; `archived`.
6. **Pothole aggregates**: fused over pass_measurements with inverse-variance weighted median; `depth_class_mm` from fused depth_p95; `water_filled_ratio`; `passes_seen / passes_total` (passes_total counted from `trip_fix` rows with `cam_active=true` passing within 8 m along the same way+direction with lane band within ±1 — **[FIX: verification radius matches cluster eps and lane]**).
7. **Repair verification**: after `repaired_claimed`, a pass is "clean" if cam_active, daylight-or-quality ok (`az_rms` and luma gates), within the same key, and no verified detection within eps. ≥ 3 clean passes within 14 days from ≥ 2 trips AND (10 % random engineer QC sample passed, or no sample drawn) → `verified_closed`.

---

## 6. Prioritisation score (per-agency editable, `priority_config` JSONB, versioned; breakdown stored on every pothole)

P = 100 · clip( wS·S + wE·E + wV·V + wT·T + wM·M , 0, 1 ) with defaults wS=0.30, wE=0.25, wV=0.20, wT=0.15, wM=0.10.

- **S (IRC:82-2015 Cl. 7.5.3.4 severity)**: Large (depth_p95 > 50 mm or W > 500 mm) → 1.0; Medium (25–50 mm or W 200–500 mm) → 0.6; Small → 0.3; water-filled unknown-depth → max(0.6, rule in §4.10, 0.8 if A ≥ 0.3 m²); +0.2 if in wheel path (|lane_offset − wheel track| < 0.35 m) or ≤ 30 m from a junction node; +0.1 if `imu_corroborated`; clip 1.0.
- **E (exposure)** **[FIX: explicit normalisation]**: E = 0.5·C + 0.5·B, where C = OSM class weight (trunk/primary 1.0, secondary 0.8, tertiary 0.6, residential/unclassified 0.4, service 0.2) and B = min(1, bus_passes_per_day / 50); bus_passes_per_day = observed `trip_fix` passes on the way (rolling 7 days) if available else GTFS trips/day through the way (from shapes ∩ way) / 50.
- **V (vulnerability)**: V = min(1, 1 − Π(1 − v_i)) over POIs within 200 m: school 0.5, hospital/clinic 0.5, bus stop (GTFS stops.txt) 0.3, metro station 0.3, market 0.2; +0.2 if OSM `lit=no` or no lighting data on a non-arterial; clip 1.0.
- **T (age vs SLA)** **[FIX: uses SLA hours, not /14]**: T = min(1, hours_since_confirmed / sla_hours) + 0.3·[breached] + min(0.3, 0.1·complaint_count); clip 1.0. sla_hours from `contract.sla_rules` or agency default: arterial 24 h (GHMC Citizen's Charter 2023), other roads 72 h in monsoon window, 168 h off-season; R&B state roads 168 h.
- **M (monsoon/weather)**: base 0.5; 0.8 during the declared monsoon window (Jun–Sep SW, Oct–Nov NE); 1.0 if IMD district rain in last 72 h > 30 mm (IMD feed adapter; manual flag in demo); +0.2 if inside a registered waterlogging hotspot polygon; +0.1 if water_filled; clip 1.0.

Stretch rating (IRC:82-2015 Table 5.3, urban): per 100 m chainage bin per way: pothole_area_pct = Σ A / (100 m × lanes × 3.5 m); > 0.5 % → Poor, 0 < x ≤ 0.5 % → Fair, 0 → Good. Rendered as a road-health layer.

---

## 7. Repair planning (IRC:82-2015 Sec 10 & Cl. 11.4, IRC:116-2014, IRC:SP:100-2014)

Inputs: fused A, L, W, depth_p95 (σ), depth_mean, water_filled, surface type (BT/CC from `road_segment.surface`), cluster neighbours.

1. **Merge**: potholes within 0.5 m on the same lane band → one cut.
2. **Square cut**: L_cut = L + 2m, W_cut = W + 2m, m = 0.125 m (configurable 0.10–0.15; IRC:82 Cl. 11.4 "cut adjoining deteriorated portion"); A_cut = L_cut · W_cut.
3. **Design depth** **[FIX: p95, not mean]**: d_design = ceil5mm( max(depth_p95, 0.050) ) (IRC:116 Cl. 5.2: ≥ 50 mm patching mix). If d_design > 0.075: layered — bottom = d_design − 0.050 of Crusher Run Macadam / angular 25 mm aggregate with MC-30 prime (layers ≤ 75 mm each), top = 0.050 patching mix. Else single layer d_patch = d_design.
4. **Volumes**: V_patch = A_cut · d_patch; V_crm = A_cut · max(0, d_design − 0.050); V_cavity = fused V (stored for audit); V_cuboid = L·W·depth_max (stored to show the 2–4× over-estimate of the cuboid method).
5. **Mass and bags**: M_mix = V_patch · ρ · (1 + w), ρ = 2,200 kg/m³ compacted (configurable; HINCOL rule 100 kg/m² at 50 mm ⇒ 2,000), w = 0.12 wastage; bags = ceil(M_mix / bag_kg), bag_kg = 25 (trade) or 50 (IRC:116 default). σ_bags from σ_M = ρ(1+w)·sqrt((A_cut·σ_d95)² + (d_patch·σ_A)²); report "bags (low–high)" = ceil((M ∓ σ_M)/bag_kg). **[FIX: no "exact bags" language anywhere.]**
6. **Ancillaries**: tack/prime emulsion SS-1/CSS-1 0.30 kg/m² × A_cut; sand blinding 1 kg/m² × A_cut; CRM tonnes = V_crm × 2.1.
7. **Labour**: 2.0 mazdoor-hours per pothole + 1.0 h per additional 0.025 m³ of V_patch; mate/supervision 10 %.
8. **Cost** (rate_card per agency per FY; defaults from IndiaMART/HINCOL evidence, all editable): cost = bags × ₹350 (25 kg) + emulsion_kg × ₹45 + CRM_t × ₹1,100 + sand_kg × ₹2 + labour_h × ₹100 + 10 % T&P/transport. Baseline comparison on the dashboard: GHMC ₹7,396/pothole (Deccan Chronicle 2021). Jet-patcher rate default ₹1,942/m² is labelled "BBMP Bengaluru 2025 reference — replace with GHMC circle tender rate". **[FIX: provenance labelled; no misattributed IRC clause numbers — rain rule cited as IRC:82 Sec 10/Appendix "pothole need not be dry; excess water shall be removed" and IRC:SP:100-2014 Sec 9 for cold maintenance mixes.]**
9. **Method selector** (`spec_code`): default `IRC116_COLDMIX` (storable cut-back MC-800 readymade mix, 4.5–5.5 % residual bitumen, 50 kg bags) or `IRCSP100_EMULSION` (SS-2/MS emulsion 7–8 % by wt of aggregate) per agency preference; `JET_PATCHER` if ≥ 8 confirmed potholes within 300 m or any A_cut > 1.0 m²; `INFRARED` if BT, d_design ≤ 40 mm, dry forecast and a unit exists in the zone; `HOT_MIX` only if plant ≤ 10 km and off-season (Chudi Bazar cold-arrival problem); `CC_MORTAR` (engineer item) for CC surface. Weather gate: patching allowed in damp holes after bailing; blocked during active rain (IMD nowcast or manual flag); surface-course cold mix never scheduled in rain.
10. **Worked example (unit test)**: pothole 0.50 × 0.40 m, depth_p95 60 mm (σ 8 mm): A_cut = 0.75 × 0.65 = 0.4875 m²; d_design 60 mm, single layer; V_patch = 0.02925 m³; M = 0.02925 × 2200 × 1.12 = 72.1 kg → 3 bags of 25 kg (range 3–4) or 2 of 50 kg; emulsion 0.146 kg; sand 0.49 kg; labour 2.2 h; cost ≈ 3×350 + 0.146×45 + 0.49×2 + 2.2×100 = ₹1,278 + 10 % ≈ **₹1,406** vs ₹7,396 baseline. V_cavity ≈ 5–7 L vs V_cuboid 12 L vs V_patch 29 L (all shown).

**Work order lifecycle**: `planned` → `work_order_issued` (contract, contractor, crew, sla_due, run_id) → `acknowledged` (contractor app) → `in_progress` (crew check-in geo+time, before photo, bag batch/lot no.) → `repaired_claimed` (after photo, actual bags; variance > 25 % vs estimate flagged) → `verification_pending` → `verified_closed` | `qc_failed` (rework WO) ; `reopened` on re-detection → `dlp_event` if inside DLP (patch warranty 6 months default; CRMP package 5 years) → rework WO at zero pay + debit-note draft, as an **evidence package** (before/after photos, re-detection frames, measurements, pass log) rather than an automatic penalty. Every transition writes `audit_log(entity, entity_id, action, actor_id, before, after, reason, geom, at)`.

**SLA & escalation**: `ops/sla.py` ticks every 5 min: at 50 % of SLA → notify AE; breach → DEE; breach + 24 h → EE; breach + 72 h → Zonal Commissioner (notification adapters: email/SMS/WhatsApp stubs logging to `notifications` table in demo).

**Daily repair run (indent)**: 06:00 IST per circle: top-N confirmed potholes by P, grouped by road owner and contractor, greedy nearest-neighbour route within 500 m hops, cap by crew-hours (default 8 h) → `repair_run` with totals (bags, CRM t, emulsion kg, sand kg, crew-h, ₹) and a printable indent (PDF via browser print CSS) and MB CSV.

**Contractor SLA/DLP data**: `contract(package, agency, contractor, geom, start, end, dlp_months, sla_rules jsonb)`; CRMP Phase-1 seven packages seeded from public descriptions (editable); unknown owner → `jurisdiction_queue` for an admin to resolve (addresses the GHMC/R&B/NHAI/SCB confusion).

---

## 8. Software stack (pinned)

**Python: use the existing `.venv` (Python 3.11.18) for everything — API, workers, ML.** Rationale: torch 2.11.0+cu128 is already installed and verified on the RTX 5080 (sm_120); the CUDA 13.2 driver (595.99) runs cu128 wheels; `rfdetr`, `onnxruntime-gpu`, `tensorrt` have no confirmed Python 3.14 wheels; one venv avoids an API/ML split. Do **not** reinstall torch. Python 3.14 is not used. **[FIX: all three designs' venv errors.]**

`pyproject.toml` (uv/pip), pins (checked on PyPI 2026-11-11/12 by research; if a pin fails to resolve, the agent records the resolved version in `LOCKFILE.md` rather than silently changing APIs):
```
torch==2.11.0+cu128 (existing, do not touch), torchvision (matching, existing)
ultralytics==8.4.211
rfdetr>=1.4,<2            # verifier; optional extra "verifier"
transformers==5.22.2      # SAM2, DA-V2 fallback
git+https://github.com/ByteDance-Seed/Depth-Anything-3   # DA3 (pin commit in LOCKFILE.md)
opencv-python-headless>=4.11,<5   # keep 4.x for ultralytics/albumentations compatibility
albumentations>=2.0,<3
numpy>=2.1,<3  scipy>=1.15  scikit-learn==1.9.1  shapely==2.2.0  pyproj==3.8.0  h3==4.5.0
open3d>=0.19 (optional extra "o3d"; numpy RANSAC is default)
fastapi==0.162.5  uvicorn[standard]==0.54.0  pydantic==2.14.3  python-multipart==0.0.33
sqlalchemy>=2.0.36,<2.1   geoalchemy2==0.21.0   psycopg[binary]==3.3.8   alembic==1.20.1
celery[redis]==5.7.0  redis>=6  boto3==1.44.1
pyjwt==2.15.1  argon2-cffi>=25  python-dotenv  httpx
huggingface_hub>=1.0  kaggle  roboflow  osmium (pyosmium)  pmtiles
pytest  pytest-asyncio  hypothesis  ruff
```
**Services (native, no Docker)**: PostgreSQL 16–18 + PostGIS 3.4–3.6, Redis 7/8, MinIO single binary. `tools/setup_dev.sh` tries `apt` (postgresql-postgis, redis-server) with sudo; **fallback without sudo**: `micromamba create -p .mamba -c conda-forge postgresql postgis redis-server minio` and user-space `initdb`/`pg_ctl` on port 5433. `DEMO_LITE=1` replaces MinIO with a local-filesystem storage adapter (`server/app/storage.py` has `S3Storage` and `FsStorage` behind one interface) and sets Celery `task_always_eager`. PostGIS is mandatory.

**Node 22 / web** (`web/package.json`): react 19.3.1, react-dom 19.3.1, vite 8.3.8, typescript 5.9, maplibre-gl 6.18.0 (ESM-only, WebGL2), pmtiles ^4, @tanstack/react-query 5.107.1, zustand 5.1.0, i18next 27.1.1, react-i18next 18.0.1, vite-plugin-pwa 2.0.0, workbox-window 7.4.1, dexie ^4, react-router ^7, recharts ^3 (charts), @fontsource/noto-sans-telugu, @fontsource/noto-sans-devanagari.

**Edge**: Android Kotlin (CameraX 1.5, LiteRT GPU delegate, Room, WorkManager) — skeleton only in hackathon; Jetson Python (GStreamer + TensorRT 10.x) — runner stub; Python simulator — fully working.

---

## 9. Repo layout

```
gunthameter/                                   (root = /home/coral/Desktop/Telangana pathole detection)
├── README.md                     quickstart (make setup, make data, make demo), architecture, IRC refs, licence table
├── LOCKFILE.md                   actually-resolved versions + DA3 commit hash (written by setup)
├── Makefile                      setup | services | data | dataset | train | export | db | api | workers | web | demo | test | truth
├── pyproject.toml                single project, extras: verifier, o3d, dev
├── .env.example                  DATABASE_URL, REDIS_URL, S3_* , DEMO_MODE, DEMO_LITE, HF_TOKEN, KAGGLE_*, ROBOFLOW_API_KEY, JWT_SECRET
├── configs/
│   ├── classes.yaml              7 classes, reporting subset
│   ├── model/india_pothole_seg.yaml   ultralytics dataset yaml (paths, names)
│   ├── irc_rules.yaml            IRC:82 7.5.3.4 bins; IRC:116/SP:100 layer rules; density; wastage; margin; bag sizes
│   ├── rate_card.ghmc_fy2026.yaml  INR unit rates with provenance strings
│   ├── priority.default.yaml     weights, SLA hours per road class, monsoon window, POI weights
│   ├── agencies.yaml             GHMC/Cyberabad MC/Malkajgiri MC/R&B/HMDA/NHAI/SCB; CRMP packages seed
│   ├── cameras/{phone_demo,car_default,bus_default,sim_bus}.json   calibration certs (certified=false for demo)
│   └── measure.yaml              depth backend, scale_convention, RANSAC params, gates, k_sigma defaults
├── data/
│   ├── download.py               all dataset/geo/checkpoint downloads into datasets/downloads/<name>/ (isolated dirs)
│   ├── build_dataset.py          VOC/YOLO/eYantra/Roboflow -> unified YOLO-seg; SAM2 pseudo-masks; negatives; splits
│   ├── splits/                   train/val/test file lists (committed)
│   ├── raw/  interim/  datasets/ tiles/   (gitignored)
│   └── truth/                    ground-truth CSV template + staged-target results
├── ml/
│   ├── segmenter.py              Segmenter interface: UltralyticsSegmenter | RFDETRSegmenter | ONNXSegmenter
│   ├── train_seg.py              YOLO26 seg training wrapper + India augmentations hook
│   ├── train_rfdetr_seg.py       verifier/teacher
│   ├── eval.py                   metrics -> reports/model_eval.json
│   ├── export.py                 litert/onnx/engine + sha256 + model_registry entry + ed25519 signature
│   ├── pseudo_label.py           teacher labels new footage; disagreement queue (Label Studio JSON)
│   └── augment/india.py
├── measure/                      (pure library, no DB imports)
│   ├── types.py                  Measurement, FrameMeasurement, CalibCert dataclasses (pydantic)
│   ├── calib.py                  intrinsics, PnP extrinsics, cert I/O, drift checks
│   ├── depth_backends/{base,da3_metric,da2_metric,metric3d_onnx,stereo_oakd,stereo_sgbm,synthetic}.py
│   ├── surface.py                RANSAC plane (numpy/open3d), height lock, road frame, local quadratic
│   ├── pothole_geom.py           mask integral: depth stats, area, volume, L/W, lane offset, σ, quality, water flag
│   ├── fuse.py                   track/pass fusion (inverse-variance, outlier rejection)
│   ├── imu.py                    jolt detection, pass-time prediction, corroboration, optional depth hint
│   ├── irc.py                    severity bins, Table 5.3 rating (single source of truth)
│   ├── truth.py                  truth ingest, MAE/MAPE per bin, fit k and g_v, certificate issue
│   ├── synthetic.py              render planar/crowned road + spherical-cap cavity with exact volume
│   ├── pipeline.py               measure_frame(img, mask, cert, cfg) -> FrameMeasurement (used by worker & CLI)
│   └── cli.py                    `python -m measure.cli run --image --mask|--detect --calib` -> JSON
├── repair/
│   ├── planner.py                square cut, layering, volumes, bags±σ, ancillaries, cost, method selector
│   ├── mb_export.py              Measurement Book CSV; indent CSV
│   └── tests/test_planner.py
├── geo/
│   ├── locate.py                 GPS + IPM offset -> ground point, σ_pos
│   ├── mapmatch_pg.py            PostGIS nearest-way + heading + Viterbi (default)
│   ├── mapmatch_osrm.py          optional OSRM client
│   ├── cluster.py                DBSCAN per way+dir, lane split, pothole upsert
│   ├── passes.py                 trip_fix -> way passes, clean-pass counting
│   ├── jurisdiction.py           road owner / contract / admin unit spatial joins; jurisdiction queue
│   ├── gtfs.py                   GTFS loader -> routes, stops, trips/day per way
│   └── osm.py                    PBF -> road_segment, pois (schools, hospitals, markets, metro), lit tag
├── priority/
│   ├── score.py                  P formula + breakdown (pure)
│   ├── exposure.py  vulnerability.py  weather.py (IMD adapter + manual flag)
├── server/
│   ├── app/main.py               FastAPI factory, routers, gzip, CORS, /healthz
│   ├── app/core/{config,security,rbac,audit}.py
│   ├── app/db/{session,models,partitions}.py   SQLAlchemy+GeoAlchemy2 models
│   ├── app/schemas/*.py          pydantic DTOs (shared contracts)
│   ├── app/storage.py            S3Storage | FsStorage
│   ├── app/routers/{auth,ingest,citizen,potholes,tiles,measure,plans,work_orders,runs,contractors,kpi,exports,admin,public,edge}.py
│   ├── app/services/{state_machine,sla,escalation,verification,dlp,runs,notify}.py
│   ├── workers/{celery_app,tasks_gpu,tasks_cpu,tasks_ops}.py
│   ├── alembic/                  migrations (postgis ext, tables, monthly partitions, indexes)
│   └── tests/                    API, state machine, dedup, verification, RBAC
├── edge/
│   ├── schema/observation.schema.json, trip_fix.schema.json   (shared JSON Schema; validated both sides)
│   ├── sim/replay.py             virtual bus: dataset frames/video + GTFS shape -> detections -> payloads -> upload
│   ├── sim/fleet.py              N virtual buses, repeated passes, repair-day simulation (potholes vanish)
│   ├── android/                  Kotlin skeleton (CameraX, LiteRT, distance sampler, Room queue, WorkManager, OTA)
│   └── jetson/                   GStreamer+TensorRT runner stub, OAK-D capture stub, systemd unit
├── gis/
│   ├── build_tiles.sh            pmtiles extract for Hyderabad bbox
│   └── layers/                   ghmc_zones.geojson, wards (unofficial, validity dates), crmp_packages.geojson, waterlogging_hotspots.geojson, road_owner_overrides.geojson
├── web/
│   ├── package.json  vite.config.ts  index.html  tsconfig.json
│   ├── src/app/{routes,layout}  src/lib/{api,auth,offlineQueue,sync,i18n}
│   ├── src/features/{map,pothole,priority,workorders,runs,field,contractors,citizen,admin,public,kpi}
│   ├── src/i18n/{te,hi,en}.json
│   └── public/                   fonts, icons, manifest
├── scripts/
│   ├── setup_dev.sh              venv deps, services (apt or micromamba), buckets, .env
│   ├── seed_demo.py              agencies, admin units, contracts, OSM roads, GTFS, POIs, rate card, priority config, users
│   ├── run_demo.sh               services -> migrate -> seed -> api -> workers -> web -> fleet simulator
│   ├── staged_pothole_truth.py   staged-target validation report
│   └── make_report.py            evaluation/validation tables -> docs/REPORT.md
├── docs/
│   ├── ARCHITECTURE.md  DEPTH_METHOD.md  CALIBRATION_SOP.md (te/en)  IRC_MAPPING.md  API.md
│   ├── DEPLOYMENT_TSRTC.md (pilot BOM, power, thermal, SIM, depot Wi-Fi)  FIELD_SOP_te.md  LICENCES.md  DEMO_SCRIPT.md
└── tests/
    ├── test_volume_synthetic.py  test_surface.py  test_fuse.py  test_imu.py  test_irc.py  test_priority.py
    └── e2e/test_demo_pipeline.py
```

---

## 10. Build order — work packages (parallelisable by independent agents)

Shared contracts are frozen in **WP0** and must not be changed by other WPs without updating WP0 files. Dependencies: WP0 → everything; WP1 (data) → WP3 (training) and WP7 (sim); WP2 (measure) → WP6 (GPU worker); WP4 (DB/API) → WP5/6/7/8/9; WP9 (web) needs only WP4's OpenAPI spec (mock with seed data).

### WP0 — Scaffold & contracts (first, single agent, ~2 h)
Dir: root, `configs/`, `edge/schema/`, `server/app/schemas/`, `measure/types.py`, `tools/setup_dev.sh`, `Makefile`, `.env.example`, `LOCKFILE.md`.
Contracts:
- `edge/schema/observation.schema.json`:
```json
{"client_uuid":"uuid","device_id":"str","trip_id":"uuid","seq":1,"captured_at":"ISO8601+05:30","mono_ms":123456,
 "source":"edge|van|jolt|citizen|sim","calib_id":"str","model_version":"str",
 "gps":{"lat":17.43,"lon":78.46,"accuracy_m":6.5,"speed_mps":9.2,"heading_deg":212.0,"fix_age_ms":120},
 "imu":{"az_rms_1s":0.08,"az_peak_g":0.42,"window_key":"imu/.../seq.npz"},
 "frame":{"key":"frames/<device>/<trip>/<seq>.jpg","w":1280,"h":720,"jpeg_q":75},
 "detections":[{"track_id":7,"cls":"pothole","conf":0.61,"bbox":[x1,y1,x2,y2],"polygon":[[u,v],"..."]}]}
```
- `edge/schema/trip_fix.schema.json`: `{device_id, trip_id, fixes:[{ts, lat, lon, accuracy_m, speed_mps, heading_deg, az_rms, cam_active}]}` (batch ≤ 600).
- `measure/types.py`: `CalibCert`, `FrameMeasurement(depth_p95_mm, depth_max_mm, depth_mean_mm, depth_sigma_mm, area_m2, area_sigma, volume_l, volume_sigma, length_m, width_m, range_m, lane_offset_m, ground_xz:[x,z], surface_model, plane_rms_mm, ring_rms_mm, scale_factor, water_filled, quality, reject_reason, method, depth_model_version)`, `PassMeasurement` (same + n_frames), `Severity` enum.
- DB tables (Alembic in WP4; names fixed here): `agency, admin_unit, user, device, calib_cert, trip, trip_fix(part. monthly), observation(part. monthly), detection, measurement, pass_measurement, pothole, pothole_observation, pothole_event, road_segment, way_pass, poi, contract, contractor, repair_plan, work_order, repair_run, repair_event, verification, dlp_event, complaint, jurisdiction_queue, priority_config, rate_card, model_registry, notification, audit_log`.
- API (`/v1`, JWT bearer; roles citizen, driver_device, field_engineer, contractor, zonal_admin, super_admin): `POST /auth/otp`, `POST /auth/token`; `POST /edge/register`, `GET /edge/config` (ETag, signed), `POST /edge/presign`, `POST /ingest/observations` (JSON batch, idempotent by client_uuid), `POST /ingest/trip-fixes`, `POST /ingest/trip-close`; `POST /citizen/reports` (multipart), `GET /citizen/reports/{ticket}`; `GET /potholes?bbox&status&severity&admin_unit&agency&min_priority&limit`, `GET /potholes/{id}`, `POST /potholes/{id}/reject`, `GET /tiles/potholes/{z}/{x}/{y}.mvt`, `GET /segments/rating?bbox`; `POST /measure` (image + calib → FrameMeasurement; demo), `POST /truth/measurements`, `GET /devices/{id}/cert`; `POST /potholes/{id}/plan`, `GET /plans/{id}`; `POST /work-orders`, `PATCH /work-orders/{id}/transition`, `POST /work-orders/{id}/photos`, `GET /work-orders?...`; `POST /runs/generate?circle&date`, `GET /runs/{id}`, `GET /runs/{id}/indent.csv`; `GET /contractors/{id}/scorecard`; `GET /kpi/{admin_unit}?window`; `GET /exports/measurement-book.csv`, `GET /exports/potholes.geojson`; `GET/PUT /admin/priority-config`, `GET/PUT /admin/rate-card`, `POST /admin/admin-units` (GeoJSON), `GET /admin/jurisdiction-queue`, `POST /admin/jurisdiction-queue/{id}/resolve`, `GET /admin/audit?entity&id`; `GET /public/wards`.
Acceptance: `make setup` creates/validates venv deps without touching torch, prints GPU check; `pytest tests/test_schema.py` validates example payloads against both JSON schemas; `LOCKFILE.md` written.

### WP1 — Data acquisition & dataset build (agent A; parallel with WP2/WP4)
Dir: `data/`, `configs/model/`, `gis/layers/`. Files: `download.py`, `build_dataset.py`, `splits/`.
Interfaces: `python -I dataset_tools/download.py --all|--rdd|--eyantra|--bharat|--roboflow|--gtfs|--osm|--boundaries|--overpass|--weights`; `python dataset_tools/build_dataset.py --out datasets/archive/india_pothole_v1_allclasses --sam2 --negatives-frac 0.2` → YOLO-seg layout + `india_pothole_seg.yaml` + `stats.json` (counts per class/split).
Acceptance: dataset yaml validates with `yolo val` dry-run (`ultralytics.data.utils.check_det_dataset`); stats.json shows ≥ 3,000 pothole instances and ≥ 500 val mask images; no file executed from `datasets/downloads`.

### WP2 — Measurement core (agent B; the differentiator; start immediately)
Dir: `measure/`, `tests/`. Files: all `measure/*`, `tests/test_volume_synthetic.py`, `test_surface.py`, `test_fuse.py`, `test_imu.py`, `test_irc.py`.
Interfaces:
- `DepthBackend.predict(img_bgr, fx_px) -> (depth_m, conf|None)`; `SyntheticBackend` returns rendered Z.
- `surface.fit_road_surface(points_cam[N,3], road_mask, cfg) -> SurfaceFit(n, d, h_hat, s, R_road, quad_coeffs|None, plane_rms, ring_rms, inlier_ratio)`.
- `pothole_geom.measure_mask(depth_m, mask, K, surface, cfg) -> FrameMeasurement`.
- `pipeline.measure_frame(img, mask, cert, cfg) -> FrameMeasurement`; `fuse.fuse_frames(list[FrameMeasurement]) -> PassMeasurement`; `imu.corroborate(window, t_pass, lane_offset, cert) -> ImuResult`; `irc.severity(depth_p95_mm, width_mm, water_filled) -> Severity`.
- CLI: `python -m measure.cli run --image X.jpg --calib configs/cameras/phone_demo.json [--mask m.png | --detect weights.pt] --json out.json`.
Acceptance: synthetic bowl tests pass (volume ±3 %, p95 ±2 mm, crowned road ±5 %, σ-coverage ≥ 90 %); `measure.cli run` on a RDD image with `phone_demo.json` returns JSON with all fields and `certified=false` flag; DA3 forward pass on the 5080 < 1.5 s/frame; `measure/truth.py` fits k on synthetic noisy runs.

### WP3 — Training, verifier, export (agent C; after WP1; GPU-heavy, run in background)
Dir: `ml/`. Files: `segmenter.py`, `train_seg.py`, `train_rfdetr_seg.py`, `eval.py`, `export.py`, `pseudo_label.py`, `augment/india.py`.
Interfaces: `Segmenter.predict(img) -> list[Detection(cls, conf, bbox, polygon, mask)]`; `Segmenter.track(frames) -> ...track_id`; `ml/export.py --weights best.pt --formats onnx,litert16,litert8,engine --register` writes `model_registry` JSON (name, version, task, format, sha256, metrics, signature).
Acceptance: `reports/model_eval.json` with box mAP50 ≥ 0.70 on RDD-India D40 test (hard floor; target 0.75), mask IoU ≥ 0.65 on eYantra test; exported ONNX runs through `ONNXSegmenter` with ≤ 2 % mAP delta; bootstrap weights path works when training is not finished (`SEG_WEIGHTS` env).

### WP4 — Database, storage, auth, core API (agent D; parallel with WP1/2)
Dir: `server/app/`, `server/alembic/`, `tools/seed_demo.py`, `tools/setup_dev.sh` services part.
Tables (key columns):
- `device(id, agency_id, vehicle_reg, kind phone|jetson|oakd|zed|sim, calib_cert_id, last_seen)`; `calib_cert(id, device_id, json, certified bool, issued_at)`.
- `trip(id, device_id, started_at, ended_at, gtfs_route_id, raw_geom linestring, matched_geom, km)`; `trip_fix(trip_id, ts, geog point, accuracy_m, speed_mps, heading, az_rms, cam_active)` partitioned monthly, BRIN(ts), GIST(geog).
- `observation(id, client_uuid unique, device_id, trip_id, seq, captured_at, source, geog_raw, geog_ground, way_id, direction, chainage_m, lane_offset_m, frame_key, imu_key, calib_cert_id, model_version, status pending|verified|rejected)` partitioned monthly.
- `detection(id, observation_id, track_id, cls, conf, bbox int[4], polygon jsonb, verifier_cls, verifier_conf, mask_key)`.
- `measurement(id, detection_id, method, depth_p95_mm, depth_max_mm, depth_mean_mm, depth_sigma_mm, area_m2, area_sigma, volume_l, volume_sigma, length_m, width_m, range_m, lane_offset_m, surface_model, plane_rms_mm, ring_rms_mm, scale_factor, water_filled, imu_corroborated, imu_depth_mm, quality, reject_reason, depth_model_version, depth_map_key)`.
- `pass_measurement(id, trip_id, pothole_id, n_frames, fused fields..., sigma fields..., quality)`.
- `pothole(id, agency_id, admin_unit_id, road_segment_id, way_id, direction, lane_band, chainage_m, geog, h3_r12, status, severity small|medium|large|unknown_water, depth_p95_mm, depth_sigma_mm, area_m2, volume_l, volume_sigma, length_m, width_m, water_filled_ratio, first_seen, last_seen, passes_seen, passes_total, trips_seen, complaint_count, imu_corroborations, priority, priority_breakdown jsonb, contract_id, contractor_id, best_method, certified_measurement bool)`; `pothole_observation(pothole_id, observation_id, detection_id)`; `pothole_event(pothole_id, ts, from_status, to_status, reason, actor_id)`.
- `road_segment(id, osm_way_id, name, highway, lanes, lit, surface, geog linestring, owner_agency_id, contract_id, admin_unit_id, gtfs_trips_per_day)`; `way_pass(way_id, direction, day, n_passes, n_cam_active)`; `poi(id, kind, name, geog, source)`.
- `agency, admin_unit(kind zone|circle|ward|division, name_en/te/hi, geom, parent_id, valid_from, valid_to)`, `contractor`, `contract(..., dlp_months, sla_rules jsonb)`, `priority_config(agency_id, version, weights jsonb, active)`, `rate_card(agency_id, fy, items jsonb)`.
- `repair_plan, work_order, repair_run, repair_event, verification, dlp_event, complaint, jurisdiction_queue, notification, model_registry, user, audit_log`.
Interfaces: SQLAlchemy models + pydantic DTOs; `FsStorage`/`S3Storage.put(key, bytes) / presign(key) / get(key)`; `require_role(*roles)` + `scope_query(user, stmt)`.
Acceptance: `alembic upgrade head` on fresh PostGIS; `tools/seed_demo.py` loads OSM Hyderabad roads (≥ 50k segments), GTFS stops/routes, POIs, agencies, admin units, 7 CRMP contracts, rate card, priority config, demo users; `pytest server/tests/test_api_smoke.py` (register device → post observations → list potholes) passes; RBAC test: contractor cannot read other contractor's WOs.

### WP5 — Geo services (agent E; after WP4 schema)
Dir: `geo/`, `priority/`. Files listed in §9.
Interfaces: `locate.ground_point(obs, fm) -> (lat, lon, sigma_m)`; `mapmatch_pg.match_trip(trip_id)` and `match_point(lat, lon, heading) -> MatchResult(way_id, direction, chainage_m, dist_m)`; `cluster.upsert_from_observations(way_id, direction, window_days=30) -> list[pothole_id]`; `passes.count_passes(pothole, since) -> (total, clean)`; `jurisdiction.resolve(pothole) -> (agency_id, contract_id, admin_unit_id) | queue`; `priority.score.compute(pothole, ctx) -> (P, breakdown)`.
Acceptance: synthetic test — 3 trips over the same pothole with GPS noise σ 8 m on a 2-lane way → exactly 1 `confirmed` pothole; 2 potholes on opposite directions stay separate; 2 potholes 2 m apart in different lanes stay separate; 3 clean passes → `verified_closed`; priority unit tests for each term; T term breaches at 24 h for arterial.

### WP6 — Workers (agent F; after WP2, WP4)
Dir: `server/workers/`. Files: `celery_app.py`, `tasks_gpu.py` (`verify_segment(observation_ids)`, `depth_measure(detection_ids)` batch 16, writes measurement rows + depth PNG16 to storage), `tasks_cpu.py` (`locate_and_match`, `fuse_pass`, `cluster_way`, `score_pothole`, `plan_pothole`), `tasks_ops.py` (`sla_tick`, `verify_repairs`, `dlp_watch`, `daily_indent`, `kpi_rollup`).
Interfaces: task chain triggered by `POST /ingest/observations` → `verify_segment.s() | depth_measure.s() | locate_and_match.s() | fuse_pass.s() | cluster_way.s() | score_pothole.s() | plan_pothole.s()`; `trip-close` → `fuse_pass` + `verify_repairs`.
Acceptance: with `DEMO_MODE=1 DEMO_LITE=1` (eager), posting 20 simulator observations results in measurement + pothole rows within 60 s on the 5080; GPU memory < 12 GB with batch 16; failure in depth (reject) still leaves observation `verified` with `measurement.reject_reason`.

### WP7 — Edge simulator & edge skeletons (agent G; after WP0; needs WP1 frames)
Dir: `edge/`. Files: `sim/replay.py`, `sim/fleet.py`, `android/` skeleton, `jetson/` stub, `protocol.md`.
Interfaces: `python edge/sim/replay.py --frames datasets/.../val --route 10H --device sim-bus-01 --api http://localhost:8000 --gps-sigma 7 --passes 3 --weights <seg weights|onnx> --fps-equiv 30 --speed-kmh 35` — distance-triggered keyframes, runs the real Segmenter (or uses dataset labels with `--use-labels` when no GPU), uploads frames via presign, posts observations + 1 Hz trip fixes with `cam_active`, closes trip; `fleet.py --buses 3 --routes 10H,218,5K --days 2 --repair-day 2` repeats passes and, on repair day, suppresses detections at "repaired" potholes so verification triggers; also emits 2 jolt observations and 1 citizen report.
Acceptance: full `fleet.py` run produces ≥ 30 confirmed potholes, ≥ 1 `verified_closed`, ≥ 1 `reopened`/`dlp_event`, ≥ 1 jurisdiction-queue item, in < 15 min on the dev box; Android project compiles (`./gradlew assembleDebug`) if the SDK is present, otherwise documented.

### WP8 — Ops services, repair planner, exports (agent H; after WP4)
Dir: `repair/`, `server/app/services/`, routers `plans, work_orders, runs, contractors, kpi, exports, admin, public`.
Interfaces: `repair.planner.plan(PothFused, rules, rate_card, context) -> RepairPlan(a_cut_m2, d_design_mm, layered, v_patch_l, v_crm_l, v_cavity_l, v_cuboid_l, mix_kg, mix_kg_sigma, bags, bags_low, bags_high, bag_kg, emulsion_kg, sand_kg, crm_t, labour_h, method, spec_code, est_cost_inr, cost_breakdown, rate_card_version)`; `state_machine.transition(entity, to, actor, reason, geom)` (validates allowed edges, writes audit_log); `runs.generate(circle_id, date) -> RepairRun`; `verification.evaluate(work_order)`; `dlp.watch(pothole)`; `mb_export.measurement_book(work_order_ids) -> CSV`.
Acceptance: `repair/tests/test_planner.py` worked example (0.5×0.4 m, p95 60 mm → 3 bags of 25 kg, range 3–4, cost ≈ ₹1,400 ± rate card); layered case 90 mm → CRM 40 mm + 50 mm mix; water-filled → plan flagged `pending_remeasure` with lower-bound bags; state machine rejects illegal transitions and logs every legal one; indent CSV totals equal sum of plans; invoice-eligible endpoint returns only `verified_closed`.

### WP9 — Web PWA (agent I; after WP0 API contract; mock server from seed)
Dir: `web/`. Routes: `/` zonal dashboard (KPIs, map, ward league, monsoon toggle), `/map` (MVT potholes coloured by severity/priority, H3 heat at z < 13, stretch rating layer, filters), `/potholes/:id` (frames with mask overlay, depth cross-section chart from depth_map, measurement ± σ, method/certified badge, priority breakdown, plan with bags low–high, timeline), `/queue` (priority list, batch to WO), `/work-orders`, `/runs/:id` (indent print), `/field` (offline list, before/after capture resized to 1280 px, bags used, Dexie queue + background sync), `/contractors/:id` (scorecard, survival curve), `/citizen` (report + track), `/admin` (weights, rate card, SLA, admin-unit upload, jurisdiction queue, audit viewer, model registry, calibration status), `/public/wards`.
Interfaces: typed API client generated from OpenAPI (`openapi-typescript`); i18n keys in `src/i18n/{te,hi,en}.json` (Telugu first); PMTiles from `/tiles/hyd.pmtiles` with online fallback style.
Acceptance: `npm run build` passes; Lighthouse PWA installable; map renders seeded potholes; language switch te/hi/en; field mode queues a photo offline and syncs when online (Playwright test with network off/on); list views use thumbnails ≤ 20 KB.

### WP10 — GIS layers & tiles (agent J; parallel)
Dir: `gis/`. `build_tiles.sh` (pmtiles extract Hyderabad), `layers/*.geojson` (GHMC zones/circles from boundaries repo, unofficial wards with `valid_from`, CRMP package polygons approximated from public descriptions and marked editable, waterlogging hotspots sample, road_owner overrides for NH/R&B/SCB via OSM `ref`/`operator`), loader in `seed_demo.py`.
Acceptance: all layers load into `admin_unit`/`contract`; every seeded road segment has owner or is in `jurisdiction_queue`; tiles file < 400 MB.

### WP11 — Validation artefacts & docs (agent K; after WP2/WP3)
Files: `tools/staged_pothole_truth.py`, `test_datasets/staged_measurement_truth/template.csv`, `tools/make_report.py`, `docs/*`.
Acceptance: `docs/DEPTH_METHOD.md` contains the formulas of §4 and the synthetic + staged error tables; `docs/CALIBRATION_SOP.md` (te/en) and `docs/DEPLOYMENT_TSRTC.md` (20-bus pilot across 3 depots, BOM, DC-DC, thermal, SIM, depot Wi-Fi) exist; `docs/LICENCES.md` lists every dataset/model licence and the AGPL/GPL caveats; `docs/DEMO_SCRIPT.md` is a 7-minute jury walkthrough.

### WP12 — End-to-end demo & hardening (integrator; last)
Files: `tools/run_demo.sh`, `tests/e2e/test_demo_pipeline.py`, `Makefile demo`.
`make demo` = services up (apt or micromamba) → migrate → seed → API (8000) → workers (or eager) → web (5173) → `edge/sim/fleet.py --buses 3 --routes 10H,218,5K --days 2 --repair-day 2` → prints URLs and a summary (potholes by status, bags in today's indent, ₹ vs baseline). Also `make demo-lite` (no Redis/MinIO; FS storage + eager Celery; PostGIS still required). `make measure-demo` = `measure.cli` on 5 RDD images + synthetic bowl report (works with no services at all — the minimum viable jury demo).
Acceptance: `pytest tests/e2e` passes on the dev box; demo completes < 20 min from clean DB; README quickstart verified by a fresh agent.

**Synthetic / demo mode summary**: (1) `SyntheticBackend` + `measure/synthetic.py` give exact-ground-truth cavities for unit tests without any model; (2) `measure.cli` runs real DA3 depth on real Indian images with a demo calibration (uncertified); (3) `edge/sim` replays real dataset frames along real TGSRTC GTFS shapes over real OSM Hyderabad roads with GPS noise, repeated passes, a repair day, jolts and a citizen report so every state (candidate → confirmed → planned → WO → run → repaired → verified_closed → reopened/DLP, jurisdiction queue) is exercised; (4) `DEMO_LITE` removes Redis/MinIO dependencies; (5) staged foam-board target provides a real-camera accuracy table.

---

## 11. Risks & mitigations; demo vs production

| Risk | Mitigation | Demo vs production |
|---|---|---|
| Monocular absolute depth accuracy on bus cameras unproven (no published GT study) | height scale-lock, local quadratic surface, propagated σ, quality gates, fusion, synthetic + staged validation, stereo Tier B for billing, certificate gating MB | Demo: synthetic + staged phone target; uncertified calibrations flagged. Production: truth campaign (50–100 potholes/zone), fitted k per device, certificates |
| DA3 scale convention disputed | configurable convention, validated by minimising ĥ error; height lock makes it non-critical | — |
| Water-filled potholes in monsoon | explicit flag, never report water as floor, lower-bound severity by width/IMU/complaints, remeasure scheduling, polarised stereo on vans | Demo: simulated via `pothole_water` labels |
| False positives (manholes, speed breakers, shadows, patches) | negative classes, RF-DETR verifier, ≥ 2-pass confirmation, IMU corroboration, engineer reject → retraining | Demo: verifier optional (`VERIFIER=off` if rfdetr fails to install) |
| Domain shift (RDD = Delhi/Haryana; no Telangana labels) | IDD negatives, India augmentations, pseudo-labelling of pilot footage, active-learning queue, label 1–2k Hyderabad frames in pilot week 1 | Production item |
| GPS 5–15 m in Ameerpet/Koti canyons; parallel service roads | ground point via IPM offset, heading-aware matching, lane bands, manual split/merge tool in admin | Residual mis-assignment accepted; admin tool |
| Hardware heat (phones throttle > 45 °C; Jetson dev kit 0–35 °C) | thermal callbacks, shaded/fan mounts, industrial carriers, uptime telemetry | Demo: simulator only. Production: 20-bus pilot decides phone vs Pi/Jetson |
| Licensing (YOLO26 AGPL, eYantra GPL-2, UniDepth/Depth Pro NC) | Segmenter interface with Apache RF-DETR-Seg path; eYantra prototype-only; LICENCES.md; Enterprise licence line item | Production: resolve before procurement |
| Toolchain (rfdetr/onnxruntime-gpu/tensorrt wheels, DA3 API changes) | single 3.11 venv already working; backends behind interfaces; ONNX Runtime CUDA fallback; commit-pinned DA3 | — |
| Services without sudo | micromamba fallback; DEMO_LITE | — |
| Scope vs time | WP2 + `make measure-demo` first (credible partial deliverable); Android/Jetson are skeletons; OSRM/Valhalla optional | Demo: simulator. Production: Android app, Jetson runner, Valhalla, PMTiles for all Telangana |
| Governance (ward boundaries litigated, 4 road owners, CRMP DLP never invoked) | pluggable admin_unit with validity dates, jurisdiction queue, DLP as evidence package not auto-penalty, configurable payment gating, audit trail | — |
| Over-claiming (volumes, IMU) | bags ± range, σ everywhere, IMU existence-only until gain fitted, honest expected-error table | — |
| Privacy (faces/plates in frames) | server-side blur before any public/citizen exposure; frames retained 90 days; phone hashes | Production policy |

**Demo (hackathon) scope**: full server pipeline, measurement core with validation artefacts, simulator fleet, PWA (all views), trained YOLO26 seg, optional verifier, PostGIS map-matching, demo calibrations (uncertified).
**Production scope (pilot plan in docs)**: 20–50 TSRTC buses from 2–3 depots with phones + DC-DC, 5 Tier B vans, depot calibration SOP, truth campaign and certificates, Hyderabad labelling + retraining, Valhalla, NIC/MeghRaj hosting, WhatsApp/MyCURE integration, Ultralytics Enterprise or RF-DETR-only edge, contractor MB/e-payment linkage with one CRMP package.
