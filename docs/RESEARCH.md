

## RESEARCH: stack-ops
# Stack Research Report — Telangana Pothole Platform (as of 2026-11-12)

Versions below were checked live on PyPI/npm today unless marked (web) or (uncertain). Dev box: Python 3.14, Node 22, CUDA 13.2 — all core packages declare a 3.14 classifier except `geoalchemy2`, `minio`, `tensorrt`, `depth-anything-v2` (see notes).

## 1. Backend

| Role | Pick | Version | Notes |
|---|---|---|---|
| API | `fastapi` | 0.162.5 | Pin >=0.128.1: PEP 649 deferred annotations on 3.14 broke TYPE_CHECKING forward refs in 0.128.0. `uvicorn` 0.54.0, `pydantic` 2.14.3, `python-multipart` 0.0.33. |
| ORM | `sqlalchemy` | 2.1.8 | 2.1 is a new line; `geoalchemy2` 0.21.0 has no 3.14 classifier and 2.1 support is **uncertain** — if import fails, pin `sqlalchemy<2.1` (2.0.x) which is safe. Alternative: skip GeoAlchemy and write PostGIS SQL via `psycopg` 3.3.8 / `asyncpg` 0.32.0 (both 3.14-ready). |
| Migrations | `alembic` | 1.20.1 | |
| DB | PostgreSQL 18 (rel. 2025-09-25, async I/O) + PostGIS 3.6.x | 3.6.0 rel. 2025-09-02; 3.6.4 seen on Aiven. Breaking change: topology columns need manual migration (irrelevant for new DB). |
| TimescaleDB | **Not needed** | — | Detection volume (~tens of thousands/day city-wide) fits native PG declarative partitioning by month on `detections(captured_at)` + BRIN index. Timescale adds ops burden and no GPU-box docker. Revisit only for per-second IMU telemetry. |
| Object store | MinIO (single binary, no docker) | `minio` 7.2.20 (no 3.14 classifier, pure-Python, should work; **uncertain**) or `boto3` 1.44.1 (S3 API, 3.14 OK). Prefer boto3 to keep cloud-portable (govt may move to NIC/MeghRaj S3). Store original JPEG + 320px WebP thumbnail + depth map PNG16. |
| Queue | `arq` 0.28.0 (Redis, asyncio, tiny) **or** `celery` 5.7.0 | Web notes call arq "maintenance-only" (unverified). Pragmatic: **Celery 5.7 + Redis 8.1** with a dedicated `gpu` queue, `worker_prefetch_multiplier=1`, `--pool=solo` per GPU process, batch frames in groups of 16 for YOLO/TensorRT. `rq` 3.0.0 / `dramatiq` 2.2.1 are viable simpler fallbacks. |
| Auth | `pyjwt` 2.15.1 + argon2 (`argon2-cffi`) | Roles: `citizen`, `field_engineer`, `contractor`, `zonal_admin`, `super_admin`. RBAC via FastAPI dependency `require_role(...)`; row scope by `zone_id`/`circle_id` (GHMC 6 zones/30 circles). Citizens: OTP (phone) only, no password. Contractors scoped to their CRMP package. |
| ML runtime | `torch` 2.15.0 (CUDA 13 wheels), `ultralytics` 8.4.211, `onnxruntime-gpu` 1.31.1, `tensorrt` 11.4.0 (no 3.14 classifier — **uncertain**, use ONNX Runtime CUDA EP as fallback), `opencv-python-headless` 5.0.0.93, `transformers` 5.22.2 (Depth-Anything-V2 metric checkpoints `depth-anything/Depth-Anything-V2-Metric-Outdoor-*-hf`). |
| Geo libs | `shapely` 2.2.0, `pyproj` 3.8.0, `h3` 4.5.0, `scikit-learn` 1.9.1 (DBSCAN). |

Detection model: YOLO26 (Ultralytics, NMS-free, end-to-end; docs say Jan 2026, other sources say late 2025 — date **uncertain**). Nano variant ~1.7 ms on T4 TensorRT FP16. Segmentation head (`yolo26n-seg`) is preferred over box-only because depth/volume needs a pixel mask.

Depth/volume (the user's hard requirement): no public paper gives validated pothole *volume* from monocular depth (2025 work: arXiv 2505.21049 does area only, with Depth Anything V2 + Kalman; a 2026 Kumoh paper notes DA-V3 gives relative not metric depth). Recommended design, two tiers:
- Tier A (buses, cheap): monocular metric DA-V2-Metric-Outdoor; calibrate scale per camera install using known lane-width/zebra stripes (IRC:35 marking widths, 100 mm stripe) and mount height; fit a RANSAC road plane to the mask's ring region, depth = plane − pixel depth, volume = Σ depth × pixel ground area. Expect ±30–40 % error; good enough for severity class (IRC:82 bins: <25 mm, 25–50 mm, >50 mm) and bulk material estimate.
- Tier B (municipal inspection vehicles): stereo (OAK-D Pro / ZED 2i / two Raspberry Pi GS cameras) for ±10 % metric depth, used for contractor billing. Same plane-fit algorithm. Mark: no head-to-head DA-V2 vs stereo error figures found on roads (**uncertain**).

## 2. Spatial dedup / clustering
- GPS noise 5–15 m in Hyderabad canyons (Ameerpet, Koti). Pipeline: (1) map-match the trip trace with **Valhalla Meili** (`/trace_attributes`, lower RAM than OSRM, tunable `search_radius`, `gps_accuracy`; returns OSM `way_id` + distance-along-edge). Build graph from Geofabrik `southern-zone` extract clipped to Telangana. OSRM is fine for prototype (`/match` API). No 2025 benchmark comparing them exists; pick Valhalla for production, OSRM for hackathon speed.
- Dedup key = (`way_id`, direction bucket of heading ±45°, `H3` res-12 cell ≈ 9 m edge, lateral lane offset from camera x-position). Then **DBSCAN** (`eps`=8 m haversine via `metric='haversine'`, `min_samples`=2) within the same way_id+direction over a rolling 30-day window. Cluster → `pothole` entity; detections → `observations`. Confidence grows with distinct vehicle passes (≥2 passes from ≥2 trips before dispatch). Store cluster centroid as PostGIS `geography(Point,4326)`, plus `ST_LineLocatePoint` chainage on the road segment for R&B km-stone reporting.
- Repair verification uses the same key: N consecutive passes with no detection in cell → status `verified_closed`.

## 3. Prioritisation
No Indian code gives a pothole-level priority formula. Relevant references: **IRC:82-2015** (Code of Practice for Maintenance of Bituminous Surfaces; defines pothole severity by depth/area and fixed-weight distress ranking used for stretch prioritisation), **IRC:116-2014** (readymade cold patching mix, potholes ≤75 mm deep, surface need not be dry), **IRC:SP:100-2014** (emulsion cold mix incl. pothole repair), MoRTH 1992 circular pointing to IRC:82 for wet-season patching. IRC:SP:72 is rural pavement design — not relevant. PMGSY **eMARG** (MP RDA) is the national model: geo-tagged inspection photos, performance-based maintenance contracts, contractor payment released only after inspection — copy its "inspection→payment" linkage. GHMC already runs an ArcGIS Field Maps "Periodic Public Safety Inspection" app (web, 2025) — position this platform as feeding/replacing it via GeoJSON export.

Suggested score (0–100), configurable in DB per zone:
`P = 100 · (0.30·S + 0.25·E + 0.20·V + 0.15·T + 0.10·M)`
- S severity: depth-dominated — depth ≥50 mm or area ≥0.5 m² = 1.0; 25–50 mm = 0.6; <25 mm = 0.3; add +0.2 if within wheel-path (lateral offset <1.2 m from lane centre) or at a junction.
- E exposure: road class (arterial/sub-arterial/collector from OSM `highway` tag) × TSRTC bus-route overlap × observed pass-count per day (buses give this for free).
- V vulnerability: POIs within 200 m — schools, hospitals, bus stops (TSRTC GTFS), metro stations, markets; night-lighting absent flag.
- T age: `min(1, days_open/14)`, with SLA breach bump.
- M monsoon: IMD district nowcast/rain last 72 h (open IMD API) × 1.0–1.5; waterlogged-pothole flag from citizen report.
Comparable literature uses PCI weight ~37 % in FAHP; TOPSIS beats fixed IRC weights (ρ=0.78) — keep weights editable, not hard-coded.

## 4. Dashboard
- `react` 19.3.1, `vite` 8.3.8, `maplibre-gl` 6.18.0 (**v6 is ESM-only, WebGL2 mandatory**, no UMD bundle; released 2026-07-22), `@tanstack/react-query` 5.107.1, `zustand` 5.1.0, `i18next` 27.1.1 + `react-i18next` 18.0.1 (te/hi/en JSON bundles; use Noto Sans Telugu/Devanagari), `vite-plugin-pwa` 2.0.0 + `workbox-window` 7.4.1.
- Tiles: self-host PMTiles (Protomaps build of Telangana) served from MinIO; cache tile pack of engineer's circle in the PWA for offline. Vector tiles of potholes via PostGIS `ST_AsMVT` endpoint in FastAPI, bucketed by H3 at low zoom.
- Low bandwidth: list views fetch only WebP thumbs (≤20 KB), full image on tap; API responses gzip/brotli; background sync queue (Workbox `BackgroundSyncPlugin`) for field photo uploads; images client-resized to 1280 px before upload.
- Field PWA offline: IndexedDB (Dexie) queue of work orders + before/after photos; conflict resolution server-wins with audit log.

## 5. Edge capture
- **Primary: Android (Kotlin, CameraX 1.5, LiteRT GPU delegate FP16)** on a ₹8–12k phone or a rugged dashcam-style Android box; INT8 YOLO showed significant mAP loss in a 2026 review, GPU FP16 ≈2× CPU; NNAPI is deprecated — use LiteRT GPU or Qualcomm QNN delegate. NCNN FP16 competitive on Mali mid-range (single blog, weak evidence) — benchmark both on target device. Use `STRATEGY_KEEP_ONLY_LATEST`.
- **Secondary: Jetson Orin Nano (JetPack 6/7) + GStreamer + TensorRT** for stereo inspection vans (Tier B) — needed for stereo depth at 15+ fps.
- Sampling by distance: trigger frame when `haversine(last_fix, now) ≥ 4 m` (≈ every 0.3 s at 40 km/h), skip when speed <5 km/h (stationary at signals) to avoid 100× duplicates. Fuse `FusedLocationProvider` 1 Hz + raw GNSS (`GnssMeasurement`, NavIC L5 on supported Snapdragon chips) + IMU heading; record `accuracy_m` per fix and down-weight >15 m.
- Store-and-forward: SQLite/Room queue; upload only detections (crop + 2 context frames, not full video) over Wi-Fi at depot (TSRTC depot Wi-Fi) or 4G when `accuracy_m` good; resumable uploads via tus or S3 multipart. Optional NMEA from external u-blox for buses with existing VTS.

## 6. Verification loop & contracts
- GHMC **CRMP**: 709 km handed to private agencies for 5 years (news; a GHMC official claimed defect-liability clause "never invoked"); CRMP Phase-2 sanctioned ₹3,145 cr across GHMC/Cyberabad/Malkajgiri (2026). Mandated agency redressal (phone/email/Twitter). No published pothole SLA found — propose: 24 h arterial / 72 h other in monsoon, 7 days otherwise, aligned with GHMC "road safety drive" practice (~20k potholes per drive).
- Data model: `pothole → work_order(contractor, sla_due, material_spec IRC:116 or IRC:SP:100, est_volume_l) → repair_event(before/after photos geo+time-stamped, batch no. of cold-mix) → verification(auto re-detection: N≥3 clean passes within 14 days; manual engineer QC sample 10 %) → recurrence_watch (DLP: re-opened within 5-yr package or 6-month patch warranty → contractor liable, auto-debit note)`.
- Payment linkage (eMARG pattern): measurement-book entry generated from estimated volume (depth × area, ×1.25 compaction/cut-back factor); contractor invoices only against `verified_closed` work orders; recurrence inside DLP → zero-pay rework. Public transparency page (per-ward open/closed counts, median repair days) for MyGHMC/citizen apps; citizen report intake via WhatsApp bot/web form merges into same cluster logic.

## Key uncertainties
geoalchemy2 × SQLAlchemy 2.1 and Python 3.14; `tensorrt` PyPI wheel on 3.14/CUDA 13.2; YOLO26 release date; monocular depth absolute accuracy (no road-scene benchmark found); exact CRMP contract SLA/DLP text (tender docs not public in search).

Sources: PyPI/npm JSON (live), fastapi.tiangolo.com/release-notes, postgis.net/2025/09/PostGIS-3.6.0, postgresql.org/docs/current/release-18.html, maplibre.org/news (Jul–Aug 2026), docs.ultralytics.com/models, arxiv.org/abs/2505.21049, arxiv.org/pdf/2509.14839, law.resource.org IRC:82-2015 / IRC:116-2014 / IRC:SP:72-2015, vikaspedia eMARG, siasat.com GHMC ArcGIS inspection app, telanganatoday.com CRMP articles, deccanchronicle.com CRMP Phase-2, developers.google.com/edge/litert/performance/delegates, towardsdatascience.com Valhalla Meili, mergify.com PEP 649 note, MagicStack/asyncpg PR 1279.

## RESEARCH: deployments
# Research report: existing pothole-AI systems, edge hardware, open GIS for Hyderabad

Facts verified via web (2026-11-12). "(unverified)" = single/weak source.

## 1. India: existing systems

**GHMC / Telangana (most relevant precedent)**
- **T-AIM "Mobility AI Grand Challenge" (Aug 2022)**: GHMC + Telangana AI Mission + Capgemini; TiHAN (IIT-H) technical, INAI-iHub Data (IIIT-H) supplied "Bodhyaan 1.0" capture platform. Task: "identify and classify pothole severity across specified routes" from live/archived video; prize up to Rs 20 lakh for GHMC pilot. No public outcome found -> treat as unfinished precedent and reuse its framing (severity classes per route). https://telanganatoday.com/t-aim-launches-mobility-ai-grand-challenge
- **Cyberabad Police CCTV pothole detection (Dec 2020)**: AI on ~5 lakh city CCTV cams to capture pothole images, forwarded to GHMC/HMDA/R&B. Key lesson: jurisdiction confusion among **four road owners (GHMC, R&B, NHAI, Secunderabad Cantonment Board)** required SOPs - your work-order router must map each segment to owning agency. https://telanganatoday.com/hyderabads-pothole-menace-to-end-soon
- **GHMC scale**: Road Safety Drive 2025: 19,380 of 20,882 identified potholes repaired (https://telanganatoday.com/road-safety-drive-over-90-of-potholes-repaired-in-greater-hyderabad-ghmc-commissioner-admitted); 2026 drive >19,000 filled (https://telanganatoday.com/ghmc-fills-over-19000-potholes-across-hyderabad-as-part-of-road-safety-drive). Deccan Chronicle 2021: GHMC spent ~Rs 7,396 per pothole vs ~Rs 1,600 budgeted; materials = bulk bitumen, crushed metal, cationic emulsion; Rs 95.77 cr on Monsoon Emergency Teams Jun-Oct (https://www.pressreader.com/india/deccan-chronicle/20210727/281517934152105). GHMC tenders "Shelmac bags + emulsion for cold-mix pothole filling during monsoon" (Serilingampally, ~Rs 3.95 lakh lots) (https://tenders.infralens.in/tender/19-ee-cir-49-slpz-cmc-1-10-2026-709807) - cold-mix is already procured; your volume output -> bags needed is a direct fit.
- **TGSRTC**: has ADAS pilots on Lahari/Rajadhani/Garuda buses (40% accident reduction claimed) - precedent for cameras on buses, but not pothole-related. https://telanganatoday.com/tgsrtc-plans-to-install-ai-powered-alert-adas-in-buses
- **GHMC restructuring (2026)**: GHMC split into 3 corporations (GHMC ~689 km², 6 zones/30 circles/150 wards; plus Cyberabad MC, Malkajgiri MC); ward count contested (150 vs 300) and litigated. Design ward/zone layer as swappable. https://newsmeter.in/top-stories/ghmc-20-explained-3-municipal-corporations-300-wards-new-administrative-setup-762906

**RoadMetrics (Bengaluru, 2019)**: smartphone camera + accelerometer/gyro; B2B to municipalities; claims 50,000 km, 90-95% accuracy (company-sourced, no independent eval); worked with ELCITA. Play Store app only ~600 downloads -> crowdsourcing by citizens failed, fleet-mounted B2B worked. https://frontiertech.niti.gov.in/story/ai-driven-road-management-enhancing-indias-infrastructure-with-roadmetrics/ ; https://www.outlookbusiness.com/enterprise/big-idea/difficult-roads-are-what-led-to-the-inception-of-the-deep-tech-start-up-roadmetrics-5992

**Bengaluru North/East City Corporations (2026)**: AI survey of 2,493 km found 14,393 potholes; East: 1,600 km, uses a VLM to measure length/width of each pothole, 5 road-condition grades, also cracks/rutting/utility cuts. Vendor unnamed. No depth. https://swarajyamag.com/news-brief/ai-survey-detects-over-14000-potholes-on-north-bengaluru-roads ; https://newsable.asianetnews.com/karnataka-news/bengaluru-east-civic-body-uses-ai-identify-potholes-across-1600-km-of-roads-articleshow-za4mqhr

**Gaurav Sen dashcam project (Aug 2026, viral)**: dashcam+GPS+accelerometer, cloud VLM (OpenAI), 3 size classes, distinguishes speed-breakers, links each detection to ~2,900 public road contracts -> tender ID/contractor/officer -> auto complaint in ~4 s. No accuracy metrics; cloud-only. Lesson: **contractor/warranty linkage** is the governance feature people want. https://www.businesstoday.in/technology/artificial-intelligence/story/no-hiding-behind-potholes-bengaluru-engineers-ai-app-tracks-road-damage-names-contractors-549536-2026-08-17

**iWatchRoad v2 (arXiv 2510.16375, IISER/PCMC-motivated)**: dashcam + external GPS logger; YOLOv8 fine-tuned on **BharatPotHole** (7k frames, 640px); negatives (shadows, manholes) added to cut FPs; severity from bbox geometry; OCR of dashcam timestamp to sync GPS (found fixed 5h30m44s offset bug); within-video dedup by Haversine clustering at **2.5 m**; cross-pass dedup listed as *unsolved*; PostGIS+Leaflet+Django/React; contractor/warranty colour-coding, Twilio/Firebase alerts, escalation, repair verified by later drive. Limits: parallel/multi-lane map-matching ambiguity, OCR errors. https://arxiv.org/html/2510.16375v1 (v1: https://arxiv.org/html/2508.10945v1)

**Others**: Intents Mobi (Intents Go) - accelerometer-only, 1.85 lakh potholes/bumps via cab/truck drivers (https://www.autocarindia.com/bike-news/intents-go-418506). Rasta.AI (Pune) - camera-based, Latur PWD survey Rs 1.99 cr, first Maharashtra PWD AI survey (https://www.freepressjournal.in/pune/pune-based-firm-uses-ai-for-road-surveys-in-latur-first-of-its-kind-project-in-state). RoadBounce (Pune) - roughness sensors. drivebuddyAI - patent on real-time road-quality mapping. BMC Mumbai "Pothole QuickFix" (Jun 2025) - citizen app + WhatsApp bot, 48-h SLA, 3,018 complaints in June; BMC 2022 tender sought AI/ML vehicle-based system, outcome unknown (https://www.freepressjournal.in/mumbai/mumbai-news-bmc-launches-pothole-quickfix-app-and-whatsapp-chatbot-for-faster-road-repair-complaints). MoRTH/PMGSY: national AI pilot detecting defects from phone video, field validation from Sept (https://yourstory.com/ai-story/ai-road-defects-system-rural-roads-india). NHAI: 3D-laser Network Survey Vehicles, 20,933 km in 23 states, 300 km/day (https://www.newsonair.gov.in/nhai-to-deploy-3d-survey-vehicles-in-23-states-for-ai-based-highway-monitoring). Kerala PWD, "Sadak Sudhar", Varanasi: nothing found.

**Recurring Indian failure modes**: counts disputed (BBMP 448 vs BTP 7,500 vs app 2x) because of no shared pothole *definition*; citizen apps ineffective; cloud-only and accelerometer-only produce FPs (speed-breakers, manholes - Boston found sunk manholes 4x potholes); no depth/volume, so material quantities and contractor billing stay unverifiable; patchwork fails after rain (https://www.deccanchronicle.com/amp/southern-states/telangana/poor-patchwork-leaves-hyderabad-roads-full-of-potholes-again-1975536).

## 2. Global systems
- **RoadBotics/Michelin RoadWay**: windshield phone, 250+ governments; segment rating 1-5 colours. https://mobilitydata.michelin.com/node/29
- **vialytics (DE)**: phone behind windshield, 15 damage classes, offline-first app with work orders syncing on reconnect (good model for low bandwidth). Dedup/severity internals not public. https://apps.apple.com/us/app/-/id1644659020
- **Blyncsy (Bentley)**: 1.5M+ crowdsourced dashcams (Nexar), automated PASER scoring, claims within 1% of LiDAR; Hawaii DOT: detections "identified, deduplicated, verified" against existing issues and user reports, "data to work orders in 3 steps". Method undisclosed. https://blyncsy.com/wp-content/uploads/2024/03/Hawaii-DOT-Case-Study.pdf
- **Boston Street Bump**: accelerometer+GPS; relied on repeat hits across users to confirm; FP <10% after BU work; manholes dominated. https://www.boston.gov/departments/new-urban-mechanics/street-bump
- **Kent CC + Amey + Route Reports (2020)**: cameras on Arriva buses (ADEPT Live Labs) - the closest "bus camera" precedent (details page gone). https://amey.co.uk/media/news/2020/november/amey-and-kent-county-council-use-buses-to-spot-pot-holes
- **Stan app (Metricell, UK)**: phone AI reports width *and depth*; 11.5M-pothole national estimate. https://www.highwaysmagazine.co.uk/App-data-suggests-11.5-million-potholes-in-UK/13988
- **Dedup across passes (academic, usable)**: Pothole Patrol (MobiSys'08) server-side clustering filters spurious detections; later work uses DBSCAN spatial-density clustering across vehicles. Combine: map-match to OSM way + direction, cluster within ~3-5 m along-track, require >=2 independent passes to "confirm", N clean passes to "auto-close".

## 3. Depth measurement literature
- Monocular: SAM area + DPT depth -> R² 0.95 area / 0.77 volume (https://trid.trb.org/View/2752100); BoT-SORT + Depth Anything V2 + triangulated area (BMVC 2025 workshop: https://bmva-archive.org.uk/bmvc/2025/assets/workshops/DIFA/Paper_2/paper.pdf); YOLOv8-seg + depth map fusion (https://arxiv.org/abs/2504.13648). Caveat: Depth Anything gives *relative* depth; need metric scaling (known camera height/pitch + ground-plane fit, or stereo).
- Stereo: SAE 2026-26-0269 shows polarized-lens stereo improves depth in harsh light; stereo fails at night; LiDAR accurate but costly (https://saemobilus.sae.org/papers/comparative-analysis-pothole-depth-estimation-using-lidar-stereo-imaging-polarized-lenses-a-doe-approach-2026-26-0269). Kinect point-cloud volume integration precedent (1.76 L example).
- Standards: IRC:SP:100-2014 (cold mix with bitumen emulsion) and IRC:116-2014 (readymade cut-back patching mix) are the Indian specs; CSIR "Patchfill" machine makes 70 kg/3 min per IRC SP 100/116 (https://techindiacsir.anusandhan.net/online/patchfill-the-pothole-reparing-machine-T-135-tech.htm). IRC:82 depth thresholds not confirmed online (unverified; commonly cited <25 mm low, 25-50 mm medium, >50 mm high - verify from IRC:82-2015 PDF).

## 4. Edge hardware (India)
- **Jetson Orin Nano Super dev kit**: ThinkRobotics Rs 54,999 in stock; blogs say ~Rs 25-35k via RP Tech/Arrow/TannaTechBiz with 1-2 month wait (unverified); bare module Rs 28,900. Dev kit rated 0-35 C (Robu) - **unsuitable in-bus in Telangana summer**; industrial carriers (RDP CARINA -20..70 C; Seeed reComputer J3011 -10..60 C, Robocraze) needed. https://thinkrobotics.com/blogs/product-reviews-buying-guides/jetson-orin-nano-price-complete-buying-guide-for-ai-robotics-development ; https://www.rdp.in/ai/j-series/orin-nano
- **Raspberry Pi 5 + Hailo-8L AI Kit (13 TOPS)**: Rs 6,600-10,000 (ThinkRobotics, Probots Rs 9,749+GST); plus Pi 5 8GB ~Rs 8-9k. https://thinkrobotics.com/collections/ai-hardware ; https://probots.co.in/official-raspberry-pi-ai-hat-13-tops.html
- **RK3588 (Orange Pi 5)**: no reliable INR price found (unverified; typically Rs 9-14k importers).
- **Android phone Rs 10-15k** running NCNN/TFLite YOLOv8n at 10-20 fps is the pragmatic pilot sensor (RoadMetrics/vialytics/RoadBotics all phone-based); but phones throttle/shut down >45 C on dashboard - needs shaded mount, hardware-store fan, or dedicated camera + Pi/Jetson in ventilated enclosure. 24 V bus power: use 24->5 V/12 V isolated DC-DC with surge/ignition-off handling (standard practice; no India-specific source).
- **4G data**: Airtel IoT plans from Rs 29/month (tiny data); reseller 4G IoT SIMs 10-20 GB/month ~Rs 90-110; consumer overage Rs 10/GB, effective ~Rs 4/GB. Upload only crops + metadata (<1 GB/bus/month). https://circuitdigest.com/articles/how-can-you-get-and-use-an-m2m-iot-sim-card-in-india

## 5. Open GIS data (Hyderabad)
- **TGSRTC GTFS**: official feed mdb-3361, 1,031 routes, updated Jul 2026, no validation errors: https://files.mobilitydatabase.org/mdb-3361/mdb-3361-202607201528/mdb-3361-202607201528.zip (source: data.opencity.in "telangana_opendata_gtfs_tgsrtc_08_february_2026.zip"); HMRL/MMTS GTFS at https://data.telangana.gov.in/story/gtfs-hyderabad-public-transport-systems. Bus stops CSV/KML: https://data.opencity.in/dataset/hyderabad-bus-stops
- **Admin boundaries**: Telangana district/mandal/revenue shapefiles (OGD licence, form required): https://data.telangana.gov.in/story/telangana-district-mandal-and-revenue-shape-files and AIKOSH; GeoJSON districts/mandals/villages: https://github.com/gggodhwani/telangana_boundaries. **GHMC ward GeoJSON**: no official download; community files via Telangana Open Data Community / DataMeet threads (https://groups.google.com/g/datameet/c/Mq3zhhRgJj8) - unofficial, and boundaries changed in 2026.
- **POIs**: "Geo Locations of Amenities present in GHMC" https://data.telangana.gov.in/dataset/geo-locations-amenities-present-ghmc-hyderabad (format unconfirmed); health centres lat/long and Anganwadi lat/long datasets on same portal; supplement with OSM amenity=school/hospital via Overpass. Roads: OSM (Geofabrik "southern-zone" India extract) with highway class + maxspeed for priority weighting.
- Portal index: https://wiki.openstreetmap.org/wiki/Open_Geospatial_Data_from_Government_of_Telangana

## Design implications (from evidence)
1. Edge detect on-vehicle, upload crops+GPS+IMU only; offline-first like vialytics.
2. Add negatives (manholes, shadows, speed-breakers, wet patches) - top FP sources in India/Boston.
3. Depth: YOLOv8-seg + metric depth (Depth Anything V2 + ground-plane scale from fixed camera height; optional stereo module) -> depth class per IRC + volume -> cold-mix bags (IRC:SP:100 / IRC:116).
4. Cross-pass dedup via map-matched DBSCAN; confirm on >=2 passes; auto-verify repair on N clean passes (nobody in India has shipped this).
5. Route work orders by road owner (GHMC/R&B/NHAI/Cantonment) and link to contractor + defect-liability period.
6. Keep ward layer pluggable (GHMC 150/300 wards, 3-corporation split).

## RESEARCH: depth
# Research report: pothole depth & volume from vehicle-mounted cameras (India focus), 2026-11-12

## 1. Monocular metric depth models (state as of Nov 2026)

| Model | HF repo id(s) | Params | Metric? | Licence | Outdoor accuracy (AbsRel, KITTI) | Notes |
|---|---|---|---|---|---|---|
| **Depth Anything V2 Metric (outdoor)** | `depth-anything/Depth-Anything-V2-Metric-VKITTI-{Small,Base,Large}` (indoor: `...-Metric-Hypersim-{Small,Base,Large}`) | 24.8M / 97.5M / 335.3M | Yes, metres, `max_depth=80` outdoor, 20 indoor | HF card for VKITTI-Large says **apache-2.0** (relative-depth Base/Large are CC-BY-NC-4.0; verify per card) | Fine-tuned on **Virtual KITTI 2** only. Official card gives no numbers; secondary source quotes AbsRel ~0.045 on KITTI (uncertain). Metric3D repo table lists DepthAnything ViT-L 0.046 / δ1 0.982 | Scale is fixed to VKITTI's camera (fx≈725 px, height ~1.5 m). On a bus camera with different focal/height the absolute scale drifts 10-30% → must be re-scaled (see §2/§5). No `-hf` transformers variant for metric; use repo code `DepthAnythingV2(...).infer_image()`. |
| **Depth Anything 3 (DA3, ByteDance-Seed, Nov 2025)** | `depth-anything/DA3METRIC-LARGE` (0.35B, **Apache-2.0**, monocular metric); `depth-anything/DA3MONO-LARGE` (relative, Apache); `DA3-SMALL`/`DA3-BASE` (0.08B/0.12B, Apache, relative); `DA3NESTED-GIANT-LARGE-1.1` (1.4B, metric, **CC-BY-NC-4.0**) | | DA3METRIC: `metric_depth = focal_px * net_output / 300` → you supply focal in px from calibration (good: scale tied to your intrinsics) | Apache for METRIC-LARGE | **No KITTI/outdoor AbsRel published in README**; claims "significantly outperforms DA2". Treat as uncertain | Repo github.com/ByteDance-Seed/Depth-Anything-3; ONNX port `Heliosoph/da3metric-large-onnx`; LightlyTrain ports `dinov2/dav3-metric-{small,large}`. Roboflow added DA3 small/base Jan 2026. Recommended primary candidate. |
| **Metric3D v2** | torch.hub `yvanyin/metric3d` (`metric3d_vit_small/large/giant2`); ONNX `onnx-community/metric3d-vit-{small,large,giant2}`; ckpt `JUGGHM/Metric3D` | ViT-S/L/g | Yes; canonical-camera model, needs intrinsics (fx) to rescale; w/o intrinsics tries 9 default focals | **BSD-2-Clause** (commercial OK) | KITTI AbsRel 0.044 (ViT-L), 0.039 (ViT-g2), δ1 0.985/0.989 (table in repo, likely fine-tuned/not pure zero-shot) | Also outputs normals (useful for road-plane). Heavy (512x1088 input). |
| **UniDepthV2** | `lpiccinelli/unidepth-v2-vits14 / vitb14 / vitl14` | ViT-S/B/L | Yes; predicts intrinsics or accepts `Pinhole(K=...)` | **CC-BY-NC-4.0** (non-commercial → not for govt deployment without licence) | KITTI δ1 98.9 (README); AbsRel 0.037 only for KITTI-fine-tuned | Research only. |
| **Depth Pro (Apple)** | github apple/ml-depth-pro; `apple/DepthPro` (HF, via `depth_pro` lib); transformers `apple/DepthPro-hf` (uncertain id) | ~0.9B (uncertain) | Yes, metres + focal length estimate, 2.25 MP in 0.3 s on GPU | **apple-amlr** (Apple ML Research licence, research/non-commercial terms, verify) | No KITTI number on card | Good boundaries; licence problematic. |
| **ZoeDepth** | `Intel/zoedepth-kitti`, `Intel/zoedepth-nyu-kitti` (transformers-native) | ~345M | Yes | **MIT** | KITTI AbsRel ~0.057 (from Metric3D table: 0.053) | Older (2023); fine as baseline. |
| Gamma-from-Mono (arXiv 2512.04303, Dec 2025) | code not confirmed | 8.88M (LiteMono) | Metric from **camera height only** (predicts road normal + height/depth ratio γ) | unclear | KITTI AbsRel 0.085 (<20 m: 0.058); RSRD AbsRel 0.034 | Exactly the "road-relative" formulation you want; tiny; worth tracking. |

**Reliability of absolute scale on dashcams:** all metric models are trained for a canonical focal; absolute scale on an unseen bus camera is typically off by 5-30% unless you (a) feed focal px (DA3METRIC, Metric3D, UniDepth) and (b) still apply a per-frame scale correction from camera height / road plane (§2). Recommended: use the model only for *relative* geometry and fix scale with known camera height (robust, free).

## 2. Technique: road plane + mask integration (the recommended pipeline)

1. Segment pothole (YOLOv8/11-seg or SAM2 refinement) + road mask (any road seg or DA3 sky seg + plane inliers).
2. Back-project depth map to point cloud with K (fx,fy,cx,cy).
3. Fit road plane with RANSAC (open3d `segment_plane`, thr ≈ 1-2 cm, 1000 iters) on road pixels **excluding** pothole masks, within a 3-12 m band in front of the vehicle. Prior works: Fan et al. "Pothole Detection Based on Disparity Transformation and Road Surface Modeling" (arXiv 1908.00894, disparity-domain quadratic road model + RANSAC); Dhiman & Klette RANSAC road manifold; Xue et al. 2020 (arXiv 2009.07714) fit road plane with LMedS on road-segmented pixels for pitch/roll.
4. **Scale correction:** the fitted plane gives camera height ĥ (distance from origin to plane). Multiply all depths by s = h_true/ĥ where h_true is the measured mounting height (e.g. 2.4 m on a TSRTC bus dash, 1.4 m on a car). This cancels most monocular scale error and is what Gamma-from-Mono formalises.
5. Pothole depth per pixel = signed distance below plane; report max depth, mean depth (robust: 95th percentile), area = Σ pixel footprint on plane (per-pixel footprint ≈ (Z/fx)·(Z/fy)/cos θ, i.e. IPM footprint), volume = Σ depth_i · area_i. Accumulate across frames with tracking (BoT-SORT) + Kalman smoothing as in Wang et al. 2025 (arXiv 2505.21049: YOLOv8 + Depth Anything V2 + "MBTP" area + constant-area Kalman; depth 104 ms/frame bottleneck, area only, no volume GT).
6. Material quantity = volume × 1.1-1.2 (compaction/overfill) × cold-mix density ~2.0-2.2 t/m³ (verify with R&B specs).
7. Related: ISPRS Annals X-4/W7-2025 p.75 (Thailand) — segmentation + IPM to get real pothole dimensions, single-camera calibration per vehicle. arXiv 2504.13648 (segmentation + depth map per pothole). 2025 single-camera SAM+DPT framework: R² 0.95 area, 0.77 volume vs ground truth (TRID 2752100). IPM error analysis: Markó et al. 2024 (Engineering Proceedings).

## 3. Alternatives and tradeoffs for bus deployment

- **Stereo (ZED 2i):** best-documented pothole result: ISPRS Archives XLVIII-4/W14-2025 p.291 — ZED 2i + GNSS (BD-8953U), pitch 20°, polarizer, 8.33 ms exposure, Mask R-CNN w/ depth attention, RANSAC ground plane → **depth rel. error 12.53%, area 18.19% at 40 km/h**; full coverage up to 72 km/h; calibration: Zhang's method, 9×6 checkerboard, 30 mm squares, focal error <0.3%, baseline 0.1 mm. Price India: ₹52,359 (no polarizer, electronicscomp.com) / ₹62,954 + 18% GST (polarizer, electropi.in); US list $499. Needs Jetson/PC host (no onboard compute).
- **OAK-D / OAK-D-Lite (Luxonis):** onboard stereo depth + NN (RVC2), 75 mm baseline → practical depth 0.5-6 m, too noisy for mm-level at 8-15 m; good as a near-range verifier on municipal trucks. India price: OAK-D-Lite ₹31,650 (thinkrobotics) / ₹35,314 (ubuy); OAK-D ₹30,530, OAK-D-IoT-40 ₹28,435 (tanotis, incl. GST+duty). Luxonis raising OAK-D-Lite list from $169→$269 from 9 Jul 2026.
- **Smartphone LiDAR (iPhone 13 Pro+):** handheld validation only (range ≤5 m): iRodd study (Indonesia, JIPD) volume avg error 5.35%, RMSE 6.47 mm; IJIRT paper (iPhone 13 Pro) reliable volumes. Use as *ground-truth collection tool* for GHMC inspectors (cheap, 1 device per zone). Android: use ARCore depth (weaker).
- **IMU/accelerometer on bus phone:** Harikrishnan & Gopi 2017: depth via double-integration / 1-DoF wheel model, 23 potholes, ~15% depth error (13% aggregated), requires vehicle+phone placement params. Good as corroboration/trigger and for night/rain; cannot give area/volume; bus suspension damps signal. Indian precedents: RGIPT ISPRS 2022, Manipal bus-cloud system.
- **Structured light / line laser:** Vupparaboina et al. camera+laser: 98% dry, 94.7% water-filled accuracy; needs night/shade; impractical on buses, fine for a repair-crew verification rig.
- **Terrestrial LiDAR TIN (Civil Eng. Journal 2025):** traditional cuboid (L×W×D) estimates are **2-4× larger** than true volume → your integrated-volume approach directly saves material/cost; cite this.
- **Thermal camera** patent (ipiq 4676260): water-filled pothole depth from surface temperature — exotic, skip.

## 4. Accuracy expectations (bus, 20-40 km/h, 5-15 m)

- Stereo ZED at 40 km/h: ~12.5% depth error (ISPRS 2025) → ~6 mm on a 50 mm pothole, ~19 mm on 150 mm, but only inside ~8 m. Beyond ~10 m stereo depth noise (≈Z²·disp_err/(f·B)) exceeds pothole depth.
- Monocular metric at 8-15 m: per-pixel depth noise after plane-fit normalisation realistic **±10-25 mm** for depth, **±20-40 % volume** single-frame; multi-frame fusion (5-15 frames as bus approaches to ≤5 m) brings depth to ±5-15 mm and volume ±15-25 % (my estimate; no published bus-mounted monocular GT study exists — flag as uncertain). Best measurements come from the last 3-6 m before passing; use time-to-pass ordering.
- Classification is what matters: IRC-82-based severity bands used in Indian course material: **low <25 mm, moderate 25-50 mm, high >50 mm** (secondary source citing IRC 82; verify in IRC:82-2015 PDF law.resource.org/pub/in/bis/irc/irc.gov.in.082.2015.pdf; IRC:82-2023 revision exists). A system with ±10 mm error reliably bins 25/50 mm classes.
- Motion blur: 40 km/h = 11 m/s; at 1/500 s exposure → 2.2 cm smear (acceptable); use global-shutter or ≤8 ms exposure (ISPRS used 8.33 ms) and ≥30 fps; Wang et al. note failure on "highly blurred potholes".
- **Water-filled potholes:** monocular depth models see a flat specular surface → depth ≈ plane (under-estimated). Detect via: (a) segmentation class "water/puddle", (b) depth-below-plane ≈ 0 while RGB detector fires, (c) polarization (SAE 2026-26-0269 shows polarized stereo helps; Jan 2026, DOE over water/lighting/distance; no numbers in abstract), (d) refraction-based estimation (Scitepress 2021 "Water Hazard Depth Estimation") is research-grade. Policy: flag `water_filled=true, depth=unknown`, set priority by area + complaint count, re-measure after rain or by IMU jolt magnitude.
- Night: Wang et al. CDKF dark-set tuning; buses run dawn/dusk — expect 20-30 % more misses.

## 5. Calibration procedure for a bus camera

1. **Intrinsics (once per camera model):** OpenCV `calibrateCamera` with 9×6 (or 10×7) checkerboard, 30-40 mm squares printed on A2/foam board, 15-25 poses; target reprojection <0.3 px; also estimate distortion (k1-k3, p1-p2) and undistort before depth model. Lock focus/zoom (fixed-focus lens preferred; OAK-D-Lite has FF variant).
2. **Extrinsics (per vehicle, 10 min at depot):** measure camera height h with tape; lay a 2×2 m floor grid or 4 cones at known distances (3, 5, 8, 12 m) on the depot apron → solve pitch/roll/yaw by PnP (`solvePnP`) from ground points; store (h, pitch, roll). Pitch ~15-25° down worked for stereo (ISPRS: 20°).
3. **Online self-check (every trip):** (a) vanishing point from lane markings → pitch/yaw; (b) **lane-width prior** (IRC:86/IRC:SP:73 urban lane 3.5 m; urban Hyderabad often 3.0-3.5 m, so use 3.5 m ± 0.25 m only on marked arterials) → roll + height, per Lee et al. arXiv 2008.03722 (lane-width-prior extrinsic calibration for IPM), ISPRS XLIII-B2-2022 p.687; (c) RANSAC plane height vs tape-measured h → scale factor; alert if drift >5 %.
4. **Known-object check:** standard 1 m × 1 m painted square or a traffic cone (IRC cone height 750 mm) placed at 5 m; verify reconstructed size within 5 %.
5. **Validation set (ground truth):** inspector measures 50-100 potholes with straightedge + depth gauge (IRC manual method) and iPhone LiDAR scans; compute depth MAE, volume MAPE per distance bin; publish per-bus calibration cert.

## 6. Recommended stack for this project (dev box: CUDA 13.2, Py 3.14)

- Detector/seg: YOLO11-seg (AGPL) or RT-DETR/RF-DETR (Apache) fine-tuned on Indian pothole datasets (RDD2022 India subset, Kaggle "Pothole detection India").
- Depth: **DA3METRIC-LARGE (Apache)** primary, feed calibrated fx; fallback **Depth-Anything-V2-Metric-VKITTI-Base**; Metric3D v2 ViT-S (BSD) as ONNX edge option. Avoid UniDepth/Depth Pro for production (NC licences).
- Geometry: open3d RANSAC plane, numpy integration, BoT-SORT tracking, Kalman fusion of depth/area/volume over frames.
- Edge: Jetson Orin Nano + global-shutter 2 MP cam (₹25-40k per bus kit); or phone dashcam uploading 1 fps keyframes over 4G with on-device YOLO trigger (low bandwidth).
- Note Python 3.14: torch wheels for 3.14/CUDA 13 — verify availability; use a 3.12 venv if needed (uncertain).

## Key URLs
- DA3: https://github.com/ByteDance-Seed/Depth-Anything-3 ; https://huggingface.co/depth-anything/DA3METRIC-LARGE
- DA2 metric: https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth ; https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large
- Metric3D: https://github.com/YvanYin/Metric3D ; UniDepth: https://github.com/lpiccinelli-eth/UniDepth ; Depth Pro: https://github.com/apple/ml-depth-pro ; ZoeDepth: https://huggingface.co/Intel/zoedepth-kitti
- Gamma-from-Mono: https://arxiv.org/abs/2512.04303 ; Wang et al. 2025: https://arxiv.org/abs/2505.21049 ; Fan et al.: https://arxiv.org/abs/1908.00894 ; seg+depth: https://arxiv.org/abs/2504.13648
- Stereo pothole ISPRS 2025: https://isprs-archives.copernicus.org/articles/XLVIII-4-W14-2025/291/2025/ ; SAE polarized stereo vs LiDAR 2026: https://saemobilus.sae.org/papers/2026-26-0269 ; IPM Thailand: https://isprs-annals.copernicus.org/articles/X-4-W7-2025/75/2025/
- LiDAR TIN vs cuboid volume: https://civilejournal.org/index.php/cej/article/view/5954 ; iRodd phone LiDAR: https://systems.enpress-publisher.com/index.php/jipd/article/download/6162/4175
- Accelerometer depth review: https://arxiv.org/pdf/1908.10187 ; lane-width calibration: https://arxiv.org/abs/2008.03722
- IRC:82-2015: https://law.resource.org/pub/in/bis/irc/irc.gov.in.082.2015.pdf
- Prices: electronicscomp.com / electropi.in (ZED 2i), thinkrobotics.com / tanotis.com (OAK-D), shop.luxonis.com, store.stereolabs.com

Uncertain items: DA3 outdoor accuracy (no published KITTI numbers), DA2-metric exact AbsRel, Depth Pro licence terms and HF transformers id, IRC 82 severity thresholds (secondary source), monocular bus-mounted accuracy estimates (no published GT study), torch support for Python 3.14/CUDA 13.2.

## RESEARCH: datasets-models
# Research report: pothole detection + segmentation for Indian roads (as of 2026-11-12)

## 1. Datasets with Indian road imagery

**RDD2022 – India subset (primary, best-documented Indian set)**
- Official download (direct S3 zip, train images + PASCAL-VOC XML; test images without labels): `https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip` (502 MB). All-countries zip: `.../RDD2022/RDD2022.zip` (~13 GB). Also on Figshare: https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547 . Index page: https://github.com/sekilab/RoadDamageDetector
- Licence: CC BY-SA 4.0 (images). Paper: arXiv 2209.08538 (IIT Roorkee + U Tokyo).
- India numbers (from paper, verified via PDF): 7,706 train / 1,959 test = 9,665 images; 6,831 train labels. Per-class India labels: D00 longitudinal crack 1,555; D10 transverse crack 68; D20 alligator crack 2,021; **D40 pothole 3,187** (India has the most potholes of any country). Capture: smartphone mounted in car, 960x720 resized to 720x720; roads in Delhi, Gurugram and Haryana (local, state and national highways) — flat plains, no Telangana data. D10 is near-absent (68) so expect poor D10 recall; consider merging crack classes.
- Note: an ISPRS 2025 paper that used "India subset: 2,047 train / 512 val" was a filtered subset, not the full set.
- Bounding boxes only (no masks). HF mirror (unofficial, re-split): `akiii1234/RoadDamageDetection` (CC BY-SA 4.0). Roboflow copies: `universe.roboflow.com/road-damage-detection-b5a1c/rdd-india`.

**BharatPotHole (iWatchRoad / NISER, 2025)** – 7,000+ 640x640 dashcam frames from Indian rural, urban, village and highway roads (May–July, monsoon-adjacent), YOLO boxes, single class. Paper says public on Kaggle: `kaggle datasets download surbhisaswatimohanty/bharatpothole` (page exists but I could not render it; licence and exact count UNVERIFIED — check before use). Papers: iWatchRoad arXiv 2508.10945, iWatchRoadv2 arXiv 2510.16375 (CC BY-NC-SA text); project page smlab.niser.ac.in/project/iwatchroad. Model: YOLOv8 fine-tune; mAP only in figures.

**Semantic Segmentation Dataset of Indian Roads (eYantra IIT Bombay, 2021)** – 3,227 images (2,475 train / 752 test), 720x1280, pixel masks for road / pothole / footpath / shallow / other (8,129 objects). Kaggle slug: `eyantraiit/semantic-segmentation-datasets-of-indian-roads`. Licence: GNU GPL 2.0 (restrictive for commercial use; fine for a hackathon prototype). Mirror: https://datasetninja.com/indian-roads-semantic-segmentation . This is the only confirmed Indian **mask** dataset for potholes.

**IDD (IIIT-Hyderabad)** – https://idd.insaan.iiit.ac.in / inai.iiit.ac.in/domains/idd.html. 10k images, 34 classes, shot in Hyderabad and Bengaluru, but **no pothole/road-damage class**. Useful only as Hyderabad-domain background/pretraining data (road, drivable area). IDD-3D is LiDAR, no damage labels.

**Kaggle (non-Indian / mixed origin)**
- `chitholian/annotated-potholes-dataset` – 665 images, 1,740 boxes, VOC XML, ODbL v1.0 (also https://public.roboflow.com/object-detection/pothole).
- `alvarobasily/road-damage` – 3,321 smartphone 1080p images, 4 classes (pothole, alligator, longitudinal, lateral), CC0 1.0.
- Mendeley "Multi-Weather Pothole Detection (MWPD)" doi 10.17632/s5hx9n2jc3 – boxes, varied weather (not India).

**Roboflow Universe pothole segmentation (polygon masks, CC BY 4.0 unless noted; origin mostly non-Indian/unverified)**: `utm-cuu2y/potholes-segmentation`, `ateeks-workspace/pothole-segmentation-qrfpf`, `segmentation-tasks/pothole-segmentation`, `vnr-project/pothole-segmentation-8esmn` (VNR = VNR VJIET Hyderabad, likely local imagery — UNVERIFIED), `hruthik-sivakumar/pothole-segmentation-ucbbw`, `nguyen-nhut-nhut/yolo_v8_segmentation`. HF mirrors: `keremberke/pothole-segmentation` (90 imgs, COCO), `manot/pothole-segmentation2`. Download via `roboflow` pip + API key, format `yolov11`/`yolo26` seg.

**Hyderabad/Telangana-specific**: none public. GHMC reports counts only (e.g., 22,797 identified / 21,511 repaired by Nov 2025; 19k+ in Jun 2026 drive, telanganatoday.com). Plan to self-collect (TSRTC dashcam/phone) and pseudo-label with the bootstrapped model.

**Depth datasets**: no open pothole depth/volume benchmark exists (arXiv 2504.13648, Apr 2025 states this explicitly; they built a private IR-camera set). Known small sets: 981 Kinect-V2 RGB-D pairs (GeneticsMR paper, not public link found); GitHub `stereo_pothole_datasets` (stereo, from Fan et al. disparity work). A 2026 ISPRS study found Depth Anything 3 yields only relative depth for potholes without calibration. Practical options: (a) metric monocular via `APaul1/DA3METRIC-LARGE` (Apache 2.0, depth scaled by focal length in px — scaling convention disputed: HF card says multiply by focal; Replicate wrapper uses focal*depth/300; validate on a calibrated hole) plus RANSAC road-plane fit, integrate (plane − depth) over mask = volume; (b) stereo (ZED 2i, OAK-D) or cheap RGB-D for a ground-truth calibration rig; (c) known-width reference (lane marking/wheel) for pixel-to-metre scale.

## 2. Model family (late 2026)

- **YOLO26 (Ultralytics, released 14 Jan 2026)** is the current default for edge: NMS-free end-to-end head, no DFL (simpler INT8/NPU export), MuSGD, STAL small-object assignment. COCO det: n 40.9 mAP / 1.7 ms T4-TRT / 2.4M params; s 48.6 / 2.5 ms; m 53.1 / 4.7 ms; l 55.0; x 57.5. **Segmentation supported**: yolo26n-seg 39.6 box / 33.9 mask, 2.1 ms T4, 2.7M; s-seg 47.3/40.0, 3.3 ms; m-seg 52.5/44.1, 6.7 ms; l-seg 54.4/45.5; x-seg 56.5/47.0 (https://docs.ultralytics.com/tasks/segment/). Licence AGPL-3.0 (commercial needs Ultralytics Enterprise licence — flag for govt deployment). YOLO26n on Jetson Orin Nano Super: 7.53 ms FP32 / 4.57 ms FP16 / 3.80 ms INT8; AGX Orin: 2.62 ms FP16 (https://docs.ultralytics.com/guides/nvidia-jetson/).
- **YOLO11** (54.7 mAP for x) still fine; YOLO12 attention-based, slower on CPU — less attractive for edge.
- **RF-DETR (Roboflow, Apache 2.0 for Nano–Large)**: DINOv2 backbone, better domain-shift robustness (RF100-VL). COCO: Nano 48.4 AP / 2.3 ms T4, Small 53.0 / 3.5, Medium 54.7 / 4.4, Large 56.5 / 6.8 (`rf-detr-large-2026.pth`). **RF-DETR Seg** exists: Seg-Nano 40.3 AP / 3.4 ms … Seg-Large 47.1 / 8.8 ms. `pip install rfdetr` (Python ≥3.10). Exports ONNX, TensorRT, TFLite/LiteRT, OpenVINO, ExecuTorch, CoreML. https://rfdetr.roboflow.com/latest/
- Road-damage-specific numbers (none directly on RDD2022-India for YOLO26): SEA-YOLOv8 63.2 mAP50 on full RDD2022 (PMC12176152); improved YOLO11 +2 mAP50 over base on RDD2022 (PMC12469154); **YOLO26-RD (arXiv 2608.15713, Aug 2026)**: YOLO26-l variant 0.809 mAP50 on a road-damage test split (+0.031 over stock YOLO26-l); YOLO26-l on Indonesian potholes 0.689 mAP50 / 0.339 mAP50-95 (JAIC Aug 2026); YOLOv8n-seg on potholes 93.8 mAP (arXiv 2505.04207); YOLOv12m-seg box and mask ~0.80–0.85 (Caltrans/UC Berkeley report Apr 2026). Independent RF-DETR vs YOLO26 road-damage comparisons do not exist; a traffic-scene blog ranked RF-DETR > YOLO26 > YOLO11.
- **Recommendation**: train YOLO26s-seg / m-seg (masks → area; e2e export) as the edge model; train RF-DETR-Seg-Small as the server-side verifier/teacher for pseudo-labelling Hyderabad data. Benchmark both on RDD2022-India D40 + eYantra masks.

## 3. Pre-trained pothole checkpoints (bootstrap)
- `mfranzon/pothole-yolo26` (HF) – YOLO26m, dashcam, imgsz 512, 30 epochs (usage snippet has placeholder text).
- `TamAko783/YOLO26n_RDD_FRDC_Distilled_v2` (HF) – YOLO26n distilled from Co-DETR + RTMDet on RDD data (metrics not checked).
- `Samdutse/pothole-yolov8` – YOLOv8s, mAP50 0.816 / mAP50-95 0.518 (Smartathon set).
- `dronefreak/hrp4k-yolov8m` – YOLOv8m on HRP4K, single class.
- `vinothvikas1987/pothole-detection-yolov8` – YOLOv8s, 5 distress classes, pothole mAP50 0.782, ships best.pt + best.onnx, Apache 2.0.
- `ZiaPratama/Yolov8_Pothole` (no metrics). `savioratharv/pothole_detection` is YOLOS (transformer), not YOLO.
- Base weights: `Ultralytics/YOLO26` HF repo / github releases (yolo26{n,s,m,l,x}[-seg].pt); RF-DETR weights auto-download via `rfdetr`.
- Load with `hf_hub_download(repo_id, "best.pt")` then `YOLO(path)`.

## 4. Toolchain
- **ultralytics 8.4.211** (PyPI, released 2026-11-11), requires-python ≥3.8, classifiers list 3.8–3.15; Python 3.14 support added in 8.4.172 (release notes; the export Docker image stays on 3.13 for deps). 8.4.143 added INT8 QAT for YOLO26. Export formats: `onnx`, `engine` (TensorRT, FP16/INT8, `device=dla:0` on Jetson), `openvino`, `ncnn` (FP16 only), `litert` (TFLite successor, INT8), `executorch`, `coreml`, `rknn`, `qnn`, `hailo`, `imx`, `edgetpu`, `mnn`, etc. `half=True/int8=True` deprecated → `quantize=16/8`.
- **PyTorch**: stable 2.12.0 (GA 13 May 2026) supports Python 3.10–3.14; default PyPI wheels on Linux are CUDA 13.0 since 2.11. Install: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130` (works on CUDA 13.2 driver; 2.12 release notes mention CUDA 13.2 builds; cu128 deprecated). 2.13 RC exists (test channel) — UNCERTAIN whether GA yet; drops 3.13t. RF-DETR needs Python ≥3.10 (3.14 untested — UNCERTAIN; use a 3.12 venv if wheels like `onnxruntime`/`tensorrt` lag on 3.14).
- **Jetson**: JetPack 7.2 (Orin/Thor) → `pip install torch torchvision --index-url .../cu130`; JetPack 6.1 → Ultralytics-hosted torch 2.10.0 / torchvision 0.25.0 ARM64 wheels + cuDSS 0.7.1. Use TensorRT 10.x (TRT 11 unsupported on Jetson and no DLA). Engines are device-specific; rebuild per target. DeepStream INT8 known broken with TRT 10.x.
- **Android**: export `litert` (INT8) or `ncnn`; YOLO26 NMS-free head removes NMS post-processing on mobile. TFLite = LiteRT.

## Uncertain / to verify
BharatPotHole Kaggle licence and image count; RF-DETR on Python 3.14; PyTorch 2.13 GA status; DA3 metric scaling convention; `vnr-project` Roboflow set's Hyderabad origin; YOLO26-seg accuracy on Indian potholes (no published numbers — must benchmark).


## RESEARCH: irc-repair
# Research report: Indian standards, materials & Telangana context for permanent pothole repair

## 1. IRC / MoRTH standards (verified from official PDF text, law.resource.org)

**IRC:82-2015 "Code of Practice for Maintenance of Bituminous Road Surfaces"** (https://law.resource.org/pub/in/bis/irc/irc.gov.in.082.2015.pdf) — the 2023 edition (IRC:82-2023, listed in IRC 2026 catalogue https://www.irc.nic.in/WriteReadData/LINKS/Updated%20Catalogue,%202026(1)ed1ad28e-67a2-4bfe-82c1-4c3b75fa9107.pdf) is paywalled; assume its pothole clauses are substantially the same (UNCERTAIN).
- Cl. 4.1: pothole filling is **Routine Maintenance** ("almost round the year").
- **Cl. 7.5.3.4 Severity (the official grading to encode):** Small = **25 mm deep, 200 mm wide**; Medium = **25–50 mm deep, 500 mm wide**; Large = **>50 mm deep, >500 mm wide**. Pothole = "bowl shaped cavity … caused by localized disintegration" (7.5.3.1). Cause: water ingress / loss of adhesion (7.5.3.3).
- Cl. 7.5.3.5 Treatment options: (1) hot/cold mixes for immediate use, (2) storable cold mixes (cutback IRC:116 / emulsion IRC:SP:100), (3) readymade mixes, (4) cold mixes by patching machines.
- **Table 5.3 Urban road rating:** Potholes % area >0.5 = Poor (rating 1); >0 and ≤0.5 = Fair; NIL = Good. (Highways Table 5.1: >1 / 0.1–1 / <0.1 %.) Rut depth >10 mm = poor. Use this for stretch-level PCI.
- Section 10 (cold-mix patching) + Appendix: pothole "need not be dry; excess water shall be removed"; apply prime/tack coat on WMM; mix for **≤75 mm depth**; deeper → **layers of 25–75 mm**, deep part may be filled with Crusher Run Macadam; compact periphery first; finished patch **~10 mm proud** (hand rammer) or **~3 mm proud** (roller); sprinkle sand before traffic. Readymade mixes: **4.5–5.5 % residual bitumen**, storable **6 months**.
- Cl. 11.4 lists mechanised options: machine-mixed spot cold mix, Mobile Mechanised Maintenance Units, **jet patching / velocity spray injection**, **infrared** heating. "Deteriorated portion adjoining the pothole should be cut" (square cutting) before patching.
- No response-time (hours/days) targets in IRC:82-2015; SLAs come from local charters (see §4).

**IRC:SP:100-2014 Cold Mix Technology (Bitumen Emulsion)** (https://law.resource.org/pub/in/bis/irc/irc.gov.in.sp.100.2014.pdf)
- Cold mixes "suitable for low to moderate traffic… not recommended for heavily trafficked roads and high temperature" **BUT "bitumen emulsion based patching mixes for repairing potholes are recommended for all traffic and climatic conditions."**
- Sec 9: hot mix preferred for patches but "not available throughout the year in … high rainfall" → use cold maintenance mixes: (1) immediate-use, (2) storable/stockpile (SS-2 or tailor-made emulsion; covered; usable 6 months), (3) RAP-based, (4) readymade.
- Table 9.1 grading: 9.5 mm 100 %, 4.75 mm 40–100, 2.36 mm 10–40, 1.18 mm 0–10, 0.075 mm 0–2. **Emulsion 7–8 % by wt of aggregate**; MS or SS-2 grade (IS:8887).
- Cl. 9.7 pothole repair procedure identical to IRC:82 (≤75 mm/layer, CRM for deep, 10 mm proud). Field density for CMBM ≥95 % of lab max (Cl. for 700 m² lots).
- Cl. 2645: cold mix base **shall not be laid during rain or on wet granular base** — i.e. surface courses no, pothole patching yes in damp (not flooded) holes.

**IRC:116-2014 Readymade Bituminous Pothole Patching Mix (Cut-back MC-800)** (https://law.resource.org/pub/in/bis/irc/irc.gov.in.116.2014.pdf)
- Supplied in **50 kg plastic-lined bags** (default), stockable ≥6 months, workable at -7 °C; "suitable for patching potholes only, not long stretches".
- Potholes **up to 75 mm deep**; deeper → 75 mm layers; ≥50 mm patching mix at top; angular 25 mm aggregate + MC-30 prime for deep fill.
- Table 3 min residual bitumen: 4.5 % (abs <1 %), 5.0 % (1.1–1.5 %), 5.5 % (1.6–2.0 %). QC tests: wet coating, static immersion, workability, bond.

**MoRTH Specifications 5th Rev. (2013), Section 3000 Maintenance**: clause 3004 is the pothole/patch-repair item (index confirmed at https://www.scribd.com/doc/208520558/Section-3000-Maintenance-of-Road; body text not retrieved — UNCERTAIN). MoRTH Standard Data Book 2019 has a "bituminous cold mix" rate-analysis item (https://law.resource.org/pub/in/bis/irc/mort.sdb.02.2019.pdf). IRC:SP:20 (rural roads manual) only references pothole filling generally.

## 2. Materials & prices (INR)

- **Ready-to-use cold mix bags (IndiaMART/trade, 25 kg):** ₹220–₹800/bag; typical **₹275–₹400/bag** (e.g. CRETOBIT Patch Mix ₹275, "Instant Patch" Cat's Whiskers ₹400, Novaphalt ₹500) → **₹11–16/kg** (https://m.indiamart.com/impcat/road-patching-materials.html). Hindustan Colas (HINCOL "RoadBond Quick"/"Road Block") quoted ₹12–13/kg, ~100 kg per m² (≈50 mm), ≈₹1,000/m² vs hot mix ₹400–500/m² (Construction World 2014, https://www.constructionworld.in/latest-construction-technology/Say-NO-To-Potholes/11831). Brands "Instamix/Elastopatch/Roadfix/Patchmix" not found as pothole products — UNCERTAIN; use HINCOL RoadBond, Bitchem, Tiki Ezrepair, Wonder Patch instead. A 2026 Cyberabad MC (Serilingampally circle) tender bought cold-mix bags + emulsion for monsoon pothole work (~₹3.95 lakh, bids closed 23 Jun 2026) (https://tenders.infralens.in/tender/19-ee-cir-49-slpz-cmc-1-10-2026-709807).
- **Jet patcher / velocity patching:** GHMC bought jetpatchers in 2017 at ~₹1 cr each, 5 machines (~₹10 cr) (https://www.thehansindia.com/posts/index/Telangana/2017-06-21/Jetpatchers-to-fill-Hyderabad-potholes/307768); Deccan Chronicle 2021 (court filing): GHMC spent **₹7,396 per pothole vs ₹1,600 earmarked** (https://www.pressreader.com/india/deccan-chronicle/20210727/281517934152105). BBMP Bengaluru 2025: jet-patcher contract ₹2.5 cr → **₹1,942/m²** vs ₹1,200/m² conventional; ~10 min/pothole (https://www.deccanherald.com/india/karnataka/bengaluru/bbmp-upgrades-pothole-repair-with-new-rs-2-5-crore-machine-3045715). GHMC still tenders "pothole repair using jet patcher" per circle (https://tender18.com/tenders-details/16471012).
- **Infrared:** GHMC "Road Doctor" launched Aug 2014 (30 min/pothole, ~40/day, works in standing water, claimed 1-yr life, 12-hour fill promise via helpline 21111111) (https://news.fullhyderabad.com/hyderabad-news/new-technology-to-make-hyderabad-free-of-potholes-12466.html). Other cities: Vijayawada ₹1.48 cr, Ludhiana 2 machines ₹3.86 cr (underused), Kerala ₹40–45 lakh (2026), Chennai 10 machines ₹15 cr. Nagpur "Insta Road Patcher" ₹1.49 cr (2026).
- CRRI PatchFill machine ₹10 lakh; op cost "₹5,000 per 100 potholes (30 cm dia × 6 cm)" (2014). Durability claims: Road Block ≥5 yr, Wonder Patch 4–5 yr, Tiki Ezrepair 1 yr. Geopolymer/cementitious rapid-set: no Indian municipal pothole deployment found — UNCERTAIN.

## 3. Volume → material → cost formula (for the app)

- Compacted density of emulsion/cut-back cold mix: **~2.1–2.3 t/m³** (CIR field densities 1,880–2,310 kg/m³; lab CBEM 2,118–2,294 kg/m³; https://link.springer.com/article/10.1186/s42252-024-00061-7 and Nassar 2016). Use **2,200 kg/m³ compacted**, loose ~1,600–1,800 kg/m³. Hindustan Colas rule "100 kg/m² at ~50 mm" ⇒ 2,000 kg/m³ — consistent.
- Suggested formula: V_repair = A_cut × max(d_avg, 50 mm) where A_cut = square-cut area (pothole bbox + 100–150 mm margin, IRC:82 11.4); if d>75 mm, split into CRM sub-base (d−50 mm) + 50 mm patching mix (IRC:116 5.2). Mass = V × 2,200 × (1 + wastage 10–15 %). Bags = ceil(mass / 25 kg) (or 50 kg per IRC:116). Cost = bags × ₹300–400 + emulsion tack (0.25–0.3 kg/m² SS-1/CSS-1 @ ₹40–50/kg) + labour (₹600–800/day/mazdoor, 2–3 man-hours per pothole) + sand. Example: 0.5 m × 0.4 m × 60 mm → cut 0.7 × 0.6 × 0.06 = 0.025 m³ → 55 kg ×1.1 ≈ 61 kg ≈ 3 bags ≈ ₹1,000 material, ₹1,500–2,000 all-in (vs GHMC's reported ₹7,396). Mark all unit costs as configurable.

## 4. Telangana / Hyderabad context

- **GHMC road network: 9,013 km** (2,846 km BT + 6,167 km CC) per GHMC officials (https://telanganatoday.com/huge-boost-for-road-infrastructure-in-hyderabad); main roads ≥3-lane 709.49 km / 401 stretches (some cite 811.96 km). Hyderabad has highest road density in Telangana.
- **CRMP Phase 1:** ₹1,839 cr, 709.49 km, 5 years, 7 packages — BSCPL (LB Nagar 138.77 km ₹238.42 cr), NCC, MEIL, M Venkatarao Infra etc.; contract since Dec 2019 (https://www.siasat.com/hyderabad-crm-launched-promises-709-km-smooth-roads-1700772/). **Phase 2 (cabinet Mar 2026): ₹3,145 cr, 1,142.54 km (744.22 Ph-1 + 398.32 new, ≥2-lane), FDR for BT roads, 5-yr maintenance, incl. Cyberabad & Malkajgiri MCs (~1,045 km in other reports)** (https://hyderabadmail.com/hyderabad-crmp-phase2-ghmc-rs3145-crore-road-project/). Note GHMC was split (2026) into GHMC/Cyberabad MC/Malkajgiri MC — design multi-ULB tenancy.
- **HRDCL:** mandate ~600 km major corridors, ~300 km white-topping; ₹1,500 cr loan 2025; 24 link roads ₹487 cr (https://www.constructionworld.in/transport-infrastructure/highways-and-roads-infrastructure/hrdcl-revives-link-roads-project-worth-rs-4.87-billion/68292). State R&B: ₹19,300 cr HAM programme, 13,541 km, 15-yr maintenance (Jun 2026); monsoon 2025 damage estimate ₹1,157 cr; Jun 2025: 1,214 potholes on 302 km pending (R&B minister).
- **Pothole volumes/SLAs:** GHMC Citizen's Charter (Jun 2023): **potholes filled within 24 hours** (https://telanganatoday.com/now-potholes-in-hyderabad-to-be-filled-within-24-hours). Road-safety drive 2025: **20,882 identified / 19,380 repaired by 31 Oct 2025** (Secunderabad 4,602, LB Nagar 3,501, Khairatabad 3,329, Kukatpally 2,922, Charminar 2,832, Serilingampally 2,194) (https://telanganatoday.com/ghmc-fills-over-19000-potholes-across-hyderabad-as-part-of-road-safety-drive); later 22,797/21,511 by 28 Nov. 2026: 10-day monsoon drive 11–21 Aug across 150 wards; ~1,000 waterlogging hotspots; local press claims ~30,000 potholes. Earlier: "3,000 bags blacktop mix, 4,000 potholes"; single hot-mix plant at Chudi Bazar causing cold-arrival hot mix (quality issue → argument for cold mix).
- **Complaint channels:** MyGHMC app replaced by **MyCURE** (Mar 2026); GHMC WhatsApp **8125966586**; call centre 040-21111111; toll-free 155304 (3rd-party listed, UNCERTAIN) (https://telanganatoday.com/ghmcs-mycure-replaces-older-myghmc-platform-for-citizens-to-report-civic-issues). Ward offices (150) handle grievances.
- **TGSRTC fleet:** ~10,081 buses statewide, 97 depots; Greater Hyderabad zone ~2,800 buses, ~7.5 lakh km/day, 19–35 lakh pax/day (https://telanganatoday.com/rtc-to-add-over-500-buses-to-hyderabad-fleet); 500 low-floor e-buses completed 2026 (Olectra), ~545 e-buses in city, target 2,800 e-buses by 2027; VTS/GPS on ~4,200 buses (Gamyam app, 2023), plan to extend to all + GHMC digital bus-stop boards (https://telanganatoday.com/tsrtc-bus-tracking-system-to-cover-4200-vehicles). 2,800 buses × ~270 km/day = ideal crowdsensing fleet.
- **Monsoon:** SW monsoon Jun–Sep; Hyderabad normal ≈ 595–745 mm (GHMC ~621 mm), annual ~756–820 mm; **2025: 897 mm (+43.7 %), wettest in 20 yrs**; 2026: −34 % deficit (https://m.hyderabadmail.com/article/hyderabad-telangana-monsoon-rainfall-2025-record/32576). NE monsoon Oct–Nov adds damage.
- **Existing AI pilots:** T-AIM Mobility AI Grand Challenge (Capgemini, 2022) for GHMC "pothole severity by surface area and depth"; winner **RoadMetrics** (Bengaluru; ₹20 lakh pilot; smartphone dashcam, claims 90 % accuracy) — pilot outcome unpublished (https://telanganatoday.com/t-aim-launches-mobility-ai-grand-challenge, https://startuppedia.in/this-bengaluru-startup-is-set-to-offer-an-ai-solution-for-potholes-in-the-city-wins-rs-20-lakhs-to-be-used-for-execution). IIIT-H CVIT/INAI: pothole + violation detection from cameras on public vehicles, India Driving Dataset (IDD), iRASTE piloted in Telangana (https://www.deccanchronicle.com/southern-states/telangana/iiit-hyderabad-develops-low-cost-ai-tools-to-tackle-indias-real-world-road-risks-1886308). IIT Roorkee SAFETRIP: single-camera pothole volume/area patent (2026). Smartphone-LiDAR pothole volume: 5.35 % error, RMSE 6.5 mm (https://systems.enpress-publisher.com/index.php/jipd/article/download/6162/4175). SAE 2026-26-0269: polarized stereo vs LiDAR depth (Pune). Datasets: IDD (IIIT-H), RDD2022 India subset, iWatchRoad (arXiv 2508.10945).

## 5. Gaps / uncertain
IRC:82-2023 revisions, MoRTH 3004 text, GHMC current cold-mix procurement volumes, named brand prices (Instamix etc.), exact TSRTC city fleet 2026, RoadMetrics pilot results.

Local extracted IRC texts: /tmp/claude-1000/-home-coral-Desktop-Telangana-pathole-detection/659d7d73-e61a-4f1c-b12d-698aed1f77dc/scratchpad/irc/irc82.txt, .../sp100.txt, .../irc116.txt