# Licences: datasets, models, software

GunthaMeter is meant for government deployment (GHMC / R&B / TGSRTC), so every licence counts: **non-commercial
(NC) material cannot go into the deployed system**, **copyleft (GPL/AGPL) material needs a decision before
procurement**, and every attribution must reach the shipped product. Status words: **OK** (permissive, attribution
only), **CAVEAT** (usable with an obligation or a decision), **PROTOTYPE ONLY** (not in a deployed build without a
different licence), **UNVERIFIED** (licence not confirmed; excluded by default).

This table records what the project used and the licence noted at download time (`data/download.py`,
`docs/RESEARCH.md`). It is not legal advice; the department's legal cell confirms each row against the source page
before the pilot.

## 1. Datasets
| Dataset | Used for | Licence | Status / handling |
|---|---|---|---|
| RDD2022 India (CRDDC2022; HF re-split mirror `akiii1234/RoadDamageDetection`) | detector training + test split (boxes; SAM2 pseudo-masks) | CC BY-SA 4.0 | **CAVEAT**: attribution; derived *datasets* (our masks, the unified split) must stay CC BY-SA if redistributed. Trained weights are generally treated as not a derivative dataset, but get this confirmed |
| eYantra "Semantic Segmentation Datasets of Indian Roads" (IIT Bombay, Kaggle) | the only Indian pothole **mask** set: training/eval masks; D13 "shallow" → manhole | GPL-2.0 (per Kaggle page) | **PROTOTYPE ONLY** for training data in a deployed model until the licence question (GPL on data/weights) is settled; eval use is fine |
| BharatPotHole (Kaggle `surbhisaswatimohanty/bharatpothole`) | simulator frames; training only with `--include-bharat` | **UNVERIFIED** (bundled `data.yaml` claims Roboflow CC BY 4.0) | excluded from `data/datasets` by default (`LICENCE_NOTE.json`) |
| TGSRTC GTFS (Mobility Database mdb-3361) | bus routes / shapes for the simulator and pass counting | open data (as listed by the Mobility Database) | **OK**: attribute TGSRTC / Telangana open data; confirm the publisher's terms |
| OpenStreetMap (Geofabrik extract), Overpass POIs | road segments, owners via `ref`/`operator`, POIs, lit tag | ODbL 1.0 | **CAVEAT**: "© OpenStreetMap contributors" on every map; a *public* derived database (our road table) must be offered under ODbL |
| OpenFreeMap "liberty" basemap style (online, D7) | web basemap | OSM data ODbL; style/tiles per OpenFreeMap terms | **OK** with attribution; production should self-host PMTiles |
| GHMC zones / circles / wards (`gis/layers`) | admin units | built from OSM boundaries + seeds; marked `official: false` | **OK** (ODbL via OSM); replace with official GHMC boundaries when issued |
| chitholian annotated potholes, Roboflow seg sets, IDD | considered in PLAN, **not used** | ODbL / CC BY 4.0 / IDD research licence | not in the build |

## 2. Models and weights
| Model | Used for | Licence | Status |
|---|---|---|---|
| Ultralytics YOLO26 (`weights/yolo26n-seg.pt`, `yolo26s-seg.pt`) and the `ultralytics` library | segmenter base, training, export | **AGPL-3.0** (Ultralytics offers an Enterprise licence) | **CAVEAT**: a networked service built on AGPL code must offer its source to users, or buy the Enterprise licence. Budget line in the pilot; the `Segmenter` interface (`ml/segmenter.py`) allows an Apache-licensed RF-DETR-Seg swap (D5) |
| Bootstrap detector `bootstrap_pothole_yolo26m.pt` (from `mfranzon/pothole-yolo26`) | day-0 detector (D9) | **UNVERIFIED** model-card licence; YOLO26 base → AGPL obligations apply in any case | replace with our trained model before the pilot |
| Bootstrap/verifier `bootstrap_pothole_yolov8s.pt` (`vinothvikas1987/pothole-detection-yolov8`) | second-model verifier (D5) | Apache-2.0 (model card); trained with Ultralytics YOLOv8 | **CAVEAT** (the AGPL question of the Ultralytics runtime, as above) |
| Depth-Anything-V2 Metric Outdoor Base (`depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf`) | default depth backend (D4) | **verify on the model card**: DA-V2 metric checkpoints were noted as Apache-2.0 for the VKITTI-Large card, while the *relative* Base/Large checkpoints are CC-BY-NC-4.0 | **must be confirmed before the pilot**. If the Base metric card is NC, switch the default to DA3METRIC-LARGE (Apache-2.0) or Metric3D v2 (BSD-2-Clause); both are backends already (`measure/depth_backends/`) |
| Depth Anything 3 `DA3METRIC-LARGE` | optional depth backend | Apache-2.0 | **OK** (avoid `DA3NESTED-GIANT-LARGE`: CC-BY-NC-4.0) |
| Metric3D v2 (ONNX) | optional depth backend | BSD-2-Clause | **OK** |
| SAM 2.1 hiera-small (`facebook/sam2.1-hiera-small`) | box → mask pseudo-labels (dataset build, pseudo-labelling) | Apache-2.0 | **OK** |
| UniDepth, Apple Depth Pro | evaluated in research only | CC-BY-NC-4.0 / Apple research licence | **not used** (non-commercial) |
| YuNet face detector (optional privacy blur, `GM_FACE_MODEL`) | blur faces before upload | per OpenCV Zoo model card (verify) | optional |

## 3. Software (main dependencies)
| Component | Licence | Note |
|---|---|---|
| ultralytics | AGPL-3.0 | see §2 |
| PyTorch, torchvision | BSD-3-Clause | |
| transformers, huggingface_hub, safetensors, timm | Apache-2.0 | |
| onnxruntime | MIT | |
| OpenCV (opencv-python) | Apache-2.0 | |
| FastAPI, Starlette, Pydantic, SQLAlchemy, Alembic, uvicorn, httpx | MIT / BSD | |
| GeoAlchemy2, shapely, pyproj, geopandas, h3, osmnx, pyrosm | MIT / BSD / Apache-2.0 | |
| numpy, scipy, pandas, scikit-learn, matplotlib | BSD-style | |
| albumentations | verify the installed version's licence (older releases MIT; the successor project changed licence) | augmentations only |
| PostgreSQL / PostGIS (production) | PostgreSQL licence / GPL-2.0 (PostGIS) | GPL applies to PostGIS itself, not to applications that connect to it |
| Web: React, react-router, @tanstack/react-query, zustand, i18next, recharts, Vite, Workbox | MIT | |
| Web: maplibre-gl | BSD-3-Clause | |
| Web: dexie | Apache-2.0 | |
| Noto Sans / Noto Sans Telugu / Devanagari (@fontsource) | SIL Open Font Licence 1.1 | |
| Android skeleton: AndroidX, CameraX, WorkManager, OkHttp, kotlinx-coroutines | Apache-2.0 | |
| Android: Google Play services location | Google APIs terms | a de-Googled build would use the platform `LocationManager` |

## 4. Decisions needed before the pilot (owner: pilot lead + legal cell)
1. **Ultralytics AGPL:** buy the Enterprise licence for the deployed segmenter, *or* publish the service source under
   AGPL, *or* switch to RF-DETR-Seg (Apache-2.0) via the `Segmenter` interface. GunthaMeter's own code can be
   released openly either way.
2. **Depth model:** confirm the DA-V2 Metric Outdoor Base licence on its card; otherwise make DA3METRIC-LARGE the
   default (one config line: `configs/measure.yaml` `depth.backend`).
3. **eYantra GPL-2.0 data:** keep it out of the training set of the deployed model, or get written permission; RDD +
   pilot-collected frames (owned by the department) then carry the model.
4. **BharatPotHole:** use only after its licence is verified.
5. **Attribution page** in the web app: OSM ©, RDD2022 citation, TGSRTC GTFS, model credits.
