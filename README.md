# RoadFix — Pothole Detection & Maintenance

Phones or bus cameras record the road → a pothole detector finds potholes → their **size** (width, length, area with
σ) is measured from the outline and the camera height → IRC:82 severity and priority → IRC:116 / SP:100 cold-mix
repair plan (bags as a low–high range, ₹) → work orders, SLA escalation, bus-verified repair and DLP tracking.
Built for the Telangana R&B / Municipal Administration challenge; India-only data and standards.

- **How it works: [`docs/WORKFLOW.md`](docs/WORKFLOW.md)** · where work stopped: [`docs/HANDOFF.md`](docs/HANDOFF.md)
- Decisions (newest at the end): [`docs/DECISIONS.md`](docs/DECISIONS.md) · plan: [`docs/PLAN.md`](docs/PLAN.md) ·
  contracts: [`docs/CONTRACTS.md`](docs/CONTRACTS.md) · latest benchmark: [`reports/BENCHMARK.md`](reports/BENCHMARK.md)
- Operations: [`docs/CALIBRATION_SOP.md`](docs/CALIBRATION_SOP.md), [`docs/DEPLOYMENT_TSRTC.md`](docs/DEPLOYMENT_TSRTC.md),
  [`docs/LICENCES.md`](docs/LICENCES.md), [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md), edge protocol
  [`edge/protocol.md`](edge/protocol.md)

## The model and the data (D38)
| What | Where | Notes |
|---|---|---|
| **Base model** (in use) | `models/base_model/civic_yolo11n.pt` (+ `.onnx`, `_int8.tflite`) | the supplied Nagar Netra *civic_services* YOLO11n: 16 classes, **pothole = class 1**; boxes, outlines from SAM2 |
| **Base dataset** (training) | `datasets/base_nagar_netra_pothole/` | the supplied pothole set (710 images), re-split **leak-free** by `make base-dataset` (597 / 57 / 56) |
| **Test sets** | `configs/datasets.yaml` | Nagar Netra test · RDD-India locked 514 · BharatPotHole · web photos · eYantra roads without potholes (false alarms) · a real phone street walk |
| **Model registry** | `configs/models.yaml` | base, adopted fine-tunes, archived models (earlier trained / downloaded) |
| **Model in use** | `.env` `SEG_WEIGHTS` | changed only by `make train` when its candidate passes the benchmark rule, or by hand |

```bash
make base-dataset   # build datasets/base_nagar_netra_pothole from "datasets/supplied_original/..."
make benchmark      # every model on every test set -> reports/BENCHMARK.md (+ benchmark.json)
make train          # fine-tune the base on the base dataset, benchmark, adopt ONLY if not worse on any test set
make combined-dataset && make train DATA=combined EPOCHS=60   # D39: + the earlier train/val (test parts never used)
make dashboard      # watch training live (http://127.0.0.1:8765)
make negatives-dataset && make train DATA=base_neg   # D42: base + earlier images WITHOUT potholes (clean roads)
make compare-video TRIP=<id> VIDEO=<mp4>          # D42: phone keyframes vs a video of the same walk
make annotate       # D41: draw the potholes yourself (http://<pc>:8775 or <capture link>/annotate); make annotate-export
```

## Quickstart (Linux)
Python **3.11** in `.venv` (never reinstall torch casually), Node 22 for the web app, an NVIDIA GPU for training.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
# RTX 50xx (sm_120) needs the CUDA 12.8 build of torch:
.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-lock.txt
make setup          # checks the venv, writes LOCKFILE.md
make test           # CPU tests, ~1 min
cd web && npm ci    # once, for the web app
```

## Run it end to end (each in its own terminal)
```bash
make api              # RoadFix API          http://localhost:8000
make web              # web app              http://localhost:5173
make capture-public   # phone page + Drives dashboard + automatic processing (prints the link + access code)
```
On a phone: open the printed link, sign in, tap **Start recording**, drive (or tick *Testing on foot*), tap Stop.
Frames upload by themselves; the PC detects potholes, measures them, places them on the map and sends finished drives
to the API. Results: `<link>/dashboard`; confirmed potholes, repair plans and work orders: the web app. Other tools:
`make playground` (test a model on your own photos/videos), `make review` (review the base dataset), `make control`
(what is left to do). `make help` lists every target by section.

Measurements are size-only (depth model off, D27); every calibration in the repo is `certified=false`, so nothing is
billing-grade. The simulated end-to-end demo (`make demo`, ~12 min, logins with password `demo1234`: `admin`,
`zonal_khairatabad`, `engineer1`, `contractor1`, `contractor2`, `citizen1`) still works for presentations.

## Project layout
Every folder name says what is in it (D43). Data and models are not in git; each data folder has a README.

| Path | What |
|---|---|
| `datasets/` | **all training data**: `base_nagar_netra_pothole/` (the base dataset, train + val), `finetune_*` (fine-tuning sets built from it), `earlier_india_pothole_v2/` (earlier train + val), `archive/` (v1 and all-class builds), `supplied_original/` (the dataset as delivered), `downloads/` (raw downloads, untrusted), `extracted/` (unpacked downloads, OSM/GTFS road data), `splits/` (split lists, in git), `labelling/` (review and annotation decisions) |
| `test_datasets/` | **all test data, never trained on**: `nagar_netra_test/`, `rdd_india_locked/`, `bharatpothole_dashcam/`, `web_pothole_photos/`, `eyantra_clean_roads/`, `phone_street_20261118/`, `phone_videos/`, `video_compare/`, `staged_measurement_truth/` |
| `models/` | `base_model/` (the model in use), `adopted_finetunes/`, `downloaded_pretrained/` (archive), `supplied_original/` (as delivered), `yolo_starting_points/` (COCO YOLO26-seg), `training_runs/` (`detect/` fine-tunes, `seg/` archive, `logs/`), `exported/`, `registry/` |
| `app_data/` | what the running app writes: `gunthameter.db`, `store/` (frames), `capture/` (phone drives), `playground/` (model test uploads), `demo/`, `fleet_sim/`, `backups/`, `tiles/` |
| `dataset_tools/` | code that builds datasets: `build_base_dataset.py`, `build_combined_dataset.py`, `test_split.py` (moves test splits to `test_datasets/`), archive builders (`download.py`, `build_dataset.py`, `derive_dataset.py`, SAM2 outlines) |
| `tools/` | web tools and scripts: phone capture (`phone_capture.py`, `capture_worker.py`, `capture_replay.py`), model test page, annotation, dataset review, project control, training dashboard, demo, reports |
| `ml/` | `benchmark.py` (all models × all test sets), `train_det.py` (fine-tune + adoption rule), detector interface (`segmenter.py`), metrics; archive: the YOLO26-seg pipeline (`pipeline.py`, `train_seg.py`, `eval.py`, `export.py`) |
| `measure/`, `repair/`, `priority/`, `geo/`, `gis/` | size measurement (ground plane), IRC severity; repair planner; priority score; map-matching and clustering; map layers and tiles |
| `server/`, `web/`, `edge/` | FastAPI app and pipeline stages; React PWA; bus protocol, simulator, Android / Jetson skeletons |
| `configs/` | `datasets.yaml`, `models.yaml`, `roadmap.yaml`, frozen contracts (classes, IRC rules, priority, rate card, cameras) |
| `reports/`, `docs/`, `tests/` | benchmark, training and demo reports; design, decisions, handoff; tests (`make test`) |

## Archive
The earlier YOLO26 segmentation pipeline (D15–D24) is kept and runnable under `make seg-*` targets; its datasets
(`datasets/earlier_india_pothole_v2`) are now the source of the RDD-India, BharatPotHole, web and eYantra test sets.
Earlier trained and downloaded models are listed in `configs/models.yaml` with role `archive`. Nothing was deleted.
