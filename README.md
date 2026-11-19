# RoadFix — Pothole Detection & Maintenance

RoadFix finds potholes in photos from phones or bus cameras, outlines each one, measures its **size** (width, length,
area in m², each with σ) from the outline and the camera height, grades it by IRC:82, ranks it by priority, plans the
IRC:116 / SP:100 cold-mix repair (bags as a low–high range, cost in ₹), and tracks the work order until a later pass
shows the repair held. Built for Telangana roads; India-only data and standards.

- How it works: [`docs/WORKFLOW.md`](docs/WORKFLOW.md) · where work stopped: [`docs/HANDOFF.md`](docs/HANDOFF.md)
- Decisions (newest at the end, D1–D50): [`docs/DECISIONS.md`](docs/DECISIONS.md) · plan: [`docs/PLAN.md`](docs/PLAN.md)
  · frozen interfaces: [`docs/CONTRACTS.md`](docs/CONTRACTS.md) · latest benchmark: [`reports/BENCHMARK.md`](reports/BENCHMARK.md)
- Operations: [`docs/CALIBRATION_SOP.md`](docs/CALIBRATION_SOP.md), [`docs/DEPLOYMENT_TSRTC.md`](docs/DEPLOYMENT_TSRTC.md),
  [`docs/LICENCES.md`](docs/LICENCES.md), [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md), edge protocol
  [`edge/protocol.md`](edge/protocol.md)

Measurements are size-only (depth off, D27). Every calibration in the repo is `certified=false`, so no number is
billing-grade, and bags are always a range, never an exact count. Times are stored in UTC and shown in IST.

## Quickstart (Linux)
Python **3.11** in `.venv` (never reinstall torch casually), Node 22 for the web app, an NVIDIA GPU for training.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
# RTX 50xx (sm_120) needs the CUDA 12.8 build of torch:
.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-lock.txt
make setup          # checks the venv, writes LOCKFILE.md
make test           # CPU tests
cd web && npm ci    # once, for the web app
```
Model weights and data are not in git (`.gitignore`, D43): copy `models/base_model/pothole_seg11n.pt` and the
`datasets/` / `test_datasets/` folders from the project machine, or rebuild them (see "Training and benchmark").

## Try it end to end with dummy data
```bash
make e2e-demo        # dummy phone drive -> model A -> size -> API (demo DB copy) -> plans, work orders, KPIs
make e2e-dashboard   # pipeline dashboard, http://<this-pc>:8795 (phone too, same Wi-Fi)
```
`make e2e-demo` builds a dummy drive along Banjara Hills Road No. 10 from public test photos (Kaggle web and Nagar
Netra pothole photos about 40 m apart, eYantra clean roads in between), runs it through the real capture processing
with model A, sends it to an API running on a copy of the demo database (`app_data/demo/`; the main database is not
touched) and lets that API create potholes, repair plans, work orders, a repair run and KPIs. The dashboard shows every
stage: a funnel from frames to work orders, the route and potholes on a map, and a card per pothole (photo with
outline, size, severity, plan, work order). **Dummy data: the photos were not taken with the assumed camera, so the
sizes, bags and costs show the flow, not field measurements.**

## Test on your real data
```bash
make api              # RoadFix API on :8000 (each command in its own terminal)
make capture-public   # phone page + automatic processing; prints a link and an access code
make e2e-dashboard    # open the real drive in the pipeline dashboard
```
On the phone: open the printed link, sign in, mount the phone facing the road, tap **Start recording**, drive (or tick
*Testing on foot*), tap Stop. Keyframes upload by themselves; the PC detects, outlines and measures potholes and sends
the drive to the API. Drives also appear at `<link>/dashboard`; confirmed potholes, plans and work orders in the web
app (`make web`, :5173). Restart `make api` / `make capture-public` after the model changes: they load it at start.

## The model and the data
| What | Where | Notes |
|---|---|---|
| **Model in use: A** | `models/base_model/pothole_seg11n.pt` (+ `.onnx`) | YOLO11n-seg: box **and** outline in one pass (no SAM2 step); one class, pothole; `.env SEG_WEIGHTS` points to it (D50) |
| How A was made | `make merged-dataset`, `make train-merged` | backbone from the supplied model, trained on the merged dataset with hard height / lens / night augmentation (`ml/augment/hard.py`, D48) |
| Supplied original | `models/supplied_original/` | the Nagar Netra civic YOLO11n exactly as delivered (16 classes, pothole = 1); starting point for training |
| Training data | `datasets/merged_pothole_seg` | 11,877 train / 1,216 val images, potholes only, leak-free split by near-duplicate groups (D48); sources in `datasets/README.md` |
| Test data | `test_datasets/`, listed in `configs/datasets.yaml` | 7 test sets + 10 generated night / zoom check sets; never used for training |
| Model registry | `configs/models.yaml` | only model A (all other pothole models were deleted, D47 / D50; their scores stay in `reports/`) |

Model A on the benchmark (`reports/BENCHMARK.md`, confidence 0.25): mean box AP50 0.529 over 17 test sets; Nagar
Netra test 0.649, RDD-India locked 0.334, Kaggle web 0.781, eYantra clean-road false alarms 0.2 %, night versions
0.66–0.78 on the Nagar Netra and web sets, phone outlines 0.274. Known gap: on the phone street walk it finds a
pothole in 11 of 26 pothole photos.

## Dashboards and tools
| Tool | Command | Address | For |
|---|---|---|---|
| Pipeline dashboard | `make e2e-dashboard` | :8795 | one drive (demo or real) through every stage |
| Phone capture + Drives | `make capture` / `make capture-public` | https :8443 or the ngrok link; `/dashboard`, `/annotate` | phone as the bus camera, automatic processing |
| Annotation + failure review | `make annotate` | :8775 (`ANNOTATE_PORT=` to change) | draw potholes; correct test labels where a model fails (D41, D49) |
| Test results | `make results` (`BUILD=0` = serve only) | :8785 | every test photo with the model's boxes and outlines, strict or lenient matching (D46) |
| Training dashboard | `make dashboard` | :8765 | training progress, live |
| Project control | `make control` | :8790 | what is left to do |
| Model test page | `make playground` | :8780 | try the model on your own photos and videos |
| API / web app | `make api` / `make web` | :8000 / :5173 | operations: potholes, plans, work orders, KPIs |
| Simulated fleet demo | `make demo` | API :8000, web :5173 | the WP12 demo with simulated buses |

The LAN pages work on a phone on the same Wi-Fi at `http://<this-pc-ip>:<port>`; the ones showing street photos ask
for the capture access code. `make help` lists every target by section.

## Training and benchmark
```bash
make merged-dataset      # rebuild datasets/merged_pothole_seg from the base and earlier sets (leak-free)
make check-sets          # rebuild the night / zoom check sets (after correcting test labels)
make train-merged ONLY=A # retrain model A (~1.6 h on the RTX 5080, PC stays usable); adopts only if not worse anywhere
make benchmark           # the registry + the model in use on every test set -> reports/BENCHMARK.md
make results             # refresh the test-results dashboard
make annotate-export     # write your corrections / outlines into the test labels
```
`make train` (box-detector fine-tune of the supplied model, `DATA=base|combined|base_neg|phone`) still works.
Candidate B of `make train-merged` needs `yolo11s-seg.pt` downloaded again (deleted in D50).

## Project layout
| Path | What |
|---|---|
| `ml/` | benchmark, training (`train_merged.py`, `train_det.py`), augmentation (`augment/`), detector interface, metrics; archive segmentation pipeline |
| `dataset_tools/` | dataset builders (base, merged, check sets, earlier sets) |
| `tools/` | web tools: phone capture, worker, dashboards, annotation, test results, demos |
| `measure/`, `repair/`, `priority/`, `geo/`, `gis/` | size measurement, IRC severity, repair planner, priority, map-matching, admin layers |
| `server/`, `web/`, `edge/` | FastAPI app and pipeline stages; React PWA; bus protocol, simulator, Android / Jetson skeletons |
| `configs/` | datasets, models, roadmap, frozen contracts (classes, IRC rules, priority, rate card, cameras) |
| `datasets/`, `test_datasets/`, `models/`, `app_data/` | data, weights and app output; not in git, each has a README |
| `reports/` | benchmark, training, test-result and demo reports |

## Archive
The YOLO26 segmentation pipeline (D15–D24) is kept under `make seg-*`; its starting weights were deleted (D47), so it
needs them downloaded before it can train again. The D39–D45 experiments (combined / clean-roads fine-tunes, the
Kaggle notebook, `make kaggle-vs-ours`) are recorded in `docs/DECISIONS.md` and `reports/`; their models were deleted.
