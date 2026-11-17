# RoadFix — Pothole Detection & Maintenance

Bus-camera pothole mapping → metric **depth / area / volume** (with σ) → IRC:82 severity and priority →
IRC:116 / SP:100 cold-mix repair plan (bags as a low–high range, ₹) → work orders, SLA escalation, bus-verified repair
and DLP tracking. Built for the Telangana R&B / Municipal Administration challenge; India-only data and standards.

- **How it works (workflow, models, where each runs): [`docs/WORKFLOW.md`](docs/WORKFLOW.md)**
- Plan: [`docs/PLAN.md`](docs/PLAN.md) · decisions/deviations: [`docs/DECISIONS.md`](docs/DECISIONS.md) ·
  contracts: [`docs/CONTRACTS.md`](docs/CONTRACTS.md) · where work stopped: [`docs/HANDOFF.md`](docs/HANDOFF.md)
- Method and validation: [`docs/DEPTH_METHOD.md`](docs/DEPTH_METHOD.md), [`reports/REPORT.md`](reports/REPORT.md)
- Operations: [`docs/CALIBRATION_SOP.md`](docs/CALIBRATION_SOP.md) (te/en),
  [`docs/DEPLOYMENT_TSRTC.md`](docs/DEPLOYMENT_TSRTC.md), [`docs/LICENCES.md`](docs/LICENCES.md),
  [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md), edge protocol [`edge/protocol.md`](edge/protocol.md)

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
```

| Step | Command |
|---|---|
| Measurement demo, no services | `make measure-demo` |
| Seed the demo DB (once) | `make seed` |
| Web app dependencies (once) | `cd web && npm ci` |
| **End-to-end demo** (~12 min) | `make demo` |
| Web tests / production build | `cd web && npm test` / `npm run build` |

`demo` copies the seeded DB to `data/demo/demo.db` (the seeded DB is never modified), starts the API on
http://localhost:8000 (docs at `/docs`) and the web app on http://localhost:5173, drives 3 simulated TGSRTC buses over
routes 10H / 218 / 5K for 2 days (survey → repair day → bus verification → DLP check), issues tomorrow's work orders
and repair runs, and prints a summary. Logins (password `demo1234`): `admin`, `zonal_khairatabad`, `engineer1`,
`contractor1`, `contractor2`, `citizen1`. `demo --exit` stops after the summary; `demo --help` lists the options.

Demo depths come from a synthetic road render (method `synthetic`); every calibration in the repo is
`certified=false`. Nothing the demo shows is a field measurement or billing-grade.

## Training the pothole segmenter
One command, everything in the foreground (DECISIONS D16–D18):

```bash
make retrain-smoke   # ~2 min trial: 1 epoch on 15 % of train, shows batch, GPU memory and load
make retrain         # check -> data -> baselines -> train -> eval -> export -> report (YOLO26m-seg, <= ~4.5 h)
make train-gpu       # the same through scripts/train_gpu.sh: lists programs holding GPU memory first (close them
                     # for a bigger batch), keeps a log in runs/logs/; options are listed at the top of the script
```

- **GPU at 95 %:** the batch is sized by Ultralytics AutoBatch to 95 % of the GPU memory that is *free* when training
  starts (the desktop and other users are subtracted first). Measured on the RTX 5080: YOLO26m-seg batch 18, 96.5 %
  of the card's memory, median GPU load 91 % while training. On CUDA out-of-memory the run resumes from its last
  checkpoint with a 10 % smaller batch. Profile: `configs/model/train_profiles.yaml: rtx5080_gpu95`
  (`PROFILE=rtx5080_gpu95_s` for the faster small model).
- **Data:** `data/datasets/india_pothole_seg4_v2` (8,911 train / 1,722 val / 1,843 test images; 14,564 training
  potholes, 70 % of them on Indian roads): RDD2022 India + foreign potholes, eYantra manholes, BharatPotHole
  (CC BY-SA 4.0, split by drive) and human-drawn pothole masks from the web (Apache-2.0); 1,034 training images
  carry no labels (plain roads, at most 15 %). The 514 RDD-India test images are locked
  (`data/splits/rdd_india_test.lock.txt`), so every score is comparable with the 0.221 laptop model.
- **Outputs:** `runs/seg/<name>/weights/best_pothole.pt` (epoch with the best validation pothole AP50),
  `reports/model_eval.json` (RDD-India acceptance score + BharatPotHole / human-mask subsets),
  `weights/export/` + `weights/registry/` (ONNX), `reports/train/<name>_gpu.json` (GPU memory and load).
- Re-running `make retrain` resumes an interrupted run. Other pieces: `make train-check`, `make baselines`
  (already-trained pothole models, incl. one from Hugging Face, scored on the same test), `make dataset` +
  `make derive-dataset` (rebuild v2), `make prune-v1` (deletes the v1 data once a v2 model beats it).

## Project layout
| Path | What |
|---|---|
| `measure/` | measurement core: depth backends, road surface fit, mask integral, σ, fusion, IMU, IRC severity |
| `repair/`, `priority/`, `geo/` | repair planner + Measurement Book export; priority score; map-matching, clustering, passes, jurisdiction |
| `server/` | FastAPI app (`server/app`), pipeline stages and ops ticks (`server/workers`), tests |
| `ml/` | segmenter interface, training (`train_seg.py`), the training pipeline (`pipeline.py`), eval, export, augmentations |
| `data/` | dataset code (`download.py`, `build_dataset.py`, `derive_dataset.py`, SAM2 masks) and the data itself: `raw/` (downloads, untrusted), `interim/` (extracted sources), `datasets/` (training sets), `splits/` (persisted splits + test lock), `demo/`, `store/` |
| `configs/` | frozen contracts (classes, IRC rules, priority weights, rate card, cameras) and `model/` (dataset yamls, training profiles) |
| `edge/` | wire protocol + JSON schemas, Python bus simulator, Jetson runner, Android skeleton |
| `web/` | React PWA (te/hi/en, offline field mode) |
| `gis/` | GIS layers and seeds |
| `weights/` | base and bootstrap checkpoints; `pretrained/` (baselines), `export/`, `registry/` |
| `runs/`, `reports/` | training runs; evaluation, validation and demo reports |
| `scripts/` | demo orchestrator, seeding, staged-truth protocol, report generator, lockfile |
| `docs/`, `tests/` | design, decisions, contracts, SOPs; cross-package tests |

## Other targets
| Target | What |
|---|---|
| `api`, `worker`, `ops-tick` | API with the configured depth model; DB job poller; ops ticks once (SLA, verification, DLP, indent, KPI) |
| `sim-api`, `fleet` | the simulator API on port 8002, and the fleet against it (`fleet --help`) |
| `train`, `train-resume`, `eval`, `export`, `pseudo-label` | single training steps (`NAME=`, `PROFILE=`); pseudo-labels for review |
| `data` | verify sources (the large raw archives were deleted after extraction, D18; `--fetch` downloads again) |
| `truth`, `report` | staged foam-board accuracy protocol; regenerate `reports/REPORT.md` |
| `lint` | ruff |

## Status
Software complete through WP12 (see `docs/HANDOFF.md`). The segmenter retraining pipeline is ready and tested on this
machine; the full run (`make retrain`) is the next step. Still open, and not code: the real-camera staged accuracy
table, certified depot calibration, the Android app beyond its skeleton (`edge/android`), and the licence decisions in
`docs/LICENCES.md`.
