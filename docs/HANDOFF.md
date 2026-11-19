# HANDOFF — RoadFix on the Linux RTX 5080 box (updated 2026-11-19)

This file is the single source of truth for "where did we stop". Claude Code memory does NOT travel between
machines; this file and `CLAUDE.md` replace it. Read section 0 first, then `docs/DECISIONS.md` **D27-D50** (size-only
measurement, phone capture, model choice, folders, merged training, failure review, model A in use). Everything below
section 0 is the earlier history.

## 0. Current state (D50, 2026-11-19)

**Model in use: model A** `models/base_model/pothole_seg11n.pt` (+ `.onnx`): YOLO11n-seg, box and outline in one pass,
one class (pothole), `.env SEG_WEIGHTS`. Trained by `make train-merged ONLY=A` (backbone from the supplied model
`models/supplied_original/weights/best.pt`) on `datasets/merged_pothole_seg` (`make merged-dataset`, 11,877 / 1,216
images) with hard height / lens / night augmentation (`ml/augment/hard.py`). All other pothole models were deleted
(D47, D50); their scores stay in `reports/` and DECISIONS. SAM2.1 stays (outlines for box-only data).

**Benchmark** (`make benchmark` -> `reports/BENCHMARK.md`, conf 0.25), model A:

| Test set | Score |
|---|---|
| mean box AP50, 17 sets | 0.529 |
| Nagar Netra test / RDD-India locked / BharatPotHole / Kaggle web | 0.649 / 0.334 / 0.098 / 0.781 |
| eYantra false alarms, day / night | 0.2 % / 0.4 % |
| night check (Nagar / web / phone) | 0.660 / 0.782 / 0.278 |
| phone street walk: outlines AP50; photos found / clean flagged | 0.274; 11 / 26 · 4 / 13 |

Known gap: phone pothole photos found 11 / 26 (the old model found 14); A was made the model in use by user decision
over the D38 adoption rule (D50). Fix: mark more phone drives and fine-tune A on them.

**Folders (D43)**: training data `datasets/`, test sets `test_datasets/` (one folder each, `configs/datasets.yaml`;
10 generated `check_*` night / zoom sets, `make check-sets`), weights and runs `models/`, app output `app_data/`,
dataset code `dataset_tools/`, tools `tools/`. Each has a README.

**Ground truth**: the phone walk (`test_datasets/phone_street_20261118`, 26 pothole / 13 clean photos) uses the
user's outlines. Corrections from the failure-review page (`make annotate`, "Failures of …" sets, D49) are applied by
`make annotate-export`; originals kept in `labels_original/`, "not sure" photos in `excluded.txt`. Saves are logged in
`datasets/labelling/annotate/events.jsonl`.

**Dashboards**: pipeline `make e2e-dashboard` (:8795), annotation / failure review `make annotate` (:8775), test
results `make results` (:8785), training `make dashboard` (:8765), project control `make control` (:8790), model test
`make playground` (:8780), phone capture `make capture-public` (:8443 / ngrok). LAN pages open on a phone on the same
Wi-Fi.

**End to end**: `make e2e-demo` runs a dummy phone drive (public test photos along Banjara Hills Road No. 10) through
model A, the API on a demo DB copy (`app_data/demo/`), plans, work orders, a repair run and KPIs; real test: `make api`
+ `make capture-public`, record on the phone, then `make e2e-dashboard`.

**Experiment history (models deleted, results in DECISIONS)**: D38 base-data fine-tune (ft_v1; best phone outlines,
24.9 % false alarms) · D39 combined fine-tune (good public scores, poor phone) · D42 clean-roads fine-tune (too
cautious) · D44 / D45 Kaggle notebook and its recipe on our data (worse than ours) · D48 candidate B (YOLO11s-seg,
mean 0.521).

**Next**
1. Record real drives (`make capture-public`), mark them in `make annotate`, `make annotate-export`, fine-tune A.
2. Make an int8 TFLite of model A for the phone / bus box (needs TensorFlow).
3. Staged real-camera size check (`make truth`); every calibration is still `certified=false`.

**Run**: `make api`, `make web`, `make capture-public` (each in its own terminal); `make help` lists everything.
Shells that once sourced .env export old SEG_WEIGHTS / DEPTH_BACKEND; the make targets ignore them, otherwise
`unset` them first. Restart `make api` / `make capture-public` after changing the model.

---

## 1. Where the project stopped

The project lives on the Linux box at `~/gunthameter` (it came back from the Windows laptop on 2026-11-16; the laptop
scripts were deleted, D18). Machine: AMD Ryzen 9 9950X3D (16 cores / 32 threads), 59 GB RAM, **RTX 5080 16 GB
(sm_120)**, NVIDIA driver 595.99. `.venv`: Python 3.11.18, torch 2.11.0+cu128, ultralytics 8.4.211 (`LOCKFILE.md`).

- `make test`: **434 passed** in ~1 min (`-m "not gpu and not data"`, 2026-11-18). `ruff check .` reports 13 findings (B008/B904/B905/E731) in `dataset_review.py`, `model_playground.py`, `ml/train_seg.py`, `tests/test_dataset_review.py`; not fixed yet.
- WP0-WP12 software is complete (status table below). New on 2026-11-16: the v2 training data (D16), the retraining
  pipeline with the GPU at ~95 % (D17), and the clean-up (D18). **2026-11-17: the first full v2 run (`y26m_seg_india_v2`)
  scored RDD-India 0.1602 < 0.2207 (optimizer auto -> MuSGD lr 0.01, early stop at epoch 67; D19). Next run: profile
  `rtx5080_gpu95_ft` (AdamW 0.00125, cosine, 80 epochs, no early stop) as `y26m_seg_india_v2ft`.**

| WP | State |
|---|---|
| WP0 contracts | Done. `configs/classes.yaml` gained the `pothole_seg_web` source map (vocabulary unchanged; CONTRACTS.md updated) |
| WP1 data | **v2 built** (D16): `datasets/earlier_india_pothole_v2`, train 8,911 / val 1,722 / test 1,843 images; 14,564 training potholes (v1: 3,926). RDD-India test locked (`datasets/splits/rdd_india_test.lock.txt`, 514 images) |
| WP2 measurement | **Depth off since 2026-11-18 (D27)**: size-only (outline on the calibrated road plane), IRC:82 severity by width (+1 level on a bus jolt), bags as a range for an assumed depth. Depth code kept (`DEPTH_BACKEND=da2_metric` restores it). Real-camera size accuracy: pending the staged session |
| WP3 training | Pipeline ready (`make retrain`, D17). Current model = `y26m_seg_india_v2ft_geo` (D21; RDD-India pothole box AP50 **0.2494**, best so far, FAILS the 0.70 floor), `SEG_WEIGHTS` in `.env` since 2026-11-17 with `VERIFIER=off` (D24). Older runs, bootstrap and laptop weights deleted (D24); their scores stay in `reports/model_eval.json`. ONNX export skipped: `onnx` package not installed |
| **Model in use (D37, 2026-11-18)** | User-supplied Nagar Netra civic_services YOLO11n (`models/downloaded_pretrained/civic_services_yolo11n_best.pt`, pothole = class 1) + SAM2 outlines; best on unseen phone frames (11/13 found, 2/20 false) and Indian dashcam images (D36). Before: D35 logesshhh, trained y26m (kept on disk) |
| WP4-WP12 | Done (see DECISIONS waves). Web, simulator, ops, demo unchanged by this session |

Scores on the v2 test split (`reports/model_eval.json`; RDD-India = the acceptance basis):

| Model | RDD-India pothole AP50 | BharatPotHole AP50 | human-mask (`ws_`) AP50 | `ws_` mask IoU |
|---|---|---|---|---|
| laptop YOLO26n (stopgap) | 0.2207 | 0.035 | 0.338 | 0.72 |
| v2 run 1 YOLO26m (`y26m_seg_india_v2`, D19) | 0.1602 | 0.070 | 0.637 | 0.77 |
| bootstrap YOLO26m detector | 0.088 | 0.021 | 0.624 | n/a |
| Hugging Face keremberke YOLOv8s-seg | 0.001 | 0.000 | 0.105 | 0.70 |

GPU trials (one full epoch each, D17): YOLO26m-seg batch 18, 96.5 % of the card's memory, load while training median
91 %, 1.7 min/epoch (the `make retrain` default); YOLO26s-seg batch 33, 91.7 %, median 87 %, 0.9 min/epoch.

## 2. Next steps

1. **Done 2026-11-17:** `y26m_seg_india_v2ft_geo` trained (80 epochs, D21) -> RDD-India AP50 0.2494 (laptop 0.2207);
   it is `SEG_WEIGHTS`, the verifier is off (its 0.004 model is deleted), and only the best run is kept (D24).
   Next model work is data, not epochs (best validation epoch was 23/80): review the dataset (`make review`, D22),
   add real bus-mounted frames, then `make derive-dataset-reviewed` and retrain with
   `make train-gpu PROFILE=rtx5080_gpu95_ft_geo NAME=<new run>`. Test models on own images: `make playground` (D23).
   **What is left overall:** `make control` (D26) lists every open item with its gate and command; start there.
   **Demo with a real drive:** `make capture` (office LAN) or `make capture-public` (any network, D29) on this PC,
   open the printed https address on a phone, sign in, Start (no setup: any orientation, tilt measured during the
   drive, height from a vehicle preset, D30), drive (upload while driving is on by default), results appear on <link>/dashboard and finished drives go to the running API by themselves (D32;
   `make capture-replay TRIP=<id>` still works by hand). If the shell exported the old
   .env, `unset DEPTH_BACKEND` first (an environment variable overrides .env).
2. **Verifier:** off. Re-enable only with a second model that actually scores (e.g. a second v2 model); the gate in
   `server/workers/stages.py` drops unconfirmed detections below conf 0.5.
3. **ONNX export:** `pip install --no-deps onnx ml_dtypes` in `.venv`, then `make export NAME=y26m_seg_india_v2ft_geo`
   (the post-training export was skipped). `make prune-v1` (v1 datasets + split files; approved 2026-11-16) can run now.
4. Still open, not code: staged foam-board accuracy session (`make truth`), licence decisions (`docs/LICENCES.md` §4:
   Ultralytics AGPL, DA-V2 Metric Outdoor Base card, eYantra GPL-2.0), Android build, Lighthouse audit, PostGIS
   production path (DECISIONS D1-D3; `mapmatch_pg` never ran against PostGIS).

### Known issues found in the 2026-11-16 code review (not fixed yet)
- `PIPELINE_INLINE=0` (production path, D2) is half-wired: no production code enqueues jobs, and trip-close runs its
  hooks inline once (`server/app/routers/ingest.py:258-265`), before the poller processes that trip's observations.
- Demo OTP: in `DEMO_MODE` the fixed code works for any phone and OTP login matches staff by phone, so it yields a
  super_admin token (`server/app/routers/auth.py:62-75`). Fine for the demo, never for a deployment.
- `edge/sim/fleet.py:247` seeds trips with Python's per-process `hash()`: fleet runs are not reproducible.
- Reported by the review, not re-checked: `recurrence_watch` never entered; jurisdiction queue resolvable across
  agencies; anonymous signed frame URLs and no face/plate blurring (DPDP); Android outbox can lose data and retries
  4xx forever; fleet verification timing (`claim_at + 20 min`) can miss on a slow survey.

## 3. How to run things (Linux, from `~/gunthameter`)
```bash
make test                 # 312 CPU tests, ~1 min
make train-check          # venv, GPU (sm_120), free memory, dataset, RDD-India test lock
make retrain-smoke        # ~2 min trial of the training pipeline (batch, GPU memory/load)
make capture              # phone as the bus camera: HTTPS page for phones on the LAN :8443 (D28)
make capture-public       # same page from ANY network (mobile data) via an ngrok https tunnel (D29)
                          # both process drives automatically; results + map at <link>/dashboard (D32)
make capture-replay TRIP=<id> API=http://127.0.0.1:8000   # detect on the phone keyframes, send the trip to the API (D28)
make control              # project control: everything left to finish end to end, checked live, :8790 (D26)
make dashboard            # live training dashboard http://127.0.0.1:8765 (read-only; own terminal)
make review               # multi-user dataset review on the LAN :8770 (D22); make derive-dataset-reviewed applies it
make playground           # test models (+ measurement, plain-English summary) on own photos/videos, LAN :8780 (D23/D25)
make retrain              # full retraining: check, data, baselines, train, eval, export, report
make baselines            # already-trained pothole models scored on the v2 test
make dataset && make derive-dataset   # rebuild v2 (SAM2 on GPU, ~2 min; cached)
make demo                 # end-to-end demo, keeps API :8000 + web :5173 up (add --exit to stop)
make measure-demo         # measurement demo -> reports/measure_demo/README.md
make report               # regenerate reports/REPORT.md
```
`cd web && npm ci` once before the demo's web app; `npm test` / `npm run build` for the web checks. Playwright uses the
installed Chrome by default: on this box set `PW_CHANNEL=bundled` and run `npx playwright install chromium` once.

## 4. Environment setup on a fresh Linux machine
```bash
python3.11 -m venv .venv && .venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-lock.txt
make setup && make test
```
Raw downloads were deleted after extraction (D18): `datasets/extracted` holds the extracted sources and is what the builds
read. On a machine without `datasets/extracted`, `python -I dataset_tools/download.py --all --fetch` downloads again (Kaggle needs
`~/.kaggle/kaggle.json`; Hugging Face mirror for RDD2022).

## 5. Machine facts that do not carry over
- The GPU is shared with the desktop (1.2-2 GB) and possibly another account (`ml`, D10): training sizes its batch
  from the memory free at start; an out-of-memory restart steps the batch down by 10 %.
- No PostgreSQL/PostGIS/Redis installed; the SQLite + in-process path is the demo (D1-D3).
- `LOCKFILE.md` describes this machine; regenerate it elsewhere with `make lock`.
