# HANDOFF — RoadFix on the Linux RTX 5080 box (updated 2026-11-17)

This file is the single source of truth for "where did we stop". Claude Code memory does NOT travel between
machines; this file and `CLAUDE.md` replace it. Read `docs/DECISIONS.md` **D16 (v2 data), D17 (training pipeline,
95 % GPU), D18 (clean-up, Linux-only) and D19 (first v2 run, fine-tuning profile)** next; D1-D15 and the wave sections are the history.

---

## 1. Where the project stopped

The project lives on the Linux box at `~/gunthameter` (it came back from the Windows laptop on 2026-11-16; the laptop
scripts were deleted, D18). Machine: AMD Ryzen 9 9950X3D (16 cores / 32 threads), 59 GB RAM, **RTX 5080 16 GB
(sm_120)**, NVIDIA driver 595.99. `.venv`: Python 3.11.18, torch 2.11.0+cu128, ultralytics 8.4.211 (`LOCKFILE.md`).

- `make test`: **312 passed** in ~1 min (`-m "not gpu and not data"`); `ruff check .` clean.
- WP0-WP12 software is complete (status table below). New on 2026-11-16: the v2 training data (D16), the retraining
  pipeline with the GPU at ~95 % (D17), and the clean-up (D18). **2026-11-17: the first full v2 run (`y26m_seg_india_v2`)
  scored RDD-India 0.1602 < 0.2207 (optimizer auto -> MuSGD lr 0.01, early stop at epoch 67; D19). Next run: profile
  `rtx5080_gpu95_ft` (AdamW 0.00125, cosine, 80 epochs, no early stop) as `y26m_seg_india_v2ft`.**

| WP | State |
|---|---|
| WP0 contracts | Done. `configs/classes.yaml` gained the `pothole_seg_web` source map (vocabulary unchanged; CONTRACTS.md updated) |
| WP1 data | **v2 built** (D16): `data/datasets/india_pothole_seg4_v2`, train 8,911 / val 1,722 / test 1,843 images; 14,564 training potholes (v1: 3,926). RDD-India test locked (`data/splits/rdd_india_test.lock.txt`, 514 images) |
| WP2 measurement | Done; synthetic 16/16, σ coverage 1.00. Real-camera accuracy: pending the staged session |
| WP3 training | Pipeline ready (`make retrain`, D17). Current model = `y26m_seg_india_v2ft_geo` (D21; RDD-India pothole box AP50 **0.2494**, best so far, FAILS the 0.70 floor), `SEG_WEIGHTS` in `.env` since 2026-11-17 with `VERIFIER=off` (D24). Older runs, bootstrap and laptop weights deleted (D24); their scores stay in `reports/model_eval.json`. ONNX export skipped: `onnx` package not installed |
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
Raw downloads were deleted after extraction (D18): `data/interim` holds the extracted sources and is what the builds
read. On a machine without `data/interim`, `python -I data/download.py --all --fetch` downloads again (Kaggle needs
`~/.kaggle/kaggle.json`; Hugging Face mirror for RDD2022).

## 5. Machine facts that do not carry over
- The GPU is shared with the desktop (1.2-2 GB) and possibly another account (`ml`, D10): training sizes its batch
  from the memory free at start; an out-of-memory restart steps the batch down by 10 %.
- No PostgreSQL/PostGIS/Redis installed; the SQLite + in-process path is the demo (D1-D3).
- `LOCKFILE.md` describes this machine; regenerate it elsewhere with `make lock`.
