# HANDOFF — GunthaMeter on the Windows laptop (updated 2026-11-15: laptop segmenter trained + evaluated; 2026-11-14: wave 2 batch B + WP12 done)

This file is the single source of truth for "where did we stop". Claude Code memory does NOT travel between
machines; this file and `CLAUDE.md` replace it. Read `docs/DECISIONS.md` D12-D14 and the status sections
("Wave 1", "Wave 2 batch A", **"Wave 2 batch B"**, **"WP12"**) next.

---

## 1. Where the project stopped

The project lives on a **native Windows 11 laptop** at `C:\gunthameter` (no WSL). i7-10510U, 15.8 GB RAM, MX330 2 GB
(torch cu126, usable but slow; D14).

**All work packages WP0-WP12 are built and verified on this laptop (2026-11-14).** The interrupted batch-B workflow
(`wf_6045d889-f84`) was finished by hand in one session; do not resume it. Details: DECISIONS "Wave 2 batch B status"
and "WP12 status".

- `pytest -m "not gpu and not data"`: **276 passed** (~5-6 min incl. the WP12 e2e fleet test). `ruff check .` clean.
- Web: vitest 38 passed, `npm run build` OK, `npm run smoke` 15 ok / 1 skip, Playwright 3/3 with `PW_CHANNEL=msedge`
  (the default Chrome channel hangs on this laptop).
- **Demo:** `.\scripts\windows\tasks.ps1 demo` → 732 s from a clean DB copy; WP7 acceptance 4/4 (43 confirmed,
  verified_closed, reopened + DLP rework, 4 jurisdiction items). Records: `reports/demo_summary.json`,
  `reports/fleet_run.json`. Demo depth is the synthetic road render (method `synthetic`), never a measurement.

| WP | State |
|---|---|
| WP0 contracts | Done (no contract file changed in batch B) |
| WP1 data | Done |
| WP2 measurement | Done; synthetic 16/16, σ coverage 1.00. Real-camera accuracy: pending the staged session |
| WP3 training | Code done. **Training dataset is now 4 classes** (`data/datasets/india_pothole_seg4`, D15: pothole, alligator_crack, crack, manhole; manholes re-split by eYantra sequence, test has 69). **Laptop model trained (2026-11-15)**: `runs/seg/y26n_seg_india_laptop/weights/best.pt` (yolo26n-seg, 20 epochs, best = epoch 15). RDD-India test pothole box mAP50 **0.221** (bootstrap 0.0885) — **FAILS** the 0.70 floor; mask IoU 0.689 on 149 matched potholes vs SAM2 pseudo-masks (pixel IoU only 0.27). Now `SEG_WEIGHTS` in `.env` as a stopgap. Verifier stays `bootstrap_pothole_yolov8s.pt` (cracks 0.64/0.56 vs 0.21/0.12) |
| WP4 DB/API, WP5 geo, WP6 workers, WP8 ops, WP10 GIS | Done (batch A) |
| WP7 simulator | Done: `replay.py`, `fleet.py`, `sim_server.py`, `protocol.md`, Jetson stub, Android skeleton (not compiled: no SDK) |
| WP9 web | Done; Lighthouse not run |
| WP11 validation + docs | Done: `DEPTH_METHOD`, `CALIBRATION_SOP` (te/en), `DEPLOYMENT_TSRTC`, `LICENCES`, `DEMO_SCRIPT`, `reports/REPORT.md` |
| WP12 demo | Done: `scripts/demo.py`, `make demo` / `tasks.ps1 demo`, `tests/e2e/test_demo_pipeline.py` |

How to run things here (PowerShell, from `C:\gunthameter`):
```powershell
.\scripts\windows\tasks.ps1 test            # 276 tests, ~5-6 min
.\scripts\windows\tasks.ps1 demo            # full demo, keeps API :8000 + web :5173 up (add --exit to stop)
.\scripts\windows\tasks.ps1 web-test        # vitest; web-build for the production build
cd web; $env:PW_CHANNEL='msedge'; npm run build; npm run test:e2e   # Playwright offline field test
.\scripts\windows\tasks.ps1 measure-demo    # ~5.5 min CPU -> reports\measure_demo\README.md
.\scripts\windows\tasks.ps1 sim-api         # simulator API on :8002 (copy of the demo DB); then: tasks.ps1 fleet
.\scripts\windows\tasks.ps1 report          # regenerate reports\REPORT.md
```
Do not pass pytest flags such as `-p` through `tasks.ps1`: PowerShell swallows them as common parameters. Call
`.venv\Scripts\python.exe -m pytest ...` directly instead. Watch RAM: the demo (API + fleet) and the test suite
together are fine; add torch models or a second demo and the laptop swaps.

## 2. Next steps (none of them is more code)

1. **Train the segmenter** (WP3 acceptance: box mAP50 ≥ 0.70 on RDD-India test, mask IoU ≥ 0.65).
   RTX 5080 box (preferred): `make train PROFILE=rtx5080_s NAME=y26s_seg_india`, then `make eval NAME=...`,
   `make export NAME=...`, `make report`. Laptop: `.\scripts\windows\tasks.ps1 train` (8-hour resumable sessions,
   ~1 h/epoch), continue with `tasks.ps1 train-resume`; then `tasks.ps1 eval --weights runs\seg\y26n_seg_india_laptop\weights\best.pt --stage trained_y26n_laptop`.
   **Laptop run done 2026-11-15** (box mAP50 0.221, below floor; `.env` `SEG_WEIGHTS` now points at it, revert line
   is in the `.env` comment). More laptop epochs will not reach 0.70 (epochs 16-20 never beat epoch 15): the remaining
   path is the RTX 5080 `y26s` run. Its best conf for F1 is ~0.08, so pipeline thresholds tuned for 0.25 may miss
   detections. A stray `$env:SEG_WEIGHTS` in a shell overrides `.env` (pydantic: env beats file) — open a fresh shell.
2. **Staged foam-board accuracy session** with a checkerboard-calibrated camera: `scripts/staged_pothole_truth.py`
   (template → film → measure → compare) → `reports/staged_truth/summary.json` → `tasks.ps1 report`.
3. **Licence decisions** (`docs/LICENCES.md` §4): Ultralytics AGPL vs Enterprise vs RF-DETR; confirm the DA-V2 Metric
   Outdoor Base card licence (else switch `depth.backend` to `da3_metric`).
4. Android: install the SDK, `gradle wrapper`, `./gradlew assembleDebug` in `edge/android`; then the LiteRT segmenter.
5. Lighthouse PWA audit; a fresh-eyes run of the README quickstart; `make demo` under WSL/Linux.
6. Production path (DECISIONS D1-D3): PostgreSQL/PostGIS (`mapmatch_pg` never ran against PostGIS), real secrets,
   DEMO_MODE off. Pilot plan: `docs/DEPLOYMENT_TSRTC.md`.

## 3. Transfer (done 2026-11-13, all checksums OK — kept for reference; `gm_transfer` can be deleted)

The Linux box (`192.168.0.121`, user `coral`, SSH on port 22) has a staging folder `~/gm_transfer/`:

| file | size | content |
|---|---|---|
| `gunthameter_code.tar.gz` | ~0.2 GB | all code, configs, docs, tests, weights, web sources (no `.venv`, no `node_modules`) |
| `gunthameter_data_processed.tar` | ~4.2 GB | `data/interim/` (extracted datasets, geo outputs, SAM2 cache) + `data/datasets/` |
| `data_raw/` | ~13 GB | original downloads: RDD2022 zip, eYantra zip, BharatPotHole zip, TGSRTC GTFS, OSM PBF |
| `SHA256SUMS` | | checksums for everything above |
| `HANDOFF.md` | | this file |
| `pull.ps1`, `verify.ps1` | | Windows helpers (copies of `scripts/windows/`) |

Pull from the laptop (Windows 10/11 has `ssh`, `scp` and `tar` built in; you will be asked for `coral`'s password):

**Native Windows (PowerShell) — one command does copy + verify + unpack:**
```powershell
mkdir C:\gunthameter; cd C:\gunthameter                       # avoid paths with spaces
scp coral@192.168.0.121:gm_transfer/pull.ps1 .
powershell -ExecutionPolicy Bypass -File .\pull.ps1           # -Dest / -Server / -User to override
```
Manual equivalent:
```powershell
cd C:\gunthameter
scp -r coral@192.168.0.121:gm_transfer .
powershell -ExecutionPolicy Bypass -File .\gm_transfer\verify.ps1
tar -xzf .\gm_transfer\gunthameter_code.tar.gz
tar -xf  .\gm_transfer\gunthameter_data_processed.tar
New-Item -ItemType Directory -Force data\raw | Out-Null
Get-ChildItem .\gm_transfer\data_raw | Move-Item -Destination data\raw\
```

**WSL2 (Ubuntu terminal):**
```bash
mkdir -p ~/gunthameter && cd ~/gunthameter
scp -r coral@192.168.0.121:gm_transfer .        # or: rsync -a --partial --progress coral@192.168.0.121:gm_transfer/ gm_transfer/
(cd gm_transfer && sha256sum -c SHA256SUMS)
tar -xzf gm_transfer/gunthameter_code.tar.gz
tar -xf  gm_transfer/gunthameter_data_processed.tar
mkdir -p data/raw && mv gm_transfer/data_raw/* data/raw/
```
Keep the project inside the WSL filesystem (`~/...`), **not** under `/mnt/c` — file access there is 5–10× slower.
Delete `gm_transfer` once everything works.

~17 GB over gigabit LAN ≈ 3–5 min; over Wi-Fi 5 ≈ 15–30 min. If `scp` is interrupted, delete the half-copied `gm_transfer` and rerun (or use `rsync --partial` from WSL, which resumes).

Not transferred on purpose:
- `.venv/` (Linux binaries, 8.6 GB) → recreate (section 4).
- `web/node_modules/` (platform-specific esbuild/rollup) → `npm ci`.
- Hugging Face model cache → re-downloads automatically on first use (DA-V2 Metric Outdoor Base ≈ 0.4 GB, SAM2.1 small
  ≈ 0.2 GB, YOLO26 seg weights are already in `weights/`).
- Kaggle API key (`~/.kaggle/kaggle.json`) — a secret; copy it yourself only if you need to re-download Kaggle data
  (`scp coral@192.168.0.121:.kaggle/kaggle.json` → `%USERPROFILE%\.kaggle\` or `~/.kaggle/` in WSL, then `chmod 600`).

## 4. Environment setup on the laptop

Requirements: Python **3.11** (exactly — torch/ultralytics/onnxruntime wheels), Node **22**, Git (optional), NVIDIA GPU
optional (CPU works for tests, the API and the simulator; depth inference and training want CUDA).

### Option A — WSL2 Ubuntu (recommended: Makefile and bash scripts work unchanged)
```powershell
# once, PowerShell as Administrator, then reboot
wsl --install -d Ubuntu-24.04
```
Install the normal Windows NVIDIA driver; do NOT install a Linux driver inside WSL. `nvidia-smi` in WSL should then work.
```bash
sudo apt update && sudo apt install -y software-properties-common make build-essential
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt install -y python3.11 python3.11-venv python3.11-dev
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt install -y nodejs
cd ~/gunthameter
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
# GPU:
.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
# CPU only instead:  ... --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-lock.txt
make setup && make test
cd web && npm ci && npm run build
```
With sudo available in WSL you may also `apt install postgresql postgis redis-server` for the production path
(DECISIONS D1–D3); the demo does not need them.

### Option B — native Windows (PowerShell)
```powershell
winget install -e --id Python.Python.3.11
winget install -e --id OpenJS.NodeJS.LTS        # Node 22
cd C:\gunthameter
powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1          # GPU (cu128)
# powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1 -Cpu   # no NVIDIA GPU
.\scripts\windows\tasks.ps1 test
```
`scripts/windows/tasks.ps1 <target>` replaces `make` (targets: setup, test, lint, db, seed, api, web, measure-demo,
dataset, train). Known native-Windows caveats: `pyrosm` (only needed to rebuild `data/interim/geo`, which was transferred)
may need `pip install pyrosm` from conda-forge wheels if it fails; bash scripts (`scripts/setup_dev.sh`, future
`run_demo.sh`) need Git Bash or WSL; set `PYTHONUTF8=1` (setup.ps1 does) so Telugu strings read correctly.

## 5. Machine-specific facts that do NOT carry over
- Linux box: no sudo, GPU shared with another account (~12 GB used) — DECISIONS D1–D3 and D10 were driven by this.
  On the laptop re-check: `nvidia-smi` free VRAM; if you have sudo/WSL you can choose PostGIS + Redis instead of SQLite +
  in-process (keep the SQLite path working; it is the zero-service demo).
- `LOCKFILE.md` describes the Linux venv; regenerate it on the laptop with `python scripts/write_lockfile.py`.
- Paths: code uses `pathlib` relative to the repo root; avoid spaces in the laptop path anyway.
