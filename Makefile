# RoadFix - see README.md. All Python runs from .venv (Python 3.11, torch 2.11.0+cu128 for the RTX 5080).
# D38 layout: BASE model weights/base/civic_yolo11n.pt + BASE dataset data/datasets/nagar_netra_pothole (training);
# every earlier dataset is a TEST set (configs/datasets.yaml). `make help` lists the targets by section.
PY      := .venv/bin/python
DASH_PORT ?= 8765
REVIEW_PORT ?= 8770
PLAY_PORT ?= 8780
CONTROL_PORT ?= 8790
LAN_HOST ?= 0.0.0.0   # read-only pages open on the office Wi-Fi (phone); LAN_HOST=127.0.0.1 = this PC only
ANNOTATE_PORT ?= 8775
CAPTURE_PORT ?= 8443
API ?= http://127.0.0.1:8000
EPOCHS ?= 80
DATA   ?= base
AUG    ?= default
SMOKE  ?=
PYTEST  := $(PY) -m pytest
export PYTHONPATH := $(CURDIR)
# D35: a shell that once ran `source .env` keeps old SEG_WEIGHTS / DEPTH_BACKEND exported, and an environment variable
# overrides .env. Services and model steps drop them so .env (the single source of truth) decides.
NOENV   := env -u SEG_WEIGHTS -u DEPTH_BACKEND

.PHONY: help setup lock test test-fast lint \
        base-dataset combined-dataset negatives-dataset benchmark train dashboard \
        api web worker ops-tick tick capture capture-public capture-replay playground annotate annotate-export compare-video review review-export control \
        demo demo-lite measure-demo truth report db seed fleet sim-api clean \
        data seg-dataset seg-derive-dataset seg-derive-dataset-reviewed seg-retrain seg-train-gpu seg-retrain-smoke \
        seg-train-check seg-baselines seg-train seg-train-resume seg-eval seg-export seg-prune-v1 pseudo-label

NAME    ?= det_yolo11n_ft_$(shell date +%Y%m%d_%H%M)
PROFILE ?= rtx5080_gpu95_ft_geo

help:             ## this list
	@awk -F':.*## ' '/^## ---/ {print "\n" substr($$0, 4)} /^[a-z0-9-]+:.*## / {printf "  %-28s %s\n", $$1, $$2}' Makefile

## --- setup and checks
setup:            ## validate venv (no torch reinstall), .env, LOCKFILE.md
	bash scripts/setup_dev.sh

lock:             ## write LOCKFILE.md for this machine
	$(PY) scripts/write_lockfile.py

test:             ## all CPU tests
	$(PYTEST) -m "not gpu and not data"

test-fast:        ## CPU tests, stop at the first failure
	$(PYTEST) -x -m "not gpu and not data and not slow"

lint:             ## ruff
	$(PY) -m ruff check .

## --- model: base dataset, benchmark, training (D38)
base-dataset:     ## build data/datasets/nagar_netra_pothole from the supplied set, leak-free split
	$(PY) data/build_base_dataset.py --overwrite

combined-dataset: ## D39 training set: base train x5 + earlier train/val (test parts never included) -> data/datasets/combined_ft
	$(PY) data/build_combined_dataset.py --overwrite

negatives-dataset: ## D42 training set: base train x5 + ONLY earlier images without potholes (clean roads, cracks, manholes) -> data/datasets/base_neg
	$(PY) data/build_combined_dataset.py --earlier negatives --out data/datasets/base_neg --overwrite

benchmark:        ## every model (configs/models.yaml + the one in use) on every test set -> reports/BENCHMARK.md
	$(NOENV) $(PY) -m ml.benchmark

train:            ## fine-tune the base model (DATA=base|combined|base_neg|phone, EPOCHS, AUG=off|light|default|strong, NAME, SMOKE=1 trial), benchmark, adopt only if not worse anywhere
	$(NOENV) $(PY) -m ml.train_det --name $(NAME) --data $(DATA) --epochs $(EPOCHS) --aug $(AUG) $(if $(SMOKE),--smoke,)

dashboard:        ## live training dashboard :$(DASH_PORT), also on the phone (office Wi-Fi) (runs/detect and runs/seg; read-only)
	$(PY) scripts/train_dashboard.py --host $(LAN_HOST) --port $(DASH_PORT)

## --- services (each in its own terminal)
api:              ## RoadFix API on :8000 (SQLite demo DB, size-only measurement)
	$(NOENV) $(PY) -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000 --reload

web:              ## web app on :5173 (cd web && npm ci once)
	cd web && npm run dev

capture:          ## phone page + Drives dashboard + automatic processing, office LAN, https :$(CAPTURE_PORT)
	$(NOENV) $(PY) scripts/phone_capture.py --port $(CAPTURE_PORT)

capture-public:   ## same from ANY network via an ngrok https link (prints the link and the access code)
	$(NOENV) $(PY) scripts/phone_capture.py --public --port $(CAPTURE_PORT)

capture-replay:   ## re-run one captured drive by hand: make capture-replay TRIP=<id> [API=...]
	$(NOENV) $(PY) scripts/capture_replay.py $(TRIP) --api $(API)

playground:       ## model test page for your own photos/videos, LAN :$(PLAY_PORT)
	$(NOENV) $(PY) scripts/model_playground.py --host 0.0.0.0 --port $(PLAY_PORT)

annotate:         ## D41 draw the potholes yourself (outline/box, model guesses shown), LAN :$(ANNOTATE_PORT); also <capture link>/annotate
	$(NOENV) $(PY) scripts/annotate.py serve --port $(ANNOTATE_PORT)

compare-video:    ## D42 phone keyframes vs a video of the same walk: make compare-video TRIP=<id> VIDEO=<mp4> [OFFSET=s] [MODELS=a,b]
	$(NOENV) $(PY) scripts/phone_vs_video.py $(TRIP) "$(VIDEO)" $(if $(OFFSET),--offset $(OFFSET),) $(if $(MODELS),--models $(MODELS),)

annotate-export:  ## your marks -> phone test-set labels (phone_street_marked) + data/datasets/phone_marked (train set 'phone')
	$(PY) scripts/annotate.py export

review:           ## multi-user review of the BASE dataset (keep / reject / fix), LAN :$(REVIEW_PORT)
	$(PY) scripts/dataset_review.py serve --host 0.0.0.0 --port $(REVIEW_PORT)

review-export:    ## write data/review/exports/* from the review decisions
	$(PY) scripts/dataset_review.py export

control:          ## project control (what is left) :$(CONTROL_PORT), also on the phone (office Wi-Fi)
	$(NOENV) $(PY) scripts/project_control.py --host $(LAN_HOST) --port $(CONTROL_PORT)

worker:           ## DB job poller + ops ticks every 300 s (production path)
	$(PY) -m server.workers.run

ops-tick:         ## run the ops ticks once (cron)
	$(PY) -m server.workers.run --tick

tick: ops-tick

## --- demo, reports, database
demo:             ## end-to-end demo with simulated buses (API :8000 + web :5173)
	bash scripts/run_demo.sh

demo-lite: demo

measure-demo:     ## measurement demo, no services needed -> reports/measure_demo
	$(PY) -m measure.cli demo --out reports/measure_demo

truth:            ## staged foam-board accuracy check
	$(PY) scripts/staged_pothole_truth.py

report:           ## regenerate reports/REPORT.md
	$(PY) scripts/make_report.py

db:               ## create tables (SQLite demo)
	$(PY) -m server.app.db.init

seed:             ## agencies, OSM roads, GTFS, POIs, contracts, users
	$(PY) scripts/seed_demo.py

fleet:            ## simulated fleet against make sim-api
	$(PY) edge/sim/fleet.py --api http://127.0.0.1:8002

sim-api:          ## API with synthetic road-render depth on a copy of the demo DB (:8002)
	$(PY) edge/sim/sim_server.py --copy-from data/gunthameter.db

clean:            ## remove caches, demo store and the demo DB
	rm -rf .pytest_cache .ruff_cache data/store data/gunthameter.db

## --- archive: segmentation pipeline (D15-D24), kept runnable; its datasets are now test sets
data:             ## verify downloaded sources (raw zips may be deleted, D18)
	$(PY) -I data/download.py --all

seg-dataset:      ## rebuild the v2 test-data source (RDD2022 + eYantra + BharatPotHole + web masks, SAM2.1)
	$(PY) data/build_dataset.py --out data/datasets/india_pothole_seg_v2 --splits-dir data/splits/v2 --sam2 \
	  --sam2-max-gpu-gb 10 --negatives-frac 0.2 --include-bharat --include-webseg

seg-derive-dataset: ## 4-class india_pothole_seg4_v2 (source of the RDD/Bharat/web/eYantra test sets)
	$(PY) data/derive_dataset.py --overwrite

seg-derive-dataset-reviewed: review-export ## same, minus rejected train/val images
	$(PY) data/derive_dataset.py --overwrite --exclude data/review/exports/rejected_trainval.txt

seg-retrain:      ## old YOLO26-seg pipeline: check, data, baseline, train, eval, export, report
	$(PY) -m ml.pipeline --name $(NAME) --profile $(PROFILE)

seg-train-gpu:    ## same via scripts/train_gpu.sh (GPU check, log in runs/logs)
	NAME=$(NAME) PROFILE=$(PROFILE) bash scripts/train_gpu.sh

seg-retrain-smoke: ## 1-epoch trial of the old pipeline
	$(PY) -m ml.pipeline --name $(NAME) --profile $(PROFILE) --smoke

seg-train-check:  ## environment + GPU + v2 dataset + RDD-India lock
	$(PY) -m ml.pipeline --stages check

seg-baselines:    ## old baseline stage (zero-shot scores on the v2 test)
	$(PY) -m ml.pipeline --stages baseline

seg-train:        ## old training stage only
	$(PY) -m ml.train_seg --profile $(PROFILE) --name $(NAME)

seg-train-resume: ## resume an old segmentation run
	$(PY) -m ml.train_seg --name $(NAME) --resume

seg-eval:         ## old eval stage (reports/model_eval.json)
	$(PY) -m ml.pipeline --stages eval --name $(NAME)

seg-export:       ## old export stage
	$(PY) -m ml.pipeline --stages export --name $(NAME)

seg-prune-v1:     ## delete v1 datasets + laptop run (approved 2026-11-16)
	$(PY) -m ml.pipeline --stages prune-v1

pseudo-label:     ## pseudo-labels for unlabelled frames (review before training on them)
	$(PY) -m ml.pseudo_label
