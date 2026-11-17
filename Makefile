# RoadFix - see README.md. All Python runs from .venv (Python 3.11, torch 2.11.0+cu128 for the RTX 5080).
PY      := .venv/bin/python
DASH_PORT ?= 8765
REVIEW_PORT ?= 8770
PLAY_PORT ?= 8780
PYTEST  := $(PY) -m pytest
export PYTHONPATH := $(CURDIR)

.PHONY: help setup lock test test-fast lint data dataset derive-dataset derive-dataset-reviewed review review-export playground retrain train-gpu dashboard retrain-smoke train-check baselines prune-v1 train train-resume eval export pseudo-label fleet sim-api db seed api worker ops-tick tick web demo demo-lite measure-demo truth report clean

NAME    ?= y26m_seg_india_v2
PROFILE ?= rtx5080_gpu95

help:
	@grep -E '^[a-z0-9-]+:' Makefile | cut -d: -f1 | tr '\n' ' '; echo

setup:            ## validate venv (no torch reinstall), .env, LOCKFILE.md
	bash scripts/setup_dev.sh

lock:
	$(PY) scripts/write_lockfile.py

test:             ## all CPU tests
	$(PYTEST) -m "not gpu and not data"

test-fast:
	$(PYTEST) -x -m "not gpu and not data and not slow"

lint:
	$(PY) -m ruff check .

data:             ## WP1: verify sources; unpack any whose extracted copy is missing (raw zips may be deleted, D18)
	$(PY) -I data/download.py --all

dataset:          ## D16: v2 build (RDD2022 + eYantra + BharatPotHole by drive + human web masks, SAM2.1 on GPU)
	$(PY) data/build_dataset.py --out data/datasets/india_pothole_seg_v2 --splits-dir data/splits/v2 --sam2 \
	  --sam2-max-gpu-gb 10 --negatives-frac 0.2 --include-bharat --include-webseg

derive-dataset:   ## D15/D16: 4-class training set india_pothole_seg4_v2 (manholes re-split by drive sequence)
	$(PY) data/derive_dataset.py --overwrite

derive-dataset-reviewed: review-export ## D22: same, minus train/val images rejected in the dataset review (test stays locked)
	$(PY) data/derive_dataset.py --overwrite --exclude data/review/exports/rejected_trainval.txt

review:           ## D22: multi-user dataset review (keep/reject/needs fix) on the office LAN, port $(REVIEW_PORT)
	$(PY) scripts/dataset_review.py serve --host 0.0.0.0 --port $(REVIEW_PORT)

playground:       ## D23: test the trained model (+ measurement) on your own photos/videos, LAN port $(PLAY_PORT)
	$(PY) scripts/model_playground.py --host 0.0.0.0 --port $(PLAY_PORT)

review-export:    ## D22: write data/review/exports/* (rejected_trainval.txt, rejected_test_pending.txt, needs_fix.txt, CSV)
	$(PY) scripts/dataset_review.py export

retrain:          ## D17: whole training pipeline in the foreground: check, data, baseline, train (95 % GPU), eval, export, report
	$(PY) -m ml.pipeline --name $(NAME) --profile $(PROFILE)

train-gpu:        ## D17: same as retrain via scripts/train_gpu.sh: GPU check (asks to close GPU-heavy apps), log in runs/logs
	NAME=$(NAME) PROFILE=$(PROFILE) bash scripts/train_gpu.sh

dashboard:        ## live training dashboard on http://127.0.0.1:$(DASH_PORT) (read-only; own terminal, Ctrl+C stops)
	$(PY) scripts/train_dashboard.py --port $(DASH_PORT)

retrain-smoke:    ## D17: trial of the pipeline (1 epoch on 15 % of train, short eval); shows batch, GPU memory and load
	$(PY) -m ml.pipeline --name $(NAME) --profile $(PROFILE) --smoke

train-check:      ## environment + GPU + dataset + RDD-India test lock
	$(PY) -m ml.pipeline --stages check

baselines:        ## zero-shot scores of already-trained pothole models (laptop, bootstrap, Hugging Face) on our test
	$(PY) -m ml.pipeline --stages baseline

prune-v1:         ## D18: delete v1 datasets + laptop run, only once a v2 model beats it and .env no longer uses it
	$(PY) -m ml.pipeline --stages prune-v1

train:            ## WP3 training only (pipeline stage train without the OOM restarts); NAME=..., PROFILE=...
	$(PY) -m ml.train_seg --profile $(PROFILE) --name $(NAME)

train-resume:
	$(PY) -m ml.train_seg --name $(NAME) --resume

eval:
	$(PY) -m ml.pipeline --stages eval --name $(NAME)

export:
	$(PY) -m ml.pipeline --stages export --name $(NAME)

pseudo-label:     ## WP3: pseudo-labels for unlabelled frames (review before training on them)
	$(PY) -m ml.pseudo_label

fleet:            ## WP7: fleet against a running API (make sim-api in another shell)
	$(PY) edge/sim/fleet.py --api http://127.0.0.1:8002

sim-api:          ## WP7: API with synthetic road-render depth on a copy of the demo DB (port 8002)
	$(PY) edge/sim/sim_server.py --copy-from data/gunthameter.db

db:               ## WP4: create tables (SQLite demo)
	$(PY) -m server.app.db.init

seed:             ## WP4/WP10: agencies, OSM roads, GTFS, POIs, contracts, users
	$(PY) scripts/seed_demo.py

api:
	$(PY) -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000 --reload

worker:           ## WP6: DB job poller + ops ticks every 300 s
	$(PY) -m server.workers.run

ops-tick:         ## WP6: run the ops ticks once (cron)
	$(PY) -m server.workers.run --tick

tick: ops-tick

web:
	cd web && npm run dev

measure-demo:     ## minimum viable jury demo: no services needed
	$(PY) -m measure.cli demo --out reports/measure_demo

truth:
	$(PY) scripts/staged_pothole_truth.py

report:
	$(PY) scripts/make_report.py

demo:             ## WP12 end-to-end
	bash scripts/run_demo.sh

demo-lite: demo

clean:
	rm -rf .pytest_cache .ruff_cache data/store data/gunthameter.db
