# GunthaMeter - see README.md. All Python runs from .venv (Python 3.11; torch cu128, or cu126 on Pascal GPUs).
PY      := .venv/bin/python
PYTEST  := $(PY) -m pytest
export PYTHONPATH := $(CURDIR)

.PHONY: help setup lock test test-fast lint data dataset derive-dataset train train-resume eval export pseudo-label fleet sim-api db seed api worker ops-tick tick web demo demo-lite measure-demo truth report clean

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1 | tr '\n' ' '; echo

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

data:             ## WP1: unpack/convert downloaded datasets (data/raw already populated)
	$(PY) -I data/download.py --all

dataset:          ## WP1: unified YOLO-seg dataset with SAM2 pseudo-masks
	$(PY) data/build_dataset.py --out data/datasets/india_pothole_seg --sam2 --negatives-frac 0.2

derive-dataset:   ## D15: 4-class training set (india_pothole_seg4) + manhole re-split by drive sequence
	$(PY) data/derive_dataset.py

train:            ## WP3 (resumable sessions; on the RTX 5080 box: PROFILE=rtx5080_s NAME=y26s_seg_india)
	$(PY) -m ml.train_seg --profile $(or $(PROFILE),laptop_mx330) --name $(or $(NAME),y26n_seg_india_laptop) --session-hours 8

train-resume:
	$(PY) -m ml.train_seg --name $(or $(NAME),y26n_seg_india_laptop) --resume --session-hours 8

eval:
	$(PY) -m ml.eval --weights runs/seg/$(or $(NAME),y26n_seg_india_laptop)/weights/best.pt --stage trained_$(or $(NAME),y26n_seg_india_laptop)

export:
	$(PY) -m ml.export --weights runs/seg/$(or $(NAME),y26n_seg_india_laptop)/weights/best.pt --formats onnx --register

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
