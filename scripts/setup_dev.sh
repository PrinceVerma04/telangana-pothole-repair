#!/usr/bin/env bash
# Dev setup without sudo (DECISIONS D1-D3). Never reinstalls torch.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

if [[ ! -x $PY ]]; then
  echo "ERROR: .venv missing. Create it with python3.11 -m venv .venv and install torch cu128 first." >&2
  exit 1
fi

$PY - <<'EOF'
import sys
assert sys.version_info[:2] == (3, 11), f"need Python 3.11 venv, got {sys.version}"
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF

# install project deps (torch excluded from pyproject on purpose); --no-deps on the editable to be safe
$PY -m pip install -q --upgrade-strategy only-if-needed -e ".[dev]" --no-deps
$PY - <<'EOF'
import importlib, sys
missing = [m for m in ["ultralytics","transformers","cv2","fastapi","sqlalchemy","shapely","pyproj","h3","jsonschema","yaml","pytest"]
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"missing packages: {missing} -> pip install them into .venv (do not touch torch)")
print("python deps ok")
EOF

[[ -f .env ]] || cp .env.example .env
mkdir -p data/store data/interim data/datasets reports runs
$PY scripts/write_lockfile.py
echo "setup ok"
