#!/usr/bin/env bash
# WP12 end-to-end demo (make demo): seeded SQLite DB copy -> API (inline pipeline) -> web -> simulated TGSRTC fleet
# -> tomorrow's indent + KPIs. All logic is in scripts/demo.py (same on Windows: .\scripts\windows\tasks.ps1 demo).
#   bash scripts/run_demo.sh            # keep API + web running until Ctrl-C
#   bash scripts/run_demo.sh --exit     # stop after the summary (timing / CI)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
export PYTHONPATH="$PWD" PYTHONUTF8=1 GM_DEVICE="${GM_DEVICE:-cpu}"
exec "$PY" scripts/demo.py "$@"
