#!/usr/bin/env bash
# Start (or resume) segmenter training with the GPU used as fully as this machine allows (DECISIONS D17).
# Runs in the FOREGROUND of this terminal; Ctrl+C stops it safely and running the script again resumes.
#
#   bash scripts/train_gpu.sh                    # full pipeline: check, data, baselines, train, eval, export, report
#   bash scripts/train_gpu.sh --smoke            # ~2 min trial (1 epoch on 15 % of train)
#   bash scripts/train_gpu.sh --stages train     # training only (resumes runs/seg/<name> if it exists)
#   PROFILE=rtx5080_gpu95_s NAME=y26s_seg_india_v2 bash scripts/train_gpu.sh   # faster small model
#   Any other ml/pipeline.py option passes through (e.g. --epochs 100, --workers 16, --set compile=true).
#
# What makes it efficient (measured 2026-11-16 on the RTX 5080, one epoch each, see configs/model/train_profiles.yaml):
#   * batch = Ultralytics AutoBatch at 95 % of the GPU memory FREE at start (YOLO26m: batch 18, 96.5 % of the card),
#     mixed precision, images cached in RAM, 16 loader processes: median GPU load 91 % while training.
#   * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: same speed, less memory fragmentation near the limit.
#   * on CUDA out-of-memory the run resumes from its last checkpoint with a 10 % smaller batch.
#   * this script asks you to close other programs that hold GPU memory first: every GiB they free goes to the batch.
#   * D20: if the monitor is on this GPU, 3 GiB stay free for the desktop (else it freezes); GM_DISPLAY_RESERVE_GB=0 turns it off.
# Not used: cuDNN autotuner (no net gain here); torch.compile (needs `sudo apt install python3.11-dev` for Triton, untested).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
NAME="${NAME:-y26m_seg_india_v2}"
PROFILE="${PROFILE:-rtx5080_gpu95}"
if [[ ! -x $PY ]]; then
  echo "ERROR: .venv missing - see README 'Quickstart (Linux)'" >&2
  exit 1
fi

echo "== GPU before training"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.limit --format=csv,noheader
others=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits 2>/dev/null || true)
if [[ -n "$others" ]]; then
  echo "Programs holding GPU memory now (MiB):"
  echo "$others" | awk -F', ' '{name=$3; sub(/ .*/, "", name); n=split(name, p, "/"); printf "  pid %-8s %6s MiB  %s\n", $1, $2, p[n]}'
  big=$(echo "$others" | awk -F', ' '$2 > 300 {print}' | wc -l)
  if [[ $big -gt 0 && -t 0 ]]; then
    echo "Closing the larger ones (e.g. the browser) gives the training a bigger batch."
    read -r -p "Press Enter to start anyway, or Ctrl+C to close them first... " _
  fi
fi

RUN="$NAME"  # the pipeline trains --smoke trials under <name>_smoke
for arg in "$@"; do [[ $arg == --smoke && $RUN != *_smoke ]] && RUN="${NAME}_smoke"; done

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
mkdir -p runs/logs
log="runs/logs/${RUN}_$(date +%Y%m%d-%H%M%S).log"
echo "== training ${RUN} (profile ${PROFILE}); log: ${log}"
set +e
"$PY" -m ml.pipeline --name "$NAME" --profile "$PROFILE" "$@" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e
echo
if [[ $rc -eq 0 ]]; then
  echo "== done. Model: runs/seg/${RUN}/weights/best_pothole.pt  Scores: reports/model_eval.json  GPU: reports/train/${RUN}_gpu.json"
else
  echo "== stopped (exit ${rc}). Run the same command again to resume from the last checkpoint. Log: ${log}"
fi
exit "$rc"
