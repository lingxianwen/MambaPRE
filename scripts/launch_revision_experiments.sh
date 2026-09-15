#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
export MAMBAPRE_PROJECT_DIR="$project_dir"
cd "$project_dir"
mkdir -p runs/logs

nohup bash -c '
  set -e
  cd "$MAMBAPRE_PROJECT_DIR"
  bash scripts/run_scaling_matrix.sh 0
  bash scripts/run_cnn_seed_pipeline.sh 1337 0
  bash scripts/run_cnn_seed_pipeline.sh 3407 0
  bash scripts/run_cnn_seed_pipeline.sh 9001 0
' >"$project_dir/runs/logs/scaling_then_cnn_gpu0.log" 2>&1 </dev/null &
gpu0_pid=$!

nohup bash -c '
  set -e
  cd "$MAMBAPRE_PROJECT_DIR"
  bash scripts/run_cnn_seed_pipeline.sh 2027 1
  bash scripts/run_cnn_seed_pipeline.sh 4701 1
' >"$project_dir/runs/logs/cnn_gpu1.log" 2>&1 </dev/null &
gpu1_pid=$!

echo "gpu0_queue_pid=$gpu0_pid log=runs/logs/scaling_then_cnn_gpu0.log"
echo "gpu1_queue_pid=$gpu1_pid log=runs/logs/cnn_gpu1.log"
