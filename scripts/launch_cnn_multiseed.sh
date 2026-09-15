#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
mkdir -p runs/logs results/multiseed
seeds=(1337 2027 3407 4701 9001)
for index in "${!seeds[@]}"; do
  gpu=$((index % 4))
  seed="${seeds[$index]}"
  log="runs/logs/cnn_seed${seed}.log"
  nohup bash scripts/run_cnn_seed_pipeline.sh "$seed" "$gpu" >"$log" 2>&1 </dev/null &
  echo "seed=$seed gpu=$gpu pid=$! log=$log"
  if [[ "$index" -eq 3 ]]; then
    wait
  fi
done
