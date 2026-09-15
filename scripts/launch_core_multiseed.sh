#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
mkdir -p runs/logs results/multiseed

seeds=(2027 3407 4701 9001)
for gpu in 0 1 2 3; do
  seed="${seeds[$gpu]}"
  log="runs/logs/core_seed${seed}.log"
  nohup bash scripts/run_seed_pipeline.sh "$seed" "$gpu" >"$log" 2>&1 </dev/null &
  echo "seed=$seed gpu=$gpu pid=$! log=$log"
done

