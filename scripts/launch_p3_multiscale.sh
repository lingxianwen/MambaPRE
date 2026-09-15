#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p runs/logs results/p3_multiscale

nohup bash -c 'bash scripts/run_p3_multiscale_seed.sh 1337 0 && bash scripts/run_p3_multiscale_seed.sh 9001 0' \
  >runs/logs/p3_multiscale_seed1337_9001.log 2>&1 </dev/null &
echo "seeds=1337,9001 gpu=0 pid=$! log=runs/logs/p3_multiscale_seed1337_9001.log"

seeds=(2027 3407 4701)
for index in 0 1 2; do
  gpu="$((index + 1))"
  seed="${seeds[$index]}"
  log="runs/logs/p3_multiscale_seed${seed}.log"
  nohup bash scripts/run_p3_multiscale_seed.sh "$seed" "$gpu" >"$log" 2>&1 </dev/null &
  echo "seed=$seed gpu=$gpu pid=$! log=$log"
done
