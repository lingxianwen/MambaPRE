#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
mkdir -p runs/logs results/key_ablation_multiseed

# Seed 1337 already has checkpoints, so GPU 0 first standardizes its evaluation
# outputs and then trains the one missing seed.  The other GPUs train one seed.
nohup bash -c 'bash scripts/run_key_ablation_seed.sh 1337 0 && bash scripts/run_key_ablation_seed.sh 2027 0' \
  >runs/logs/key_ablation_seed1337_2027.log 2>&1 </dev/null &
echo "seeds=1337,2027 gpu=0 pid=$! log=runs/logs/key_ablation_seed1337_2027.log"

seeds=(3407 4701 9001)
for index in 0 1 2; do
  gpu="$((index + 1))"
  seed="${seeds[$index]}"
  log="runs/logs/key_ablation_seed${seed}.log"
  nohup bash scripts/run_key_ablation_seed.sh "$seed" "$gpu" >"$log" 2>&1 </dev/null &
  echo "seed=$seed gpu=$gpu pid=$! log=$log"
done
