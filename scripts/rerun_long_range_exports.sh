#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${1:-0}"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_dir"

for seed in 1337 2027 3407 4701 9001; do
  for model in full_mamba matched_transformer supervised_cnn bimamba_only; do
    "$python_bin" scripts/export_long_range_per_message.py \
      --checkpoint "runs/controlled_long_range/${model}_seed${seed}/best.pt" \
      --evaluation "results/controlled_long_range/seed${seed}/${model}/evaluation.json" \
      --model "$model" --seed "$seed" --device cuda \
      --output "results/controlled_long_range/seed${seed}/${model}/per_message.csv"
  done
done

"$python_bin" scripts/aggregate_controlled_long_range.py
