#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${1:-0}"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_dir"

for seed in 1337 2027 3407 4701 9001; do
  output="results/bimamba_ablation_multiseed/seed${seed}/neupre_full_novel.json"
  "$python_bin" -m mambapre.cli evaluate \
    --checkpoint "runs/bimamba_param_matched_seed${seed}/best.pt" \
    --data data/processed/neupre_pdml/test_external_full_novel.jsonl \
    --device cuda --batch-size 32 --workers 4 --output "$output"
done
