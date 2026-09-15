#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SEED GPU_INDEX" >&2
  exit 2
fi

seed="$1"
gpu="$2"
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$project_dir"

run_one() {
  local tag="$1"
  local base="$2"
  local config="runs/configs/controlled_long_range_${tag}_seed${seed}.json"
  local run_dir="runs/controlled_long_range/${tag}_seed${seed}"
  local result_dir="results/controlled_long_range/seed${seed}/${tag}"
  mkdir -p "$result_dir"
  "$python_bin" scripts/make_long_range_config.py --base "$base" --seed "$seed" --tag "$tag" --output "$config"
  if [[ ! -s "$run_dir/best.pt" ]]; then
    "$python_bin" -m mambapre.cli train --config "$config"
  fi
  "$python_bin" -m mambapre.cli calibrate-evaluate \
    --checkpoint "$run_dir/best.pt" \
    --calibration-data data/processed/controlled_long_range/validation.jsonl \
    --test-data data/processed/controlled_long_range/test.jsonl \
    --device cuda --batch-size 4 --workers 4 \
    --output "$result_dir/evaluation.json"
  "$python_bin" scripts/export_long_range_per_message.py \
    --checkpoint "$run_dir/best.pt" --evaluation "$result_dir/evaluation.json" \
    --model "$tag" --seed "$seed" --device cuda \
    --output "$result_dir/per_message.csv"
}

run_one full_mamba configs/dual_bimamba_guided_dedup.json
run_one matched_transformer configs/transformer_param_matched_dedup.json
run_one supervised_cnn configs/cnn_param_matched_dedup.json
run_one bimamba_only configs/bimamba_param_matched.json
