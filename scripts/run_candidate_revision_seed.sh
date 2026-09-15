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
  local config="runs/configs/candidate_revision_${tag}_seed${seed}.json"
  local run_dir="runs/controlled_candidate_selection/pw8_revision/${tag}_seed${seed}"
  local result_dir="results/controlled_candidate_revision_pw8/seed${seed}/${tag}"
  mkdir -p "$result_dir"
  "$python_bin" scripts/make_candidate_selection_config.py --base "$base" --seed "$seed" --tag "pw8_revision/${tag}" --output "$config"
  if [[ ! -s "$run_dir/best.pt" ]]; then
    "$python_bin" -m mambapre.cli train --config "$config"
  fi
  "$python_bin" -m mambapre.cli calibrate-evaluate \
    --checkpoint "$run_dir/best.pt" \
    --calibration-data data/processed/controlled_candidate_selection/validation.jsonl \
    --test-data data/processed/controlled_candidate_selection/test.jsonl \
    --device cuda --batch-size 4 --workers 4 --output "$result_dir/evaluation.json"
  "$python_bin" scripts/export_long_range_per_message.py \
    --checkpoint "$run_dir/best.pt" --evaluation "$result_dir/evaluation.json" \
    --data data/processed/controlled_candidate_selection/test.jsonl \
    --model "$tag" --seed "$seed" --device cuda --output "$result_dir/per_message.csv"
}

run_one fixed_fusion configs/dual_bimamba_fixed.json
run_one learned_unguided configs/dual_bimamba.json
run_one guided_no_semantic_aux configs/dual_bimamba_guided_no_aux.json
run_one structural_only configs/dual_bimamba_guided_structural_only.json
