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

evaluate_revision() {
  local checkpoint="$1"
  local result_dir="$2"
  mkdir -p "$result_dir"
  "$python_bin" -m mambapre.cli evaluate \
    --checkpoint "$checkpoint" \
    --data data/processed/neupre_pdml/test_external_full_novel.jsonl \
    --device cuda --batch-size 32 --workers 4 \
    --output "$result_dir/external_full.json"
  "$python_bin" -m mambapre.cli calibrate-evaluate \
    --checkpoint "$checkpoint" \
    --calibration-data data/processed/neupre_pdml/calibration_long_envelope_novel.jsonl \
    --test-data data/processed/neupre_pdml/test_long_envelope_session_disjoint_novel.jsonl \
    --device cuda --batch-size 4 --workers 4 \
    --output "$result_dir/long_envelope.json"
  "$python_bin" -m mambapre.cli calibrate-evaluate \
    --checkpoint "$checkpoint" \
    --calibration-data data/processed/long_full_strict/calibration_long_full_capture_disjoint_novel.jsonl \
    --test-data data/processed/long_full_strict/test_long_full_capture_disjoint_novel.jsonl \
    --device cuda --batch-size 4 --workers 4 \
    --output "$result_dir/strict_fins.json"
}

train_and_evaluate() {
  local tag="$1"
  local base="$2"
  local run_dir="runs/revision_core/${tag}_seed${seed}"
  local config="runs/configs/revision_core_${tag}_seed${seed}.json"
  "$python_bin" scripts/make_seed_config.py \
    --base "$base" --seed "$seed" --output-dir "$run_dir" \
    --config-output "$config"
  if [[ ! -s "$run_dir/best.pt" ]]; then
    "$python_bin" -m mambapre.cli train --config "$config"
  fi
  evaluate_revision "$run_dir/best.pt" "results/revision_core/seed${seed}/${tag}"
}

# B in the factorial ablation and the Phase-3 less-coarse target.
train_and_evaluate fixed_fusion configs/dual_bimamba_fixed.json
train_and_evaluate structural_only configs/dual_bimamba_guided_structural_only.json
