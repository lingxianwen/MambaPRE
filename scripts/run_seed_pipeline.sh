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
export PYTHONUNBUFFERED=1
cd "$project_dir"

run_model() {
  local tag="$1"
  local base_config="$2"
  local run_prefix="$3"
  local run_dir="runs/${run_prefix}_seed${seed}"
  local result_dir="results/multiseed/seed${seed}/${tag}"
  local seeded_config="runs/configs/${tag}_seed${seed}.json"
  mkdir -p "$result_dir"
  "$python_bin" scripts/make_seed_config.py \
    --base "$base_config" --seed "$seed" --output-dir "$run_dir" \
    --config-output "$seeded_config"
  if [[ ! -s "$run_dir/best.pt" ]]; then
    "$python_bin" -m mambapre.cli train --config "$seeded_config"
  fi
  "$python_bin" -m mambapre.cli evaluate \
    --checkpoint "$run_dir/best.pt" \
    --data data/processed/core_corpus/test_novel.jsonl \
    --device cuda --batch-size 32 --workers 4 \
    --output "$result_dir/test_novel.json"
  "$python_bin" -m mambapre.cli evaluate \
    --checkpoint "$run_dir/best.pt" \
    --data data/processed/core_corpus/ood_novel.jsonl \
    --device cuda --batch-size 32 --workers 4 \
    --output "$result_dir/ood_novel.json"
  "$python_bin" -m mambapre.cli evaluate \
    --checkpoint "$run_dir/best.pt" \
    --data data/processed/neupre_pdml/test_external_full_novel.jsonl \
    --device cuda --batch-size 32 --workers 4 \
    --output "$result_dir/neupre_full_novel.json"
  "$python_bin" -m mambapre.cli calibrate-evaluate \
    --checkpoint "$run_dir/best.pt" \
    --calibration-data data/processed/neupre_pdml/calibration_long_envelope_novel.jsonl \
    --test-data data/processed/neupre_pdml/test_long_envelope_session_disjoint_novel.jsonl \
    --device cuda --batch-size 4 --workers 4 \
    --output "$result_dir/neupre_long_session_calibrated.json"
}

run_model guided_mamba configs/dual_bimamba_guided_dedup.json dual_bimamba_guided_dedup
run_model matched_transformer configs/transformer_param_matched_dedup.json transformer_param_matched_dedup
