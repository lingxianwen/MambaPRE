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
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_dir"

run_variant() {
  local tag="$1"
  local base_config="$2"
  local run_dir="runs/p4_${tag}_seed${seed}"
  local result_dir="results/p4_screen/seed${seed}/${tag}"
  local seeded_config="runs/configs/p4_${tag}_seed${seed}.json"

  mkdir -p "$result_dir"
  "$python_bin" scripts/make_seed_config.py \
    --base "$base_config" --seed "$seed" --output-dir "$run_dir" \
    --config-output "$seeded_config"
  if [[ ! -s "$run_dir/best.pt" ]]; then
    "$python_bin" -m mambapre.cli train --config "$seeded_config"
  fi

  # These are development diagnostics only. No sealed-test path appears here.
  "$python_bin" -m mambapre.cli evaluate \
    --checkpoint "$run_dir/best.pt" \
    --data data/processed/core_corpus/test_novel.jsonl \
    --device cuda --batch-size 32 --workers 4 \
    --output "$result_dir/short_test_novel.json"
  "$python_bin" -m mambapre.cli calibrate-evaluate \
    --checkpoint "$run_dir/best.pt" \
    --calibration-data data/processed/neupre_pdml/calibration_long_envelope_novel.jsonl \
    --test-data data/processed/neupre_pdml/test_long_envelope_session_disjoint_novel.jsonl \
    --device cuda --batch-size 4 --workers 4 \
    --output "$result_dir/long_envelope_diagnostic.json"
  "$python_bin" -m mambapre.cli evaluate \
    --checkpoint "$run_dir/best.pt" \
    --data data/processed/p4/dev_fins_viewed_diagnostic_only.jsonl \
    --device cuda --batch-size 4 --workers 4 \
    --output "$result_dir/viewed_fins_diagnostic_only.json"
}

run_variant role_guided configs/p4_role_guided_long20.json
run_variant boundary_multiscale configs/p4_boundary_multiscale_long20.json
run_variant matched_transformer configs/p4_transformer_matched_long20.json
