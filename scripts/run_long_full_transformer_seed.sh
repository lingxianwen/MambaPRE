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
checkpoint="runs/transformer_param_matched_dedup_seed${seed}/best.pt"
result_dir="results/multiseed/seed${seed}/matched_transformer"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
cd "$project_dir"
mkdir -p "$result_dir"

"$python_bin" -m mambapre.cli calibrate-evaluate \
  --checkpoint "$checkpoint" \
  --calibration-data data/processed/long_full_strict/calibration_long_full_capture_disjoint_novel.jsonl \
  --test-data data/processed/long_full_strict/test_long_full_capture_disjoint_novel.jsonl \
  --device cuda --batch-size 4 --workers 4 \
  --output "$result_dir/long_full_strict_calibrated.json"
"$python_bin" -m mambapre.cli calibrate-evaluate \
  --checkpoint "$checkpoint" \
  --calibration-data data/processed/long_full_strict/calibration_long_full_capture_disjoint_novel.jsonl \
  --test-data data/processed/long_full_strict/test_long_full_public_real_novel.jsonl \
  --device cuda --batch-size 1 --workers 1 \
  --output "$result_dir/long_full_public_real.json"
"$python_bin" -m mambapre.cli calibrate-evaluate \
  --checkpoint "$checkpoint" \
  --calibration-data data/processed/long_full_strict/calibration_long_full_capture_disjoint_novel.jsonl \
  --test-data data/processed/long_full_strict/test_long_full_controlled_capture_disjoint_novel.jsonl \
  --device cuda --batch-size 4 --workers 4 \
  --output "$result_dir/long_full_controlled.json"
