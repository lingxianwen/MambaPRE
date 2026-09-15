#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="$1"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_dir"

for seed in 1337 2027 3407 4701 9001; do
  output_dir="results/multiseed/seed${seed}/matched_cnn"
  mkdir -p "$output_dir"
  for item in \
    "test_long_full_capture_disjoint_novel.jsonl:long_full_strict_calibrated.json:4" \
    "test_long_full_public_real_novel.jsonl:long_full_public_real.json:1" \
    "test_long_full_controlled_capture_disjoint_novel.jsonl:long_full_controlled.json:4"; do
    IFS=: read -r test_file output_file batch_size <<<"$item"
    if [[ ! -s "$output_dir/$output_file" ]]; then
      "$python_bin" -m mambapre.cli calibrate-evaluate \
        --checkpoint "runs/cnn_param_matched_dedup_seed${seed}/best.pt" \
        --calibration-data data/processed/long_full_strict/calibration_long_full_capture_disjoint_novel.jsonl \
        --test-data "data/processed/long_full_strict/$test_file" \
        --device cuda --batch-size "$batch_size" --workers 2 \
        --output "$output_dir/$output_file"
    fi
  done
done
