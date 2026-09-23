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

run_dir="runs/bigru_param_matched_dedup_seed${seed}"
result_dir="results/multiseed/seed${seed}/matched_bigru"
config="runs/configs/matched_bigru_seed${seed}.json"
mkdir -p "$result_dir"

"$python_bin" scripts/make_seed_config.py \
  --base configs/bigru_param_matched_dedup.json --seed "$seed" \
  --output-dir "$run_dir" --config-output "$config"
if [[ ! -s "$run_dir/best.pt" ]]; then
  "$python_bin" -m mambapre.cli train --config "$config"
fi

"$python_bin" -m mambapre.cli evaluate \
  --checkpoint "$run_dir/best.pt" --data data/processed/core_corpus/test_novel.jsonl \
  --device cuda --batch-size 32 --workers 4 --output "$result_dir/test_novel.json"
"$python_bin" -m mambapre.cli evaluate \
  --checkpoint "$run_dir/best.pt" --data data/processed/core_corpus/ood_novel.jsonl \
  --device cuda --batch-size 32 --workers 4 --output "$result_dir/ood_novel.json"
"$python_bin" -m mambapre.cli evaluate \
  --checkpoint "$run_dir/best.pt" --data data/processed/neupre_pdml/test_external_full_novel.jsonl \
  --device cuda --batch-size 32 --workers 4 --output "$result_dir/neupre_full_novel.json"
"$python_bin" -m mambapre.cli calibrate-evaluate \
  --checkpoint "$run_dir/best.pt" \
  --calibration-data data/processed/neupre_pdml/calibration_long_envelope_novel.jsonl \
  --test-data data/processed/neupre_pdml/test_long_envelope_session_disjoint_novel.jsonl \
  --device cuda --batch-size 4 --workers 4 \
  --output "$result_dir/neupre_long_session_calibrated.json"
"$python_bin" -m mambapre.cli calibrate-evaluate \
  --checkpoint "$run_dir/best.pt" \
  --calibration-data data/processed/long_full_strict/calibration_long_full_capture_disjoint_novel.jsonl \
  --test-data data/processed/long_full_strict/test_long_full_capture_disjoint_novel.jsonl \
  --device cuda --batch-size 4 --workers 4 \
  --output "$result_dir/long_full_strict_calibrated.json"
"$python_bin" -m mambapre.cli evaluate \
  --checkpoint "$run_dir/best.pt" \
  --data data/processed/p4_sealed/test_long_full_opcua_capture_isolated.jsonl \
  --device cuda --batch-size 2 --workers 2 --output "$result_dir/sealed_opcua.json"

"$python_bin" scripts/evaluate_offset_bins.py \
  --checkpoint "$run_dir/best.pt" \
  --data data/processed/long_full_strict/test_long_full_capture_disjoint_novel.jsonl \
  --evaluation-report "$result_dir/long_full_strict_calibrated.json" \
  --model-name matched_bigru --seed "$seed" --device cuda --batch-size 4 --workers 4 \
  --output "$result_dir/strict_fins_offset_bins.json"
"$python_bin" scripts/evaluate_offset_bins.py \
  --checkpoint "$run_dir/best.pt" \
  --data data/processed/p4_sealed/test_long_full_opcua_capture_isolated.jsonl \
  --evaluation-report "$result_dir/sealed_opcua.json" \
  --model-name matched_bigru --seed "$seed" --device cuda --batch-size 2 --workers 2 \
  --output "$result_dir/sealed_opcua_offset_bins.json"
