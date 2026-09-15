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
checkpoint="runs/dual_bimamba_guided_dedup_seed$seed/best.pt"
result_dir="results/batch_invariant_revision/pretrained_reval/seed$seed"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$project_dir"
cd "$project_dir"
mkdir -p "$result_dir"

if [[ ! -s "$checkpoint" ]]; then
  echo "missing checkpoint: $checkpoint" >&2
  exit 1
fi

"$python_bin" -m mambapre.cli evaluate \
  --checkpoint "$checkpoint" \
  --data data/processed/core_corpus/test_novel.jsonl \
  --device cuda --batch-size 32 --workers 4 \
  --output "$result_dir/test_novel.json"
"$python_bin" -m mambapre.cli evaluate \
  --checkpoint "$checkpoint" \
  --data data/processed/core_corpus/ood_novel.jsonl \
  --device cuda --batch-size 32 --workers 4 \
  --output "$result_dir/ood_novel.json"
"$python_bin" -m mambapre.cli evaluate \
  --checkpoint "$checkpoint" \
  --data data/processed/neupre_pdml/test_external_full_novel.jsonl \
  --device cuda --batch-size 32 --workers 4 \
  --output "$result_dir/neupre_full_novel.json"
"$python_bin" -m mambapre.cli calibrate-evaluate \
  --checkpoint "$checkpoint" \
  --calibration-data data/processed/neupre_pdml/calibration_long_envelope_novel.jsonl \
  --test-data data/processed/neupre_pdml/test_long_envelope_session_disjoint_novel.jsonl \
  --device cuda --batch-size 4 --workers 4 \
  --output "$result_dir/neupre_long_session_calibrated.json"
"$python_bin" -m mambapre.cli calibrate-evaluate \
  --checkpoint "$checkpoint" \
  --calibration-data data/processed/long_full_strict/calibration_long_full_capture_disjoint_novel.jsonl \
  --test-data data/processed/long_full_strict/test_long_full_capture_disjoint_novel.jsonl \
  --device cuda --batch-size 4 --workers 4 \
  --output "$result_dir/long_full_strict_calibrated.json"
