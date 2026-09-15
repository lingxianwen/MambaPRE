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
run_dir="runs/batch_invariant_revision/guided_mamba_seed$seed"
result_dir="results/batch_invariant_revision/retrained/seed$seed"
seeded_config="runs/configs/batch_invariant_guided_mamba_seed$seed.json"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$project_dir"
cd "$project_dir"
mkdir -p "$result_dir"

"$python_bin" scripts/make_seed_config.py \
  --base configs/dual_bimamba_guided_dedup.json --seed "$seed" \
  --output-dir "$run_dir" --config-output "$seeded_config"
if [[ ! -s "$run_dir/best.pt" ]]; then
  "$python_bin" -m mambapre.cli train --config "$seeded_config"
fi

checkpoint="$run_dir/best.pt"
"$python_bin" -m mambapre.cli evaluate \
  --checkpoint "$checkpoint" --data data/processed/core_corpus/test_novel.jsonl \
  --device cuda --batch-size 32 --workers 4 --output "$result_dir/test_novel.json"
"$python_bin" -m mambapre.cli evaluate \
  --checkpoint "$checkpoint" --data data/processed/core_corpus/ood_novel.jsonl \
  --device cuda --batch-size 32 --workers 4 --output "$result_dir/ood_novel.json"
"$python_bin" -m mambapre.cli evaluate \
  --checkpoint "$checkpoint" --data data/processed/neupre_pdml/test_external_full_novel.jsonl \
  --device cuda --batch-size 32 --workers 4 --output "$result_dir/neupre_full_novel.json"
"$python_bin" -m mambapre.cli calibrate-evaluate \
  --checkpoint "$checkpoint" \
  --calibration-data data/processed/neupre_pdml/calibration_long_envelope_novel.jsonl \
  --test-data data/processed/neupre_pdml/test_long_envelope_session_disjoint_novel.jsonl \
  --device cuda --batch-size 4 --workers 4 --output "$result_dir/neupre_long_session_calibrated.json"
"$python_bin" -m mambapre.cli calibrate-evaluate \
  --checkpoint "$checkpoint" \
  --calibration-data data/processed/long_full_strict/calibration_long_full_capture_disjoint_novel.jsonl \
  --test-data data/processed/long_full_strict/test_long_full_capture_disjoint_novel.jsonl \
  --device cuda --batch-size 4 --workers 4 --output "$result_dir/long_full_strict_calibrated.json"
