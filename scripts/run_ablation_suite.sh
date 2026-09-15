#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi

gpu="$1"
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
cd "$project_dir"

evaluate_checkpoint() {
  local tag="$1"
  local checkpoint="$2"
  local output_dir="results/ablations_seed1337/$tag"
  if [[ ! -s "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    return 0
  fi
  mkdir -p "$output_dir"
  "$python_bin" -m mambapre.cli evaluate \
    --checkpoint "$checkpoint" --data data/processed/core_corpus/test_novel.jsonl \
    --device cuda --batch-size 32 --workers 4 --output "$output_dir/test_novel.json"
  "$python_bin" -m mambapre.cli evaluate \
    --checkpoint "$checkpoint" --data data/processed/core_corpus/ood_novel.jsonl \
    --device cuda --batch-size 32 --workers 4 --output "$output_dir/ood_novel.json"
  "$python_bin" -m mambapre.cli calibrate-evaluate \
    --checkpoint "$checkpoint" \
    --calibration-data data/processed/neupre_pdml/calibration_long_envelope_novel.jsonl \
    --test-data data/processed/neupre_pdml/test_long_envelope_session_disjoint_novel.jsonl \
    --device cuda --batch-size 4 --workers 4 \
    --output "$output_dir/neupre_long_session_calibrated.json"
}

evaluate_checkpoint unidirectional_mamba runs/mamba_seed1337/best.pt
evaluate_checkpoint bidirectional_mamba runs/bimamba_seed1337/best.pt
evaluate_checkpoint dual_fixed runs/dual_bimamba_fixed_seed1337/best.pt
evaluate_checkpoint dual_unguided runs/dual_bimamba_seed1337/best.pt

no_aux_dir=runs/dual_bimamba_guided_no_aux_seed1337
if [[ ! -s "$no_aux_dir/best.pt" ]]; then
  "$python_bin" -m mambapre.cli train --config configs/dual_bimamba_guided_no_aux.json
fi
evaluate_checkpoint guided_no_semantic_aux "$no_aux_dir/best.pt"
