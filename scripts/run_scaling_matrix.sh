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
cd "$project_dir"
mkdir -p results/scaling

for precision in float32 bfloat16; do
  "$python_bin" -m mambapre.cli benchmark \
    --config configs/dual_bimamba_guided_dedup.json \
    --lengths 256,512,1024,2048,4096 --batch-size 8 --device cuda \
    --warmup 10 --repetitions 30 --precision "$precision" \
    --output "results/scaling/guided_mamba_${precision}.json"
  "$python_bin" -m mambapre.cli benchmark \
    --config configs/transformer_param_matched_dedup.json \
    --lengths 256,512,1024,2048,4096 --batch-size 8 --device cuda \
    --warmup 10 --repetitions 30 --precision "$precision" \
    --output "results/scaling/matched_transformer_${precision}.json"
  if [[ "$precision" == "float32" ]]; then
    sdpa_backend="memory_efficient"
  else
    sdpa_backend="flash"
  fi
  "$python_bin" -m mambapre.cli benchmark \
    --config configs/transformer_sdpa_param_matched.json \
    --lengths 256,512,1024,2048,4096 --batch-size 8 --device cuda \
    --warmup 10 --repetitions 30 --precision "$precision" \
    --attention-backend "$sdpa_backend" \
    --output "results/scaling/matched_transformer_sdpa_${precision}.json"
done
