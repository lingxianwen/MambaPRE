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
mkdir -p results/scaling

"$python_bin" -m mambapre.cli benchmark \
  --config configs/transformer_sdpa_param_matched.json \
  --lengths 256,512,1024,2048,4096 --batch-size 8 --device cuda \
  --warmup 10 --repetitions 30 --precision float32 \
  --attention-backend memory_efficient \
  --output results/scaling/matched_transformer_sdpa_float32.json

"$python_bin" -m mambapre.cli benchmark \
  --config configs/transformer_sdpa_param_matched.json \
  --lengths 256,512,1024,2048,4096 --batch-size 8 --device cuda \
  --warmup 10 --repetitions 30 --precision bfloat16 \
  --attention-backend flash \
  --output results/scaling/matched_transformer_sdpa_bfloat16.json
