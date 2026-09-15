#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-1}"
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_dir"
mkdir -p runs/logs results/p3_multiscale

for seed in 1337 2027 3407 4701 9001; do
  bash scripts/run_p3_multiscale_seed.sh "$seed" "$gpu"
done

"$python_bin" scripts/aggregate_p3_multiscale.py \
  > runs/logs/p3_multiscale_aggregate.log 2>&1
