#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_dir"

"$python_bin" scripts/verify_p4_freeze.py --freeze results/p4/freeze_manifest.json
sealed_data="data/processed/p4_sealed/test_long_full_opcua_capture_isolated.jsonl"

for seed in 1337 2027 3407 4701 9001; do
  for model in role_guided matched_transformer; do
    checkpoint="runs/p4_${model}_seed${seed}/best.pt"
    output="results/p4_sealed/seed${seed}/${model}/sealed_opcua.json"
    "$python_bin" -m mambapre.cli evaluate \
      --checkpoint "$checkpoint" --data "$sealed_data" \
      --device cuda --batch-size 2 --workers 2 --output "$output"
  done
done
