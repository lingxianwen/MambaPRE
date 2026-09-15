#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 BASELINE_ROOT DATA_JSONL OUTPUT_DIR [PYTHON]" >&2
  exit 2
fi

baseline_root="$1"
data_file="$2"
output_dir="$3"
python_bin="${4:-python}"
bi_python="${BI_PYTHON:-$python_bin}"
classical_python="${CLASSICAL_PYTHON:-$python_bin}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
baseline_root="$(cd "$baseline_root" && pwd)"

if [[ "$data_file" != /* ]]; then
  data_file="$project_root/$data_file"
fi
if [[ "$output_dir" != /* ]]; then
  output_dir="$project_root/$output_dir"
fi

cd "$project_root"
export PYTHONPATH="$project_root:$baseline_root/netzob-master/netzob-master/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output_dir"

"$bi_python" -m baselines.run_binaryinferno \
  --data "$data_file" \
  --source-dir "$baseline_root/binaryinferno-main/binaryinferno" \
  --python "$bi_python" \
  --output "$output_dir/binaryinferno.json"

"$classical_python" -m baselines.run_netzob \
  --data "$data_file" \
  --source-dir "$baseline_root/netzob-master/netzob-master/src" \
  --split static \
  --output "$output_dir/netzob.json"

"$classical_python" -m baselines.run_nemesys \
  --data "$data_file" \
  --source-dir "$baseline_root/nemesys-master/nemesys-master/src" \
  --output "$output_dir/nemesys.json"

"$classical_python" -m baselines.run_netplier \
  --data "$data_file" \
  --source-dir "$baseline_root/NetPlier-master/NetPlier-master/netplier" \
  --output "$output_dir/netplier.json"
