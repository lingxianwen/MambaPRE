#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_INDEX SEED [SEED ...]" >&2
  exit 2
fi

gpu="$1"
shift
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
for seed in "$@"; do
  bash scripts/run_batch_invariant_unguided_seed.sh "$seed" "$gpu"
done
