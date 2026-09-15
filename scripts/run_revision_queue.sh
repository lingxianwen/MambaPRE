#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_INDEX SEED [SEED ...]" >&2
  exit 2
fi

gpu="$1"
shift
for seed in "$@"; do
  ./scripts/run_core_revision_seed.sh "$seed" "$gpu"
  ./scripts/run_controlled_revision_seed.sh "$seed" "$gpu"
done

