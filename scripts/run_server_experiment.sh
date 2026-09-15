#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 CONFIG_JSON [GPU_INDEX]" >&2
  exit 2
fi

CONFIG_JSON="$1"
GPU_INDEX="${2:-0}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" -m mambapre.cli train --config "$CONFIG_JSON"
