#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 WAIT_PID JOB_SCRIPT GPU_INDEX" >&2
  exit 2
fi

wait_pid="$1"
job_script="$2"
gpu="$3"
while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 30
done
exec bash "$job_script" "$gpu"

