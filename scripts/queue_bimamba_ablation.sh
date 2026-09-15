#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
log_dir="$project_dir/runs/logs"
queue_log="$log_dir/bimamba_ablation_queue.log"
mkdir -p "$log_dir"

echo "waiting for GPU $gpu to have <4096 MiB allocated and <20% utilization" >> "$queue_log"
stable=0
while (( stable < 3 )); do
  sample="$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  memory="${sample%%,*}"
  utilization="${sample##*,}"
  memory="${memory// /}"
  utilization="${utilization// /}"
  if (( memory < 4096 && utilization < 20 )); then
    stable=$((stable + 1))
  else
    stable=0
  fi
  printf '%s gpu=%s memory=%sMiB utilization=%s%% stable=%s/3\n' \
    "$(date --iso-8601=seconds)" "$gpu" "$memory" "$utilization" "$stable" >> "$queue_log"
  if (( stable < 3 )); then
    sleep 60
  fi
done

cd "$project_dir"
for seed in 1337 2027 3407 4701 9001; do
  echo "$(date --iso-8601=seconds) starting seed $seed on GPU $gpu" >> "$queue_log"
  bash scripts/run_bimamba_ablation_seed.sh "$seed" "$gpu" \
    > "$log_dir/bimamba_ablation_seed${seed}.log" 2>&1
  echo "$(date --iso-8601=seconds) completed seed $seed" >> "$queue_log"
done

export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/aggregate_bimamba_ablation.py >> "$queue_log" 2>&1
echo "$(date --iso-8601=seconds) all Bi-Mamba ablations completed" >> "$queue_log"
