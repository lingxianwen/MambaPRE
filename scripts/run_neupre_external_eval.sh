#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_DIR="${RESULT_DIR:-results/neupre_external_seed1337}"
GPU_MAMBA="${GPU_MAMBA:-0}"
GPU_TRANSFORMER="${GPU_TRANSFORMER:-1}"

mkdir -p "${RESULT_DIR}"

run_model() {
  local model_name="$1"
  local gpu="$2"
  local checkpoint="$3"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m mambapre.cli evaluate \
    --checkpoint "${checkpoint}" \
    --data data/processed/neupre_pdml/test_external_full_novel.jsonl \
    --batch-size 64 --workers 4 \
    --output "${RESULT_DIR}/${model_name}_full_novel.json"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m mambapre.cli evaluate \
    --checkpoint "${checkpoint}" \
    --data data/processed/neupre_pdml/test_external_long_envelope_novel.jsonl \
    --batch-size 8 --workers 4 \
    --output "${RESULT_DIR}/${model_name}_long_envelope_novel.json"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m mambapre.cli evaluate \
    --checkpoint "${checkpoint}" \
    --data data/processed/neupre_pdml/test_external_long_full.jsonl \
    --batch-size 1 --workers 0 \
    --output "${RESULT_DIR}/${model_name}_long_full_seen_diagnostic.json"
}

run_model \
  guided_mamba "${GPU_MAMBA}" \
  runs/dual_bimamba_guided_dedup_seed1337/best.pt \
  >"${RESULT_DIR}/guided_mamba.log" 2>&1 &
mamba_pid=$!

run_model \
  matched_transformer "${GPU_TRANSFORMER}" \
  runs/transformer_param_matched_dedup_seed1337/best.pt \
  >"${RESULT_DIR}/matched_transformer.log" 2>&1 &
transformer_pid=$!

wait "${mamba_pid}"
wait "${transformer_pid}"

"${PYTHON_BIN}" - "${RESULT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("*.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    overall = report["overall"]
    summary[path.stem] = {
        "n": overall["n"],
        "boundary_f1": overall["boundary_f1"],
        "exact_field_f1": overall["exact_field_f1"],
        "message_perfection": overall["message_perfection"],
        "threshold": report["threshold"],
    }
print(json.dumps(summary, indent=2))
PY
