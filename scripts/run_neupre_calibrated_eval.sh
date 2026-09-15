#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_DIR="${RESULT_DIR:-results/neupre_external_seed1337}"
GPU_MAMBA="${GPU_MAMBA:-0}"
GPU_TRANSFORMER="${GPU_TRANSFORMER:-1}"
CALIBRATION=data/processed/neupre_pdml/calibration_long_envelope_novel.jsonl
TEST=data/processed/neupre_pdml/test_long_envelope_session_disjoint_novel.jsonl

mkdir -p "${RESULT_DIR}"

run_model() {
  local model_name="$1"
  local gpu="$2"
  local checkpoint="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m mambapre.cli calibrate-evaluate \
    --checkpoint "${checkpoint}" \
    --calibration-data "${CALIBRATION}" \
    --test-data "${TEST}" \
    --batch-size 8 --workers 4 \
    --output "${RESULT_DIR}/${model_name}_long_envelope_session_calibrated.json"
}

run_model guided_mamba "${GPU_MAMBA}" \
  runs/dual_bimamba_guided_dedup_seed1337/best.pt \
  >"${RESULT_DIR}/guided_mamba_calibrated.log" 2>&1 &
mamba_pid=$!

run_model matched_transformer "${GPU_TRANSFORMER}" \
  runs/transformer_param_matched_dedup_seed1337/best.pt \
  >"${RESULT_DIR}/matched_transformer_calibrated.log" 2>&1 &
transformer_pid=$!

wait "${mamba_pid}"
wait "${transformer_pid}"

"${PYTHON_BIN}" - "${RESULT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("*_session_calibrated.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    summary[path.stem] = {
        "selected_threshold": report["selected_threshold"],
        "calibration_n": report["calibration"]["overall"]["n"],
        "calibration_boundary_f1": report["calibration"]["overall"]["boundary_f1"],
        "test_n": report["test"]["overall"]["n"],
        "test_boundary_f1": report["test"]["overall"]["boundary_f1"],
        "test_exact_field_f1": report["test"]["overall"]["exact_field_f1"],
        "test_message_perfection": report["test"]["overall"]["message_perfection"],
    }
print(json.dumps(summary, indent=2))
PY
