import json
from pathlib import Path

import torch

from mambapre.data import normalized_row
from mambapre.losses import multitask_loss
from mambapre.neupre import (
    boundary_only_row,
    prepare_neupre_dataset,
    session_disjoint_calibration_test,
    stable_session_id,
)


def test_boundary_only_row_preserves_unknown_semantics():
    row = boundary_only_row(
        "sample",
        "toy",
        b"\x01\x02\x03\x04",
        [0, 1, 3, 4],
    )
    assert row["boundaries"] == [1, 3]
    assert row["role_ids"] == [-100] * 4
    assert row["type_ids"] == [-100] * 4


def test_session_id_is_direction_independent():
    forward = {"src_ip": "10.0.0.1", "src_port": 123, "dst_ip": "10.0.0.2", "dst_port": 456}
    reverse = {"src_ip": "10.0.0.2", "src_port": 456, "dst_ip": "10.0.0.1", "dst_port": 123}
    assert stable_session_id(forward) == stable_session_id(reverse)


def test_prepare_boundary_maps_removes_reference_bytes(tmp_path: Path):
    maps = tmp_path / "maps"
    maps.mkdir()
    mapping = {"01020304": [0, 1, 4], "0506070809": [0, 2, 5]}
    (maps / "modbus.json").write_text(json.dumps(mapping), encoding="utf-8")

    reference = normalized_row(
        "ref",
        "modbus",
        "c2s",
        bytes.fromhex("01020304"),
        [{"off": 0, "len": 4, "role": "payload", "type": "bytes"}],
    )
    reference_path = tmp_path / "reference.jsonl"
    reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")

    stats = prepare_neupre_dataset(
        pdml_dir=maps,
        output_dir=tmp_path / "output",
        reference_paths=[reference_path],
    )
    assert stats["test_external_full"]["n"] == 2
    assert stats["test_external_full_novel"]["n"] == 1
    assert stats["overlap_audit"]["external_exact_bytes_regardless_of_protocol"] == 1


def test_multitask_loss_skips_missing_semantics_and_gate_targets():
    outputs = {
        "boundary_logits": torch.zeros(1, 4),
        "role_logits": torch.zeros(1, 4, 8),
        "type_logits": torch.zeros(1, 4, 5),
        "spatial_gates": torch.full((1, 2, 4, 3), 0.5),
    }
    batch = {
        "mask": torch.ones(1, 4, dtype=torch.bool),
        "boundary_labels": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        "role_labels": torch.full((1, 4), -100, dtype=torch.long),
        "type_labels": torch.full((1, 4), -100, dtype=torch.long),
    }
    loss, parts = multitask_loss(outputs, batch, gate_weight=0.1)
    assert torch.isfinite(loss)
    assert set(parts) == {"boundary_loss", "loss"}


def test_session_disjoint_calibration_partition():
    rows = [
        {"id": str(index), "session_id": f"session-{index // 3}"}
        for index in range(12)
    ]
    calibration, test = session_disjoint_calibration_test(rows, 0.3, seed=1337)
    calibration_sessions = {row["session_id"] for row in calibration}
    test_sessions = {row["session_id"] for row in test}
    assert calibration_sessions
    assert test_sessions
    assert calibration_sessions.isdisjoint(test_sessions)
