import json
from pathlib import Path

import pytest

from mambapre.data import (
    FixedRatioBatchSampler,
    ProtocolDataset,
    collate_samples,
    load_jsonl,
    normalized_row,
)
from mambapre.prepare import stratified_train_validation_split


def test_normalized_row_and_collation(tmp_path: Path):
    row = normalized_row(
        "sample-1",
        "toy",
        "c2s",
        bytes.fromhex("0102030405"),
        [
            {"off": 0, "len": 2, "role": "constant", "type": "uint16_be"},
            {"off": 2, "len": 3, "role": "payload", "type": "bytes"},
        ],
    )
    assert row["boundaries"] == [2]
    path = tmp_path / "toy.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset = ProtocolDataset(load_jsonl(path))
    batch = collate_samples([dataset[0]])
    assert batch["bytes"].shape == (1, 5)
    assert batch["boundary_labels"][0].nonzero().flatten().tolist() == [2]


def test_noncontiguous_fields_rejected():
    with pytest.raises(ValueError, match="non-contiguous"):
        normalized_row(
            "bad",
            "toy",
            "c2s",
            b"\x00\x01\x02",
            [{"off": 1, "len": 2, "role": "payload", "type": "bytes"}],
        )


def test_grouped_split_keeps_duplicate_bytes_together():
    template = {
        "protocol": "toy",
        "direction": "c2s",
        "boundaries": [1],
        "role_ids": [0, 6],
        "type_ids": [0, 4],
    }
    rows = []
    for index, values in enumerate(([1, 2], [1, 2], [3, 4], [5, 6], [7, 8])):
        rows.append({"id": str(index), "bytes": values, **template})
    train, validation = stratified_train_validation_split(rows, 0.4, seed=7)
    train_keys = {(row["protocol"], bytes(row["bytes"])) for row in train}
    validation_keys = {(row["protocol"], bytes(row["bytes"])) for row in validation}
    assert train_keys.isdisjoint(validation_keys)


def test_fixed_ratio_sampler_is_exact_and_covers_base_set():
    sampler = FixedRatioBatchSampler(
        base_size=65,
        long_size=7,
        batch_size=32,
        long_fraction=0.2,
        seed=1337,
    )
    batches = list(sampler)
    assert len(batches) == 3
    assert all(len(batch) == 32 for batch in batches)
    assert all(sum(index >= 65 for index in batch) == 6 for batch in batches)
    assert set(range(65)).issubset({index for batch in batches for index in batch})
    assert sampler.effective_long_fraction == pytest.approx(0.1875)


def test_fixed_ratio_sampler_reproducible_for_same_seed():
    kwargs = dict(
        base_size=53,
        long_size=5,
        batch_size=16,
        long_fraction=0.2,
        seed=2027,
    )
    assert list(FixedRatioBatchSampler(**kwargs)) == list(
        FixedRatioBatchSampler(**kwargs)
    )
