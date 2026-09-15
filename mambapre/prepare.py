from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .constants import length_bucket
from .data import normalized_row

HEX_RE = re.compile(r"Bytes \(hex\):\s*(.*)$", re.DOTALL)
PORT_RE = re.compile(r"src_port=(\d+).*?dst_port=(\d+)")
DIRECTION_RE = re.compile(r"direction=(\w+)")
PROTOCOL_PORTS = {
    "modbus": {502, 1502},
    "s7comm": {102},
    "dnp3": {20000},
    "iec104": {2404},
    "cip_pccc": {44818},
    "omron_fins": {9600},
}


def _parse_input(text: str) -> tuple[str, str, bytes]:
    port_match = PORT_RE.search(text)
    ports = set(map(int, port_match.groups())) if port_match else set()
    protocol = next(
        (name for name, candidates in PROTOCOL_PORTS.items() if ports & candidates), "unknown"
    )
    direction_match = DIRECTION_RE.search(text)
    direction = direction_match.group(1) if direction_match else "?"
    hex_match = HEX_RE.search(text)
    if not hex_match:
        raise ValueError("missing hex byte stream")
    tokens = [token for token in hex_match.group(1).split() if len(token) == 2]
    byte_values = bytes(int(token, 16) for token in tokens)
    return protocol, direction, byte_values


def convert_legacy_jsonl(path: str | Path, prefix: str) -> list[dict]:
    converted = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            source = json.loads(line)
            protocol, direction, byte_values = _parse_input(source["input"])
            fields = json.loads(source["output"])["fields"]
            converted.append(
                normalized_row(
                    f"{prefix}-{index:06d}", protocol, direction, byte_values, fields
                )
            )
    return converted


def stratified_train_validation_split(
    rows: list[dict], validation_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    strata: dict[tuple[str, str], dict[tuple, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        stratum = (row["protocol"], length_bucket(len(row["bytes"])))
        strata[stratum][_identity(row)].append(row)
    rng = random.Random(seed)
    train, validation = [], []
    for identity_groups in strata.values():
        groups = list(identity_groups.values())
        rng.shuffle(groups)
        n_validation_groups = (
            max(1, round(len(groups) * validation_fraction)) if len(groups) > 1 else 0
        )
        for group in groups[:n_validation_groups]:
            validation.extend(group)
        for group in groups[n_validation_groups:]:
            train.extend(group)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def _identity(row: dict) -> tuple[str, bytes]:
    """Identity used to prevent exact-message leakage across splits."""
    return row["protocol"], bytes(row["bytes"])


def _without_identity_overlap(rows: list[dict], reference: list[dict]) -> list[dict]:
    seen = {_identity(row) for row in reference}
    return [row for row in rows if _identity(row) not in seen]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")


def prepare_dataset(
    source_train: str,
    source_test: str,
    output_dir: str,
    validation_fraction: float = 0.1,
    seed: int = 1337,
    source_ood: str | None = None,
) -> dict:
    all_train = convert_legacy_jsonl(source_train, "source-train")
    train, validation = stratified_train_validation_split(
        all_train, validation_fraction, seed
    )
    test = convert_legacy_jsonl(source_test, "source-test")
    output = Path(output_dir)
    splits = {
        "train": train,
        "validation": validation,
        "test": test,
        "test_novel": _without_identity_overlap(test, all_train),
    }
    if source_ood:
        ood = convert_legacy_jsonl(source_ood, "source-ood")
        splits["ood"] = ood
        splits["ood_novel"] = _without_identity_overlap(ood, all_train)
    for name, rows in splits.items():
        _write_jsonl(output / f"{name}.jsonl", rows)

    stats = {}
    for name, rows in splits.items():
        stats[name] = {
            "n": len(rows),
            "protocols": dict(Counter(row["protocol"] for row in rows)),
            "length_buckets": dict(Counter(length_bucket(len(row["bytes"])) for row in rows)),
            "max_length": max(len(row["bytes"]) for row in rows),
            "unique_messages": len({_identity(row) for row in rows}),
        }
    stats["overlap_audit"] = {
        "train_validation": len({_identity(row) for row in train} & {_identity(row) for row in validation}),
        "source_train_test": len({_identity(row) for row in all_train} & {_identity(row) for row in test}),
        "source_train_ood": (
            len({_identity(row) for row in all_train} & {_identity(row) for row in splits["ood"]})
            if "ood" in splits
            else 0
        ),
    }
    (output / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats
