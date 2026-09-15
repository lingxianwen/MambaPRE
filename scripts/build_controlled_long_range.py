from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from mambapre.constants import ROLE_TO_ID, TYPE_TO_ID


DISTANCES = (64, 128, 256, 512, 1024)
MARKER = bytes.fromhex("f39ac75d")
FILL = 0xA5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_prefix(rng: random.Random, length: int) -> bytes:
    # Values below 0x80 make the 0xF3 marker byte impossible in the prefix.
    return bytes(rng.randrange(0, 128) for _ in range(length))


def make_row(split: str, distance: int, index: int, rng: random.Random) -> dict:
    # The target-position distribution is identical in every distance bucket.
    target_position = rng.randint(1280, 1792)
    trailer_length = rng.randint(96, 192)
    prefix_length = target_position - len(MARKER) - 2 - distance
    if prefix_length <= 0:
        raise ValueError("negative prefix length")
    prefix = make_prefix(rng, prefix_length)
    length_bytes = distance.to_bytes(2, "big")
    body_start = prefix_length + len(MARKER) + len(length_bytes)
    raw = prefix + MARKER + length_bytes + bytes([FILL]) * (distance + trailer_length)
    if body_start + distance != target_position:
        raise AssertionError("target-position construction failed")

    roles = [ROLE_TO_ID["payload"]] * len(raw)
    types = [TYPE_TO_ID["bytes"]] * len(raw)
    marker_start = prefix_length
    length_start = marker_start + len(MARKER)
    roles[marker_start:length_start] = [ROLE_TO_ID["constant"]] * len(MARKER)
    roles[length_start:body_start] = [ROLE_TO_ID["length"]] * 2
    types[length_start:body_start] = [TYPE_TO_ID["uint16_be"]] * 2
    digest = hashlib.sha256(raw).hexdigest()[:20]
    return {
        "id": f"controlled-lr-{split}-d{distance:04d}-{index:04d}-{digest}",
        "protocol": f"controlled_lr_{distance:04d}",
        "direction": "c2s",
        "bytes": list(raw),
        # Only the dependency-controlled boundary is in scope for this stress test.
        "boundaries": [target_position],
        "role_ids": roles,
        "type_ids": types,
        "dependency_distance": distance,
        "target_boundary": target_position,
        "length_field_offset": length_start,
        "body_start": body_start,
        "trailer_length": trailer_length,
        "label_scope": "dependency_controlled_boundary_only",
        "source_kind": "controlled_synthetic_stress_test",
    }


def parse_target(row: dict) -> int:
    raw = bytes(row["bytes"])
    marker_start = raw.find(MARKER)
    if marker_start < 0 or raw.find(MARKER, marker_start + 1) >= 0:
        raise ValueError(f"{row['id']}: marker must occur exactly once")
    length_start = marker_start + len(MARKER)
    distance = int.from_bytes(raw[length_start:length_start + 2], "big")
    return length_start + 2 + distance


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a controlled long-range boundary stress test")
    parser.add_argument("--output-dir", default="data/processed/controlled_long_range")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--train-per-distance", type=int, default=100)
    parser.add_argument("--validation-per-distance", type=int, default=30)
    parser.add_argument("--test-per-distance", type=int, default=50)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_counts = {
        "train": args.train_per_distance,
        "validation": args.validation_per_distance,
        "test": args.test_per_distance,
    }
    split_rows = {}
    for split_index, (split, count) in enumerate(split_counts.items()):
        rows = []
        # Separate deterministic streams preserve equal target-position distributions
        # across buckets without sharing raw messages across splits.
        base_rng = random.Random(args.seed + 10000 * split_index)
        positions = [base_rng.randint(1280, 1792) for _ in range(count)]
        trailers = [base_rng.randint(96, 192) for _ in range(count)]
        for distance_index, distance in enumerate(DISTANCES):
            rng = random.Random(args.seed + 10000 * split_index + 1000 * distance_index)
            for index in range(count):
                row = make_row(split, distance, index, rng)
                # Force paired absolute positions and lengths across distance buckets.
                desired_target = positions[index]
                desired_trailer = trailers[index]
                prefix_length = desired_target - len(MARKER) - 2 - distance
                prefix = make_prefix(rng, prefix_length)
                raw = prefix + MARKER + distance.to_bytes(2, "big") + bytes([FILL]) * (distance + desired_trailer)
                row.update({
                    "bytes": list(raw),
                    "boundaries": [desired_target],
                    "target_boundary": desired_target,
                    "length_field_offset": prefix_length + len(MARKER),
                    "body_start": prefix_length + len(MARKER) + 2,
                    "trailer_length": desired_trailer,
                })
                roles = [ROLE_TO_ID["payload"]] * len(raw)
                types = [TYPE_TO_ID["bytes"]] * len(raw)
                marker_start = prefix_length
                length_start = marker_start + len(MARKER)
                body_start = length_start + 2
                roles[marker_start:length_start] = [ROLE_TO_ID["constant"]] * len(MARKER)
                roles[length_start:body_start] = [ROLE_TO_ID["length"]] * 2
                types[length_start:body_start] = [TYPE_TO_ID["uint16_be"]] * 2
                row["role_ids"] = roles
                row["type_ids"] = types
                row["id"] = f"controlled-lr-{split}-d{distance:04d}-{index:04d}-{hashlib.sha256(raw).hexdigest()[:20]}"
                if parse_target(row) != row["target_boundary"]:
                    raise AssertionError(f"parser rejected {row['id']}")
                rows.append(row)
        split_rows[split] = rows
        write_jsonl(output / f"{split}.jsonl", rows)

    hashes = {
        split: {hashlib.sha256(bytes(row["bytes"])).hexdigest() for row in rows}
        for split, rows in split_rows.items()
    }
    overlap = {
        "train_validation": len(hashes["train"] & hashes["validation"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "validation_test": len(hashes["validation"] & hashes["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"exact-byte overlap detected: {overlap}")
    local_windows = Counter()
    for row in split_rows["test"]:
        target = row["target_boundary"]
        local_windows[bytes(row["bytes"][target - 30:target + 31])] += 1
    if len(local_windows) != 1:
        raise RuntimeError("target local neighborhoods are not identical")
    train_bytes = sum(len(row["bytes"]) for row in split_rows["train"])
    train_positives = sum(len(row["boundaries"]) for row in split_rows["train"])
    pos_weight = (train_bytes - train_positives) / train_positives
    manifest = {
        "name": "controlled long-range stress test",
        "protocol_valid": False,
        "reason_not_protocol_valid": "A synthetic diagnostic grammar is used because the available capture-isolated fully dissected long corpus has only five OPC UA messages and cannot support paired five-distance construction.",
        "distances": list(DISTANCES),
        "seeds_reserved_for_models": [1337, 2027, 3407, 4701, 9001],
        "generation_seed": args.seed,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "per_distance_counts": split_counts,
        "target_position_range": [1280, 1792],
        "target_position_distribution_paired_across_distances": True,
        "target_local_radius": 30,
        "unique_target_local_windows_in_test": len(local_windows),
        "parser_validation": "marker + uint16 big-endian distance exactly reconstructs every target boundary",
        "label_scope": "one dependency-controlled boundary per message",
        "exact_byte_overlap": overlap,
        "training_boundary_pos_weight": pos_weight,
        "files": {
            split: {"path": f"{split}.jsonl", "sha256": sha256(output / f"{split}.jsonl")}
            for split in split_rows
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
