from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from mambapre.constants import ROLE_TO_ID, TYPE_TO_ID


DISTANCES = (16, 32, 64, 128, 256, 512, 1024)
SYNC = bytes.fromhex("f39ac75d")
CANDIDATE = bytes.fromhex("d44dd44d")
FILL = 0xA5
TRAIL = 0xA6
LOCAL_RADIUS = 30
TRAIL_LENGTH = 80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prefix_bytes(rng: random.Random, length: int) -> bytes:
    # Keep sync/candidate leading bytes impossible outside their reserved sites.
    return bytes(rng.randrange(0, 128) for _ in range(length))


def make_block(distance: int, active: int) -> tuple[bytes, int, int]:
    raw = SYNC + bytes([active]) + bytes([FILL]) * distance + CANDIDATE + bytes([TRAIL]) * TRAIL_LENGTH
    selector = len(SYNC)
    candidate = selector + 1 + distance
    return raw, selector, candidate


def parse_candidates(raw: bytes) -> tuple[list[int], list[int], list[int]]:
    selectors, candidates, active = [], [], []
    cursor = 0
    while True:
        start = raw.find(SYNC, cursor)
        if start < 0:
            break
        selector = start + len(SYNC)
        value = raw[selector]
        if value not in (0, 1):
            raise ValueError("selector must be binary")
        candidate = raw.find(CANDIDATE, selector + 1)
        if candidate < 0:
            raise ValueError("candidate marker missing")
        selectors.append(selector)
        candidates.append(candidate)
        active.append(value)
        cursor = candidate + len(CANDIDATE)
    if len(candidates) != 2 or sum(active) != 1:
        raise ValueError("expected two candidates and exactly one active selector")
    return selectors, candidates, active


def make_row(split: str, distance: int, index: int, rng: random.Random) -> dict:
    active_first = rng.randrange(2)
    prefix = prefix_bytes(rng, rng.randint(32, 192))
    suffix = prefix_bytes(rng, rng.randint(32, 192))
    first, _, _ = make_block(distance, active_first)
    second, _, _ = make_block(distance, 1 - active_first)
    raw = prefix + first + second + suffix
    selectors, candidates, active = parse_candidates(raw)
    target = candidates[active.index(1)]
    observed = target - (selectors[active.index(1)] + 1)
    if observed != distance:
        raise AssertionError("selector-to-candidate distance mismatch")

    roles = [ROLE_TO_ID["payload"]] * len(raw)
    types = [TYPE_TO_ID["bytes"]] * len(raw)
    for selector in selectors:
        sync_start = selector - len(SYNC)
        roles[sync_start:selector] = [ROLE_TO_ID["constant"]] * len(SYNC)
        roles[selector] = ROLE_TO_ID["count"]
        types[selector] = TYPE_TO_ID["uint8"]
    digest = hashlib.sha256(raw).hexdigest()[:20]
    return {
        "id": f"controlled-candidate-{split}-d{distance:04d}-{index:04d}-{digest}",
        "protocol": f"controlled_candidate_{distance:04d}",
        "direction": "c2s",
        "bytes": list(raw),
        "boundaries": [target],
        "role_ids": roles,
        "type_ids": types,
        "dependency_distance": distance,
        "target_boundary": target,
        "candidate_boundaries": candidates,
        "selector_offsets": selectors,
        "active_candidate": active.index(1),
        "label_scope": "one selector-controlled candidate boundary per message",
        "source_kind": "controlled_synthetic_stress_test",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired-candidate long-range stress test")
    parser.add_argument("--output-dir", default="data/processed/controlled_candidate_selection")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--train-per-distance", type=int, default=100)
    parser.add_argument("--validation-per-distance", type=int, default=30)
    parser.add_argument("--test-per-distance", type=int, default=50)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested = {
        "train": args.train_per_distance,
        "validation": args.validation_per_distance,
        "test": args.test_per_distance,
    }
    split_rows: dict[str, list[dict]] = {}
    for split_index, (split, count) in enumerate(requested.items()):
        rows = []
        for distance_index, distance in enumerate(DISTANCES):
            rng = random.Random(args.seed + 10000 * split_index + 1000 * distance_index)
            rows.extend(make_row(split, distance, index, rng) for index in range(count))
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

    local_windows = set()
    parser_checks = 0
    for row in split_rows["test"]:
        raw = bytes(row["bytes"])
        selectors, candidates, active = parse_candidates(raw)
        if row["dependency_distance"] >= 64:
            for candidate in candidates:
                local_windows.add(raw[candidate - LOCAL_RADIUS:candidate + LOCAL_RADIUS + 1])
        target = candidates[active.index(1)]
        parser_checks += int(target == row["target_boundary"])
    if len(local_windows) != 1:
        raise RuntimeError(f"candidate local windows differ: {len(local_windows)} unique")

    train_bytes = sum(len(row["bytes"]) for row in split_rows["train"])
    train_positives = sum(len(row["boundaries"]) for row in split_rows["train"])
    manifest = {
        "name": "paired-candidate controlled long-range stress test",
        "protocol_valid": False,
        "reason_not_protocol_valid": "Synthetic diagnostic grammar used to isolate access to distant evidence from exact pointer arithmetic.",
        "distances": list(DISTANCES),
        "primary_distance_buckets": [64, 128, 256, 512, 1024],
        "near_context_controls": [16, 32],
        "cnn_receptive_field": 61,
        "cnn_effective_radius": 30,
        "generation_seed": args.seed,
        "model_seeds": [1337, 2027, 3407, 4701, 9001],
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "per_distance_counts": requested,
        "label_scope": "one selector-controlled candidate boundary per message",
        "candidate_local_radius": LOCAL_RADIUS,
        "unique_candidate_local_windows_in_primary_test_buckets": len(local_windows),
        "parser_validated_test_messages": parser_checks,
        "exact_byte_overlap": overlap,
        "training_boundary_pos_weight": (train_bytes - train_positives) / train_positives,
        "files": {
            split: {"path": f"{split}.jsonl", "sha256": sha256(output / f"{split}.jsonl")}
            for split in split_rows
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
