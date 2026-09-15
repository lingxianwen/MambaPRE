from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from mambapre.neupre import convert_rich_record


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(rows: list[dict]) -> dict:
    return {
        "messages": len(rows),
        "captures": len({str(row.get("capture_id", "")) for row in rows}),
        "sessions": len({str(row.get("session_id", "")) for row in rows}),
        "protocols": dict(Counter(str(row["protocol"]) for row in rows)),
        "min_length": min(len(row["bytes"]) for row in rows),
        "max_length": max(len(row["bytes"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the P4 training and diagnostic ledger")
    parser.add_argument(
        "--controlled-rich",
        default="data/processed/long_full_strict/controlled_fins_accepted.jsonl",
    )
    parser.add_argument(
        "--viewed-strict",
        default="data/processed/long_full_strict/all_long_full_strict_novel.jsonl",
    )
    parser.add_argument("--output-dir", default="data/processed/p4")
    args = parser.parse_args()

    rich_path = Path(args.controlled_rich)
    viewed_path = Path(args.viewed_strict)
    output = Path(args.output_dir)
    rich = read_jsonl(rich_path)
    invalid = [
        row.get("id", "?")
        for row in rich
        if row.get("source_kind") != "controlled_valid_protocol_traffic"
        or row.get("annotation_scope") != "full_field_pdml"
        or len(bytes.fromhex(str(row.get("payload_hex", "")))) < 513
    ]
    if invalid:
        raise ValueError(f"controlled inputs fail the P4 admission policy: {invalid[:5]}")

    controlled = [convert_rich_record(row) for row in rich]
    if any(len(row["role_ids"]) != len(row["bytes"]) for row in controlled):
        raise ValueError("role labels do not continuously cover a controlled message")
    if any(len(row["type_ids"]) != len(row["bytes"]) for row in controlled):
        raise ValueError("type labels do not continuously cover a controlled message")

    viewed = read_jsonl(viewed_path)
    controlled_path = output / "train_long_controlled.jsonl"
    diagnostic_path = output / "dev_fins_viewed_diagnostic_only.jsonl"
    write_jsonl(controlled_path, controlled)
    write_jsonl(diagnostic_path, viewed)

    controlled_bytes = {bytes(row["bytes"]) for row in controlled}
    viewed_bytes = {bytes(row["bytes"]) for row in viewed}
    manifest = {
        "policy": {
            "controlled_data_usage": "training only",
            "paper_label": "protocol-valid controlled traffic",
            "viewed_strict_fins_status": "development/diagnostic only",
            "viewed_strict_fins_final_test_eligible": False,
            "sealed_test_status": "tracked_separately",
            "sealed_test_manifest": "data/processed/p4_sealed/manifest.json",
            "sealed_test_evaluation_allowed_before_model_freeze": False,
        },
        "long_training": {
            "path": controlled_path.as_posix(),
            "source_kind": "protocol-valid controlled traffic",
            "operational_traffic": False,
            "annotation": "full-field Wireshark/tshark PDML",
            "stats": summary(controlled),
            "sha256": sha256(controlled_path),
            "input_sha256": sha256(rich_path),
        },
        "viewed_diagnostic": {
            "path": diagnostic_path.as_posix(),
            "reason": "repeatedly inspected in earlier stages; controlled subset is now training data",
            "stats": summary(viewed),
            "sha256": sha256(diagnostic_path),
            "exact_message_overlap_with_long_training": len(controlled_bytes & viewed_bytes),
        },
        "sealed_test_acceptance_gate": {
            "capture_source_isolated_from_training": True,
            "minimum_message_length": 513,
            "continuous_full_field_coverage": True,
            "target_independent_capture_sources": "2-3",
            "model_outputs_must_remain_unread_until_freeze": True,
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
