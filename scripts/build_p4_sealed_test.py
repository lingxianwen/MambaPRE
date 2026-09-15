from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mambapre.full_field import audit_paths
from mambapre.neupre import convert_rich_record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unopened P4 sealed OPC UA test")
    parser.add_argument("--accepted-rich", required=True)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--output-dir", default="data/processed/p4_sealed")
    args = parser.parse_args()

    accepted_path = Path(args.accepted_rich)
    rows, audit = audit_paths([accepted_path], reference_paths=args.reference)
    capture_ids = sorted({str(row.get("capture_id", "")) for row in rows})
    session_ids = sorted({str(row.get("session_id", "")) for row in rows})
    if len(capture_ids) < 3:
        raise ValueError(f"sealed test requires 3 capture sources, found {capture_ids}")
    if any(len(bytes.fromhex(row["payload_hex"])) < 513 for row in rows):
        raise ValueError("sealed test contains a message shorter than 513 bytes")
    if any(row.get("source_kind") != "public_real_capture" for row in rows):
        raise ValueError("sealed test may contain public real captures only")

    normalized = [convert_rich_record(row) for row in rows]
    output_dir = Path(args.output_dir)
    test_path = output_dir / "test_long_full_opcua_capture_isolated.jsonl"
    write_jsonl(test_path, normalized)
    manifest = {
        "status": "built_unopened",
        "model_evaluation_permitted": False,
        "unlock_condition": "write results/p4/freeze_manifest.json before first model evaluation",
        "test_path": test_path.as_posix(),
        "test_sha256": sha256(test_path),
        "accepted_rich_path": accepted_path.as_posix(),
        "accepted_rich_sha256": sha256(accepted_path),
        "source": "CISA ICSNPP OPC UA Binary official public test captures",
        "source_kind": "public_real_capture",
        "protocol": "opcua",
        "annotation": "Wireshark/tshark PDML with value-verified OPC UA Int32 length prefixes",
        "messages": len(normalized),
        "capture_sources": capture_ids,
        "capture_count": len(capture_ids),
        "sessions": session_ids,
        "session_count": len(session_ids),
        "min_length": min(len(row["bytes"]) for row in normalized),
        "max_length": max(len(row["bytes"]) for row in normalized),
        "capture_sha256": {
            str(row["capture_id"]): str(row["capture_sha256"]) for row in rows
        },
        "training_isolation": {
            "exact_reference_overlap": 0,
            "reference_files": args.reference,
            "protocol_absent_from_training": True,
            "controlled_generated_data_present": False,
        },
        "strict_audit": audit,
        "limitations": [
            "five messages are sufficient for a sealed pilot but not a broad protocol census",
            "three captures come from one public repository; two named client implementations are represented",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
