from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def payload_hash(row: dict) -> str:
    values = bytes(row["bytes"]) if "bytes" in row else bytes.fromhex(row["payload_hex"])
    return hashlib.sha256(values).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the P4 sealed-candidate admission gate")
    parser.add_argument(
        "--length-report", default="results/p4/cisa_candidate_transport_lengths.json"
    )
    parser.add_argument(
        "--viewed-data",
        default="data/processed/long_full_strict/all_long_full_strict_novel.jsonl",
    )
    parser.add_argument("--output", default="results/p4/sealed_candidate_ledger.json")
    args = parser.parse_args()

    report = json.loads(Path(args.length_report).read_text(encoding="utf-8"))
    viewed_hashes = {payload_hash(row) for row in read_jsonl(Path(args.viewed_data))}
    decisions = []
    for capture in report["captures"]:
        qualifying = capture.get("qualifying_transport_payload_sha256", [])
        overlap = sorted(set(qualifying) & viewed_hashes)
        if not qualifying:
            status, reason = "rejected", "no transport payload reaches 513 bytes"
        elif overlap:
            status, reason = "rejected", "qualifying application payload was already viewed"
        else:
            status, reason = "pending_full_field_audit", "length and novelty pre-screen passed"
        decisions.append(
            {
                **capture,
                "status": status,
                "reason": reason,
                "viewed_payload_sha256_overlap": overlap,
            }
        )

    accepted = [item for item in decisions if item["status"] == "accepted"]
    pending = [item for item in decisions if item["status"].startswith("pending")]
    output = {
        "source": "CISA ICSNPP official public test captures",
        "gate": {
            "minimum_length": 513,
            "reject_previously_viewed_payload": True,
            "requires_contiguous_full_field_audit_after_prescreen": True,
        },
        "captures": decisions,
        "summary": {
            "candidate_captures": len(decisions),
            "accepted_captures": len(accepted),
            "pending_captures": len(pending),
            "rejected_captures": len(decisions) - len(accepted) - len(pending),
            "sealed_test_ready": False,
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
