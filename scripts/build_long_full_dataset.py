from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from mambapre.full_field import audit_paths
from mambapre.neupre import convert_rich_record


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def ordered_groups(values: set[str], seed: int) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest(),
    )


def stats(rows: list[dict]) -> dict:
    return {
        "messages": len(rows),
        "captures": sorted({str(row["capture_id"]) for row in rows}),
        "sessions": sorted({str(row["session_id"]) for row in rows}),
        "protocols": dict(Counter(str(row["protocol"]) for row in rows)),
        "min_length": min((len(row["bytes"]) for row in rows), default=0),
        "max_length": max((len(row["bytes"]) for row in rows), default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build capture-disjoint strict long full-field calibration/test sets"
    )
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--holdout-source-kind",
        action="append",
        default=["public_real_capture"],
        help="source kinds forced into test; repeat as needed",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for value in args.inputs:
        path = Path(value)
        paths.extend(sorted(path.rglob("*.jsonl"))) if path.is_dir() else paths.append(path)
    accepted, audit = audit_paths(paths, reference_paths=args.reference)
    captures = {str(row.get("capture_id", "")) for row in accepted}
    if "" in captures or len(captures) < 2:
        raise ValueError("capture-disjoint split requires at least two named captures")

    holdout_captures = {
        str(row["capture_id"])
        for row in accepted
        if row.get("source_kind") in set(args.holdout_source_kind)
    }
    eligible_captures = captures - holdout_captures
    if not eligible_captures:
        raise ValueError("no captures remain eligible for calibration")
    ordered = ordered_groups(eligible_captures, args.seed)
    target = len(accepted) * args.calibration_fraction
    reserve = 1 if not holdout_captures else 0
    selectable = ordered[:-reserve] if reserve else ordered
    if not selectable:
        raise ValueError("no capture is selectable for calibration")
    prefix_counts = []
    cumulative = 0
    for capture_id in selectable:
        cumulative += sum(str(row["capture_id"]) == capture_id for row in accepted)
        prefix_counts.append(cumulative)
    selected_count = min(
        range(1, len(selectable) + 1),
        key=lambda count: abs(prefix_counts[count - 1] - target),
    )
    calibration_captures = set(selectable[:selected_count])
    calibration_rich = [
        row for row in accepted if str(row["capture_id"]) in calibration_captures
    ]
    test_rich = [
        row for row in accepted if str(row["capture_id"]) not in calibration_captures
    ]
    calibration = [convert_rich_record(row) for row in calibration_rich]
    test = [convert_rich_record(row) for row in test_rich]
    public_real_test = [
        row for row in test if row.get("source_kind") == "public_real_capture"
    ]
    controlled_test = [
        row
        for row in test
        if row.get("source_kind") == "controlled_valid_protocol_traffic"
    ]
    combined = calibration + test

    output = Path(args.output_dir)
    write_jsonl(output / "calibration_long_full_capture_disjoint_novel.jsonl", calibration)
    write_jsonl(output / "test_long_full_capture_disjoint_novel.jsonl", test)
    write_jsonl(output / "test_long_full_public_real_novel.jsonl", public_real_test)
    write_jsonl(
        output / "test_long_full_controlled_capture_disjoint_novel.jsonl",
        controlled_test,
    )
    write_jsonl(output / "all_long_full_strict_novel.jsonl", combined)

    calibration_sessions = {row["session_id"] for row in calibration}
    test_sessions = {row["session_id"] for row in test}
    calibration_bytes = {bytes(row["bytes"]) for row in calibration}
    test_bytes = {bytes(row["bytes"]) for row in test}
    report = {
        "audit": audit,
        "split": {
            "seed": args.seed,
            "calibration_fraction_target": args.calibration_fraction,
            "grouping": "capture_id",
            "holdout_source_kinds": args.holdout_source_kind,
            "forced_test_captures": sorted(holdout_captures),
            "calibration": stats(calibration),
            "test": stats(test),
            "public_real_test": stats(public_real_test),
            "controlled_test": stats(controlled_test),
            "capture_overlap": len(
                set(stats(calibration)["captures"]) & set(stats(test)["captures"])
            ),
            "session_overlap": len(calibration_sessions & test_sessions),
            "exact_byte_overlap": len(calibration_bytes & test_bytes),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
