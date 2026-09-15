from __future__ import annotations

import argparse
import json
from pathlib import Path

from mambapre.full_field import audit_paths


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit true long full-field PDML labels without gap/opaque filling"
    )
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--min-length", type=int, default=513)
    parser.add_argument("--min-fields", type=int, default=5)
    parser.add_argument("--min-structural-fields", type=int, default=3)
    parser.add_argument("--accepted-output")
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    paths: list[Path] = []
    for value in args.inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.jsonl")))
        else:
            paths.append(path)
    accepted, report = audit_paths(
        paths,
        reference_paths=args.reference,
        min_length=args.min_length,
        min_fields=args.min_fields,
        min_structural_fields=args.min_structural_fields,
    )
    if args.accepted_output:
        write_jsonl(Path(args.accepted_output), accepted)
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
