"""Audit exact byte-message overlap and within-split duplication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def identities(path: Path) -> tuple[list[tuple[str, bytes]], int]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                values.append((row["protocol"], bytes(row["bytes"])))
    return values, len(set(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    split_paths = sorted(args.data_dir.glob("*.jsonl"))
    split_data = {path.stem: identities(path) for path in split_paths}
    report = {
        "splits": {
            name: {
                "rows": len(values),
                "unique_messages": unique,
                "duplicate_rows": len(values) - unique,
            }
            for name, (values, unique) in split_data.items()
        },
        "pairwise_unique_overlap": {},
    }
    names = sorted(split_data)
    for index, left in enumerate(names):
        left_set = set(split_data[left][0])
        for right in names[index + 1 :]:
            overlap = left_set & set(split_data[right][0])
            report["pairwise_unique_overlap"][f"{left}__{right}"] = len(overlap)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

