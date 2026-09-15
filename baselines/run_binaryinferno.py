from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from baselines.common import group_by_protocol_direction, load_rows, score_predictions, tile_lengths


_SPEC_LINE = re.compile(r"^\s*\S+\s+(\*|\d+)V")


def parse_spec(stdout: str) -> list[int | None]:
    lines = stdout.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "SPECSTART")
        end = next(i for i, line in enumerate(lines[start + 1 :], start + 1) if line.strip() == "SPECEND")
    except StopIteration:
        return []
    lengths: list[int | None] = []
    for line in lines[start + 1 : end]:
        match = _SPEC_LINE.match(line)
        if match:
            lengths.append(None if match.group(1) == "*" else int(match.group(1)))
    return lengths


def run_group(
    source_dir: Path,
    python_executable: str,
    hex_messages: list[str],
    endian: str,
    timeout: int,
) -> list[int | None]:
    try:
        completed = subprocess.run(
            [python_executable, "blackboard.py", "--detectors", endian],
            input=("\n".join(hex_messages) + "\n").encode(),
            cwd=source_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return []
    return parse_spec(completed.stdout.decode(errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the supplied BinaryInferno source")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-dir", required=True, help="directory containing blackboard.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--endian", choices=("BE", "LE"), default="BE")
    parser.add_argument("--max-per-group", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    rows = load_rows(args.data)
    predictions: list[list[tuple[int, int]] | None] = [None] * len(rows)
    for (protocol, direction), indices in sorted(group_by_protocol_direction(rows).items()):
        sample = indices[: args.max_per_group]
        lengths = run_group(
            Path(args.source_dir),
            args.python,
            [rows[index]["raw"].hex() for index in sample],
            args.endian,
            args.timeout,
        )
        print(f"[{protocol}/{direction}] n={len(indices)} fields={len(lengths)}")
        for index in indices:
            message_length = len(rows[index]["raw"])
            predictions[index] = tile_lengths(lengths, message_length) if lengths else [(0, message_length)]

    score_predictions(
        rows,
        [value for value in predictions if value is not None],
        args.output,
        "BinaryInferno",
        str(Path(args.source_dir).resolve()),
        {"endian": args.endian, "max_per_group": args.max_per_group, "timeout": args.timeout},
    )


if __name__ == "__main__":
    main()
