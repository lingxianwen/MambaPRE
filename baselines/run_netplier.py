from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from baselines.common import group_by_protocol_direction, load_rows, score_predictions, tile_lengths


_MODE = {"dnp3": "linsi"}


def parse_field_columns(path: str | Path) -> list[int]:
    bounds: list[int] = []
    position = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 4:
                continue
            position += int(parts[2]) // 8
            bounds.append(position)
    return bounds


def aligned_to_spans(aligned_hex: str, bounds: list[int], message_length: int) -> list[tuple[int, int]]:
    lengths: list[int] = []
    start = 0
    for end in bounds:
        length = len(aligned_hex[start:end].replace("-", "")) // 2
        if length:
            lengths.append(length)
        start = end
    tail = len(aligned_hex[start:].replace("-", "")) // 2
    if tail:
        lengths.append(tail)
    return tile_lengths(lengths, message_length)


def run_group(alignment_class, raws: list[bytes], mode: str, ep: float) -> list[list[tuple[int, int]]]:
    from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage

    with tempfile.TemporaryDirectory() as temporary:
        alignment = alignment_class(
            messages=[RawMessage(raw) for raw in raws],
            output_dir=temporary + os.sep,
            mode=mode,
            ep=ep,
        )
        alignment.execute()
        bounds = parse_field_columns(alignment.filepath_fields_info)
        aligned = Path(alignment.filepath_output_oneline).read_text(encoding="utf-8").splitlines()
    return [aligned_to_spans(value, bounds, len(raw)) for raw, value in zip(raws, aligned)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the supplied NetPlier alignment source")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-dir", required=True, help="directory containing alignment.py")
    parser.add_argument("--max-per-group", type=int, default=80)
    parser.add_argument("--ep", type=float, default=0.123)
    args = parser.parse_args()
    sys.path.insert(0, args.source_dir)
    from alignment import Alignment

    rows = load_rows(args.data)
    predictions: list[list[tuple[int, int]] | None] = [None] * len(rows)
    for (protocol, direction), indices in sorted(group_by_protocol_direction(rows).items()):
        mode = _MODE.get(protocol, "ginsi")
        for start in range(0, len(indices), args.max_per_group):
            chunk = indices[start : start + args.max_per_group]
            try:
                spans = run_group(Alignment, [rows[index]["raw"] for index in chunk], mode, args.ep)
            except Exception as error:
                print(f"[{protocol}/{direction}] chunk={start} failed: {type(error).__name__}: {error}")
                spans = []
            for local_index, index in enumerate(chunk):
                predictions[index] = spans[local_index] if local_index < len(spans) else [(0, len(rows[index]["raw"]))]

    score_predictions(
        rows,
        [value for value in predictions if value is not None],
        args.output,
        "NetPlier alignment segmentation",
        args.source_dir,
        {"max_per_group": args.max_per_group, "ep": args.ep},
    )


if __name__ == "__main__":
    main()
