from __future__ import annotations

import argparse
import sys

from baselines.common import group_by_protocol_direction, load_rows, score_predictions, tile_lengths


def process_group(raws: list[bytes], aligned: bool) -> dict[int, list[int]]:
    from netzob.Inference.Vocabulary.Format import Format
    from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
    from netzob.Model.Vocabulary.Symbol import Symbol

    messages = [RawMessage(raw) for raw in raws]
    symbol = Symbol(messages=messages)
    if aligned:
        Format.splitAligned(symbol, doInternalSlick=True)
    else:
        try:
            Format.splitStatic(symbol)
        except ValueError:
            symbol = Symbol(messages=messages)
            Format.splitAligned(symbol, doInternalSlick=True)
    positions = {id(message): index for index, message in enumerate(messages)}
    return {
        positions[id(message)]: [len(cell) for cell in cells if len(cell) > 0]
        for message, cells in symbol.getMessageCells().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the supplied Netzob source")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--split", choices=("aligned", "static"), default="static")
    parser.add_argument("--max-per-group", type=int, default=120)
    args = parser.parse_args()
    sys.path.insert(0, args.source_dir)

    rows = load_rows(args.data)
    predictions: list[list[tuple[int, int]] | None] = [None] * len(rows)
    for (protocol, direction), indices in sorted(group_by_protocol_direction(rows).items()):
        for start in range(0, len(indices), args.max_per_group):
            chunk = indices[start : start + args.max_per_group]
            try:
                cells = process_group([rows[index]["raw"] for index in chunk], args.split == "aligned")
            except Exception as error:
                print(f"[{protocol}/{direction}] chunk={start} failed: {type(error).__name__}: {error}")
                cells = {}
            for local_index, index in enumerate(chunk):
                message_length = len(rows[index]["raw"])
                lengths = cells.get(local_index)
                predictions[index] = tile_lengths(lengths, message_length) if lengths else [(0, message_length)]

    score_predictions(
        rows,
        [value for value in predictions if value is not None],
        args.output,
        f"Netzob ({args.split})",
        args.source_dir,
        {"split": args.split, "max_per_group": args.max_per_group},
    )


if __name__ == "__main__":
    main()
