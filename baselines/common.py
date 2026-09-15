from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def length_bucket(length: int) -> str:
    for name, low, high in (
        ("0009-0032", 0, 32),
        ("0033-0128", 33, 128),
        ("0129-0256", 129, 256),
        ("0257-0512", 257, 512),
        ("0513-1024", 513, 1024),
        ("1025+", 1025, 10**9),
    ):
        if low <= length <= high:
            return name
    raise ValueError(f"invalid message length: {length}")


@dataclass
class BoundaryCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    exact_fields: int = 0
    predicted_fields: int = 0
    gold_fields: int = 0
    perfect_messages: int = 0
    messages: int = 0

    @staticmethod
    def spans(cuts: set[int], length: int) -> set[tuple[int, int]]:
        points = [0, *sorted(cuts), length]
        return {(start, end - start) for start, end in zip(points, points[1:])}

    def add(self, pred_cuts: set[int], gold_cuts: set[int], length: int) -> None:
        self.true_positive += len(pred_cuts & gold_cuts)
        self.false_positive += len(pred_cuts - gold_cuts)
        self.false_negative += len(gold_cuts - pred_cuts)
        predicted = self.spans(pred_cuts, length)
        gold = self.spans(gold_cuts, length)
        self.exact_fields += len(predicted & gold)
        self.predicted_fields += len(predicted)
        self.gold_fields += len(gold)
        self.perfect_messages += int(pred_cuts == gold_cuts)
        self.messages += 1

    @staticmethod
    def prf(true_positive: int, predicted: int, gold: int) -> tuple[float, float, float]:
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / gold if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    def summary(self) -> dict[str, float | int]:
        precision, recall, f1 = self.prf(
            self.true_positive,
            self.true_positive + self.false_positive,
            self.true_positive + self.false_negative,
        )
        field_precision, field_recall, field_f1 = self.prf(
            self.exact_fields, self.predicted_fields, self.gold_fields
        )
        return {
            "n": self.messages,
            "boundary_precision": precision,
            "boundary_recall": recall,
            "boundary_f1": f1,
            "exact_field_precision": field_precision,
            "exact_field_recall": field_recall,
            "exact_field_f1": field_f1,
            "message_perfection": self.perfect_messages / self.messages if self.messages else 0.0,
        }


def load_rows(path: str | Path) -> list[dict]:
    """Load Mamba-PRE normalized JSONL without changing the evaluation split."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            raw = bytes(row["bytes"])
            cuts = {int(value) for value in row["boundaries"] if 0 < int(value) < len(raw)}
            if not raw:
                raise ValueError(f"empty message at {path}:{line_number}")
            rows.append(
                {
                    "id": row.get("id", f"row-{line_number}"),
                    "protocol": row.get("protocol", "unknown"),
                    "direction": row.get("direction", "unknown"),
                    "raw": raw,
                    "gold_cuts": cuts,
                }
            )
    if not rows:
        raise ValueError(f"no messages found in {path}")
    return rows


def group_by_protocol_direction(rows: list[dict]) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["protocol"], row["direction"])].append(index)
    return groups


def tile_lengths(lengths: list[int | None], message_length: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for length in lengths:
        if offset >= message_length:
            break
        if length is None or length <= 0:
            spans.append((offset, message_length - offset))
            offset = message_length
            break
        take = min(int(length), message_length - offset)
        spans.append((offset, take))
        offset += take
    if offset < message_length:
        spans.append((offset, message_length - offset))
    return spans


def spans_to_cuts(spans: list[tuple[int, int]], message_length: int) -> set[int]:
    cuts: set[int] = set()
    for offset, length in spans:
        for cut in (offset, offset + length):
            if 0 < cut < message_length:
                cuts.add(int(cut))
    return cuts


def score_predictions(
    rows: list[dict],
    spans_per_row: list[list[tuple[int, int]]],
    output_path: str | Path,
    tool: str,
    implementation: str,
    settings: dict | None = None,
) -> dict:
    if len(rows) != len(spans_per_row):
        raise ValueError("prediction count does not match row count")
    overall = BoundaryCounts()
    by_protocol: dict[str, BoundaryCounts] = defaultdict(BoundaryCounts)
    by_length: dict[str, BoundaryCounts] = defaultdict(BoundaryCounts)
    predictions: list[dict] = []
    for row, spans in zip(rows, spans_per_row):
        pred_cuts = spans_to_cuts(spans, len(row["raw"]))
        length = len(row["raw"])
        overall.add(pred_cuts, row["gold_cuts"], length)
        by_protocol[row["protocol"]].add(pred_cuts, row["gold_cuts"], length)
        by_length[length_bucket(length)].add(pred_cuts, row["gold_cuts"], length)
        predictions.append({"id": row["id"], "predicted_boundaries": sorted(pred_cuts)})
    result = {
        "overall": overall.summary(),
        "by_protocol": {key: value.summary() for key, value in sorted(by_protocol.items())},
        "by_length": {key: value.summary() for key, value in sorted(by_length.items())},
    }
    result.update(
        {
            "tool": tool,
            "implementation": implementation,
            "settings": settings or {},
            "predictions": predictions,
        }
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
