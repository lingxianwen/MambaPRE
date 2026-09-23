from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mambapre.data import load_jsonl
from mambapre.engine import load_checkpoint, make_loader, move_batch, resolve_device


def parse_bins(specification: str) -> list[tuple[int, int | None]]:
    bins: list[tuple[int, int | None]] = []
    for token in specification.split(","):
        lower, upper = token.split("-", maxsplit=1)
        bins.append((int(lower), None if upper == "inf" else int(upper)))
    return bins


def bin_name(lower: int, upper: int | None) -> str:
    return f"{lower}+" if upper is None else f"{lower}-{upper}"


def select_threshold(checkpoint: dict, report_path: str | None) -> float:
    if report_path:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        if "selected_threshold" in report:
            return float(report["selected_threshold"])
        if "threshold" in report:
            return float(report["threshold"])
        if "test" in report and "threshold" in report["test"]:
            return float(report["test"]["threshold"])
    return float(checkpoint.get("validation", {}).get("threshold", 0.5))


def summarize_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def dataset_max_length(path: str | None) -> int | None:
    if not path or not Path(path).is_file():
        return None
    return max((row.length for row in load_jsonl(path)), default=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate micro boundary precision/recall/F1 by absolute byte offset"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evaluation-report")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--bins", default="1-32,33-128,129-256,257-512,513-inf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    threshold = select_threshold(checkpoint, args.evaluation_report)
    loader = make_loader(
        args.data,
        args.batch_size,
        model.config.max_length,
        False,
        args.workers,
        args.seed,
    )
    bins = parse_bins(args.bins)
    counts = {
        bin_name(lower, upper): {"tp": 0, "fp": 0, "fn": 0, "gold": 0, "predicted": 0}
        for lower, upper in bins
    }
    message_lengths: list[int] = []
    model.eval()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            probabilities = torch.sigmoid(
                model(batch["bytes"], batch["mask"])["boundary_logits"]
            ).detach().cpu().numpy()
            gold = cpu_batch["boundary_labels"].numpy()
            for index, length_tensor in enumerate(cpu_batch["lengths"]):
                length = int(length_tensor)
                message_lengths.append(length)
                predicted = set(np.flatnonzero(probabilities[index, :length] >= threshold))
                reference = set(np.flatnonzero(gold[index, :length] > 0.5))
                predicted.discard(0)
                reference.discard(0)
                for lower, upper in bins:
                    name = bin_name(lower, upper)
                    in_bin = lambda value: value >= lower and (upper is None or value <= upper)
                    pred_bin = {value for value in predicted if in_bin(value)}
                    gold_bin = {value for value in reference if in_bin(value)}
                    counts[name]["tp"] += len(pred_bin & gold_bin)
                    counts[name]["fp"] += len(pred_bin - gold_bin)
                    counts[name]["fn"] += len(gold_bin - pred_bin)
                    counts[name]["gold"] += len(gold_bin)
                    counts[name]["predicted"] += len(pred_bin)

    training_config = checkpoint.get("training_config", {})
    training_path = training_config.get("data", {}).get("train")
    train_max = dataset_max_length(training_path)
    result = {
        "model": args.model_name,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "data": args.data,
        "threshold": threshold,
        "position_embedding": "learned absolute table",
        "training_data": training_path,
        "training_max_length": train_max,
        "test_min_length": min(message_lengths),
        "test_max_length": max(message_lengths),
        "test_uses_untrained_absolute_indices": (
            train_max is not None and max(message_lengths) > train_max
        ),
        "bins": {name: summarize_counts(value) for name, value in counts.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
