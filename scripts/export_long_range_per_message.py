from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from mambapre.engine import load_checkpoint, make_loader, move_batch, resolve_device
from mambapre.metrics import BoundaryAccumulator


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Export controlled-long-range per-message predictions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--data", default="data/processed/controlled_long_range/test.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    model.eval()
    report = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    report["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    report["model_config"] = checkpoint["model_config"]
    threshold = float(report["selected_threshold"])
    loader = make_loader(args.data, 4, model.config.max_length, False, 0, args.seed)
    metadata = {}
    with Path(args.data).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata[row["id"]] = row
    overall = BoundaryAccumulator()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "model", "seed", "message_id", "distance", "length",
            "target_boundary", "threshold", "predicted_boundaries",
            "tp", "fp", "fn", "top1_boundary", "top1_correct",
            "top1_absolute_error", "target_probability", "target_rank",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            probabilities = torch.sigmoid(model(batch["bytes"], batch["mask"])["boundary_logits"]).cpu().numpy()
            for index, length_tensor in enumerate(cpu_batch["lengths"]):
                length = int(length_tensor)
                message_id = cpu_batch["ids"][index]
                row = metadata[message_id]
                gold = {int(row["target_boundary"])}
                target = int(row["target_boundary"])
                valid_probabilities = probabilities[index, 1:length]
                top1_boundary = int(np.argmax(valid_probabilities)) + 1
                target_probability = float(probabilities[index, target])
                target_rank = 1 + int(np.sum(valid_probabilities > target_probability))
                predicted = set(np.flatnonzero(probabilities[index, :length] >= threshold).tolist())
                predicted.discard(0)
                overall.add(predicted, gold, length)
                writer.writerow({
                    "model": args.model,
                    "seed": args.seed,
                    "message_id": message_id,
                    "distance": row["dependency_distance"],
                    "length": length,
                    "target_boundary": row["target_boundary"],
                    "threshold": threshold,
                    "predicted_boundaries": " ".join(map(str, sorted(predicted))),
                    "tp": len(predicted & gold),
                    "fp": len(predicted - gold),
                    "fn": len(gold - predicted),
                    "top1_boundary": top1_boundary,
                    "top1_correct": int(top1_boundary == target),
                    "top1_absolute_error": abs(top1_boundary - target),
                    "target_probability": target_probability,
                    "target_rank": target_rank,
                })
    expected = report["test"]["overall"]
    observed = overall.summary()
    maximum = max(abs(float(observed[key]) - float(expected[key])) for key in ("boundary_precision", "boundary_recall", "boundary_f1"))
    if maximum > 1e-12:
        raise RuntimeError(f"per-message export mismatch: {maximum}")
    Path(args.evaluation).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"exact {args.model} seed={args.seed} n={observed['n']} threshold={threshold}")


if __name__ == "__main__":
    main()
