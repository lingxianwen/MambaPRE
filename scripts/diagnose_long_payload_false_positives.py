from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mambapre.constants import ROLE_TO_ID
from mambapre.engine import load_checkpoint, make_loader, move_batch, resolve_device
from mambapre.reporting import summarize_values


DISTANCE_BINS = (
    ("001-030", 1, 30),
    ("031-128", 31, 128),
    ("129-256", 129, 256),
    ("257-512", 257, 512),
    ("513+", 513, 10**9),
)


def distance_bin(distance: int) -> str:
    for name, low, high in DISTANCE_BINS:
        if low <= distance <= high:
            return name
    raise ValueError(f"invalid payload distance: {distance}")


@torch.no_grad()
def evaluate_one(
    checkpoint_path: Path,
    threshold: float,
    data_path: Path,
    device: torch.device,
    batch_size: int,
) -> dict:
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    loader = make_loader(
        str(data_path),
        batch_size,
        model.config.max_length,
        False,
        0,
        int(checkpoint["training_config"].get("seed", 1337)),
    )
    counts = defaultdict(lambda: {"eligible_bytes": 0, "false_positives": 0})
    beyond_radius = {"eligible_bytes": 0, "false_positives": 0}
    payload_starts = []
    gold_boundary_offsets = set()
    messages = 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        probabilities = torch.sigmoid(model(batch["bytes"], batch["mask"])["boundary_logits"])
        predictions = probabilities >= threshold
        for index, length_tensor in enumerate(cpu_batch["lengths"]):
            length = int(length_tensor)
            gold = cpu_batch["boundary_labels"][index, :length].numpy() > 0.5
            roles = cpu_batch["role_labels"][index, :length].numpy()
            predicted = predictions[index, :length].cpu().numpy()
            payload_positions = np.flatnonzero(roles == ROLE_TO_ID["payload"])
            if not len(payload_positions):
                raise ValueError(f"{cpu_batch['ids'][index]} has no payload annotation")
            payload_start = int(payload_positions[0])
            payload_starts.append(payload_start)
            gold_boundary_offsets.update(np.flatnonzero(gold).tolist())
            for position in payload_positions:
                position = int(position)
                if gold[position]:
                    continue
                name = distance_bin(position - payload_start)
                counts[name]["eligible_bytes"] += 1
                counts[name]["false_positives"] += int(predicted[position])
                if position - payload_start > 30:
                    beyond_radius["eligible_bytes"] += 1
                    beyond_radius["false_positives"] += int(predicted[position])
            messages += 1

    rows = []
    for name, _, _ in DISTANCE_BINS:
        eligible = counts[name]["eligible_bytes"]
        false_positives = counts[name]["false_positives"]
        rows.append(
            {
                "distance_from_payload_start": name,
                "eligible_payload_bytes": eligible,
                "false_positives": false_positives,
                "false_positives_per_10000_bytes": 10000 * false_positives / eligible,
            }
        )
    return {
        "messages": messages,
        "threshold": threshold,
        "payload_start_offsets": sorted(set(payload_starts)),
        "gold_boundary_offsets": sorted(gold_boundary_offsets),
        "beyond_cnn_radius": {
            **beyond_radius,
            "false_positives_per_10000_bytes": (
                10000
                * beyond_radius["false_positives"]
                / beyond_radius["eligible_bytes"]
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose false boundaries beyond a CNN's finite receptive field"
    )
    parser.add_argument("--results-root", default="results/multiseed")
    parser.add_argument(
        "--data",
        default="data/processed/neupre_pdml/test_long_envelope_session_disjoint_novel.jsonl",
    )
    parser.add_argument("--seeds", default="1337,2027,3407,4701,9001")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--output", default="results/diagnostics/long_payload_false_positives.json"
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    seeds = [int(value) for value in args.seeds.split(",")]
    models = {
        "Mamba-PRE": "guided_mamba",
        "Supervised CNN": "matched_cnn",
    }
    run_names = {
        "Mamba-PRE": "dual_bimamba_guided_dedup",
        "Supervised CNN": "cnn_param_matched_dedup",
    }
    device = resolve_device(args.device)
    per_seed = {name: {} for name in models}
    for display_name, result_name in models.items():
        for seed in seeds:
            evaluation_path = (
                root / f"seed{seed}" / result_name / "neupre_long_session_calibrated.json"
            )
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            checkpoint = Path("runs") / f"{run_names[display_name]}_seed{seed}" / "best.pt"
            per_seed[display_name][str(seed)] = evaluate_one(
                checkpoint,
                float(evaluation["selected_threshold"]),
                Path(args.data),
                device,
                args.batch_size,
            )

    aggregate = {}
    paired = {}
    for bin_name, _, _ in DISTANCE_BINS:
        aggregate[bin_name] = {}
        for model_name in models:
            values = [
                next(
                    row["false_positives_per_10000_bytes"]
                    for row in per_seed[model_name][str(seed)]["rows"]
                    if row["distance_from_payload_start"] == bin_name
                )
                for seed in seeds
            ]
            aggregate[bin_name][model_name] = summarize_values(values)
        differences = [
            next(
                row["false_positives_per_10000_bytes"]
                for row in per_seed["Mamba-PRE"][str(seed)]["rows"]
                if row["distance_from_payload_start"] == bin_name
            )
            - next(
                row["false_positives_per_10000_bytes"]
                for row in per_seed["Supervised CNN"][str(seed)]["rows"]
                if row["distance_from_payload_start"] == bin_name
            )
            for seed in seeds
        ]
        paired[bin_name] = summarize_values(differences)

    beyond_radius_aggregate = {}
    for model_name in models:
        values = [
            per_seed[model_name][str(seed)]["beyond_cnn_radius"][
                "false_positives_per_10000_bytes"
            ]
            for seed in seeds
        ]
        beyond_radius_aggregate[model_name] = summarize_values(values)
    beyond_radius_paired = summarize_values(
        [
            per_seed["Mamba-PRE"][str(seed)]["beyond_cnn_radius"][
                "false_positives_per_10000_bytes"
            ]
            - per_seed["Supervised CNN"][str(seed)]["beyond_cnn_radius"][
                "false_positives_per_10000_bytes"
            ]
            for seed in seeds
        ]
    )

    output = {
        "diagnostic_question": (
            "Does Mamba reduce false boundaries after the 61-byte CNN receptive field?"
        ),
        "scope": (
            "Opaque payload non-boundary bytes only; distance is measured from the first "
            "payload byte. Gold payload-start boundaries are excluded."
        ),
        "cnn_receptive_field": {
            "width_bytes": 61,
            "radius_bytes": 30,
            "derivation": "1 + 2 convolutions * (kernel_size - 1) * sum(1,2,4,8)",
        },
        "seeds": seeds,
        "distance_bins_fixed_before evaluation": [name for name, _, _ in DISTANCE_BINS],
        "per_seed": per_seed,
        "aggregate_false_positives_per_10000_bytes": aggregate,
        "paired_mamba_minus_cnn": paired,
        "primary_beyond_radius_false_positives_per_10000_bytes": beyond_radius_aggregate,
        "primary_paired_mamba_minus_cnn": beyond_radius_paired,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "primary_beyond_radius": beyond_radius_aggregate,
                "primary_paired": beyond_radius_paired,
                "descriptive_bins": aggregate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
