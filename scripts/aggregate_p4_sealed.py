from __future__ import annotations

import argparse
import json
from pathlib import Path

from mambapre.reporting import load_json, summarize_values


METRICS = (
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "exact_field_f1",
    "message_perfection",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the frozen P4 sealed pilot")
    parser.add_argument("--root", default="results/p4_sealed")
    parser.add_argument("--seeds", default="1337,2027,3407,4701,9001")
    parser.add_argument("--output", default="results/p4_sealed/aggregate.json")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    root = Path(args.root)
    loaded = {"role_guided": {}, "matched_transformer": {}}
    missing = []
    for model in loaded:
        for seed in seeds:
            path = root / f"seed{seed}" / model / "sealed_opcua.json"
            if path.is_file():
                loaded[model][seed] = load_json(path)
            else:
                missing.append(path.as_posix())
    paired = [
        seed for seed in seeds
        if seed in loaded["role_guided"] and seed in loaded["matched_transformer"]
    ]
    result = {
        "frozen_evaluation": True,
        "freeze_manifest": "results/p4/freeze_manifest.json",
        "test_manifest": "data/processed/p4_sealed/manifest.json",
        "test_scope": "five-message, three-capture OPC UA sealed pilot",
        "requested_seeds": seeds,
        "missing": missing,
        "models": {
            model: {
                metric: summarize_values(
                    [float(loaded[model][seed]["overall"][metric]) for seed in seeds if seed in loaded[model]]
                )
                for metric in METRICS
                if loaded[model]
            }
            for model in loaded
        },
        "paired_role_guided_minus_transformer": {
            "paired_seeds": paired,
            "metrics": {
                metric: summarize_values(
                    [
                        float(loaded["role_guided"][seed]["overall"][metric])
                        - float(loaded["matched_transformer"][seed]["overall"][metric])
                        for seed in paired
                    ]
                )
                for metric in METRICS
            } if paired else {},
        },
        "limitations": [
            "the sealed set contains five messages, so results are pilot evidence",
            "all captures are hosted by one public repository",
            "no hyperparameter or threshold changes are permitted after this evaluation",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
