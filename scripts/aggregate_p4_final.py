from __future__ import annotations

import argparse
import json
from pathlib import Path

from mambapre.reporting import load_json, summarize_values


MODELS = ("role_guided", "matched_transformer")
DATASETS = ("short_test_novel", "long_envelope_diagnostic", "viewed_fins_diagnostic_only")
METRICS = (
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "exact_field_f1",
    "message_perfection",
)


def root(report: dict) -> dict:
    return report["test"] if "test" in report else report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the frozen P4 final pair")
    parser.add_argument("--results-root", default="results/p4_screen")
    parser.add_argument("--seeds", default="1337,2027,3407,4701,9001")
    parser.add_argument("--output", default="results/p4_final/aggregate_diagnostic.json")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    base = Path(args.results_root)
    loaded = {dataset: {model: {} for model in MODELS} for dataset in DATASETS}
    missing = []
    for dataset in DATASETS:
        for model in MODELS:
            for seed in seeds:
                path = base / f"seed{seed}" / model / f"{dataset}.json"
                if path.is_file():
                    loaded[dataset][model][seed] = root(load_json(path))
                else:
                    missing.append(path.as_posix())

    result = {
        "requested_seeds": seeds,
        "primary": "role_guided",
        "comparator": "matched_transformer",
        "sealed_test_used": False,
        "missing": missing,
        "datasets": {},
    }
    for dataset in DATASETS:
        item = {"models": {}, "paired_role_guided_minus_transformer": {}}
        for model in MODELS:
            reports = loaded[dataset][model]
            item["models"][model] = {
                metric: summarize_values(
                    [float(reports[seed]["overall"][metric]) for seed in seeds if seed in reports]
                )
                for metric in METRICS
                if reports
            }
        paired = [
            seed for seed in seeds
            if seed in loaded[dataset]["role_guided"]
            and seed in loaded[dataset]["matched_transformer"]
        ]
        item["paired_role_guided_minus_transformer"] = {
            "paired_seeds": paired,
            "metrics": {
                metric: summarize_values(
                    [
                        float(loaded[dataset]["role_guided"][seed]["overall"][metric])
                        - float(loaded[dataset]["matched_transformer"][seed]["overall"][metric])
                        for seed in paired
                    ]
                )
                for metric in METRICS
            } if paired else {},
        }
        result["datasets"][dataset] = item

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
