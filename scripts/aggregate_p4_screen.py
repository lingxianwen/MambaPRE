from __future__ import annotations

import argparse
import json
from pathlib import Path

from mambapre.reporting import load_json, summarize_values


MODELS = ("role_guided", "boundary_multiscale", "matched_transformer")
DATASETS = (
    "short_test_novel",
    "long_envelope_diagnostic",
    "viewed_fins_diagnostic_only",
)
METRICS = (
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "exact_field_f1",
    "message_perfection",
)


def metric_root(report: dict) -> dict:
    return report["test"] if "test" in report else report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the two-seed P4 screen")
    parser.add_argument("--root", default="results/p4_screen")
    parser.add_argument("--seeds", default="1337,2027")
    parser.add_argument("--output", default="results/p4_screen/aggregate.json")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    root = Path(args.root)
    loaded = {dataset: {model: {} for model in MODELS} for dataset in DATASETS}
    missing = []
    for dataset in DATASETS:
        for model in MODELS:
            for seed in seeds:
                path = root / f"seed{seed}" / model / f"{dataset}.json"
                if path.is_file():
                    loaded[dataset][model][seed] = metric_root(load_json(path))
                else:
                    missing.append(path.as_posix())

    result = {
        "requested_seeds": seeds,
        "sealed_test_used": False,
        "warning": "viewed FINS is training-overlapping diagnostic data and cannot support a generalization claim",
        "missing": missing,
        "datasets": {},
    }
    for dataset in DATASETS:
        item = {"models": {}, "paired_differences": {}}
        for model in MODELS:
            reports = loaded[dataset][model]
            item["models"][model] = {
                metric: summarize_values(
                    [float(reports[seed]["overall"][metric]) for seed in seeds if seed in reports]
                )
                for metric in METRICS
                if reports
            }
        for left, right in (
            ("role_guided", "matched_transformer"),
            ("boundary_multiscale", "matched_transformer"),
            ("boundary_multiscale", "role_guided"),
        ):
            paired = [
                seed for seed in seeds
                if seed in loaded[dataset][left] and seed in loaded[dataset][right]
            ]
            item["paired_differences"][f"{left}_minus_{right}"] = {
                "paired_seeds": paired,
                "metrics": {
                    metric: summarize_values(
                        [
                            float(loaded[dataset][left][seed]["overall"][metric])
                            - float(loaded[dataset][right][seed]["overall"][metric])
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
