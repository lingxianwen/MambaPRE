from __future__ import annotations

import argparse
import json
from pathlib import Path

from mambapre.reporting import METRICS, load_json, summarize_values


def metric_root(report: dict) -> dict:
    return report["test"] if "test" in report else report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paired Mamba/Transformer seeds")
    parser.add_argument("--results-root", default="results/multiseed")
    parser.add_argument("--seeds", default="1337,2027,3407,4701,9001")
    parser.add_argument("--output", default="results/multiseed/aggregate.json")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    datasets = (
        "test_novel",
        "ood_novel",
        "neupre_full_novel",
        "neupre_long_session_calibrated",
        "long_full_strict_calibrated",
        "long_full_public_real",
        "long_full_controlled",
    )
    models = ("guided_mamba", "matched_transformer", "matched_cnn")
    loaded: dict[str, dict[str, dict[int, dict]]] = {
        dataset: {model: {} for model in models} for dataset in datasets
    }
    missing = []
    root = Path(args.results_root)
    for seed in seeds:
        for model in models:
            for dataset in datasets:
                path = root / f"seed{seed}" / model / f"{dataset}.json"
                if path.exists():
                    loaded[dataset][model][seed] = metric_root(load_json(path))
                else:
                    missing.append(str(path))

    aggregate = {"requested_seeds": seeds, "missing": missing, "datasets": {}}
    for dataset in datasets:
        dataset_result = {
            "models": {},
            "paired_guided_minus_transformer": {},
            "paired_guided_minus_cnn": {},
        }
        for model in models:
            reports = loaded[dataset][model]
            dataset_result["models"][model] = {
                metric: summarize_values(
                    [float(reports[seed]["overall"][metric]) for seed in seeds if seed in reports]
                )
                for metric in METRICS
                if reports
            }
            protocols = sorted(
                set.intersection(
                    *(set(report.get("by_protocol", {})) for report in reports.values())
                )
            ) if reports else []
            dataset_result["models"][model]["by_protocol"] = {
                protocol: {
                    metric: summarize_values(
                        [
                            float(reports[seed]["by_protocol"][protocol][metric])
                            for seed in seeds
                            if seed in reports
                        ]
                    )
                    for metric in METRICS
                }
                for protocol in protocols
            }
        paired_seeds = [
            seed
            for seed in seeds
            if seed in loaded[dataset]["guided_mamba"]
            and seed in loaded[dataset]["matched_transformer"]
        ]
        for metric in METRICS:
            if paired_seeds:
                differences = [
                    float(loaded[dataset]["guided_mamba"][seed]["overall"][metric])
                    - float(loaded[dataset]["matched_transformer"][seed]["overall"][metric])
                    for seed in paired_seeds
                ]
                summary = summarize_values(differences)
                summary["seeds"] = paired_seeds
                dataset_result["paired_guided_minus_transformer"][metric] = summary
        paired_cnn_seeds = [
            seed
            for seed in seeds
            if seed in loaded[dataset]["guided_mamba"]
            and seed in loaded[dataset]["matched_cnn"]
        ]
        for metric in METRICS:
            if paired_cnn_seeds:
                differences = [
                    float(loaded[dataset]["guided_mamba"][seed]["overall"][metric])
                    - float(loaded[dataset]["matched_cnn"][seed]["overall"][metric])
                    for seed in paired_cnn_seeds
                ]
                summary = summarize_values(differences)
                summary["seeds"] = paired_cnn_seeds
                dataset_result["paired_guided_minus_cnn"][metric] = summary
        aggregate["datasets"][dataset] = dataset_result

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
