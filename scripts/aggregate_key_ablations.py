from __future__ import annotations

import argparse
import json
from pathlib import Path

from mambapre.reporting import METRICS, load_json, summarize_values


DATASETS = (
    "test_novel",
    "ood_novel",
    "neupre_full_novel",
    "neupre_long_session_calibrated",
    "long_full_strict_calibrated",
    "long_full_public_real",
    "long_full_controlled",
)
VARIANTS = ("full_guided", "dual_unguided", "guided_no_semantic_aux")


def metric_root(report: dict) -> dict:
    return report["test"] if "test" in report else report


def report_path(
    results_root: Path, main_results_root: Path, seed: int, variant: str, dataset: str
) -> Path:
    if variant == "full_guided":
        return main_results_root / f"seed{seed}" / "guided_mamba" / f"{dataset}.json"
    return results_root / f"seed{seed}" / variant / f"{dataset}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate five-seed guided/gating/semantic-loss ablations"
    )
    parser.add_argument("--results-root", default="results/key_ablation_multiseed")
    parser.add_argument("--main-results-root", default="results/multiseed")
    parser.add_argument("--seeds", default="1337,2027,3407,4701,9001")
    parser.add_argument(
        "--output", default="results/key_ablation_multiseed/aggregate.json"
    )
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",")]
    results_root = Path(args.results_root)
    main_results_root = Path(args.main_results_root)
    loaded: dict[str, dict[str, dict[int, dict]]] = {
        dataset: {variant: {} for variant in VARIANTS} for dataset in DATASETS
    }
    missing: list[str] = []

    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in seeds:
                path = report_path(results_root, main_results_root, seed, variant, dataset)
                if path.exists():
                    loaded[dataset][variant][seed] = metric_root(load_json(path))
                else:
                    missing.append(str(path))

    aggregate: dict = {
        "requested_seeds": seeds,
        "design": {
            "full_guided": "guided gate + role/type auxiliary losses",
            "dual_unguided": "learned gate without structural gate supervision; role/type auxiliary losses retained",
            "guided_no_semantic_aux": "guided gate retained; role/type auxiliary losses set to zero",
            "pairing": "same seed, training data, architecture, optimizer, and schedule",
        },
        "missing": missing,
        "datasets": {},
    }

    for dataset in DATASETS:
        dataset_result: dict = {"variants": {}, "paired_full_minus_ablation": {}}
        for variant in VARIANTS:
            reports = loaded[dataset][variant]
            dataset_result["variants"][variant] = {
                metric: summarize_values(
                    [float(reports[seed]["overall"][metric]) for seed in seeds if seed in reports]
                )
                for metric in METRICS
                if reports
            }

        for ablation in ("dual_unguided", "guided_no_semantic_aux"):
            paired_seeds = [
                seed
                for seed in seeds
                if seed in loaded[dataset]["full_guided"]
                and seed in loaded[dataset][ablation]
            ]
            comparison = {"paired_seeds": paired_seeds, "metrics": {}}
            for metric in METRICS:
                if paired_seeds:
                    comparison["metrics"][metric] = summarize_values(
                        [
                            float(loaded[dataset]["full_guided"][seed]["overall"][metric])
                            - float(loaded[dataset][ablation][seed]["overall"][metric])
                            for seed in paired_seeds
                        ]
                    )
            dataset_result["paired_full_minus_ablation"][ablation] = comparison
        aggregate["datasets"][dataset] = dataset_result

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
