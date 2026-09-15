from __future__ import annotations

import argparse
import json
from pathlib import Path

from mambapre.reporting import load_json, summarize_values


DATASETS = (
    "test_novel",
    "ood_novel",
    "neupre_full_novel",
    "neupre_long_session_calibrated",
    "long_full_strict_calibrated",
    "long_full_public_real",
    "long_full_controlled",
)
VARIANTS = ("role_guided", "boundary_multiscale", "hybrid_multiscale")
METRICS = (
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "exact_field_f1",
    "message_perfection",
)


def metric_root(report: dict) -> dict:
    return report["test"] if "test" in report else report


def report_path(p3_root: Path, main_root: Path, seed: int, variant: str, dataset: str) -> Path:
    if variant == "role_guided":
        return main_root / f"seed{seed}" / "guided_mamba" / f"{dataset}.json"
    return p3_root / f"seed{seed}" / variant / f"{dataset}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate P3 multiscale gate experiments")
    parser.add_argument("--p3-root", default="results/p3_multiscale")
    parser.add_argument("--main-root", default="results/multiseed")
    parser.add_argument("--seeds", default="1337,2027,3407,4701,9001")
    parser.add_argument("--output", default="results/p3_multiscale/aggregate.json")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",")]
    p3_root, main_root = Path(args.p3_root), Path(args.main_root)
    loaded = {dataset: {variant: {} for variant in VARIANTS} for dataset in DATASETS}
    missing: list[str] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in seeds:
                path = report_path(p3_root, main_root, seed, variant, dataset)
                if path.exists():
                    loaded[dataset][variant][seed] = metric_root(load_json(path))
                else:
                    missing.append(str(path))

    output = {
        "requested_seeds": seeds,
        "design": {
            "role_guided": "original structural-versus-payload gate target",
            "boundary_multiscale": "soft local target at radii 0/1/3 around field cuts",
            "hybrid_multiscale": "role target plus payload-internal multiscale boundary neighborhoods",
            "controlled_conditions": "same architecture, parameters, data, optimizer, schedule, threshold rules, and paired seeds",
        },
        "missing": missing,
        "datasets": {},
    }
    for dataset in DATASETS:
        item = {"variants": {}, "paired_variant_minus_role_guided": {}}
        for variant in VARIANTS:
            reports = loaded[dataset][variant]
            item["variants"][variant] = {
                metric: summarize_values(
                    [float(reports[seed]["overall"][metric]) for seed in seeds if seed in reports]
                )
                for metric in METRICS
                if reports
            }
        for variant in ("boundary_multiscale", "hybrid_multiscale"):
            paired = [
                seed for seed in seeds
                if seed in loaded[dataset]["role_guided"] and seed in loaded[dataset][variant]
            ]
            item["paired_variant_minus_role_guided"][variant] = {
                "paired_seeds": paired,
                "metrics": {
                    metric: summarize_values([
                        float(loaded[dataset][variant][seed]["overall"][metric])
                        - float(loaded[dataset]["role_guided"][seed]["overall"][metric])
                        for seed in paired
                    ])
                    for metric in METRICS
                } if paired else {},
            }
        output["datasets"][dataset] = item

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
