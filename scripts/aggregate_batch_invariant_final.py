#!/usr/bin/env python3
"""Aggregate final batch-invariant guided/unguided runs with unchanged controls."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = (1337, 2027, 3407, 4701, 9001)
T95_DF4 = 2.7764451051977987
DATASETS = {
    "id_novel": "test_novel.json",
    "source_held_out": "ood_novel.json",
    "external_full": "neupre_full_novel.json",
    "long_envelope": "neupre_long_session_calibrated.json",
    "strict_fins": "long_full_strict_calibrated.json",
}


def read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    return report["test"] if "test" in report else report


def summarize(values: list[float]) -> dict[str, object]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    half = T95_DF4 * std / math.sqrt(len(values))
    return {
        "values": values,
        "mean": mean,
        "sample_std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def guided_path(seed: int, dataset: str) -> Path:
    return RESULTS / "batch_invariant_revision" / "retrained" / f"seed{seed}" / DATASETS[dataset]


def transformer_path(seed: int, dataset: str) -> Path:
    return RESULTS / "multiseed" / f"seed{seed}" / "matched_transformer" / DATASETS[dataset]


def variant_path(variant: str, seed: int, dataset: str) -> Path:
    if variant == "cnn":
        return RESULTS / "multiseed" / f"seed{seed}" / "matched_cnn" / DATASETS[dataset]
    if variant == "bimamba_only":
        return RESULTS / "bimamba_ablation_multiseed" / f"seed{seed}" / DATASETS[dataset]
    if variant == "fixed_dual_stream":
        names = {
            "external_full": "external_full.json",
            "long_envelope": "long_envelope.json",
            "strict_fins": "strict_fins.json",
        }
        return RESULTS / "revision_core" / f"seed{seed}" / "fixed_fusion" / names[dataset]
    if variant == "unguided_learned_fusion":
        return RESULTS / "batch_invariant_revision" / "unguided_retrained" / f"seed{seed}" / DATASETS[dataset]
    if variant == "full_mambapre":
        return guided_path(seed, dataset)
    raise KeyError(variant)


def f1(path: Path) -> float:
    return float(read(path)["overall"]["boundary_f1"])


def overall_metric(path: Path, metric: str) -> float:
    return float(read(path)["overall"][metric])


def paired(left: list[float], right: list[float]) -> dict[str, object]:
    return summarize([a - b for a, b in zip(left, right)])


def main() -> None:
    output: dict[str, object] = {
        "seeds": list(SEEDS),
        "main": {},
        "auxiliary_heads": {},
        "ablations": {},
        "gate": {},
    }
    for dataset in DATASETS:
        guided = [f1(guided_path(seed, dataset)) for seed in SEEDS]
        transformer = [f1(transformer_path(seed, dataset)) for seed in SEEDS]
        output["main"][dataset] = {
            "mambapre": summarize(guided),
            "transformer": summarize(transformer),
            "paired_mambapre_minus_transformer": paired(guided, transformer),
        }

        output["auxiliary_heads"][dataset] = {
            "byte_role_accuracy": summarize([
                overall_metric(guided_path(seed, dataset), "byte_role_accuracy")
                for seed in SEEDS
            ]),
            "byte_type_accuracy": summarize([
                overall_metric(guided_path(seed, dataset), "byte_type_accuracy")
                for seed in SEEDS
            ]),
        }

    ablation_datasets = ("long_envelope", "external_full", "strict_fins")
    variants = ("cnn", "bimamba_only", "fixed_dual_stream", "unguided_learned_fusion", "full_mambapre")
    raw: dict[str, dict[str, list[float]]] = {}
    for variant in variants:
        raw[variant] = {}
        output["ablations"][variant] = {}
        for dataset in ablation_datasets:
            values = [f1(variant_path(variant, seed, dataset)) for seed in SEEDS]
            raw[variant][dataset] = values
            output["ablations"][variant][dataset] = summarize(values)

    output["ablation_paired"] = {}
    for variant in variants[:-1]:
        output["ablation_paired"][f"{variant}_minus_full"] = {
            dataset: paired(raw[variant][dataset], raw["full_mambapre"][dataset])
            for dataset in ablation_datasets
        }
    output["ablation_paired"]["fixed_minus_bimamba_only"] = {
        dataset: paired(raw["fixed_dual_stream"][dataset], raw["bimamba_only"][dataset])
        for dataset in ablation_datasets
    }

    protocol_macro_values = []
    for seed in SEEDS:
        report = read(guided_path(seed, "external_full"))
        protocol_macro_values.append(statistics.fmean(
            float(value["boundary_f1"]) for value in report["by_protocol"].values()
        ))
    transformer_macro_values = []
    for seed in SEEDS:
        report = read(transformer_path(seed, "external_full"))
        transformer_macro_values.append(statistics.fmean(
            float(value["boundary_f1"]) for value in report["by_protocol"].values()
        ))
    output["external_protocol_macro"] = {
        "mambapre": summarize(protocol_macro_values),
        "transformer": summarize(transformer_macro_values),
    }

    guided_per_layer: dict[int, list[float]] = {layer: [] for layer in range(4)}
    unguided_per_layer: dict[int, list[float]] = {layer: [] for layer in range(4)}
    for seed in SEEDS:
        guided_analysis = read(guided_path(seed, "long_envelope"))["gate_analysis"]
        unguided_analysis = read(
            variant_path("unguided_learned_fusion", seed, "long_envelope")
        )["gate_analysis"]
        for row in guided_analysis["per_layer"]:
            guided_per_layer[int(row["layer"])].append(float(row["separation"]))
        for row in unguided_analysis["per_layer"]:
            unguided_per_layer[int(row["layer"])].append(float(row["separation"]))
    output["gate"] = {
        "guided": {
            f"block_{layer + 1}": summarize(values)
            for layer, values in guided_per_layer.items()
        },
        "unguided": {
            f"block_{layer + 1}": summarize(values)
            for layer, values in unguided_per_layer.items()
        },
        "paired_guided_minus_unguided": {
            f"block_{layer + 1}": paired(guided_per_layer[layer], unguided_per_layer[layer])
            for layer in guided_per_layer
        },
    }

    destination = RESULTS / "batch_invariant_revision" / "final_aggregate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
