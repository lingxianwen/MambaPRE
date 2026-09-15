#!/usr/bin/env python3
"""Compare batch-invariant position-feature results with the published run files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEEDS = (1337, 2027, 3407, 4701, 9001)
DATASETS = {
    "id_novel": ("test_novel.json", "overall"),
    "source_held_out": ("ood_novel.json", "overall"),
    "external_full": ("neupre_full_novel.json", "overall"),
    "long_envelope": ("neupre_long_session_calibrated.json", "test.overall"),
    "strict_fins": ("long_full_strict_calibrated.json", "test.overall"),
}


def read_f1(path: Path, dotted_key: str) -> float:
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in dotted_key.split("."):
        value = value[key]
    return float(value["boundary_f1"])


def describe(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=float)
    return {
        "values": values,
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pretrained_reval", "retrained"), required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    revised_root = args.results_root / "batch_invariant_revision" / args.mode
    old_root = args.results_root / "multiseed"
    summary: dict[str, object] = {"mode": args.mode, "seeds": list(SEEDS), "datasets": {}}
    for dataset, (filename, key) in DATASETS.items():
        old = [read_f1(old_root / f"seed{seed}" / "guided_mamba" / filename, key) for seed in SEEDS]
        revised = [read_f1(revised_root / f"seed{seed}" / filename, key) for seed in SEEDS]
        delta = [new - previous for new, previous in zip(revised, old)]
        summary["datasets"][dataset] = {
            "old": describe(old),
            "revised": describe(revised),
            "paired_delta": describe(delta),
            "max_abs_seed_delta": max(abs(value) for value in delta),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
