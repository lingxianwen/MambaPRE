from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


SEEDS = (1337, 2027, 3407, 4701, 9001)
DATASETS = {
    "test_novel": ("test_novel.json", "overall"),
    "ood_novel": ("ood_novel.json", "overall"),
    "long_envelope": ("neupre_long_session_calibrated.json", "test"),
    "strict_fins": ("long_full_strict_calibrated.json", "test"),
}
T95_DF4 = 2.7764451051977987


def summary(values: list[float]) -> dict[str, object]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = T95_DF4 * std / math.sqrt(len(values)) if len(values) == 5 else None
    return {
        "n_seeds": len(values),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width if half_width is not None else None,
        "ci95_high": mean + half_width if half_width is not None else None,
        "values": values,
    }


def read_boundary_f1(path: Path, section: str) -> float:
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report[section]
    if section != "overall":
        metrics = metrics["overall"]
    return float(metrics["boundary_f1"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the parameter-matched Bi-Mamba ablation")
    parser.add_argument("--root", default="results/bimamba_ablation_multiseed")
    parser.add_argument("--output", default="results/bimamba_ablation_multiseed/aggregate.json")
    parser.add_argument("--full-root", default="results/multiseed")
    parser.add_argument("--unguided-root", default="results/key_ablation_multiseed")
    args = parser.parse_args()

    root = Path(args.root)
    result: dict[str, object] = {
        "seeds": list(SEEDS),
        "variant": "Bi-Mamba only, d_model=140",
        "parameters": 2_199_274,
        "target_parameters": 2_191_502,
        "parameter_delta_fraction": (2_199_274 - 2_191_502) / 2_191_502,
        "datasets": {},
    }
    missing: list[str] = []
    for dataset, (filename, section) in DATASETS.items():
        values: list[float] = []
        for seed in SEEDS:
            path = root / f"seed{seed}" / filename
            if not path.exists():
                missing.append(str(path))
                continue
            values.append(read_boundary_f1(path, section))
        if values:
            result["datasets"][dataset] = summary(values)

        comparisons: dict[str, object] = {}
        for name, comparison_root, variant in (
            ("full_minus_bimamba", Path(args.full_root), "guided_mamba"),
            ("dual_unguided_minus_bimamba", Path(args.unguided_root), "dual_unguided"),
        ):
            differences: list[float] = []
            for seed in SEEDS:
                bimamba_path = root / f"seed{seed}" / filename
                comparison_path = comparison_root / f"seed{seed}" / variant / filename
                if not bimamba_path.exists() or not comparison_path.exists():
                    missing.extend(str(path) for path in (bimamba_path, comparison_path) if not path.exists())
                    continue
                differences.append(
                    read_boundary_f1(comparison_path, section)
                    - read_boundary_f1(bimamba_path, section)
                )
            if differences:
                comparisons[name] = summary(differences)
        if comparisons:
            result["datasets"][dataset]["paired_comparisons"] = comparisons
    result["missing"] = missing
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if missing:
        raise SystemExit(f"missing {len(missing)} evaluation files; partial aggregate written")


if __name__ == "__main__":
    main()
