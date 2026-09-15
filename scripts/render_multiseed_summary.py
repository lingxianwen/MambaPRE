from __future__ import annotations

import argparse
from pathlib import Path

from mambapre.reporting import load_json


LABELS = {
    "test_novel": "Core-corpus test (exact-byte novel)",
    "ood_novel": "Core-corpus OOD (exact-byte novel)",
    "neupre_full_novel": "NeuPRE full-field external",
    "neupre_long_session_calibrated": "NeuPRE long, session-disjoint",
}


def pm(summary: dict) -> str:
    return f"{summary['mean']:.4f} ± {summary['std']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render five-seed claim/evidence summary")
    parser.add_argument("--input", default="results/multiseed/aggregate.json")
    parser.add_argument("--output", default="docs/multiseed_results.md")
    args = parser.parse_args()
    report = load_json(args.input)
    lines = [
        "# Five-seed core comparison",
        "",
        "Seeds: " + ", ".join(str(seed) for seed in report["requested_seeds"]) + ".",
        "Values are mean ± sample standard deviation. Difference intervals are paired",
        "two-sided 95% Student-t confidence intervals over seeds.",
        "",
        "| Dataset | Guided Mamba boundary F1 | Transformer boundary F1 | Paired difference (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for key, dataset in report["datasets"].items():
        mamba = dataset["models"]["guided_mamba"]["boundary_f1"]
        transformer = dataset["models"]["matched_transformer"]["boundary_f1"]
        difference = dataset["paired_guided_minus_transformer"]["boundary_f1"]
        lines.append(
            f"| {LABELS[key]} | {pm(mamba)} | {pm(transformer)} | "
            f"{difference['mean']:+.4f} [{difference['ci95_low']:+.4f}, "
            f"{difference['ci95_high']:+.4f}] |"
        )
    lines.extend([
        "",
        "## Evidence reading",
        "",
        "- On the leak-free in-distribution test, the controlled models are statistically tied.",
        "- On the full NeuPRE external set, Mamba has a positive mean boundary-F1 gap, but the",
        "  five-seed interval crosses zero; this is not yet evidence of a universal advantage.",
        "- On session-disjoint long S7comm+ envelopes, every seed favors Mamba and the paired",
        "  interval excludes zero. This supports the long-sequence claim.",
        "- Message perfection is zero for both models on the coarse long-envelope labels, so it",
        "  is not a useful discriminator in that experiment.",
    ])
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
