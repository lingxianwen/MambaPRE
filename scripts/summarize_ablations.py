from __future__ import annotations

import argparse
import json
from pathlib import Path

from mambapre.reporting import load_json


def root_metrics(path: Path) -> dict:
    report = load_json(path)
    return report["test"]["overall"] if "test" in report else report["overall"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize seed-1337 architecture ablations")
    parser.add_argument("--ablation-dir", default="results/ablations_seed1337")
    parser.add_argument("--output-json", default="results/ablations_seed1337/comparison.json")
    parser.add_argument("--output-md", default="docs/ablation_results_seed1337.md")
    args = parser.parse_args()
    root = Path(args.ablation_dir)
    methods = {
        "Unidirectional Mamba": root / "unidirectional_mamba",
        "Bidirectional Mamba": root / "bidirectional_mamba",
        "Dual stream, fixed fusion": root / "dual_fixed",
        "Dual stream, unguided gate": root / "dual_unguided",
        "Guided gate, no semantic auxiliary loss": root / "guided_no_semantic_aux",
    }
    datasets = {
        "test_novel": "test_novel.json",
        "ood_novel": "ood_novel.json",
        "long_session": "neupre_long_session_calibrated.json",
    }
    rows = {}
    for name, directory in methods.items():
        rows[name] = {
            dataset: root_metrics(directory / filename) for dataset, filename in datasets.items()
        }
    rows["Guided dual-stream Mamba-PRE"] = {
        "test_novel": root_metrics(Path("results/server_guided_dedup_seed1337/test_novel.json")),
        "ood_novel": root_metrics(Path("results/server_guided_dedup_seed1337/ood_novel.json")),
        "long_session": root_metrics(
            Path("results/neupre_external_seed1337/guided_mamba_long_envelope_session_calibrated.json")
        ),
    }
    report = {
        "seed": 1337,
        "status": "mechanism-screening ablation; repeat selected contrasts over five seeds",
        "methods": rows,
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Architecture ablations (seed 1337)",
        "",
        "These are mechanism-screening results from one seed. They are not yet the final",
        "statistical ablation table.",
        "",
        "| Variant | Test boundary F1 | OOD boundary F1 | Long session boundary F1 | Long exact-field F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in rows.items():
        lines.append(
            f"| {name} | {values['test_novel']['boundary_f1']:.4f} | "
            f"{values['ood_novel']['boundary_f1']:.4f} | "
            f"{values['long_session']['boundary_f1']:.4f} | "
            f"{values['long_session']['exact_field_f1']:.4f} |"
        )
    lines.extend([
        "",
        "At this seed, bidirectionality improves the long-session boundary F1 from 0.6031",
        "to 0.7121. Dual streams with an unguided learned gate reach 0.7434, and structural",
        "gate guidance reaches 0.7866. Removing role/type auxiliary losses reduces it to",
        "0.6256. The selected guided-versus-unguided and auxiliary-loss contrasts should be",
        "repeated across seeds before making a causal claim.",
    ])
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

