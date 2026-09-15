from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "revision"
SEEDS = (1337, 2027, 3407, 4701, 9001)
T95_DF4 = 2.7764451051977987
METRICS = ("boundary_precision", "boundary_recall", "boundary_f1")
PARAMETERS = {
    "A_bimamba_only": 2_199_274,
    "B_fixed_fusion": 1_992_846,
    "C_E_learned_unguided": 2_191_502,
    "D_guided_no_semantic_aux": 2_191_502,
    "F_full": 2_191_502,
}


def read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metric_root(report: dict) -> dict:
    return report["test"] if "test" in report else report


def summarize(values: list[float]) -> dict[str, float | list[float]]:
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


def phase2_path(variant: str, dataset: str, seed: int) -> Path:
    if variant == "A_bimamba_only":
        names = {
            "long_envelope": RESULTS / "bimamba_ablation_multiseed" / f"seed{seed}" / "neupre_long_session_calibrated.json",
            "controlled": RESULTS / "controlled_long_range" / f"seed{seed}" / "bimamba_only" / "evaluation.json",
            "strict_fins": RESULTS / "bimamba_ablation_multiseed" / f"seed{seed}" / "long_full_strict_calibrated.json",
            "external_full": RESULTS / "bimamba_ablation_multiseed" / f"seed{seed}" / "neupre_full_novel.json",
        }
    elif variant == "B_fixed_fusion":
        names = {
            "long_envelope": RESULTS / "revision_core" / f"seed{seed}" / "fixed_fusion" / "long_envelope.json",
            "controlled": RESULTS / "controlled_revision" / f"seed{seed}" / "fixed_fusion" / "evaluation.json",
            "strict_fins": RESULTS / "revision_core" / f"seed{seed}" / "fixed_fusion" / "strict_fins.json",
            "external_full": RESULTS / "revision_core" / f"seed{seed}" / "fixed_fusion" / "external_full.json",
        }
    elif variant == "C_E_learned_unguided":
        names = {
            "long_envelope": RESULTS / "key_ablation_multiseed" / f"seed{seed}" / "dual_unguided" / "neupre_long_session_calibrated.json",
            "controlled": RESULTS / "controlled_revision" / f"seed{seed}" / "learned_unguided" / "evaluation.json",
            "strict_fins": RESULTS / "key_ablation_multiseed" / f"seed{seed}" / "dual_unguided" / "long_full_strict_calibrated.json",
            "external_full": RESULTS / "key_ablation_multiseed" / f"seed{seed}" / "dual_unguided" / "neupre_full_novel.json",
        }
    elif variant == "D_guided_no_semantic_aux":
        names = {
            "long_envelope": RESULTS / "key_ablation_multiseed" / f"seed{seed}" / "guided_no_semantic_aux" / "neupre_long_session_calibrated.json",
            "controlled": RESULTS / "controlled_revision" / f"seed{seed}" / "guided_no_semantic_aux" / "evaluation.json",
            "strict_fins": RESULTS / "key_ablation_multiseed" / f"seed{seed}" / "guided_no_semantic_aux" / "long_full_strict_calibrated.json",
            "external_full": RESULTS / "key_ablation_multiseed" / f"seed{seed}" / "guided_no_semantic_aux" / "neupre_full_novel.json",
        }
    elif variant == "F_full":
        names = {
            "long_envelope": RESULTS / "multiseed" / f"seed{seed}" / "guided_mamba" / "neupre_long_session_calibrated.json",
            "controlled": RESULTS / "controlled_long_range" / f"seed{seed}" / "full_mamba" / "evaluation.json",
            "strict_fins": RESULTS / "multiseed" / f"seed{seed}" / "guided_mamba" / "long_full_strict_calibrated.json",
            "external_full": RESULTS / "multiseed" / f"seed{seed}" / "guided_mamba" / "neupre_full_novel.json",
        }
    else:
        raise KeyError(variant)
    return names[dataset]


def phase3_path(variant: str, dataset: str, seed: int) -> Path:
    if variant == "binary_guidance":
        names = {
            "long_envelope": RESULTS / "multiseed" / f"seed{seed}" / "guided_mamba" / "neupre_long_session_calibrated.json",
            "controlled": RESULTS / "controlled_long_range" / f"seed{seed}" / "full_mamba" / "evaluation.json",
            "strict_fins": RESULTS / "multiseed" / f"seed{seed}" / "guided_mamba" / "long_full_strict_calibrated.json",
        }
    elif variant == "structural_only":
        names = {
            "long_envelope": RESULTS / "revision_core" / f"seed{seed}" / "structural_only" / "long_envelope.json",
            "controlled": RESULTS / "controlled_revision" / f"seed{seed}" / "structural_only" / "evaluation.json",
            "strict_fins": RESULTS / "revision_core" / f"seed{seed}" / "structural_only" / "strict_fins.json",
        }
    elif variant == "unguided":
        names = {
            "long_envelope": RESULTS / "key_ablation_multiseed" / f"seed{seed}" / "dual_unguided" / "neupre_long_session_calibrated.json",
            "controlled": RESULTS / "controlled_revision" / f"seed{seed}" / "learned_unguided" / "evaluation.json",
            "strict_fins": RESULTS / "key_ablation_multiseed" / f"seed{seed}" / "dual_unguided" / "long_full_strict_calibrated.json",
        }
    else:
        raise KeyError(variant)
    return names[dataset]


def aggregate_grid(variants: tuple[str, ...], datasets: tuple[str, ...], path_fn) -> dict:
    loaded = {
        variant: {
            dataset: {seed: metric_root(read(path_fn(variant, dataset, seed))) for seed in SEEDS}
            for dataset in datasets
        }
        for variant in variants
    }
    output = {"seeds": list(SEEDS), "variants": {}, "paired": {}}
    for variant in variants:
        output["variants"][variant] = {}
        for dataset in datasets:
            output["variants"][variant][dataset] = {
                metric: summarize([
                    float(loaded[variant][dataset][seed]["overall"][metric]) for seed in SEEDS
                ])
                for metric in METRICS
            }
    reference = variants[-1]
    for variant in variants[:-1]:
        output["paired"][f"{variant}_minus_{reference}"] = {}
        for dataset in datasets:
            output["paired"][f"{variant}_minus_{reference}"][dataset] = {
                metric: summarize([
                    float(loaded[variant][dataset][seed]["overall"][metric])
                    - float(loaded[reference][dataset][seed]["overall"][metric])
                    for seed in SEEDS
                ])
                for metric in METRICS
            }
    return output


def write_phase2_csv(aggregate: dict) -> None:
    path = OUT / "factorial_ablation.csv"
    rows = []
    for variant, datasets in aggregate["variants"].items():
        for dataset, metrics in datasets.items():
            row = {
                "variant": variant,
                "dataset": dataset,
                "parameters": PARAMETERS[variant],
                "parameter_delta_percent_vs_full": 100.0 * (PARAMETERS[variant] / PARAMETERS["F_full"] - 1.0),
            }
            for metric, summary in metrics.items():
                for stat in ("mean", "sample_std", "ci95_low", "ci95_high"):
                    row[f"{metric}_{stat}"] = summary[stat]
            rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_phase3_gate_csv() -> None:
    path = OUT / "gate_target_per_layer.csv"
    rows = []
    for variant in ("binary_guidance", "structural_only", "unguided"):
        for dataset in ("long_envelope", "controlled", "strict_fins"):
            for seed in SEEDS:
                report = metric_root(read(phase3_path(variant, dataset, seed)))
                analysis = report.get("gate_analysis", {})
                for layer in analysis.get("per_layer", []):
                    rows.append({
                        "variant": variant,
                        "dataset": dataset,
                        "seed": seed,
                        "layer": layer["layer"],
                        "structural_mean": layer["structural_mean"],
                        "payload_mean": layer["payload_mean"],
                        "separation": layer["separation"],
                    })
    if not rows:
        raise RuntimeError("no per-layer gate statistics found")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    for variant in ("binary_guidance", "structural_only", "unguided"):
        for dataset in ("long_envelope", "controlled", "strict_fins"):
            for layer in range(4):
                selected = [
                    row for row in rows
                    if row["variant"] == variant and row["dataset"] == dataset and row["layer"] == layer
                ]
                output = {"variant": variant, "dataset": dataset, "layer": layer}
                for metric in ("structural_mean", "payload_mean", "separation"):
                    stats = summarize([float(row[metric]) for row in selected])
                    for stat in ("mean", "sample_std", "ci95_low", "ci95_high"):
                        output[f"{metric}_{stat}"] = stats[stat]
                summary_rows.append(output)
    with (OUT / "gate_target_per_layer_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def write_tex(phase2: dict, phase3: dict) -> None:
    labels = {
        "A_bimamba_only": "A: Bi-Mamba only",
        "B_fixed_fusion": "B: fixed dual stream",
        "C_E_learned_unguided": "C/E: learned gate, no $L_g$",
        "D_guided_no_semantic_aux": "D: $L_g$, no semantic aux.",
        "F_full": "F: full",
    }
    lines = [
        "% Automatically generated by scripts/aggregate_revision_ablations.py",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Variant & Params & Long & External & Strict FINS \\\\",
        "\\midrule",
    ]
    for variant in PARAMETERS:
        values = phase2["variants"][variant]
        lines.append(
            f"{labels[variant]} & {PARAMETERS[variant] / 1e6:.3f}M & "
            f"{values['long_envelope']['boundary_f1']['mean']:.4f} & "
            f"{values['external_full']['boundary_f1']['mean']:.4f} & "
            f"{values['strict_fins']['boundary_f1']['mean']:.4f} \\\\" 
        )
    lines.extend(("\\bottomrule", "\\end{tabular}", ""))
    (OUT / "factorial_ablation.tex").write_text("\n".join(lines), encoding="utf-8")

    labels3 = {
        "binary_guidance": "Binary guidance",
        "structural_only": "Structural-only mask",
        "unguided": "Unguided gate",
    }
    lines = [
        "% Automatically generated by scripts/aggregate_revision_ablations.py",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Target & Long F1 & Controlled F1 & FINS P & FINS R & FINS F1 \\\\",
        "\\midrule",
    ]
    for variant in ("binary_guidance", "structural_only", "unguided"):
        values = phase3["variants"][variant]
        lines.append(
            f"{labels3[variant]} & {values['long_envelope']['boundary_f1']['mean']:.4f} & "
            f"{values['controlled']['boundary_f1']['mean']:.4f} & "
            f"{values['strict_fins']['boundary_precision']['mean']:.4f} & "
            f"{values['strict_fins']['boundary_recall']['mean']:.4f} & "
            f"{values['strict_fins']['boundary_f1']['mean']:.4f} \\\\" 
        )
    lines.extend(("\\bottomrule", "\\end{tabular}", ""))
    (OUT / "gate_target_ablation.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phase2_variants = tuple(PARAMETERS)
    phase2 = aggregate_grid(
        phase2_variants,
        ("long_envelope", "controlled", "external_full", "strict_fins"),
        phase2_path,
    )
    phase2["design_note"] = (
        "Cells C and E in the requested matrix are definitionally identical: both retain the "
        "learned gate and role/type auxiliary losses while setting gate_weight=0. They share "
        "one five-seed execution and are reported as C/E, not as independent evidence."
    )
    phase2["parameters"] = PARAMETERS
    (OUT / "factorial_ablation.json").write_text(json.dumps(phase2, indent=2) + "\n", encoding="utf-8")
    write_phase2_csv(phase2)

    phase3 = aggregate_grid(
        ("structural_only", "unguided", "binary_guidance"),
        ("long_envelope", "controlled", "strict_fins"),
        phase3_path,
    )
    (OUT / "gate_target_ablation.json").write_text(json.dumps(phase3, indent=2) + "\n", encoding="utf-8")
    write_phase3_gate_csv()
    write_tex(phase2, phase3)
    print(OUT / "factorial_ablation.json")
    print(OUT / "gate_target_ablation.json")


if __name__ == "__main__":
    main()
