from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


SEEDS = (1337, 2027, 3407, 4701, 9001)
DISTANCES = (16, 32, 64, 128, 256, 512, 1024)
PRIMARY = (64, 128, 256, 512, 1024)
MODELS = (
    "full_mamba", "matched_transformer", "supervised_cnn", "bimamba_only",
    "fixed_fusion", "learned_unguided", "guided_no_semantic_aux", "structural_only",
)
MAIN_MODELS = MODELS[:4]
DISPLAY = {
    "full_mamba": "Mamba-PRE", "matched_transformer": "Transformer",
    "supervised_cnn": "Supervised CNN", "bimamba_only": "Bi-Mamba only",
    "fixed_fusion": "Fixed fusion", "learned_unguided": "Unguided gate",
    "guided_no_semantic_aux": "No semantic aux.", "structural_only": "Structural-only $L_g$",
}
T95_DF4 = 2.7764451051977987


def summarize(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    half = T95_DF4 * std / math.sqrt(len(values))
    return {"values": values, "mean": mean, "sample_std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def result_path(root: Path, seed: int, model: str) -> Path:
    base = "controlled_candidate_selection_pw8" if model in MAIN_MODELS else "controlled_candidate_revision_pw8"
    return root / base / f"seed{seed}" / model / "evaluation.json"


def prediction_path(root: Path, seed: int, model: str) -> Path:
    base = "controlled_candidate_selection_pw8" if model in MAIN_MODELS else "controlled_candidate_revision_pw8"
    return root / base / f"seed{seed}" / model / "per_message.csv"


def main() -> None:
    root = Path("results")
    figure_root = Path("figures")
    revision_root = root / "revision"
    figure_root.mkdir(parents=True, exist_ok=True)
    revision_root.mkdir(parents=True, exist_ok=True)
    reports = {(seed, model): json.loads(result_path(root, seed, model).read_text(encoding="utf-8")) for seed in SEEDS for model in MODELS}
    localization = {}
    for seed in SEEDS:
        for model in MODELS:
            with prediction_path(root, seed, model).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for distance in DISTANCES:
                selected = [row for row in rows if int(row["distance"]) == distance]
                localization[(seed, model, distance)] = {
                    "top1_accuracy": statistics.fmean(float(row["top1_correct"]) for row in selected),
                    "target_rank": statistics.fmean(float(row["target_rank"]) for row in selected),
                }

    per_seed = []
    for seed in SEEDS:
        for model in MODELS:
            report = reports[(seed, model)]
            for distance in DISTANCES:
                metric = report["test"]["by_protocol"][f"controlled_candidate_{distance:04d}"]
                per_seed.append({
                    "seed": seed, "model": model, "distance": distance,
                    "precision": metric["boundary_precision"], "recall": metric["boundary_recall"],
                    "boundary_f1": metric["boundary_f1"], **localization[(seed, model, distance)],
                    "selected_threshold": report["selected_threshold"],
                    "parameters": report.get("parameters", 0),
                })
    with (root / "candidate_selection_per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(per_seed[0]))
        writer.writeheader(); writer.writerows(per_seed)

    summary_rows = []
    for distance in DISTANCES:
        for model in MODELS:
            selected = [row for row in per_seed if row["distance"] == distance and row["model"] == model]
            row = {"distance": distance, "model": model}
            for metric in ("precision", "recall", "boundary_f1", "top1_accuracy", "target_rank"):
                for stat, value in summarize([float(item[metric]) for item in selected]).items():
                    if stat != "values": row[f"{metric}_{stat}"] = value
            if model == "full_mamba":
                for baseline in ("supervised_cnn", "matched_transformer"):
                    base = [row for row in per_seed if row["distance"] == distance and row["model"] == baseline]
                    paired = summarize([float(a["boundary_f1"]) - float(b["boundary_f1"]) for a, b in zip(selected, base)])
                    for stat in ("mean", "ci95_low", "ci95_high"):
                        row[f"mamba_minus_{baseline}_f1_{stat}"] = paired[stat]
            summary_rows.append(row)
    fields = sorted({key for row in summary_rows for key in row}, key=lambda key: (key not in ("distance", "model"), key))
    with (root / "candidate_selection_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(summary_rows)

    # Pre-specified primary-distance aggregates and trends, paired by seed.
    primary_aggregate = {}
    for model in MODELS:
        primary_aggregate[model] = {}
        for metric in ("boundary_f1", "top1_accuracy"):
            values = []
            for seed in SEEDS:
                values.append(statistics.fmean(
                    float(row[metric]) for row in per_seed
                    if row["seed"] == seed and row["model"] == model
                    and row["distance"] in PRIMARY
                ))
            primary_aggregate[model][metric] = summarize(values)
    paired_primary = {}
    for baseline in ("supervised_cnn", "matched_transformer", "bimamba_only"):
        paired_primary[baseline] = {}
        for metric in ("boundary_f1", "top1_accuracy"):
            differences = []
            for seed in SEEDS:
                mamba = statistics.fmean(
                    float(row[metric]) for row in per_seed
                    if row["seed"] == seed and row["model"] == "full_mamba"
                    and row["distance"] in PRIMARY
                )
                other = statistics.fmean(
                    float(row[metric]) for row in per_seed
                    if row["seed"] == seed and row["model"] == baseline
                    and row["distance"] in PRIMARY
                )
                differences.append(mamba - other)
            paired_primary[baseline][metric] = summarize(differences)

    x = [math.log2(value) for value in PRIMARY]
    x_mean = statistics.fmean(x)
    x_ss = sum((value - x_mean) ** 2 for value in x)
    trend = {"primary_distance_aggregate": primary_aggregate, "paired_primary": paired_primary}
    for metric in ("boundary_f1", "top1_accuracy"):
        slopes = []
        for seed in SEEDS:
            differences = []
            for distance in PRIMARY:
                m = next(row[metric] for row in per_seed if row["seed"] == seed and row["model"] == "full_mamba" and row["distance"] == distance)
                c = next(row[metric] for row in per_seed if row["seed"] == seed and row["model"] == "supervised_cnn" and row["distance"] == distance)
                differences.append(float(m) - float(c))
            y_mean = statistics.fmean(differences)
            slopes.append(sum((a - x_mean) * (b - y_mean) for a, b in zip(x, differences)) / x_ss)
        trend[f"mamba_minus_cnn_{metric}_slope_per_distance_doubling"] = summarize(slopes)
    (root / "candidate_selection_trend.json").write_text(json.dumps(trend, indent=2) + "\n", encoding="utf-8")

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "legend.fontsize": 7, "axes.linewidth": 0.8, "lines.linewidth": 1.5,
        "lines.markersize": 4.5, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    styles = {
        "full_mamba": dict(color="#0B5CAD", marker="o", linestyle="-"),
        "matched_transformer": dict(color="#0F766E", marker="s", linestyle="--"),
        "supervised_cnn": dict(color="#B64040", marker="^", linestyle=":"),
        "bimamba_only": dict(color="#6B7280", marker="D", linestyle="-."),
    }
    for metric, ylabel, stem in (
        ("boundary_f1", "Boundary F1", "candidate_selection_f1"),
        ("top1_accuracy", "Top-1 boundary localization", "candidate_selection_top1"),
    ):
        fig, ax = plt.subplots(figsize=(3.5, 1.60))
        for model in ("full_mamba", "matched_transformer", "supervised_cnn"):
            means, low, high = [], [], []
            for distance in DISTANCES:
                row = next(item for item in summary_rows if item["distance"] == distance and item["model"] == model)
                means.append(float(row[f"{metric}_mean"])); low.append(float(row[f"{metric}_ci95_low"])); high.append(float(row[f"{metric}_ci95_high"]))
            errors = [[a - b for a, b in zip(means, low)], [b - a for a, b in zip(means, high)]]
            ax.errorbar(DISTANCES, means, yerr=errors, capsize=2.2, label=DISPLAY[model], **styles[model])
        ax.axvspan(12, 32, color="#E8EDF4", alpha=0.7, zorder=-2)
        ax.axvline(32, color="#6B7280", linewidth=0.7, linestyle="--", zorder=-1)
        if metric == "top1_accuracy":
            ax.axhline(0.5, color="#4B5563", linewidth=0.8, linestyle=(0, (2, 2)), zorder=-1)
            ax.text(1000, 0.515, "chance", color="#4B5563", fontsize=6.8, ha="right", va="bottom")
        ax.set_xscale("log", base=2); ax.set_xticks(DISTANCES, [str(value) for value in DISTANCES])
        ax.set_xlabel("Selector-to-candidate distance (bytes)"); ax.set_ylabel(ylabel); ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#D7DEE8", linewidth=0.55); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, ncol=2, loc="lower left")
        fig.tight_layout(pad=0.45)
        for suffix, dpi in (("pdf", None), ("svg", None), ("png", 400), ("tiff", 600)):
            fig.savefig((figure_root / stem).with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    print(root / "candidate_selection_summary.csv")


if __name__ == "__main__":
    main()
