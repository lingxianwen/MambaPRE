from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


SEEDS = (1337, 2027, 3407, 4701, 9001)
DISTANCES = (64, 128, 256, 512, 1024)
MODELS = ("full_mamba", "matched_transformer", "supervised_cnn", "bimamba_only")
DISPLAY = {
    "full_mamba": "Mamba-PRE",
    "matched_transformer": "Transformer",
    "supervised_cnn": "Supervised CNN",
    "bimamba_only": "Bi-Mamba only",
}
T95_DF4 = 2.7764451051977987


def summarize(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    half = T95_DF4 * std / math.sqrt(len(values))
    return {"mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def load_reports(root: Path) -> dict:
    reports = {}
    for seed in SEEDS:
        for model in MODELS:
            path = root / f"seed{seed}" / model / "evaluation.json"
            reports[(seed, model)] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the controlled long-range five-seed experiment")
    parser.add_argument("--results-root", default="results/controlled_long_range")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--figure-root", default="figures")
    args = parser.parse_args()
    root = Path(args.results_root)
    output_root = Path(args.output_root)
    figure_root = Path(args.figure_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    reports = load_reports(root)

    localization = {}
    for seed in SEEDS:
        for model in MODELS:
            path = root / f"seed{seed}" / model / "per_message.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for distance in DISTANCES:
                selected = [row for row in rows if int(row["distance"]) == distance]
                localization[(seed, model, distance)] = {
                    "top1_accuracy": statistics.fmean(float(row["top1_correct"]) for row in selected),
                    "top1_absolute_error": statistics.fmean(float(row["top1_absolute_error"]) for row in selected),
                    "target_rank": statistics.fmean(float(row["target_rank"]) for row in selected),
                }

    per_seed_rows = []
    for seed in SEEDS:
        for model in MODELS:
            report = reports[(seed, model)]
            parameters = int(report.get("parameters", 0))
            for distance in DISTANCES:
                metric = report["test"]["by_protocol"][f"controlled_lr_{distance:04d}"]
                per_seed_rows.append({
                    "seed": seed,
                    "model": model,
                    "parameters": parameters,
                    "parameter_delta_vs_full": parameters - int(reports[(seed, "full_mamba")].get("parameters", 0)),
                    "distance": distance,
                    "precision": metric["boundary_precision"],
                    "recall": metric["boundary_recall"],
                    "boundary_f1": metric["boundary_f1"],
                    **localization[(seed, model, distance)],
                    "selected_threshold": report["selected_threshold"],
                })
    per_seed_path = output_root / "long_range_per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(per_seed_rows[0]))
        writer.writeheader()
        writer.writerows(per_seed_rows)

    per_message_path = output_root / "long_range_per_message.csv"
    with per_message_path.open("w", newline="", encoding="utf-8") as output:
        writer = None
        for seed in SEEDS:
            for model in MODELS:
                path = root / f"seed{seed}" / model / "per_message.csv"
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    if writer is None:
                        writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    writer.writerows(reader)

    summary_rows = []
    for distance in DISTANCES:
        model_summary = {}
        for model in MODELS:
            selected = [row for row in per_seed_rows if row["distance"] == distance and row["model"] == model]
            model_summary[model] = {metric: summarize([float(row[metric]) for row in selected]) for metric in ("precision", "recall", "boundary_f1", "top1_accuracy", "top1_absolute_error", "target_rank")}
        mamba_f1 = [float(row["boundary_f1"]) for row in per_seed_rows if row["distance"] == distance and row["model"] == "full_mamba"]
        cnn_f1 = [float(row["boundary_f1"]) for row in per_seed_rows if row["distance"] == distance and row["model"] == "supervised_cnn"]
        transformer_f1 = [float(row["boundary_f1"]) for row in per_seed_rows if row["distance"] == distance and row["model"] == "matched_transformer"]
        mamba_cnn = summarize([left - right for left, right in zip(mamba_f1, cnn_f1)])
        mamba_transformer = summarize([left - right for left, right in zip(mamba_f1, transformer_f1)])
        for model in MODELS:
            row = {"distance": distance, "model": model}
            for metric in ("precision", "recall", "boundary_f1", "top1_accuracy", "top1_absolute_error", "target_rank"):
                for stat, value in model_summary[model][metric].items():
                    row[f"{metric}_{stat}"] = value
            row.update({
                "mamba_minus_cnn_f1_mean": mamba_cnn["mean"] if model == "full_mamba" else "",
                "mamba_minus_cnn_f1_ci95_low": mamba_cnn["ci95_low"] if model == "full_mamba" else "",
                "mamba_minus_cnn_f1_ci95_high": mamba_cnn["ci95_high"] if model == "full_mamba" else "",
                "mamba_minus_transformer_f1_mean": mamba_transformer["mean"] if model == "full_mamba" else "",
                "mamba_minus_transformer_f1_ci95_low": mamba_transformer["ci95_low"] if model == "full_mamba" else "",
                "mamba_minus_transformer_f1_ci95_high": mamba_transformer["ci95_high"] if model == "full_mamba" else "",
            })
            summary_rows.append(row)
    summary_path = output_root / "long_range_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    log_distances = [math.log2(value) for value in DISTANCES]
    x_mean = statistics.fmean(log_distances)
    x_ss = sum((value - x_mean) ** 2 for value in log_distances)
    slopes = []
    endpoint_increases = []
    for seed in SEEDS:
        differences = []
        for distance in DISTANCES:
            mamba = next(float(row["boundary_f1"]) for row in per_seed_rows if row["seed"] == seed and row["model"] == "full_mamba" and row["distance"] == distance)
            cnn = next(float(row["boundary_f1"]) for row in per_seed_rows if row["seed"] == seed and row["model"] == "supervised_cnn" and row["distance"] == distance)
            differences.append(mamba - cnn)
        y_mean = statistics.fmean(differences)
        slopes.append(sum((x - x_mean) * (y - y_mean) for x, y in zip(log_distances, differences)) / x_ss)
        endpoint_increases.append(differences[-1] - differences[0])
    trend = {
        "meaning": "Seed-wise OLS slope of (Mamba-PRE minus CNN boundary F1) per doubling of dependency distance; endpoint increase is the paired 1024-byte difference minus the paired 64-byte difference.",
        "slope_per_distance_doubling": summarize(slopes),
        "endpoint_increase_1024_minus_64": summarize(endpoint_increases),
    }
    (output_root / "long_range_trend.json").write_text(json.dumps(trend, indent=2) + "\n", encoding="utf-8")

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0, "axes.labelsize": 8.0, "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2, "legend.fontsize": 7.0, "axes.linewidth": 0.8,
        "lines.linewidth": 1.5, "lines.markersize": 4.5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })
    styles = {
        "full_mamba": dict(color="#0B5CAD", marker="o", linestyle="-"),
        "matched_transformer": dict(color="#0F766E", marker="s", linestyle="--"),
        "supervised_cnn": dict(color="#B64040", marker="^", linestyle=":"),
        "bimamba_only": dict(color="#6B7280", marker="D", linestyle="-."),
    }
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    for model in MODELS:
        means, lows, highs = [], [], []
        for distance in DISTANCES:
            summary = next(row for row in summary_rows if row["distance"] == distance and row["model"] == model)
            means.append(float(summary["boundary_f1_mean"]))
            lows.append(float(summary["boundary_f1_ci95_low"]))
            highs.append(float(summary["boundary_f1_ci95_high"]))
        errors = [[mean - low for mean, low in zip(means, lows)], [high - mean for mean, high in zip(means, highs)]]
        ax.errorbar(DISTANCES, means, yerr=errors, capsize=2.2, label=DISPLAY[model], **styles[model])
    ax.set_xscale("log", base=2)
    ax.set_xticks(DISTANCES, [str(value) for value in DISTANCES])
    ax.set_xlabel("Length-field dependency distance (bytes)")
    ax.set_ylabel("Boundary F1 (higher is better)")
    upper = max(
        float(row["boundary_f1_ci95_high"])
        for row in summary_rows
    )
    ax.set_ylim(0, max(0.03, 1.12 * upper))
    ax.grid(axis="y", color="#D7DEE8", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout(pad=0.45)
    stem = figure_root / "long_range_distance"
    for suffix, dpi in (("pdf", None), ("svg", None), ("png", 400), ("tiff", 600)):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    for model in MODELS:
        means, lows, highs = [], [], []
        for distance in DISTANCES:
            summary = next(row for row in summary_rows if row["distance"] == distance and row["model"] == model)
            means.append(float(summary["top1_accuracy_mean"]))
            lows.append(float(summary["top1_accuracy_ci95_low"]))
            highs.append(float(summary["top1_accuracy_ci95_high"]))
        errors = [[mean - low for mean, low in zip(means, lows)], [high - mean for mean, high in zip(means, highs)]]
        ax.errorbar(DISTANCES, means, yerr=errors, capsize=2.2, label=DISPLAY[model], **styles[model])
    ax.set_xscale("log", base=2)
    ax.set_xticks(DISTANCES, [str(value) for value in DISTANCES])
    ax.set_xlabel("Length-field dependency distance (bytes)")
    ax.set_ylabel("Top-1 boundary localization")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#D7DEE8", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(pad=0.45)
    stem = figure_root / "long_range_localization"
    for suffix, dpi in (("pdf", None), ("svg", None), ("png", 400), ("tiff", 600)):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(summary_path)


if __name__ == "__main__":
    main()
