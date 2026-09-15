"""Generate the ICASSP scaling figure from cached benchmark JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "scaling" / "comparison.json"
FIGURE_DIR = ROOT / "figures"
DATA_DIR = FIGURE_DIR / "data"


def main() -> None:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    fp32 = report["precisions"]["float32"]["rows"]
    bf16 = report["precisions"]["bfloat16"]["rows"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / "scaling_speed_memory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["precision", "bytes", "speedup_transformer_over_mamba", "memory_reduction_percent"])
        for precision, rows in (("FP32", fp32), ("BF16", bf16)):
            for row in rows:
                writer.writerow([
                    precision,
                    row["length"],
                    row["mamba_speedup"],
                    100.0 * row["mamba_memory_reduction"],
                ])

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.72), constrained_layout=True)
    styles = {
        "FP32": dict(color="#0B5CAD", marker="o", linestyle="-"),
        "BF16": dict(color="#B7791F", marker="s", linestyle="--"),
    }
    for precision, rows in (("FP32", fp32), ("BF16", bf16)):
        lengths = [row["length"] for row in rows]
        speedups = [row["mamba_speedup"] for row in rows]
        reductions = [100.0 * row["mamba_memory_reduction"] for row in rows]
        axes[0].plot(lengths, speedups, label=precision, **styles[precision])
        axes[1].plot(lengths, reductions, label=precision, **styles[precision])

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks([256, 512, 1024, 2048, 4096])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value / 1024)}k" if value >= 1024 else f"{int(value)}"))
        axis.set_xlabel("Message bytes")
        axis.grid(True, axis="y", color="#D7DEE8", linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].axhline(1.0, color="#6B7280", linewidth=0.8, linestyle=":", zorder=0)
    axes[0].set_ylabel("Speedup (Transformer/Mamba)")
    axes[0].set_ylim(0.0, 4.35)
    axes[0].text(0.02, 0.96, "(a) Latency", transform=axes[0].transAxes, va="top", fontweight="bold")
    axes[0].annotate("4.00x", xy=(4096, fp32[-1]["mamba_speedup"]), xytext=(-17, -11), textcoords="offset points", color="#0B5CAD")

    axes[1].set_ylabel("Peak-memory reduction (%)")
    axes[1].set_ylim(0.0, 101.0)
    axes[1].text(0.02, 0.96, "(b) Memory", transform=axes[1].transAxes, va="top", fontweight="bold")
    axes[1].annotate("95.7%", xy=(4096, 100.0 * fp32[-1]["mamba_memory_reduction"]), xytext=(-23, -11), textcoords="offset points", color="#0B5CAD")
    axes[1].legend(frameon=False, loc="lower right", handlelength=2.3)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(FIGURE_DIR / f"scaling_tradeoff.{suffix}", dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
