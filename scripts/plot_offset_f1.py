from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ORDER = ("1-32", "33-128", "129-256", "257-512", "513+")
DISPLAY_SAFE = ("1\u201332", "33\u2013128", "129\u2013256", "257\u2013512", "513+")
DISPLAY = ("1–32", "33–128", "129–256", "257–512", "513+")
STYLES = {
    "mambapre": ("Mamba-PRE", "#0B5CAD", "o", "-"),
    "mambapre_core": ("Mamba-PRE", "#0B5CAD", "o", "-"),
    "transformer": ("Transformer", "#0F766E", "s", "--"),
    "transformer_core": ("Transformer", "#0F766E", "s", "--"),
    "matched_bigru": ("BiGRU", "#B64040", "^", ":"),
}


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["models"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot five-seed boundary F1 by byte offset")
    parser.add_argument("--fins", required=True)
    parser.add_argument("--opcua", required=True)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()

    datasets = [
        ("(a) Strict FINS: 29 messages", load(args.fins)),
        ("(b) Public OPC UA pilot: 5 messages", load(args.opcua)),
    ]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.3,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.3,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.72), sharex=True)
    rows: list[list[str | int | float]] = []
    handles = {}
    x = np.arange(len(ORDER))
    for ax, (title, models) in zip(axes, datasets):
        for key, values in models.items():
            if key not in STYLES:
                continue
            label, color, marker, linestyle = STYLES[key]
            means = [float(values["bins"][name]["f1"]["mean"]) for name in ORDER]
            stds = [float(values["bins"][name]["f1"]["std"]) for name in ORDER]
            line = ax.errorbar(
                x,
                means,
                yerr=stds,
                color=color,
                marker=marker,
                linestyle=linestyle,
                capsize=2,
                elinewidth=0.8,
                label=label,
            )
            handles[label] = line
            for offset, mean, std in zip(ORDER, means, stds):
                rows.append([title[4:], label, offset, mean, std, 5])
        ax.axvline(1.5, color="#9CA3AF", linewidth=0.7, linestyle=(0, (2, 2)))
        ax.set_ylim(0.0, 0.66)
        ax.set_ylabel("Boundary F1")
        ax.set_title(title, loc="left", fontweight="bold", pad=2)
        ax.grid(axis="y", color="#D7DEE8", linewidth=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xticks(x, DISPLAY_SAFE)
    axes[-1].set_xlabel("Absolute boundary offset (byte)")
    fig.legend(
        list(handles.values()),
        list(handles.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=max(1, len(handles)),
        frameon=False,
        handlelength=2.5,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.6)

    stem = Path(args.output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, options in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 300},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", **options)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "model", "offset_bin", "f1_mean", "f1_std", "seeds"])
        writer.writerows(rows)
    plt.close(fig)


if __name__ == "__main__":
    main()
