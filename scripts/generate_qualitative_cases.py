#!/usr/bin/env python3
"""Select representative boundary cases deterministically and plot Fig. 3."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT.parent / "ICASSP2026_Paper_Templates"
SELECTION = RESULTS / "reproduction" / "qualitative_case_selection.json"


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.4,
    "axes.titlesize": 7.8,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.7,
    "ytick.labelsize": 7.0,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

COLORS = {
    "gold": "#172033",
    "mamba": "#0B5CAD",
    "transformer": "#0F766E",
    "false": "#B64040",
    "grid": "#D7DEE8",
}


def parse_set(value: str) -> set[int]:
    return set(map(int, value.split())) if value.strip() else set()


def read_csv(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def boundary_f1(predicted: set[int], gold: set[int]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    return 2.0 * len(predicted & gold) / (len(predicted) + len(gold))


def select_case(aggregate: dict, dataset: str, filename: str) -> dict:
    summary = aggregate["main"][dataset]
    target_m = float(summary["mambapre"]["mean"])
    target_t = float(summary["transformer"]["mean"])
    seed_rows = []
    for index, seed in enumerate(aggregate["seeds"]):
        m = float(summary["mambapre"]["values"][index])
        t = float(summary["transformer"]["values"][index])
        seed_rows.append((abs(m - target_m) + abs(t - target_t), int(seed), m, t))
    _, seed, seed_m, seed_t = min(seed_rows)

    mamba_path = (
        RESULTS / "reproduction" / "batch_invariant_per_message"
        / f"seed{seed}" / f"{filename}.csv"
    )
    transformer_path = (
        RESULTS / "reproduction" / "per_message" / "matched_transformer"
        / f"seed{seed}" / f"{filename}.csv"
    )
    mamba = read_csv(mamba_path)
    transformer = read_csv(transformer_path)
    candidates = []
    for message_id, m_row in mamba.items():
        t_row = transformer[message_id]
        gold = parse_set(m_row["gold_boundaries"])
        m_pred = parse_set(m_row["predicted_boundaries"])
        t_pred = parse_set(t_row["predicted_boundaries"])
        m_f1 = boundary_f1(m_pred, gold)
        t_f1 = boundary_f1(t_pred, gold)
        distance = abs(m_f1 - target_m) + abs(t_f1 - target_t)
        candidates.append((distance, message_id, m_row, gold, m_pred, t_pred, m_f1, t_f1))
    distance, message_id, row, gold, m_pred, t_pred, m_f1, t_f1 = min(candidates)
    return {
        "dataset": dataset,
        "selection_rule": (
            "Choose the seed minimizing L1 distance between its Mamba/Transformer "
            "aggregate-F1 pair and the five-seed mean pair; within that seed choose "
            "the message minimizing the same distance, breaking ties by message ID."
        ),
        "seed": seed,
        "seed_boundary_f1": {"mambapre": seed_m, "transformer": seed_t},
        "five_seed_mean_boundary_f1": {"mambapre": target_m, "transformer": target_t},
        "message_distance": distance,
        "id": message_id,
        "protocol": row["protocol"],
        "length": int(row["length"]),
        "threshold": float(row["threshold"]),
        "gold": sorted(gold),
        "mambapre": sorted(m_pred),
        "transformer": sorted(t_pred),
        "message_boundary_f1": {"mambapre": m_f1, "transformer": t_f1},
        "counts": {
            "gold": len(gold),
            "mambapre_tp": len(m_pred & gold),
            "mambapre_fp": len(m_pred - gold),
            "mambapre_fn": len(gold - m_pred),
            "transformer_tp": len(t_pred & gold),
            "transformer_fp": len(t_pred - gold),
            "transformer_fn": len(gold - t_pred),
        },
        "sources": {"mambapre": str(mamba_path), "transformer": str(transformer_path)},
    }


def draw_track(ax, values: set[int], gold: set[int], y: float, color: str) -> None:
    true = sorted(values & gold)
    false = sorted(values - gold)
    if true:
        ax.scatter(true, [y] * len(true), marker="|", s=48, linewidths=0.8,
                   color=color, clip_on=True, zorder=3)
    if false:
        ax.scatter(false, [y] * len(false), marker="x", s=12, linewidths=0.75,
                   color=COLORS["false"], clip_on=True, zorder=4)


def format_track_axis(ax, xlim, ticks, show_y: bool) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.45, 2.45)
    ax.set_xticks(ticks)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["Gold", "Mamba-PRE", "Transformer"])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45)
    ax.tick_params(axis="both", length=2, pad=1.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if not show_y:
        ax.tick_params(axis="y", labelleft=False)
    for y in (0, 1, 2):
        ax.hlines(y, xlim[0], xlim[1], color="#C8CDD5", linewidth=0.45, zorder=0)


def plot_case_on_axis(ax, case: dict, xlim, ticks, show_y=True) -> None:
    gold = set(case["gold"])
    mamba = set(case["mambapre"])
    transformer = set(case["transformer"])
    ax.scatter(sorted(gold), [2] * len(gold), marker="|", s=48, linewidths=0.7,
               color=COLORS["gold"], clip_on=True, zorder=3)
    draw_track(ax, mamba, gold, 1, COLORS["mamba"])
    draw_track(ax, transformer, gold, 0, COLORS["transformer"])
    format_track_axis(ax, xlim, ticks, show_y)


def main() -> None:
    aggregate = json.loads(
        (RESULTS / "batch_invariant_revision" / "final_aggregate.json").read_text()
    )
    long_case = select_case(aggregate, "long_envelope", "neupre_long_session_calibrated")
    fins_case = select_case(aggregate, "strict_fins", "long_full_strict_calibrated")
    selection = {"long_envelope": long_case, "strict_fins": fins_case}
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    SELECTION.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

    # Keep the plot compact enough for the remaining one-column space on page 4.
    # Text stays at IEEE-readable sizes; vertical whitespace, not font size, is reduced.
    fig = plt.figure(figsize=(3.5, 1.42))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1), width_ratios=(1.22, 0.78),
                            left=0.22, right=0.985, top=0.90, bottom=0.23,
                            hspace=0.76, wspace=0.06)
    ax_prefix = fig.add_subplot(grid[0, 0])
    ax_payload = fig.add_subplot(grid[0, 1], sharey=ax_prefix)
    plot_case_on_axis(ax_prefix, long_case, (0, 32), [0, 8, 16, 24], True)
    plot_case_on_axis(ax_payload, long_case, (32, 1024), [32, 512, 1024], False)
    ax_prefix.set_title("(a) Long envelope: prefix", loc="left", fontweight="bold", pad=2)
    ax_payload.set_title("opaque payload", loc="center", pad=2)
    # The upper track omits an axis title to preserve separation from panel (b).
    ax_prefix.spines["right"].set_visible(False)
    ax_payload.spines["left"].set_visible(False)
    ax_payload.tick_params(axis="y", left=False)
    long_f1 = long_case["message_boundary_f1"]
    ax_payload.text(1000, 1.18, f"F1={long_f1['mambapre']:.3f}", ha="right",
                    color=COLORS["mamba"], fontsize=6.7)
    ax_payload.text(1000, 0.18, f"F1={long_f1['transformer']:.3f}", ha="right",
                    color=COLORS["transformer"], fontsize=6.7)

    # Expand the dense 0--20-byte prefix without dropping or jittering any cut.
    # The two adjacent axes retain true offsets, with explicitly different scales.
    fins_grid = grid[1, :].subgridspec(1, 2, width_ratios=(0.38, 0.62), wspace=0.12)
    ax_fins_prefix = fig.add_subplot(fins_grid[0, 0])
    ax_fins = fig.add_subplot(fins_grid[0, 1], sharey=ax_fins_prefix)
    plot_case_on_axis(ax_fins_prefix, fins_case, (0, 20), [0, 5, 10, 15], True)
    plot_case_on_axis(ax_fins, fins_case, (20, 540), [20, 200, 400, 540], False)
    ax_fins_prefix.set_title("(b) Strict FINS: split offset scales", loc="left",
                      fontweight="bold", pad=2)
    ax_fins_prefix.set_xlabel("Offset (byte)", labelpad=1)
    ax_fins.set_xlabel("Offset (byte)", labelpad=1)
    fins_f1 = fins_case["message_boundary_f1"]
    ax_fins.text(530, 1.18, f"F1={fins_f1['mambapre']:.3f}", ha="right",
                 color=COLORS["mamba"], fontsize=6.7)
    ax_fins.text(530, 0.18, f"F1={fins_f1['transformer']:.3f}", ha="right",
                 color=COLORS["transformer"], fontsize=6.7)

    for suffix, kwargs in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 300},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(PAPER / f"Fig3_qualitative{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
