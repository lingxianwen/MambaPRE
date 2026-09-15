from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


METRICS = ("boundary_f1", "exact_field_f1", "message_perfection")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_classical(report: dict) -> dict:
    def row(value: dict) -> dict:
        return {
            "n": value["n"],
            "boundary_f1": value["bpos_F1"],
            "exact_field_f1": value["exact_boundary_F1"],
            "message_perfection": value["msg_perfection"],
        }

    return {
        "overall": row(report["overall"]),
        "by_protocol": {
            value["protocol"]: row(value) for value in report["per_protocol"]
        },
    }


def add_macro(report: dict) -> dict:
    values = list(report["by_protocol"].values())
    report = dict(report)
    report["macro_protocol"] = {
        metric: statistics.fmean(float(value[metric]) for value in values)
        for metric in METRICS
    }
    return report


def _t_critical_95(degrees_of_freedom: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.7764451051977987, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
             30: 2.042}
    if degrees_of_freedom in table:
        return table[degrees_of_freedom]
    if degrees_of_freedom < 15:
        return table[10]
    if degrees_of_freedom < 20:
        return table[15]
    if degrees_of_freedom < 30:
        return table[20]
    return 1.96


def summarize_values(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty list")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = (
        _t_critical_95(len(values) - 1) * std / math.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )
    return {
        "n_seeds": len(values),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "values": values,
    }
