from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from mambapre.reporting import summarize_values


def overall_metrics(report: dict) -> dict[str, float]:
    tp = sum(int(value["tp"]) for value in report["bins"].values())
    fp = sum(int(value["fp"]) for value in report["bins"].values())
    fn = sum(int(value["fn"]) for value in report["bins"].values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate five-seed offset-bin reports")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for value in args.inputs:
        report = json.loads(Path(value).read_text(encoding="utf-8"))
        grouped[report["model"]].append(report)
    result = {"models": {}, "inputs": args.inputs}
    for model, reports in sorted(grouped.items()):
        bins = reports[0]["bins"].keys()
        result["models"][model] = {
            "seeds": sorted(report["seed"] for report in reports),
            "training_max_length": sorted(
                {report["training_max_length"] for report in reports}
            ),
            "test_uses_untrained_absolute_indices": any(
                report["test_uses_untrained_absolute_indices"] for report in reports
            ),
            "overall": {
                metric: summarize_values(
                    [overall_metrics(report)[metric] for report in reports]
                )
                for metric in ("precision", "recall", "f1")
            },
            "bins": {
                name: {
                    metric: summarize_values(
                        [float(report["bins"][name][metric]) for report in reports]
                    )
                    for metric in ("precision", "recall", "f1")
                }
                | {
                    "gold_per_seed": sorted(
                        {int(report["bins"][name]["gold"]) for report in reports}
                    )
                }
                for name in bins
            },
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
