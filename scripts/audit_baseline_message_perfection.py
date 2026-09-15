from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from baselines.common import BoundaryCounts, load_rows


def audit_one(report_path: Path, rows_by_id: dict[str, dict]) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    predictions = report.get("predictions", [])
    pred_by_id: dict[str, set[int]] = {}
    duplicate_ids: list[str] = []
    for item in predictions:
        message_id = str(item["id"])
        if message_id in pred_by_id:
            duplicate_ids.append(message_id)
        pred_by_id[message_id] = {int(value) for value in item["predicted_boundaries"]}

    missing_ids = sorted(set(rows_by_id) - set(pred_by_id))
    unknown_ids = sorted(set(pred_by_id) - set(rows_by_id))
    counts = BoundaryCounts()
    error_sizes: list[int] = []
    misses_per_message: list[int] = []
    extras_per_message: list[int] = []
    exact_ids: list[str] = []
    closest: list[dict] = []
    by_protocol = defaultdict(lambda: {"n": 0, "perfect": 0})

    for message_id, row in rows_by_id.items():
        if message_id not in pred_by_id:
            continue
        length = len(row["raw"])
        predicted = {cut for cut in pred_by_id[message_id] if 0 < cut < length}
        gold = set(row["gold_cuts"])
        missing = sorted(gold - predicted)
        extra = sorted(predicted - gold)
        error = len(missing) + len(extra)
        counts.add(predicted, gold, length)
        error_sizes.append(error)
        misses_per_message.append(len(missing))
        extras_per_message.append(len(extra))
        protocol = row["protocol"]
        by_protocol[protocol]["n"] += 1
        if error == 0:
            exact_ids.append(message_id)
            by_protocol[protocol]["perfect"] += 1
        closest.append({
            "id": message_id,
            "protocol": protocol,
            "length": length,
            "gold_cuts": len(gold),
            "predicted_cuts": len(predicted),
            "missing_cuts": len(missing),
            "extra_cuts": len(extra),
            "symmetric_difference": error,
        })

    recomputed = counts.summary()
    reported = report.get("overall", {})
    metric_deltas = {
        metric: abs(float(recomputed[metric]) - float(reported[metric]))
        for metric in ("boundary_precision", "boundary_recall", "boundary_f1",
                       "exact_field_f1", "message_perfection")
        if metric in reported
    }
    histogram = Counter(error_sizes)
    return {
        "tool": report.get("tool", report_path.stem),
        "report": str(report_path),
        "coverage": {
            "expected_messages": len(rows_by_id),
            "prediction_records": len(predictions),
            "unique_prediction_ids": len(pred_by_id),
            "missing_ids": missing_ids,
            "unknown_ids": unknown_ids,
            "duplicate_ids": sorted(set(duplicate_ids)),
        },
        "independent_recompute": recomputed,
        "max_abs_reported_metric_delta": max(metric_deltas.values(), default=0.0),
        "perfect_message_count": len(exact_ids),
        "perfect_message_ids": exact_ids,
        "cut_error_distribution": {
            "minimum": min(error_sizes) if error_sizes else None,
            "median": statistics.median(error_sizes) if error_sizes else None,
            "mean": statistics.fmean(error_sizes) if error_sizes else None,
            "messages_with_1_cut_error": histogram[1],
            "messages_with_at_most_2_cut_errors": sum(v for k, v in histogram.items() if k <= 2),
            "total_missing_cuts": sum(misses_per_message),
            "total_extra_cuts": sum(extras_per_message),
            "histogram": {str(k): histogram[k] for k in sorted(histogram)},
        },
        "by_protocol": dict(sorted(by_protocol.items())),
        "ten_closest_messages": sorted(
            closest, key=lambda item: (item["symmetric_difference"], item["id"])
        )[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently audit whole-message perfection for classical PRE baselines"
    )
    parser.add_argument("--data", default="data/processed/core_corpus/test_novel.jsonl")
    parser.add_argument("--results-dir", default="results/baselines_public")
    parser.add_argument("--output-json", default="results/baselines_public/message_perfection_audit.json")
    parser.add_argument("--output-md", default="docs/message_perfection_audit.md")
    args = parser.parse_args()

    rows = load_rows(args.data)
    rows_by_id = {row["id"]: row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError("evaluation data contains duplicate message IDs")

    result_dir = Path(args.results_dir)
    reports = [
        audit_one(result_dir / filename, rows_by_id)
        for filename in ("netplier.json", "binaryinferno.json", "nemesys.json", "netzob.json")
    ]
    output = {
        "definition": "A message is perfect iff its predicted and gold internal-cut sets are identical.",
        "data": args.data,
        "n": len(rows),
        "reports": reports,
    }
    Path(args.output_json).write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Table 2 整报文完全正确率审计",
        "",
        "定义：仅当一条消息的预测内部切点集合与标注集合完全相同时，该消息计为完全正确。",
        f"评测集：`{args.data}`，共 {len(rows)} 条 exact-byte-novel 消息。",
        "",
        "| 方法 | 覆盖消息 | 完全正确数 | 最少切点错误 | 中位切点错误 | 仅差 1 个切点 | 漏切总数 | 多切总数 | 与原报告最大差值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        coverage = report["coverage"]
        errors = report["cut_error_distribution"]
        lines.append(
            f"| {report['tool']} | {coverage['unique_prediction_ids']}/{coverage['expected_messages']} "
            f"| {report['perfect_message_count']} | {errors['minimum']} | {errors['median']:.1f} "
            f"| {errors['messages_with_1_cut_error']} | {errors['total_missing_cuts']} "
            f"| {errors['total_extra_cuts']} | {report['max_abs_reported_metric_delta']:.2e} |"
        )
    lines.extend([
        "",
        "所有预测均使用同一 `BoundaryCounts` 定义独立重算；JSON 同时保存误差直方图、分协议完全正确计数和最接近完全正确的消息。",
        "若完全正确数为 0 且最少切点错误大于 0，则表中 0.0000 是严格集合相等定义的真实结果，而不是四舍五入。",
    ])
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
