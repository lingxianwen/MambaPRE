from __future__ import annotations

import argparse
import json
from pathlib import Path

from mambapre.reporting import add_macro, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="生成无完全重复消息的测试集对比面板")
    parser.add_argument("--guided", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--classical-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    reports = {
        "Guided Mamba-PRE": add_macro(load_json(args.guided)),
        "Param-matched Transformer": add_macro(load_json(args.transformer)),
    }
    display = {
        "binaryinferno": "BinaryInferno",
        "netzob": "Netzob",
        "netplier": "NetPlier",
        "nemesys": "NEMESYS",
    }
    for filename, name in display.items():
        reports[name] = add_macro(load_json(Path(args.classical_dir) / f"{filename}.json"))

    protocols = sorted(
        set.intersection(*(set(report["by_protocol"]) for report in reports.values()))
    )
    result = {
        "scope": "已排除与 Mamba-PRE 训练集和验证集完全相同字节串的测试消息",
        "n": next(iter(reports.values()))["overall"]["n"],
        "methods": reports,
        "comparison_notes": {
            "Guided Mamba-PRE": "有监督字节级模型；此面板使用种子 1337",
            "Param-matched Transformer": "监督、优化器、数据划分和参数预算相同",
            "classical": "无监督流量分析工具，按协议和方向分组推断",
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# 无完全重复消息的测试集对比",
        "",
        f"与训练池不存在完全相同字节串的测试消息：**{result['n']}** 条。",
        "",
        "## 总体微平均与协议宏平均指标",
        "",
        "| 方法 | 边界 F1 | 严格字段 F1 | 整报文完全正确率 | 协议宏平均边界 F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        overall = report["overall"]
        lines.append(
            f"| {name} | {overall['boundary_f1']:.4f} | {overall['exact_field_f1']:.4f} "
            f"| {overall['message_perfection']:.4f} | {report['macro_protocol']['boundary_f1']:.4f} |"
        )
    lines.extend([
        "",
        "## 分协议边界 F1",
        "",
        "| 协议 | " + " | ".join(reports) + " |",
        "|---|" + "---:|" * len(reports),
    ])
    for protocol in protocols:
        values = " | ".join(
            f"{report['by_protocol'][protocol]['boundary_f1']:.4f}"
            for report in reports.values()
        )
        lines.append(f"| {protocol} | {values} |")
    lines.extend([
        "",
        "Mamba-PRE 与 Transformer 构成受控神经架构比较；四个传统工具是无监督流量分析参照。",
        "两类方法的监督条件和计算方式不同，论文中必须持续披露这一点。",
        "",
        "所有方法均使用 `mambapre.metrics` 中相同的边界、严格字段和整报文定义重新评分。",
    ])
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
