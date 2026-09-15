from __future__ import annotations

import argparse
import json
from pathlib import Path

from mambapre.reporting import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Mamba and Transformer scaling")
    parser.add_argument("--input-dir", default="results/scaling")
    parser.add_argument("--output-json", default="results/scaling/comparison.json")
    parser.add_argument("--output-md", default="docs/scaling_results.md")
    args = parser.parse_args()
    root = Path(args.input_dir)
    output = {"precisions": {}}
    lines = [
        "# 运行时间与显存缩放结果",
        "",
        "单张 RTX A6000，batch size 8，预热 10 次，正式测量 30 次。标准 Transformer 使用 PyTorch 自动路径；优化 SDPA 对照在 FP32 强制 memory-efficient kernel，在 BF16 强制 Flash kernel。",
    ]
    for precision in ("float32", "bfloat16"):
        mamba = load_json(root / f"guided_mamba_{precision}.json")
        transformer = load_json(root / f"matched_transformer_{precision}.json")
        transformer_sdpa = load_json(root / f"matched_transformer_sdpa_{precision}.json")
        mamba_rows = {row["length"]: row for row in mamba["results"]}
        transformer_rows = {row["length"]: row for row in transformer["results"]}
        sdpa_rows = {row["length"]: row for row in transformer_sdpa["results"]}
        rows = []
        for length in sorted(set(mamba_rows) & set(transformer_rows) & set(sdpa_rows)):
            m_row, t_row, f_row = mamba_rows[length], transformer_rows[length], sdpa_rows[length]
            if any(row["status"] != "ok" for row in (m_row, t_row, f_row)):
                continue
            latency_speedup = t_row["median_latency_ms"] / m_row["median_latency_ms"]
            memory_reduction = 1.0 - m_row["peak_memory_mb"] / t_row["peak_memory_mb"]
            flash_speedup = f_row["median_latency_ms"] / m_row["median_latency_ms"]
            flash_memory_reduction = 1.0 - m_row["peak_memory_mb"] / f_row["peak_memory_mb"]
            rows.append({
                "length": length,
                "mamba_latency_ms": m_row["median_latency_ms"],
                "transformer_latency_ms": t_row["median_latency_ms"],
                "mamba_speedup": latency_speedup,
                "mamba_memory_mb": m_row["peak_memory_mb"],
                "transformer_memory_mb": t_row["peak_memory_mb"],
                "mamba_memory_reduction": memory_reduction,
                "sdpa_transformer_latency_ms": f_row["median_latency_ms"],
                "mamba_speedup_vs_sdpa": flash_speedup,
                "sdpa_transformer_memory_mb": f_row["peak_memory_mb"],
                "mamba_memory_reduction_vs_sdpa": flash_memory_reduction,
            })
        crossover = next((row["length"] for row in rows if row["mamba_speedup"] > 1), None)
        output["precisions"][precision] = {
            "mamba_parameters": mamba["parameters"],
            "transformer_parameters": transformer["parameters"],
            "sdpa_transformer_parameters": transformer_sdpa["parameters"],
            "standard_attention_backend": transformer.get("attention_backend_requested", "auto"),
            "sdpa_attention_backend": transformer_sdpa["attention_backend_requested"],
            "first_measured_mamba_latency_crossover": crossover,
            "rows": rows,
        }
        lines.extend([
            "",
            f"## {precision}",
            "",
            "| 字节数 | Mamba ms | 标准 T ms | 优化 SDPA T ms | T/M 标准 | T/M SDPA | Mamba MB | 标准 T MB | SDPA T MB | 相对 SDPA 节省显存 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                f"| {row['length']} | {row['mamba_latency_ms']:.3f} | "
                f"{row['transformer_latency_ms']:.3f} | {row['sdpa_transformer_latency_ms']:.3f} | "
                f"{row['mamba_speedup']:.2f}x | {row['mamba_speedup_vs_sdpa']:.2f}x | "
                f"{row['mamba_memory_mb']:.1f} | {row['transformer_memory_mb']:.1f} | "
                f"{row['sdpa_transformer_memory_mb']:.1f} | "
                f"{100 * row['mamba_memory_reduction_vs_sdpa']:.1f}% |"
            )
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
