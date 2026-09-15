from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from mambapre.constants import ROLES, TYPES


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    rows = read_rows(ROOT / "data/processed/core_corpus/train.jsonl")
    nonpadding = sum(len(row["bytes"]) for row in rows)
    role_labeled = sum(sum(int(value) >= 0 for value in row["role_ids"]) for row in rows)
    type_labeled = sum(sum(int(value) >= 0 for value in row["type_ids"]) for row in rows)
    role_message_coverage = sum(all(int(value) >= 0 for value in row["role_ids"]) for row in rows)
    type_message_coverage = sum(all(int(value) >= 0 for value in row["type_ids"]) for row in rows)
    boundary_positives = sum(len(row["boundaries"]) for row in rows)
    parameters = {
        "Mamba-PRE/full learned gate": 2191502,
        "parameter-matched Transformer": 2180802,
        "supervised CNN": 2188270,
        "Bi-Mamba only": 2199274,
        "fixed-fusion dual stream": 1992846,
        "unguided learned gate": 2191502,
        "guided without semantic losses": 2191502,
        "optimized SDPA efficiency control": 2191886,
    }
    target = parameters["Mamba-PRE/full learned gate"]
    parameter_deviation = {name: (value - target) / target for name, value in parameters.items()}
    report = {
        "roles": list(ROLES),
        "types": list(TYPES),
        "structural_role_set": [role for role in ROLES if role != "payload"],
        "payload_handling": "payload role maps to gate target 0",
        "opaque_handling": "opaque byte spans use primitive type 'bytes'; no separate opaque role exists",
        "unknown_or_missing_role_handling": "role id -100 is excluded from role loss and role-guidance mask; padding is excluded by the message mask",
        "gate_tensor_shape": "B x N x L x d; each per-layer G^l is B x L x d",
        "gate_target": "t_i=1 for every known non-payload role and t_i=0 for payload",
        "gate_score_for_loss": "gbar_i=(1/(N*d))*sum_{layer,channel} G[layer,i,channel]",
        "gate_loss": "mean byte-wise probability-space BCE(gbar_i,t_i) over known-role non-padding bytes; probabilities are clipped to [1e-6,1-1e-6]",
        "boundary_loss": "mean BCEWithLogits over non-padding bytes with fixed pos_weight=8.0",
        "boundary_positive_weight": 8.0,
        "boundary_positive_weight_computed_from_data": False,
        "observed_training_negative_positive_ratio": (nonpadding - boundary_positives) / boundary_positives,
        "role_cardinality": len(ROLES),
        "type_cardinality": len(TYPES),
        "training_messages": len(rows),
        "training_protocol_counts": dict(sorted(Counter(row["protocol"] for row in rows).items())),
        "nonpadding_training_bytes": nonpadding,
        "role_labeled_bytes": role_labeled,
        "type_labeled_bytes": type_labeled,
        "role_byte_coverage": role_labeled / nonpadding,
        "type_byte_coverage": type_labeled / nonpadding,
        "fully_role_labeled_messages": role_message_coverage,
        "fully_type_labeled_messages": type_message_coverage,
        "cnn": {
            "width": 224,
            "layers": 4,
            "kernel_sizes": [3, 3],
            "dilations": [1, 2, 4, 8],
            "residual": True,
            "normalization": "pre-convolution LayerNorm and post-residual LayerNorm",
            "activation": "GELU after the first convolution",
            "dropout": 0.1,
            "receptive_field": 61,
            "receptive_field_formula": "1 + 2*(3-1)*sum(1,2,4,8)",
        },
        "parameters": parameters,
        "parameter_deviation_vs_full": parameter_deviation,
        "code_evidence": {
            "roles": "mambapre/constants.py",
            "gate_and_losses": "mambapre/losses.py",
            "gate_shape_and_cnn": "mambapre/model.py",
        },
    }
    output_json = ROOT / "results/reproducibility/method_audit.json"
    output_md = ROOT / "docs/method_reproducibility_audit.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 方法可复现性审计",
        "",
        "本文件由 `scripts/audit_method_reproducibility.py` 从实现和训练数据生成。",
        "",
        "## 门控目标与损失",
        "",
        f"- 角色集合：{', '.join(ROLES)}；结构角色是除 `payload` 外的 7 类。",
        "- `payload` 的门控目标为 0；已知非 payload 角色的目标为 1；缺失角色 `-100` 和 padding 不参与门控损失。",
        "- 每层门控 `G^l` 的真实形状是 `B x L x d`，完整输出为 `B x N x L x d`，不是 `L x 1`。",
        "- 实现先在层和通道维求均值，再对每个有效字节计算 probability-space BCE；最后按有效字节取均值。",
        "- 边界损失固定 `pos_weight=8.0`，不是由当前训练集动态计算。",
        "",
        "## 标签覆盖",
        "",
        f"- 训练消息：{len(rows)}；非 padding 字节：{nonpadding}。",
        f"- role/type 均覆盖 {role_labeled}/{nonpadding} 字节和 {role_message_coverage}/{len(rows)} 消息。",
        f"- 数据中的负/正边界比为 {(nonpadding - boundary_positives) / boundary_positives:.6f}，仅作为审计值；主实验仍使用固定权重 8。",
        "",
        "## CNN",
        "",
        "- 四个 residual layers，宽度 224；每层两次 kernel-3 卷积，dilations={1,2,4,8}。",
        "- 第一卷积前 LayerNorm，第一卷积后 GELU，残差相加后再次 LayerNorm，dropout=0.1。",
        "- 感受野为 `1 + 2*(3-1)*(1+2+4+8) = 61` bytes。",
        "",
        "## 参数量",
        "",
        "| Variant | Parameters | Deviation vs full |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | {value:,} | {100 * parameter_deviation[name]:+.2f}% |" for name, value in parameters.items())
    lines.extend((
        "",
        "## 已发现的稿件偏差",
        "",
        "- 当前稿件把角色写成包含 `identifier`，但代码中没有该类，应改为真实八类。",
        "- 当前 Bi-Mamba 公式是示意性相加，但实现是双向输出拼接后 Linear-GELU-Dropout，再残差与 LayerNorm；需按代码改写。",
        "- 当前融合公式额外写了输入残差，但实现的 SpatialFusion 是加权融合后 LayerNorm；残差位于各分支内部，需按代码改写。",
        "- `mean gate response` 需明确为跨层、跨通道求均值后进行 byte-wise BCE，不能写成仅约束全局均值。",
        "",
    ))
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    main()
