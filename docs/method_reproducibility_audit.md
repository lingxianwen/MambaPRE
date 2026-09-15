# 方法可复现性审计

本文件由 `scripts/audit_method_reproducibility.py` 从实现和训练数据生成。

## 门控目标与损失

- 角色集合：constant, length, sequence, opcode, address, count, payload, checksum；结构角色是除 `payload` 外的 7 类。
- `payload` 的门控目标为 0；已知非 payload 角色的目标为 1；缺失角色 `-100` 和 padding 不参与门控损失。
- 每层门控 `G^l` 的真实形状是 `B x L x d`，完整输出为 `B x N x L x d`，不是 `L x 1`。
- 实现先在层和通道维求均值，再对每个有效字节计算 probability-space BCE；最后按有效字节取均值。
- 边界损失固定 `pos_weight=8.0`，不是由当前训练集动态计算。

## 标签覆盖

- 训练消息：5997；非 padding 字节：218072。
- role/type 均覆盖 218072/218072 字节和 5997/5997 消息。
- 数据中的负/正边界比为 2.282388，仅作为审计值；主实验仍使用固定权重 8。

## CNN

- 四个 residual layers，宽度 224；每层两次 kernel-3 卷积，dilations={1,2,4,8}。
- 第一卷积前 LayerNorm，第一卷积后 GELU，残差相加后再次 LayerNorm，dropout=0.1。
- 感受野为 `1 + 2*(3-1)*(1+2+4+8) = 61` bytes。

## 参数量

| Variant | Parameters | Deviation vs full |
|---|---:|---:|
| Mamba-PRE/full learned gate | 2,191,502 | +0.00% |
| parameter-matched Transformer | 2,180,802 | -0.49% |
| supervised CNN | 2,188,270 | -0.15% |
| Bi-Mamba only | 2,199,274 | +0.35% |
| fixed-fusion dual stream | 1,992,846 | -9.06% |
| unguided learned gate | 2,191,502 | +0.00% |
| guided without semantic losses | 2,191,502 | +0.00% |
| optimized SDPA efficiency control | 2,191,886 | +0.02% |

## 已发现的稿件偏差

- 当前稿件把角色写成包含 `identifier`，但代码中没有该类，应改为真实八类。
- 当前 Bi-Mamba 公式是示意性相加，但实现是双向输出拼接后 Linear-GELU-Dropout，再残差与 LayerNorm；需按代码改写。
- 当前融合公式额外写了输入残差，但实现的 SpatialFusion 是加权融合后 LayerNorm；残差位于各分支内部，需按代码改写。
- `mean gate response` 需明确为跨层、跨通道求均值后进行 byte-wise BCE，不能写成仅约束全局均值。
