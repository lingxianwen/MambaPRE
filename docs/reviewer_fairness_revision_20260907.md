# 投稿前公平性修订记录（2026-09-07）

## 结论

本轮修订解决了损失描述、Transformer 后端、FINS 数据独立性命名、Source OOD 命名、监督神经基线和 Table 2 整报文正确率六类问题。所有新增数字均来自保存的逐种子 JSON 或独立审计结果。

## 监督与损失

- 全部神经模型共享相同数据、边界/角色/类型标注、优化器设置、训练轮数上限、早停和五个随机种子。
- Mamba-PRE 额外使用架构专属门控正则项 $L_g$；该项由同一角色标注派生。
- 论文不再使用“identical losses”，也不再声称实验只隔离 encoder architecture；改为评估完整 structure-aware architecture。

## 新增监督 CNN 对照

- 四层残差膨胀 CNN，宽度 224；每层包含两个核宽 3 的 GELU Conv1D、pre/post LayerNorm 和 0.1 dropout，dilation 为 `{1,2,4,8}`，总感受野 61 字节、双侧半径 30 字节。
- 参数量 2,188,270；Mamba-PRE 为 2,191,502，差异为 -0.15%。
- 使用与 Transformer 相同的边界/角色/类型损失，不含门控损失。
- 五种子 Boundary F1：

| 测试集 | 监督 CNN（均值±标准差） | Mamba-PRE − CNN，配对 95% CI |
|---|---:|---:|
| ID novel | 0.9998±0.0001 | -0.0002 [-0.0004, 0.0001] |
| Source-held-out | 0.8895±0.0112 | +0.0062 [-0.0084, 0.0209] |
| External full | 0.7063±0.0170 | +0.0250 [-0.0105, 0.0604] |
| Long session | 0.6860±0.0572 | +0.0209 [-0.0966, 0.1384] |
| Strict long FINS | 0.2016±0.0304 | -0.0735 [-0.1294, -0.0176] |

长包络上 CNN 与 Mamba-PRE 差异不显著，因此论文将结论限制为 Mamba-PRE 优于匹配的标准 Transformer，而不声称优于所有神经编码器。

## 长程依赖诊断

- 长包络测试的全部金标边界都位于 offset 1--7，之后为不透明 payload；因此该数据不能直接验证远距离边界恢复。
- 诊断的唯一主对比是 CNN 30 字节半径之外的 payload 非边界位置，指标为每万字节假边界数。
- Mamba-PRE：12.73；监督 CNN：17.75；五种子配对差为 -5.02，95% CI `[-18.64, 8.61]`。
- 方向性趋势有利于 Mamba-PRE，但区间跨 0，不能据此断言 long-range modeling 是性能增益来源。摘要、实验讨论与结论均已收缩该主张。

机器可读结果见 `results/diagnostics/long_payload_false_positives.json`，复现脚本为 `scripts/diagnose_long_payload_false_positives.py`。

## Transformer 效率后端

- 标准对照：PyTorch 2.3 `TransformerEncoderLayer`，参数量 2,180,802。
- 优化对照：四层、四头、$d=176$、FFN=652，参数量 2,191,886（相对 Mamba-PRE +0.02%）。每头维度 44，满足高效 kernel 的整除约束。
- FP32 强制 memory-efficient SDPA；BF16 强制 Flash SDPA；同时禁用 MHA fused fastpath，保证请求的 SDPA 后端实际执行。
- 定长 benchmark 中所有位置均有效，因此显式省略无作用的 padding mask；报告记录 `full_length_padding_mask_elided=true`。
- 4096 字节、batch 8：

| 精度/对照 | Mamba ms | Transformer ms | T/M | Mamba MB | Transformer MB | Mamba 节省显存 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 标准 | 34.813 | 83.335 | 2.39× | 356.8 | 2,195.2 | 83.7% |
| FP32 SDPA | 34.813 | 79.676 | 2.29× | 356.8 | 269.7 | -32.3% |
| BF16 标准 | 20.743 | 16.694 | 0.80× | 213.0 | 194.7 | -9.4% |
| BF16 Flash | 20.743 | 15.500 | 0.75× | 213.0 | 188.8 | -12.9% |

因此最终稿仅保留“FP32 长序列延迟优势”，并明确 BF16 Flash attention 反转延迟与显存排序。

## 数据命名

- `Source OOD` 统一改为 `Source-held-out`；正文明确 302 条中 286 条属于训练出现过的五个协议，仅 16 条为未见 IEC104。
- FINS 改为“controlled protocol-valid variants and one held-out public capture”。正文明确三组 controlled variants 用于校准，七组用于测试；这些组不是独立真实采集的 captures。

## Table 2 整报文正确率审计

- 四个传统工具均覆盖 577/577 条唯一消息，无缺失、未知或重复 ID；正文改称 “A separate verification script”，不再使用可能暗示第三方审计的 “independent audit”。
- 独立重算与原报告所有指标最大绝对差为 0。
- 最接近完全正确的输出仍分别相差 3（NetPlier）、3（BinaryInferno）、2（NEMESYS）和 5（Netzob）个内部切点。
- 因为整报文正确率要求预测切点集合与金标集合完全相等，所以 0.0000 是真实严格结果，不是四舍五入或漏跑。

详细机器可读证据见 `results/baselines_public/message_perfection_audit.json`。

## 验证

- 服务器完整测试：28 passed，1 条 PyTorch nested-tensor 性能提示，无失败。
- LaTeX：5 页，其中正文 4 页、参考文献第 5 页；无 overfull、未定义引用或未定义文献。
- PDF：Letter 尺寸，全部字体已嵌入；逐页渲染未发现裁切、重叠或乱码。

## 额外复现参数

- 边界损失使用固定 `pos_weight=8`，所有模型一致；该数值不是由当前训练集的负/正样本比动态计算。
- role/type 标注覆盖全部 5,997 条训练消息，即 218,072/218,072 个非 padding 字节（100%）。
- `source-held-out` 中的 source 明确定义为单独提供的评测输入文件，不代表 capture-level 独立性。
