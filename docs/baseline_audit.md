# 传统基线直接复现实验审计

## 比较范围

本实验只把能够从消息字节直接产生字段切分的公开实现放入同一张字段边界表。所有方法读取同一份 `test_novel.jsonl`，预测结果统一转换为逐消息字段区间，再由 `baselines/common.py` 计算边界 precision、recall、F1、严格 exact-field F1 和整报文完全正确率。测试集含 577 条与训练池不存在完全相同字节串的消息；阈值不会在该测试集上重新选择。

| 方法 | 实际运行的公开源码路径 | 用于比较的输出 | 关键披露 |
|---|---|---|---|
| BinaryInferno | `Baseline/binaryinferno-main/binaryinferno` | 语义驱动字段切分 | 通过原项目入口逐协议、逐方向运行 |
| NEMESYS | `Baseline/nemesys-master/nemesys-master/src` | BCDG 字段切分 | 固定 `sigma=0.6`；不启用后续 PCA refinement |
| Netzob | `Baseline/netzob-master/netzob-master/src` | 静态对齐后的字段切分 | 调用源码中的 `Format.splitStatic` |
| NetPlier | `Baseline/NetPlier-master/NetPlier-master/netplier` | 对齐阶段产生的切分 | NetPlier 的论文主任务是关键词推断；这里只评估其公开对齐子模块，不能解释为完整 NetPlier 任务精度 |

ICE-PRE 依赖固件解析器或可执行实现，ProtocolGPT 的主要输出是协议状态机。二者输入假设或预测目标与纯流量字段边界检测不一致，因此只在相关工作中讨论，不纳入同一 F1 排名。

## 公平性约束

- 四个传统方法使用完全相同的 577 条测试消息和相同的协议、方向分组。
- 所有边界和字段指标均由同一个评分器从逐消息预测重新计算，而不是抄录原论文数字。
- Mamba-PRE 与 Transformer 共享训练数据、轮数、长报文采样比例、验证集阈值规则和评估代码。
- 传统方法是无监督的组级分析器，神经模型是有监督学习器；表格用于给出 PRE 语境，不把性能差距完全归因于编码器。
- 传统工具通常没有随机种子；五种子置信区间只用于神经模型之间的受控比较。

## 主张—证据对应

| 论文主张 | 必需证据 | 当前状态 |
|---|---|---|
| Mamba-PRE 在公开测试上优于传统字段切分方法 | 四个源码适配器的统一评分结果 | 已完成，结果位于 `results/baselines_public/` |
| 优势不是旧脚本或指标口径造成的 | 逐消息预测、统一评分器、源码路径和设置 | 已由适配器 JSON 保存 |
| 长序列优势来自受控架构比较 | 参数匹配 Transformer、相同训练与阈值、长度分桶、速度和显存 | 由 P4 与 scaling 实验提供，不使用传统方法排名替代 |

## 完整运行结果

| 方法 | Boundary P | Boundary R | Boundary F1 | 严格字段 F1 | 整报文完全正确率 |
|---|---:|---:|---:|---:|---:|
| NetPlier alignment | 0.4483 | 0.6764 | 0.5392 | 0.2445 | 0.0000 |
| BinaryInferno | 0.7284 | 0.3312 | 0.4553 | 0.1496 | 0.0000 |
| NEMESYS BCDG | 0.5983 | 0.2925 | 0.3929 | 0.1044 | 0.0000 |
| Netzob static | 0.6342 | 0.2660 | 0.3748 | 0.2004 | 0.0000 |

这些数值由 `results/baselines_public/*.json` 直接汇总，并已写入论文系统对比表。NetPlier 行始终标记为 alignment segmentation，避免把子模块的字段切分分数误述为其完整关键词推断性能。
