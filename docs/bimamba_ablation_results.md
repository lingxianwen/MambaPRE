# 参数匹配 Bi-Mamba-only 五种子消融

## 目的

该实验检验完整 Mamba-PRE 相对纯双向状态空间编码器的收益，补足“局部流是否有实际价值”的审稿证据。为避免容量差异，Bi-Mamba-only 将隐藏维度设为 140，共 2,199,274 个参数；完整 Mamba-PRE 为 2,191,502 个参数，差异为 +0.35%。

两者使用相同训练数据、优化器、训练轮数、早停规则、阈值选择方法和五个配对种子：1337、2027、3407、4701、9001。

## 结果

| 测试集 | Bi-Mamba-only 边界 F1 | 完整模型减去 Bi-Mamba | 配对 95% CI |
|---|---:|---:|---:|
| exact-byte-novel 短测试 | 0.9995 ± 0.0001 | +0.0001 | [-0.0002, 0.0005] |
| source-held-out | 0.8884 ± 0.0038 | +0.0074 | [0.0034, 0.0113] |
| session-disjoint 长包络 | 0.6180 ± 0.0445 | +0.0890 | [0.0238, 0.1541] |
| capture-disjoint strict FINS | 0.1315 ± 0.0167 | -0.0034 | [-0.0329, 0.0261] |

无引导双流相对 Bi-Mamba-only 的长包络增益为 +0.0486，配对 95% CI 为 [-0.0282, 0.1254]，区间跨零。由此只能认为完整的“局部流 + 角色引导融合”组合在长包络任务上显著优于参数匹配 Bi-Mamba，不能声称局部流在没有引导时独立产生显著收益。所有结构收益均未转移到 strict FINS。

## 复现

- 配置：`configs/bimamba_param_matched.json`
- 单种子脚本：`scripts/run_bimamba_ablation_seed.sh`
- 安全队列：`scripts/queue_bimamba_ablation.sh`
- 聚合脚本：`scripts/aggregate_bimamba_ablation.py`
- 聚合结果：`results/bimamba_ablation_multiseed/aggregate.json`

聚合脚本同时计算完整模型和无引导双流相对 Bi-Mamba-only 的逐种子差值、样本标准差及双侧 Student-$t$ 95% 置信区间。
