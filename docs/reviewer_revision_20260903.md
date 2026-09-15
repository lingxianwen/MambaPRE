# 2026-09-03 审稿风险修改记录

本轮按“主张必须与证据范围一致”的原则修改论文，图 1 及其 caption 暂不改动。

## 已完成修改

- 将任务重新表述为离散字节序列的密集分割，强调边界检测需要局部模式与长程上下文证据，提高 ICASSP 的序列信号处理契合度。
- 将摘要中的门控描述从既成事实改为训练目标：对结构字节偏向局部处理、对载荷字节偏向全局建模。
- 将短测试和传统基线表中的 `leakage-resistant` 收缩为可直接证明的 `exact-byte-novel`。
- 明确 source-held-out 集不是协议完全隔离：286 条消息来自五个训练协议，另有 16 条未见 IEC104；因此不再把它笼统描述为 unseen-protocol OOD。
- 将错误的 `direction embedding` 改为代码实际实现的 learned absolute-position embedding。
- 明确最大输入和训练窗口为 4,096 字节，且当前训练样本均未超过该长度。
- 在外部七协议表增加宏平均：Mamba-PRE 为 0.8101，Transformer 为 0.8229，避免 LonTalk 的 12,028 条消息掩盖协议间差异。
- 明确效率比较的实现条件：PyTorch 2.3 inference mode、相同精度、不使用 `torch.compile`；Transformer 使用标准 `nn.TransformerEncoderLayer`/多头注意力，Mamba 使用 `mamba-ssm` 2.2.2 selective-scan CUDA kernel。速度和显存结论限定为相对于该标准 PyTorch Transformer 实现。

## 已完成的关键消融

普通 `d=128` Bi-Mamba 只有 1,920,654 个参数，直接比较会混入容量差异。本轮新增 `d=140` 的参数匹配 Bi-Mamba-only：

- Bi-Mamba-only：2,199,274 参数；
- 完整 Mamba-PRE：2,191,502 参数；
- 差异：+0.35%；
- 训练数据、优化器、轮数、早停、阈值选择和五个随机种子均保持一致；
- 评测包括 exact-byte-novel 短测试、source-held-out、session-disjoint 长包络和 capture-disjoint strict FINS。

安全队列在 GPU 0 连续三次满足显存占用低于 4,096 MiB 且利用率低于 20% 后启动，五个种子均已完成。参数匹配 Bi-Mamba-only 的长包络 F1 为 $0.6180\pm0.0445$，strict FINS F1 为 $0.1315\pm0.0167$。完整模型相对 Bi-Mamba-only 的长包络配对增益为 0.0890，95% CI 为 $[0.0238,0.1541]$；strict FINS 差异为 -0.0034，区间跨零。

无引导双流相对 Bi-Mamba-only 的长包络点增益为 0.0486，但 95% CI $[-0.0282,0.1254]$ 跨零。因此论文只主张“角色引导的双流整体”获得支持，不主张局部卷积分支脱离引导后具有独立显著收益。Table 4 已加入 Bi-Mamba-only 行。
