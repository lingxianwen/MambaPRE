# 审稿问题—证据—论文动作矩阵

| 审稿风险 | 新证据 | 结论 | 已完成的论文动作 |
|---|---|---|---|
| “identical losses” 与专属 $L_g$ 冲突 | 训练配置与 loss 代码审计 | Transformer 共享 boundary/role/type 监督；Mamba 额外使用由同一 role annotation 导出的结构正则 | 改为 “matched data, shared task supervision, and parameter budgets”；实验部分逐项说明共享项和额外 $L_g$ |
| CNN 与 Mamba 在 long envelope 统计持平 | 二候选距离诊断，五种子，64–1024 字节 | Mamba Top-1 0.4856，CNN 0.5112，差值 CI 跨 0；距离斜率 CI 跨 0 | 采用 Branch B；引言、摘要、结论均不再把收益归因于 long-range modeling |
| Transformer 内存理论与 Flash/SDPA 冲突 | 标准、memory-efficient SDPA、Flash 三 backend 测量 | Flash/SDPA 避免物化 $L\times L$ score tensor，但算术仍为二次 | 修订复杂度段；效率 claim 限定到当前实现与 FP32 latency |
| 效率可能来自 kernel 差异 | optimized SDPA/Flash 参数匹配对照 | FP32 Mamba 快，但显存更大；BF16 Flash 速度和显存均反转 | 摘要、图注、正文同时报告反转结果 |
| `capture-disjoint FINS` 过强 | 数据生成审计 | 40 条为 protocol-valid controlled variants，不是独立真实采集；另有 1 条 held-out public message | 删除 `capture-disjoint`；明确逻辑组隔离、零字节重叠及独立性限制 |
| `Source OOD` 误导 | 302 条中 286 条为 seen protocol，仅 16 条 IEC104 | 不是 protocol-level OOD | 全文改为 `source-held-out`，并定义 source 为单独提供的评估文件 |
| 缺少 supervised neural PRE baseline | 任务接口/粒度审计 + common-pipeline CNN/Transformer | 既有系统无法在相同 byte-label protocol 下公平复现 | Related Work 说明真实原因；加入参数匹配 CNN 与 Transformer 控制 |
| 主表 strict FINS 数值疑似不一致 | 105 次检查点逐报文复算 | 精确均值 0.12814958，正确四位小数是 0.1281 | 摘要、表 1、消融表统一为 0.1281 |
| 消融未完全分离组件 | A–F 五种子析因矩阵 | 双流相对 Bi-Mamba 有益；guidance 在 learned gate 内有益；但 fixed fusion 与 full 无法区分 | 表 3 换成 A/B/C-E/D/F；明确 learned routing 非必要结论 |
| 原 role gate 可能导致 FINS 漏检 | binary、structural-only、unguided 五种子与逐层 gate | structural-only FINS 增益不显著且损害 long envelope | 保留为失败诊断，不替换模型，不再继续调门控目标 |
| 方法公式与代码不一致 | `model.py`/`losses.py` 自动审计 | Bi-Mamba 是 concat-projection；分支内残差；融合后无额外残差；gate 接收位置；$L_g$ 是跨层/通道均值后的逐字节 BCE | 重写公式；修正图 1 的错误残差箭头；写明 mask、taxonomy 和 label coverage |
| CNN 不可复现 | 实现审计 | $d=224$，4 residual layers，每层两次 $k=3$，dilation 1/2/4/8，RF 61，2,188,270 参数 | 数据集/控制段补齐一行完整描述 |
| Table 2 perfection 全为 0 像失败 | 577 条逐报文输出覆盖与 cut-error audit | 输出完整；每个工具最接近预测仍差 2–5 cuts | 表附近解释严格定义、完整覆盖与复算；传统工具只作语境对照 |
| 2026 相关工作缺失 | 出版社/会议/DBLP 元数据核验 | TransRE、ICPPRAG、FieldWeaver 已正式出版；PVParser 保守按 CCS 2026 full version 引用 | Related Work 增加 transfer、knowledge/LLM、visual-language、long-payload search 四条路线 |
| 内容页数超限 | 完整 TeX 编译与逐页渲染 | 5 页总长，前 4 页内容，第 5 页参考文献 | 删除重复引言，压缩措辞与图纵横比；未缩小正文字号 |

## 当前科学定位

论文能够支持：结构感知双流 Mamba 在统一监督管线下显著优于 matched standard Transformer 的 long-envelope F1，并在 FP32 长序列延迟上优于 optimized SDPA。

论文不能支持：Mamba 普遍优于 CNN、性能增益由长距离依赖导致、learned gate 是必要组件、Mamba 在所有 backend/precision 下更快或更省显存、对完整长协议字段具有普遍泛化性。

