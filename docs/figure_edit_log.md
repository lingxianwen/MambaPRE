# 论文图形生成与编辑记录

## 图 1：Mamba-PRE Overview

- 目标版式：双栏宽度；
- 编辑源：`ICASSP2026_Paper_Templates/Template.tex` 内的 TikZ；
- 数据流：字节输入 → 嵌入 → 双向 Mamba/局部卷积 → 角色引导门控 → 边界、角色和类型输出；
- 虚线仅表示训练期监督，推理期不输入协议角色；
- 未使用任何外部图标或手工栅格图片。

## 图 2：长度扩展性

- 目标版式：单栏宽度、双面板；
- 原始数据：`results/scaling/comparison.json`；
- 可复现数据：`figures/data/scaling_speed_memory.csv`；
- 生成脚本：`scripts/plot_scaling_figure.py`；
- 输出：`figures/scaling_tradeoff.pdf`、`.svg`、`.png`；
- 面板 (a)：Transformer 延迟除以 Mamba 延迟，1 表示交叉点；
- 面板 (b)：Mamba 相对 Transformer 的 CUDA 峰值显存降幅；
- 未手工修改数值、坐标轴、测点、基线名称或曲线。

## 版面与导出核验

- 图 1 置于正文第 2 页顶部，按双栏宽度显示；
- 图 2 替换原 FP32 缩放表，置于正文第 4 页单栏，并同时报告 FP32/BF16；
- 正文与技术图表仅占第 1--4 页，参考文献由 `\clearpage` 独占第 5 页；
- 最终 PDF 为 Letter 尺寸、共 5 页，无 overfull、未定义引用或未嵌入字体。
# 2026-09-05 图 1 与图 2 重绘

- 图 1 改为 LaTeX 目录内的原生 TikZ 矢量图，编辑源为 `fig1_overview.tex`，独立导出入口为 `Fig1_standalone.tex`。
- 图 1 保留字节输入、字节/位置嵌入、双向 Mamba 全局流、局部 Conv1D 流、字节级门控、融合和三个预测任务；删除了正文中已给出的完整损失公式与拥挤的中间张量标签。
- 实线灰色箭头表示训练与推理共享的计算路径，橙色虚线表示仅训练监督；`H^N` 明确连接预测头。
- 图 2 改由 LaTeX 目录内的 `make_fig2_scaling.py` 从冻结的 `comparison.json` 重新生成，未手工修改数值、坐标范围或基线身份。
- 图 2 已在 2026-09-07 按后端公平性审计重绘：同时区分标准 PyTorch 与强制 SDPA/Flash 路径，并标出 FP32 相对优化 SDPA 的 2.29 倍延迟优势及 -32.3% 显存差值；旧版 4.00 倍/95.7% 标注不再使用。
- 最终产物包括 PDF/SVG、PNG 预览和 TIFF 高分辨率版本；论文使用矢量版本。

## 本轮未修改的数据元素

- 所有延迟、显存和相对比例数值；
- 横轴消息长度和纵轴范围；
- FP32/BF16 身份；
- RTX A6000、batch size 8、10 次预热和 30 次计时条件。
