# 2026 年相关工作元数据核验

核验日期：2026-09-08。以下条目只采用出版社、会议官网、DBLP 或作者提交的 arXiv 页面；在修改论文前均已核对题名、作者、年份与出版状态。

## 可作为正式出版物引用

1. **TransRE** — Yanyang Zhao, Zhengxiong Luo, Wenlong Zhang, Feifan Wu, Yuanliang Chen, Fuchen Ma, Qi Xu, Heyuan Shi, and Yu Jiang, “Protocol Reverse Engineering via Deep Transfer Learning,” *Proceedings of the ACM on Software Engineering*, vol. 3, FSE, pp. 636–656, 2026. DOI: `10.1145/3797124`。FSE 2026 官方页面和 DBLP 均确认。定位：利用标准协议的先验语法和域适配向目标私有协议迁移。

2. **ICPPRAG** — Yahui Yang, Yangyang Geng, Mufeng Wang, and Qiang Wei, “Reverse engineering for industrial control proprietary protocols based on multi-source knowledge fusion and LLM inference,” *Information Fusion*, vol. 135, Art. 104468, 2026. DOI: `10.1016/j.inffus.2026.104468`。Elsevier 页面确认卷、文章号和 DOI。定位：通过多源协议知识库、RAG 和 LLM 做跨协议结构/语义推断。

3. **FieldWeaver** — Qichao Yang, Fangfang Zhao, Xiaokang Yin, Ruijie Cai, and Shengli Liu, “FieldWeaver: A visual–language approach to binary protocol format inference,” *Computer Networks*, vol. 287, Art. 112573, 2026. DOI: `10.1016/j.comnet.2026.112573`。Elsevier 页面确认卷、文章号和 DOI。定位：把报文可视化纹理边界与 LLM 语义区域融合。

## 已确认论文身份，但最终卷期页码尚不宜臆造

4. **PVParser** — Chuan Sheng, Shan Jiang, Xiaogang Zhu, Wanlun Ma, Jianming Zhao, Yu Yao, Sheng Wen, and Yang Xiang, “Recovering Process Variables from Industrial Network Traffic via Search-Based Optimization,” CCS 2026 full version, arXiv:`2608.16403`, 2026。arXiv 作者页面明确标注其为 CCS 2026 论文全文；在尚未取得 ACM proceedings 元数据前，BibTeX 应保守写为 CCS 2026，并附 arXiv 标识，不填写未经核验的页码或 ACM DOI。定位：使用周期检测与 MCTS，在长、部署相关 payload 中搜索过程变量字段，避免顺序分割误差传播。

## 对 Mamba-PRE 的准确定位

- TransRE 主要回答“如何迁移已有协议语法知识”；Mamba-PRE 研究统一监督条件下的字节级序列编码与局部/全局路由。
- ICPPRAG 与 FieldWeaver 引入外部知识、视觉表示或 LLM 语义推理；Mamba-PRE 不依赖推理时的外部知识或 LLM。
- PVParser 针对过程变量和长 payload 的搜索式恢复；Mamba-PRE 是逐字节边界序列标注器。二者任务范围和评价粒度不能直接等同。
- 因此，新颖性不能写成笼统的“首个 Mamba for PRE”，而应落在可检验的结构感知双流架构、门控监督设计，以及受控远距离依赖条件下的证据上。

