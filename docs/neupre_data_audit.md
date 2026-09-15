# NeuPRE PDML/PCAP 数据审计

本审计记录 NeuPRE 捕获数据对 Mamba-PRE 论文能够证明和不能证明的内容。生成的数据文件是评测资产，本身不是发表结果。

## 来源与重新生成

从用户提供的 NeuPRE 工作区复制了 11 个 PCAP 文件和 7 个原始 `experiments/pdml_gt/*.json` 映射到 `data/raw/neupre_pdml/`，未修改原始 NeuPRE 项目。

完整字段记录通过本地 PDML 解析器，利用 Wireshark/tshark PDML 从 PCAP 重新生成。重新生成会处理每个数据包，而不是沿用原生成器的 1,200 包上限，并保留捕获文件名、数据包索引、方向，以及与方向无关的传输会话哈希。精确的 PCAP SHA-256 和 tshark 版本保存在 `data/generated/neupre_pdml_records/manifest.json`。

```powershell
$env:PYTHONPATH='.'
python scripts/generate_neupre_records.py `
  --pcap-dir data/raw/neupre_pdml/pcap `
  --records-dir data/generated/neupre_pdml_records `
  --pdml-module-dir "<pdml-parser-root>" `
  --include-s7comm-plus-envelope

python -m mambapre.cli prepare-neupre `
  --records-dir data/generated/neupre_pdml_records `
  --output-dir data/processed/neupre_pdml `
  --reference-data data/processed/core_corpus/train.jsonl `
  --reference-data data/processed/core_corpus/validation.jsonl
```

## 按标注范围分离的数据集

两种标注范围必须分开汇报。

| 评测文件 | 标注范围 | 消息数 | 最大字节数 | 长度 >512 的消息数 |
|---|---|---:|---:|---:|
| `test_external_full.jsonl` | 完整 Wireshark PDML 字段 | 21,937 | 1,028 | 3 |
| `test_external_full_novel.jsonl` | 完整字段，精确字节不在 核心训练池 | 21,121 | 480 | 0 |
| `test_external_envelope.jsonl` | S7comm+ TPKT/COTP 包络及不透明载荷 | 1,718 | 1,024 | 530 |
| `test_external_long_envelope_novel.jsonl` | 相同粗粒度范围，精确字节新颖且长度 >512 | 530 | 1,024 | 530 |
| `calibration_long_envelope_novel.jsonl` | 用于阈值选择的 4 个完整长报文会话 | 264 | 1,024 | 264 |
| `test_long_envelope_session_disjoint_novel.jsonl` | 6 个留出的长报文会话 | 266 | 1,024 | 266 |

完整字段协议包括 Modbus、Delta 的 Modbus 方言、DNP3、S7comm、IEC104、Omron FINS 和 LonTalk，共覆盖 7 个捕获及 58 个观测到的传输会话。S7comm+ 包络集覆盖 1 个捕获和 22 个会话，其中 10 个会话包含长度超过 512 字节的消息。长报文校准集与测试集之间不存在会话或精确消息重合。

## 完整性发现

- 所有源 PDML 映射均包含有效十六进制载荷和有序、范围合法的边界列表，且同时包含起止点。
- 重新生成的记录使用连续字段覆盖完整消息。
- 完整字段集与包络集之间不存在精确 `(protocol, bytes)` 重合。
- 完整字段/包络记录中有 816 条的原始字节出现在完整核心源训练池中；若同时要求协议别名匹配，则为 706 条。
- 3 条长度超过 512 字节的完整解析消息全部与训练池重合。因此，当前项目中长度超过 512 字节、精确字节新颖且具有完整字段标签的消息数量为 **0**。
- 精确字节新颖不能证明捕获或会话层面的独立性。核心语料规范化源数据缺少捕获/会话标识，且部分重合说明两个语料库可能共享捕获来源。

## 主张—证据边界

| 候选主张 | 当前证据 | 状态 |
|---|---|---|
| 在真实协议上的跨语料兼容性 | 21,121 条精确字节新颖的完整字段记录 | 可用，但必须说明上述来源限制 |
| 真实长报文精度 | 3 条完整字段长报文，全部在训练中见过 | 不支持 |
| 长报文包络分段 | 530 条长度为 513–1,024 字节的精确字节新颖 S7comm+ 消息 | 仅支持 TPKT/COTP 包络边界 |
| 普遍的完整字段长报文优势 | 不存在新颖完整字段长报文集 | 关键证据缺失 |

下一步数据采集目标是：具有完整解析器覆盖、与现有捕获独立，并包含足量 512 和 1,024 字节以上消息的协议语料。在此之前，论文可以汇报长序列运行时/显存，以及粗粒度 S7comm+ 包络实验，但不能将后者描述为完整协议字段恢复。

第一轮单种子模型结果及其限制记录在 `docs/neupre_first_results.md`。

## 阶段二更新（2026-08-04）

上述审计结论描述的是原始 NeuPRE 语料。项目现已另行加入捕获独立的严格 540 字节 FINS 全字段压力测试：41 条消息、11 个捕获、63 个连续字段，校准集与测试集的捕获、会话和精确字节重合均为 0。该集合包含 1 条公开真实消息和 40 条受控协议有效消息，两类来源分开汇报。

五种子结果没有支持原先期望的完整字段精度优势：Mamba-PRE 为 0.1282 ± 0.0162，参数匹配 Transformer 为 0.3262 ± 0.1026，配对差值 95% CI 为 [-0.3292, -0.0669]。因此论文已将主张收窄为“长包络分段和长序列效率优势”，并把严格全字段结果作为失败边界。详见 `docs/long_full_field_stage.md`。
