# Mamba-PRE

Mamba-PRE is a reproducible research codebase for byte-level protocol field
segmentation. Its main model combines complementary local and global
representations through:

1. a bidirectional selective-state-space Mamba stream for broader contextual structure;
2. a depthwise-convolution stream for rigid local header patterns; and
3. a position-conditioned, channel-wise byte gate that fuses the two streams.

The structure-aware variant uses field-role labels only during training to
regularize the learned fusion behavior. Gold roles are never supplied at
inference. Unguided and fixed-fusion variants remain explicit controls; the
released results do not establish that learned routing is necessary or that
the model recovers distant dependencies.

The repository also contains parameter-matched Transformer, supervised
dilated-CNN, and packed bidirectional recurrent controls, plus unidirectional
Mamba and plain BiMamba ablations. The four-layer BiGRU configuration has
2,191,922 parameters versus 2,191,502 for Mamba-PRE and uses the same byte
embeddings, semantic heads, training data, optimization, and threshold policy.
Its five-seed runner is `scripts/run_bigru_seed_pipeline.sh`; the CNN runner is
`scripts/run_cnn_seed_pipeline.sh`.

The default server backend is the original official Mamba selective-scan CUDA
kernel. Mamba-2 remains available as `backend=mamba2`; its `headdim=32` keeps a
128-wide model's fused channel-last projection strides 8-aligned. The current
server's NVIDIA 520 driver cannot load Triton 2.3 kernels, so Mamba-2 results
must not be reported from that machine unless its driver is upgraded. Changing
width or backend should always be followed by `scripts/smoke_cuda.py`.

## Current dataset

The first reproducible benchmark is a normalized five-protocol industrial corpus:

- five protocols: Modbus, S7comm, DNP3, CIP/PCCC, and Omron FINS;
- strict contiguous field annotations with boundary, role, and scalar type labels;
- 6,672 original training examples and 742 held-out test examples;
- message lengths from 9 to 1,028 bytes.

The preparation command creates a protocol/length-stratified 90/10 split while
assigning identical byte messages as one group, so train and validation have no
exact-message overlap. It preserves the original 742-row test set for
comparability and additionally writes `test_novel.jsonl`, which removes every
message that occurs in the source training pool. The held-out source corpus is
treated the same way, producing `ood.jsonl` and the exact-byte-novel
`ood_novel.jsonl`. This split is not protocol-disjoint: it contains the five
training protocols plus 16 IEC104 messages, so paper text describes it as
source-held-out rather than unseen-protocol OOD.

```powershell
python -m mambapre.cli prepare `
  --source-train "<legacy-pre-root>\train.jsonl" `
  --source-test "<legacy-pre-root>\test.jsonl" `
  --source-ood "<legacy-pre-root>\ood.jsonl" `
  --output-dir data/processed/core_corpus
```

## Installation

Local CPU smoke tests need only PyTorch and NumPy:

```bash
python -m pip install -e .[dev]
pytest -q
```

GPU experiments use the official `mamba-ssm` CUDA kernels. Install a wheel or
build matching the server's PyTorch/CUDA versions, following the
[official Mamba repository](https://github.com/state-spaces/mamba). Do not use
the `reference` backend for paper results; it is a portable recurrent test
double, not Mamba.

```bash
python -m pip install -e .
python -m pip install causal-conv1d mamba-ssm --no-build-isolation
```

## Training and evaluation

Run from the repository root so relative data paths resolve consistently.

```bash
python -m mambapre.cli train --config configs/transformer.json
python -m mambapre.cli train --config configs/bimamba.json
python -m mambapre.cli train --config configs/dual_bimamba.json

python -m mambapre.cli evaluate \
  --checkpoint runs/dual_bimamba_seed1337/best.pt \
  --data data/processed/core_corpus/test.jsonl \
  --output results/dual_bimamba_test.json
```

Every evaluation reports micro boundary precision/recall/F1, strict exact-field
F1, whole-message perfection, byte-role/type accuracy, per-protocol results, and
length buckets. The length buckets are important: only 21 core-corpus test
messages exceed 256 bytes, so a pooled average alone cannot substantiate a
long-sequence claim.

The original source files contain exact byte duplicates across their source
train/test pools. Results on `test.jsonl` and `ood.jsonl` are retained only for
backward comparability; paper claims must use the `*_novel.jsonl` variants and
the group-disjoint validation split.

During training, the boundary threshold is selected only on the validation set
from the configured grid. The best checkpoint stores that threshold, and the
evaluation command uses it by default without retuning on test data.

## NeuPRE PDML/PCAP external evaluation

The NeuPRE adapter regenerates seven full-field protocol sets from real PCAPs
through Wireshark PDML and preserves capture/session metadata. It also creates
a separate S7comm+ long-message set whose labels cover only the TPKT/COTP
envelope and opaque payload. The two scopes are never pooled into one metric.

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

The audit found 21,121 exact-byte-novel full-field records, but none is longer
than 512 bytes. The S7comm+ set contains 530 exact-byte-novel messages from 513
to 1,024 bytes, but only at envelope granularity. See
[`docs/neupre_data_audit.md`](docs/neupre_data_audit.md) before making a
long-message accuracy claim.

The first frozen-threshold diagnostics are in
[`docs/neupre_first_results.md`](docs/neupre_first_results.md). The completed
five-seed controlled comparison is in
[`docs/multiseed_results.md`](docs/multiseed_results.md).

## Deep-boundary and offset diagnostics

The strict FINS and public OPC UA diagnostics contain fully dissected messages
with boundaries beyond the header region. Run the parameter-matched recurrent
control for one seed with:

```bash
bash scripts/run_bigru_seed_pipeline.sh 1337 0
```

`scripts/evaluate_offset_bins.py` reports micro precision, recall, and F1 in
the fixed offset bins 1--32, 33--128, 129--256, 257--512, and 513+. Aggregate
five seeds with `scripts/aggregate_offset_bins.py`; generate the publication
plot with `scripts/plot_offset_f1.py`. The compact plotted values are released
as [`figures/Fig3_offset_f1.csv`](figures/Fig3_offset_f1.csv). These tests are
at most 620 bytes, below the 1,028-byte training maximum, and therefore diagnose
deep-offset transfer rather than unseen absolute-position indices.

## Scaling benchmark

Use identical batch sizes and model widths for both architectures. CUDA peak
memory, median latency, p90 latency, message throughput, and byte throughput are
recorded. OOM is a valid result and is saved rather than silently reducing the
batch size.

```bash
python -m mambapre.cli benchmark --config configs/transformer_param_matched_dedup.json \
  --lengths 256,512,1024,2048,4096 --batch-size 8 --precision bfloat16 \
  --output results/scaling_transformer.json

python -m mambapre.cli benchmark --config configs/dual_bimamba_guided_dedup.json \
  --lengths 256,512,1024,2048,4096 --batch-size 8 --precision bfloat16 \
  --output results/scaling_dual_bimamba.json
```

Backend-sensitive efficiency is measured with both the standard PyTorch path
and an efficiency-only parameter-matched Transformer (`d=176`, head dimension
44). The latter forces memory-efficient SDPA in FP32 and Flash SDPA in BF16 via
`scripts/run_optimized_sdpa.sh`. Full-length batches omit the no-op padding mask,
and each JSON records the requested backend and this mask policy. The completed FP32/BF16 comparison is in
[`docs/scaling_results.md`](docs/scaling_results.md). The exact-byte-novel
traditional-tool comparison is generated from the supplied public sources with
`scripts/run_classical_baselines.sh`. The direct comparison includes
BinaryInferno, NEMESYS BCDG, Netzob static alignment, and NetPlier's alignment
segmentation. ICE-PRE and ProtocolGPT remain related systems but are not mixed
into the field-boundary table because they require different inputs or predict
a different target. Exact adapter choices and fairness limitations are recorded
in [`docs/baseline_audit.md`](docs/baseline_audit.md).

```bash
BI_PYTHON=/path/to/baseline-python \
CLASSICAL_PYTHON=/path/to/baseline-python \
bash scripts/run_classical_baselines.sh \
  ../Baseline data/processed/core_corpus/test_novel.jsonl results/baselines_public
```

The architecture screening table is in
[`docs/ablation_results_seed1337.md`](docs/ablation_results_seed1337.md).
The five-seed parameter-matched Bi-Mamba-only ablation is in
[`docs/bimamba_ablation_results.md`](docs/bimamba_ablation_results.md). It is
reproduced with `scripts/run_bimamba_ablation_seed.sh` and aggregated with
`scripts/aggregate_bimamba_ablation.py`.

The complete claim-to-evidence plan is in
[`docs/experiment_plan.md`](docs/experiment_plan.md).
The completed five-seed boundary/hybrid multiscale gate experiment and its
pre-registered stop decision are in
[`docs/p3_multiscale_results.md`](docs/p3_multiscale_results.md).
The validated A6000 environment and exact CUDA wheel choices are recorded in
[`docs/server_setup.md`](docs/server_setup.md).
The first seed's leakage audit, accuracy ablations, gate diagnostics, scaling
results, and go/no-go decision are in
[`docs/first_run_results.md`](docs/first_run_results.md).
