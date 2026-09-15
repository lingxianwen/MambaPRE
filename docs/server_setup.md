# 已验证的 GPU 服务器环境

本仓库不保存 SSH 主机、密码、认证令牌或私钥。

## 硬件与基础运行时

- GPU：4 × NVIDIA RTX A6000，48 GB，计算能力 8.6
- 驱动：520.61.05
- CUDA 工具包：11.8
- 基础 Python：现有 `qwen3_serve` 环境中的 3.10.20
- PyTorch：2.3.0+cu118，禁用 CXX11 ABI
- 隔离 venv：`<workspace>/venvs/mambapre`
- 服务器项目镜像：`<workspace>/projects/Mamba-PRE`

venv 使用 `--system-site-packages` 创建，以复用与 CUDA 匹配的 PyTorch，同时不修改源 Conda 环境。

## 已验证的 CUDA wheel

- `causal_conv1d-1.4.0+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
- `mamba_ssm-2.2.2+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`

两个 wheel 均为 GitHub 官方发布文件。服务器当前存在间歇性 DNS 故障，因此可复现安装路径是：在另一台主机下载这些精确版本，核对文件名和校验和，上传到用户 wheel 目录，再使用 `--no-deps` 安装。

## 后端决策

官方原始 Mamba CUDA selective-scan 路径已在该服务器通过前向和反向测试。Mamba-2 的融合 Triton 路径无法运行：在已安装的 NVIDIA 520 驱动下，Triton 2.3 会产生 `device kernel image is invalid`。代码仍保留 `backend=mamba2`，但该服务器上的论文实验必须使用 `backend=mamba1`；除非升级驱动并重新通过 CUDA 冒烟测试。

## 启动方式

前台运行：

```bash
scripts/run_server_experiment.sh configs/dual_bimamba.json 0
```

后台运行并记录无缓冲日志：

```bash
nohup scripts/run_server_experiment.sh configs/dual_bimamba.json 0 \
  > runs/dual_bimamba_seed1337.log 2>&1 < /dev/null &
```

每次运行前确认所选 GPU 空闲。在扩展性对比中不要静默修改 batch size；显存不足本身也是实验结果的一部分。
