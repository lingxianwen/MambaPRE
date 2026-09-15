from __future__ import annotations

from contextlib import nullcontext
import time
import statistics

import torch

from .constants import PAD_BYTE
from .engine import resolve_device, set_seed
from .model import ModelConfig, build_model


def benchmark_architecture(
    model_config: dict,
    lengths: list[int],
    batch_size: int,
    device_name: str,
    warmup: int = 10,
    repetitions: int = 30,
    seed: int = 1337,
    precision: str = "float32",
    attention_backend: str = "auto",
) -> dict:
    set_seed(seed)
    device = resolve_device(device_name)
    config = ModelConfig(**model_config)
    model = build_model(config).to(device).eval()
    if config.architecture == "transformer":
        model.backbone.omit_padding_mask = True
    if precision not in {"float32", "bfloat16"}:
        raise ValueError(f"unsupported precision: {precision}")
    if precision == "bfloat16" and device.type != "cuda":
        raise ValueError("bfloat16 benchmark currently requires CUDA")
    if attention_backend not in {"auto", "math", "memory_efficient", "flash"}:
        raise ValueError(f"unsupported attention backend: {attention_backend}")
    if attention_backend != "auto" and config.architecture != "transformer":
        raise ValueError("a forced attention backend is only valid for Transformer")
    if attention_backend != "auto" and device.type != "cuda":
        raise ValueError("a forced attention backend requires CUDA")
    use_autocast = precision == "bfloat16"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    previous_fastpath = None
    if attention_backend != "auto":
        previous_fastpath = torch.backends.mha.get_fastpath_enabled()
        # TransformerEncoderLayer otherwise may bypass SDPA through its fused fast path.
        torch.backends.mha.set_fastpath_enabled(False)

    def attention_context():
        if attention_backend == "auto":
            return nullcontext()
        return torch.backends.cuda.sdp_kernel(
            enable_flash=attention_backend == "flash",
            enable_math=attention_backend == "math",
            enable_mem_efficient=attention_backend == "memory_efficient",
        )

    rows = []
    try:
        for length in lengths:
            if length > config.max_length:
                rows.append({"length": length, "status": "exceeds_model_max_length"})
                continue
            try:
                byte_values = torch.randint(
                    0, PAD_BYTE, (batch_size, length), device=device, dtype=torch.long
                )
                mask = torch.ones((batch_size, length), device=device, dtype=torch.bool)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                with torch.inference_mode(), attention_context():
                    for _ in range(warmup):
                        with torch.autocast(
                            device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast
                        ):
                            model(byte_values, mask)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    samples = []
                    for _ in range(repetitions):
                        start = time.perf_counter()
                        with torch.autocast(
                            device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast
                        ):
                            model(byte_values, mask)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        samples.append(time.perf_counter() - start)
                samples.sort()
                median = samples[len(samples) // 2]
                row = {
                    "length": length,
                    "status": "ok",
                    "median_latency_ms": median * 1000,
                    "mean_latency_ms": statistics.fmean(samples) * 1000,
                    "std_latency_ms": statistics.stdev(samples) * 1000 if len(samples) > 1 else 0.0,
                    "p90_latency_ms": samples[int(0.9 * (len(samples) - 1))] * 1000,
                    "messages_per_second": batch_size / median,
                    "bytes_per_second": batch_size * length / median,
                }
                if device.type == "cuda":
                    row["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
                rows.append(row)
            except RuntimeError as exc:
                if isinstance(exc, RuntimeError) and "No available kernel" not in str(exc):
                    if not isinstance(exc, torch.cuda.OutOfMemoryError):
                        raise
                rows.append({
                    "length": length,
                    "status": "out_of_memory" if isinstance(exc, torch.cuda.OutOfMemoryError) else "backend_unavailable",
                    "error": str(exc),
                })
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if previous_fastpath is not None:
            torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    return {
        "architecture": config.architecture,
        "backend": config.backend,
        "parameters": parameter_count,
        "batch_size": batch_size,
        "device": str(device),
        "torch_version": torch.__version__,
        "precision": precision,
        "attention_backend_requested": attention_backend,
        "transformer_mha_fastpath_disabled": attention_backend != "auto",
        "full_length_padding_mask_elided": config.architecture == "transformer",
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_compute_capability": (
            list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None
        ),
        "warmup": warmup,
        "repetitions": repetitions,
        "results": rows,
    }
