from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .constants import ROLE_TO_ID
from .data import FixedRatioBatchSampler, ProtocolDataset, collate_samples, load_jsonl
from .losses import gate_supervision_target, multitask_loss
from .metrics import StratifiedMetrics, evaluate_batch
from .model import ModelConfig, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = dict(batch)
    for key in ("bytes", "mask", "boundary_labels", "role_labels", "type_labels"):
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def make_loader(
    path: str,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    dataset = ProtocolDataset(
        load_jsonl(path), max_length=max_length, random_crop=shuffle, seed=seed
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_samples,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


def make_train_loader(
    data_config: dict,
    batch_size: int,
    max_length: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, dict]:
    """Create either the standard loader or a fixed-ratio long-message loader."""
    long_path = data_config.get("long_train")
    long_fraction = float(data_config.get("long_fraction", 0.0))
    if not long_path:
        loader = make_loader(
            data_config["train"], batch_size, max_length, True, workers, seed
        )
        return loader, {
            "strategy": "standard_shuffle",
            "base_path": data_config["train"],
            "long_path": None,
            "requested_long_fraction": 0.0,
        }

    base_dataset = ProtocolDataset(
        load_jsonl(data_config["train"]),
        max_length=max_length,
        random_crop=True,
        seed=seed,
    )
    long_dataset = ProtocolDataset(
        load_jsonl(long_path),
        max_length=max_length,
        random_crop=True,
        seed=seed + 1,
    )
    sampler = FixedRatioBatchSampler(
        base_size=len(base_dataset),
        long_size=len(long_dataset),
        batch_size=batch_size,
        long_fraction=long_fraction,
        seed=seed,
    )
    loader = DataLoader(
        ConcatDataset([base_dataset, long_dataset]),
        batch_sampler=sampler,
        num_workers=workers,
        collate_fn=collate_samples,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return loader, {
        "strategy": "fixed_ratio_per_batch",
        "base_path": data_config["train"],
        "long_path": long_path,
        "long_source_kind": data_config.get(
            "long_source_kind", "protocol-valid controlled traffic"
        ),
        "base_messages": len(base_dataset),
        "long_messages": len(long_dataset),
        "batch_size": batch_size,
        "base_per_batch": sampler.base_per_batch,
        "long_per_batch": sampler.long_per_batch,
        "requested_long_fraction": long_fraction,
        "effective_long_fraction": sampler.effective_long_fraction,
        "batches_per_epoch": len(sampler),
    }
@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float | list[float],
    loss_config: dict,
) -> dict:
    model.eval()
    thresholds = [float(threshold)] if isinstance(threshold, (int, float)) else threshold
    if not thresholds:
        raise ValueError("at least one evaluation threshold is required")
    metrics = {value: StratifiedMetrics() for value in thresholds}
    loss_sums: dict[str, float] = defaultdict(float)
    gate_structural_sum = None
    gate_payload_sum = None
    gate_structural_count = 0
    gate_payload_count = 0
    gate_target_positive_sum = None
    gate_target_negative_sum = None
    gate_target_positive_count = 0
    gate_target_negative_count = 0
    batches = 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        outputs = model(batch["bytes"], batch["mask"])
        _, parts = multitask_loss(outputs, batch, **loss_config)
        for key, value in parts.items():
            loss_sums[key] += value
        for value, accumulator in metrics.items():
            evaluate_batch(accumulator, outputs, cpu_batch, threshold=value)
        if "spatial_gates" in outputs:
            gate = outputs["spatial_gates"].detach().float().mean(dim=-1)
            known_roles = batch["role_labels"] != -100
            structural = batch["mask"] & known_roles & (
                batch["role_labels"] != ROLE_TO_ID["payload"]
            )
            payload = batch["mask"] & (
                batch["role_labels"] == ROLE_TO_ID["payload"]
            )
            structural_sum = (gate * structural[:, None, :]).sum(dim=(0, 2)).cpu()
            payload_sum = (gate * payload[:, None, :]).sum(dim=(0, 2)).cpu()
            gate_structural_sum = (
                structural_sum
                if gate_structural_sum is None
                else gate_structural_sum + structural_sum
            )
            gate_payload_sum = (
                payload_sum if gate_payload_sum is None else gate_payload_sum + payload_sum
            )
            gate_structural_count += int(structural.sum())
            gate_payload_count += int(payload.sum())
            target, target_mask = gate_supervision_target(
                batch,
                mode=loss_config.get("gate_target_mode", "role"),
                boundary_radii=loss_config.get("gate_boundary_radii", (0, 1, 3)),
                boundary_values=loss_config.get("gate_boundary_values", (1.0, 0.67, 0.33)),
            )
            target_positive = target_mask & (target > 0)
            target_negative = target_mask & (target == 0)
            target_positive_sum = (gate * target_positive[:, None, :]).sum(dim=(0, 2)).cpu()
            target_negative_sum = (gate * target_negative[:, None, :]).sum(dim=(0, 2)).cpu()
            gate_target_positive_sum = (
                target_positive_sum
                if gate_target_positive_sum is None
                else gate_target_positive_sum + target_positive_sum
            )
            gate_target_negative_sum = (
                target_negative_sum
                if gate_target_negative_sum is None
                else gate_target_negative_sum + target_negative_sum
            )
            gate_target_positive_count += int(target_positive.sum())
            gate_target_negative_count += int(target_negative.sum())
        batches += 1
    reports = {value: accumulator.summary() for value, accumulator in metrics.items()}
    best_threshold = max(
        reports, key=lambda value: reports[value]["overall"]["boundary_f1"]
    )
    report = reports[best_threshold]
    report["losses"] = {key: value / max(batches, 1) for key, value in loss_sums.items()}
    report["threshold"] = best_threshold
    if len(thresholds) > 1:
        report["threshold_search"] = {
            str(value): reports[value]["overall"]["boundary_f1"] for value in thresholds
        }
    if gate_structural_sum is not None:
        structural_means = gate_structural_sum / max(gate_structural_count, 1)
        payload_means = gate_payload_sum / max(gate_payload_count, 1)
        report["gate_analysis"] = {
            "meaning": "1 selects local rigid stream; 0 selects global SSM stream",
            "supervision": loss_config.get("gate_target_mode", "role"),
            "structural_bytes": gate_structural_count,
            "payload_bytes": gate_payload_count,
            "per_layer": [
                {
                    "layer": index,
                    "structural_mean": float(structural),
                    "payload_mean": float(payload),
                    "separation": float(structural - payload),
                }
                for index, (structural, payload) in enumerate(
                    zip(structural_means, payload_means)
                )
            ],
        }
        if gate_target_positive_sum is not None:
            positive_means = gate_target_positive_sum / max(gate_target_positive_count, 1)
            negative_means = gate_target_negative_sum / max(gate_target_negative_count, 1)
            report["gate_analysis"]["target_partition"] = {
                "positive_bytes": gate_target_positive_count,
                "negative_bytes": gate_target_negative_count,
                "per_layer": [
                    {
                        "layer": index,
                        "positive_mean": float(positive),
                        "negative_mean": float(negative),
                        "separation": float(positive - negative),
                    }
                    for index, (positive, negative) in enumerate(
                        zip(positive_means, negative_means)
                    )
                ],
            }
    return report


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    training_config: dict,
    report: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "model_config": model.config.to_dict(),
            "training_config": training_config,
            "validation": report,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model, checkpoint


def train_from_config(config: dict) -> dict:
    seed = int(config.get("seed", 1337))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    model_config = ModelConfig(**config["model"])
    model = build_model(model_config).to(device)
    data_config = config["data"]
    training = config["training"]
    loss_config = config.get("loss", {})
    train_loader, sampling_report = make_train_loader(
        data_config,
        training["batch_size"],
        model_config.max_length,
        training.get("workers", 0),
        seed,
    )
    val_loader = make_loader(
        data_config["validation"],
        training.get("eval_batch_size", training["batch_size"]),
        model_config.max_length,
        False,
        training.get("workers", 0),
        seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.get("learning_rate", 3e-4),
        weight_decay=training.get("weight_decay", 0.01),
    )
    epochs = int(training.get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_amp = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip = float(training.get("gradient_clip", 1.0))
    threshold = float(training.get("threshold", 0.5))
    validation_thresholds = [
        float(value) for value in training.get("threshold_grid", [threshold])
    ]
    patience = int(training.get("patience", 8))
    output_dir = Path(config.get("output_dir", "runs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "history.jsonl"
    best_f1 = -1.0
    bad_epochs = 0
    best_report = {}

    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_start = time.perf_counter()
            train_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            accumulation = int(training.get("gradient_accumulation", 1))
            for step, cpu_batch in enumerate(train_loader, 1):
                batch = move_batch(cpu_batch, device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(batch["bytes"], batch["mask"])
                    loss, _ = multitask_loss(outputs, batch, **loss_config)
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                train_loss += float(loss.detach())

            scheduler.step()
            validation = evaluate_model(
                model, val_loader, device, validation_thresholds, loss_config
            )
            f1 = validation["overall"]["boundary_f1"]
            record = {
                "epoch": epoch,
                "train_loss": train_loss / max(len(train_loader), 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - epoch_start,
                "validation": validation,
            }
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            print(
                f"epoch={epoch:03d} train_loss={record['train_loss']:.5f} "
                f"val_f1={f1:.4f} perfect={validation['overall']['message_perfection']:.4f}"
            )
            save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, config, validation)
            if f1 > best_f1:
                best_f1 = f1
                best_report = validation
                bad_epochs = 0
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, config, validation)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"early stopping after {bad_epochs} epochs without validation improvement")
                    break

    summary = {
        "best_boundary_f1": best_f1,
        "best_validation": best_report,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "device": str(device),
        "training_sampling": sampling_report,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
