from __future__ import annotations

import torch
from torch.nn import functional as F

from .constants import ROLE_TO_ID


def _multiscale_boundary_target(
    boundary_labels: torch.Tensor,
    radii: list[int] | tuple[int, ...],
    values: list[float] | tuple[float, ...],
) -> torch.Tensor:
    """Create a soft local-stream target around each internal field cut."""
    if len(radii) != len(values) or not radii:
        raise ValueError("gate_boundary_radii and gate_boundary_values must have equal nonzero length")
    if any(int(radius) < 0 for radius in radii):
        raise ValueError("gate boundary radii must be non-negative")
    if any(not 0.0 <= float(value) <= 1.0 for value in values):
        raise ValueError("gate boundary values must lie in [0, 1]")

    cuts = boundary_labels.float().unsqueeze(1)
    target = torch.zeros_like(boundary_labels, dtype=torch.float32)
    for radius, value in zip(radii, values):
        radius = int(radius)
        nearby = F.max_pool1d(cuts, kernel_size=2 * radius + 1, stride=1, padding=radius)
        target = torch.maximum(target, nearby.squeeze(1) * float(value))
    return target


def gate_supervision_target(
    batch: dict,
    mode: str = "role",
    boundary_radii: list[int] | tuple[int, ...] = (0, 1, 3),
    boundary_values: list[float] | tuple[float, ...] = (1.0, 0.67, 0.33),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a soft gate target and the bytes on which it is supervised.

    Gate=1 selects the local stream. ``role`` preserves the original
    structural-versus-payload target. ``structural_only`` applies positive
    supervision to known structural bytes and excludes payload bytes from
    :math:`L_g`. ``boundary_multiscale`` selects local
    processing only near gold cuts. ``role_boundary_multiscale`` retains the
    original structural target while allowing payload-internal cut
    neighborhoods to use local evidence.
    """
    mask = batch["mask"]
    known_roles = batch["role_labels"] != -100
    role_target = (batch["role_labels"] != ROLE_TO_ID["payload"]).float()
    if mode == "role":
        return role_target, mask & known_roles
    if mode == "structural_only":
        structural = known_roles & (
            batch["role_labels"] != ROLE_TO_ID["payload"]
        )
        return torch.ones_like(role_target), mask & structural

    boundary_target = _multiscale_boundary_target(
        batch["boundary_labels"], boundary_radii, boundary_values
    ).to(mask.device)
    if mode == "boundary_multiscale":
        return boundary_target, mask
    if mode == "role_boundary_multiscale":
        structural_target = role_target * known_roles.to(role_target.dtype)
        return torch.maximum(boundary_target, structural_target), mask
    raise ValueError(f"unknown gate_target_mode: {mode}")


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    boundary_positive_weight: float = 8.0,
    role_weight: float = 0.2,
    type_weight: float = 0.1,
    gate_weight: float = 0.0,
    gate_target_mode: str = "role",
    gate_boundary_radii: list[int] | tuple[int, ...] = (0, 1, 3),
    gate_boundary_values: list[float] | tuple[float, ...] = (1.0, 0.67, 0.33),
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"]
    logits = outputs["boundary_logits"][mask]
    targets = batch["boundary_labels"][mask]
    boundary_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=torch.tensor(boundary_positive_weight, device=logits.device),
    )
    total = boundary_loss
    parts = {"boundary_loss": float(boundary_loss.detach())}

    if "role_logits" in outputs:
        flat_roles = batch["role_labels"].reshape(-1)
        flat_types = batch["type_labels"].reshape(-1)
        if (flat_roles != -100).any():
            role_loss = F.cross_entropy(
                outputs["role_logits"].reshape(-1, outputs["role_logits"].shape[-1]),
                flat_roles,
                ignore_index=-100,
            )
            total = total + role_weight * role_loss
            parts["role_loss"] = float(role_loss.detach())
        if (flat_types != -100).any():
            type_loss = F.cross_entropy(
                outputs["type_logits"].reshape(-1, outputs["type_logits"].shape[-1]),
                flat_types,
                ignore_index=-100,
            )
            total = total + type_weight * type_loss
            parts["type_loss"] = float(type_loss.detach())

    if gate_weight > 0.0:
        if "spatial_gates" not in outputs:
            raise ValueError("gate_weight > 0 requires a dual-stream model with spatial_gates")
        # Gate=1 selects the rigid local stream; Gate=0 selects the global SSM stream.
        # Gold annotations supervise the gate during training only and are never model inputs.
        gate_scores = outputs["spatial_gates"].mean(dim=(1, 3))
        gate_target, gate_mask = gate_supervision_target(
            batch,
            mode=gate_target_mode,
            boundary_radii=gate_boundary_radii,
            boundary_values=gate_boundary_values,
        )
        # Explicit FP32 probability-space BCE avoids PyTorch's unsafe-autocast
        # guard for ``binary_cross_entropy`` while preserving the same objective.
        if gate_mask.any():
            gate_prob = gate_scores[gate_mask].float().clamp(1e-6, 1.0 - 1e-6)
            gate_target = gate_target[gate_mask].float()
            gate_loss = -(
                gate_target * gate_prob.log()
                + (1.0 - gate_target) * torch.log1p(-gate_prob)
            ).mean()
            total = total + gate_weight * gate_loss
            parts["gate_loss"] = float(gate_loss.detach())
    parts["loss"] = float(total.detach())
    return total, parts
