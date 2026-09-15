from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .constants import PAD_BYTE, ROLES, TYPES


@dataclass
class ModelConfig:
    architecture: str = "dual_bimamba"
    backend: str = "mamba1"
    d_model: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 32
    transformer_heads: int = 4
    transformer_ffn: int = 512
    max_length: int = 2048
    local_kernel_size: int = 7
    fusion_mode: str = "learned"
    predict_semantics: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def reverse_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reverse only the valid prefix in each padded sequence."""
    batch, seq_len, width = x.shape
    lengths = mask.sum(dim=1)
    positions = torch.arange(seq_len, device=x.device).expand(batch, seq_len)
    gather_positions = torch.where(
        positions < lengths[:, None], lengths[:, None] - 1 - positions, positions
    )
    return x.gather(1, gather_positions[:, :, None].expand(batch, seq_len, width))


class BytePositionEmbedding(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.byte = nn.Embedding(PAD_BYTE + 1, config.d_model, padding_idx=PAD_BYTE)
        self.position = nn.Embedding(config.max_length, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, byte_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len = byte_values.shape[1]
        if seq_len > self.position.num_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds configured max_length "
                f"{self.position.num_embeddings}"
            )
        positions = torch.arange(seq_len, device=byte_values.device)
        x = self.byte(byte_values) + self.position(positions)[None, :, :]
        return self.dropout(self.norm(x)) * mask[:, :, None]


class ReferenceSSM(nn.Module):
    """Portable linear-time recurrent mixer for tests, not the paper Mamba backend.

    GPU experiments must use ``backend=mamba2``. This fallback exists so data,
    masking, losses, and metrics can be tested without custom CUDA extensions.
    """

    def __init__(self, d_model: int, d_conv: int, expand: int, dropout: float) -> None:
        super().__init__()
        hidden = d_model * expand
        self.in_proj = nn.Linear(d_model, hidden * 2)
        self.depthwise = nn.Conv1d(
            hidden, hidden, kernel_size=d_conv, padding=d_conv - 1, groups=hidden
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out_proj = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, gates = self.in_proj(x).chunk(2, dim=-1)
        values = self.depthwise(values.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        values, _ = self.gru(torch.nn.functional.silu(values))
        return self.dropout(self.out_proj(values * torch.sigmoid(gates)))


def make_mixer(config: ModelConfig) -> nn.Module:
    if config.backend == "reference":
        return ReferenceSSM(
            config.d_model, config.d_conv, config.expand, config.dropout
        )
    if config.backend not in {"mamba1", "mamba2"}:
        raise ValueError(f"unknown SSM backend: {config.backend}")
    try:
        from mamba_ssm import Mamba, Mamba2
    except Exception as exc:
        raise RuntimeError(
            f"backend={config.backend} requires a CUDA-matched mamba-ssm installation. "
            "Use backend=reference only for CPU smoke tests."
        ) from exc
    common = dict(
        d_model=config.d_model,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
    )
    if config.backend == "mamba1":
        return Mamba(**common)
    return Mamba2(**common, headdim=config.headdim)


class BiMambaLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_fwd = nn.LayerNorm(config.d_model)
        self.norm_bwd = nn.LayerNorm(config.d_model)
        self.forward_mixer = make_mixer(config)
        self.backward_mixer = make_mixer(config)
        self.fuse = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        fwd = self.forward_mixer(self.norm_fwd(x))
        reversed_x = reverse_valid(self.norm_bwd(x), mask)
        bwd = reverse_valid(self.backward_mixer(reversed_x), mask)
        update = self.fuse(torch.cat((fwd, bwd), dim=-1))
        return self.output_norm(x + update) * mask[:, :, None]


class UniMambaLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mixer = make_mixer(config)
        self.dropout = nn.Dropout(config.dropout)
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.output_norm(x + self.dropout(self.mixer(self.norm(x)))) * mask[:, :, None]


class LocalRigidStream(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        kernel = config.local_kernel_size
        self.norm = nn.LayerNorm(config.d_model)
        self.depthwise = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=kernel,
            padding=kernel // 2,
            groups=config.d_model,
        )
        self.pointwise = nn.Conv1d(config.d_model, config.d_model, kernel_size=1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x).transpose(1, 2)
        x = self.pointwise(torch.nn.functional.gelu(self.depthwise(x))).transpose(1, 2)
        return (residual + self.dropout(x)) * mask[:, :, None]


class SpatialFusion(nn.Module):
    """Position-conditioned fusion of rigid local and flexible global streams."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.fusion_mode not in {"learned", "fixed"}:
            raise ValueError(f"unknown fusion mode: {config.fusion_mode}")
        self.fusion_mode = config.fusion_mode
        # Use the model's fixed position range rather than the padded length of
        # the current batch.  This keeps predictions invariant to which other
        # message lengths happen to share a batch.
        self.absolute_position_scale = math.log1p(max(config.max_length, 2))
        self.gate = (
            nn.Sequential(
                nn.Linear(2 * config.d_model + 2, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
                nn.Sigmoid(),
            )
            if config.fusion_mode == "learned"
            else None
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self, global_features: torch.Tensor, local_features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = global_features.shape
        positions = torch.arange(seq_len, device=global_features.device, dtype=global_features.dtype)
        lengths = mask.sum(dim=1).clamp_min(1).to(global_features.dtype)
        relative = positions[None, :] / (lengths[:, None] - 1).clamp_min(1)
        absolute = torch.log1p(positions)[None, :] / self.absolute_position_scale
        spatial = torch.stack((relative.expand(batch, -1), absolute.expand(batch, -1)), dim=-1)
        gate = (
            self.gate(torch.cat((global_features, local_features, spatial), dim=-1))
            if self.gate is not None
            else torch.full_like(global_features, 0.5)
        )
        fused = gate * local_features + (1.0 - gate) * global_features
        return self.norm(fused) * mask[:, :, None], gate * mask[:, :, None]


class TransformerBackbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ffn,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.num_layers, nn.LayerNorm(config.d_model))
        self.omit_padding_mask = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Scaling uses full-length batches and explicitly enables this no-op-mask
        # elision so PyTorch's fused SDPA kernels remain eligible.
        padding_mask = None if self.omit_padding_mask else ~mask
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x * mask[:, :, None], None


class ResidualCNNLayer(nn.Module):
    """Finite-context supervised control used in the matched neural baseline."""

    def __init__(self, config: ModelConfig, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.norm = nn.LayerNorm(config.d_model)
        self.conv1 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        update = self.norm(x).transpose(1, 2)
        update = torch.nn.functional.gelu(self.conv1(update))
        update = self.conv2(self.dropout(update)).transpose(1, 2)
        return self.output_norm(x + self.dropout(update)) * mask[:, :, None]


class CNNBackbone(nn.Module):
    """Parameter-matched, dilated CNN for a common-pipeline supervised control."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ResidualCNNLayer(config, dilation=2**index)
            for index in range(config.num_layers)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, None]:
        for layer in self.layers:
            x = layer(x, mask)
        return x, None


class MambaBackbone(nn.Module):
    def __init__(self, config: ModelConfig, bidirectional: bool) -> None:
        super().__init__()
        layer_type = BiMambaLayer if bidirectional else UniMambaLayer
        self.layers = nn.ModuleList(layer_type(config) for _ in range(config.num_layers))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, None]:
        for layer in self.layers:
            x = layer(x, mask)
        return x, None


class DualStreamBiMambaBackbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.global_layers = nn.ModuleList(BiMambaLayer(config) for _ in range(config.num_layers))
        self.local_layers = nn.ModuleList(LocalRigidStream(config) for _ in range(config.num_layers))
        self.fusions = nn.ModuleList(SpatialFusion(config) for _ in range(config.num_layers))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_x = local_x = x
        gates = []
        for global_layer, local_layer, fusion in zip(
            self.global_layers, self.local_layers, self.fusions
        ):
            global_x = global_layer(global_x, mask)
            local_x = local_layer(local_x, mask)
            global_x, gate = fusion(global_x, local_x, mask)
            local_x = global_x
            gates.append(gate)
        return global_x, torch.stack(gates, dim=1)


class ProtocolBoundaryModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = BytePositionEmbedding(config)
        if config.architecture == "transformer":
            self.backbone = TransformerBackbone(config)
        elif config.architecture == "cnn":
            self.backbone = CNNBackbone(config)
        elif config.architecture == "mamba":
            self.backbone = MambaBackbone(config, bidirectional=False)
        elif config.architecture == "bimamba":
            self.backbone = MambaBackbone(config, bidirectional=True)
        elif config.architecture == "dual_bimamba":
            self.backbone = DualStreamBiMambaBackbone(config)
        else:
            raise ValueError(f"unknown architecture: {config.architecture}")

        self.boundary_head = nn.Linear(config.d_model, 1)
        self.role_head = nn.Linear(config.d_model, len(ROLES)) if config.predict_semantics else None
        self.type_head = nn.Linear(config.d_model, len(TYPES)) if config.predict_semantics else None

    def forward(self, byte_values: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.embedding(byte_values, mask)
        x, gates = self.backbone(x, mask)
        output = {"boundary_logits": self.boundary_head(x).squeeze(-1)}
        if self.role_head is not None:
            output["role_logits"] = self.role_head(x)
            output["type_logits"] = self.type_head(x)
        if gates is not None:
            output["spatial_gates"] = gates
        return output


def build_model(config: ModelConfig | dict) -> ProtocolBoundaryModel:
    if isinstance(config, dict):
        config = ModelConfig(**config)
    return ProtocolBoundaryModel(config)
