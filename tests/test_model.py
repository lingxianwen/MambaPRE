import math

import torch

from mambapre.losses import gate_supervision_target, multitask_loss
from mambapre.model import ModelConfig, build_model, reverse_valid


def test_reverse_valid_keeps_padding_at_end():
    x = torch.tensor([[[1.0], [2.0], [3.0], [0.0]], [[4.0], [5.0], [0.0], [0.0]]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    reversed_x = reverse_valid(x, mask)
    assert reversed_x.squeeze(-1).tolist() == [[3.0, 2.0, 1.0, 0.0], [5.0, 4.0, 0.0, 0.0]]


def test_all_architectures_shape_with_reference_backend():
    bytes_ = torch.tensor([[1, 2, 3, 4, 256], [5, 6, 7, 256, 256]])
    mask = bytes_ != 256
    for architecture in ("transformer", "cnn", "mamba", "bimamba", "dual_bimamba"):
        model = build_model(
            ModelConfig(
                architecture=architecture,
                backend="reference",
                d_model=16,
                num_layers=1,
                d_state=8,
                transformer_heads=4,
                transformer_ffn=32,
                max_length=16,
            )
        )
        output = model(bytes_, mask)
        assert output["boundary_logits"].shape == (2, 5)
        assert output["role_logits"].shape == (2, 5, 8)
        assert output["type_logits"].shape == (2, 5, 5)
        if architecture == "dual_bimamba":
            assert output["spatial_gates"].shape == (2, 1, 5, 16)


def test_fixed_fusion_gate_is_half():
    bytes_ = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(bytes_, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            architecture="dual_bimamba",
            backend="reference",
            fusion_mode="fixed",
            d_model=16,
            num_layers=1,
            d_state=8,
            transformer_heads=4,
            transformer_ffn=32,
            max_length=16,
        )
    )
    output = model(bytes_, mask)
    assert torch.all(output["spatial_gates"] == 0.5)


def test_learned_fusion_is_invariant_to_other_message_lengths_in_batch():
    torch.manual_seed(17)
    model = build_model(
        ModelConfig(
            architecture="dual_bimamba",
            backend="reference",
            d_model=16,
            num_layers=1,
            d_state=8,
            transformer_heads=4,
            transformer_ffn=32,
            max_length=16,
        )
    ).eval()
    message = torch.tensor([[1, 2, 3, 4]])
    message_mask = torch.ones_like(message, dtype=torch.bool)
    mixed = torch.tensor(
        [
            [1, 2, 3, 4, 256, 256, 256, 256],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ]
    )
    mixed_mask = mixed != 256

    with torch.inference_mode():
        single_output = model(message, message_mask)
        mixed_output = model(mixed, mixed_mask)

    assert torch.allclose(
        single_output["boundary_logits"][0],
        mixed_output["boundary_logits"][0, :4],
        atol=1e-6,
    )
    assert torch.allclose(
        single_output["spatial_gates"][0],
        mixed_output["spatial_gates"][0, :, :4],
        atol=1e-6,
    )
    for fusion in model.backbone.fusions:
        assert fusion.absolute_position_scale == math.log1p(16)


def test_guided_gate_loss_backpropagates():
    bytes_ = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(bytes_, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            architecture="dual_bimamba",
            backend="reference",
            d_model=16,
            num_layers=1,
            d_state=8,
            transformer_heads=4,
            transformer_ffn=32,
            max_length=16,
        )
    )
    batch = {
        "bytes": bytes_,
        "mask": mask,
        "boundary_labels": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        "role_labels": torch.tensor([[0, 1, 6, 6]]),
        "type_labels": torch.tensor([[0, 0, 4, 4]]),
    }
    outputs = model(bytes_, mask)
    loss, parts = multitask_loss(outputs, batch, gate_weight=0.05)
    loss.backward()
    assert parts["gate_loss"] > 0
    assert any("gate" in name and parameter.grad is not None for name, parameter in model.named_parameters())


def test_structural_only_gate_target_excludes_payload_unknown_and_padding():
    batch = {
        "mask": torch.tensor([[True, True, True, True, False]]),
        "boundary_labels": torch.zeros(1, 5),
        "role_labels": torch.tensor([[0, 6, -100, 1, 0]]),
    }
    target, target_mask = gate_supervision_target(batch, mode="structural_only")
    assert target.tolist() == [[1.0, 1.0, 1.0, 1.0, 1.0]]
    assert target_mask.tolist() == [[True, False, False, True, False]]


def test_multiscale_boundary_gate_target():
    batch = {
        "mask": torch.ones(1, 9, dtype=torch.bool),
        "boundary_labels": torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        "role_labels": torch.full((1, 9), 6, dtype=torch.long),
    }
    target, target_mask = gate_supervision_target(
        batch,
        mode="boundary_multiscale",
        boundary_radii=(0, 1, 3),
        boundary_values=(1.0, 0.6, 0.2),
    )
    assert target_mask.all()
    expected = torch.tensor([[0.0, 0.2, 0.2, 0.6, 1.0, 0.6, 0.2, 0.2, 0.0]])
    assert torch.allclose(target, expected)


def test_hybrid_gate_keeps_structure_and_adds_payload_boundaries():
    batch = {
        "mask": torch.ones(1, 6, dtype=torch.bool),
        "boundary_labels": torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
        "role_labels": torch.tensor([[0, 0, 6, 6, 6, 6]]),
    }
    target, target_mask = gate_supervision_target(
        batch,
        mode="role_boundary_multiscale",
        boundary_radii=(0, 1),
        boundary_values=(1.0, 0.5),
    )
    assert target_mask.all()
    expected = torch.tensor([[1.0, 1.0, 0.5, 1.0, 0.5, 0.0]])
    assert torch.allclose(target, expected)
