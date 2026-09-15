"""One-batch CUDA smoke test for the official Mamba backend."""

import torch

from mambapre.losses import multitask_loss
from mambapre.model import ModelConfig, build_model


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    config = ModelConfig(
        architecture="dual_bimamba",
        backend="mamba1",
        d_model=128,
        num_layers=2,
        max_length=256,
    )
    model = build_model(config).to(device).train()
    batch_size, length = 2, 64
    batch = {
        "bytes": torch.randint(0, 256, (batch_size, length), device=device),
        "mask": torch.ones((batch_size, length), dtype=torch.bool, device=device),
        "boundary_labels": torch.zeros((batch_size, length), device=device),
        "role_labels": torch.randint(0, 8, (batch_size, length), device=device),
        "type_labels": torch.randint(0, 5, (batch_size, length), device=device),
    }
    batch["boundary_labels"][:, 8::8] = 1.0
    outputs = model(batch["bytes"], batch["mask"])
    loss, parts = multitask_loss(outputs, batch)
    loss.backward()
    gradient_parameters = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    print(
        {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "output_shape": list(outputs["boundary_logits"].shape),
            "losses": parts,
            "finite_gradient_parameters": gradient_parameters,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    )


if __name__ == "__main__":
    main()
