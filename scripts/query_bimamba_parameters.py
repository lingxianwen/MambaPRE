from __future__ import annotations

import argparse

from mambapre.model import ModelConfig, build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Count Bi-Mamba-only parameter candidates")
    parser.add_argument("--target", type=int, default=2_191_502)
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 132, 136, 140, 144])
    args = parser.parse_args()

    for width in args.widths:
        config = ModelConfig(
            architecture="bimamba",
            backend="mamba1",
            d_model=width,
            num_layers=4,
            d_state=64,
            d_conv=4,
            expand=2,
            max_length=4096,
            predict_semantics=True,
        )
        parameters = sum(parameter.numel() for parameter in build_model(config).parameters())
        print(f"d_model={width}: parameters={parameters}, delta={parameters - args.target:+d}")


if __name__ == "__main__":
    main()
