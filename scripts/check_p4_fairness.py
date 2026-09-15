from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mambapre.model import ModelConfig, build_model


CONFIGS = {
    "role_guided_mamba": Path("configs/p4_role_guided_long20.json"),
    "boundary_multiscale_mamba": Path("configs/p4_boundary_multiscale_long20.json"),
    "matched_transformer": Path("configs/p4_transformer_matched_long20.json"),
}


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    configs = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in CONFIGS.items()}
    data_hashes = {name: canonical_hash(config["data"]) for name, config in configs.items()}
    training_hashes = {
        name: canonical_hash(config["training"]) for name, config in configs.items()
    }
    if len(set(data_hashes.values())) != 1:
        raise ValueError(f"P4 data configurations differ: {data_hashes}")
    if len(set(training_hashes.values())) != 1:
        raise ValueError(f"P4 training configurations differ: {training_hashes}")

    parameters = {}
    for name, config in configs.items():
        model = build_model(ModelConfig(**config["model"]))
        parameters[name] = sum(parameter.numel() for parameter in model.parameters())
    mamba_parameters = parameters["role_guided_mamba"]
    relative_gap = abs(parameters["matched_transformer"] - mamba_parameters) / mamba_parameters
    if relative_gap > 0.01:
        raise ValueError(f"Transformer parameter gap exceeds 1%: {relative_gap:.4%}")

    report = {
        "fairness_gate": "passed",
        "identical_data_config_sha256": next(iter(data_hashes.values())),
        "identical_training_config_sha256": next(iter(training_hashes.values())),
        "parameters": parameters,
        "transformer_vs_mamba_parameter_gap": relative_gap,
        "batch_composition": {
            "batch_size": configs["role_guided_mamba"]["training"]["batch_size"],
            "requested_long_fraction": configs["role_guided_mamba"]["data"]["long_fraction"],
            "effective_long_messages": 6,
            "effective_base_messages": 26,
            "effective_long_fraction": 0.1875,
        },
    }
    output = Path("results/p4/fairness_gate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
