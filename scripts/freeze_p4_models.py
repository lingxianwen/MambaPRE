from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze P4 checkpoints before sealed evaluation")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--sealed-manifest", required=True)
    parser.add_argument("--primary-family", required=True)
    parser.add_argument("--output", default="results/p4/freeze_manifest.json")
    args = parser.parse_args()

    sealed_manifest = Path(args.sealed_manifest)
    if not sealed_manifest.is_file():
        raise FileNotFoundError("sealed dataset manifest is required before model freeze")
    sealed = json.loads(sealed_manifest.read_text(encoding="utf-8"))
    if sealed.get("status") != "built_unopened":
        raise ValueError("sealed manifest must be in built_unopened state")
    checkpoints = []
    for value in args.checkpoint:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        training_config = checkpoint["training_config"]
        if "p4_sealed" in json.dumps(training_config):
            raise ValueError(f"sealed path leaked into training config: {path}")
        checkpoints.append(
            {
                "path": path.as_posix(),
                "sha256": sha256(path),
                "seed": int(training_config["seed"]),
                "architecture": checkpoint["model_config"]["architecture"],
                "validation_selected_threshold": float(checkpoint["validation"]["threshold"]),
                "validation_boundary_f1": float(
                    checkpoint["validation"]["overall"]["boundary_f1"]
                ),
                "training_config_sha256": hashlib.sha256(
                    json.dumps(training_config, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    report = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_family": args.primary_family,
        "checkpoints": checkpoints,
        "sealed_manifest": sealed_manifest.as_posix(),
        "sealed_manifest_sha256": sha256(sealed_manifest),
        "sealed_test_path": sealed["test_path"],
        "sealed_test_sha256": sealed["test_sha256"],
        "rule": "checkpoint, validation-selected threshold, and sealed dataset are immutable before first sealed evaluation",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
