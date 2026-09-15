from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fair controlled-long-range training config")
    parser.add_argument("--base", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.base).read_text(encoding="utf-8"))
    manifest = json.loads(Path("data/processed/controlled_long_range/manifest.json").read_text(encoding="utf-8"))
    config["seed"] = args.seed
    config["output_dir"] = f"runs/controlled_long_range/{args.tag}_seed{args.seed}"
    config["data"] = {
        "train": "data/processed/controlled_long_range/train.jsonl",
        "validation": "data/processed/controlled_long_range/validation.jsonl",
    }
    config["training"].update({"batch_size": 8, "eval_batch_size": 4, "workers": 4})
    config["loss"]["boundary_positive_weight"] = float(manifest["training_boundary_pos_weight"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
