from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fair paired-candidate stress-test config")
    parser.add_argument("--base", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.base).read_text(encoding="utf-8"))
    config["seed"] = args.seed
    config["output_dir"] = f"runs/controlled_candidate_selection/{args.tag}_seed{args.seed}"
    config["data"] = {
        "train": "data/processed/controlled_candidate_selection/train.jsonl",
        "validation": "data/processed/controlled_candidate_selection/validation.jsonl",
    }
    config["training"].update({"batch_size": 8, "eval_batch_size": 4, "workers": 4})
    # Use the paper's fixed task loss.  A ratio-weighted pilot (979.99) made
    # false candidate markers effectively free and collapsed to predicting
    # both candidates; that pilot is retained separately and is not analyzed.
    config["loss"]["boundary_positive_weight"] = 8.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
