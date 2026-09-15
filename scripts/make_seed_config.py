from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a seeded training configuration")
    parser.add_argument("--base", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.base).read_text(encoding="utf-8"))
    config["seed"] = args.seed
    config["output_dir"] = args.output_dir
    output = Path(args.config_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

