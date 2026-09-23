from __future__ import annotations

import argparse
import json
from pathlib import Path

from mambapre.reporting import summarize_values


DATASETS = {
    "id_novel": ("test_novel.json", ("overall",)),
    "source_held_out": ("ood_novel.json", ("overall",)),
    "external_full": ("neupre_full_novel.json", ("overall",)),
    "long_envelope": ("neupre_long_session_calibrated.json", ("test", "overall")),
    "strict_fins": ("long_full_strict_calibrated.json", ("test", "overall")),
    "sealed_opcua": ("sealed_opcua.json", ("overall",)),
}


def metric(path: Path, keys: tuple[str, ...], name: str) -> float:
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in keys:
        value = value[key]
    return float(value[name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate matched BiGRU five-seed results")
    parser.add_argument("--root", default="results/multiseed")
    parser.add_argument(
        "--mamba-root", default="results/batch_invariant_revision/retrained"
    )
    parser.add_argument("--seeds", default="1337,2027,3407,4701,9001")
    parser.add_argument("--output", default="results/bigru/aggregate.json")
    args = parser.parse_args()
    root = Path(args.root)
    mamba_root = Path(args.mamba_root)
    seeds = [int(value) for value in args.seeds.split(",")]
    result = {
        "parameters": 2_191_922,
        "matched_transformer_parameters": 2_191_502,
        "relative_parameter_difference": (2_191_922 - 2_191_502) / 2_191_502,
        "seeds": seeds,
        "datasets": {},
    }
    for dataset, (filename, keys) in DATASETS.items():
        bigru = [
            metric(root / f"seed{seed}" / "matched_bigru" / filename, keys, "boundary_f1")
            for seed in seeds
        ]
        mamba = [
            metric(mamba_root / f"seed{seed}" / filename, keys, "boundary_f1")
            for seed in seeds
        ] if dataset != "sealed_opcua" else []
        entry = {"bigru": summarize_values(bigru)}
        if mamba:
            entry["mambapre"] = summarize_values(mamba)
            entry["paired_mambapre_minus_bigru"] = summarize_values(
                [left - right for left, right in zip(mamba, bigru)]
            )
        result["datasets"][dataset] = entry
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
