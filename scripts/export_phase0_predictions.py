from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from mambapre.engine import load_checkpoint, make_loader, move_batch, resolve_device
from mambapre.metrics import BoundaryAccumulator


SEEDS = (1337, 2027, 3407, 4701, 9001)
DATASETS = {
    "test_novel": "data/processed/core_corpus/test_novel.jsonl",
    "ood_novel": "data/processed/core_corpus/ood_novel.jsonl",
    "neupre_full_novel": "data/processed/neupre_pdml/test_external_full_novel.jsonl",
    "neupre_long_session_calibrated": "data/processed/neupre_pdml/test_long_envelope_session_disjoint_novel.jsonl",
    "long_full_strict_calibrated": "data/processed/long_full_strict/test_long_full_capture_disjoint_novel.jsonl",
}
EVAL_BATCH_SIZE = {
    "test_novel": 32,
    "ood_novel": 32,
    "neupre_full_novel": 32,
    "neupre_long_session_calibrated": 4,
    "long_full_strict_calibrated": 4,
}
MODELS = {
    "batch_invariant_guided": {
        "checkpoint": "runs/batch_invariant_revision/guided_mamba_seed{seed}/best.pt",
        "result": "results/batch_invariant_revision/retrained/seed{seed}/{dataset}.json",
        "datasets": tuple(DATASETS),
    },
    "guided_mamba": {
        "run": "dual_bimamba_guided_dedup",
        "result": "results/multiseed/seed{seed}/guided_mamba/{dataset}.json",
        "datasets": tuple(DATASETS),
    },
    "matched_transformer": {
        "run": "transformer_param_matched_dedup",
        "result": "results/multiseed/seed{seed}/matched_transformer/{dataset}.json",
        "datasets": tuple(DATASETS),
    },
    "matched_cnn": {
        "run": "cnn_param_matched_dedup",
        "result": "results/multiseed/seed{seed}/matched_cnn/{dataset}.json",
        "datasets": tuple(DATASETS),
    },
    "bimamba_only": {
        "run": "bimamba_param_matched",
        "result": "results/bimamba_ablation_multiseed/seed{seed}/{dataset}.json",
        "datasets": ("neupre_long_session_calibrated", "long_full_strict_calibrated"),
    },
    "dual_unguided": {
        "run": "dual_bimamba",
        "result": "results/key_ablation_multiseed/seed{seed}/dual_unguided/{dataset}.json",
        "datasets": ("neupre_long_session_calibrated", "long_full_strict_calibrated"),
    },
    "guided_no_semantic_aux": {
        "run": "dual_bimamba_guided_no_aux",
        "result": "results/key_ablation_multiseed/seed{seed}/guided_no_semantic_aux/{dataset}.json",
        "datasets": ("neupre_long_session_calibrated", "long_full_strict_calibrated"),
    },
}


def metric_root(report: dict) -> dict:
    return report["test"] if "test" in report else report


def threshold_from(report: dict) -> float:
    return float(report.get("selected_threshold", metric_root(report).get("threshold", 0.5)))


@torch.no_grad()
def export_one(checkpoint_path: Path, data_path: Path, threshold: float, output: Path, device: torch.device, batch_size: int) -> dict:
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    model.eval()
    seed = int(checkpoint["training_config"].get("seed", 1337))
    loader = make_loader(str(data_path), batch_size, model.config.max_length, False, 0, seed)
    counts = BoundaryAccumulator()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "protocol", "length", "threshold", "gold_boundaries", "predicted_boundaries"))
        writer.writeheader()
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            probabilities = torch.sigmoid(model(batch["bytes"], batch["mask"])["boundary_logits"]).detach().cpu().numpy()
            gold = cpu_batch["boundary_labels"].numpy()
            for index, length_tensor in enumerate(cpu_batch["lengths"]):
                length = int(length_tensor)
                predicted = set(np.flatnonzero(probabilities[index, :length] >= threshold).tolist())
                expected = set(np.flatnonzero(gold[index, :length] > 0.5).tolist())
                predicted.discard(0)
                expected.discard(0)
                counts.add(predicted, expected, length)
                writer.writerow({
                    "id": cpu_batch["ids"][index],
                    "protocol": cpu_batch["protocols"][index],
                    "length": length,
                    "threshold": threshold,
                    "gold_boundaries": " ".join(map(str, sorted(expected))),
                    "predicted_boundaries": " ".join(map(str, sorted(predicted))),
                })
    return counts.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and verify Phase-0 per-message predictions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--datasets", default="", help="Optional comma-separated dataset subset")
    parser.add_argument("--output-root", default="results/reproduction/per_message")
    args = parser.parse_args()
    device = resolve_device(args.device)
    requested_models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    seeds = tuple(int(value) for value in args.seeds.split(","))
    requested_datasets = {
        value.strip() for value in args.datasets.split(",") if value.strip()
    }
    output_root = Path(args.output_root)
    manifest = {"device": str(device), "models": {}, "checks": []}
    for model_name in requested_models:
        spec = MODELS[model_name]
        manifest["models"][model_name] = {}
        for seed in seeds:
            checkpoint = (
                Path(str(spec["checkpoint"]).format(seed=seed))
                if "checkpoint" in spec
                else Path("runs") / f"{spec['run']}_seed{seed}" / "best.pt"
            )
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            seed_records = {}
            for dataset in spec["datasets"]:
                if requested_datasets and dataset not in requested_datasets:
                    continue
                result_path = Path(str(spec["result"]).format(seed=seed, dataset=dataset))
                report = json.loads(result_path.read_text(encoding="utf-8"))
                threshold = threshold_from(report)
                output = output_root / model_name / f"seed{seed}" / f"{dataset}.csv"
                observed = export_one(
                    checkpoint,
                    Path(DATASETS[dataset]),
                    threshold,
                    output,
                    device,
                    EVAL_BATCH_SIZE.get(dataset, args.batch_size),
                )
                expected = metric_root(report)["overall"]
                deltas = {metric: abs(float(observed[metric]) - float(expected[metric])) for metric in ("boundary_precision", "boundary_recall", "boundary_f1", "exact_field_f1", "message_perfection")}
                maximum = max(deltas.values())
                if maximum > 1e-12:
                    raise RuntimeError(f"metric mismatch {model_name} seed={seed} dataset={dataset}: {deltas}")
                seed_records[dataset] = {"checkpoint": str(checkpoint), "result": str(result_path), "predictions": str(output), "threshold": threshold, "n": observed["n"], "max_metric_delta": maximum}
                manifest["checks"].append({"model": model_name, "seed": seed, "dataset": dataset, "status": "exact", "max_metric_delta": maximum})
                print(f"exact {model_name} seed={seed} dataset={dataset} n={observed['n']} threshold={threshold}")
            manifest["models"][model_name][str(seed)] = seed_records
    manifest["all_exact"] = all(item["status"] == "exact" for item in manifest["checks"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root.parent / "per_message_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
