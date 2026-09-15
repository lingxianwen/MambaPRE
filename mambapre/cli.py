from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import benchmark_architecture
from .engine import evaluate_model, load_checkpoint, make_loader, resolve_device, train_from_config
from .neupre import prepare_neupre_dataset
from .prepare import prepare_dataset


def read_json(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_report(report: dict, path: str | None) -> None:
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


def command_prepare(args: argparse.Namespace) -> None:
    report = prepare_dataset(
        source_train=args.source_train,
        source_test=args.source_test,
        source_ood=args.source_ood,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    write_report(report, None)


def command_prepare_neupre(args: argparse.Namespace) -> None:
    report = prepare_neupre_dataset(
        records_dir=args.records_dir,
        pdml_dir=args.pdml_dir,
        output_dir=args.output_dir,
        reference_paths=args.reference_data,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    write_report(report, None)


def command_train(args: argparse.Namespace) -> None:
    config = read_json(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    write_report(train_from_config(config), None)


def command_evaluate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["training_config"]
    loader = make_loader(
        args.data,
        args.batch_size,
        model.config.max_length,
        False,
        args.workers,
        int(config.get("seed", 1337)),
    )
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(checkpoint.get("validation", {}).get("threshold", 0.5))
    )
    report = evaluate_model(
        model,
        loader,
        device,
        threshold,
        config.get("loss", {}),
    )
    report["checkpoint"] = args.checkpoint
    report["data"] = args.data
    write_report(report, args.output)


def command_calibrate_evaluate(args: argparse.Namespace) -> None:
    """Select a threshold on calibration data and freeze it for held-out test."""
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["training_config"]
    seed = int(config.get("seed", 1337))
    calibration_loader = make_loader(
        args.calibration_data,
        args.batch_size,
        model.config.max_length,
        False,
        args.workers,
        seed,
    )
    test_loader = make_loader(
        args.test_data,
        args.batch_size,
        model.config.max_length,
        False,
        args.workers,
        seed,
    )
    thresholds = [float(value) for value in args.threshold_grid.split(",")]
    calibration = evaluate_model(
        model,
        calibration_loader,
        device,
        thresholds,
        config.get("loss", {}),
    )
    selected = float(calibration["threshold"])
    test = evaluate_model(
        model,
        test_loader,
        device,
        selected,
        config.get("loss", {}),
    )
    report = {
        "checkpoint": args.checkpoint,
        "calibration_data": args.calibration_data,
        "test_data": args.test_data,
        "selection_rule": "maximize boundary F1 on calibration data only",
        "selected_threshold": selected,
        "calibration": calibration,
        "test": test,
    }
    write_report(report, args.output)


def command_benchmark(args: argparse.Namespace) -> None:
    config = read_json(args.config)
    model_config = config.get("model", config)
    report = benchmark_architecture(
        model_config=model_config,
        lengths=[int(value) for value in args.lengths.split(",")],
        batch_size=args.batch_size,
        device_name=args.device,
        warmup=args.warmup,
        repetitions=args.repetitions,
        seed=args.seed,
        precision=args.precision,
        attention_backend=args.attention_backend,
    )
    write_report(report, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mamba-PRE experiment runner")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="convert legacy PRE JSONL")
    prepare.add_argument("--source-train", required=True)
    prepare.add_argument("--source-test", required=True)
    prepare.add_argument("--source-ood")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=1337)
    prepare.set_defaults(func=command_prepare)

    prepare_neupre = commands.add_parser(
        "prepare-neupre", help="prepare scope-separated NeuPRE external tests"
    )
    source = prepare_neupre.add_mutually_exclusive_group(required=True)
    source.add_argument("--records-dir", help="rich JSONL records regenerated from PCAP")
    source.add_argument("--pdml-dir", help="boundary-only NeuPRE pdml_gt JSON maps")
    prepare_neupre.add_argument("--output-dir", required=True)
    prepare_neupre.add_argument(
        "--reference-data",
        action="append",
        default=[],
        help="training JSONL used for byte-level overlap removal; repeat as needed",
    )
    prepare_neupre.add_argument("--calibration-fraction", type=float, default=0.3)
    prepare_neupre.add_argument("--seed", type=int, default=1337)
    prepare_neupre.set_defaults(func=command_prepare_neupre)

    train = commands.add_parser("train", help="train one architecture")
    train.add_argument("--config", required=True)
    train.add_argument("--output-dir")
    train.set_defaults(func=command_train)

    evaluate = commands.add_parser("evaluate", help="evaluate a checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--output")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--workers", type=int, default=0)
    evaluate.add_argument(
        "--threshold",
        type=float,
        help="override the checkpoint's validation-selected threshold",
    )
    evaluate.set_defaults(func=command_evaluate)

    calibrate = commands.add_parser(
        "calibrate-evaluate",
        help="select threshold on calibration sessions and freeze it on test sessions",
    )
    calibrate.add_argument("--checkpoint", required=True)
    calibrate.add_argument("--calibration-data", required=True)
    calibrate.add_argument("--test-data", required=True)
    calibrate.add_argument("--output")
    calibrate.add_argument("--device", default="auto")
    calibrate.add_argument("--batch-size", type=int, default=8)
    calibrate.add_argument("--workers", type=int, default=0)
    calibrate.add_argument(
        "--threshold-grid",
        default="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9",
    )
    calibrate.set_defaults(func=command_calibrate_evaluate)

    benchmark = commands.add_parser("benchmark", help="measure scaling and memory")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--lengths", default="256,512,1024,2048,4096")
    benchmark.add_argument("--batch-size", type=int, default=8)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--repetitions", type=int, default=30)
    benchmark.add_argument("--seed", type=int, default=1337)
    benchmark.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    benchmark.add_argument(
        "--attention-backend",
        choices=("auto", "math", "memory_efficient", "flash"),
        default="auto",
        help="force the Transformer SDPA kernel; auto preserves the standard PyTorch path",
    )
    benchmark.add_argument("--output")
    benchmark.set_defaults(func=command_benchmark)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
