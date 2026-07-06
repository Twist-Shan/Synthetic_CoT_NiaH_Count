from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _run(cmd: list[str], *, skip_if: Path | None = None) -> None:
    if skip_if is not None and skip_if.exists():
        print(f"[skip] {skip_if}")
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    print("$", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def _python_module(module: str, *args: str) -> list[str]:
    return [sys.executable, "-u", "-m", module, *args]


def generate_data(args: argparse.Namespace, *, task_format: str, out_dir: Path) -> None:
    _run(
        _python_module(
            "trace_counting.generate_data",
            "--out_dir",
            str(out_dir),
            "--max_count",
            str(args.max_count),
            "--noise_vocab_size",
            str(args.noise_vocab_size),
            "--train_lengths",
            args.lengths,
            "--train_counts",
            args.id_counts,
            "--val_id_lengths",
            args.lengths,
            "--val_id_counts",
            args.id_counts,
            "--val_count_ood_lengths",
            args.lengths,
            "--val_count_ood_counts",
            args.ood_counts,
            "--examples_per_pair_train",
            str(args.examples_per_pair_train),
            "--examples_per_pair_val",
            str(args.examples_per_pair_val),
            "--seeds",
            str(args.seed),
            "--task_format",
            task_format,
            "--no_legacy_shifts",
        ),
        skip_if=out_dir / "dataset_metadata.json" if args.skip_completed else None,
    )


def train_run(args: argparse.Namespace, *, data_dir: Path, run_dir: Path) -> None:
    _run(
        _python_module(
            "trace_counting.train",
            "--data_dir",
            str(data_dir),
            "--model_config",
            str(ROOT / args.model_config),
            "--loss_mask",
            "full_sequence",
            "--final_weight",
            "1",
            "--seed",
            str(args.seed),
            "--out_dir",
            str(run_dir),
            "--batch_size",
            str(args.batch_size),
            "--max_steps",
            str(args.max_steps),
            "--learning_rate",
            str(args.learning_rate),
            "--warmup_steps",
            str(args.warmup_steps),
            "--eval_every",
            str(args.eval_every),
            "--eval_limit",
            str(args.eval_limit),
            "--save_every",
            str(args.save_every),
            "--progress_every",
            str(args.progress_every),
        ),
        skip_if=run_dir / "checkpoints" / "final" / "config.json" if args.skip_completed else None,
    )


def eval_run(args: argparse.Namespace, *, data_dir: Path, run_dir: Path) -> None:
    cmd = _python_module(
        "trace_counting.eval",
        "--checkpoint",
        str(run_dir / "checkpoints" / "final"),
        "--data_dir",
        str(data_dir),
        "--splits",
        "val_id,val_count_ood",
        "--out_dir",
        str(run_dir / "eval"),
        "--limit",
        str(args.eval_limit),
    )
    _run(cmd, skip_if=run_dir / "eval" / "summary_metrics.json" if args.skip_completed else None)


def probe_run(args: argparse.Namespace, *, data_dir: Path, run_dir: Path) -> None:
    checkpoint = run_dir / "checkpoints" / "final"
    _run(
        _python_module(
            "trace_counting.probes",
            "--checkpoint",
            str(checkpoint),
            "--data_dir",
            str(data_dir),
            "--split",
            "val_id",
            "--out_dir",
            str(run_dir / "probes"),
            "--anchors",
            "ans,think_close,source_marker,trace_index,trace_marker",
            "--layers",
            "all",
            "--limit",
            str(args.probe_limit),
        ),
        skip_if=run_dir / "probes" / "probe_summary.json" if args.skip_completed else None,
    )
    _run(
        _python_module(
            "trace_counting.directions",
            "--checkpoint",
            str(checkpoint),
            "--data_dir",
            str(data_dir),
            "--split",
            "val_id",
            "--out_dir",
            str(run_dir / "directions"),
            "--anchors",
            "ans,think_close,source_marker,trace_index,trace_marker",
            "--layers",
            "all",
            "--targets",
            "total_count,running_count,k",
            "--limit",
            str(args.probe_limit),
            "--seed",
            str(args.seed),
        ),
        skip_if=run_dir / "directions" / "direction_metadata.json" if args.skip_completed else None,
    )


def steering_run(args: argparse.Namespace, *, data_dir: Path, run_dir: Path) -> None:
    _run(
        _python_module(
            "trace_counting.steering",
            "--checkpoint",
            str(run_dir / "checkpoints" / "final"),
            "--data_dir",
            str(data_dir),
            "--split",
            "val_count_ood",
            "--direction_dir",
            str(run_dir / "directions"),
            "--out_dir",
            str(run_dir / "steering"),
            "--limit",
            str(args.steering_limit),
            "--layer",
            "final",
            "--anchor",
            "ans",
            "--target",
            "total_count",
            f"--alphas={args.steering_alphas}",
        ),
        skip_if=run_dir / "steering" / "steering_summary.csv" if args.skip_completed else None,
    )


def attention_run(args: argparse.Namespace, *, data_dir: Path, run_dir: Path) -> None:
    _run(
        _python_module(
            "trace_counting.attention_analysis",
            "--checkpoint",
            str(run_dir / "checkpoints" / "final"),
            "--data_dir",
            str(data_dir),
            "--splits",
            "val_id,val_count_ood",
            "--out_dir",
            str(run_dir / "attention"),
            "--limit",
            str(args.attention_limit),
            "--query_anchors",
            "ans,think_close",
        ),
        skip_if=run_dir / "attention" / "attention_summary.csv" if args.skip_completed else None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Trace Count v1 NiaH-like think/no-think experiment.")
    parser.add_argument("--out_root", default="runs/trace_count_v1_seed0")
    parser.add_argument("--data_root", default="data/trace_count_v1_seed0")
    parser.add_argument("--model_config", default="configs/model/small_main.yaml")
    parser.add_argument("--model_name", default="small_main")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lengths", default="50,100,200")
    parser.add_argument("--id_counts", default="0:5")
    parser.add_argument("--ood_counts", default="5:10")
    parser.add_argument("--max_count", type=int, default=10)
    parser.add_argument("--noise_vocab_size", type=int, default=64)
    parser.add_argument("--examples_per_pair_train", type=int, default=512)
    parser.add_argument("--examples_per_pair_val", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_limit", type=int, default=2048)
    parser.add_argument("--probe_limit", type=int, default=2048)
    parser.add_argument("--steering_limit", type=int, default=1024)
    parser.add_argument("--attention_limit", type=int, default=512)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--steering_alphas", default="-4,-2,-1,0,1,2,4")
    parser.add_argument("--stage", default="all", choices=["all", "data", "train", "eval", "probe", "steering", "attention"])
    parser.add_argument("--skip_completed", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_root = ROOT / args.out_root
    data_root = ROOT / args.data_root
    variants = [
        ("think_trace", "think_trace_full_sequence_seed0"),
        ("answer_only", "answer_only_full_sequence_seed0"),
    ]
    stages = ["data", "train", "eval", "probe", "steering", "attention"] if args.stage == "all" else [args.stage]

    for task_format, run_name in variants:
        data_dir = data_root / task_format
        run_dir = out_root / args.model_name / run_name
        if "data" in stages:
            generate_data(args, task_format=task_format, out_dir=data_dir)
        if "train" in stages:
            train_run(args, data_dir=data_dir, run_dir=run_dir)
        if "eval" in stages:
            eval_run(args, data_dir=data_dir, run_dir=run_dir)
        if "probe" in stages:
            probe_run(args, data_dir=data_dir, run_dir=run_dir)
        if "steering" in stages:
            steering_run(args, data_dir=data_dir, run_dir=run_dir)
        if "attention" in stages:
            attention_run(args, data_dir=data_dir, run_dir=run_dir)


if __name__ == "__main__":
    main()
