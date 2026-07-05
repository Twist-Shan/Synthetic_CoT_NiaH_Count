from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trace_counting.io_utils import load_yaml


def _run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def _list_arg(value) -> str:
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def _path(value: str | Path) -> str:
    value = Path(value)
    return str(value if value.is_absolute() else ROOT / value)


def data_cmd(config: dict) -> list[str]:
    data = config["data"]
    cmd = [
        sys.executable,
        "-m",
        "trace_counting.generate_data",
        "--out_dir",
        _path(data["out_dir"]),
        "--max_count",
        str(data.get("max_count", 64)),
        "--noise_vocab_size",
        str(data.get("noise_vocab_size", 64)),
        "--train_lengths",
        _list_arg(data["train_lengths"]),
        "--train_counts",
        _list_arg(data["train_counts"]),
        "--examples_per_pair_train",
        str(data.get("examples_per_pair_train", 512)),
        "--examples_per_pair_val",
        str(data.get("examples_per_pair_val", 128)),
        "--seeds",
        _list_arg(data.get("seeds", [0, 1, 2])),
    ]
    for key in [
        "val_id_lengths",
        "val_id_counts",
        "val_length_ood_lengths",
        "val_length_ood_counts",
        "val_density_shift_low_lengths",
        "val_density_shift_low_counts",
        "val_density_shift_high_lengths",
        "val_density_shift_high_counts",
    ]:
        if key in data and data[key] is not None:
            cmd += [f"--{key}", _list_arg(data[key])]
    return cmd


def train_cmd(config: dict) -> list[str]:
    data = config["data"]
    train = config["training"]
    cmd = [
        sys.executable,
        "-m",
        "trace_counting.train",
        "--data_dir",
        _path(data["out_dir"]),
        "--model_config",
        _path(train["model_config"]),
        "--loss_mask",
        str(train["loss_mask"]),
        "--final_weight",
        str(train.get("final_weight", 10.0)),
        "--eos_weight",
        str(train.get("eos_weight", 1.0)),
        "--seed",
        str(train.get("seed", 0)),
        "--out_dir",
        _path(train["out_dir"]),
        "--batch_size",
        str(train.get("batch_size", 128)),
        "--grad_accum_steps",
        str(train.get("grad_accum_steps", 1)),
        "--max_steps",
        str(train.get("max_steps", 50000)),
        "--learning_rate",
        str(train.get("learning_rate", 3.0e-4)),
        "--warmup_steps",
        str(train.get("warmup_steps", 1000)),
        "--weight_decay",
        str(train.get("weight_decay", 0.1)),
        "--grad_clip_norm",
        str(train.get("grad_clip_norm", 1.0)),
        "--eval_every",
        str(train.get("eval_every", 1000)),
        "--save_every",
        str(train.get("save_every", 5000)),
    ]
    if train.get("device"):
        cmd += ["--device", str(train["device"])]
    if train.get("final_count_only_include_eos", False):
        cmd += ["--final_count_only_include_eos"]
    return cmd


def eval_cmd(config: dict) -> list[str]:
    data = config["data"]
    train = config["training"]
    eval_cfg = config.get("eval", {})
    run_dir = Path(_path(train["out_dir"]))
    cmd = [
        sys.executable,
        "-m",
        "trace_counting.eval",
        "--checkpoint",
        str(run_dir / "checkpoints" / "final"),
        "--data_dir",
        _path(data["out_dir"]),
        "--splits",
        _list_arg(eval_cfg.get("splits", ["val_id", "val_length_ood", "val_density_shift_low", "val_density_shift_high"])),
        "--out_dir",
        str(run_dir / "eval"),
    ]
    if eval_cfg.get("limit") is not None:
        cmd += ["--limit", str(eval_cfg["limit"])]
    return cmd


def probe_cmd(config: dict) -> list[str]:
    data = config["data"]
    train = config["training"]
    probe = config.get("probe", {})
    run_dir = Path(_path(train["out_dir"]))
    cmd = [
        sys.executable,
        "-m",
        "trace_counting.probes",
        "--checkpoint",
        str(run_dir / "checkpoints" / "final"),
        "--data_dir",
        _path(data["out_dir"]),
        "--split",
        str(probe.get("split", "val_id")),
        "--out_dir",
        str(run_dir / "probes"),
        "--anchors",
        str(probe.get("anchors", "ans,think_open,think_close,source,trace_index,trace_marker")),
        "--layers",
        str(probe.get("layers", "-1")),
    ]
    if probe.get("limit") is not None:
        cmd += ["--limit", str(probe["limit"])]
    return cmd


def plots_cmd(config: dict) -> list[str]:
    run_dir = _path(config["training"]["out_dir"])
    return [sys.executable, "-m", "trace_counting.plots", "--run_dir", run_dir]


def summarize_cmd(config: dict) -> list[str]:
    exp_name = config.get("experiment", {}).get("name", "trace_count_v0")
    runs_dir = ROOT / "runs" / exp_name
    return [
        sys.executable,
        "-m",
        "trace_counting.summarize",
        "--runs_dir",
        str(runs_dir),
        "--out_csv",
        str(runs_dir / "summary.csv"),
        "--print_markdown",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trace-counting experiment pipeline from YAML.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="all", choices=["all", "data", "train", "eval", "probe", "plots", "summarize"])
    args = parser.parse_args()

    config = load_yaml(_path(args.config))
    stages = ["data", "train", "eval", "probe", "plots"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "data":
            _run(data_cmd(config))
        elif stage == "train":
            _run(train_cmd(config))
        elif stage == "eval":
            _run(eval_cmd(config))
        elif stage == "probe":
            _run(probe_cmd(config))
        elif stage == "plots":
            _run(plots_cmd(config))
        elif stage == "summarize":
            _run(summarize_cmd(config))


if __name__ == "__main__":
    main()
