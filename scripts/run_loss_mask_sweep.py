from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


REGIMES = [
    ("full_sequence", None),
    ("full_sequence_final_weighted", 5.0),
    ("full_sequence_final_weighted", 10.0),
    ("completion_only", None),
    ("completion_final_weighted", 5.0),
    ("completion_final_weighted", 10.0),
    ("final_count_only", None),
]


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def regime_name(loss_mask: str, final_weight: float | None) -> str:
    if final_weight is None:
        return loss_mask
    return f"{loss_mask}_fw{final_weight:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recommended v0 loss-mask sweep.")
    parser.add_argument("--data_dir", default="data/trace_count_v0")
    parser.add_argument("--model_config", default="configs/model/tiny_debug.yaml")
    parser.add_argument("--model_name", default="tiny_debug")
    parser.add_argument("--out_root", default="runs/trace_count_v0")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_splits", default="val_id,val_length_ood,val_density_shift_low,val_density_shift_high")
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    for seed in seeds:
        for loss_mask, final_weight in REGIMES:
            name = regime_name(loss_mask, final_weight)
            run_dir = Path(args.out_root) / args.model_name / f"{name}_seed{seed}"
            train_cmd = [
                sys.executable,
                "-m",
                "trace_counting.train",
                "--data_dir",
                str(args.data_dir),
                "--model_config",
                str(args.model_config),
                "--loss_mask",
                loss_mask,
                "--seed",
                str(seed),
                "--out_dir",
                str(run_dir),
                "--max_steps",
                str(args.max_steps),
                "--batch_size",
                str(args.batch_size),
            ]
            if final_weight is not None:
                train_cmd += ["--final_weight", str(final_weight)]
            if args.device:
                train_cmd += ["--device", args.device]
            _run(train_cmd, dry_run=args.dry_run)

            eval_cmd = [
                sys.executable,
                "-m",
                "trace_counting.eval",
                "--checkpoint",
                str(run_dir / "checkpoints" / "final"),
                "--data_dir",
                str(args.data_dir),
                "--splits",
                args.eval_splits,
                "--out_dir",
                str(run_dir / "eval"),
            ]
            if args.eval_limit is not None:
                eval_cmd += ["--limit", str(args.eval_limit)]
            if args.device:
                eval_cmd += ["--device", args.device]
            _run(eval_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
