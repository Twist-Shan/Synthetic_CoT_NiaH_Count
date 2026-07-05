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


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("$", " ".join(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def regime_name(loss_mask: str, final_weight: float | None) -> str:
    if final_weight is None:
        return loss_mask
    return f"{loss_mask}_fw{final_weight:g}"


def preflight_data_dir(data_dir: str | Path, eval_splits: str) -> None:
    data_path = _resolve(data_dir)
    required = ["vocab.json", "train.jsonl"]
    required.extend(f"{split.strip()}.jsonl" for split in eval_splits.split(",") if split.strip())
    missing = [name for name in required if not (data_path / name).exists()]
    if not missing:
        return

    missing_text = ", ".join(missing)
    raise FileNotFoundError(
        f"Missing dataset files under {data_path}: {missing_text}\n\n"
        "Generate the dataset before running the sweep. For the notebook's single-seed full v0 sweep, run:\n"
        "  python scripts/run_pipeline.py --config configs/experiment/v0_seed0.yaml --stage data\n\n"
        "For the three-seed full v0 sweep, run:\n"
        "  python scripts/run_pipeline.py --config configs/experiment/v0.yaml --stage data\n\n"
        "For the small Colab/debug dataset, run:\n"
        "  python scripts/run_pipeline.py --config configs/experiment/debug.yaml --stage data\n"
        "  python scripts/run_loss_mask_sweep.py --data_dir data/trace_count_v0_debug "
        "--out_root runs/trace_count_v0_debug --model_config configs/model/tiny_debug.yaml "
        "--model_name tiny_debug --max_steps 100 --eval_limit 128"
    )


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
    parser.add_argument("--skip_completed", action="store_true", help="Skip runs that already have checkpoints/final/config.json.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        preflight_data_dir(args.data_dir, args.eval_splits)

    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    run_specs = [(seed, loss_mask, final_weight) for seed in seeds for loss_mask, final_weight in REGIMES]
    print(
        "\n".join(
            [
                "",
                "=" * 88,
                f"Loss-mask sweep: {len(run_specs)} training runs",
                f"data_dir={args.data_dir}",
                f"model={args.model_name} config={args.model_config}",
                f"seeds={','.join(str(seed) for seed in seeds)} max_steps={args.max_steps}",
                "=" * 88,
            ]
        ),
        flush=True,
    )
    for run_idx, (seed, loss_mask, final_weight) in enumerate(run_specs, start=1):
        name = regime_name(loss_mask, final_weight)
        run_dir = Path(args.out_root) / args.model_name / f"{name}_seed{seed}"
        final_checkpoint = run_dir / "checkpoints" / "final" / "config.json"
        print(
            "\n".join(
                [
                    "",
                    "-" * 88,
                    f"RUN {run_idx}/{len(run_specs)}: seed={seed} loss_mask={loss_mask}"
                    + ("" if final_weight is None else f" final_weight={final_weight:g}"),
                    f"run_dir={run_dir}",
                    "-" * 88,
                ]
            ),
            flush=True,
        )
        if args.skip_completed and final_checkpoint.exists():
            print(f"[skip] RUN {run_idx}/{len(run_specs)}: found {final_checkpoint}", flush=True)
        else:
            train_cmd = [
                sys.executable,
                "-u",
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

        print(f"[eval] RUN {run_idx}/{len(run_specs)}: evaluating {run_dir}", flush=True)
        eval_cmd = [
            sys.executable,
            "-u",
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
