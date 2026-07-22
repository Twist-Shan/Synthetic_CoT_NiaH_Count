#!/usr/bin/env python3
"""Run the one-factor-at-a-time v20 phase-transition controls.

The suite changes exactly one of objective timing, accepted-count sampling, or
data-window length relative to the v20 baseline.  All runs use the thinking
model only, seed 1234, 100-step scientific snapshots, and the same diagnostic
stages.  This keeps the comparison focused and cuts the compute roughly in half.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS = (
    {
        "name": "baseline_switch1500_natural_L256",
        "factor": "baseline",
        "max_steps_for_language_pred": 1500,
        "training_count_distribution": "natural",
        "seq_len": 256,
        "n_positions": 384,
    },
    {
        "name": "switch0000_natural_L256",
        "factor": "objective_switch",
        "max_steps_for_language_pred": 0,
        "training_count_distribution": "natural",
        "seq_len": 256,
        "n_positions": 384,
    },
    {
        "name": "switch3000_natural_L256",
        "factor": "objective_switch",
        "max_steps_for_language_pred": 3000,
        "training_count_distribution": "natural",
        "seq_len": 256,
        "n_positions": 384,
    },
    {
        "name": "switch10000_natural_L256",
        "factor": "objective_switch",
        "max_steps_for_language_pred": 10000,
        "training_count_distribution": "natural",
        "seq_len": 256,
        "n_positions": 384,
    },
    {
        "name": "switch1500_uniform_L256",
        "factor": "count_distribution",
        "max_steps_for_language_pred": 1500,
        "training_count_distribution": "uniform",
        "seq_len": 256,
        "n_positions": 384,
    },
    {
        "name": "switch1500_natural_L128",
        "factor": "sequence_length",
        "max_steps_for_language_pred": 1500,
        "training_count_distribution": "natural",
        "seq_len": 128,
        "n_positions": 256,
    },
    {
        "name": "switch1500_natural_L384",
        "factor": "sequence_length",
        "max_steps_for_language_pred": 1500,
        "training_count_distribution": "natural",
        "seq_len": 384,
        "n_positions": 512,
    },
)


def command(experiment: dict[str, object], out_root: Path, device: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "synthetic_counting_v20.run_v20",
        "--preset",
        "main",
        "--stage",
        "prepare,train,phase,plots",
        "--device",
        device,
        "--seed",
        "1234",
        "--model-variant",
        "rope/thinking",
        "--checkpoint-every",
        "100",
        "--max-steps-for-language-pred",
        str(experiment["max_steps_for_language_pred"]),
        "--training-count-distribution",
        str(experiment["training_count_distribution"]),
        "--seq-len",
        str(experiment["seq_len"]),
        "--n-positions",
        str(experiment["n_positions"]),
        "--out-root",
        str(out_root),
        "--run-name",
        f"v20_phase_{experiment['name']}_seed1234",
        "--skip-completed",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=ROOT / "colab_results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    selected = [item for item in EXPERIMENTS if not args.only or item["name"] in args.only]
    manifest = {
        "design": "one-factor-at-a-time relative to baseline_switch1500_natural_L256",
        "shared": {
            "version": "v20",
            "seed": 1234,
            "mode": "rope/thinking",
            "count_range": "1-30",
            "checkpoint_every": 100,
            "train_steps": 10000,
        },
        "analysis_rule": (
            "Compare candidate transition both by optimizer step and by "
            "training_token_exposure_by_k; a switch-locked change supports curriculum, "
            "an exposure-locked change supports exposure, and neither alone establishes "
            "a seed-stable phase transition."
        ),
        "experiments": selected,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "v20_phase_ablation_design.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for experiment in selected:
        cmd = command(experiment, args.out_root, args.device)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
