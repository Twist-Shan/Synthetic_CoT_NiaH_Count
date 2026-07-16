from __future__ import annotations

import argparse
from pathlib import Path

from synthetic_counting_v10.final_bridge_causal import run_final_bridge_causal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run v10 CoT final-source and non-thinking MLP bridge causality analyses."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--behavior-examples-per-pair", type=int, default=4)
    parser.add_argument("--patch-examples-per-pair", type=int, default=2)
    parser.add_argument("--feature-fit-examples-per-pair", type=int, default=1)
    parser.add_argument("--feature-eval-examples-per-pair", type=int, default=2)
    parser.add_argument("--random-replicates", type=int, default=3)
    parser.add_argument("--no-skip-completed", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_final_bridge_causal(
        args.run_dir,
        device=args.device,
        behavior_examples_per_pair=args.behavior_examples_per_pair,
        patch_examples_per_pair=args.patch_examples_per_pair,
        feature_fit_examples_per_pair=args.feature_fit_examples_per_pair,
        feature_eval_examples_per_pair=args.feature_eval_examples_per_pair,
        random_replicates=args.random_replicates,
        skip_completed=not args.no_skip_completed,
    )
    for name, frame in outputs.items():
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()

