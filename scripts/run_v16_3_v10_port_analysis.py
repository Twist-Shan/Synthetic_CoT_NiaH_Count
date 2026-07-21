#!/usr/bin/env python3
"""Run the complete v10-style representation/causal suite on v16.3 RoPE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from synthetic_counting_v16_3.v10_port_analysis import PortOptions, run_v10_port_analysis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=ROOT / "colab_results" / "v16_3_main_rope_seed1234",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--examples-per-count", type=int, default=4)
    parser.add_argument("--centroid-train-per-count", type=int, default=10)
    parser.add_argument("--retrieval-selection-examples", type=int, default=10)
    parser.add_argument("--retrieval-reporting-examples", type=int, default=24)
    parser.add_argument("--random-paths", type=int, default=6)
    parser.add_argument("--seed", type=int, default=6162)
    args = parser.parse_args()
    options = PortOptions(
        examples_per_count=args.examples_per_count,
        centroid_train_per_count=args.centroid_train_per_count,
        retrieval_selection_examples=args.retrieval_selection_examples,
        retrieval_reporting_examples=args.retrieval_reporting_examples,
        random_paths=args.random_paths,
        seed=args.seed,
    )
    output = run_v10_port_analysis(args.run_dir, device=args.device, options=options)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

