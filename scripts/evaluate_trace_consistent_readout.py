from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synthetic_counting_v20.trace_readout import write_trace_readout_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a gold-free trace-to-answer readout on a completed run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--count-max-threshold", type=int, required=True)
    args = parser.parse_args()
    manifest = write_trace_readout_analysis(
        args.run_dir,
        count_max_threshold=args.count_max_threshold,
    )
    if manifest is None:
        raise RuntimeError("completed Thinking final-autoregressive detail is required")
    print(pd.read_csv(args.run_dir / "tables" / "trace_readout_summary.csv").to_string(index=False))
    print(pd.read_csv(args.run_dir / "tables" / "trace_readout_by_count.csv").to_string(index=False))


if __name__ == "__main__":
    main()
