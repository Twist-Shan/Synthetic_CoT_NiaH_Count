from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


if "pyarrow" not in sys.modules:
    pyarrow_stub = types.ModuleType("pyarrow")
    pyarrow_stub.__version__ = "0.0.0"
    pyarrow_stub.Array = type("Array", (), {})
    pyarrow_stub.ChunkedArray = type("ChunkedArray", (), {})
    sys.modules["pyarrow"] = pyarrow_stub
for optional_module in ("numexpr", "bottleneck"):
    sys.modules.setdefault(optional_module, None)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_counting_v10.report_stratified import build_stratified_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute v10 head ablations and stratify every causal table by count range"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--examples-per-count", type=int, default=8)
    parser.add_argument("--random-replicates", type=int, default=8)
    parser.add_argument("--reuse-ablation", action="store_true")
    parser.add_argument(
        "--reuse-position-local-ablation",
        action="store_true",
        help="Reuse an existing position_local_ablation_by_bin.csv instead of rerunning it",
    )
    args = parser.parse_args()
    outputs = build_stratified_tables(
        args.run_dir,
        device=args.device,
        rerun_ablation=not args.reuse_ablation,
        rerun_position_local_ablation=not args.reuse_position_local_ablation,
        ablation_examples_per_count=args.examples_per_count,
        random_replicates=args.random_replicates,
    )
    out_dir = args.run_dir / "analysis" / "report_stratified"
    print(f"V10_STRATIFIED_ANALYSIS={out_dir}")
    for name, frame in outputs.items():
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
