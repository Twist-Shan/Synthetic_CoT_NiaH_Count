#!/usr/bin/env python3
"""Build checkpoint-wise PCA coordinates for the interactive v16.2 report view."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from synthetic_counting_v16_2.interactive_geometry import write_interactive_geometry_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=ROOT / "colab_results" / "v16_2_main_rope_seed1234",
    )
    args = parser.parse_args()
    print(write_interactive_geometry_table(args.run_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
