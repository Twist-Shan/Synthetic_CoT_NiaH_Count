from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_counting_v20.data import load_corpus_text
from synthetic_counting_v20.pipeline import load_prepared_v20_data
from synthetic_counting_v20.training import (
    autoregressive_task_evaluation,
    load_v20_checkpoint_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved synthetic-counting checkpoint on the fixed test task suite"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--mode", choices=("nonthinking", "thinking"), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg, vocab, _, _, model = load_v20_checkpoint_model(
        args.run_dir,
        "rope",
        args.mode,
        step=args.step,
        device=args.device,
    )
    text = load_corpus_text()
    _, _, _, test_suites = load_prepared_v20_data(cfg, vocab, text, args.run_dir)
    frame = autoregressive_task_evaluation(
        model,
        cfg,
        vocab,
        test_suites["task"],
        position_encoding="rope",
        mode=args.mode,
        step=args.step,
    )
    by_count = {
        str(int(count)): float(group["ar_accuracy"].mean())
        for count, group in frame.groupby("count", sort=True)
    }
    result: dict[str, object] = {
        "version": cfg.version,
        "mode": args.mode,
        "step": args.step,
        "evaluation_split": "fixed_test_task",
        "num_examples": int(len(frame)),
        "examples_per_count": cfg.final_examples_per_count,
        "accuracy": float(frame["ar_accuracy"].mean()),
        "per_count_accuracy": by_count,
    }
    if args.mode == "thinking":
        result.update(
            {
                "trace_exact": float(frame["trace_exact"].mean()),
                "ordered_marker_accuracy": float(
                    frame["trace_ordered_marker_accuracy"].mean()
                ),
                "marker_count_accuracy": float(
                    frame["trace_marker_count_accuracy"].mean()
                ),
                "mean_marker_count_delta": float(
                    (frame["trace_generated_marker_count"] - frame["count"]).mean()
                ),
            }
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
