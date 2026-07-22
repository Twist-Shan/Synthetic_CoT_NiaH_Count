from __future__ import annotations

import argparse

from .config import ALL_MODEL_VARIANTS, preset_config


def build_parser(version: str = "v20") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run {version}: query-first RoPE synthetic counting (counts 1..30)"
    )
    parser.add_argument("--preset", choices=("debug", "main"), default="debug")
    parser.add_argument(
        "--stage",
        default="all",
        help="all or comma-separated prepare,train,phase,causal,attention,state,plots",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--precision", choices=("float32", "bf16"), default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--recovery-every", type=int, default=None)
    parser.add_argument("--snapshot-shard-every", type=int, default=None)
    parser.add_argument("--snapshot-dtype", choices=("float16", "bfloat16", "float32"), default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--ar-eval-every", type=int, default=None)
    parser.add_argument("--max-steps-for-language-pred", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--eval-examples-per-count", type=int, default=None)
    parser.add_argument("--final-examples-per-count", type=int, default=None)
    parser.add_argument("--ar-examples-per-count", type=int, default=None)
    parser.add_argument("--permutation-examples-per-count", type=int, default=None)
    parser.add_argument("--phase-examples-per-count", type=int, default=None)
    parser.add_argument("--phase-head-selection-examples-per-count", type=int, default=None)
    parser.add_argument("--final-count-loss-weight", type=float, default=None)
    parser.add_argument("--cot-trace-loss-weight", type=float, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--count-max-threshold", type=int, default=None)
    parser.add_argument("--task-occurrence-ratio", type=float, default=None)
    parser.add_argument("--needle-pool-size", type=int, default=None)
    parser.add_argument("--needle-pool-frequency-threshold", type=float, default=None)
    parser.add_argument("--needle-pool-frequency-bins", type=int, default=None)
    parser.add_argument("--needle-pool-seed", type=int, default=None)
    parser.add_argument("--candidate-filter-max-attempts", type=int, default=None)
    parser.add_argument(
        "--model-variant",
        action="append",
        choices=ALL_MODEL_VARIANTS,
        default=None,
    )
    parser.add_argument(
        "--out-root", default=f"runs/synthetic_counting_{version.replace('-', '_')}"
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-sync-root", default=None)
    parser.add_argument("--skip-completed", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, version: str = "v20") -> None:
    args = build_parser(version).parse_args(argv)
    names = (
        "device",
        "seed",
        "train_steps",
        "batch_size",
        "precision",
        "checkpoint_every",
        "recovery_every",
        "snapshot_shard_every",
        "snapshot_dtype",
        "eval_every",
        "ar_eval_every",
        "max_steps_for_language_pred",
        "weight_decay",
        "eval_examples_per_count",
        "final_examples_per_count",
        "ar_examples_per_count",
        "permutation_examples_per_count",
        "phase_examples_per_count",
        "phase_head_selection_examples_per_count",
        "final_count_loss_weight",
        "cot_trace_loss_weight",
        "seq_len",
        "count_max_threshold",
        "task_occurrence_ratio",
        "needle_pool_size",
        "needle_pool_frequency_threshold",
        "needle_pool_frequency_bins",
        "needle_pool_seed",
        "candidate_filter_max_attempts",
    )
    overrides = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    overrides.update(
        version=version,
        count_tokenization="atomic" if version == "v20" else "digitwise",
    )
    if args.model_variant is not None:
        overrides["enabled_model_variants"] = tuple(args.model_variant)
    cfg = preset_config(args.preset, **overrides)
    from .pipeline import run_v20_pipeline

    run_v20_pipeline(
        cfg,
        stage=args.stage,
        out_root=args.out_root,
        run_name=args.run_name,
        checkpoint_sync_root=args.checkpoint_sync_root,
        skip_completed=args.skip_completed,
    )


__all__ = ["build_parser", "main"]
