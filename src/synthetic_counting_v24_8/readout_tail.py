from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from synthetic_counting_v20.data import (
    V20Example,
    V20Rendered,
    collate_v20,
    load_corpus_text,
)
from synthetic_counting_v20.training import (
    _training_batch,
    atomic_csv,
    autoregressive_task_evaluation,
    load_v20_checkpoint_model,
    teacher_forced_task_evaluation,
)


def _load_prepared_v20_data(*args: Any, **kwargs: Any) -> Any:
    # Importing the complete pipeline also imports plotting dependencies. Keep
    # that optional dependency boundary out of lightweight gate/unit tests.
    from synthetic_counting_v20.pipeline import load_prepared_v20_data

    return load_prepared_v20_data(*args, **kwargs)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    learning_rate: float
    steps: int
    warmup_steps: int

    def validate(self) -> None:
        if not self.name:
            raise ValueError("candidate name must be nonempty")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("candidate learning rate must be finite and positive")
        if self.steps <= 0:
            raise ValueError("candidate steps must be positive")
        if self.warmup_steps < 0 or self.warmup_steps >= self.steps:
            raise ValueError("warmup steps must lie in [0, steps)")


@dataclass(frozen=True)
class GateSummary:
    overall_accuracy: float
    minimum_count_accuracy: float
    count_accuracy_spread: float
    trace_exact_accuracy: float
    answer_rate: float
    success_criteria_met: bool

    @property
    def selection_score(self) -> tuple[float, ...]:
        # Minimum-class performance is primary because it rules out the
        # v24.7 failure mode where whole semantic counts are never emitted.
        return (
            float(self.success_criteria_met),
            self.minimum_count_accuracy,
            self.overall_accuracy,
            -self.count_accuracy_spread,
            self.trace_exact_accuracy,
            self.answer_rate,
        )


def default_candidate_specs() -> tuple[CandidateSpec, ...]:
    """Conservative-to-aggressive native-head calibration schedule."""

    return (
        CandidateSpec("number_rows_lr3e-4", 3e-4, 1_200, 50),
        CandidateSpec("number_rows_lr1e-3", 1e-3, 1_200, 50),
        CandidateSpec("number_rows_lr3e-3", 3e-3, 1_200, 50),
    )


def summarize_gate(frame: pd.DataFrame, *, mode: str) -> tuple[GateSummary, pd.DataFrame]:
    required = {"count", "ar_accuracy", "ar_answered"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"autoregressive frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("cannot summarize an empty autoregressive frame")
    by_count = (
        frame.groupby("count", as_index=False)
        .agg(
            examples=("ar_accuracy", "size"),
            ar_final_accuracy=("ar_accuracy", "mean"),
            ar_answer_rate=("ar_answered", "mean"),
            trace_exact=("trace_exact", "mean")
            if "trace_exact" in frame.columns
            else ("ar_accuracy", lambda values: float("nan")),
        )
        .sort_values("count")
        .reset_index(drop=True)
    )
    minimum = float(by_count["ar_final_accuracy"].min())
    spread = float(
        by_count["ar_final_accuracy"].max()
        - by_count["ar_final_accuracy"].min()
    )
    trace_exact = (
        float(frame["trace_exact"].mean())
        if mode == "thinking" and "trace_exact" in frame.columns
        else 1.0
    )
    overall = float(frame["ar_accuracy"].mean())
    answer_rate = float(frame["ar_answered"].mean())
    summary = GateSummary(
        overall_accuracy=overall,
        minimum_count_accuracy=minimum,
        count_accuracy_spread=spread,
        trace_exact_accuracy=trace_exact,
        answer_rate=answer_rate,
        success_criteria_met=(
            overall >= 0.90
            and minimum >= 0.85
            and spread <= 0.10
            and trace_exact >= 0.90
        ),
    )
    return summary, by_count


def _learning_rate(spec: CandidateSpec, step: int) -> float:
    if step <= spec.warmup_steps:
        return spec.learning_rate * step / max(1, spec.warmup_steps)
    progress = (step - spec.warmup_steps) / max(1, spec.steps - spec.warmup_steps)
    return spec.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _answer_logits_and_targets(
    logits: torch.Tensor,
    rendered: list[V20Rendered],
    number_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    targets: list[int] = []
    for row, item in enumerate(rendered):
        if item.spans is None or item.count is None:
            raise ValueError("readout calibration requires counting-task rows")
        if len(item.spans.count_positions) != 1:
            raise ValueError("v24.8 requires one atomic final-count token")
        rows.append(logits[row, item.spans.count_pos - 1, number_ids])
        targets.append(int(item.count) - 1)
    return torch.stack(rows), torch.tensor(targets, dtype=torch.long, device=logits.device)


def _validation_examples(
    examples: list[V20Example],
    *,
    per_count: int,
    count_max: int,
) -> list[V20Example]:
    selected: list[V20Example] = []
    for count in range(1, count_max + 1):
        values = [item for item in examples if int(item.count or 0) == count]
        if len(values) < per_count:
            raise ValueError(
                f"validation suite has {len(values)} examples for count {count}; "
                f"need {per_count}"
            )
        selected.extend(values[:per_count])
    return selected


def _evaluate(
    model: torch.nn.Module,
    cfg: Any,
    vocab: Any,
    examples: list[V20Example],
    *,
    mode: str,
    step: int,
) -> tuple[GateSummary, pd.DataFrame, pd.DataFrame]:
    frame = autoregressive_task_evaluation(
        model,
        cfg,
        vocab,
        examples,
        position_encoding="rope",
        mode=mode,
        step=step,
    )
    summary, by_count = summarize_gate(frame, mode=mode)
    return summary, by_count, frame


def _summary_row(
    summary: GateSummary,
    *,
    mode: str,
    candidate: str,
    step: int,
    split: str,
) -> dict[str, Any]:
    return {
        "experiment": "v24.8",
        "source_version": "v24.7",
        "mode": mode,
        "candidate": candidate,
        "step": int(step),
        "evaluation_split": split,
        **asdict(summary),
    }


def _save_model(
    path: Path,
    model: torch.nn.Module,
    *,
    source_run: Path,
    source_config: Any,
    mode: str,
    candidate: CandidateSpec,
    step: int,
    validation_summary: GateSummary,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "experiment": "v24.8",
            "source_version": "v24.7",
            "source_run": str(source_run),
            "mode": mode,
            "position_encoding": "rope",
            "tail_candidate": asdict(candidate),
            "tail_step": int(step),
            "trainable_parameters": "existing native lm_head number-token rows only",
            "validation_summary": asdict(validation_summary),
            "source_config": asdict(source_config),
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        },
        temporary,
    )
    temporary.replace(path)


def _run_candidate(
    source_run: Path,
    output_dir: Path,
    *,
    mode: str,
    spec: CandidateSpec,
    device: str,
    batch_size: int,
    eval_every: int,
    validation_per_count: int,
    seed: int,
) -> dict[str, Any]:
    spec.validate()
    cfg, vocab, pool, split, model = load_v20_checkpoint_model(
        source_run,
        "rope",
        mode,
        label="final",
        device=device,
    )
    if cfg.version != "v24.7":
        raise ValueError(f"v24.8 requires a v24.7 source checkpoint, got {cfg.version}")
    if cfg.trace_format != "separator" or cfg.count_tokenization != "atomic":
        raise ValueError("v24.8 requires the unchanged atomic separator-trace setting")
    if model.lm_head is None:
        raise ValueError("v24.8 requires v24.7's untied native LM head")
    text = load_corpus_text()
    _, _, curve_suites, _ = _load_prepared_v20_data(cfg, vocab, text, source_run)
    validation = _validation_examples(
        curve_suites["heldout"]["task"],
        per_count=validation_per_count,
        count_max=cfg.count_max_threshold,
    )
    tail_cfg = replace(
        cfg,
        batch_size=batch_size,
        train_steps=spec.steps,
        warmup_steps=spec.warmup_steps,
        max_steps_for_language_pred=0,
        answer_query_contrastive_weight=0.0,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.lm_head.weight.requires_grad_(True)
    trainable = [model.lm_head.weight]
    optimizer = AdamW(trainable, lr=0.0, betas=(0.9, 0.999), weight_decay=0.0)
    number_ids = torch.tensor(vocab.number_ids, dtype=torch.long, device=device)
    rng = random.Random(seed)
    history_rows: list[dict[str, Any]] = []
    by_count_rows: list[pd.DataFrame] = []

    initial_summary, initial_by_count, _ = _evaluate(
        model, cfg, vocab, validation, mode=mode, step=0
    )
    history_rows.append(
        _summary_row(
            initial_summary,
            mode=mode,
            candidate=spec.name,
            step=0,
            split="validation",
        )
    )
    initial_by_count = initial_by_count.assign(
        mode=mode, candidate=spec.name, step=0, evaluation_split="validation"
    )
    by_count_rows.append(initial_by_count)
    best_summary = initial_summary
    best_step = 0
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    started = time.perf_counter()
    for step in range(1, spec.steps + 1):
        model.train()
        _, rendered = _training_batch(
            tail_cfg,
            vocab,
            text,
            split,
            pool,
            mode,
            rng,
            require_task=True,
        )
        ids, _, mask = collate_v20(rendered, vocab, device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        answer_logits, targets = _answer_logits_and_targets(logits, rendered, number_ids)
        loss = F.cross_entropy(answer_logits.float(), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        rate = _learning_rate(spec, step)
        optimizer.param_groups[0]["lr"] = rate
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == spec.steps:
            print(
                f"[tail] mode={mode} candidate={spec.name} step={step}/{spec.steps} "
                f"loss={float(loss.detach().cpu()):.6f} lr={rate:.3e} "
                f"grad_norm={gradient_norm:.3f}",
                flush=True,
            )
        if step % eval_every == 0 or step == spec.steps:
            summary, by_count, _ = _evaluate(
                model, cfg, vocab, validation, mode=mode, step=step
            )
            history_rows.append(
                _summary_row(
                    summary,
                    mode=mode,
                    candidate=spec.name,
                    step=step,
                    split="validation",
                )
            )
            by_count_rows.append(
                by_count.assign(
                    mode=mode,
                    candidate=spec.name,
                    step=step,
                    evaluation_split="validation",
                )
            )
            print(
                f"[validation] mode={mode} candidate={spec.name} step={step} "
                f"overall={summary.overall_accuracy:.3f} "
                f"min_count={summary.minimum_count_accuracy:.3f} "
                f"spread={summary.count_accuracy_spread:.3f} "
                f"trace_exact={summary.trace_exact_accuracy:.3f} "
                f"gate={summary.success_criteria_met}",
                flush=True,
            )
            if summary.selection_score > best_summary.selection_score:
                best_summary = summary
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                _save_model(
                    output_dir / "candidates" / spec.name / mode / "best_checkpoint.pt",
                    model,
                    source_run=source_run,
                    source_config=cfg,
                    mode=mode,
                    candidate=spec,
                    step=step,
                    validation_summary=summary,
                )
    model.load_state_dict(best_state)
    candidate_dir = output_dir / "candidates" / spec.name / mode
    candidate_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(pd.DataFrame(history_rows), candidate_dir / "validation_history.csv")
    atomic_csv(pd.concat(by_count_rows, ignore_index=True), candidate_dir / "validation_by_count.csv")
    _save_model(
        candidate_dir / "selected_checkpoint.pt",
        model,
        source_run=source_run,
        source_config=cfg,
        mode=mode,
        candidate=spec,
        step=best_step,
        validation_summary=best_summary,
    )
    return {
        "model": model,
        "cfg": cfg,
        "vocab": vocab,
        "best_summary": best_summary,
        "best_step": best_step,
        "candidate": spec,
        "duration_seconds": time.perf_counter() - started,
    }


def _final_test(
    result: dict[str, Any],
    source_run: Path,
    output_dir: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    model = result["model"]
    cfg = result["cfg"]
    vocab = result["vocab"]
    text = load_corpus_text()
    _, _, _, test_suites = _load_prepared_v20_data(cfg, vocab, text, source_run)
    summary, by_count, detail = _evaluate(
        model,
        cfg,
        vocab,
        test_suites["task"],
        mode=mode,
        step=int(result["best_step"]),
    )
    teacher_forced = teacher_forced_task_evaluation(
        model,
        cfg,
        vocab,
        test_suites["task"],
        position_encoding="rope",
        mode=mode,
        step=int(result["best_step"]),
    )
    final_dir = output_dir / "final" / mode
    final_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(
        pd.DataFrame(
            [
                _summary_row(
                    summary,
                    mode=mode,
                    candidate=result["candidate"].name,
                    step=int(result["best_step"]),
                    split="test",
                )
            ]
        ),
        final_dir / "final_autoregressive_summary.csv",
    )
    atomic_csv(by_count.assign(mode=mode), final_dir / "final_autoregressive_by_count.csv")
    bounded = detail.drop(columns=["generated_tokens"], errors="ignore")
    atomic_csv(bounded, final_dir / "final_autoregressive_detail.csv")
    atomic_csv(teacher_forced, final_dir / "final_teacher_forced_detail.csv")
    _save_model(
        final_dir / "checkpoint.pt",
        model,
        source_run=source_run,
        source_config=cfg,
        mode=mode,
        candidate=result["candidate"],
        step=int(result["best_step"]),
        validation_summary=result["best_summary"],
    )
    return {
        "mode": mode,
        "candidate": result["candidate"].name,
        "selected_step": int(result["best_step"]),
        "validation": asdict(result["best_summary"]),
        "test": asdict(summary),
    }


def run_readout_tail(
    source_run: str | Path,
    output_dir: str | Path,
    *,
    device: str,
    batch_size: int = 128,
    eval_every: int = 100,
    validation_per_count: int = 10,
    seed: int = 2478,
    candidates: tuple[CandidateSpec, ...] | None = None,
) -> Path:
    source_run = Path(source_run).resolve()
    output_dir = Path(output_dir).resolve()
    if not source_run.exists():
        raise FileNotFoundError(f"source v24.7 run does not exist: {source_run}")
    if batch_size <= 0 or eval_every <= 0 or validation_per_count <= 0:
        raise ValueError("batch size, eval interval, and validation count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_specs = candidates or default_candidate_specs()
    for spec in candidate_specs:
        spec.validate()
    manifest: dict[str, Any] = {
        "experiment": "v24.8",
        "source_version": "v24.7",
        "source_run": str(source_run),
        "trace_change": False,
        "inference_change": False,
        "auxiliary_decoder": False,
        "trainable_parameters": "existing native lm_head number-token rows only",
        "selection_split": "validation",
        "test_used_for_selection": False,
        "batch_size": batch_size,
        "eval_every": eval_every,
        "validation_examples_per_count": validation_per_count,
        "seed": seed,
        "candidates": [asdict(spec) for spec in candidate_specs],
        "status": "running",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    thinking_results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for index, spec in enumerate(candidate_specs):
        result = _run_candidate(
            source_run,
            output_dir,
            mode="thinking",
            spec=spec,
            device=device,
            batch_size=batch_size,
            eval_every=eval_every,
            validation_per_count=validation_per_count,
            seed=seed,
        )
        thinking_results.append(result)
        if selected is None or (
            result["best_summary"].selection_score
            > selected["best_summary"].selection_score
        ):
            selected = result
        if result["best_summary"].success_criteria_met:
            break
    assert selected is not None

    # Apply the Thinking-selected training setting to the paired Non-thinking
    # checkpoint. The test split remains untouched until both modes are fixed.
    nonthinking = _run_candidate(
        source_run,
        output_dir,
        mode="nonthinking",
        spec=selected["candidate"],
        device=device,
        batch_size=batch_size,
        eval_every=eval_every,
        validation_per_count=validation_per_count,
        seed=seed,
    )
    final_results = [
        _final_test(selected, source_run, output_dir, mode="thinking"),
        _final_test(nonthinking, source_run, output_dir, mode="nonthinking"),
    ]
    manifest.update(
        status="complete",
        selected_candidate=selected["candidate"].name,
        final_results=final_results,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    atomic_csv(
        pd.DataFrame(
            [
                {
                    "mode": item["mode"],
                    "candidate": item["candidate"],
                    "selected_step": item["selected_step"],
                    **{f"validation_{key}": value for key, value in item["validation"].items()},
                    **{f"test_{key}": value for key, value in item["test"].items()},
                }
                for item in final_results
            ]
        ),
        output_dir / "final_summary.csv",
    )
    print(json.dumps(final_results, indent=2, sort_keys=True), flush=True)
    print(f"V24_8_OUTPUT_DIR={output_dir}", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate v24.7's native atomic-number LM-head rows without changing its trace"
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-per-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2478)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_readout_tail(
        args.source_run,
        args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        validation_per_count=args.validation_per_count,
        seed=args.seed,
    )


__all__ = [
    "CandidateSpec",
    "GateSummary",
    "default_candidate_specs",
    "run_readout_tail",
    "summarize_gate",
]
