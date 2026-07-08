from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .data import BaseExample, balanced_examples, count_bin, nonthinking_query, thinking_query
from .generation import greedy_generate_one, next_token_logits
from .model import make_model
from .train import checkpoint_steps, load_checkpoint
from .vocab import MARKER_TOKENS, Vocab, token_to_count


@dataclass(frozen=True)
class ParsedThinkingGeneration:
    trace_tokens: list[str]
    trace_markers: list[str]
    final_count: int | None
    has_close: bool
    premature_close: bool
    missing_close: bool
    invalid_count: bool
    duplicate_rate: float
    reason: str


def _lcs_len(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            old = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def parse_thinking_generation(generated_tokens: list[str], gold_markers: list[str] | None = None) -> ParsedThinkingGeneration:
    gold_markers = gold_markers or []
    if "</Think>" in generated_tokens:
        close_idx = generated_tokens.index("</Think>")
        trace_tokens = generated_tokens[:close_idx]
        rest = generated_tokens[close_idx + 1 :]
        has_close = True
        missing_close = False
    else:
        trace_tokens = list(generated_tokens)
        rest = []
        has_close = False
        missing_close = True
    trace_markers = [token for token in trace_tokens if token in MARKER_TOKENS]
    final_count = token_to_count(rest[0]) if rest else None
    invalid_count = final_count is None
    premature_close = has_close and len(trace_markers) < len(gold_markers)
    duplicate_rate = max(0, len(trace_markers) - len(set(trace_markers))) / max(1, len(trace_markers))
    if missing_close:
        reason = "missing_close"
    elif invalid_count:
        reason = "invalid_count"
    elif premature_close:
        reason = "premature_close"
    else:
        reason = "ok"
    return ParsedThinkingGeneration(
        trace_tokens=trace_tokens,
        trace_markers=trace_markers,
        final_count=final_count,
        has_close=has_close,
        premature_close=premature_close,
        missing_close=missing_close,
        invalid_count=invalid_count,
        duplicate_rate=float(duplicate_rate),
        reason=reason,
    )


def trace_metric_dict(parsed: ParsedThinkingGeneration, gold_markers: list[str]) -> dict[str, float]:
    lcs = _lcs_len(parsed.trace_markers, gold_markers)
    return {
        "trace_exact": float(parsed.trace_markers == gold_markers),
        "trace_marker_precision": float(lcs / max(1, len(parsed.trace_markers))),
        "trace_marker_recall": float(lcs / max(1, len(gold_markers))),
        "trace_duplicate_rate": float(parsed.duplicate_rate),
        "premature_close_rate": float(parsed.premature_close),
        "missing_close_rate": float(parsed.missing_close),
        "invalid_count_rate": float(parsed.invalid_count),
    }


def examples_for_eval(cfg: dict[str, Any], seed_offset: int = 5000) -> list[BaseExample]:
    train = cfg["train"]
    return balanced_examples(
        int(train["seq_len"]),
        int(train["eval_examples_per_count"]),
        int(train["seed"]) + seed_offset,
        int(train["count_min"]),
        int(train["count_max"]),
    )


@torch.no_grad()
def evaluate_nonthinking(
    model,
    examples: list[BaseExample],
    vocab: Vocab,
    device: str | torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    prefixes = [nonthinking_query(ex, vocab) for ex in examples]
    logits = next_token_logits(model, prefixes, vocab, device, batch_size=batch_size)
    count_ids = torch.tensor(vocab.count_ids, dtype=torch.long)
    restricted = logits.index_select(1, count_ids)
    preds = restricted.argmax(dim=-1).tolist()
    rows: list[dict[str, Any]] = []
    for ex, pred_offset in zip(examples, preds):
        pred = int(pred_offset) + 1
        rows.append(
            {
                "mode": "nonthinking",
                "count": ex.count,
                "count_bin": count_bin(ex.count),
                "pred_count": pred,
                "final_accuracy": float(pred == ex.count),
                "final_mae": abs(pred - ex.count),
                "undercount_rate": float(pred < ex.count),
                "overcount_rate": float(pred > ex.count),
                "trace_exact": math.nan,
                "trace_marker_precision": math.nan,
                "trace_marker_recall": math.nan,
                "trace_duplicate_rate": math.nan,
                "premature_close_rate": math.nan,
                "missing_close_rate": math.nan,
                "invalid_count_rate": 0.0,
                "generated": "",
            }
        )
    return rows


@torch.no_grad()
def evaluate_thinking(
    model,
    examples: list[BaseExample],
    vocab: Vocab,
    cfg: dict[str, Any],
    device: str | torch.device,
) -> list[dict[str, Any]]:
    max_new = 2 * int(cfg["train"]["count_max"]) + 6
    rows: list[dict[str, Any]] = []
    for ex in examples:
        gen_ids = greedy_generate_one(model, thinking_query(ex, vocab), vocab, device, max_new_tokens=max_new, continue_after_close=2)
        gen_tokens = vocab.decode(gen_ids)
        parsed = parse_thinking_generation(gen_tokens, ex.needle_markers)
        pred = parsed.final_count
        rows.append(
            {
                "mode": "thinking",
                "count": ex.count,
                "count_bin": count_bin(ex.count),
                "pred_count": pred if pred is not None else -1,
                "final_accuracy": float(pred == ex.count),
                "final_mae": abs(pred - ex.count) if pred is not None else math.nan,
                "undercount_rate": float(pred < ex.count) if pred is not None else math.nan,
                "overcount_rate": float(pred > ex.count) if pred is not None else math.nan,
                "generated": " ".join(gen_tokens),
                **trace_metric_dict(parsed, ex.needle_markers),
            }
        )
    return rows


@torch.no_grad()
def ambiguous_prefix_diagnostic(
    model,
    examples: list[BaseExample],
    vocab: Vocab,
    device: str | torch.device,
    batch_size: int,
) -> pd.DataFrame:
    import pandas as pd

    logits = next_token_logits(model, [thinking_query(ex, vocab) for ex in examples], vocab, device, batch_size=batch_size)
    probs = F.softmax(logits, dim=-1)
    marker_ids = torch.tensor(vocab.marker_ids, dtype=torch.long)
    rows: list[dict[str, Any]] = []
    for ex, prob in zip(examples, probs):
        argmax_id = int(prob.argmax().item())
        gold_first_id = vocab.token_to_id[ex.needle_markers[0]]
        rows.append(
            {
                "count": ex.count,
                "count_bin": count_bin(ex.count),
                "p_close_after_think": float(prob[vocab.think_close_id].item()),
                "p_any_marker_after_think": float(prob.index_select(0, marker_ids).sum().item()),
                "p_gold_first_marker_after_think": float(prob[gold_first_id].item()),
                "argmax_token_after_think": vocab.id_to_token[argmax_id],
                "argmax_is_close": float(argmax_id == vocab.think_close_id),
                "argmax_is_gold_first_marker": float(argmax_id == gold_first_id),
            }
        )
    return pd.DataFrame(rows)


def summarize_eval_rows(rows: pd.DataFrame, step: int) -> pd.DataFrame:
    import pandas as pd

    schema = [
        "step",
        "mode",
        "count",
        "count_bin",
        "n_examples",
        "final_accuracy",
        "final_mae",
        "undercount_rate",
        "overcount_rate",
        "trace_exact",
        "trace_marker_precision",
        "trace_marker_recall",
        "premature_close_rate",
        "missing_close_rate",
        "invalid_count_rate",
    ]
    if rows.empty:
        return pd.DataFrame(columns=schema)
    metrics = [col for col in schema if col not in {"step", "mode", "count", "count_bin", "n_examples"}]
    grouped = rows.groupby(["mode", "count", "count_bin"], dropna=False)
    out = grouped[metrics].mean(numeric_only=True).reset_index()
    out.insert(0, "step", int(step))
    counts = grouped.size().reset_index(name="n_examples")
    out = out.merge(counts, on=["mode", "count", "count_bin"], how="left")
    return out[schema]


def summarize_ambiguous_rows(rows: pd.DataFrame, step: int) -> pd.DataFrame:
    import pandas as pd

    schema = [
        "step",
        "count",
        "count_bin",
        "n_examples",
        "p_close_after_think",
        "p_any_marker_after_think",
        "p_gold_first_marker_after_think",
        "argmax_token_after_think",
        "argmax_is_close",
        "argmax_is_gold_first_marker",
    ]
    if rows.empty:
        return pd.DataFrame(columns=schema)
    numeric = [
        "p_close_after_think",
        "p_any_marker_after_think",
        "p_gold_first_marker_after_think",
        "argmax_is_close",
        "argmax_is_gold_first_marker",
    ]
    out = rows.groupby(["count", "count_bin"], dropna=False)[numeric].mean().reset_index()
    argmax = (
        rows.groupby(["count", "count_bin"], dropna=False)["argmax_token_after_think"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )
    counts = rows.groupby(["count", "count_bin"], dropna=False).size().reset_index(name="n_examples")
    out = out.merge(argmax, on=["count", "count_bin"], how="left").merge(counts, on=["count", "count_bin"], how="left")
    out.insert(0, "step", int(step))
    return out[schema]


def run_evaluation(cfg: dict[str, Any], vocab: Vocab, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import pandas as pd

    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    examples = examples_for_eval(cfg)
    batch_size = min(256, int(cfg["train"]["batch_size"]))
    eval_parts: list[pd.DataFrame] = []
    example_parts: list[pd.DataFrame] = []
    ambiguous_parts: list[pd.DataFrame] = []
    steps = checkpoint_steps(run_dir, int(cfg["train"]["train_steps"]))
    if not steps:
        raise FileNotFoundError("No checkpoints found. Run --stage train first.")
    for step, ckpt_path in steps:
        print(f"[eval] step={step}", flush=True)
        model = make_model(cfg["model"], cfg["device"])
        load_checkpoint(model, ckpt_path, cfg["device"])
        rows = evaluate_nonthinking(model, examples, vocab, cfg["device"], batch_size)
        rows.extend(evaluate_thinking(model, examples, vocab, cfg, cfg["device"]))
        per_df = pd.DataFrame(rows)
        per_df.insert(0, "step", int(step))
        example_parts.append(per_df)
        eval_parts.append(summarize_eval_rows(per_df, step))
        ambiguous_parts.append(summarize_ambiguous_rows(ambiguous_prefix_diagnostic(model, examples, vocab, cfg["device"], batch_size), step))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    eval_by_step = pd.concat(eval_parts, ignore_index=True)
    eval_examples = pd.concat(example_parts, ignore_index=True)
    ambiguous = pd.concat(ambiguous_parts, ignore_index=True)
    eval_by_step.to_csv(tables / "eval_by_step.csv", index=False)
    eval_examples.to_csv(tables / "eval_examples.csv", index=False)
    ambiguous.to_csv(tables / "ambiguous_prefix.csv", index=False)
    return eval_by_step, ambiguous, eval_examples
