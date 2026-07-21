from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .analysis import (
    apply_ridge,
    collect_v16_2_attention,
    collect_v16_2_states,
    fit_ridge,
    nearest_centroid,
)
from .config import V16_2Config, config_from_dict
from .data import (
    V16_2Example,
    V16_2Vocab,
    load_corpus_split,
    load_corpus_text,
    load_suite_manifests,
    render_v16_2,
    render_v16_2_shortened_trace,
)
from .needle_pool import load_needle_pool
from .timing import record_cached_event, timed_event
from .training import (
    atomic_csv,
    autoregressive_task_evaluation,
    checkpoint_steps,
    load_v16_2_checkpoint_model,
)


@dataclass(frozen=True)
class DynamicsOptions:
    attention_examples_per_count: int = 20
    ar_examples_per_count: int = 10
    state_train_examples_per_count: int = 40
    state_eval_examples_per_count: int = 15
    run_attention: bool = True
    run_states: bool = True
    run_generated: bool = True
    run_counterfactual: bool = True
    run_similarity: bool = True
    force: bool = False

    def validate(self) -> None:
        for name in (
            "attention_examples_per_count",
            "ar_examples_per_count",
            "state_train_examples_per_count",
            "state_eval_examples_per_count",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.attention_examples_per_count < 2:
            raise ValueError("attention_examples_per_count must allow selection/reporting splits")
        if not (self.run_attention or self.run_states):
            raise ValueError("enable at least one checkpoint-dynamics metric family")


PART_TABLES = (
    "attention_detail",
    "attention_summary",
    "attention_by_count",
    "attention_by_k",
    "autoregressive",
    "attention_behavior_link",
    "state_probe_summary",
    "state_by_count",
    "state_geometry",
    "state_cross_site",
    "state_counterfactual_trace",
)


def _json_fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _balanced_subset(
    examples: list[V16_2Example], per_count: int, count_max: int
) -> list[V16_2Example]:
    result: list[V16_2Example] = []
    for count in range(1, count_max + 1):
        available = [item for item in examples if item.count == count]
        if not available:
            raise RuntimeError(f"diagnostic suite has no examples for count {count}")
        result.extend(available[: min(per_count, len(available))])
    return result


def _attention_splits(
    examples: list[V16_2Example], per_count: int, count_max: int
) -> tuple[list[V16_2Example], dict[str, str]]:
    selected: list[V16_2Example] = []
    split_by_prompt: dict[str, str] = {}
    selection_size = per_count // 2
    reporting_size = per_count - selection_size
    for count in range(1, count_max + 1):
        available = [item for item in examples if item.count == count]
        requested = min(per_count, len(available))
        if requested < 2:
            raise RuntimeError(f"need at least two attention examples for count {count}")
        local_selection = min(selection_size, requested // 2)
        local_reporting = min(reporting_size, requested - local_selection)
        values = available[: local_selection + local_reporting]
        for index, item in enumerate(values):
            split_by_prompt[item.prompt_sha256] = (
                "head_selection" if index < local_selection else "heldout_reporting"
            )
        selected.extend(values)
    return selected, split_by_prompt


def _r2(y_true: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.square(y_true - y_true.mean()).sum())
    if denominator <= 1e-12:
        return math.nan
    return 1.0 - float(np.square(y_true - prediction).sum()) / denominator


def _geometry_metrics(vectors: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique = np.unique(labels)
    centroids = np.stack([vectors[labels == value].mean(axis=0) for value in unique])
    centered = centroids - centroids.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    variance = np.square(singular)
    ratios = variance / max(float(variance.sum()), 1e-12)
    pc1 = centered @ vh[0]
    label_values = unique.astype(float)
    correlation = math.nan
    if len(unique) > 1 and np.std(pc1) > 1e-12 and np.std(label_values) > 1e-12:
        correlation = float(np.corrcoef(pc1, label_values)[0, 1])
    deltas = np.diff(centroids, axis=0)
    mean_delta = deltas.mean(axis=0) if len(deltas) else np.zeros(centroids.shape[1])
    denominator = np.linalg.norm(deltas, axis=1) * max(float(np.linalg.norm(mean_delta)), 1e-12)
    consistency = np.divide(
        deltas @ mean_delta,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    adjacent = np.linalg.norm(deltas, axis=1) if len(deltas) else np.asarray([])
    projected = pc1 if not np.isfinite(correlation) or correlation >= 0 else -pc1
    violations = int(np.sum(np.diff(projected) <= 0)) if len(projected) > 1 else 0
    return {
        "pc1_label_r2": float(correlation**2) if np.isfinite(correlation) else math.nan,
        "pc1_adjacent_consistency": float(consistency.mean()) if len(consistency) else math.nan,
        "effective_dimension": float(variance.sum() ** 2 / max(float(np.square(variance).sum()), 1e-12)),
        "pc1_variance": float(ratios[0]) if len(ratios) else math.nan,
        "pc1_to_pc6_variance": float(ratios[:6].sum()) if len(ratios) else math.nan,
        "adjacent_distance_mean": float(adjacent.mean()) if len(adjacent) else math.nan,
        "adjacent_distance_std": float(adjacent.std()) if len(adjacent) else math.nan,
        "monotonic_order_violations": violations,
    }


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("linear CKA requires aligned matrices with the same shape")
    x = left - left.mean(axis=0, keepdims=True)
    y = right - right.mean(axis=0, keepdims=True)
    cross = float(np.square(x.T @ y).sum())
    denominator = math.sqrt(
        float(np.square(x.T @ x).sum()) * float(np.square(y.T @ y).sum())
    )
    return cross / denominator if denominator > 1e-12 else math.nan


def _probe_rows(
    train: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    heldout: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    *,
    position_encoding: str,
    mode: str,
    step: int,
    context: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    for site, layer in sorted(set(train) & set(heldout)):
        train_x, train_y = train[(site, layer)]
        test_x, test_y = heldout[(site, layer)]
        nearest = nearest_centroid(train_x, train_y, test_x)
        ridge = apply_ridge(fit_ridge(train_x, train_y.astype(float)), test_x)
        summary_rows.append(
            {
                "position_encoding": position_encoding,
                "mode": mode,
                "step": step,
                "context": context,
                "site": site,
                "layer": layer,
                "nearest_centroid_accuracy": float(np.mean(nearest == test_y)),
                "ridge_mae": float(np.mean(np.abs(ridge - test_y))),
                "ridge_r2": _r2(test_y.astype(float), ridge),
                "train_states": len(train_y),
                "heldout_states": len(test_y),
            }
        )
        for label in np.unique(test_y):
            selected = test_y == label
            count_rows.append(
                {
                    "position_encoding": position_encoding,
                    "mode": mode,
                    "step": step,
                    "context": context,
                    "site": site,
                    "layer": layer,
                    "label": int(label),
                    "examples": int(selected.sum()),
                    "nearest_centroid_accuracy": float(np.mean(nearest[selected] == test_y[selected])),
                    "ridge_prediction_mean": float(ridge[selected].mean()),
                    "ridge_bias": float(ridge[selected].mean() - label),
                    "ridge_mae": float(np.mean(np.abs(ridge[selected] - label))),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(count_rows)


def _geometry_rows(
    states: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    *,
    position_encoding: str,
    mode: str,
    step: int,
    context: str,
) -> pd.DataFrame:
    rows = []
    for (site, layer), (vectors, labels) in sorted(states.items()):
        rows.append(
            {
                "position_encoding": position_encoding,
                "mode": mode,
                "step": step,
                "context": context,
                "site": site,
                "layer": layer,
                **_geometry_metrics(vectors, labels),
            }
        )
    return pd.DataFrame(rows)


def _cross_site_rows(
    train: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    heldout: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    *,
    position_encoding: str,
    mode: str,
    step: int,
) -> pd.DataFrame:
    if mode != "thinking":
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for layer in range(0, 5):
        final_key = ("final_answer", layer)
        trace_key = ("trace_marker", layer)
        if final_key not in train or trace_key not in train or final_key not in heldout or trace_key not in heldout:
            continue
        final_train_x, final_train_y = train[final_key]
        trace_train_x, trace_train_y = train[trace_key]
        final_test_x, final_test_y = heldout[final_key]
        trace_test_x, trace_test_y = heldout[trace_key]
        final_fit = fit_ridge(final_train_x, final_train_y.astype(float))
        trace_fit = fit_ridge(trace_train_x, trace_train_y.astype(float))
        final_direction = final_fit[2][1:]
        trace_direction = trace_fit[2][1:]
        cosine_denominator = np.linalg.norm(final_direction) * np.linalg.norm(trace_direction)
        cosine = float(final_direction @ trace_direction / cosine_denominator) if cosine_denominator > 1e-12 else math.nan
        for direction, fitted, test_x, test_y in (
            ("trace_to_answer", trace_fit, final_test_x, final_test_y),
            ("answer_to_trace", final_fit, trace_test_x, trace_test_y),
        ):
            prediction = apply_ridge(fitted, test_x)
            slope, intercept = np.polyfit(test_y.astype(float), prediction, 1)
            rows.append(
                {
                    "position_encoding": position_encoding,
                    "mode": mode,
                    "step": step,
                    "layer": layer,
                    "direction": direction,
                    "mae": float(np.mean(np.abs(prediction - test_y))),
                    "r2": _r2(test_y.astype(float), prediction),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "direction_cosine": cosine,
                    "examples": len(test_y),
                }
            )
    return pd.DataFrame(rows)


def _collect_hidden_at_positions(
    model,
    cfg: V16_2Config,
    vocab: V16_2Vocab,
    token_lists: list[list[str]],
    positions: list[int],
    labels: list[int],
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    if not token_lists:
        return {}
    vectors: dict[int, list[np.ndarray]] = {}
    batch_size = min(cfg.analysis_batch_size, len(token_lists))
    for start in range(0, len(token_lists), batch_size):
        chunk = token_lists[start : start + batch_size]
        chunk_positions = positions[start : start + batch_size]
        encoded = [vocab.encode(tokens) for tokens in chunk]
        max_len = max(map(len, encoded))
        ids = torch.full((len(chunk), max_len), vocab.pad_id, dtype=torch.long, device=cfg.device)
        mask = torch.zeros_like(ids)
        for row, values in enumerate(encoded):
            ids[row, : len(values)] = torch.tensor(values, device=cfg.device)
            mask[row, : len(values)] = 1
        hidden_states = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states or ()
        for layer, hidden in enumerate(hidden_states):
            for row, position in enumerate(chunk_positions):
                vectors.setdefault(layer, []).append(
                    hidden[row, position].detach().float().cpu().numpy()
                )
    y = np.asarray(labels, dtype=int)
    return {
        ("final_answer", layer): (np.stack(values), y.copy())
        for layer, values in vectors.items()
    }


def _generated_final_states(
    model,
    cfg: V16_2Config,
    vocab: V16_2Vocab,
    ar: pd.DataFrame,
) -> tuple[dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    token_lists: list[list[str]] = []
    positions: list[int] = []
    labels: list[int] = []
    status_rows: list[dict[str, Any]] = []
    for _, row in ar.iterrows():
        tokens = str(row.generated_tokens).split()
        status = "valid" if "<Ans>" in tokens else "missing_ans"
        status_rows.append(
            {
                "prompt_sha256": row.prompt_sha256,
                "count": int(row["count"]),
                "generated_state_status": status,
            }
        )
        if status != "valid":
            continue
        ans_pos = tokens.index("<Ans>")
        token_lists.append(tokens[: ans_pos + 1])
        positions.append(ans_pos)
        labels.append(int(row["count"]))
    return (
        _collect_hidden_at_positions(model, cfg, vocab, token_lists, positions, labels),
        pd.DataFrame(status_rows),
    )


def _numeric_mean(
    frame: pd.DataFrame, group_columns: list[str], excluded: set[str] | None = None
) -> pd.DataFrame:
    excluded = excluded or set()
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in set(group_columns) | excluded
    ]
    if frame.empty or not numeric:
        return pd.DataFrame()
    return frame.groupby(group_columns, as_index=False, dropna=False)[numeric].mean()


def _save_states(path: Path, states: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]) -> None:
    payload: dict[str, np.ndarray] = {}
    for (site, layer), (vectors, labels) in states.items():
        payload[f"{site}__{layer}__x"] = vectors
        payload[f"{site}__{layer}__y"] = labels
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _load_states(path: Path) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return {}
    result: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    with np.load(path) as archive:
        for key in archive.files:
            if not key.endswith("__x"):
                continue
            site, layer, _ = key.rsplit("__", 2)
            result[(site, int(layer))] = (archive[key], archive[f"{site}__{layer}__y"])
    return result


@torch.no_grad()
def _counterfactual_rows(
    model,
    cfg: V16_2Config,
    vocab: V16_2Vocab,
    examples: list[V16_2Example],
    train_states: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    *,
    position_encoding: str,
    mode: str,
    step: int,
) -> pd.DataFrame:
    if mode != "thinking":
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for example in examples:
        if int(example.count or 0) < 2:
            continue
        normal = render_v16_2(example, vocab, "thinking")
        shortened = render_v16_2_shortened_trace(example, vocab)
        for condition, item in (("gold_trace", normal), ("remove_final_pair", shortened)):
            assert item.spans is not None and example.count is not None
            ids = torch.tensor([item.input_ids], device=cfg.device)
            output = model(input_ids=ids, output_hidden_states=True)
            answer_logits = output.logits[0, item.spans.ans_pos]
            probabilities = answer_logits.softmax(dim=-1)
            gold_index = int(example.count) - 1
            lower_index = int(example.count) - 2
            for layer, hidden in enumerate(output.hidden_states or ()):
                state = hidden[0, item.spans.ans_pos].detach().float().cpu().numpy()[None]
                key = ("final_answer", layer)
                prediction = math.nan
                if key in train_states:
                    train_x, train_y = train_states[key]
                    prediction = float(apply_ridge(fit_ridge(train_x, train_y.astype(float)), state)[0])
                rows.append(
                    {
                        "position_encoding": position_encoding,
                        "mode": mode,
                        "step": step,
                        "condition": condition,
                        "prompt_sha256": example.prompt_sha256,
                        "count": int(example.count),
                        "layer": layer,
                        "gold_probability": float(probabilities[vocab.number_ids[gold_index]]),
                        "count_minus_one_probability": float(probabilities[vocab.number_ids[lower_index]]),
                        "gold_logit_margin_vs_count_minus_one": float(
                            answer_logits[vocab.number_ids[gold_index]]
                            - answer_logits[vocab.number_ids[lower_index]]
                        ),
                        "ridge_count_prediction": prediction,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    baseline = frame[frame.condition == "gold_trace"].set_index(
        ["position_encoding", "mode", "step", "prompt_sha256", "layer"]
    )
    for metric in (
        "gold_probability",
        "gold_logit_margin_vs_count_minus_one",
        "ridge_count_prediction",
    ):
        lookup = baseline[metric]
        keys = pd.MultiIndex.from_frame(
            frame[["position_encoding", "mode", "step", "prompt_sha256", "layer"]]
        )
        frame[f"delta_{metric}"] = frame[metric].to_numpy() - lookup.reindex(keys).to_numpy()
    return frame


def _part_dir(analysis_dir: Path, position_encoding: str, mode: str, step: int) -> Path:
    return analysis_dir / "parts" / f"{position_encoding}_{mode}_step_{step:06d}"


def _write_part(frame: pd.DataFrame, part_dir: Path, name: str) -> None:
    if not frame.empty:
        atomic_csv(frame, part_dir / f"{name}.csv")


def _part_complete(part_dir: Path, options_fingerprint: str) -> bool:
    marker = part_dir / "complete.json"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("options_fingerprint") == options_fingerprint
    except (json.JSONDecodeError, OSError):
        return False


def _analyze_checkpoint(
    run_dir: Path,
    analysis_dir: Path,
    cfg: V16_2Config,
    train_examples: list[V16_2Example],
    heldout_examples: list[V16_2Example],
    options: DynamicsOptions,
    options_fingerprint: str,
    *,
    position_encoding: str,
    mode: str,
    step: int,
) -> None:
    part_dir = _part_dir(analysis_dir, position_encoding, mode, step)
    if not options.force and _part_complete(part_dir, options_fingerprint):
        record_cached_event(
            run_dir,
            scope="checkpoint_dynamics",
            block="checkpoint_total",
            position_encoding=position_encoding,
            mode=mode,
            step=step,
            device=cfg.device,
        )
        print(f"[dynamics:cached] {position_encoding}/{mode} step={step}", flush=True)
        return
    # A changed option fingerprint defines a different partition schema/subset.
    # Rebuild it from an empty directory so disabled families cannot leave stale CSVs.
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    with timed_event(
        run_dir,
        scope="checkpoint_dynamics",
        block="checkpoint_total",
        position_encoding=position_encoding,
        mode=mode,
        step=step,
    ):
        with timed_event(
            run_dir, scope="checkpoint_dynamics", block="checkpoint_model_load",
            position_encoding=position_encoding, mode=mode, step=step,
            device=cfg.device,
        ):
            loaded_cfg, vocab, _, _, model = load_v16_2_checkpoint_model(
                run_dir, position_encoding, mode, step=step, device=cfg.device
            )
        attention: pd.DataFrame = pd.DataFrame()
        ar: pd.DataFrame = pd.DataFrame()
        train_states: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        heldout_states: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

        if options.run_attention:
            attention_examples, split_by_prompt = _attention_splits(
                heldout_examples, options.attention_examples_per_count, cfg.count_max_threshold
            )
            with timed_event(
                run_dir, scope="checkpoint_dynamics", block="attention_metrics",
                position_encoding=position_encoding, mode=mode, step=step,
                num_examples=len(attention_examples),
            ):
                attention = collect_v16_2_attention(
                    model, loaded_cfg, vocab, attention_examples,
                    position_encoding=position_encoding, mode=mode,
                )
                attention["step"] = step
                attention["diagnostic_split"] = attention.prompt_sha256.map(split_by_prompt)
                attention["correct_top1_minus_chance"] = (
                    attention.correct_top1 - attention.chance_top1
                )
                _write_part(attention, part_dir, "attention_detail")
                groups = ["position_encoding", "mode", "step", "diagnostic_split", "query_kind", "layer", "head"]
                _write_part(_numeric_mean(attention, groups, {"count", "query_k", "example_id"}), part_dir, "attention_summary")
                by_count = groups[:-2] + ["count", "layer", "head"]
                _write_part(_numeric_mean(attention, by_count, {"query_k", "example_id"}), part_dir, "attention_by_count")
                trace = attention[attention.query_kind == "trace_index"]
                _write_part(_numeric_mean(trace, groups[:-2] + ["query_k", "layer", "head"], {"count", "example_id"}), part_dir, "attention_by_k")

        if options.run_generated:
            ar_examples = _balanced_subset(
                heldout_examples, options.ar_examples_per_count, cfg.count_max_threshold
            )
            with timed_event(
                run_dir, scope="checkpoint_dynamics", block="autoregressive_evaluation",
                position_encoding=position_encoding, mode=mode, step=step,
                num_examples=len(ar_examples),
            ):
                ar = autoregressive_task_evaluation(
                    model, loaded_cfg, vocab, ar_examples,
                    position_encoding=position_encoding, mode=mode, step=step,
                )
                _write_part(ar, part_dir, "autoregressive")
                if not attention.empty:
                    per_prompt = _numeric_mean(
                        attention[attention.query_kind == "final_answer"],
                        ["position_encoding", "mode", "step", "prompt_sha256"],
                        {"count", "query_k", "example_id", "layer", "head"},
                    )
                    link = per_prompt.merge(
                        ar[["position_encoding", "mode", "step", "prompt_sha256", "count", "ar_accuracy", "ar_abs_error"]],
                        on=["position_encoding", "mode", "step", "prompt_sha256"], how="inner",
                    )
                    _write_part(link, part_dir, "attention_behavior_link")

        if options.run_states:
            train_subset = _balanced_subset(
                train_examples, options.state_train_examples_per_count, cfg.count_max_threshold
            )
            heldout_subset = _balanced_subset(
                heldout_examples, options.state_eval_examples_per_count, cfg.count_max_threshold
            )
            with timed_event(
                run_dir, scope="checkpoint_dynamics", block="hidden_state_metrics",
                position_encoding=position_encoding, mode=mode, step=step,
                num_examples=len(train_subset) + len(heldout_subset),
            ):
                train_states = collect_v16_2_states(
                    model, loaded_cfg, vocab, train_subset, mode,
                    options.state_train_examples_per_count,
                )
                heldout_states = collect_v16_2_states(
                    model, loaded_cfg, vocab, heldout_subset, mode,
                    options.state_eval_examples_per_count,
                )
                probes, by_count = _probe_rows(
                    train_states, heldout_states, position_encoding=position_encoding,
                    mode=mode, step=step, context="teacher_forced",
                )
                _write_part(probes, part_dir, "state_probe_summary")
                _write_part(by_count, part_dir, "state_by_count")
                _write_part(
                    _geometry_rows(
                        heldout_states, position_encoding=position_encoding, mode=mode,
                        step=step, context="teacher_forced",
                    ),
                    part_dir, "state_geometry",
                )
                _write_part(
                    _cross_site_rows(
                        train_states, heldout_states, position_encoding=position_encoding,
                        mode=mode, step=step,
                    ),
                    part_dir, "state_cross_site",
                )
                if options.run_similarity:
                    _save_states(part_dir / "heldout_states.npz", heldout_states)

            if options.run_generated and not ar.empty:
                with timed_event(
                    run_dir, scope="checkpoint_dynamics", block="generated_state_metrics",
                    position_encoding=position_encoding, mode=mode, step=step,
                    num_examples=len(ar),
                ):
                    generated_states, statuses = _generated_final_states(
                        model, loaded_cfg, vocab, ar
                    )
                    probes, by_count = _probe_rows(
                        train_states, generated_states, position_encoding=position_encoding,
                        mode=mode, step=step, context="generated_prefix",
                    )
                    if not probes.empty:
                        probes = probes.merge(
                            pd.DataFrame({"generated_state_status": ["valid"]}), how="cross"
                        )
                    _write_part(probes, part_dir, "state_probe_summary_generated")
                    _write_part(by_count, part_dir, "state_by_count_generated")
                    _write_part(statuses.assign(
                        position_encoding=position_encoding, mode=mode, step=step
                    ), part_dir, "generated_state_status")

            if options.run_counterfactual and mode == "thinking":
                counterfactual_examples = _balanced_subset(
                    heldout_examples, options.state_eval_examples_per_count,
                    cfg.count_max_threshold,
                )
                with timed_event(
                    run_dir, scope="checkpoint_dynamics", block="counterfactual_trace_metrics",
                    position_encoding=position_encoding, mode=mode, step=step,
                    num_examples=len(counterfactual_examples),
                ):
                    _write_part(
                        _counterfactual_rows(
                            model, loaded_cfg, vocab, counterfactual_examples, train_states,
                            position_encoding=position_encoding, mode=mode, step=step,
                        ),
                        part_dir, "state_counterfactual_trace",
                    )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _atomic_json(
            {
                "status": "complete",
                "position_encoding": position_encoding,
                "mode": mode,
                "step": step,
                "options_fingerprint": options_fingerprint,
            },
            part_dir / "complete.json",
        )


def _aggregate_parts(analysis_dir: Path, table_dir: Path) -> dict[str, pd.DataFrame]:
    mapping = {
        "attention_detail": "checkpoint_attention_detail.csv",
        "attention_summary": "checkpoint_attention_summary.csv",
        "attention_by_count": "checkpoint_attention_by_count.csv",
        "attention_by_k": "checkpoint_attention_by_k.csv",
        "autoregressive": "checkpoint_dynamics_autoregressive.csv",
        "attention_behavior_link": "checkpoint_attention_behavior_link.csv",
        "state_probe_summary": "checkpoint_state_probe_summary.csv",
        "state_by_count": "checkpoint_state_by_count.csv",
        "state_geometry": "checkpoint_state_geometry.csv",
        "state_cross_site": "checkpoint_state_cross_site.csv",
        "state_counterfactual_trace": "checkpoint_counterfactual_trace_readout.csv",
    }
    extra = {
        "state_probe_summary_generated": "checkpoint_state_probe_summary.csv",
        "state_by_count_generated": "checkpoint_state_by_count.csv",
        "generated_state_status": "checkpoint_generated_state_status.csv",
    }
    result: dict[str, pd.DataFrame] = {}
    for part_name, output_name in {**mapping, **extra}.items():
        frames = [_read_csv(path) for path in sorted((analysis_dir / "parts").glob(f"*/{part_name}.csv"))]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        if output_name in result:
            combined = pd.concat((result[output_name], combined), ignore_index=True)
        result[output_name] = combined
    for output_name in set(mapping.values()) | set(extra.values()):
        if output_name not in result:
            (table_dir / output_name).unlink(missing_ok=True)
    for output_name, frame in result.items():
        atomic_csv(frame, table_dir / output_name)
    return result


def _head_stability(attention: pd.DataFrame, table_dir: Path) -> pd.DataFrame:
    if attention.empty:
        (table_dir / "checkpoint_head_stability.csv").unlink(missing_ok=True)
        return pd.DataFrame()
    metrics = {
        "needle_retrieval": "needle_attention_enrichment",
        "needle_coverage": "top_n_needle_recall",
        "trace_routing": "trace_readout_mass",
        "ordered_trace": "correct_top1_minus_chance",
    }
    rows: list[dict[str, Any]] = []
    for (position_encoding, mode), variant in attention.groupby(["position_encoding", "mode"]):
        final_step = int(variant.step.max())
        selection = variant[
            (variant.step == final_step) & (variant.diagnostic_split == "head_selection")
        ]
        for role, metric in metrics.items():
            if mode != "thinking" and role in {"trace_routing", "ordered_trace"}:
                continue
            eligible = selection
            if role == "ordered_trace":
                eligible = eligible[eligible.query_kind == "trace_index"]
            elif role == "trace_routing":
                eligible = eligible[eligible.query_kind == "final_answer"]
            else:
                eligible = eligible[eligible.query_kind == "final_answer"]
            scores = eligible.groupby(["layer", "head"])[metric].mean().dropna()
            if scores.empty:
                continue
            chosen_layer, chosen_head = scores.idxmax()
            for step, current in variant[variant.diagnostic_split == "heldout_reporting"].groupby("step"):
                current = current[(current.layer == chosen_layer) & (current["head"] == chosen_head)]
                current = current[current.query_kind == ("trace_index" if role == "ordered_trace" else "final_answer")]
                all_scores = variant[
                    (variant.step == step)
                    & (variant.diagnostic_split == "heldout_reporting")
                    & (variant.query_kind == ("trace_index" if role == "ordered_trace" else "final_answer"))
                ].groupby(["layer", "head"])[metric].mean().sort_values(ascending=False)
                rank = math.nan
                if (chosen_layer, chosen_head) in all_scores.index:
                    rank = int(all_scores.index.get_loc((chosen_layer, chosen_head))) + 1
                rows.append({
                    "position_encoding": position_encoding, "mode": mode,
                    "step": int(step), "role": role, "metric": metric,
                    "selected_at_step": final_step, "layer": int(chosen_layer),
                    "head": int(chosen_head), "heldout_value": float(current[metric].mean()),
                    "heldout_best_current_value": (
                        float(all_scores.iloc[0]) if not all_scores.empty else math.nan
                    ),
                    "heldout_rank": rank,
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        atomic_csv(result, table_dir / "checkpoint_head_stability.csv")
    else:
        (table_dir / "checkpoint_head_stability.csv").unlink(missing_ok=True)
    return result


def _state_similarity(analysis_dir: Path, table_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = list(sorted((analysis_dir / "parts").glob("*/heldout_states.npz")))
    variants: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    for path in pattern:
        marker = json.loads((path.parent / "complete.json").read_text(encoding="utf-8"))
        variants.setdefault((marker["position_encoding"], marker["mode"]), []).append((int(marker["step"]), path))
    for (position_encoding, mode), entries in variants.items():
        entries.sort()
        loaded = [(step, _load_states(path)) for step, path in entries]
        final_step, final_states = loaded[-1]
        for index, (step, states) in enumerate(loaded):
            comparisons: list[tuple[str, int, dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]]] = [
                ("final", final_step, final_states)
            ]
            if index:
                comparisons.append(("previous", loaded[index - 1][0], loaded[index - 1][1]))
            for reference, reference_step, other in comparisons:
                for site, layer in sorted(set(states) & set(other)):
                    left, left_y = states[(site, layer)]
                    right, right_y = other[(site, layer)]
                    if not np.array_equal(left_y, right_y):
                        raise ValueError("state-similarity examples are not aligned")
                    rows.append({
                        "position_encoding": position_encoding, "mode": mode,
                        "step": step, "reference": reference,
                        "reference_step": reference_step, "site": site, "layer": layer,
                        "linear_cka": linear_cka(left, right), "examples": len(left),
                    })
    result = pd.DataFrame(rows)
    if not result.empty:
        atomic_csv(result, table_dir / "checkpoint_state_similarity.csv")
    else:
        (table_dir / "checkpoint_state_similarity.csv").unlink(missing_ok=True)
    return result


def run_v16_2_checkpoint_dynamics(
    run_dir: str | Path,
    options: DynamicsOptions | None = None,
    *,
    device: str | None = None,
) -> Path:
    run_dir = Path(run_dir).resolve()
    options = options or DynamicsOptions()
    options.validate()
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    if device is not None:
        cfg = replace(cfg, device=device)
    text = load_corpus_text()
    split = load_corpus_split(run_dir / "data" / "corpus_split.json", cfg, text)
    vocab = V16_2Vocab.load(run_dir / "vocab.json")
    pool = load_needle_pool(
        run_dir / "data" / "needle_pool.json", cfg,
        split_fingerprint=split.split_fingerprint, vocab_fingerprint=vocab.fingerprint,
    )
    suites, _ = load_suite_manifests(
        run_dir / "data" / "loss_suite_manifests.json",
        split_fingerprint=split.split_fingerprint, pool_fingerprint=pool.pool_fingerprint,
    )
    train_examples = suites["train"]["task"]
    heldout_examples = suites["heldout"]["task"]
    analysis_dir = run_dir / "analysis" / "checkpoint_dynamics"
    table_dir = run_dir / "tables"
    options_payload = asdict(options)
    options_fingerprint = _json_fingerprint({
        "version": "v16_2_revision_5", "config": cfg.to_dict(),
        "options": {key: value for key, value in options_payload.items() if key != "force"},
        "split": split.split_fingerprint, "pool": pool.pool_fingerprint,
    })
    inventory: list[dict[str, Any]] = []
    for position_encoding, mode in cfg.model_variants:
        steps = checkpoint_steps(run_dir, position_encoding, mode)
        if not steps:
            raise FileNotFoundError(f"no numeric checkpoints for {position_encoding}/{mode}")
        for step, _ in steps:
            inventory.append({"position_encoding": position_encoding, "mode": mode, "step": step})
        print(
            f"[dynamics:inventory] {position_encoding}/{mode} "
            f"steps={[step for step, _ in steps]}",
            flush=True,
        )
    _atomic_json(
        {"status": "running", "options": options_payload,
         "options_fingerprint": options_fingerprint, "inventory": inventory},
        analysis_dir / "manifest.json",
    )
    for item in inventory:
        _analyze_checkpoint(
            run_dir, analysis_dir, cfg, train_examples, heldout_examples,
            options, options_fingerprint, **item,
        )
    with timed_event(run_dir, scope="checkpoint_dynamics", block="aggregate_tables"):
        tables = _aggregate_parts(analysis_dir, table_dir)
        attention = tables.get("checkpoint_attention_detail.csv", pd.DataFrame())
        _head_stability(attention, table_dir)
        if options.run_similarity:
            _state_similarity(analysis_dir, table_dir)
        else:
            (table_dir / "checkpoint_state_similarity.csv").unlink(missing_ok=True)
    with timed_event(run_dir, scope="checkpoint_dynamics", block="visualization_plots"):
        from .plots import plot_v16_2_checkpoint_dynamics

        plot_v16_2_checkpoint_dynamics(run_dir)
    _atomic_json(
        {"status": "complete", "options": options_payload,
         "options_fingerprint": options_fingerprint, "inventory": inventory},
        analysis_dir / "manifest.json",
    )
    return analysis_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze v16_2 checkpoints across training")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attention-examples-per-count", type=int, default=20)
    parser.add_argument("--ar-examples-per-count", type=int, default=10)
    parser.add_argument("--state-train-examples-per-count", type=int, default=40)
    parser.add_argument("--state-eval-examples-per-count", type=int, default=15)
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--skip-states", action="store_true")
    parser.add_argument("--skip-generated", action="store_true")
    parser.add_argument("--skip-counterfactual", action="store_true")
    parser.add_argument("--skip-similarity", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    options = DynamicsOptions(
        attention_examples_per_count=args.attention_examples_per_count,
        ar_examples_per_count=args.ar_examples_per_count,
        state_train_examples_per_count=args.state_train_examples_per_count,
        state_eval_examples_per_count=args.state_eval_examples_per_count,
        run_attention=not args.skip_attention,
        run_states=not args.skip_states,
        run_generated=not args.skip_generated,
        run_counterfactual=not args.skip_counterfactual,
        run_similarity=not args.skip_similarity,
        force=args.force,
    )
    result = run_v16_2_checkpoint_dynamics(args.run_dir, options, device=args.device)
    print(f"Checkpoint dynamics saved under {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
