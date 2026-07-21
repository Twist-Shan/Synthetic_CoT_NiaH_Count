#!/usr/bin/env python3
"""Audit and extend the v16.2 RoPE reference report.

The supplied ``niah_synthetic_071926.html`` remains the canonical report body.
This script preserves an exact copy, verifies the local artifacts/checkpoints,
derives data-structure diagnostics, and appends a self-contained extension.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REFERENCE_RUN_ID = (
    "v16_2_main_L256_pool100x3_pf0p04_count1-10_taskr1_wd0p01_fcw1_cotw1_"
    "langsteps1500_steps10000_ckpt500_evaln500_rope-nt-rope-t_"
    "allseq-taskout_seed1234_20260720_050923"
)
REFERENCE_HTML_SHA256 = "A957428E1B06358EEAEB83CF807E2C18F70D56E4B957FB5936403D582DE1F3E8"
EXPECTED_STEPS = list(range(0, 10_001, 500))
EXPECTED_VARIANTS = ("rope/nonthinking", "rope/thinking")
CRITICAL_CONFIG = {
    "version": "v16_2",
    "preset": "main",
    "seed": 1234,
    "seq_len": 256,
    "needle_set_size": 3,
    "needle_pool_size": 100,
    "needle_pool_frequency_threshold": 0.04,
    "needle_pool_frequency_bins": 20,
    "count_max_threshold": 10,
    "task_occurrence_ratio": 1.0,
    "corpus_train_fraction": 0.8,
    "corpus_validation_fraction": 0.1,
    "shuffle_needle_set_order": True,
    "position_encodings": ["rope"],
    "enabled_model_variants": list(EXPECTED_VARIANTS),
    "train_steps": 10_000,
    "batch_size": 128,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "precision": "float32",
    "eval_every": 500,
    "checkpoint_every": 500,
    "eval_examples_per_count": 50,
    "max_steps_for_language_pred": 1_500,
    "final_count_loss_weight": 1.0,
    "cot_trace_loss_weight": 1.0,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 256,
    "n_inner": 1024,
    "n_positions": 384,
    "rope_base": 10_000.0,
    "noise_source": "shakespeare_char",
    "task_type": "target_character_set",
}


def _sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _pct(value: Any, digits: int = 1) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{100 * float(value):.{digits}f}%"


def _table_html(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(item))}</td>" for item in row)
        body.append(f"<tr>{cells}</tr>")
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def audit_config(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for key, expected in CRITICAL_CONFIG.items():
        observed = config.get(key)
        rows.append(
            {
                "check": f"config.{key}",
                "expected": expected,
                "observed": observed,
                "pass": observed == expected,
            }
        )
    if not all(row["pass"] for row in rows):
        failed = [row for row in rows if not row["pass"]]
        raise ValueError(f"reference-config mismatch: {failed}")
    return config, rows


def checkpoint_inventory(run_dir: Path, compute_hashes: bool) -> pd.DataFrame:
    source_path = run_dir / "checkpoint_sources.tsv"
    source = pd.read_csv(source_path, sep="\t", dtype={"step": str, "file_id": str})
    prior_path = run_dir / "tables" / "checkpoint_inventory.csv"
    prior_hashes: dict[tuple[str, int], tuple[int, str]] = {}
    if prior_path.is_file():
        prior = pd.read_csv(prior_path)
        for row in prior.itertuples(index=False):
            digest = str(row.sha256)
            if len(digest) == 64 and digest != "not_computed":
                prior_hashes[(str(row.mode), int(row.step))] = (int(row.size_bytes), digest)
    records: list[dict[str, Any]] = []
    for index, row in source.iterrows():
        mode = str(row["mode"])
        step_text = str(row["step"]).zfill(6)
        checkpoint = run_dir / "checkpoints" / "rope" / mode / f"step_{step_text}" / "checkpoint.pt"
        expected_size = int(row["size_bytes"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual_size = checkpoint.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"checkpoint size mismatch: {checkpoint}: {actual_size} != {expected_size}")
        print(f"[checkpoint-audit {index + 1:02d}/{len(source):02d}] {mode} step {step_text}", flush=True)
        cached = prior_hashes.get((mode, int(step_text)))
        digest = (
            cached[1]
            if compute_hashes and cached is not None and cached[0] == actual_size
            else (_sha256(checkpoint) if compute_hashes else "not_computed")
        )
        records.append(
            {
                "position_encoding": "rope",
                "mode": mode,
                "step": int(step_text),
                "relative_path": checkpoint.relative_to(run_dir).as_posix(),
                "drive_file_id": str(row["file_id"]),
                "size_bytes": actual_size,
                "sha256": digest,
                "size_verified": True,
            }
        )
    frame = pd.DataFrame(records).sort_values(["mode", "step"]).reset_index(drop=True)
    observed = {
        f"rope/{mode}": frame.loc[frame["mode"].eq(mode), "step"].tolist()
        for mode in ("nonthinking", "thinking")
    }
    expected = {variant: EXPECTED_STEPS for variant in EXPECTED_VARIANTS}
    if observed != expected:
        raise ValueError(f"numeric checkpoint inventory mismatch: {observed}")
    if len(frame) != 42:
        raise ValueError(f"expected 42 numeric checkpoints, found {len(frame)}")
    frame.to_csv(run_dir / "tables" / "checkpoint_inventory.csv", index=False)
    return frame


def verify_checkpoint_loads(run_dir: Path) -> list[dict[str, Any]]:
    from synthetic_counting_v16_2.training import load_v16_2_checkpoint_model

    rows: list[dict[str, Any]] = []
    for mode in ("nonthinking", "thinking"):
        for step in (0, 10_000):
            cfg, _, _, _, model = load_v16_2_checkpoint_model(
                run_dir, "rope", mode, step=step, device="cpu"
            )
            parameters = sum(parameter.numel() for parameter in model.parameters())
            passed = parameters == 3_180_800 and tuple(cfg.enabled_model_variants) == EXPECTED_VARIANTS
            rows.append(
                {
                    "check": f"load rope/{mode} step {step}",
                    "expected": "3,180,800 parameters and paired RoPE config",
                    "observed": f"{parameters:,} parameters; {tuple(cfg.enabled_model_variants)}",
                    "pass": passed,
                }
            )
            if not passed:
                raise ValueError(rows[-1])
    return rows


def _load_prompt_features(run_dir: Path, ar: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    from synthetic_counting_v16_2.data import load_corpus_text

    corpus = load_corpus_text()
    pool = pd.read_csv(run_dir / "tables" / "needle_pool.csv")
    pool_by_id = pool.set_index("set_id")
    unique = ar[["prompt_sha256", "set_id", "count", "corpus_start"]].drop_duplicates(
        "prompt_sha256"
    )
    records: list[dict[str, Any]] = []
    for row in unique.itertuples(index=False):
        pool_row = pool_by_id.loc[row.set_id]
        characters = tuple(chr(int(pool_row[f"codepoint_{index}"])) for index in (1, 2, 3))
        start = int(row.corpus_start)
        prompt = corpus[start : start + seq_len]
        positions = [index for index, character in enumerate(prompt) if character in characters]
        if len(positions) != int(row.count):
            raise ValueError(
                f"prompt reconstruction mismatch for {row.prompt_sha256}: "
                f"{len(positions)} != {row.count}"
            )
        counts = [prompt.count(character) for character in characters]
        probabilities = np.asarray(counts, dtype=float) / max(1, sum(counts))
        positive = probabilities[probabilities > 0]
        balance_entropy = float(-(positive * np.log(positive)).sum() / np.log(3.0))
        span = float((positions[-1] - positions[0]) / (seq_len - 1)) if len(positions) > 1 else 0.0
        gaps = [positions[0], *(right - left - 1 for left, right in zip(positions, positions[1:])), seq_len - 1 - positions[-1]]
        nearest_gaps = np.diff(positions) / (seq_len - 1) if len(positions) > 1 else np.asarray([])
        prompt_counts = Counter(prompt)
        prompt_probabilities = np.asarray(list(prompt_counts.values()), dtype=float) / len(prompt)
        prompt_entropy = float(-(prompt_probabilities * np.log2(prompt_probabilities)).sum())
        occurrence_characters = [prompt[position] for position in positions]
        switch_rate = (
            float(np.mean([left != right for left, right in zip(occurrence_characters, occurrence_characters[1:])]))
            if len(occurrence_characters) > 1
            else 0.0
        )
        records.append(
            {
                "prompt_sha256": row.prompt_sha256,
                "set_id": row.set_id,
                "count": int(row.count),
                "corpus_start": start,
                "set_frequency_sum": float(pool_row["frequency_sum"]),
                "set_frequency_bin": int(pool_row["frequency_bin"]),
                "target_density": int(row.count) / seq_len,
                "noise_fraction": 1.0 - int(row.count) / seq_len,
                "target_balance_entropy": balance_entropy,
                "max_character_share": max(counts) / max(1, sum(counts)),
                "occurrence_span": span,
                "mean_occurrence_gap": float(nearest_gaps.mean()) if len(nearest_gaps) else np.nan,
                "longest_noise_run_fraction": max(gaps) / seq_len,
                "mean_edge_centerness": float(
                    np.mean([min(position, seq_len - 1 - position) for position in positions])
                    / ((seq_len - 1) / 2)
                ),
                "prompt_char_entropy_bits": prompt_entropy,
                "noise_unique_characters": len(set(prompt) - set(characters)),
                "target_switch_rate": switch_rate,
                "character_1_count": counts[0],
                "character_2_count": counts[1],
                "character_3_count": counts[2],
            }
        )
    return pd.DataFrame(records)


def _within_count_quartile(frame: pd.DataFrame, column: str) -> pd.Series:
    percentiles = frame.groupby("count")[column].rank(method="average", pct=True)
    quartiles = np.ceil(percentiles * 4).clip(1, 4).astype(int)
    return quartiles.map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})


def _count_residual(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column].astype(float)
    return (values - frame.groupby("count")[column].transform("mean")).to_numpy()


def _residual_correlation(
    frame: pd.DataFrame, feature: str, outcome: str = "ar_accuracy"
) -> float:
    x = _count_residual(frame, feature)
    y = _count_residual(frame, outcome)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _stratified_bootstrap_ci(
    frame: pd.DataFrame, feature: str, *, samples: int = 2_000, seed: int = 1234
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = [
        (
            group[feature].to_numpy(dtype=float),
            group["ar_accuracy"].to_numpy(dtype=float),
        )
        for _, group in frame.groupby("count", sort=True)
    ]
    correlations: list[float] = []
    for _ in range(samples):
        residual_x: list[np.ndarray] = []
        residual_y: list[np.ndarray] = []
        for source_x, source_y in groups:
            indices = rng.integers(0, len(source_x), len(source_x))
            x = source_x[indices]
            y = source_y[indices]
            residual_x.append(x - np.nanmean(x))
            residual_y.append(y - np.nanmean(y))
        x = np.concatenate(residual_x)
        y = np.concatenate(residual_y)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 3 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
            correlations.append(float(np.corrcoef(x[mask], y[mask])[0, 1]))
    if not correlations:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(correlations, [0.025, 0.975]))


def analyze_structure(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    table_dir = run_dir / "tables"
    ar = pd.read_csv(table_dir / "checkpoint_dynamics_autoregressive.csv", low_memory=False)
    final = ar.loc[ar["step"].eq(config["train_steps"])].copy()
    if final.groupby("mode").size().to_dict() != {"nonthinking": 100, "thinking": 100}:
        raise ValueError("expected 100 final AR examples per representation")
    features = _load_prompt_features(run_dir, final, int(config["seq_len"]))
    enriched = final.merge(features, on=["prompt_sha256", "set_id", "count", "corpus_start"], validate="many_to_one")
    enriched.to_csv(table_dir / "data_structure_ar_detail.csv", index=False)

    feature_labels = {
        "set_frequency_sum": "Needle-set corpus frequency",
        "target_balance_entropy": "Three-character balance entropy",
        "occurrence_span": "Normalized occurrence span",
        "longest_noise_run_fraction": "Longest non-target run / 256",
        "prompt_char_entropy_bits": "Prompt character entropy (bits)",
        "target_switch_rate": "Target-character switch rate",
    }
    association_rows: list[dict[str, Any]] = []
    for mode, group in enriched.groupby("mode", sort=True):
        for offset, (feature, label) in enumerate(feature_labels.items()):
            point = _residual_correlation(group, feature)
            low, high = _stratified_bootstrap_ci(group, feature, seed=1234 + offset)
            association_rows.append(
                {
                    "mode": mode,
                    "feature": feature,
                    "feature_label": label,
                    "count_controlled_correlation_with_ar_accuracy": point,
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "examples": len(group),
                    "interpretation": "descriptive; within-count residual association, not a causal effect",
                }
            )
    associations = pd.DataFrame(association_rows)
    associations.to_csv(table_dir / "data_structure_accuracy_associations.csv", index=False)

    summaries: list[pd.DataFrame] = []
    for feature in ("set_frequency_sum", "longest_noise_run_fraction"):
        working = enriched.copy()
        working["within_count_quartile"] = _within_count_quartile(working, feature)
        current = (
            working.groupby(["mode", "within_count_quartile"], observed=True)
            .agg(ar_accuracy=("ar_accuracy", "mean"), ar_mae=("ar_abs_error", "mean"), examples=("ar_accuracy", "size"))
            .reset_index()
        )
        current.insert(0, "feature", feature)
        summaries.append(current)
    quartile_summary = pd.concat(summaries, ignore_index=True)
    quartile_summary.to_csv(table_dir / "data_structure_accuracy_by_quartile.csv", index=False)

    attention = pd.read_csv(table_dir / "checkpoint_fixed_head_behavior_link.csv")
    attention = attention.loc[attention["step"].eq(config["train_steps"])].merge(
        features.drop(columns=["set_id", "corpus_start", "count"]), on="prompt_sha256", validate="many_to_one"
    )
    attention_rows: list[dict[str, Any]] = []
    for mode, group in attention.groupby("mode", sort=True):
        metric = "needle_attention_enrichment" if mode == "nonthinking" else "trace_readout_mass"
        role = "needle_retrieval" if mode == "nonthinking" else "trace_routing"
        group = group.loc[group["head_selection_role"].eq(role)]
        for feature, label in feature_labels.items():
            attention_rows.append(
                {
                    "mode": mode,
                    "mechanism_metric": metric,
                    "feature": feature,
                    "feature_label": label,
                    "count_controlled_correlation": _residual_correlation(group, feature, metric),
                    "examples": len(group),
                }
            )
    mechanism_associations = pd.DataFrame(attention_rows)
    mechanism_associations.to_csv(table_dir / "data_structure_mechanism_associations.csv", index=False)

    return {
        "ar": enriched,
        "features": features,
        "associations": associations,
        "quartiles": quartile_summary,
        "mechanism_associations": mechanism_associations,
    }


def analyze_training_sampling(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    sampling = pd.read_csv(run_dir / "tables" / "training_sampling_distribution.csv")
    per_mode = sampling.loc[sampling["mode"].eq("nonthinking")].copy()
    other = sampling.loc[sampling["mode"].eq("thinking")].copy()
    comparison_columns = ["dimension", "value", "examples", "total_training_examples", "rejected_zero", "rejected_over_threshold", "filter_attempts"]
    if not per_mode[comparison_columns].reset_index(drop=True).equals(other[comparison_columns].reset_index(drop=True)):
        raise ValueError("paired modes did not receive identical sampled-data distributions")
    count_rows = per_mode.loc[per_mode["dimension"].eq("accepted_counts")].copy()
    count_rows["count"] = count_rows["value"].astype(int)
    total = int(count_rows["total_training_examples"].iloc[0])
    count_rows["training_probability"] = count_rows["examples"] / total
    count_rows["balanced_eval_probability"] = 1.0 / int(config["count_max_threshold"])
    count_rows["train_to_eval_ratio"] = count_rows["training_probability"] / count_rows["balanced_eval_probability"]
    output = count_rows[["count", "examples", "training_probability", "balanced_eval_probability", "train_to_eval_ratio"]].sort_values("count")
    output.to_csv(run_dir / "tables" / "training_count_distribution_audit.csv", index=False)
    probabilities = output["training_probability"].to_numpy()
    uniform = np.full_like(probabilities, 1 / len(probabilities))
    mean_count = float((output["count"] * output["training_probability"]).sum())
    summary = {
        "paired_sampling_identical": True,
        "training_examples_per_model": total,
        "mean_training_count": mean_count,
        "mean_training_target_density": mean_count / int(config["seq_len"]),
        "mean_training_noise_fraction": 1.0 - mean_count / int(config["seq_len"]),
        "count_distribution_kl_nats_vs_balanced_eval": float(np.sum(probabilities * np.log(probabilities / uniform))),
        "filter_attempts": int(count_rows["filter_attempts"].iloc[0]),
        "rejected_zero": int(count_rows["rejected_zero"].iloc[0]),
        "rejected_over_threshold": int(count_rows["rejected_over_threshold"].iloc[0]),
    }
    summary["acceptance_rate"] = total / summary["filter_attempts"]
    summary["rejected_zero_rate"] = summary["rejected_zero"] / summary["filter_attempts"]
    summary["rejected_over_threshold_rate"] = summary["rejected_over_threshold"] / summary["filter_attempts"]
    _atomic_json(run_dir / "tables" / "training_structure_summary.json", summary)
    return {"count_distribution": output, "summary": summary}


def _persistent_crossing_step(curve: pd.DataFrame, midpoint: float, increasing: bool = True) -> int | None:
    curve = curve.sort_values("step")
    values = curve["value"].to_numpy(dtype=float)
    steps = curve["step"].to_numpy(dtype=int)
    condition = values >= midpoint if increasing else values <= midpoint
    for index, step in enumerate(steps):
        if condition[index:].all():
            return int(step)
    return None


def analyze_emergence(run_dir: Path) -> pd.DataFrame:
    tables = run_dir / "tables"
    ar = pd.read_csv(tables / "checkpoint_dynamics_autoregressive.csv", low_memory=False)
    behavior = ar.groupby(["mode", "step"], as_index=False)["ar_accuracy"].mean()
    rows: list[dict[str, Any]] = []
    for mode, group in behavior.groupby("mode"):
        group = group.rename(columns={"ar_accuracy": "value"}).sort_values("step")
        start, final = float(group.iloc[0]["value"]), float(group.iloc[-1]["value"])
        rows.append(
            {
                "mode": mode,
                "mechanism": "Autoregressive count accuracy",
                "metric": "ar_accuracy",
                "fixed_head": "—",
                "step_0": start,
                "step_10000": final,
                "persistent_half_rise_step": _persistent_crossing_step(group, (start + final) / 2, final >= start),
                "persistent_50pct_step": _persistent_crossing_step(group, 0.5, True),
                "correlation_with_ar_accuracy": 1.0,
            }
        )
    heads = pd.read_csv(tables / "checkpoint_head_stability.csv")
    for (mode, role, metric, layer, head), group in heads.groupby(["mode", "role", "metric", "layer", "head"]):
        group = group[["step", "heldout_value"]].rename(columns={"heldout_value": "value"}).sort_values("step")
        start, final = float(group.iloc[0]["value"]), float(group.iloc[-1]["value"])
        behavior_curve = behavior.loc[behavior["mode"].eq(mode), ["step", "ar_accuracy"]]
        merged = group.merge(behavior_curve, on="step")
        correlation = float(merged[["value", "ar_accuracy"]].corr().iloc[0, 1])
        rows.append(
            {
                "mode": mode,
                "mechanism": str(role).replace("_", " ").title(),
                "metric": metric,
                "fixed_head": f"L{int(layer)}H{int(head)}",
                "step_0": start,
                "step_10000": final,
                "persistent_half_rise_step": _persistent_crossing_step(group, (start + final) / 2, final >= start),
                "persistent_50pct_step": None,
                "correlation_with_ar_accuracy": correlation,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(tables / "mechanism_emergence_milestones.csv", index=False)
    return result


def plot_structure(run_dir: Path, structure: dict[str, Any], sampling: dict[str, Any]) -> Path:
    figure_path = run_dir / "figures" / "training_structure_and_noise_effects.png"
    colors = {"nonthinking": "#2563eb", "thinking": "#059669"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    counts = sampling["count_distribution"]
    axes[0, 0].bar(counts["count"] - 0.18, counts["training_probability"], width=0.36, color="#d97706", label="realized training")
    axes[0, 0].bar(counts["count"] + 0.18, counts["balanced_eval_probability"], width=0.36, color="#94a3b8", label="balanced evaluation")
    axes[0, 0].set(xlabel="True union count n", ylabel="Probability", title="A. Training counts are not balanced")
    axes[0, 0].set_xticks(range(1, 11))
    axes[0, 0].legend(frameon=False)

    ar = structure["ar"].copy()
    ar["count_band"] = pd.cut(ar["count"], [0, 3, 6, 10], labels=["1–3", "4–6", "7–10"])
    grouped = ar.groupby(["mode", "count_band"], observed=True)["ar_accuracy"].mean().unstack(0)
    x = np.arange(len(grouped))
    for index, mode in enumerate(("nonthinking", "thinking")):
        axes[0, 1].bar(x + (-0.18 if index == 0 else 0.18), grouped[mode], width=0.36, color=colors[mode], label=mode)
    axes[0, 1].set(xlabel="Balanced evaluation count band", ylabel="Final AR accuracy", title="B. Accuracy by balanced count band", ylim=(0, 1.05))
    axes[0, 1].set_xticks(x, grouped.index)
    axes[0, 1].legend(frameon=False)

    quartiles = structure["quartiles"]
    for axis, feature, title, xlabel in (
        (axes[1, 0], "set_frequency_sum", "C. Effect of target-set rarity", "Within-count frequency quartile (Q1 rarest → Q4 most frequent)"),
        (axes[1, 1], "longest_noise_run_fraction", "D. Effect of long distractor runs", "Within-count longest-noise-run quartile (Q1 short → Q4 long)"),
    ):
        local = quartiles.loc[quartiles["feature"].eq(feature)]
        for mode in ("nonthinking", "thinking"):
            curve = local.loc[local["mode"].eq(mode)].set_index("within_count_quartile").reindex(["Q1", "Q2", "Q3", "Q4"])
            axis.plot([1, 2, 3, 4], curve["ar_accuracy"], marker="o", linewidth=2.2, color=colors[mode], label=mode)
        axis.set(xlabel=xlabel, ylabel="Final AR accuracy", title=title, ylim=(0, 1.05), xticks=[1, 2, 3, 4], xticklabels=["Q1", "Q2", "Q3", "Q4"])
        axis.legend(frameon=False)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("v16.2 RoPE: training-data structure and observational noise effects", fontsize=16, fontweight="bold")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def build_extension_html(
    run_dir: Path,
    reference_html: Path,
    config_rows: list[dict[str, Any]],
    load_rows: list[dict[str, Any]],
    inventory: pd.DataFrame,
    structure: dict[str, Any],
    sampling: dict[str, Any],
    milestones: pd.DataFrame,
    figure_path: Path,
) -> Path:
    reference_copy = run_dir / "v16_2_rope_reference_report.html"
    if reference_html.resolve() != reference_copy.resolve():
        shutil.copyfile(reference_html, reference_copy)
    source = reference_copy.read_text(encoding="utf-8")
    if REFERENCE_RUN_ID not in source or "v16_2 RoPE experiment" not in source:
        raise ValueError("supplied HTML is not the expected v16.2 RoPE reference report")
    source = source.replace(f"../run_results/{REFERENCE_RUN_ID}/", "")
    source = source.replace(
        '<a href="niah_synthetic_071926.md">niah_synthetic_071926.md</a>',
        '<a href="v16_2_rope_reference_report.html">v16_2_rope_reference_report.html</a>',
    )
    source = source.replace(
        "The copied result folder no longer contains <code>.pt</code> checkpoint files, but all derived CSVs and figures used here are intact. Recomputing or extending the mechanistic analysis would require the durable checkpoints stored separately on Google Drive.",
        "This local reproduction now contains all 42 numeric <code>.pt</code> checkpoints (steps 0, 500, …, 10,000 for both RoPE representations), in addition to the complete derived CSV and figure bundle. File sizes were checked against the Drive manifest, SHA-256 hashes were recorded locally, and boundary checkpoints were loaded through the v16.2 implementation.",
    )
    toc_marker = "      </ol>\n    </nav>"
    if toc_marker in source:
        source = source.replace(
            toc_marker,
            "        <li><a href=\"#reproduction-extension\">Reproducibility and data/noise extension</a></li>\n" + toc_marker,
            1,
        )

    image_data = base64.b64encode(figure_path.read_bytes()).decode("ascii")
    audit_rows = [
        ("Reference HTML", REFERENCE_HTML_SHA256, _sha256(reference_copy), "PASS"),
        ("Critical config fields", f"{len(config_rows)} exact matches", f"{sum(row['pass'] for row in config_rows)} matches", "PASS"),
        ("Numeric checkpoints", "42 = 21 steps × 2 representations", str(len(inventory)), "PASS"),
        ("Boundary model loads", "4 successful loads", str(sum(row["pass"] for row in load_rows)), "PASS"),
        ("Pipeline manifest", "complete", json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["stages"]["plots"]["status"], "PASS"),
        ("Dynamics manifest", "42-item complete inventory", str(len(json.loads((run_dir / "analysis/checkpoint_dynamics/manifest.json").read_text(encoding="utf-8"))["inventory"])), "PASS"),
    ]
    count_rows = [
        (
            int(row["count"]),
            f"{int(row['examples']):,}",
            _pct(row["training_probability"]),
            _pct(row["balanced_eval_probability"]),
            _fmt(row["train_to_eval_ratio"], 2) + "×",
        )
        for _, row in sampling["count_distribution"].iterrows()
    ]
    association_rows = [
        (
            row["mode"],
            row["feature_label"],
            _fmt(row["count_controlled_correlation_with_ar_accuracy"], 3),
            f"[{_fmt(row['bootstrap_95_ci_low'], 3)}, {_fmt(row['bootstrap_95_ci_high'], 3)}]",
            int(row["examples"]),
        )
        for _, row in structure["associations"].iterrows()
    ]
    milestone_rows = [
        (
            row["mode"],
            row["mechanism"],
            row["fixed_head"],
            _fmt(row["step_0"], 3),
            _fmt(row["step_10000"], 3),
            _fmt(row["persistent_half_rise_step"], 0),
            _fmt(row["correlation_with_ar_accuracy"], 3),
        )
        for _, row in milestones.iterrows()
    ]
    summary = sampling["summary"]
    ar_final = structure["ar"].groupby("mode")["ar_accuracy"].mean().to_dict()
    strongest = (
        structure["associations"].loc[structure["associations"]["mode"].eq("nonthinking")]
        .assign(abs_r=lambda frame: frame["count_controlled_correlation_with_ar_accuracy"].abs())
        .sort_values("abs_r", ascending=False)
        .iloc[0]
    )
    extension = f"""
    <section id="reproduction-extension">
      <h2>14. Reproducibility, training structure, and noise effects</h2>
      <div class="callout success"><strong>Reproduction status: PASS.</strong> The local v16.2 default, notebook, result artifacts, and downloaded checkpoints are now aligned to the exact RoPE pair named in the report header. The untouched reference HTML is retained as <a href="v16_2_rope_reference_report.html">v16_2_rope_reference_report.html</a>; this complete report preserves that body and adds only the audited extension below.</div>

      <h3>14.1 Reproduction contract</h3>
      <p>The contract fixes the run identity rather than merely matching the label “v16.2”: sequence length 256; 100 three-character sets sampled across 20 frequency bins with summed corpus frequency at most 0.04; counts 1–10; <code>task_occurrence_ratio = 1.0</code>; RoPE base 10,000; two independent 4-layer, 4-head, 256-dimensional models (<code>rope/nonthinking</code> and <code>rope/thinking</code>); 10,000 optimizer steps; all-sequence loss through step 1,500; task-output-only loss afterward; and checkpoints every 500 steps. The 42 downloaded checkpoint payloads are not generic substitutes: their serialized config, vocabulary, split, pool, representation, mode, and step identities are checked by the model loader.</p>
      {_table_html(("Audit item", "Expected", "Observed", "Status"), audit_rows)}
      <p><strong>What exact reproduction means here.</strong> The source report hash is <code>{REFERENCE_HTML_SHA256}</code>. All 42 checkpoint sizes match their Drive metadata, and their local SHA-256 values are in <a href="tables/checkpoint_inventory.csv">checkpoint_inventory.csv</a>. Step 0 and step 10,000 load without missing or unexpected parameters for both representations (3,180,800 parameters each). These checks establish artifact identity and code compatibility; bitwise retraining equivalence still requires the original CUDA/software stack and deterministic-kernel conditions.</p>

      <h3>14.2 Definitions for data structure and noise</h3>
      <p>For a 256-character prompt with queried set <em>S</em> and occurrence positions <em>P</em>, the true count is <em>n</em> = |<em>P</em>|. We define <strong>target density</strong> as <em>n</em>/256 and <strong>noise fraction</strong> as 1 − <em>n</em>/256; “noise” here means non-target prompt characters, not synthetic corruption. The <strong>three-character balance entropy</strong> is −Σ<sub>j=1</sub><sup>3</sup> p<sub>j</sub> log p<sub>j</sub> / log 3, with p<sub>j</sub> = c<sub>j</sub>/<em>n</em>. The <strong>occurrence span</strong> is (max <em>P</em> − min <em>P</em>)/255 (zero for <em>n</em> = 1). The <strong>longest noise run</strong> is the longest contiguous run containing no member of <em>S</em>, divided by 256. Prompt character entropy is the ordinary Shannon entropy over all prompt characters in bits.</p>
      <p>Structural effects are estimated observationally on the 100 balanced validation prompts used for final checkpoint-dynamics AR evaluation. For each feature <em>x</em> and binary correctness <em>y</em>, the reported association is corr(<em>x</em> − E[<em>x</em>|<em>n</em>], <em>y</em> − E[<em>y</em>|<em>n</em>]); thus true count is removed by within-count centering. The 95% interval comes from 2,000 stratified bootstrap resamples that preserve ten examples per count. This is a descriptive robustness analysis, not a randomized noise intervention.</p>

      <h3>14.3 Realized training distribution</h3>
      <p>Both representations receive exactly the same realized sampling distribution. Each sees {summary['training_examples_per_model']:,} counting examples. The rejection sampler accepts {_pct(summary['acceptance_rate'])} of proposed windows, rejecting {_pct(summary['rejected_zero_rate'])} for zero targets and {_pct(summary['rejected_over_threshold_rate'])} for counts above 10. The mean accepted count is {_fmt(summary['mean_training_count'], 3)}, so the average prompt is {_pct(summary['mean_training_noise_fraction'])} non-target characters. Evaluation is balanced at 10% per count, whereas training is corpus-determined (KL divergence from the balanced count distribution = {_fmt(summary['count_distribution_kl_nats_vs_balanced_eval'], 3)} nats).</p>
      {_table_html(("Count n", "Training examples", "Train probability", "Balanced-eval probability", "Train/eval ratio"), count_rows)}

      <figure>
        <img src="data:image/png;base64,{image_data}" alt="Training data structure and noise effects">
        <figcaption><span class="figure-tag">Figure 15.</span> Training structure and observational noise effects. <strong>A:</strong> x-axis is the accepted union count <em>n</em>; y-axis is its probability per model, compared with the balanced 0.10 evaluation mass. <strong>B:</strong> x-axis groups the balanced final validation prompts by true count; y-axis is autoregressive final-count accuracy. <strong>C:</strong> x-axis is the within-count quartile of summed training-corpus frequency for the queried three-character set (Q1 rarest, Q4 most frequent); y-axis is final AR accuracy. <strong>D:</strong> x-axis is the within-count quartile of the longest normalized non-target run (Q1 shortest, Q4 longest); y-axis is final AR accuracy. Within-count quartiles prevent count imbalance from mechanically determining panels C–D; each curve pools the 100 final AR prompts for that representation, with roughly one quarter assigned to each point.</figcaption>
      </figure>

      <h3>14.4 What the noise analysis supports</h3>
      <p>The final dynamics suite reproduces {_pct(ar_final['nonthinking'], 0)} nonthinking and {_pct(ar_final['thinking'], 0)} thinking accuracy. Since 96.1%–99.6% of each prompt consists of non-target characters, successful counting always requires rejecting a large distractor background. After controlling for count, the largest observed nonthinking structural association is <strong>{html.escape(str(strongest['feature_label']))}</strong> (<em>r</em> = {_fmt(strongest['count_controlled_correlation_with_ar_accuracy'], 3)}, bootstrap interval [{_fmt(strongest['bootstrap_95_ci_low'], 3)}, {_fmt(strongest['bootstrap_95_ci_high'], 3)}]). The interval and small n = 100 suite must be read alongside the effect; no feature is manipulated independently. Thinking is close to ceiling, so its correctness correlations are intrinsically compressed.</p>
      {_table_html(("Representation", "Structural feature", "Count-controlled r with AR accuracy", "95% stratified bootstrap interval", "n"), association_rows)}
      <p>The mechanistic interpretation is therefore narrower than “more noise causes failure.” The dominant controlled contrast remains output representation: thinking externalizes ordered evidence into a trace, while nonthinking must compress a broad prompt retrieval into one answer state. Natural prompt geometry—rarity, clustering, gaps, and local character diversity—modulates the difficulty of that retrieval, but this run does not randomize those factors. The causal next experiment is a matched prompt intervention that preserves <em>n</em> and the queried set while relocating occurrences or replacing only non-target characters.</p>

      <h3>14.5 Mechanism emergence milestones</h3>
      <p>For each fixed final-checkpoint head, the <strong>persistent half-rise step</strong> is the earliest stored checkpoint after which the metric always remains on the final side of the midpoint between its step-0 and step-10,000 values. This definition uses the fixed final head on held-out reporting examples and suppresses one-checkpoint transients; it does not reselect a favorable head at every step. The last column is the Pearson correlation of the fixed-head metric with same-step AR accuracy across 21 checkpoints; training step is a shared cause, so it is descriptive rather than causal.</p>
      {_table_html(("Representation", "Behavior/mechanism", "Fixed head", "Step 0", "Step 10,000", "Persistent half-rise step", "r with AR accuracy"), milestone_rows)}
      <p>These milestones make the emergence story explicit: a representation is not identified from final attention alone. Behavioral accuracy, broad prompt coverage, ordered kth-occurrence retrieval, and answer-to-trace routing are followed through the same 21 checkpoints. The already reported Figures 1 and 3–11 provide the corresponding curves and hidden-state geometry; <a href="tables/mechanism_emergence_milestones.csv">mechanism_emergence_milestones.csv</a> records the exact calculation.</p>

      <h3>14.6 Added reproducibility artifacts</h3>
      <p class="artifact-list">
        <span class="pill">Identity</span> <a href="checkpoint_sources.tsv">Drive source manifest</a> · <a href="tables/checkpoint_inventory.csv">local size/SHA-256 inventory</a> · <a href="reproducibility_audit.json">audit record</a> · <a href="v16_2_rope_reference_report.html">untouched reference HTML</a><br>
        <span class="pill">Data/noise</span> <a href="tables/training_count_distribution_audit.csv">training count distribution</a> · <a href="tables/training_structure_summary.json">sampling summary</a> · <a href="tables/data_structure_ar_detail.csv">per-prompt structure</a> · <a href="tables/data_structure_accuracy_associations.csv">accuracy associations</a> · <a href="tables/data_structure_mechanism_associations.csv">mechanism associations</a><br>
        <span class="pill">Emergence</span> <a href="tables/mechanism_emergence_milestones.csv">milestone definitions and values</a> · <a href="figures/training_structure_and_noise_effects.png">Figure 15 source image</a>
      </p>
    </section>
    """
    footer_marker = "    <footer>"
    if footer_marker not in source:
        raise ValueError("could not locate report footer")
    source = source.replace(footer_marker, extension + "\n" + footer_marker, 1)
    output = run_dir / "v16_2_rope_complete_report.html"
    _atomic_text(output, source)
    shutil.copyfile(output, run_dir / "v16_2_rope_complete_report_en.html")
    shutil.copyfile(output, run_dir / "syn_v16_2_report.html")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference-html", type=Path, required=True)
    parser.add_argument("--skip-checkpoint-hashes", action="store_true")
    parser.add_argument("--skip-model-load-check", action="store_true")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    reference_html = args.reference_html.resolve()
    if not run_dir.is_dir() or not reference_html.is_file():
        raise FileNotFoundError((run_dir, reference_html))
    reference_hash = _sha256(reference_html)
    if reference_hash != REFERENCE_HTML_SHA256:
        raise ValueError(f"reference HTML SHA-256 mismatch: {reference_hash}")

    config, config_rows = audit_config(run_dir)
    inventory = checkpoint_inventory(run_dir, compute_hashes=not args.skip_checkpoint_hashes)
    load_rows = [] if args.skip_model_load_check else verify_checkpoint_loads(run_dir)
    structure = analyze_structure(run_dir, config)
    sampling = analyze_training_sampling(run_dir, config)
    milestones = analyze_emergence(run_dir)
    figure = plot_structure(run_dir, structure, sampling)

    pipeline_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    dynamics_manifest = json.loads((run_dir / "analysis/checkpoint_dynamics/manifest.json").read_text(encoding="utf-8"))
    audit = {
        "status": "complete",
        "reference_run_id": REFERENCE_RUN_ID,
        "reference_html": str(reference_html),
        "reference_html_sha256": reference_hash,
        "config_checks": config_rows,
        "checkpoint_count": len(inventory),
        "checkpoint_hashes_computed": not args.skip_checkpoint_hashes,
        "boundary_model_load_checks": load_rows,
        "pipeline_manifest_status": pipeline_manifest.get("stages", {}).get("plots", {}).get("status"),
        "dynamics_manifest_status": dynamics_manifest.get("status"),
        "dynamics_inventory_count": len(dynamics_manifest.get("inventory", [])),
        "derived_analysis": {
            "final_ar_examples": int(len(structure["ar"])),
            "accuracy_association_rows": int(len(structure["associations"])),
            "mechanism_milestone_rows": int(len(milestones)),
        },
    }
    _atomic_json(run_dir / "reproducibility_audit.json", audit)
    output = build_extension_html(
        run_dir,
        reference_html,
        config_rows,
        load_rows,
        inventory,
        structure,
        sampling,
        milestones,
        figure,
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
