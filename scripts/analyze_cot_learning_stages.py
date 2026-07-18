from __future__ import annotations

import argparse
import base64
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_counting_v11.config import config_from_dict  # noqa: E402
from synthetic_counting_v11.data import (  # noqa: E402
    Vocab,
    balanced_examples,
    collate,
    render,
)
from synthetic_counting_v11.model import build_model  # noqa: E402


VERSIONS = ("v11", "v12", "v13", "v14")
COLORS = {
    "final": "#2563eb",
    "index": "#7c3aed",
    "marker": "#16a34a",
    "ar_marker": "#0f766e",
    "exact": "#dc2626",
    "muted": "#64748b",
}


@dataclass(frozen=True)
class Trajectory:
    version: str
    run_dir: Path
    position_encoding: str

    @property
    def key(self) -> str:
        return f"{self.version}:{self.position_encoding}"

    @property
    def label(self) -> str:
        suffix = self.position_encoding.upper()
        return f"{self.version.upper()} {suffix}"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def discover_runs(results_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for version in VERSIONS:
        candidates = sorted(
            path
            for path in results_root.glob(f"{version}_main_seed1234_*")
            if path.is_dir() and (path / "config.json").exists()
        )
        if not candidates:
            raise FileNotFoundError(f"No formal {version} run found under {results_root}")
        found[version] = candidates[-1]
    return found


def trajectories_for_runs(runs: dict[str, Path]) -> list[Trajectory]:
    result: list[Trajectory] = []
    for version, run_dir in runs.items():
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        for position in cfg["position_encodings"]:
            result.append(Trajectory(version, run_dir, str(position)))
    return result


def aggregate_thinking_tables(trajectory: Trajectory) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = trajectory.run_dir / "tables"
    tf = read_csv(tables / "eval_by_bin.csv")
    losses = read_csv(tables / "eval_losses.csv")
    ar = read_csv(tables / "autoregressive_by_bin.csv")

    selector = (tf["position_encoding"] == trajectory.position_encoding) & (tf["mode"] == "thinking")
    tf = (
        tf.loc[selector]
        .groupby("step", as_index=False)[
            ["tf_final_accuracy", "tf_trace_marker_accuracy", "tf_trace_index_accuracy"]
        ]
        .mean()
    )
    selector = (losses["position_encoding"] == trajectory.position_encoding) & (
        losses["mode"] == "thinking"
    )
    losses = losses.loc[selector].sort_values("step").reset_index(drop=True)
    selector = (ar["position_encoding"] == trajectory.position_encoding) & (ar["mode"] == "thinking")
    ar = (
        ar.loc[selector]
        .groupby("step", as_index=False)[
            ["ar_final_accuracy", "trace_exact", "trace_marker_recall"]
        ]
        .mean()
    )
    return tf, losses, ar


def first_stable_step(frame: pd.DataFrame, column: str, threshold: float, floor: float | None = None) -> float:
    if frame.empty or column not in frame:
        return math.nan
    ordered = frame.sort_values("step")
    values = ordered[column].to_numpy(dtype=float)
    steps = ordered["step"].to_numpy(dtype=float)
    allowed_floor = threshold if floor is None else floor
    for index, value in enumerate(values):
        tail = values[index:]
        if value >= threshold and np.nanmin(tail) >= allowed_floor:
            return float(steps[index])
    return math.nan


def two_segment_breakpoint(frame: pd.DataFrame, column: str, min_points: int = 4) -> float:
    """Return the split step minimizing two independent linear-regression SSEs."""

    if frame.empty or column not in frame:
        return math.nan
    ordered = frame[["step", column]].dropna().sort_values("step")
    x = ordered["step"].to_numpy(dtype=float)
    y = ordered[column].to_numpy(dtype=float)
    if len(x) < 2 * min_points:
        return math.nan
    x = (x - x.min()) / max(float(x.max() - x.min()), 1.0)

    def segment_sse(left: int, right: int) -> float:
        xs = x[left:right]
        ys = y[left:right]
        design = np.column_stack((np.ones(len(xs)), xs))
        beta = np.linalg.lstsq(design, ys, rcond=None)[0]
        return float(np.square(ys - design @ beta).sum())

    candidates: list[tuple[float, int]] = []
    for split in range(min_points, len(x) - min_points + 1):
        candidates.append((segment_sse(0, split) + segment_sse(split, len(x)), split))
    _, split = min(candidates)
    return float(ordered.iloc[split]["step"])


def stage_summary(trajectories: Iterable[Trajectory]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        tf, _, ar = aggregate_thinking_tables(trajectory)
        rows.append(
            {
                "trajectory": trajectory.label,
                "version": trajectory.version,
                "position_encoding": trajectory.position_encoding,
                "final_count_stable_99_step": first_stable_step(tf, "tf_final_accuracy", 0.99, 0.95),
                "trace_index_stable_99_step": first_stable_step(tf, "tf_trace_index_accuracy", 0.99, 0.95),
                "marker_piecewise_break_step": two_segment_breakpoint(tf, "tf_trace_marker_accuracy"),
                "marker_20_step": first_stable_step(tf, "tf_trace_marker_accuracy", 0.20, 0.18),
                "marker_50_step": first_stable_step(tf, "tf_trace_marker_accuracy", 0.50, 0.45),
                "marker_90_step": first_stable_step(tf, "tf_trace_marker_accuracy", 0.90, 0.85),
                "ar_exact_50_step": first_stable_step(ar, "trace_exact", 0.50, 0.45),
                "ar_exact_90_step": first_stable_step(ar, "trace_exact", 0.90, 0.85),
                "final_tf_marker_accuracy": float(tf.iloc[-1]["tf_trace_marker_accuracy"]),
                "final_ar_marker_recall": float(ar.iloc[-1]["trace_marker_recall"]),
                "final_trace_exact": float(ar.iloc[-1]["trace_exact"]),
            }
        )
    return pd.DataFrame(rows)


def expected_exact_from_local(marker_recall: float, count_bin: str) -> float:
    lo, hi = (int(part) for part in str(count_bin).split("-"))
    values = [float(marker_recall) ** count for count in range(lo, hi + 1)]
    return float(np.mean(values))


def local_exact_decomposition(trajectories: Iterable[Trajectory]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for trajectory in trajectories:
        frame = read_csv(trajectory.run_dir / "tables" / "autoregressive_by_bin.csv")
        frame = frame[
            (frame["position_encoding"] == trajectory.position_encoding)
            & (frame["mode"] == "thinking")
        ].copy()
        frame["expected_exact_if_independent"] = frame.apply(
            lambda row: expected_exact_from_local(row["trace_marker_recall"], row["count_bin"]), axis=1
        )
        frame["trajectory"] = trajectory.label
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _checkpoint_steps(run_dir: Path, position: str) -> list[tuple[int, Path]]:
    root = run_dir / "checkpoints" / position / "thinking"
    result: list[tuple[int, Path]] = []
    for directory in root.glob("step_*"):
        path = directory / "checkpoint.pt"
        if not path.exists():
            continue
        try:
            step = int(directory.name.removeprefix("step_"))
        except ValueError:
            continue
        result.append((step, path))
    return sorted(result)


def _normalized_entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if len(values) <= 1 or total <= 1e-12:
        return 0.0
    probabilities = values / total
    return float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum() / math.log(len(values)))


def _geometry_metrics(vectors: list[np.ndarray], labels: list[int]) -> dict[str, float]:
    if not vectors:
        return {
            "pc1_label_r2": math.nan,
            "pc1_adjacent_consistency": math.nan,
            "effective_dimension": math.nan,
            "pc1_variance": math.nan,
            "pc1_to_pc6_variance": math.nan,
        }
    frame = pd.DataFrame({"label": labels, "index": range(len(labels))})
    matrix = np.stack(vectors)
    centroids: list[np.ndarray] = []
    unique_labels = sorted(int(value) for value in frame["label"].unique())
    for label in unique_labels:
        indices = frame.loc[frame["label"] == label, "index"].to_numpy(dtype=int)
        centroids.append(matrix[indices].mean(axis=0))
    values = np.stack(centroids)
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    ratios = variance / max(float(variance.sum()), 1e-12)
    pc1 = centered @ vh[0]
    label_values = np.asarray(unique_labels, dtype=float)
    correlation = np.corrcoef(pc1, label_values)[0, 1] if len(pc1) > 1 else math.nan
    deltas = np.diff(values, axis=0)
    mean_delta = deltas.mean(axis=0) if len(deltas) else np.zeros(values.shape[1])
    denominator = np.linalg.norm(deltas, axis=1) * max(float(np.linalg.norm(mean_delta)), 1e-12)
    consistency = np.divide(deltas @ mean_delta, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    effective = float(variance.sum() ** 2 / max(float(np.square(variance).sum()), 1e-12))
    return {
        "pc1_label_r2": float(correlation**2) if np.isfinite(correlation) else math.nan,
        "pc1_adjacent_consistency": float(np.mean(consistency)) if len(consistency) else math.nan,
        "effective_dimension": effective,
        "pc1_variance": float(ratios[0]),
        "pc1_to_pc6_variance": float(ratios[:6].sum()),
    }


def checkpoint_diagnostics(
    trajectories: Iterable[Trajectory],
    output_dir: Path,
    *,
    examples_per_count: int,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attention_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        cfg = config_from_dict(json.loads((trajectory.run_dir / "config.json").read_text(encoding="utf-8")))
        cfg = type(cfg)(**{**cfg.__dict__, "device": device})
        vocab = Vocab.load(trajectory.run_dir / "vocab.json")
        examples = balanced_examples(
            cfg,
            vocab,
            examples_per_count,
            cfg.seed + 330_000,
        )
        rendered = [render(example, vocab, "thinking") for example in examples]
        checkpoints = _checkpoint_steps(trajectory.run_dir, trajectory.position_encoding)
        previous_state: dict[str, torch.Tensor] | None = None
        print(f"[checkpoint diagnostics] {trajectory.label}: {len(checkpoints)} checkpoints", flush=True)
        for step, checkpoint_path in checkpoints:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_model(cfg, vocab, trajectory.position_encoding, device)
            model.load_state_dict(payload["model_state_dict"])
            model.eval()

            current_state = {name: value.detach().float().cpu() for name, value in model.state_dict().items()}
            if previous_state is not None:
                module_groups = {
                    "embedding": lambda name: "embedding" in name,
                    "attention": lambda name: ".attention." in name,
                    "mlp": lambda name: ".mlp." in name,
                    "normalization": lambda name: "ln_" in name or "norm" in name,
                }
                for group_name, predicate in module_groups.items():
                    names = [name for name in current_state if predicate(name) and name in previous_state]
                    delta_sq = sum(float(torch.square(current_state[name] - previous_state[name]).sum()) for name in names)
                    base_sq = sum(float(torch.square(previous_state[name]).sum()) for name in names)
                    drift_rows.append(
                        {
                            "trajectory": trajectory.label,
                            "step": step,
                            "module_group": group_name,
                            "relative_update_norm": math.sqrt(delta_sq / max(base_sq, 1e-24)),
                        }
                    )
            previous_state = current_state

            attention_accumulator: dict[tuple[int, int], dict[str, list[float]]] = {}
            state_accumulator: dict[tuple[int, str], tuple[list[np.ndarray], list[int]]] = {}
            batch_size = 4 if cfg.seq_len >= 512 else 8
            with torch.no_grad():
                for start in range(0, len(rendered), batch_size):
                    chunk = rendered[start : start + batch_size]
                    ids, _, mask = collate(chunk, vocab, device)
                    output = model(
                        input_ids=ids,
                        attention_mask=mask,
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                    attentions = output.attentions or ()
                    hidden_states = output.hidden_states or ()
                    for row_index, item in enumerate(chunk):
                        for layer_index, layer_attention in enumerate(attentions, start=1):
                            matrix = layer_attention[row_index].detach().float().cpu().numpy()
                            for head in range(matrix.shape[0]):
                                key = (layer_index, head)
                                bucket = attention_accumulator.setdefault(
                                    key,
                                    {
                                        "k_mass": [],
                                        "k_top1": [],
                                        "diagonal": [],
                                        "trace_readout_mass": [],
                                        "final_broad_score": [],
                                    },
                                )
                                final_weights = matrix[head, item.spans.ans_pos]
                                needle_weights = final_weights[item.prompt_needle_positions]
                                bucket["final_broad_score"].append(
                                    float(needle_weights.sum()) * _normalized_entropy(needle_weights)
                                )
                                bucket["trace_readout_mass"].append(
                                    float(final_weights[item.spans.trace_marker_positions].sum())
                                )
                                for k, query_position in enumerate(item.spans.trace_index_positions):
                                    weights = matrix[head, query_position]
                                    prompt_weights = weights[item.prompt_needle_positions]
                                    correct = float(prompt_weights[k])
                                    bucket["k_mass"].append(correct)
                                    bucket["k_top1"].append(float(int(np.argmax(prompt_weights) == k)))
                                    bucket["diagonal"].append(
                                        correct / max(float(prompt_weights.sum()), 1e-12)
                                    )
                        for layer_index, hidden in enumerate(hidden_states):
                            final_bucket = state_accumulator.setdefault(
                                (layer_index, "final_answer"), ([], [])
                            )
                            final_bucket[0].append(
                                hidden[row_index, item.spans.ans_pos].detach().float().cpu().numpy()
                            )
                            final_bucket[1].append(int(item.count))
                            trace_bucket = state_accumulator.setdefault(
                                (layer_index, "trace_progress"), ([], [])
                            )
                            for k, position in enumerate(item.spans.trace_marker_positions, start=1):
                                trace_bucket[0].append(
                                    hidden[row_index, position].detach().float().cpu().numpy()
                                )
                                trace_bucket[1].append(k)

            for (layer, head), bucket in attention_accumulator.items():
                attention_rows.append(
                    {
                        "trajectory": trajectory.label,
                        "step": step,
                        "layer": layer,
                        "head": head,
                        **{name: float(np.mean(values)) for name, values in bucket.items()},
                    }
                )
            for (layer, site), (vectors, labels) in state_accumulator.items():
                geometry_rows.append(
                    {
                        "trajectory": trajectory.label,
                        "step": step,
                        "layer": layer,
                        "site": site,
                        **_geometry_metrics(vectors, labels),
                    }
                )
            del model, payload

    attention = pd.DataFrame(attention_rows)
    geometry = pd.DataFrame(geometry_rows)
    drift = pd.DataFrame(drift_rows)
    attention.to_csv(output_dir / "checkpoint_attention.csv", index=False)
    geometry.to_csv(output_dir / "checkpoint_state_geometry.csv", index=False)
    drift.to_csv(output_dir / "checkpoint_parameter_drift.csv", index=False)
    return attention, geometry, drift


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)


def plot_stage_overview(trajectories: list[Trajectory], stages: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle("CoT learning has an early scaffold stage and a delayed content-binding stage", fontsize=18, weight="bold")
    for ax, trajectory in zip(axes.flat, trajectories):
        tf, _, ar = aggregate_thinking_tables(trajectory)
        stage_row = stages[stages["trajectory"] == trajectory.label].iloc[0]
        breakpoint = stage_row["marker_piecewise_break_step"]
        if np.isfinite(breakpoint):
            ax.axvspan(0, breakpoint, color="#dbeafe", alpha=0.6, label="Stage A: scaffold")
            ax.axvspan(breakpoint, 10_000, color="#dcfce7", alpha=0.45, label="Stage B: binding")
            ax.axvline(breakpoint, color="#475569", linestyle=":", linewidth=1.3)
        ax.plot(tf.step, tf.tf_final_accuracy, color=COLORS["final"], label="TF final count", linewidth=2)
        ax.plot(tf.step, tf.tf_trace_index_accuracy, color=COLORS["index"], label="TF trace index", linewidth=1.8)
        ax.plot(tf.step, tf.tf_trace_marker_accuracy, color=COLORS["marker"], label="TF marker identity", linewidth=2.4)
        ax.plot(ar.step, ar.trace_exact, color=COLORS["exact"], label="AR exact trace", linewidth=2, marker="o", markersize=3)
        ax.set_title(trajectory.label)
        ax.set_ylim(-0.03, 1.04)
        ax.set_xlabel("training step")
        ax.set_ylabel("accuracy / recall")
        _style_axis(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_component_losses(trajectories: list[Trajectory], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.4), sharex=True, constrained_layout=True)
    fig.suptitle("Loss decomposition: easy syntax/count targets fall before marker binding", fontsize=18, weight="bold")
    columns = [
        ("eval_final_count_loss", "final count", COLORS["final"]),
        ("eval_trace_index_loss", "trace index", COLORS["index"]),
        ("eval_trace_marker_loss", "trace marker", COLORS["marker"]),
        ("eval_think_close_loss", "close trace", "#ea580c"),
        ("eval_eos_loss", "EOS", "#64748b"),
    ]
    for ax, trajectory in zip(axes.flat, trajectories):
        _, losses, _ = aggregate_thinking_tables(trajectory)
        for column, label, color in columns:
            if column in losses and losses[column].notna().any():
                ax.plot(losses.step, losses[column].clip(lower=1e-4), label=label, color=color, linewidth=2)
        ax.set_yscale("log")
        ax.set_title(trajectory.label)
        ax.set_xlabel("training step")
        ax.set_ylabel("teacher-forced token CE (log scale)")
        _style_axis(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_threshold_timeline(stages: pd.DataFrame, path: Path) -> None:
    events = [
        ("final_count_stable_99_step", "final count stable >=99%", COLORS["final"]),
        ("marker_piecewise_break_step", "marker-learning breakpoint", "#0f766e"),
        ("marker_50_step", "marker accuracy stable >=50%", COLORS["marker"]),
        ("ar_exact_50_step", "AR exact trace stable >=50%", COLORS["exact"]),
    ]
    fig, ax = plt.subplots(figsize=(13, 5.6), constrained_layout=True)
    y = np.arange(len(stages))
    offsets = np.linspace(-0.24, 0.24, len(events))
    for offset, (column, label, color) in zip(offsets, events):
        values = stages[column].to_numpy(dtype=float)
        ax.scatter(values, y + offset, s=72, label=label, color=color, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y, stages["trajectory"])
    ax.set_xlim(0, 10_300)
    ax.set_xlabel("first stable step / piecewise breakpoint")
    ax.set_title("Stage timing: final answer and index scaffold precede marker binding", fontsize=17, weight="bold")
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)
    _style_axis(ax)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_local_exact(decomposition: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.2), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle("Exact-trace accuracy is a multiplicative sequence metric", fontsize=18, weight="bold")
    for ax, (trajectory, frame) in zip(axes.flat, decomposition.groupby("trajectory", sort=False)):
        for count_bin, part in frame.groupby("count_bin"):
            ax.plot(part.trace_marker_recall, part.trace_exact, marker="o", linewidth=1.7, label=f"observed {count_bin}")
            ax.plot(
                part.trace_marker_recall,
                part.expected_exact_if_independent,
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label=f"p^n approximation {count_bin}",
            )
        ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle=":", linewidth=1)
        ax.set_title(trajectory)
        ax.set_xlabel("AR marker recall p")
        ax.set_ylabel("whole-trace exact match")
        _style_axis(ax)
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_attention_emergence(attention: pd.DataFrame, path: Path) -> None:
    trajectories = list(dict.fromkeys(attention["trajectory"]))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.4), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle("Checkpoint tracking: when does a stable k-to-k retrieval head emerge?", fontsize=18, weight="bold")
    for ax, trajectory in zip(axes.flat, trajectories):
        frame = attention[attention["trajectory"] == trajectory]
        final_step = frame.step.max()
        final = frame[frame.step == final_step].sort_values("k_mass", ascending=False).iloc[0]
        final_layer = int(final["layer"])
        final_head = int(final["head"])
        selected = frame[
            (frame["layer"] == final_layer) & (frame["head"] == final_head)
        ].sort_values("step")
        best = frame.groupby("step", as_index=False).agg(
            best_k_mass=("k_mass", "max"), best_top1=("k_top1", "max")
        )
        ax.plot(
            selected["step"],
            selected["k_mass"],
            color="#16a34a",
            marker="o",
            label=f"fixed final head L{final_layer}H{final_head} mass",
        )
        ax.plot(selected.step, selected.k_top1, color="#2563eb", marker="o", label="fixed final head top-1")
        ax.plot(best.step, best.best_k_mass, color="#86efac", linestyle="--", label="best head at each step: mass")
        ax.set_title(trajectory)
        ax.set_xlabel("checkpoint step")
        ax.set_ylabel("attention metric")
        ax.set_ylim(-0.03, 1.03)
        _style_axis(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_geometry_emergence(geometry: pd.DataFrame, path: Path) -> None:
    trajectories = list(dict.fromkeys(geometry["trajectory"]))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.4), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle("Checkpoint tracking: count/progress geometry appears at different semantic sites", fontsize=18, weight="bold")
    for ax, trajectory in zip(axes.flat, trajectories):
        frame = geometry[geometry["trajectory"] == trajectory]
        final_layer = frame.layer.max()
        for site, color, label in (
            ("final_answer", "#2563eb", "final-answer count geometry"),
            ("trace_progress", "#16a34a", "trace-progress geometry"),
        ):
            selected = frame[(frame.site == site) & (frame.layer == final_layer)].sort_values("step")
            ax.plot(selected.step, selected.pc1_label_r2, marker="o", color=color, label=label)
        ax.set_title(trajectory)
        ax.set_xlabel("checkpoint step")
        ax.set_ylabel("PC1 vs count/progress R-squared")
        ax.set_ylim(-0.03, 1.03)
        _style_axis(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_parameter_drift(drift: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6), constrained_layout=True)
    grouped = drift.groupby(["module_group", "step"], as_index=False).relative_update_norm.mean()
    for group, frame in grouped.groupby("module_group"):
        ax.plot(frame.step, frame.relative_update_norm, marker="o", linewidth=2, label=group)
    ax.set_title("Mean relative parameter update between consecutive checkpoints", fontsize=17, weight="bold")
    ax.set_xlabel("ending checkpoint step")
    ax.set_ylabel("||theta_t - theta_(t-1000)|| / ||theta_(t-1000)||")
    ax.legend(frameon=False, ncol=4)
    _style_axis(ax)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _image_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _table_html(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if "step" in column:
            display[column] = display[column].map(lambda value: "未达到" if pd.isna(value) else f"{int(value):,}")
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.3f}")
    return display.to_html(index=False, border=0, classes="data-table")


def _figure(path: Path, title: str, caption: str) -> str:
    return (
        '<figure class="figure">'
        f"<h3>{html.escape(title)}</h3>"
        f'<img src="{_image_uri(path)}" alt="{html.escape(title)}">'
        f"<figcaption>{caption}</figcaption>"
        "</figure>"
    )


def write_report(
    output_dir: Path,
    stages: pd.DataFrame,
    figures: dict[str, Path],
    attention: pd.DataFrame,
    geometry: pd.DataFrame,
) -> Path:
    strongest_marker = stages.sort_values("final_tf_marker_accuracy", ascending=False).iloc[0]
    weakest_marker = stages.sort_values("final_tf_marker_accuracy", ascending=True).iloc[0]
    final_head_rows: list[dict[str, Any]] = []
    if not attention.empty:
        for trajectory, frame in attention.groupby("trajectory"):
            final = frame[frame.step == frame.step.max()].sort_values("k_mass", ascending=False).iloc[0]
            final_head_rows.append(
                {
                    "trajectory": trajectory,
                    "best final k-to-k head": f"L{int(final['layer'])}H{int(final['head'])}",
                    "k-to-k raw mass": float(final.k_mass),
                    "needle-subset top-1": float(final.k_top1),
                    "diagonal dominance": float(final.diagonal),
                }
            )
    head_table = pd.DataFrame(final_head_rows)

    css = """
    :root { --ink:#172033; --muted:#526077; --line:#dbe3ee; --blue:#2563eb; --green:#15803d; --red:#b91c1c; }
    * { box-sizing:border-box; }
    body { margin:0; background:#edf2f7; color:var(--ink); font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.75; }
    main { width:min(1320px,calc(100% - 36px)); margin:24px auto 60px; background:white; padding:42px 54px 64px; box-shadow:0 10px 35px rgba(15,23,42,.08); }
    h1 { font-size:2.25rem; line-height:1.2; margin:.15rem 0 .8rem; }
    h2 { margin:2.4rem 0 1rem; padding-top:1.1rem; border-top:1px solid var(--line); font-size:1.55rem; }
    h3 { font-size:1.13rem; margin:.15rem 0 1rem; }
    p,li { font-size:1.02rem; }
    .lead { color:var(--muted); font-size:1.1rem; }
    .callout { padding:18px 22px; margin:20px 0; border-left:5px solid var(--blue); background:#eff6ff; border-radius:6px; }
    .callout.green { border-color:var(--green); background:#f0fdf4; }
    .callout.warn { border-color:#d97706; background:#fffbeb; }
    .stage-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin:20px 0; }
    .stage { border:1px solid var(--line); border-radius:8px; padding:20px; }
    .stage.a { background:#eff6ff; } .stage.b { background:#f0fdf4; }
    .formula { font-family:"Cambria Math","STIX Two Math",serif; background:#f8fafc; border:1px solid var(--line); border-radius:6px; padding:12px 16px; text-align:center; font-size:1.15rem; }
    .figure { margin:26px 0; border:1px solid var(--line); border-radius:8px; padding:18px; background:#fff; }
    .figure img { width:100%; height:auto; display:block; }
    figcaption { color:var(--muted); margin-top:12px; font-size:.96rem; }
    .data-table { width:100%; border-collapse:collapse; font-size:.92rem; margin:18px 0; }
    .data-table th,.data-table td { padding:9px 10px; border:1px solid var(--line); text-align:left; }
    .data-table th { background:#eaf0f8; }
    code { background:#edf2f7; padding:.12rem .35rem; border-radius:4px; }
    @media(max-width:850px){ main{width:100%;margin:0;padding:24px 18px}.stage-grid{grid-template-columns:1fr} }
    """
    body = f"""
    <h1>CoT 为什么呈现两个学习阶段？</h1>
    <p class="lead">对 v11–v14 共六条 thinking 训练轨迹做 loss 分解、first-passage / piecewise change-point、局部到序列误差分解，以及逐 checkpoint attention、hidden-state geometry 与参数漂移分析。</p>

    <div class="callout green"><strong>核心结论。</strong> 这里的“两阶段”不是同一个计数能力缓慢变好。Stage A 先学会低复杂度的<strong>输出骨架</strong>：数字 index 的确定性递增、最终 count、关闭 trace 与 EOS；此时 marker identity 仍约等于十类随机猜测的 10%。Stage B 才形成 prompt 位置与 marker identity 的<strong>内容绑定 / targeted retrieval</strong>。整条 trace exact-match 更晚，并不一定代表第三套机制，而主要是局部错误沿长度连乘后的阈值效应。</div>

    <div class="stage-grid">
      <section class="stage a"><h3>Stage A：algorithmic scaffold</h3><p>模型快速掌握“第几个”“最后答案是多少”“何时关闭 trace”等低熵规则。teacher-forced trace-index accuracy 很早接近 1，但 marker accuracy 仍约 0.1。这说明模型知道 trace 的计数语法，却不知道第 k 个 needle 的具体 marker identity。</p></section>
      <section class="stage b"><h3>Stage B：content binding / retrieval</h3><p>marker cross-entropy 开始持续下降，teacher-forced 与 autoregressive marker accuracy 同步上升，随后出现稳定的 k-to-k attention 与 trace-progress representation。位置编码和 haystack 结构强烈控制这一阶段的速度与终点。</p></section>
    </div>

    <h2>1. 指标和阶段判据</h2>
    <ul>
      <li><strong>TF final count accuracy：</strong>提供完整 gold trace，在 <code>&lt;Ans&gt;</code> 位置只检验最终数字；它测 readout，不测模型能否自己生成正确 trace。</li>
      <li><strong>TF trace-index accuracy：</strong>在 <code>&lt;Think&gt;</code> 或前一个 marker 后预测下一个数字 index。该任务几乎是确定性的 successor / syntax 学习。</li>
      <li><strong>TF marker identity accuracy：</strong>给定 gold 数字 index，在该位置预测匹配的 prompt marker；十种 marker 的 chance 为 0.1，直接测内容绑定。</li>
      <li><strong>AR marker recall / exact trace：</strong>模型自由生成。前者是局部 marker 命中率，后者要求整条数字-marker 序列完全正确。</li>
      <li><strong>piecewise breakpoint：</strong>在 marker-accuracy 曲线上枚举一个分割点，分别拟合前后两条直线，选择总平方误差最小的 step。它不是任意阈值，而是数据驱动的斜率变化点。</li>
    </ul>
    <div class="formula">expected exact trace under independent local errors = mean over n in bin of p<sup>n</sup></div>
    <p>其中 p 是该 checkpoint / count-bin 的 autoregressive marker recall，n 是 needle 数。该近似忽略 index、关闭符和错误相关性，只用来判断 exact-match 的晚出现是否可由误差连乘解释。</p>

    <h2>2. 两阶段的直接证据</h2>
    {_figure(figures['overview'], 'Figure 1. 六条 CoT 训练轨迹的阶段分解', '横轴是训练 step，纵轴是 accuracy/recall。蓝色阴影为 marker-learning breakpoint 之前的 Stage A，绿色阴影为之后的 Stage B。final count 与 trace index 普遍先饱和；marker identity 延迟数千 step 才离开 chance。')}
    {_figure(figures['losses'], 'Figure 2. 按 token 类型分解的 teacher-forced cross-entropy', '横轴是训练 step；纵轴为对数刻度 token CE。trace-index、final-count、think-close 与 EOS 先下降，而 trace-marker loss 长时间停留在接近 log(10) 的区域，随后才发生相变式下降。')}
    {_figure(figures['timeline'], 'Figure 3. 阶段事件的 first-passage 时间', '每一行是一条模型轨迹；点表示 final count 稳定达到 99%、marker 曲线 piecewise breakpoint、marker 稳定达到 50%、以及 autoregressive exact trace 稳定达到 50% 的时间。缺失点表示 10k step 内未达到。')}
    {_table_html(stages[['trajectory','final_count_stable_99_step','marker_piecewise_break_step','marker_50_step','marker_90_step','ar_exact_50_step','final_tf_marker_accuracy','final_trace_exact']])}

    <h2>3. 为什么看起来“还没收敛”</h2>
    <p>总 loss 被大量 trace-marker targets 主导。Stage A 结束时，最终 count 已经正确，数字 index 也几乎全对，但每个样本有 n 个 marker targets；只要 marker 仍接近 chance，总 loss 就不会接近 0。因而“final accuracy 已收敛”和“CoT objective 已收敛”并不矛盾。</p>
    {_figure(figures['local_exact'], 'Figure 4. 局部 marker recall 与整条 trace exact-match', '横轴是自由生成时单个 marker 的平均 recall p，纵轴是整条 trace exact-match。实线为观测值，虚线为 mean(p^n) 独立误差近似。长 trace 即使局部 recall 已很高，exact-match 仍会被误差连乘显著压低。')}
    <div class="callout warn"><strong>解释边界。</strong> exact-match 晚出现本身不能证明存在新的“第三阶段”。若 observed exact 与 p<sup>n</sup> 同量级，更简洁的解释是 Stage B 内部的局部检索持续抛光；只有当 exact 出现无法由局部准确率解释的突变时，才需要引入额外的 sequence-level coordination 假设。</div>

    <h2>4. checkpoint 机制追踪</h2>
    {_figure(figures['attention'], 'Figure 5. k-to-k retrieval head 的形成轨迹', '每个 checkpoint 都用同一批平衡样本。在最终 checkpoint 选 raw k-to-k mass 最大的固定 head，再向前追踪它；同时画每一步当时最强 head。横轴是 checkpoint step，纵轴分别为匹配 needle 的 raw attention mass 与 needle 子集内 top-1。')}
    {_table_html(head_table)}
    {_figure(figures['geometry'], 'Figure 6. final-count 与 trace-progress hidden-state geometry 的形成', '在每层 residual state 中，先按 count 或 trace step 求 centroid，再做 PCA；纵轴是 PC1 与 count/progress 数值的相关系数平方。它是描述性有序几何，不等于 causal count direction。')}
    {_figure(figures['drift'], 'Figure 7. checkpoint 间模块参数漂移', '横轴为后一 checkpoint 的 step，纵轴为相邻 1000-step checkpoint 间参数变化范数除以前一 checkpoint 参数范数；曲线先对六条轨迹取平均。它定位 optimizer 主要仍在更新 embedding、attention 还是 MLP，但不能单独提供机制因果性。')}

    <h2>5. 不同实验为什么第二阶段差很多</h2>
    <ul>
      <li><strong>{html.escape(str(strongest_marker.trajectory))}</strong> 的最终 TF marker accuracy 最高（{strongest_marker.final_tf_marker_accuracy:.3f}）：结构化 Tiny Shakespeare haystack 使 noise token 的局部统计更可压缩，marker 更突出，APE 也更容易形成稳定绑定。</li>
      <li><strong>{html.escape(str(weakest_marker.trajectory))}</strong> 最终 TF marker accuracy 只有 {weakest_marker.final_tf_marker_accuracy:.3f}：这不是最终 count 不会，而是内容绑定阶段在 10k step 内仍未完成。</li>
      <li><strong>RPE 的优势主要发生在 Stage B：</strong>相对距离 bias 不必从绝对位置组合中间接学出 needle-to-trace 对齐，因此 marker binding 明显更早、更完整；final count / index scaffold 的优势小得多。</li>
      <li><strong>长 context 的 exact trace 更苛刻：</strong>v12 的局部 marker recall 已高，但 count 最多 50，任何小错误都会显著降低整条序列 exact-match。</li>
      <li><strong>fixed dataset 不保证学到算法：</strong>v13 的 final answer 很快饱和，而 marker binding 慢于流式数据，符合有限 prompt 记忆 / shortcut 与可泛化位置绑定竞争的解释。</li>
    </ul>

    <h2>6. 还值得采用的常见学习动态分析</h2>
    <ol>
      <li><strong>first-passage 与 change-point：</strong>本报告已做，避免只凭肉眼说“两阶段”。</li>
      <li><strong>loss decomposition：</strong>本报告已做，区分 syntax、successor、retrieval、readout。</li>
      <li><strong>local-to-global error model：</strong>本报告已做，用 p<sup>n</sup> 判断 exact-match 是否只是序列长度效应。</li>
      <li><strong>checkpoint attention specialization：</strong>本报告已做；下一步可对最终选中的 head 做 position-local ablation，验证形成时间与 causal necessity 是否同步。</li>
      <li><strong>representation emergence：</strong>本报告已追踪 centroid geometry；更严格版本应增加 held-out ridge probe、CKA/SVCCA 以及 probe-direction patching，避免把可读性误认成因果性。</li>
      <li><strong>gradient attribution：</strong>未来训练应按 component 分别记录 gradient norm / cosine conflict，直接检验 early easy targets 是否压制 marker gradient，而不只看参数总漂移。</li>
      <li><strong>seed 与 grokking 检验：</strong>至少 3–5 seeds，比较 breakpoint 分布；同时延长 APE/RoPE 到 20k–30k，判断它们是永不学会还是 delayed generalization。</li>
    </ol>

    <h2>7. 结论</h2>
    <div class="callout green"><strong>最可信的两阶段解释：</strong>Stage A 学会可由 count 标签和局部 successor 规则解决的 scaffold；Stage B 学会把第 k 个 trace query 与 prompt 中第 k 个 needle 的 identity 绑定。不同位置编码、context 长度和数据分布主要改变 Stage B。exact trace 是 Stage B 局部质量经过长度放大的严格读数，而不是一个可靠的独立阶段指标。</div>
    <p>复现产物包括 <code>stage_summary.csv</code>、<code>local_exact_decomposition.csv</code>、<code>checkpoint_attention.csv</code>、<code>checkpoint_state_geometry.csv</code> 与 <code>checkpoint_parameter_drift.csv</code>。报告中的图片均以内嵌 base64 保存，单独发送 HTML 也能完整显示。</p>
    """
    report_path = output_dir / "syn_cot_learning_stages_report.html"
    report_path.write_text(
        f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CoT learning stages</title><style>{css}</style></head><body><main>{body}</main></body></html>",
        encoding="utf-8",
    )
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze two-stage CoT learning dynamics across v11-v14.")
    parser.add_argument("--results-root", type=Path, default=ROOT / "colab_results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "colab_results" / "cot_learning_dynamics_v11_v14",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--examples-per-count", type=int, default=1)
    parser.add_argument("--skip-checkpoint-diagnostics", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.results_root)
    trajectories = trajectories_for_runs(runs)
    stages = stage_summary(trajectories)
    decomposition = local_exact_decomposition(trajectories)
    stages.to_csv(args.output_dir / "stage_summary.csv", index=False)
    decomposition.to_csv(args.output_dir / "local_exact_decomposition.csv", index=False)

    attention_path = args.output_dir / "checkpoint_attention.csv"
    geometry_path = args.output_dir / "checkpoint_state_geometry.csv"
    drift_path = args.output_dir / "checkpoint_parameter_drift.csv"
    if args.skip_checkpoint_diagnostics and attention_path.exists() and geometry_path.exists() and drift_path.exists():
        attention = read_csv(attention_path)
        geometry = read_csv(geometry_path)
        drift = read_csv(drift_path)
    elif args.skip_checkpoint_diagnostics:
        raise FileNotFoundError("Checkpoint diagnostics were skipped but cached CSV files are missing")
    else:
        attention, geometry, drift = checkpoint_diagnostics(
            trajectories,
            args.output_dir,
            examples_per_count=args.examples_per_count,
            device=args.device,
        )

    figure_paths = {
        "overview": figures_dir / "two_stage_overview.png",
        "losses": figures_dir / "component_loss_dynamics.png",
        "timeline": figures_dir / "stage_threshold_timeline.png",
        "local_exact": figures_dir / "local_to_exact_decomposition.png",
        "attention": figures_dir / "checkpoint_attention_emergence.png",
        "geometry": figures_dir / "checkpoint_geometry_emergence.png",
        "drift": figures_dir / "checkpoint_parameter_drift.png",
    }
    plot_stage_overview(trajectories, stages, figure_paths["overview"])
    plot_component_losses(trajectories, figure_paths["losses"])
    plot_threshold_timeline(stages, figure_paths["timeline"])
    plot_local_exact(decomposition, figure_paths["local_exact"])
    plot_attention_emergence(attention, figure_paths["attention"])
    plot_geometry_emergence(geometry, figure_paths["geometry"])
    plot_parameter_drift(drift, figure_paths["drift"])
    report = write_report(args.output_dir, stages, figure_paths, attention, geometry)
    print(f"REPORT={report}")


if __name__ == "__main__":
    main()
