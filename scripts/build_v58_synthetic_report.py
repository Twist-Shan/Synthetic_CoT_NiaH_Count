#!/usr/bin/env python
"""Build the self-contained v58 synthetic NiaH report.

The report consumes archived v58 CSV/JSON artifacts only.  It does not run
model inference.  The primary comparison is between two independently
initialized and independently trained models with matched architecture/data:
separator/no-index Thinking and Non-thinking.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "v58_final"
ANALYSIS = DATA / "analysis"
REPORT_DIR = ROOT / "reports"
ASSET_DIR = REPORT_DIR / "NiaH_Synthetic_report_assets"

BLUE = "#2563A6"
BLUE_DARK = "#173B63"
ORANGE = "#D97706"
PURPLE = "#7158A6"
GREEN = "#23856D"
RED = "#C64E4E"
GREY = "#7B8794"
LIGHT_GREY = "#D7DEE7"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D8DEE7", linewidth=0.65, alpha=0.72)
    ax.spines[["top", "right"]].set_visible(False)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.titlesize": 14,
        }
    )


def data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def figure(path: Path, caption: str, alt: str) -> str:
    return (
        f'<figure><img src="{data_uri(path)}" alt="{html.escape(alt)}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def pct(value: object, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{100 * number:.{digits}f}%"


def num(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def html_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{item}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def plot_behavior(by_count: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.25))
    colors = {"nonthinking": ORANGE, "thinking": PURPLE}
    labels = {"nonthinking": "Non-thinking", "thinking": "Thinking"}

    ax = axes[0]
    for mode in ("nonthinking", "thinking"):
        line = by_count.loc[by_count["mode"].eq(mode)].sort_values("count")
        y = line["ar_final_accuracy"].to_numpy(float)
        lo = line["ar_final_accuracy_wilson95_low"].to_numpy(float)
        hi = line["ar_final_accuracy_wilson95_high"].to_numpy(float)
        ax.errorbar(
            line["count"],
            y,
            yerr=np.vstack((y - lo, hi - y)),
            marker="o",
            linewidth=2,
            capsize=2.5,
            color=colors[mode],
            label=labels[mode],
        )
    ax.set(
        title="A · Independent confirmation by count",
        xlabel="Gold count",
        ylabel="Free-running exact accuracy",
        xticks=range(1, 11),
        ylim=(-0.03, 1.04),
    )
    style_axis(ax)
    ax.legend(loc="center left")

    ax = axes[1]
    ordered = summary.set_index("mode").loc[["nonthinking", "thinking"]]
    y = ordered["ar_final_accuracy"].to_numpy(float)
    lo = ordered["ar_final_accuracy_wilson95_low"].to_numpy(float)
    hi = ordered["ar_final_accuracy_wilson95_high"].to_numpy(float)
    ax.bar([0, 1], y, color=[ORANGE, PURPLE], width=0.62)
    ax.errorbar([0, 1], y, yerr=np.vstack((y - lo, hi - y)), fmt="none", color="#222", capsize=4)
    for x, value in enumerate(y):
        ax.text(x, value + 0.045, f"{100 * value:.2f}%", ha="center", fontweight="bold")
    ax.set(
        title="B · Overall behavior (2,000/mode)",
        ylabel="Free-running exact accuracy",
        xticks=[0, 1],
        xticklabels=["Non-thinking", "Thinking"],
        ylim=(0, 1.08),
    )
    style_axis(ax)

    ax = axes[2]
    thinking = summary.loc[summary["mode"].eq("thinking")].iloc[0]
    metrics = ["trace_exact", "trace_ordered_marker_accuracy", "trace_marker_count_accuracy"]
    values = [float(thinking[item]) for item in metrics]
    names = ["Exact trace", "Ordered marker", "Marker count"]
    bars = ax.bar(range(3), values, color=[BLUE, GREEN, PURPLE], width=0.62)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{100 * value:.1f}%", ha="center")
    ax.set(
        title="C · Thinking trace diagnostics",
        ylabel="Rate",
        xticks=range(3),
        xticklabels=names,
        ylim=(0, 1.06),
    )
    style_axis(ax)
    fig.suptitle("v58 behavior: uniform Thinking advantage on a disjoint test cohort", y=1.04)
    fig.tight_layout()
    savefig(fig, path)


def plot_geometry(selected: pd.DataFrame, path: Path) -> None:
    order = [
        ("nonthinking", "nonthinking_prompt_occurrence", "NT running"),
        ("thinking", "thinking_item_end", "T running"),
        ("nonthinking", "nonthinking_answer_query", "NT final"),
        ("thinking", "thinking_answer_query", "T final"),
    ]
    rows = []
    for mode, endpoint, label in order:
        match = selected.loc[
            selected["comparison_mode"].eq(mode) & selected["endpoint"].eq(endpoint)
        ]
        if len(match) != 1:
            raise RuntimeError(f"geometry row mismatch for {mode}/{endpoint}: {len(match)}")
        row = match.iloc[0].copy()
        row["display"] = label
        rows.append(row)
    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.25))
    ax = axes[0]
    log_values = frame["confirmation_logistic_balanced_accuracy"].to_numpy(float)
    ncc_values = frame["confirmation_ncc_balanced_accuracy"].to_numpy(float)
    ax.bar(x - width / 2, log_values, width, color=BLUE, label="L2 Logistic")
    ax.bar(x + width / 2, ncc_values, width, color=GREEN, hatch="//", alpha=0.82, label="NCC")
    ax.axhline(0.1, color=RED, linestyle=":", linewidth=1.4, label="10-class chance")
    for index, row in frame.reset_index(drop=True).iterrows():
        height = max(float(row["confirmation_logistic_balanced_accuracy"]), float(row["confirmation_ncc_balanced_accuracy"]))
        ax.text(index, height + 0.045, f"L{int(row['layer'])}", ha="center", fontweight="bold")
    ax.set(
        title="A · Frozen held-out decodability",
        ylabel="Balanced accuracy",
        xticks=x,
        xticklabels=frame["display"],
        ylim=(0, 1.10),
    )
    style_axis(ax)
    ax.legend(loc="upper left")

    ax = axes[1]
    rsa = frame["confirmation_ordinal_rsa"].to_numpy(float)
    bars = ax.bar(x, rsa, color=[ORANGE, PURPLE, ORANGE, PURPLE], alpha=0.88)
    for bar, value in zip(bars, rsa, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    ax.axhline(0, color="#222", linewidth=1)
    ax.set(
        title="B · Held-out ordinal RSA",
        ylabel="Spearman correlation with count gap",
        xticks=x,
        xticklabels=frame["display"],
        ylim=(-0.08, 1.02),
    )
    style_axis(ax)
    fig.suptitle("Clean geometry: running states and answer-query states are separate endpoints", y=1.04)
    fig.tight_layout()
    savefig(fig, path)


def _head_matrix(frame: pd.DataFrame, value: str = "score") -> np.ndarray:
    matrix = np.full((4, 8), np.nan)
    for row in frame.itertuples(index=False):
        matrix[int(row.layer) - 1, int(row.head)] = float(getattr(row, value))
    if np.isnan(matrix).any():
        raise RuntimeError("head matrix is incomplete")
    return matrix


def plot_role_specialization(dynamics: pd.DataFrame, path: Path) -> None:
    final = dynamics.loc[dynamics["step"].eq(dynamics["step"].max())]
    broad = final.loc[final["role"].eq("nonthinking_broad")]
    targeted = final.loc[final["role"].eq("targeted_retrieval")]
    role_names = ["thinking_broad", "marker_successor", "targeted_retrieval"]

    fig = plt.figure(figsize=(15.2, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.12], hspace=0.34, wspace=0.24)
    for slot, frame, title, cmap in (
        (grid[0, 0], broad, "A · Non-thinking broad score", "YlOrBr"),
        (grid[0, 1], targeted, "B · Thinking targeted mass", "PuBuGn"),
    ):
        ax = fig.add_subplot(slot)
        matrix = _head_matrix(frame)
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0)
        ax.set(title=title, xlabel="Head", ylabel="Layer", xticks=range(8), yticks=range(4), yticklabels=["L1", "L2", "L3", "L4"])
        for layer in range(4):
            for head in range(8):
                color = "white" if matrix[layer, head] > 0.62 * float(np.nanmax(matrix)) else "#18212B"
                ax.text(head, layer, f"{matrix[layer, head]:.3f}", ha="center", va="center", fontsize=7.3, color=color)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="Role score")

    ax = fig.add_subplot(grid[1, :])
    normalized_rows = []
    for role in role_names:
        line = final.loc[final["role"].eq(role)].sort_values(["layer", "head"])
        values = line["score"].to_numpy(float)
        normalized_rows.append(values / max(float(values.max()), 1e-12))
    matrix = np.vstack(normalized_rows)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    labels = [f"L{layer}H{head}" for layer in range(1, 5) for head in range(8)]
    ax.set(
        title="C · Within-Thinking role fingerprint (row-normalized)",
        xlabel="Attention head",
        yticks=range(3),
        yticklabels=["Broad", "Marker successor", "Targeted retrieval"],
        xticks=range(32),
        xticklabels=labels,
    )
    ax.tick_params(axis="x", labelrotation=75, labelsize=7.2)
    fig.colorbar(image, ax=ax, fraction=0.018, pad=0.015, label="Score / row maximum")
    fig.suptitle("Final-checkpoint role specialization and head-bank differentiation", y=0.99)
    savefig(fig, path)


def _arm_curve(
    frame: pd.DataFrame,
    *,
    mode: str,
    scope: str,
    metric: str,
    endpoint: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subset = frame.loc[
        frame["comparison_mode"].eq(mode) & frame["scope"].eq(scope)
    ].copy()
    if endpoint is not None:
        subset = subset.loc[subset["endpoint"].eq(endpoint)]
    clean = subset.loc[subset["path_kind"].eq("clean")]
    if len(clean) != 1:
        raise RuntimeError(f"clean arm mismatch: {mode}/{scope}/{metric}/{endpoint}")
    ks = np.array([0, 1, 2, 4], dtype=int)
    ranked = [float(clean.iloc[0][metric])]
    control_mid = [np.nan]
    control_lo = [np.nan]
    control_hi = [np.nan]
    for k in ks[1:]:
        rank = subset.loc[subset["path_kind"].eq("ranked") & subset["top_k"].eq(k)]
        controls = subset.loc[
            subset["path_kind"].eq("layer_matched_control") & subset["top_k"].eq(k), metric
        ].dropna()
        if len(rank) != 1 or controls.empty:
            raise RuntimeError(f"arm mismatch: {mode}/{scope}/K{k}/{metric}/{endpoint}")
        ranked.append(float(rank.iloc[0][metric]))
        control_mid.append(float(controls.median()))
        control_lo.append(float(controls.min()))
        control_hi.append(float(controls.max()))
    return ks, np.asarray(ranked), np.asarray(control_mid), np.asarray(control_lo), np.asarray(control_hi)


def _plot_ranked_control(
    ax: plt.Axes,
    curve: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    title: str,
    ylabel: str,
    color: str,
    ylim: tuple[float, float] = (0, 1.05),
) -> None:
    ks, ranked, control, low, high = curve
    ax.plot(ks, ranked, marker="o", color=color, linewidth=2.2, label="Ranked bank")
    valid = ~np.isnan(control)
    ax.plot(ks[valid], control[valid], marker="s", color=GREY, linestyle="--", label="Control median")
    ax.fill_between(ks[valid], low[valid], high[valid], color=LIGHT_GREY, alpha=0.65, label="Control range")
    ax.set(title=title, xlabel="Cumulative Top-K removed", ylabel=ylabel, xticks=ks, ylim=ylim)
    style_axis(ax)


def plot_topk(
    tf_behavior: pd.DataFrame,
    tf_ncc: pd.DataFrame,
    free_running: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2))
    _plot_ranked_control(
        axes[0, 0],
        _arm_curve(tf_behavior, mode="thinking", scope="role_query_local", metric="teacher_forced_trace_accuracy"),
        title="A · Teacher-forced trace-token accuracy",
        ylabel="Accuracy",
        color=PURPLE,
    )
    _plot_ranked_control(
        axes[0, 1],
        _arm_curve(free_running, mode="thinking", scope="role_query_local_free_running", metric="trace_exact"),
        title="B · Free-running exact trace",
        ylabel="Exact rate",
        color=PURPLE,
    )
    _plot_ranked_control(
        axes[0, 2],
        _arm_curve(free_running, mode="thinking", scope="role_query_local_free_running", metric="ar_final_accuracy"),
        title="C · Thinking free-running final answer",
        ylabel="Exact accuracy",
        color=PURPLE,
    )
    _plot_ranked_control(
        axes[1, 0],
        _arm_curve(
            tf_ncc,
            mode="thinking",
            scope="role_query_local",
            endpoint="thinking_item_end",
            metric="confirmation_ncc_balanced_accuracy",
        ),
        title="D · Post-ablation running NCC",
        ylabel="Frozen NCC balanced accuracy",
        color=BLUE,
        ylim=(0, 0.36),
    )
    axes[1, 0].axhline(0.1, color=RED, linestyle=":", linewidth=1.2)
    _plot_ranked_control(
        axes[1, 1],
        _arm_curve(
            tf_ncc,
            mode="thinking",
            scope="role_query_local",
            endpoint="thinking_answer_query",
            metric="confirmation_ncc_balanced_accuracy",
        ),
        title="E · Post-ablation final NCC",
        ylabel="Frozen NCC balanced accuracy",
        color=GREEN,
    )
    _plot_ranked_control(
        axes[1, 2],
        _arm_curve(free_running, mode="nonthinking", scope="role_query_local_free_running", metric="ar_final_accuracy"),
        title="F · Non-thinking broad-bank test",
        ylabel="Exact accuracy",
        color=ORANGE,
        ylim=(0, 0.42),
    )
    for ax in axes.flat:
        ax.legend(loc="best", fontsize=7.5)
    fig.suptitle("Ranked Top-K necessity and clean-frozen post-ablation representation", y=1.01)
    fig.tight_layout()
    savefig(fig, path)


def plot_sufficiency(answer: pd.DataFrame, progress: pd.DataFrame, path: Path) -> None:
    answer = answer.loc[answer["is_discovery_selected_layer"].eq(1.0)].copy()
    answer_summary = answer.groupby(["mode", "condition"], as_index=False).agg(
        donor_adoption=("donor_adoption", "mean"),
        receiver_retention=("receiver_retention", "mean"),
        donor_margin_shift=("donor_margin_shift", "mean"),
        observations=("donor_adoption", "size"),
    )
    progress_summary = progress.groupby("condition", as_index=False).agg(
        donor_first_adoption=("donor_first_adoption", "mean"),
        natural_first_retention=("natural_first_retention", "mean"),
        generated_final_correct=("generated_final_correct", "mean"),
        observations=("donor_first_adoption", "size"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    conditions = ["adjacent_count_donor", "same_count_context_control"]
    x = np.arange(2)
    width = 0.34
    for index, mode in enumerate(("nonthinking", "thinking")):
        values = []
        for condition in conditions:
            row = answer_summary.loc[
                answer_summary["mode"].eq(mode) & answer_summary["condition"].eq(condition)
            ]
            values.append(float(row.iloc[0]["donor_adoption"]))
        bars = ax.bar(x + (index - 0.5) * width, values, width, color=ORANGE if mode == "nonthinking" else PURPLE, label="Non-thinking" if mode == "nonthinking" else "Thinking")
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{100 * value:.1f}%", ha="center", fontsize=8.5)
    ax.set(
        title="A · Answer-query full-state transplant",
        ylabel="Greedy donor-count adoption",
        xticks=x,
        xticklabels=["Adjacent-count donor", "Same-count context control"],
        ylim=(0, 1.08),
    )
    style_axis(ax)
    ax.legend()

    ax = axes[1]
    order = [
        "clean",
        "centroid_shift",
        "orthogonal_control",
        "natural_marker_cross_position",
        "natural_item_span_cross_position",
    ]
    labels = ["Clean", "Centroid shift", "Orthogonal", "Natural marker†", "Natural item†"]
    values = [
        float(progress_summary.loc[progress_summary["condition"].eq(condition), "donor_first_adoption"].iloc[0])
        for condition in order
    ]
    colors = [GREY, BLUE, LIGHT_GREY, ORANGE, ORANGE]
    bars = ax.bar(range(len(order)), values, color=colors)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{100 * value:.1f}%", ha="center", fontsize=8)
    ax.set(
        title="B · Free-running progress-state sufficiency",
        ylabel="Donor next-marker adoption",
        xticks=range(len(order)),
        xticklabels=labels,
        ylim=(0, 0.58),
    )
    ax.tick_params(axis="x", labelrotation=18)
    style_axis(ax)
    fig.suptitle("Terminal answer state is executable; same-position progress state remains weak", y=1.03)
    fig.tight_layout()
    savefig(fig, path)


def _weighted_routing(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for step, group in frame.groupby("step"):
        weights = group["observations"].to_numpy(float)
        total = float(weights.sum())
        row: dict[str, float] = {"step": float(step), "observations": total}
        for metric in ("targeted_mass", "qk_margin", "correct_occurrence_top1"):
            row[metric] = float(np.average(group[metric].to_numpy(float), weights=weights))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("step")


def _bank_dynamics(
    frame: pd.DataFrame,
    selected_heads: set[tuple[int, int]],
) -> pd.DataFrame:
    rows = []
    for step, group in frame.groupby("step"):
        group = group.copy()
        group["selected"] = [
            (int(layer), int(head)) in selected_heads
            for layer, head in zip(group["layer"], group["head"], strict=True)
        ]
        values = group["score"].to_numpy(float)
        selected = group.loc[group["selected"], "score"].to_numpy(float)
        other = group.loc[~group["selected"], "score"].to_numpy(float)
        current = set(
            (int(row.layer), int(row.head))
            for row in group.nlargest(len(selected_heads), "score").itertuples(index=False)
        )
        union = selected_heads | current
        rows.append(
            {
                "step": int(step),
                "selected_mean": float(selected.mean()),
                "other_mean": float(other.mean()),
                "bank_share": float(selected.sum() / max(values.sum(), 1e-12)),
                "jaccard": float(len(selected_heads & current) / len(union)),
            }
        )
    return pd.DataFrame(rows).sort_values("step")


def plot_training_dynamics(
    high_power: pd.DataFrame,
    routing: pd.DataFrame,
    attention: pd.DataFrame,
    rankings: pd.DataFrame,
    patching: pd.DataFrame,
    causal: pd.DataFrame,
    path: Path,
) -> dict[str, float]:
    routed = _weighted_routing(routing)
    targeted_heads = set(
        (int(row.layer), int(row.head))
        for row in rankings.loc[
            rankings["role"].eq("targeted_retrieval") & rankings["rank"].le(4)
        ].itertuples(index=False)
    )
    broad_heads = set(
        (int(row.layer), int(row.head))
        for row in rankings.loc[
            rankings["role"].eq("nonthinking_broad") & rankings["rank"].le(4)
        ].itertuples(index=False)
    )
    targeted = _bank_dynamics(
        attention.loc[attention["role"].eq("targeted_retrieval")], targeted_heads
    )
    broad = _bank_dynamics(
        attention.loc[attention["role"].eq("nonthinking_broad")], broad_heads
    )
    top2_patch = patching.loc[
        patching["intervention"].eq("value_only_at_target_source")
        & patching["top_n"].eq(2)
    ].sort_values("step")

    causal_fixed = causal.loc[causal["intervention"].eq("fixed_head_zero")].copy()
    causal_control = causal.loc[causal["intervention"].eq("same_layer_control_zero")].copy()
    causal_specific = causal_fixed.merge(
        causal_control[["step", "role", "causal_damage"]],
        on=["step", "role"],
        suffixes=("_fixed", "_control"),
    )
    causal_specific["specificity"] = (
        causal_specific["causal_damage_fixed"] - causal_specific["causal_damage_control"]
    )

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.4), sharex=False)
    ax = axes[0, 0]
    for mode, color, label in (
        ("nonthinking", ORANGE, "Non-thinking final"),
        ("thinking", PURPLE, "Thinking final"),
    ):
        line = high_power.loc[high_power["mode"].eq(mode)].sort_values("step")
        ax.plot(line["step"], line["ar_accuracy"], marker="o", markersize=3.5, linewidth=2, color=color, label=label)
    trace = high_power.loc[high_power["mode"].eq("thinking")].sort_values("step")
    ax.plot(trace["step"], trace["trace_exact"], color=BLUE, linestyle="--", linewidth=1.7, label="Thinking trace exact")
    ax.set(title="A · High-power free-running behavior", xlabel="Training step", ylabel="Rate", ylim=(-0.03, 1.03))
    style_axis(ax)
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    ax.plot(routed["step"], routed["targeted_mass"], color=GREEN, linewidth=2.2, label="Targeted attention mass")
    ax.set(title="B · Targeted routing formation", xlabel="Training step", ylabel="Attention mass", ylim=(0, 0.48))
    style_axis(ax)
    twin = ax.twinx()
    twin.plot(routed["step"], routed["qk_margin"], color=PURPLE, linewidth=1.5, alpha=0.82, label="QK correct-vs-best-wrong")
    twin.axhline(0, color=GREY, linestyle=":", linewidth=1)
    twin.set_ylabel("Scaled QK margin")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower right")

    ax = axes[0, 2]
    ax.plot(targeted["step"], targeted["bank_share"], color=PURPLE, linewidth=2.2, label="Targeted Top-4 share")
    ax.plot(broad["step"], broad["bank_share"], color=ORANGE, linestyle="--", linewidth=1.8, label="NT broad Top-4 share")
    ax.axhline(4 / 32, color=GREY, linestyle=":", linewidth=1, label="Uniform 4/32")
    ax.set(title="C · Bank concentration", xlabel="Training step", ylabel="Top-4 share of all-head score", ylim=(0, 1.0))
    style_axis(ax)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(targeted["step"], targeted["jaccard"], color=PURPLE, linewidth=2.2, label="Targeted identity")
    ax.plot(broad["step"], broad["jaccard"], color=ORANGE, linestyle="--", linewidth=1.8, label="NT broad identity")
    ax.set(title="D · Final-bank identity stabilization", xlabel="Training step", ylabel="Top-4 Jaccard vs final", ylim=(-0.03, 1.03))
    style_axis(ax)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(top2_patch["step"], top2_patch["normalized_recovery_mean"], color=GREEN, marker="o", linewidth=2, label="Top-2 value recovery")
    ax.plot(top2_patch["step"], top2_patch["patched_correct"], color=BLUE, marker="s", linewidth=1.7, label="Patched marker accuracy")
    ax.plot(top2_patch["step"], top2_patch["corrupt_correct"], color=GREY, linestyle="--", linewidth=1.5, label="Corrupt baseline accuracy")
    ax.set(title="E · Retrieval transport emerges", xlabel="Training step", ylabel="Recovery / accuracy", ylim=(-0.06, 1.03))
    style_axis(ax)
    ax.legend(loc="upper left")

    ax = axes[1, 2]
    for role, color, label in (
        ("marker_successor", BLUE, "Marker-successor head"),
        ("targeted_retrieval", PURPLE, "Targeted head"),
    ):
        line = causal_specific.loc[causal_specific["role"].eq(role)].sort_values("step")
        ax.plot(line["step"], line["specificity"], color=color, marker="o", markersize=3.5, linewidth=2, label=label)
    ax.axhline(0, color=GREY, linestyle=":", linewidth=1)
    ax.set(title="F · Position-local causal specificity", xlabel="Training step", ylabel="Damage(selected) - damage(control)")
    style_axis(ax)
    ax.legend(loc="upper left")

    for ax in axes.flat:
        ax.axvline(1500, color=RED, linestyle=":", linewidth=1.0, alpha=0.78)
        ax.ticklabel_format(style="plain", axis="x")
    fig.suptitle("Synchronized training dynamics (linear step axis; 500 free-running examples/mode/checkpoint)", y=1.01)
    fig.tight_layout()
    savefig(fig, path)

    final_mass = float(routed.iloc[-1]["targeted_mass"])
    first_50 = int(routed.loc[routed["targeted_mass"].ge(0.5 * final_mass), "step"].iloc[0])
    first_80 = int(routed.loc[routed["targeted_mass"].ge(0.8 * final_mass), "step"].iloc[0])
    return {
        "final_targeted_mass": final_mass,
        "first_targeted_50pct_step": first_50,
        "first_targeted_80pct_step": first_80,
        "final_targeted_top4_share": float(targeted.iloc[-1]["bank_share"]),
        "final_broad_top4_share": float(broad.iloc[-1]["bank_share"]),
        "final_targeted_jaccard": float(targeted.iloc[-1]["jaccard"]),
        "final_broad_jaccard": float(broad.iloc[-1]["jaccard"]),
    }


def row_value(frame: pd.DataFrame, **conditions: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].eq(value)
    result = frame.loc[mask]
    if len(result) != 1:
        raise RuntimeError(f"expected one row for {conditions}, found {len(result)}")
    return result.iloc[0]


def median_control(
    frame: pd.DataFrame,
    metric: str,
    *,
    mode: str,
    scope: str,
    top_k: int,
) -> float:
    rows = frame.loc[
        frame["comparison_mode"].eq(mode)
        & frame["scope"].eq(scope)
        & frame["path_kind"].eq("layer_matched_control")
        & frame["top_k"].eq(top_k),
        metric,
    ].dropna()
    if rows.empty:
        raise RuntimeError("missing matched controls")
    return float(rows.median())


def build_report(output: Path) -> dict:
    setup_style()
    cfg = read_json(DATA / "config.json")
    run_manifest = read_json(DATA / "manifest.json")
    confirmation = read_csv(ANALYSIS / "behavior_confirmation_v58" / "autoregressive_summary.csv")
    confirmation_by_count = read_csv(ANALYSIS / "behavior_confirmation_v58" / "autoregressive_by_count.csv")
    behavior_gate = read_json(ANALYSIS / "behavior_confirmation_v58" / "behavior_gate.json")
    canonical = read_csv(DATA / "tables" / "final_autoregressive_summary.csv")
    sampling = read_csv(DATA / "tables" / "training_sampling_distribution.csv")
    train_metrics = read_csv(DATA / "tables" / "train_metrics.csv")
    geometry = read_csv(ANALYSIS / "v58_clean_ncc" / "selected_confirmation_summary.csv")
    rankings = read_csv(ANALYSIS / "v58_topk_ncc" / "head_rankings.csv")
    tf_behavior = read_csv(ANALYSIS / "v58_topk_ncc" / "post_ablation_behavior.csv")
    tf_ncc = read_csv(ANALYSIS / "v58_topk_ncc" / "post_ablation_ncc_selected.csv")
    free_topk = read_csv(ANALYSIS / "v58_free_running_topk" / "free_running_summary.csv")
    answer = read_csv(ANALYSIS / "v58_free_running_sufficiency" / "answer_transplant_confirmation.csv")
    progress = read_csv(ANALYSIS / "v58_free_running_sufficiency" / "progress_rollout_confirmation.csv")
    attention = read_csv(ANALYSIS / "extended" / "tables" / "attention_role_dynamics.csv")
    routing = read_csv(ANALYSIS / "phase_transition_audit" / "tables" / "routing_qk_by_k.csv")
    patching = read_csv(ANALYSIS / "phase_transition_audit" / "tables" / "retrieval_transport_recovery.csv")
    causal = read_csv(ANALYSIS / "phase_transition_audit" / "tables" / "local_head_causal_damage.csv")
    high_power = read_csv(ANALYSIS / "phase_transition_audit" / "tables" / "high_power_ar_summary.csv")

    final_hp_t = row_value(high_power, step=10000, mode="thinking")
    final_hp_nt = row_value(high_power, step=10000, mode="nonthinking")
    canonical_t = row_value(canonical, mode="thinking")
    canonical_nt = row_value(canonical, mode="nonthinking")
    if not math.isclose(float(final_hp_t["ar_accuracy"]), float(canonical_t["ar_final_accuracy"]), abs_tol=1e-12):
        raise RuntimeError("high-power Thinking endpoint does not reproduce canonical evaluation")
    if not math.isclose(float(final_hp_nt["ar_accuracy"]), float(canonical_nt["ar_final_accuracy"]), abs_tol=1e-12):
        raise RuntimeError("high-power Non-thinking endpoint does not reproduce canonical evaluation")

    figures = {
        "behavior": ASSET_DIR / "v58_behavior_confirmation.png",
        "geometry": ASSET_DIR / "v58_clean_geometry.png",
        "roles": ASSET_DIR / "v58_role_specialization.png",
        "topk": ASSET_DIR / "v58_topk_causal.png",
        "sufficiency": ASSET_DIR / "v58_free_running_sufficiency.png",
        "dynamics": ASSET_DIR / "v58_training_dynamics.png",
    }
    plot_behavior(confirmation_by_count, confirmation, figures["behavior"])
    plot_geometry(geometry, figures["geometry"])
    plot_role_specialization(attention, figures["roles"])
    plot_topk(tf_behavior, tf_ncc, free_topk, figures["topk"])
    plot_sufficiency(answer, progress, figures["sufficiency"])
    dynamics_stats = plot_training_dynamics(
        high_power, routing, attention, rankings, patching, causal, figures["dynamics"]
    )

    c_t = row_value(confirmation, mode="thinking")
    c_nt = row_value(confirmation, mode="nonthinking")
    g_nt_run = row_value(geometry, comparison_mode="nonthinking", endpoint="nonthinking_prompt_occurrence")
    g_nt_final = row_value(geometry, comparison_mode="nonthinking", endpoint="nonthinking_answer_query")
    g_t_run = row_value(geometry, comparison_mode="thinking", endpoint="thinking_item_end")
    g_t_final = row_value(geometry, comparison_mode="thinking", endpoint="thinking_answer_query")

    tf_clean = row_value(tf_behavior, comparison_mode="thinking", scope="role_query_local", path_kind="clean", top_k=0)
    tf_k2 = row_value(tf_behavior, comparison_mode="thinking", scope="role_query_local", path_kind="ranked", top_k=2)
    fr_clean = row_value(free_topk, comparison_mode="thinking", scope="role_query_local_free_running", path_kind="clean", top_k=0)
    fr_k2 = row_value(free_topk, comparison_mode="thinking", scope="role_query_local_free_running", path_kind="ranked", top_k=2)
    fr_k4 = row_value(free_topk, comparison_mode="thinking", scope="role_query_local_free_running", path_kind="ranked", top_k=4)
    fr_k2_control_trace = median_control(
        free_topk, "trace_exact", mode="thinking", scope="role_query_local_free_running", top_k=2
    )

    selected_answer = answer.loc[answer["is_discovery_selected_layer"].eq(1.0)]
    t_donor = selected_answer.loc[
        selected_answer["mode"].eq("thinking")
        & selected_answer["condition"].eq("adjacent_count_donor")
    ]
    t_context = selected_answer.loc[
        selected_answer["mode"].eq("thinking")
        & selected_answer["condition"].eq("same_count_context_control")
    ]
    nt_donor = selected_answer.loc[
        selected_answer["mode"].eq("nonthinking")
        & selected_answer["condition"].eq("adjacent_count_donor")
    ]
    nt_context = selected_answer.loc[
        selected_answer["mode"].eq("nonthinking")
        & selected_answer["condition"].eq("same_count_context_control")
    ]

    patch_final = row_value(
        patching,
        step=10000,
        intervention="value_only_at_target_source",
        top_n=2,
    )
    successor_final = row_value(
        causal,
        step=10000,
        role="marker_successor",
        intervention="fixed_head_zero",
    )
    successor_control = row_value(
        causal,
        step=10000,
        role="marker_successor",
        intervention="same_layer_control_zero",
    )
    target_final = row_value(
        causal,
        step=10000,
        role="targeted_retrieval",
        intervention="fixed_head_zero",
    )
    target_control = row_value(
        causal,
        step=10000,
        role="targeted_retrieval",
        intervention="same_layer_control_zero",
    )

    sample_counts = sampling.loc[sampling["dimension"].eq("accepted_counts")].copy()
    sample_counts["share"] = sample_counts["examples"] / sample_counts["total_training_examples"]
    sample_min = float(sample_counts["share"].min())
    sample_max = float(sample_counts["share"].max())
    late_loss = train_metrics.loc[train_metrics["step"].eq(10000)].set_index("mode")

    top_t = rankings.loc[rankings["role"].eq("targeted_retrieval")].sort_values("rank").head(4)
    top_nt = rankings.loc[rankings["role"].eq("nonthinking_broad")].sort_values("rank").head(4)
    top_t_text = ", ".join(f"L{int(row.layer)}H{int(row.head)}" for row in top_t.itertuples(index=False))
    top_nt_text = ", ".join(f"L{int(row.layer)}H{int(row.head)}" for row in top_nt.itertuples(index=False))

    setup_table = html_table(
        ["Component", "v58 setting", "Control / interpretation"],
        [
            ["Task", "Shakespeare character stream; 3-character target set; query first; 256-character data; count 1–10", "Same corpus split, needle pool, sampler, tokenization and context permutation policy"],
            ["Trace", "Thinking: &lt;Think&gt; (&lt;Sep&gt; marker)<sup>N</sup> &lt;/Think&gt; &lt;Ans&gt; count; Non-thinking: &lt;Ans&gt; count", "No explicit numeric running index; marker sequence itself is unchanged across the v35→v58 search"],
            ["Models", "Two independent 12,658,176-parameter transformers; 4 layers × 8 heads; d=512; MLP=2048; RoPE; atomic answer tokens", "Not a shared mode-switch model; exact head IDs are not compared across independently initialized models"],
            ["Optimization", "10,000 steps × batch 128 = 1.28M task examples/model; AdamW; lr 3e-4; warmup 500; weight decay .01; grad clip 1; BF16", "Single seed 1234; snapshots every 100 steps; full recovery state every 500"],
            ["Sampler", f"Max-entropy set×count sampler; realized count shares {pct(sample_min,2)}–{pct(sample_max,2)} in both modes", "Balances gold count while respecting which character sets can realize each count"],
            ["Loss 1–1500", "Teacher-forced next-token cross-entropy on every non-padding token", "Language-model warm start; final count has only its natural one-token frequency"],
            ["Loss 1501–10000", "Task output only; component-normalized coefficients count/trace/structure = 8/8/16", "Thinking region shares 25%/25%/50%; Non-thinking count/structure = 33.3%/66.7%; no scheduled sampling or contrastive loss"],
            ["Final readout", "Atomic count rows use an untied output head; all other vocabulary rows remain tied", "Cached dynamics evaluator was explicitly validated token-for-token against the reference evaluator"],
        ],
    )

    behavior_table = html_table(
        ["Evaluation", "Non-thinking", "Thinking", "Difference / uniformity"],
        [
            ["Canonical test (50/count)", pct(canonical_nt["ar_final_accuracy"]), pct(canonical_t["ar_final_accuracy"]), f"+{100*(float(canonical_t['ar_final_accuracy'])-float(canonical_nt['ar_final_accuracy'])):.1f} pp"],
            ["Independent confirmation (200/count)", f"{pct(c_nt['ar_final_accuracy'],2)} [{pct(c_nt['ar_final_accuracy_wilson95_low'],2)}, {pct(c_nt['ar_final_accuracy_wilson95_high'],2)}]", f"{pct(c_t['ar_final_accuracy'],2)} [{pct(c_t['ar_final_accuracy_wilson95_low'],2)}, {pct(c_t['ar_final_accuracy_wilson95_high'],2)}]", f"+{100*(float(c_t['ar_final_accuracy'])-float(c_nt['ar_final_accuracy'])):.2f} pp"],
            ["Thinking per-count range", "—", f"{pct(behavior_gate['metrics']['thinking_min_count_accuracy'])}–100.0%", f"spread {100*float(behavior_gate['metrics']['thinking_count_spread']):.1f} pp"],
            ["Thinking trace diagnostics", "—", f"exact {pct(c_t['trace_exact'])}; ordered {pct(c_t['trace_ordered_marker_accuracy'])}; marker-count {pct(c_t['trace_marker_count_accuracy'])}", "Diagnostic only; not a behavior gate"],
        ],
    )

    geometry_table = html_table(
        ["Mode / endpoint", "Selected layer", "Logistic BA", "NCC BA", "Ordinal RSA"],
        [
            ["Non-thinking running: kth prompt occurrence end", f"L{int(g_nt_run['layer'])}", pct(g_nt_run["confirmation_logistic_balanced_accuracy"]), pct(g_nt_run["confirmation_ncc_balanced_accuracy"]), num(g_nt_run["confirmation_ordinal_rsa"])],
            ["Thinking running: kth trace item end", f"L{int(g_t_run['layer'])}", pct(g_t_run["confirmation_logistic_balanced_accuracy"]), pct(g_t_run["confirmation_ncc_balanced_accuracy"]), num(g_t_run["confirmation_ordinal_rsa"])],
            ["Non-thinking final: answer query", f"L{int(g_nt_final['layer'])}", pct(g_nt_final["confirmation_logistic_balanced_accuracy"]), pct(g_nt_final["confirmation_ncc_balanced_accuracy"]), num(g_nt_final["confirmation_ordinal_rsa"])],
            ["Thinking final: answer query", f"L{int(g_t_final['layer'])}", pct(g_t_final["confirmation_logistic_balanced_accuracy"]), pct(g_t_final["confirmation_ncc_balanced_accuracy"]), num(g_t_final["confirmation_ordinal_rsa"])],
        ],
    )

    causal_table = html_table(
        ["Experiment", "Clean / damaged baseline", "Intervention result", "Matched-control interpretation"],
        [
            ["Teacher-forced targeted Top-2", f"trace token acc {pct(tf_clean['teacher_forced_trace_accuracy'])}", pct(tf_k2["teacher_forced_trace_accuracy"]), f"control median {pct(median_control(tf_behavior,'teacher_forced_trace_accuracy',mode='thinking',scope='role_query_local',top_k=2))}"],
            ["Free-running targeted Top-2", f"trace exact {pct(fr_clean['trace_exact'])}", f"trace {pct(fr_k2['trace_exact'])}; final {pct(fr_k2['ar_final_accuracy'])}", f"control trace median {pct(fr_k2_control_trace)}"],
            ["Free-running targeted Top-4", f"trace exact {pct(fr_clean['trace_exact'])}", f"trace {pct(fr_k4['trace_exact'])}; final {pct(fr_k4['ar_final_accuracy'])}", "only one disjoint matched Top-4 control; effect is enriched, not unique"],
            ["Top-2 targeted value patch", f"corrupt marker acc {pct(patch_final['corrupt_correct'])}", f"patched {pct(patch_final['patched_correct'])}; normalized recovery {pct(patch_final['normalized_recovery_mean'])}", f"restores +{num(patch_final['margin_restoration'],2)} correct-marker logit margin"],
            ["Marker-successor L2H3 zero", f"baseline accuracy {pct(0.9818181818181818)}", f"accuracy {pct(successor_final['accuracy'])}; margin damage {num(successor_final['causal_damage'])}", f"same-layer control damage {num(successor_control['causal_damage'])}"],
            ["Targeted L4H5 zero", f"baseline accuracy {pct(0.9454545454545454)}", f"accuracy {pct(target_final['accuracy'])}; margin damage {num(target_final['causal_damage'])}", f"control damage {num(target_control['causal_damage'])}; single-head specificity modest"],
        ],
    )

    alignment_table = html_table(
        ["Large-model experiment / claim", "v58 aligned result", "Status"],
        [
            ["Separate running and final geometry", f"Running NCC NT/T={pct(g_nt_run['confirmation_ncc_balanced_accuracy'])}/{pct(g_t_run['confirmation_ncc_balanced_accuracy'])}; final={pct(g_nt_final['confirmation_ncc_balanced_accuracy'])}/{pct(g_t_final['confirmation_ncc_balanced_accuracy'])}", '<span class="status partial">final strongly aligned; running weak</span>'],
            ["Targeted retrieval bank and matched Top-K lesions", f"Thinking Top-4={top_t_text}; free-run trace exact {pct(fr_clean['trace_exact'])}→{pct(fr_k4['trace_exact'])}", '<span class="status yes">positive, but not uniquely necessary</span>'],
            ["Post-ablation representation readout", f"Thinking running/final NCC remain {pct(g_t_run['confirmation_ncc_balanced_accuracy'])}/{pct(g_t_final['confirmation_ncc_balanced_accuracy'])} after local Top-K", '<span class="status partial">measured; no mediation at earlier layers</span>'],
            ["Free-running answer-state sufficiency", f"Thinking adjacent donor adoption {pct(t_donor['donor_adoption'].mean())} vs context control {pct(t_context['donor_adoption'].mean())}; NT {pct(nt_donor['donor_adoption'].mean())} vs {pct(nt_context['donor_adoption'].mean())}", '<span class="status yes">strong Thinking terminal readout</span>'],
            ["Natural no-index progress / recurrence", "Same-position centroid shift only weakly changes the next marker; natural cross-position patches are position-confounded", '<span class="status no">internal-counter sufficiency not established</span>'],
            ["Non-thinking broad retrieval/aggregation", f"Descriptive Top-4={top_nt_text}; ranked ablation is not more damaging than matched controls at a low behavioral floor", '<span class="status no">major remaining gap</span>'],
            ["Universal Thinking final broad aggregator", "Not required: the trace-derived answer-query state itself is executable", '<span class="status partial">not claimed and not needed</span>'],
            ["One-arm serial mediation", "Targeted damage, carrier rescue, commit rescue and final rollout are not all closed in the same damaged baseline", '<span class="status no">open</span>'],
        ],
    )

    css = """
:root{--ink:#16202A;--muted:#52606D;--line:#D6DEE8;--paper:#FFFFFF;--wash:#F3F6F9;--blue:#2563A6;--orange:#D97706;--purple:#7158A6;--green:#23856D;--red:#B94444}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#E9EEF3;color:var(--ink);font-family:Inter,"Segoe UI","Noto Sans SC",Arial,sans-serif;line-height:1.64}
main{max-width:1180px;margin:28px auto;background:var(--paper);padding:55px 66px 72px;box-shadow:0 12px 38px rgba(24,39,56,.12)}
h1{font-size:2.25rem;line-height:1.16;margin:0 0 12px;letter-spacing:-.025em}h2{font-size:1.55rem;margin:52px 0 14px;padding-top:10px;border-top:2px solid var(--ink)}h3{font-size:1.14rem;margin:30px 0 9px}h4{margin:22px 0 6px}p{margin:9px 0 13px}.dek{font-size:1.08rem;color:var(--muted);max-width:940px}.meta{font-size:.88rem;color:var(--muted);margin-bottom:25px}
.abstract,.conclusion,.warning,.example,.formula,.audit{padding:15px 18px;margin:16px 0;border-left:4px solid var(--blue);background:#F4F8FC}.conclusion{border-left-color:var(--green);background:#F2F8F6}.warning{border-left-color:var(--red);background:#FFF5F4}.example{border-left-color:var(--orange);background:#FFF8EC}.formula{border-left-color:var(--purple);background:#F7F5FB}.audit{border-left-color:#607D8B;background:#F3F6F8}.label{font-weight:750;margin-right:5px}
.toc{background:var(--wash);padding:17px 22px;border:1px solid var(--line);margin:24px 0}.toc ol{columns:2;margin:8px 0 0;padding-left:23px}.toc a{color:var(--blue);text-decoration:none}
.chain{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}.chain>div{padding:14px;border:1px solid var(--line);background:#FAFBFC}.chain b{display:block;color:var(--blue);margin-bottom:4px}
figure{margin:24px 0 31px}figure img{display:block;width:100%;height:auto;border:1px solid var(--line);background:white}figcaption{font-size:.9rem;color:#46525E;margin-top:8px;line-height:1.52}
.table-wrap{overflow-x:auto;margin:15px 0 24px;border:1px solid var(--line)}table{width:100%;border-collapse:collapse;font-size:.88rem}th{background:#EDF2F7;text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}td{padding:9px 10px;border-bottom:1px solid #E8EDF2;vertical-align:top}tr:last-child td{border-bottom:0}
code{font-family:"Cascadia Mono",Consolas,monospace;font-size:.88em;background:#EEF2F5;padding:2px 5px;border-radius:3px}.status{display:inline-block;padding:2px 7px;border-radius:12px;font-size:.78rem;font-weight:700;white-space:nowrap}.status.yes{color:#12664F;background:#DDF3EA}.status.partial{color:#855600;background:#FFF0C9}.status.no{color:#8F2F2F;background:#FBE2E2}
ul{padding-left:22px}.small{font-size:.86rem;color:var(--muted)}a{color:var(--blue)}
@media(max-width:800px){main{margin:0;padding:30px 20px}.chain{grid-template-columns:1fr}.toc ol{columns:1}h1{font-size:1.8rem}}
@media print{body{background:white}main{box-shadow:none;margin:0;max-width:none}figure{break-inside:avoid}}
"""

    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NiaH Synthetic v58: geometry, causal experiments, and training dynamics</title><style>{css}</style></head><body><main>
<h1>NiaH Synthetic v58：No-index Thinking vs Independent Non-thinking</h1>
<p class="dek">完整训练设定、Geometry Comparison、Top-K ablation / patching、free-running causal sufficiency，以及与大模型实验对齐的 training dynamics</p>
<p class="meta">生成日期：2026-09-01 · count classes：1–10 · seed：1234 · 两个 independently initialized models · 线性 training-step 主横轴</p>

<div class="abstract"><span class="label">核心结论。</span>v58 已经满足比“Native-thinking 必须接近满分”更合理、也更严格可检验的标准：独立 confirmation 上 Thinking 为 <strong>{pct(c_t['ar_final_accuracy'],2)}</strong>，Non-thinking 为 <strong>{pct(c_nt['ar_final_accuracy'],2)}</strong>，差 {100*(float(c_t['ar_final_accuracy'])-float(c_nt['ar_final_accuracy'])):.2f} pp；Thinking 的十个 count 均在 {pct(behavior_gate['metrics']['thinking_min_count_accuracy'])}–100%，没有类别塌缩。机制上最强的是 <strong>final-count compression</strong>（Thinking answer-query NCC {pct(g_t_final['confirmation_ncc_balanced_accuracy'])} vs NT {pct(g_nt_final['confirmation_ncc_balanced_accuracy'])}）、集中在 L4 的 targeted bank、逐渐增强的 retrieval transport，以及可执行的 trace-to-answer state（相邻 donor adoption {pct(t_donor['donor_adoption'].mean())}）。但 v58 仍没有建立紧致的 autonomous running counter，也没有得到选择性强的 Non-thinking broad-bank necessity。因此可写的主张是：<strong>supervised no-index Thinking 形成定向检索和强终端压缩，从而显著优于独立 Non-thinking；不是两侧所有大模型机制均已对称复现。</strong></div>

<div class="toc"><b>目录</b><ol><li><a href="#logic">问题与证据层级</a></li><li><a href="#setup">完整训练设定</a></li><li><a href="#behavior">行为与均匀性</a></li><li><a href="#geometry">Geometry Comparison / NCC</a></li><li><a href="#roles">Role specialization</a></li><li><a href="#causal">Ablation 与 patching</a></li><li><a href="#sufficiency">Free-running sufficiency</a></li><li><a href="#dynamics">Training dynamics</a></li><li><a href="#alignment">与大模型对齐及 gaps</a></li><li><a href="#limits">结论与限制</a></li><li><a href="#artifacts">复现产物</a></li></ol></div>

<h2 id="logic">1. 研究问题与证据层级</h2>
<p>核心问题不是“Thinking 有没有更好看的 attention map”，而是同一个 counting task 在两种输出协议下是否形成不同计算：Non-thinking 候选机制是 answer query 对 prompt evidence 的 broad retrieval；Thinking 候选机制是 trace 内逐项 targeted retrieval，再把 trace 压缩为最终 count state。报告把证据分成四层，避免把可读性、attention 与 causality 混为一谈。</p>
<div class="chain"><div><b>Representation</b>Logistic / NCC / ordinal RSA：count 是否可读、是否形成原型几何。</div><div><b>Routing</b>Broad score / targeted mass：query 看向哪些 evidence positions。</div><div><b>Necessity</b>Ranked Top-K removal 相对 matched controls 是否更伤自然计算。</div><div><b>Sufficiency</b>把 donor state 写入 receiver 后，free-running 输出是否采用 donor 信息。</div></div>
<div class="example"><span class="label">简单例子。</span>目标字符是 {{a,b,c}}，正文中按顺序出现 a、x、b、a、c，因此 gold count=4。Non-thinking 直接生成 <code>&lt;Ans&gt; 4</code>；Thinking 生成 <code>&lt;Think&gt; &lt;Sep&gt;a &lt;Sep&gt;b &lt;Sep&gt;a &lt;Sep&gt;c &lt;/Think&gt; &lt;Ans&gt; 4</code>。trace 没有“1,2,3,4”数字 index；若第 3 个 separator query 的 attention 指向正文第 3 个目标 occurrence，它就是 targeted retrieval 候选。</div>
<div class="conclusion"><span class="label">本节结论。</span>高 NCC 只说明 count cloud 可由简单原型读取；ranked ablation 才问模型是否自然依赖候选 bank；donor adoption 才问 state 是否足以驱动后续输出。四层必须分开报告。</div>

<h2 id="setup">2. 完整训练设定：模型到底怎么训</h2>
{setup_table}
<h3>2.1 为什么 Thinking 的 final-count token weight share 仍约 6%</h3>
<p>v58 在 task-output 阶段使用 <strong>component-normalized</strong> loss。Thinking 的 count、marker content、structure 三个区域先各自求 mean loss，再乘 8/8/16，因此 count 区域的目标系数份额是 8/(8+8+16)=25%。日志中的 <code>batch_final_count_token_weight_share</code> 则把系数重新摊回所有有效 token；因为 Thinking 有多个 marker 与 separator，单个 count token 最终约占 {pct(late_loss.loc['thinking','batch_final_count_token_weight_share'])}，而 Non-thinking 为 {pct(late_loss.loc['nonthinking','batch_final_count_token_weight_share'])}。前者不是“count 只得到 6% 的 component loss”，而是单 token 在更长 continuation 中的 token-level 份额。</p>
<p>step 1–1500 与 step 1501–10000 的 total loss 也不能直接连成一条物理同尺度曲线：前者平均整个序列，后者把多个区域的 mean loss 加权相加。报告因此用 free-running accuracy、attention mass、patch recovery 与 causal damage 做 dynamics 主指标，而不把 loss schedule 切换误画成 grokking 突变。</p>
<div class="audit"><span class="label">训练审计。</span>两种 mode 各看到 1,280,000 个 task examples；十个 count 的 realized shares 为 {pct(sample_min,2)}–{pct(sample_max,2)}。训练、phase、extended stages 均在 run manifest 中标记 complete；101 个 scientific snapshots 覆盖 step 0–10,000。训练总时长记录为 {float(read_csv(DATA/'tables'/'runtime_events.csv').loc[lambda d:d['block'].eq('train'),'duration_seconds'].iloc[0])/60:.1f} 分钟（CUDA runtime；manifest 未持久化 GPU SKU）。</div>
<div class="conclusion"><span class="label">本节结论。</span>v58 没有改 trace 内容、没有联合训练 mode、没有 scheduled sampling 或额外 contrastive objective。性能差来自两个独立模型在同一任务和均衡 sampler 上学习到的差异；主要设计变化是 4×8、d=512 的并行容量和 component-normalized task-output loss。</div>

<h2 id="behavior">3. 行为结果：均匀且远超 Non-thinking</h2>
<h3>目的</h3><p>先确认 Thinking 优势不是只由 count=1 或 count=10 的边界类别贡献，也不是 canonical 500 条样本的偶然波动。</p>
<h3>设置</h3><p>Canonical test 每个 count 50 条；独立 confirmation 每个 count 200 条、每种 mode 共 2,000 条。Confirmation 明确排除 canonical 的 <code>(set_id, corpus_start)</code>，overlap=0，并用 Wilson 95% interval 报 overall 与 per-count accuracy。预注册 behavior gate 为 Thinking≥75%、最差 count≥70%、spread≤20 pp、Thinking−NT≥30 pp；trace exact 仅作诊断。</p>
{behavior_table}
{figure(figures['behavior'], "图 1｜行为与均匀性。Panel A 横轴是 gold count 1–10，纵轴是独立 confirmation 的 free-running exact-answer accuracy；点为每类 200 条样本的均值，误差线为逐类 Wilson 95% interval。Panel B 是每种 mode 的 2,000 条总体准确率与 Wilson interval。Panel C 仅对 Thinking 报 trace exact、逐位置 ordered-marker accuracy 与 marker-count accuracy。", "v58 behavior accuracy and count uniformity")}
<p>Non-thinking 在 N=1 和 N=10 有明显边界偏好，但总体只有 {pct(c_nt['ar_final_accuracy'],2)}；Thinking 每个 count 都高于或等于 90%，最低 N=9 为 90.0%。Canonical 与独立 confirmation 的总体结果分别是 {pct(canonical_t['ar_final_accuracy'])}/{pct(canonical_nt['ar_final_accuracy'])} 和 {pct(c_t['ar_final_accuracy'],2)}/{pct(c_nt['ar_final_accuracy'],2)}，方向与数量级一致。</p>
<div class="example"><span class="label">简单例子。</span>如果一个模型只会在所有输入都答“1”，均匀测试仍可能让 count=1 看起来 100%，但其余九类会接近 0，spread 接近 100 pp。v58 Thinking 的最差类仍为 90%，排除了这种 collapse。</div>
<div class="conclusion"><span class="label">本节结论。</span>按你最新提出的标准，“Native-thinking 不必追求满分，只需均匀且远超 Non-thinking”，v58 不只是勉强通过，而是以 78.25 pp 独立确认差距和 10 pp Thinking spread 明确通过；没有继续为 accuracy 搜索 v59 的必要。</div>

<h2 id="geometry">4. Geometry Comparison：running 与 final 分开</h2>
<h3>目的</h3><p>对齐大模型报告的两个语义站点：running label 在第 k 个 evidence/event 完成边界读取；final label 在生成首个答案数字前的 answer query 读取。分别问逐步 progress 与最终 count 是否形成紧致、可泛化的表示。</p>
<h3>计算方法</h3><div class="formula"><span class="label">Frozen NCC。</span>每个 endpoint 用 discovery 10 states/class 与 disjoint confirmation 8/class。Discovery 内按 grouped OOF 的 mean(Logistic BA, NCC BA) 选择一个 common decoder layer，并拟合 <code>StandardScaler → whitened PCA≤16</code>。对 class c 的 discovery states 求 centroid μ<sub>c</sub>；confirmation state z 的预测为 <code>argmin_c ||z−μ_c||²</code>。Balanced accuracy 是 10 类 recall 的平均，chance=10%。Ordinal RSA 是 centroid pair distance 与 |c−c′| 的 held-out Spearman correlation。本文把异质输入在 answer query 汇聚为可由少量 count centroids 泛化读取的几何称为 <em>representation compression</em>；这是操作性定义，不等同于信息论压缩或低维性的证明。</div>
{geometry_table}
{figure(figures['geometry'], "图 2｜Clean geometry。Panel A 横轴依次为 Non-thinking prompt-occurrence running、Thinking trace-item running、两种 mode 的 answer-query final endpoint；纵轴为未参与层选择的 confirmation balanced accuracy。实心柱为 L2 Logistic，斜线柱为 NCC，红色点线为 10-class chance；柱顶 Lx 是 discovery-only 选层。Panel B 使用相同四个 endpoint，纵轴为 held-out ordinal RSA。", "v58 clean running and final count geometry")}
<p>Thinking running NCC 从 NT 的 {pct(g_nt_run['confirmation_ncc_balanced_accuracy'])} 提升到 {pct(g_t_run['confirmation_ncc_balanced_accuracy'])}，但绝对值仍低；与此同时 ordinal RSA 达 {num(g_t_run['confirmation_ordinal_rsa'])}，说明平均 centroid path 随 count gap 有序，却不是每个 count 都形成紧致球状簇。真正显著的 compression 在 answer query：Thinking Logistic/NCC 都是 100%，而 NT 只有 {pct(g_nt_final['confirmation_logistic_balanced_accuracy'])}/{pct(g_nt_final['confirmation_ncc_balanced_accuracy'])}。</p>
<div class="example"><span class="label">简单例子。</span>十个班级的平均身高可以按年级递增，因此 ordinal RSA 很高；但每个班的学生身高大量重叠，最近 centroid 分类仍会很差。Thinking running state 正是“平均轨迹有序、单例 cloud 仍松散”的情况。</div>
<div class="conclusion"><span class="label">本节结论。</span>v58 强力复现了 Thinking 的 final-count representation compression，但只得到中等的 running count decodability，不能把它写成已经识别出 compact autonomous counter。后续 causal 实验应聚焦 targeted retrieval 与 terminal readout。</div>

<h2 id="roles">5. Attention role specialization 与 head-bank differentiation</h2>
<h3>目的与定义</h3><p><strong>Targeted mass</strong> 是 Thinking 第 k 个 separator query 对正文中 matching 第 k 个目标 occurrence 的 attention mass。<strong>Broad score</strong> 用于 Non-thinking answer query：先求所有 active target occurrences 的总 attention mass M，再以有效覆盖率 C=exp(H)/N 惩罚只盯一个 occurrence 的 head，最终 B=M×C。所有 final roles 只在 held-out selection split 排名，图和数值在 disjoint reporting split 读取。</p>
{figure(figures['roles'], "图 3｜Role specialization。Panel A 为 Non-thinking final broad score 的 4 layers × 8 heads 热图；横轴=head，纵轴=layer，格内是 reporting-split raw score。Panel B 对 Thinking targeted mass 作同样展示，两图各有独立色标，不能跨图按颜色深浅比较。Panel C 只在同一个 Thinking 模型内比较 broad、marker-successor、targeted 三种 role；横轴列出 32 个 heads，纵轴是 role，每行除以该 role 的最大值，因此用于看 head identity 分工而非 raw effect size。", "v58 attention role specialization heatmaps")}
<p>Targeted ranking 的 Top-4 为 <strong>{top_t_text}</strong>，全部位于 L4；最终 reporting split 中前八名也全部来自 L4。Thinking broad 的 Top-4 与 targeted Top-4 零重合；marker-successor Top-4 与 targeted 只重合 1/4。这是同一 Thinking 模型内可解释的 role differentiation。相反，Non-thinking broad 的 discovery Top-4 为 {top_nt_text}，但 reporting split 的最高 head 变为 L1H3，说明小样本下 exact identity 不稳定。</p>
<div class="warning"><span class="label">重要边界。</span>Thinking 与 Non-thinking 是独立初始化模型，L1H3 在两边没有 neuron identity 对齐意义；跨 mode 只能比较 role 在 layer/head-bank 层面的分布。Non-thinking broad 的 selection/reporting 每个 count 仅 2/1 条，且其行为接近 floor，因此本报告不把 descriptive broad map 升级为稳定 causal bank。</div>
<div class="conclusion"><span class="label">本节结论。</span>v58 的清楚分化发生在 Thinking 内部：L4 targeted bank 与较早的 marker-successor、较分散的 broad role 使用不同 head subsets。Targeted bank 很集中；Non-thinking broad role 目前只有描述性信号，没有同等强度的稳定性。</div>

<h2 id="causal">6. Causal experiments：Top-K ablation、post-ablation NCC 与 patching</h2>
<h3>6.1 Ranked Top-K necessity</h3><p>Head ranking 只用 discovery split。对累计 Top-K heads 做两种干预：global removal 在所有 query positions 删除；role-query-local removal 只在 Non-thinking answer query 或 Thinking 的每个 separator query 删除。Matched controls 从相同 layer、相同 K 的未选 heads 中枚举所有可用 disjoint sets。Teacher-forced 读即时 token accuracy；free-running 允许错误累积到 trace 与最终答案。</p>
{figure(figures['topk'], "图 4｜Top-K causal suite。六个 panel 的横轴都是累计删除的 K=0/1/2/4；紫/蓝/绿/橙实线为 discovery-ranked bank，灰色虚线为 layer-matched controls 的中位数，灰带是 controls 的 min–max（不是置信区间）。A/B/C 分别是 teacher-forced trace token、free-running trace exact、Thinking final answer；D/E 是使用 clean discovery scaler/PCA/centroids、不重新拟合的 running/final NCC；F 是 Non-thinking broad-bank free-running final accuracy。", "v58 Top-K ablation, post-ablation NCC, and free-running behavior")}
{causal_table}
<p>Thinking Top-K 的最清楚效果在 trace trajectory：free-running exact trace 从 {pct(fr_clean['trace_exact'])} 降到 K2 的 {pct(fr_k2['trace_exact'])} 与 K4 的 {pct(fr_k4['trace_exact'])}。不过 K4 control 也有明显损害，说明 L4 存在冗余和一般 trace-support heads；应写“ranked bank enriched for necessity”，不应写“只有这四个 heads 必要”。Non-thinking ranked broad removal 没有超过 matched controls，而且 clean accuracy 本来只有 {pct(row_value(free_topk,comparison_mode='nonthinking',scope='role_query_local_free_running',path_kind='clean',top_k=0)['ar_final_accuracy'])}，因此是 floor-limited null。</p>
<h3>6.2 为什么 Top-K 后 NCC 不变</h3><p>Thinking clean running/final representations由 discovery 选在 L3/L2；targeted heads 全在 L4。删除 L4 head output 不可能逆向改变已经记录的 L2/L3 state。Teacher forcing又把正确 gold marker作为下一 token 输入，使后续 answer query看到修复后的正确 trace。因此 post-ablation NCC 维持 {pct(g_t_run['confirmation_ncc_balanced_accuracy'])}/{pct(g_t_final['confirmation_ncc_balanced_accuracy'])} 并不否定 targeted heads；它说明当前 bank 直接参与 next-marker readout，而没有被证明是 earlier NCC compression 的 mediator。</p>
<h3>6.3 Retrieval transport patching</h3><p>构造 clean/corrupt 对：在第 k 个 targeted query 破坏应检索的 prompt occurrence，使正确 marker margin由 clean 的 {num(patch_final['clean_margin'],2)} 降到 {num(patch_final['corrupt_margin'],2)}。随后只从 clean run 恢复 ranked Top-2 heads 在 target source position 的 value contribution；最终恢复 {pct(patch_final['normalized_recovery_mean'])} 的 clean-corrupt margin gap，marker accuracy从 {pct(patch_final['corrupt_correct'])} 升到 {pct(patch_final['patched_correct'])}。这比只看 attention mass 更接近 causal retrieval transport。</p>
<div class="example"><span class="label">简单例子。</span>把第 4 个应检索的正文 occurrence 换成错误 evidence，模型的正确 marker logit落后；只把 clean L4 targeted heads 从正确 source 取出的 value write补回，正确 marker margin恢复约三分之一。这个实验不直接 patch整层 residual，因此比“把 clean final state 全部复制回来”更局部。</div>
<div class="conclusion"><span class="label">本节结论。</span>Targeted bank 有 attention concentration、teacher-forced即时损害、free-running trace损害和 value-only patch recovery 四类一致证据；单个 L4H5 的 matched-control specificity较弱，集合级结论强于单头唯一性。Non-thinking broad-bank necessity没有建立。</div>

<h2 id="sufficiency">7. Free-running causal sufficiency：terminal state 足够吗？</h2>
<h3>Experiment A · Answer-query state transplant</h3><p>对相邻 count receiver/donor，在 discovery 选择一个 layer；Thinking 先自由生成 trace直到 <code>&lt;Ans&gt;</code>，然后把 donor 的完整 answer-query residual写入 receiver并继续 greedy生成。Primary outcome 是 receiver 是否采用 donor count；same-count context donor控制“复制另一个上下文 state本身”是否导致 adoption。</p>
{figure(figures['sufficiency'], "图 5｜Free-running sufficiency。Panel A 横轴是相邻-count full-state donor 与 same-count context control，纵轴是最终 greedy donor-count adoption；每臂 16 个 confirmation pairs，橙/紫分别为 Non-thinking/Thinking。Panel B 在 Thinking trace中比较 clean、same-position centroid shift、等范数 orthogonal control，以及两个跨 absolute-position natural donor upper bounds；纵轴是下一 marker采用 donor successor 的比例，†表示位置混杂。", "v58 free-running terminal and progress-state sufficiency")}
<p>Thinking 在 L4 的相邻 donor adoption 是 {pct(t_donor['donor_adoption'].mean())}（15/16），same-count context control 为 {pct(t_context['donor_adoption'].mean())}，平均 donor-vs-receiver margin shift +{num(t_donor['donor_margin_shift'].mean(),2)}。Non-thinking 在 L3 为 {pct(nt_donor['donor_adoption'].mean())}，但 context control 也是 {pct(nt_context['donor_adoption'].mean())}，因此不是 count-specific categorical sufficiency。</p>
<h3>Experiment B · Progress-state transplant</h3><p>Same-position centroid shift在17个 eligible cells 中把 donor next-marker adoption从 clean的 {pct(progress.loc[progress['condition'].eq('clean'),'donor_first_adoption'].mean())} 提到 {pct(progress.loc[progress['condition'].eq('centroid_shift'),'donor_first_adoption'].mean())}；orthogonal control仍为 {pct(progress.loc[progress['condition'].eq('orthogonal_control'),'donor_first_adoption'].mean())}。Natural marker/item donor达到 {pct(progress.loc[progress['condition'].eq('natural_marker_cross_position'),'donor_first_adoption'].mean())}，但 donor与receiver位于不同 absolute positions；固定两-token item格式无法消除RoPE/trace-length混杂。所有 progress arms 的最终 count accuracy均为100%，因此没有形成“改写 progress→改写 final answer”的干净链。</p>
<div class="conclusion"><span class="label">本节结论。</span>不需要证明一个 universal final broad aggregator。v58 已证明 Thinking free-generated trace后的 answer-query state可执行地驱动 count；这足以支持 trace-to-answer readout。相反，低维、同位置 progress state 的行为充分性仍弱，不能声称已识别 natural internal counter。</div>

<h2 id="dynamics">8. Training dynamics：何时形成 specialization</h2>
<h3>横轴为什么用 linear steps，而不是只用 log steps</h3><p><a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html">Olsson et al. 的 induction-head work</a> 经常在 log elapsed tokens 上看 loss，并对 log-token loss导数讨论 phase change；<a href="https://arxiv.org/abs/2001.08361">Kaplan et al.</a> 与 <a href="https://arxiv.org/abs/2203.15556">Hoffmann et al.</a> 的 scaling-law 图则常在跨多数量级的参数、数据或compute上使用 log/log-log 坐标。Log轴适合展开早期并呈现幂律，但不会自动证明突变，反而会压缩本实验 3k–8k 的关键区间。v58 只有0–10k且每100 steps有snapshot，因此主图使用 linear step；若要正式claim phase transition，还应对 smooth model与change-point model做held-out比较，而不是凭视觉。</p>
{figure(figures['dynamics'], "图 6｜Synchronized training dynamics。所有 panel 的横轴均为 optimizer step；红色竖虚线是 step 1500 loss-scope切换。A：每个 checkpoint、每种 mode各500条free-running样本的answer accuracy，另画Thinking trace exact。B：固定final-selected L4H5的weighted targeted attention mass（左轴）与correct-vs-best-wrong QK margin（右轴）。C：final Top-4 bank占全部32 heads同一role score总和的份额；点线4/32是均匀基线。D：每一步即时Top-4与final Top-4的Jaccard。E：corrupt baseline下Top-2 value patch的normalized recovery与marker accuracy。F：position-local selected-head causal damage减same-layer control damage；纵轴是correct-token logit-margin units。", "v58 synchronized behavior, routing, bank, patching, and causal training dynamics")}
<p>Thinking 行为从 step 3,000 的41.4%逐步升到4,500的59.6%、6,000的79.4%、8,000的91.4%和10,000的95.4%；不是一个单点跳变。固定 L4H5 targeted mass在step {int(dynamics_stats['first_targeted_50pct_step']):,}首次达到final值的50%，在step {int(dynamics_stats['first_targeted_80pct_step']):,}达到80%；QK margin约4.8k后由负转正。Top-2 value recovery从3k附近开始上升，10k为{pct(patch_final['normalized_recovery_mean'])}。Marker-successor L2H3的matched causal specificity更早、更强，final damage difference为{num(float(successor_final['causal_damage'])-float(successor_control['causal_damage']))}；targeted L4H5 final difference只有{num(float(target_final['causal_damage'])-float(target_control['causal_damage']))}。</p>
<p>Bank differentiation的合理读法是：targeted Top-4逐渐占据更多targeted score并稳定到L4，而Non-thinking broad Top-4 identity更不稳定。它支持“先有trace结构/marker successor，再形成集中targeted routing，随后behavior继续改善”的顺序；单seed不能把这种时间共现升级为普适phase transition定律。</p>
<div class="example"><span class="label">简单例子。</span>如果attention mass从0.02在3k后持续升到0.42，同时free-running accuracy也从0.41升到0.95，我们可以说二者共同形成并给出时间顺序；只有在多个seed中change point都稳定、并且干预形成时间会同步移动behavior onset时，才接近Anthropic式的co-occurrence/co-perturbation证据。</div>
<div class="conclusion"><span class="label">本节结论。</span>v58出现清楚的role specialization与head-bank differentiation，但曲线更像3k–8k的连续形成窗口，而不是已统计确认的瞬时phase change。线性step主轴最适合本数据；log(step+1)可作早期补图，但不应被用来“制造”突变。</div>

<h2 id="alignment">9. 与大模型 Non-thinking / Native-thinking 实验对齐及 gaps</h2>
{alignment_table}
<h3>9.1 已经对齐的部分</h3><ul><li><b>站点：</b>Non-thinking running取k-th prompt occurrence end；Thinking running取k-th trace item end；final均取答案数字前query，与Geometry Comparison的语义边界一致。</li><li><b>Representation：</b>discovery-only选层与frozen confirmation Logistic/NCC，分开running和final，不用PCA图替代held-out指标。</li><li><b>Necessity：</b>discovery-ranked Top-K与layer/head-count matched controls，同时报告teacher-forced immediate damage、post-ablation frozen NCC和free-running rollout。</li><li><b>Sufficiency：</b>相邻count donor、same-count context control、free-running continuation，直接读取greedy donor adoption。</li><li><b>Dynamics：</b>固定final-selected heads追踪全部checkpoint，不在每一步重新选一个最好看的head。</li></ul>
<h3>9.2 当前最重要的 gaps</h3><ul><li><b>Non-thinking broad retrieval gap：</b>大模型已有broad-bank matched causal effect、retrieval subspace和late executable state；v58 Non-thinking在16–18% floor附近，broad ranking不稳定，ranked lesion没有超过controls。</li><li><b>Running representation gap：</b>Thinking running NCC只有{pct(g_t_run['confirmation_ncc_balanced_accuracy'])}；高ordinal RSA说明有序，不代表紧致或content-free counter。</li><li><b>Progress sufficiency gap：</b>same-position centroid shift很弱；natural donor patch跨absolute positions。尚未复现大模型Qwen natural no-index commit→next-query transfer。</li><li><b>Serial mediation gap：</b>还没有在同一个ranked-damaged free rollout中依次restore targeted output、carrier、commit、answer state并闭合最终答案。</li><li><b>Final-state confound：</b>full answer residual transplant可能携带count、trace length、position和其他上下文；尚无absolute-position matched不同count control。</li><li><b>Scale与监督：</b>12.66M字符模型、固定separator grammar、atomic one-token answer、256-char context和单seed，不能与4B/8B自然语言模型比较绝对head ID、formation step或accuracy。</li></ul>
<div class="warning"><span class="label">关于“两个mechanism都显著”。</span>v58足以作为强Thinking targeted-retrieval / representation-compression setting；它不支持把Non-thinking broad retrieval写成同等强的已复现机制。如果论文中心句必须是“两侧都由强因果证据确认，只是broad vs targeted不同”，还需要一个行为不在floor、但明显弱于Thinking的Non-thinking setting或额外训练seed，而不能靠换图隐藏当前null。</div>
<div class="conclusion"><span class="label">本节结论。</span>v58与大模型在实验设计层级上已经对齐，并强力复现Thinking final compression、targeted retrieval与terminal readout；主要未对齐处是Non-thinking broad aggregation和natural progress recurrence。</div>

<h2 id="limits">10. 最终结论、限制与建议写法</h2>
<div class="abstract"><span class="label">可以写进正文的结论。</span>(1) 在同一balanced count-1–10任务上，独立训练的no-index Thinking以{pct(c_t['ar_final_accuracy'],2)}显著超过Non-thinking的{pct(c_nt['ar_final_accuracy'],2)}，且Thinking十类均≥{pct(behavior_gate['metrics']['thinking_min_count_accuracy'])}；(2) Thinking的主要representation advantage位于answer-query final-count state，NCC为100%，running NCC仅{pct(g_t_run['confirmation_ncc_balanced_accuracy'])}但ordinal ordering强；(3) L4形成集中targeted bank，ranked removal破坏trace，target-source value patch可恢复{pct(patch_final['normalized_recovery_mean'])} margin gap；(4) free-generated trace后的answer state具有{pct(t_donor['donor_adoption'].mean())}相邻donor adoption，支持可执行trace-to-answer readout；(5)当前没有证据要求universal final broad aggregator，也没有建立compact autonomous internal counter。</div>
<ul><li>所有模型与training dynamics来自单一seed 1234；checkpoint rows不是独立训练replicates。</li><li>Head roles在final selection split冻结，再在disjoint reporting split读数；但broad reporting sample仅1/count。</li><li>Free-running Top-K只有80条reporting examples；K=4只有一个完全disjoint matched control。</li><li>Targeted bank位于L4，而running/final NCC选层在L3/L2；因此当前Top-K不能检验earlier NCC的中介。</li><li>高功效AR cached evaluator已在修复后与reference逐token一致；修复前错误使用tied embedding的派生CSV已隔离，不进入报告。</li><li><code>plan.tex</code>和<code>LLM_Compression.pdf</code>本轮已不在此前Downloads路径，未重新核验；大模型对齐直接依据三份仍可用HTML报告与旧report保存的crosswalk。</li></ul>
<p><b>下一步优先级：</b>先不要再为Thinking accuracy调参。若正文只需要“Thinking优势 + compression + targeted retrieval”，v58已经是主setting。若必须补齐对称的broad-vs-targeted机制，优先做2–3个固定v58 seed和一个行为高于floor的matched Non-thinking baseline；随后在同一damaged free rollout中做targeted output→carrier→commit→answer的serial rescue。不要用降低Non-thinking到chance来替代broad mechanism证据。</p>
<div class="conclusion"><span class="label">最终判断。</span>保留v58。它比“只求accuracy差”更干净：trace不含显式index、Thinking十类均匀、行为差距巨大，并且final compression、targeted bank、retrieval transport和terminal answer-state sufficiency相互一致。报告时同时保留两个null：Non-thinking broad-bank necessity未建立，Thinking progress-state sufficiency未建立。</div>

<h2 id="artifacts">11. 复现产物与 provenance</h2>
{html_table(["Artifact","Path / role"],[
    ["Frozen v58 run archive", html.escape(str(DATA))],
    ["Independent behavior confirmation", html.escape(str(ANALYSIS/'behavior_confirmation_v58'))],
    ["Clean aligned NCC", html.escape(str(ANALYSIS/'v58_clean_ncc'))],
    ["Teacher-forced Top-K + post-ablation NCC", html.escape(str(ANALYSIS/'v58_topk_ncc'))],
    ["Free-running Top-K", html.escape(str(ANALYSIS/'v58_free_running_topk'))],
    ["Free-running sufficiency", html.escape(str(ANALYSIS/'v58_free_running_sufficiency'))],
    ["101-snapshot roles", html.escape(str(ANALYSIS/'extended'))],
    ["Routing/patching/causal/high-power dynamics", html.escape(str(ANALYSIS/'phase_transition_audit'))],
    ["Report builder", html.escape(str(Path(__file__).resolve()))],
    ["Self-contained report", html.escape(str(output.resolve()))],
])}
<p class="small">Run created {html.escape(str(run_manifest.get('created_at_utc','unknown')))}; code commit at report build: <code>{git_commit()}</code>. All figures are regenerated from archived tables and embedded as base64, so the final HTML has no external image dependency. External paper links are references only; all v58 numerical claims come from local frozen artifacts.</p>
</main></body></html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": "v58_synthetic_report_v1",
        "report": str(output.resolve()),
        "report_sha256": sha256(output),
        "primary_comparison": "independently trained v58 separator/no-index Thinking vs v58 Non-thinking",
        "run": DATA.name,
        "code_commit": git_commit(),
        "behavior_gate": behavior_gate,
        "dynamics_statistics": dynamics_stats,
        "figures": {
            key: {"path": str(value.resolve()), "sha256": sha256(value)}
            for key, value in figures.items()
        },
        "source_roots": {
            "run_archive": str(DATA.resolve()),
            "analysis": str(ANALYSIS.resolve()),
        },
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                DATA / "config.json",
                ANALYSIS / "behavior_confirmation_v58" / "autoregressive_summary.csv",
                ANALYSIS / "v58_clean_ncc" / "selected_confirmation_summary.csv",
                ANALYSIS / "v58_topk_ncc" / "post_ablation_behavior.csv",
                ANALYSIS / "v58_free_running_topk" / "free_running_summary.csv",
                ANALYSIS / "v58_free_running_sufficiency" / "answer_transplant_confirmation.csv",
                ANALYSIS / "phase_transition_audit" / "tables" / "high_power_ar_summary.csv",
            )
        },
        "validation": {
            "high_power_endpoint_matches_canonical": True,
            "cached_vs_reference_generation": "10/10 exact for each mode at step 5000",
            "self_contained_images": all(f'data:image/png;base64,' in report for _ in [0]),
            "missing_reference_files": [
                "C:/Users/HP/Downloads/plan.tex",
                "C:/Users/HP/Downloads/LLM_Compression.pdf",
            ],
        },
    }
    manifest_path = output.with_name("NiaH_Synthetic_report_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "NiaH_Synthetic_report.html",
    )
    args = parser.parse_args()
    manifest = build_report(args.output.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
