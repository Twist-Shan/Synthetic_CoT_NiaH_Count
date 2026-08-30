#!/usr/bin/env python3
"""Build Anthropic-style training-dynamics figures for the v20 run.

The figures align macroscopic behavior, per-head role formation, routing
quality, and causal dependence on the same optimizer-step axis.  They are
descriptive for the frozen seed-1234 run; fitted 10--90% windows are shown as
formation intervals, not claimed as universal phase transitions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "colab_results" / "v20_main_RoPE_count1-30_seed1234"
ROLE_ORDER = (
    "nonthinking_broad",
    "thinking_broad",
    "targeted_retrieval",
    "marker_successor",
)
ROLE_LABEL = {
    "nonthinking_broad": "Nonthinking broad",
    "thinking_broad": "Thinking broad",
    "targeted_retrieval": "Thinking targeted",
    "marker_successor": "Thinking successor",
}
ROLE_COLOR = {
    "nonthinking_broad": "#4C78A8",
    "thinking_broad": "#F28E2B",
    "targeted_retrieval": "#8C6BB1",
    "marker_successor": "#2A9D8F",
}
MODE_COLOR = {"nonthinking": "#4C78A8", "thinking": "#F28E2B"}
LAYER_COLOR = {1: "#4C78A8", 2: "#59A14F", 3: "#E15759", 4: "#B279A2"}
OBJECTIVE_SWITCH = 1500


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _weighted_rows(
    frame: pd.DataFrame,
    group_columns: Iterable[str],
    metric_columns: Iterable[str],
    weight_column: str = "observations",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    group_columns = list(group_columns)
    metric_columns = list(metric_columns)
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in frame.groupby(grouper, sort=True, dropna=False):
        keys = (key,) if len(group_columns) == 1 else tuple(key)
        row: dict[str, float | int | str] = dict(zip(group_columns, keys, strict=True))
        weights = pd.to_numeric(group[weight_column], errors="coerce").to_numpy(dtype=float)
        valid_weight = np.isfinite(weights) & (weights > 0)
        row["observations"] = int(weights[valid_weight].sum())
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            valid = valid_weight & np.isfinite(values)
            row[metric] = (
                float(np.average(values[valid], weights=weights[valid]))
                if valid.any()
                else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _wilson_interval(successes: float, observations: int) -> tuple[float, float]:
    if observations <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    estimate = float(successes) / observations
    denominator = 1.0 + z * z / observations
    center = (estimate + z * z / (2.0 * observations)) / denominator
    radius = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / observations
            + z * z / (4.0 * observations * observations)
        )
        / denominator
    )
    return center - radius, center + radius


def _step_axis(axis: plt.Axes) -> None:
    axis.axvline(
        OBJECTIVE_SWITCH,
        color="#303030",
        linestyle=":",
        linewidth=1.1,
        zorder=0,
    )
    axis.set_xlim(-150, 10_150)
    axis.set_xticks([0, 1500, 3000, 5000, 7000, 10_000])
    axis.set_xticklabels(["0", "1.5k", "3k", "5k", "7k", "10k"])
    axis.set_xlabel("optimizer step")


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.11,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _fixed_role_lines(attention: pd.DataFrame) -> pd.DataFrame:
    fixed = attention[attention["is_fixed_role_head"].astype(float) == 1.0].copy()
    duplicate_counts = fixed.groupby(["step", "role"]).size()
    if not duplicate_counts.eq(1).all():
        raise ValueError("expected exactly one frozen head per role and step")
    return fixed.sort_values(["role", "step"])


def _fit_window(fits: pd.DataFrame, role: str) -> tuple[float, float] | None:
    selected = fits[
        (fits["evidence_family"] == "attention_role")
        & (fits["metric"] == "fixed_head_role_score")
        & (fits["group"] == role)
    ]
    if selected.empty:
        return None
    row = selected.iloc[0]
    center = float(row["smooth_center_x"])
    width = float(row["smooth_width_10_90"])
    if not np.isfinite(center) or not np.isfinite(width):
        return None
    return max(0.0, center - width / 2.0), min(10_000.0, center + width / 2.0)


def _shade_formation_window(
    axis: plt.Axes,
    fits: pd.DataFrame,
    role: str,
) -> None:
    window = _fit_window(fits, role)
    if window is None:
        return
    axis.axvspan(
        window[0],
        window[1],
        color=ROLE_COLOR[role],
        alpha=0.075,
        linewidth=0,
        zorder=0,
    )


def _behavior_frame(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dense = _read(
        run_dir / "analysis/phase_transition/tables/dense_behavior_by_count.csv"
    )
    dense = dense[dense["outcome"] == "final_answer_teacher_forced_exact"]
    teacher_forced = _weighted_rows(
        dense,
        ("step", "mode"),
        ("accuracy",),
    )

    high_power = _read(
        run_dir / "analysis/phase_transition_audit/tables/high_power_ar_summary.csv"
    )[["step", "mode", "examples", "successes", "ar_accuracy"]].copy()
    final = _read(run_dir / "tables/final_autoregressive_summary.csv").copy()
    final = final.rename(
        columns={
            "ar_final_accuracy": "ar_accuracy",
            "ar_final_accuracy_wilson95_low": "wilson_low",
            "ar_final_accuracy_wilson95_high": "wilson_high",
        }
    )
    final["successes"] = final["ar_accuracy"] * final["examples"]
    high_power["wilson_low"] = np.nan
    high_power["wilson_high"] = np.nan
    for index, row in high_power.iterrows():
        low, high = _wilson_interval(float(row["successes"]), int(row["examples"]))
        high_power.loc[index, ["wilson_low", "wilson_high"]] = (low, high)
    autoregressive = pd.concat(
        (
            high_power,
            final[
                [
                    "step",
                    "mode",
                    "examples",
                    "successes",
                    "ar_accuracy",
                    "wilson_low",
                    "wilson_high",
                ]
            ],
        ),
        ignore_index=True,
    ).sort_values(["mode", "step"])
    return teacher_forced, autoregressive


def _routing_summary(run_dir: Path) -> pd.DataFrame:
    routing = _read(
        run_dir / "analysis/phase_transition_audit/tables/routing_qk_by_k.csv"
    )
    return _weighted_rows(
        routing,
        ("step",),
        ("targeted_mass", "qk_margin", "correct_occurrence_top1"),
    ).sort_values("step")


def plot_overview(
    run_dir: Path,
    output: Path,
    attention: pd.DataFrame,
    fits: pd.DataFrame,
) -> None:
    teacher_forced, autoregressive = _behavior_frame(run_dir)
    fixed = _fixed_role_lines(attention)
    routing = _routing_summary(run_dir)
    causal = _read(
        run_dir / "analysis/phase_transition_audit/tables/local_head_causal_damage.csv"
    )
    if "causal_damage" not in causal:
        causal["causal_damage"] = -causal["margin_change_from_baseline"]

    figure, axes = plt.subplots(4, 1, figsize=(11.2, 14.2), sharex=True)

    axis = axes[0]
    for mode in ("nonthinking", "thinking"):
        color = MODE_COLOR[mode]
        tf = teacher_forced[teacher_forced["mode"] == mode]
        axis.plot(
            tf["step"],
            tf["accuracy"],
            color=color,
            linestyle="--",
            linewidth=1.3,
            alpha=0.72,
            label=f"{mode.title()} teacher-forced",
        )
        ar = autoregressive[autoregressive["mode"] == mode]
        y = ar["ar_accuracy"].to_numpy(dtype=float)
        low = ar["wilson_low"].to_numpy(dtype=float)
        high = ar["wilson_high"].to_numpy(dtype=float)
        axis.errorbar(
            ar["step"],
            y,
            yerr=np.vstack((y - low, high - y)),
            color=color,
            marker="o",
            markersize=4.2,
            capsize=2.5,
            linewidth=2.0,
            label=f"{mode.title()} AR (95% Wilson)",
            zorder=4,
        )
        final = ar[ar["step"] == 10_000]
        axis.scatter(
            final["step"],
            final["ar_accuracy"],
            marker="*",
            s=115,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
    axis.set_title("Macroscopic behavior")
    axis.set_ylabel("final-count exact accuracy")
    axis.set_ylim(-0.03, 1.04)
    axis.legend(ncol=2, loc="upper left", fontsize=8.2)
    _panel_label(axis, "A")

    axis = axes[1]
    for role in ROLE_ORDER:
        line = fixed[fixed["role"] == role]
        axis.plot(
            line["step"],
            line["score"],
            color=ROLE_COLOR[role],
            linewidth=2.1,
            label=ROLE_LABEL[role],
        )
    _shade_formation_window(axis, fits, "marker_successor")
    _shade_formation_window(axis, fits, "targeted_retrieval")
    axis.set_title("Frozen-head role formation")
    axis.set_ylabel("role score")
    axis.set_ylim(-0.03, 1.04)
    axis.legend(ncol=2, loc="center right", fontsize=8.2)
    _panel_label(axis, "B")

    axis = axes[2]
    axis.plot(
        routing["step"],
        routing["targeted_mass"],
        color=ROLE_COLOR["targeted_retrieval"],
        linewidth=2.1,
        label="correct-occurrence attention mass",
    )
    axis.plot(
        routing["step"],
        routing["correct_occurrence_top1"],
        color="#D45087",
        linewidth=1.8,
        label="correct occurrence is top-1",
    )
    axis.set_title("Targeted-routing fidelity")
    axis.set_ylabel("mass / probability")
    axis.set_ylim(-0.03, 1.04)
    margin_axis = axis.twinx()
    margin_axis.plot(
        routing["step"],
        routing["qk_margin"],
        color="#5B5B5B",
        linestyle="--",
        linewidth=1.8,
        label="QK correct − best wrong",
    )
    margin_axis.axhline(0.0, color="#5B5B5B", linewidth=0.8, alpha=0.7)
    margin_axis.set_ylabel("QK margin")
    handles, labels = axis.get_legend_handles_labels()
    second_handles, second_labels = margin_axis.get_legend_handles_labels()
    axis.legend(handles + second_handles, labels + second_labels, loc="upper left", fontsize=8.2)
    _panel_label(axis, "C")

    axis = axes[3]
    for role in ("targeted_retrieval", "marker_successor"):
        for intervention, linestyle, suffix in (
            ("fixed_head_zero", "-", "frozen rank-1"),
            ("same_layer_control_zero", "--", "same-layer peer"),
        ):
            line = causal[
                (causal["role"] == role) & (causal["intervention"] == intervention)
            ].sort_values("step")
            if line.empty:
                continue
            head = line.iloc[0]
            head_label = f"L{int(head['layer'])}H{int(head['head'])}"
            axis.plot(
                line["step"],
                line["causal_damage"],
                color=ROLE_COLOR[role],
                linestyle=linestyle,
                marker="o" if intervention == "fixed_head_zero" else None,
                markersize=3.2,
                linewidth=2.0 if intervention == "fixed_head_zero" else 1.4,
                alpha=1.0 if intervention == "fixed_head_zero" else 0.72,
                label=f"{ROLE_LABEL[role]} {head_label} · {suffix}",
            )
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_title("Local causal use at the role query positions")
    axis.set_ylabel("−Δ correct-token logit margin")
    axis.legend(ncol=2, loc="upper left", fontsize=7.7)
    _panel_label(axis, "D")

    for index, axis in enumerate(axes):
        _step_axis(axis)
        axis.grid(True, color="#D7D7D7", linewidth=0.55, alpha=0.7)
    figure.suptitle(
        "v20 training dynamics: behavior → attention roles → routing → causal use",
        fontsize=15,
        y=0.995,
    )
    figure.text(
        0.5,
        0.977,
        "RoPE · count 1–30 · seed 1234 · dotted line: objective switch · shading: fitted 10–90% formation intervals",
        ha="center",
        va="top",
        fontsize=9,
        color="#4A4A4A",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.962), h_pad=2.0)
    _save_figure(figure, output)


def plot_broad_decomposition(
    output: Path,
    attention: pd.DataFrame,
) -> None:
    required = {
        "total_target_mass",
        "effective_coverage",
        "broad_score",
        "legacy_entropy_broad_score",
    }
    missing = required.difference(attention.columns)
    if missing:
        raise ValueError(
            "attention dynamics must be regenerated with effective coverage; "
            f"missing columns: {sorted(missing)}"
        )
    fixed = _fixed_role_lines(attention)
    fixed = fixed[fixed["role"].isin(("nonthinking_broad", "thinking_broad"))]
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.35), sharex=True)
    panels = (
        ("total_target_mass", "Total target mass $M$", "attention mass"),
        ("effective_coverage", "Effective occurrence coverage $C_{eff}$", "coverage"),
        ("broad_score", "Broad retrieval $B_{eff}=M C_{eff}$", "broad score"),
    )
    for panel_index, (metric, title, ylabel) in enumerate(panels):
        axis = axes[panel_index]
        for role in ("nonthinking_broad", "thinking_broad"):
            line = fixed[fixed["role"] == role].sort_values("step")
            axis.plot(
                line["step"],
                line[metric],
                color=ROLE_COLOR[role],
                linewidth=2.2,
                label=ROLE_LABEL[role],
            )
            if metric == "broad_score":
                axis.plot(
                    line["step"],
                    line["legacy_entropy_broad_score"],
                    color=ROLE_COLOR[role],
                    linestyle=":",
                    linewidth=1.6,
                    alpha=0.9,
                    label=f"{ROLE_LABEL[role]} · legacy $MH/\\log N$",
                )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(-0.02, 1.02)
        _step_axis(axis)
        axis.grid(True, color="#D7D7D7", linewidth=0.55, alpha=0.7)
        _panel_label(axis, chr(ord("A") + panel_index))
    axes[0].legend(loc="upper left", fontsize=8.3)
    axes[2].legend(loc="upper left", fontsize=7.7)
    figure.suptitle(
        "Broad retrieval must separate how much is retrieved from how broadly it is covered",
        fontsize=14.5,
        y=1.03,
    )
    figure.text(
        0.5,
        0.985,
        "Dotted vertical line: task-output-only objective begins at step 1,500",
        ha="center",
        va="top",
        fontsize=8.8,
        color="#4A4A4A",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(figure, output)


def plot_per_head_formation(
    output: Path,
    attention: pd.DataFrame,
    fits: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 8.6), sharex=True)
    for panel_index, (axis, role) in enumerate(zip(axes.flat, ROLE_ORDER, strict=True)):
        selected = attention[attention["role"] == role].copy()
        fixed_row = selected[selected["is_fixed_role_head"].astype(float) == 1.0].iloc[0]
        fixed_head = (int(fixed_row["layer"]), int(fixed_row["head"]))
        final = selected[selected["step"] == selected["step"].max()].sort_values(
            "score", ascending=False
        )
        next_row = final[
            (final["layer"].astype(int) != fixed_head[0])
            | (final["head"].astype(int) != fixed_head[1])
        ].iloc[0]
        second_head = (int(next_row["layer"]), int(next_row["head"]))
        for (layer, head), line in selected.groupby(["layer", "head"], sort=True):
            identity = (int(layer), int(head))
            is_fixed = identity == fixed_head
            is_second = identity == second_head
            axis.plot(
                line["step"],
                line["score"],
                color=LAYER_COLOR[int(layer)],
                linewidth=2.8 if is_fixed else (1.8 if is_second else 0.75),
                linestyle="--" if is_second and not is_fixed else "-",
                alpha=1.0 if is_fixed else (0.82 if is_second else 0.28),
                zorder=4 if is_fixed else (3 if is_second else 1),
            )
        _shade_formation_window(axis, fits, role)
        axis.set_title(ROLE_LABEL[role])
        axis.set_ylabel("role score")
        y_max = max(0.08, float(selected["score"].max()) * 1.08)
        axis.set_ylim(-0.02 * y_max, y_max)
        axis.text(
            0.02,
            0.96,
            f"frozen head = L{fixed_head[0]}H{fixed_head[1]}\n"
            f"next final head = L{second_head[0]}H{second_head[1]}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#333333",
        )
        _step_axis(axis)
        axis.grid(True, color="#D7D7D7", linewidth=0.5, alpha=0.65)
        _panel_label(axis, chr(ord("A") + panel_index))

    layer_handles = [
        Line2D([0], [0], color=LAYER_COLOR[layer], linewidth=2, label=f"Layer {layer}")
        for layer in range(1, 5)
    ]
    emphasis_handles = [
        Line2D([0], [0], color="#444444", linewidth=2.8, label="frozen rank-1"),
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=1.8,
            linestyle="--",
            label="next final head",
        ),
    ]
    figure.legend(
        layer_handles + emphasis_handles,
        [handle.get_label() for handle in layer_handles + emphasis_handles],
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle(
        "Per-head role formation across 101 checkpoints",
        fontsize=14.5,
        y=1.01,
    )
    figure.text(
        0.5,
        0.982,
        "Layer color follows the induction-head convention; dotted line = objective switch; shading = fitted 10–90% window",
        ha="center",
        va="top",
        fontsize=9,
        color="#4A4A4A",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.965), h_pad=1.6, w_pad=1.2)
    _save_figure(figure, output)


def _distribution(group: pd.DataFrame) -> np.ndarray:
    ordered = group.sort_values(["layer", "head"])
    values = np.clip(ordered["score"].to_numpy(dtype=float), 0.0, None)
    total = float(values.sum())
    if total <= 0:
        return np.full(len(values), np.nan)
    return values / total


def _normalized_js(first: np.ndarray, second: np.ndarray) -> float:
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return math.nan
    midpoint = 0.5 * (first + second)

    def divergence(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log(values[mask] / midpoint[mask])))

    return 0.5 * (divergence(first) + divergence(second)) / math.log(2.0)


def build_bank_metrics(
    attention: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    distribution_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for (step, role), group in attention.groupby(["step", "role"], sort=True):
        probabilities = _distribution(group)
        entropy = -float(
            np.sum(
                probabilities[probabilities > 0]
                * np.log(probabilities[probabilities > 0])
            )
        )
        summary_rows.append(
            {
                "step": int(step),
                "role": str(role),
                "effective_heads": float(np.exp(entropy)),
                "top2_share": float(np.sort(probabilities)[-2:].sum()),
            }
        )
        ordered = group.sort_values(["layer", "head"])
        for row, probability in zip(
            ordered.itertuples(index=False), probabilities, strict=True
        ):
            distribution_rows.append(
                {
                    "step": int(step),
                    "role": str(role),
                    "layer": int(row.layer),
                    "head": int(row.head),
                    "role_share": float(probability),
                }
            )
    distributions = pd.DataFrame(distribution_rows)
    summaries = pd.DataFrame(summary_rows)

    pairs = (
        ("thinking_broad", "targeted_retrieval"),
        ("targeted_retrieval", "marker_successor"),
        ("thinking_broad", "marker_successor"),
        ("nonthinking_broad", "targeted_retrieval"),
    )
    js_rows: list[dict[str, float | int | str]] = []
    for step in sorted(distributions["step"].unique()):
        at_step = distributions[distributions["step"] == step]
        by_role = {
            role: group.sort_values(["layer", "head"])["role_share"].to_numpy(
                dtype=float
            )
            for role, group in at_step.groupby("role")
        }
        for first, second in pairs:
            js_rows.append(
                {
                    "step": int(step),
                    "pair": f"{first}__{second}",
                    "normalized_js_divergence": _normalized_js(
                        by_role[first], by_role[second]
                    ),
                }
            )
    js = pd.DataFrame(js_rows)
    return summaries, distributions, js


def plot_bank_differentiation(
    output: Path,
    bank_metrics: pd.DataFrame,
    distributions: pd.DataFrame,
    js: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 8.7))

    axis = axes[0, 0]
    for role in ROLE_ORDER:
        line = bank_metrics[bank_metrics["role"] == role].sort_values("step")
        axis.plot(
            line["step"],
            line["effective_heads"],
            color=ROLE_COLOR[role],
            linewidth=2.0,
            label=ROLE_LABEL[role],
        )
    axis.set_title("Role concentration across the 16 heads")
    axis.set_ylabel("effective number of heads")
    axis.set_ylim(1.0, 16.4)
    axis.legend(ncol=2, loc="lower left", fontsize=8)
    _step_axis(axis)
    _panel_label(axis, "A")

    axis = axes[0, 1]
    for role in ROLE_ORDER:
        line = bank_metrics[bank_metrics["role"] == role].sort_values("step")
        axis.plot(
            line["step"],
            line["top2_share"],
            color=ROLE_COLOR[role],
            linewidth=2.0,
            label=ROLE_LABEL[role],
        )
    axis.set_title("How much of each role is carried by its top two heads")
    axis.set_ylabel("top-2 share of role score")
    axis.set_ylim(0.08, 1.02)
    _step_axis(axis)
    _panel_label(axis, "B")

    axis = axes[1, 0]
    pair_style = {
        "thinking_broad__targeted_retrieval": ("Broad ↔ targeted", "#7A5195", "-"),
        "targeted_retrieval__marker_successor": ("Targeted ↔ successor", "#EF5675", "-"),
        "thinking_broad__marker_successor": ("Broad ↔ successor", "#FFA600", "--"),
        "nonthinking_broad__targeted_retrieval": (
            "Nonthinking broad ↔ targeted",
            "#4C78A8",
            ":",
        ),
    }
    for pair, (label, color, linestyle) in pair_style.items():
        line = js[js["pair"] == pair].sort_values("step")
        axis.plot(
            line["step"],
            line["normalized_js_divergence"],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    axis.set_title("Role-map differentiation")
    axis.set_ylabel("normalized Jensen–Shannon divergence")
    axis.set_ylim(-0.02, 1.02)
    axis.legend(loc="lower right", fontsize=7.8)
    _step_axis(axis)
    _panel_label(axis, "C")

    axis = axes[1, 1]
    final_step = int(distributions["step"].max())
    final = distributions[distributions["step"] == final_step]
    matrix = np.vstack(
        [
            final[final["role"] == role]
            .sort_values(["layer", "head"])["role_share"]
            .to_numpy(dtype=float)
            for role in ROLE_ORDER
        ]
    )
    image = axis.imshow(matrix, cmap="magma", aspect="auto", vmin=0.0, vmax=0.65)
    axis.set_title("Final normalized role maps")
    axis.set_yticks(range(len(ROLE_ORDER)))
    axis.set_yticklabels([ROLE_LABEL[role] for role in ROLE_ORDER])
    head_labels = [f"L{layer}H{head}" for layer in range(1, 5) for head in range(4)]
    axis.set_xticks(range(16))
    axis.set_xticklabels(head_labels, rotation=55, ha="right", fontsize=7.5)
    for row_index, role in enumerate(ROLE_ORDER):
        top_two = np.argsort(matrix[row_index])[-2:]
        for column_index in top_two:
            axis.add_patch(
                Rectangle(
                    (column_index - 0.5, row_index - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="white",
                    linewidth=1.4,
                )
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("share of total role score")
    _panel_label(axis, "D")

    for axis in axes.flat[:3]:
        axis.grid(True, color="#D7D7D7", linewidth=0.5, alpha=0.65)
    figure.suptitle(
        "Head-bank specialization and differentiation during v20 training",
        fontsize=14.5,
        y=1.01,
    )
    figure.text(
        0.5,
        0.982,
        "Dotted vertical line: task-output-only objective begins at step 1,500",
        ha="center",
        va="top",
        fontsize=8.8,
        color="#4A4A4A",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0, w_pad=1.6)
    _save_figure(figure, output)


def _log_step_axis(axis: plt.Axes) -> None:
    """Use a transparent offset so that the step-0 checkpoint remains visible."""

    axis.set_xscale("log")
    axis.axvline(
        OBJECTIVE_SWITCH + 100,
        color="#303030",
        linestyle=":",
        linewidth=1.1,
        zorder=0,
    )
    ticks = np.asarray([100, 200, 600, 1600, 3100, 5100, 10_100], dtype=float)
    axis.set_xlim(90, 11_000)
    axis.set_xticks(ticks)
    axis.set_xticklabels(["0", "100", "500", "1.5k", "3k", "5k", "10k"])
    axis.minorticks_off()
    axis.set_xlabel("optimizer step + 100 (log scale)")


def plot_axis_comparison(
    output: Path,
    attention: pd.DataFrame,
    fits: pd.DataFrame,
    js: pd.DataFrame,
) -> None:
    """Put identical dynamics on linear and logarithmic training axes."""

    fixed = _fixed_role_lines(attention)
    roles = ("thinking_broad", "targeted_retrieval")
    pair = js[js["pair"] == "thinking_broad__targeted_retrieval"].sort_values(
        "step"
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 8.3))
    window = _fit_window(fits, "targeted_retrieval")

    for column, scale in enumerate(("linear", "log")):
        role_axis = axes[0, column]
        js_axis = axes[1, column]
        for role in roles:
            line = fixed[fixed["role"] == role].sort_values("step")
            x = (
                line["step"].to_numpy(dtype=float)
                if scale == "linear"
                else line["step"].to_numpy(dtype=float) + 100.0
            )
            role_axis.plot(
                x,
                line["score"],
                color=ROLE_COLOR[role],
                linewidth=2.3,
                label=ROLE_LABEL[role],
            )
        js_x = (
            pair["step"].to_numpy(dtype=float)
            if scale == "linear"
            else pair["step"].to_numpy(dtype=float) + 100.0
        )
        js_axis.plot(
            js_x,
            pair["normalized_js_divergence"],
            color="#7A5195",
            linewidth=2.3,
            label="Thinking broad ↔ targeted",
        )
        if window is not None:
            offset = 0.0 if scale == "linear" else 100.0
            for axis in (role_axis, js_axis):
                axis.axvspan(
                    window[0] + offset,
                    window[1] + offset,
                    color=ROLE_COLOR["targeted_retrieval"],
                    alpha=0.075,
                    linewidth=0,
                    zorder=0,
                )
        if scale == "linear":
            _step_axis(role_axis)
            _step_axis(js_axis)
            role_axis.set_title("Linear training axis")
            js_axis.set_title("Linear training axis")
        else:
            _log_step_axis(role_axis)
            _log_step_axis(js_axis)
            role_axis.set_title("Log training axis")
            js_axis.set_title("Log training axis")
        role_axis.set_ylim(-0.02, 0.67)
        role_axis.set_ylabel("frozen-head role score")
        js_axis.set_ylim(-0.02, 1.02)
        js_axis.set_ylabel("normalized JS divergence")
        role_axis.grid(True, color="#D7D7D7", linewidth=0.55, alpha=0.7)
        js_axis.grid(True, color="#D7D7D7", linewidth=0.55, alpha=0.7)

    axes[0, 0].legend(loc="upper left", fontsize=8.4)
    axes[1, 0].legend(loc="lower right", fontsize=8.4)
    for label, axis in zip(("A", "B", "C", "D"), axes.flat, strict=True):
        _panel_label(axis, label)
    figure.suptitle(
        "Broad versus targeted retrieval: linear and log axes show the same data",
        fontsize=14.5,
        y=1.015,
    )
    figure.text(
        0.5,
        0.985,
        "Log spacing reallocates visual width; it does not change the fitted 10–90% transition width",
        ha="center",
        va="top",
        fontsize=9,
        color="#4A4A4A",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.8, w_pad=1.2)
    _save_figure(figure, output)


def plot_focused_role_specialization(
    output: Path,
    attention: pd.DataFrame,
) -> None:
    """Show only broad and targeted role trajectories for all 16 heads."""

    roles = ("nonthinking_broad", "thinking_broad", "targeted_retrieval")
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), sharex=True)
    for panel_index, (axis, role) in enumerate(zip(axes, roles, strict=True)):
        selected = attention[attention["role"] == role].copy()
        frozen = selected[selected["is_fixed_role_head"].astype(float) == 1.0].iloc[0]
        frozen_head = (int(frozen["layer"]), int(frozen["head"]))
        final = selected[selected["step"] == selected["step"].max()].sort_values(
            "score", ascending=False
        )
        next_row = final[
            (final["layer"].astype(int) != frozen_head[0])
            | (final["head"].astype(int) != frozen_head[1])
        ].iloc[0]
        second_head = (int(next_row["layer"]), int(next_row["head"]))
        for (layer, head), line in selected.groupby(["layer", "head"], sort=True):
            identity = (int(layer), int(head))
            is_frozen = identity == frozen_head
            is_second = identity == second_head
            axis.plot(
                line["step"],
                line["score"],
                color=LAYER_COLOR[int(layer)],
                linewidth=2.8 if is_frozen else (1.8 if is_second else 0.7),
                linestyle="--" if is_second and not is_frozen else "-",
                alpha=1.0 if is_frozen else (0.82 if is_second else 0.25),
                zorder=4 if is_frozen else (3 if is_second else 1),
            )
        axis.set_title(ROLE_LABEL[role])
        axis.set_ylabel("role score (panel-specific scale)")
        y_max = max(0.08, float(selected["score"].max()) * 1.08)
        axis.set_ylim(-0.02 * y_max, y_max)
        axis.text(
            0.025,
            0.95,
            f"frozen L{frozen_head[0]}H{frozen_head[1]}\n"
            f"next L{second_head[0]}H{second_head[1]}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color="#333333",
        )
        _step_axis(axis)
        axis.grid(True, color="#D7D7D7", linewidth=0.5, alpha=0.65)
        _panel_label(axis, chr(ord("A") + panel_index))

    layer_handles = [
        Line2D([0], [0], color=LAYER_COLOR[layer], linewidth=2, label=f"Layer {layer}")
        for layer in range(1, 5)
    ]
    emphasis_handles = [
        Line2D([0], [0], color="#444444", linewidth=2.8, label="frozen head"),
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=1.8,
            linestyle="--",
            label="next final head",
        ),
    ]
    handles = layer_handles + emphasis_handles
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    figure.suptitle(
        "Role specialization: broad retrieval and targeted retrieval",
        fontsize=14.5,
        y=1.035,
    )
    figure.text(
        0.5,
        0.995,
        "Nonthinking and Thinking are separate models; compare specialization profiles, not head identities across models",
        ha="center",
        va="top",
        fontsize=9,
        color="#4A4A4A",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.95), w_pad=1.2)
    _save_figure(figure, output)


def plot_thinking_bank_dynamics(
    output: Path,
    bank_metrics: pd.DataFrame,
    distributions: pd.DataFrame,
    js: pd.DataFrame,
) -> None:
    """Visualize legitimate within-model broad/targeted bank differentiation."""

    roles = ("thinking_broad", "targeted_retrieval")
    steps = np.asarray(sorted(distributions["step"].unique()), dtype=float)
    head_labels = [f"L{layer}H{head}" for layer in range(1, 5) for head in range(4)]
    matrices: dict[str, np.ndarray] = {}
    for role in roles:
        selected = distributions[distributions["role"] == role]
        matrices[role] = np.column_stack(
            [
                selected[selected["step"] == step]
                .sort_values(["layer", "head"])["role_share"]
                .to_numpy(dtype=float)
                for step in steps
            ]
        )

    figure, axes = plt.subplots(2, 2, figsize=(14.2, 8.8))
    heatmap_axes = axes[0]
    vmax = max(float(np.nanmax(matrix)) for matrix in matrices.values())
    meshes = []
    for axis, role in zip(heatmap_axes, roles, strict=True):
        mesh = axis.imshow(
            matrices[role],
            cmap="magma",
            aspect="auto",
            origin="upper",
            extent=(-50, 10_050, 15.5, -0.5),
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        meshes.append(mesh)
        axis.axvline(OBJECTIVE_SWITCH, color="white", linestyle=":", linewidth=1.1)
        for boundary in (3.5, 7.5, 11.5):
            axis.axhline(boundary, color="white", linewidth=0.6, alpha=0.55)
        axis.set_title(ROLE_LABEL[role])
        axis.set_yticks(range(16))
        axis.set_yticklabels(head_labels, fontsize=7.7)
        axis.set_xticks([0, 1500, 3000, 5000, 7000, 10_000])
        axis.set_xticklabels(["0", "1.5k", "3k", "5k", "7k", "10k"])
        axis.set_xlabel("optimizer step")
        axis.set_ylabel("attention head")
    axis = axes[1, 0]
    for role in roles:
        line = bank_metrics[bank_metrics["role"] == role].sort_values("step")
        axis.plot(
            line["step"],
            line["effective_heads"],
            color=ROLE_COLOR[role],
            linewidth=2.2,
            label=ROLE_LABEL[role],
        )
    axis.set_title("Within-role head concentration")
    axis.set_ylabel("effective number of heads")
    axis.set_ylim(1.0, 16.4)
    axis.legend(loc="lower left", fontsize=8.3)
    _step_axis(axis)

    axis = axes[1, 1]
    pair = js[js["pair"] == "thinking_broad__targeted_retrieval"].sort_values(
        "step"
    )
    axis.plot(
        pair["step"],
        pair["normalized_js_divergence"],
        color="#7A5195",
        linewidth=2.3,
        label="Broad ↔ targeted JS divergence",
    )
    for role, linestyle in zip(roles, ("--", ":"), strict=True):
        line = bank_metrics[bank_metrics["role"] == role].sort_values("step")
        axis.plot(
            line["step"],
            line["top2_share"],
            color=ROLE_COLOR[role],
            linestyle=linestyle,
            linewidth=1.8,
            label=f"{ROLE_LABEL[role]} top-2 share",
        )
    axis.set_title("Bank differentiation and top-head concentration")
    axis.set_ylabel("normalized score")
    axis.set_ylim(-0.02, 1.02)
    axis.legend(loc="lower right", fontsize=7.8)
    _step_axis(axis)

    for label, axis in zip(("A", "B", "C", "D"), axes.flat, strict=True):
        _panel_label(axis, label)
    figure.suptitle(
        "Broad and targeted head banks differentiate within the Thinking model",
        fontsize=14.5,
        y=1.015,
    )
    figure.text(
        0.5,
        0.985,
        "Heatmaps normalize role score across the same 16 heads at every checkpoint",
        ha="center",
        va="top",
        fontsize=9,
        color="#4A4A4A",
    )
    figure.tight_layout(rect=(0, 0, 0.92, 0.96), h_pad=2.0, w_pad=1.4)
    colorbar_axis = figure.add_axes((0.94, 0.58, 0.012, 0.27))
    colorbar = figure.colorbar(meshes[-1], cax=colorbar_axis, orientation="vertical")
    colorbar.set_label("share of total role score")
    _save_figure(figure, output)


def _formation_table(fits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for role in ("marker_successor", "targeted_retrieval"):
        window = _fit_window(fits, role)
        if window is None:
            continue
        selected = fits[
            (fits["evidence_family"] == "attention_role")
            & (fits["metric"] == "fixed_head_role_score")
            & (fits["group"] == role)
        ].iloc[0]
        rows.append(
            {
                "role": role,
                "smooth_center_step": float(selected["smooth_center_x"]),
                "smooth_width_10_90_steps": float(selected["smooth_width_10_90"]),
                "window_start_step": window[0],
                "window_end_step": window[1],
                "classification": str(selected["classification"]),
            }
        )
    return pd.DataFrame(rows)


def build_figures(run_dir: Path, output_dir: Path) -> dict[str, Path]:
    attention_path = run_dir / "analysis/extended/tables/attention_role_dynamics.csv"
    attention = _read(attention_path)
    fits = _read(
        run_dir
        / "analysis/phase_transition_audit/tables/aggregate_transition_model_comparison.csv"
    )
    required_attention_columns = {
        "step",
        "role",
        "layer",
        "head",
        "score",
        "is_fixed_role_head",
        "total_target_mass",
        "effective_coverage",
        "broad_score",
        "legacy_entropy_broad_score",
    }
    missing = required_attention_columns.difference(attention.columns)
    if missing:
        raise ValueError(
            "regenerate dense attention roles before plotting; "
            f"missing columns: {sorted(missing)}"
        )
    if set(attention["role"].unique()) != set(ROLE_ORDER):
        raise ValueError("unexpected or missing attention roles")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
        }
    )

    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    bank_metrics, distributions, js = build_bank_metrics(attention)
    outputs = {
        "overview": figure_dir / "v20_training_dynamics_overview.png",
        "broad_decomposition": figure_dir / "v20_broad_metric_decomposition.png",
        "per_head": figure_dir / "v20_per_head_role_formation.png",
        "bank_differentiation": figure_dir / "v20_head_bank_differentiation.png",
        "broad_targeted_axis_comparison": (
            figure_dir / "v20_broad_targeted_linear_vs_log.png"
        ),
        "focused_role_specialization": (
            figure_dir / "v20_broad_targeted_role_specialization.png"
        ),
        "thinking_bank_dynamics": (
            figure_dir / "v20_thinking_broad_targeted_bank_dynamics.png"
        ),
    }
    plot_overview(run_dir, outputs["overview"], attention, fits)
    plot_broad_decomposition(outputs["broad_decomposition"], attention)
    plot_per_head_formation(outputs["per_head"], attention, fits)
    plot_bank_differentiation(
        outputs["bank_differentiation"],
        bank_metrics,
        distributions,
        js,
    )
    plot_axis_comparison(
        outputs["broad_targeted_axis_comparison"],
        attention,
        fits,
        js,
    )
    plot_focused_role_specialization(
        outputs["focused_role_specialization"],
        attention,
    )
    plot_thinking_bank_dynamics(
        outputs["thinking_bank_dynamics"],
        bank_metrics,
        distributions,
        js,
    )

    table_dir.mkdir(parents=True, exist_ok=True)
    bank_metrics.to_csv(table_dir / "head_bank_differentiation.csv", index=False)
    distributions.to_csv(table_dir / "head_role_distributions.csv", index=False)
    js.to_csv(table_dir / "head_role_js_divergence.csv", index=False)
    _formation_table(fits).to_csv(table_dir / "formation_windows.csv", index=False)
    manifest = {
        "run": run_dir.name,
        "source_tables": [
            str(attention_path.relative_to(run_dir)),
            "analysis/phase_transition/tables/dense_behavior_by_count.csv",
            "analysis/phase_transition_audit/tables/high_power_ar_summary.csv",
            "analysis/phase_transition_audit/tables/routing_qk_by_k.csv",
            "analysis/phase_transition_audit/tables/local_head_causal_damage.csv",
            "analysis/phase_transition_audit/tables/aggregate_transition_model_comparison.csv",
            "tables/final_autoregressive_summary.csv",
        ],
        "broad_score_definition": "mean(M * exp(H(p)) / N)",
        "formation_window_interpretation": (
            "seed-1234 fitted sigmoid 10-90% intervals; descriptive, not a universal phase claim"
        ),
        "figures": {name: str(path.relative_to(output_dir)) for name, path in outputs.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to <run>/analysis/training_dynamics_anthropic_style",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "analysis/training_dynamics_anthropic_style"
    )
    outputs = build_figures(run_dir, output_dir)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
