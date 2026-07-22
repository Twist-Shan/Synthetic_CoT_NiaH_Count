#!/usr/bin/env python3
"""Build Chinese mechanism/training-dynamics reports for the paired v20/v21 runs."""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = (
    ROOT / "colab_results" / "v20_main_RoPE_count1-30_seed1234",
    ROOT / "colab_results" / "v21_main_RoPE_count1-30_digit_seed1234",
)
REPORT_NAME = {
    "v20": "v20_counting_mechanism_report.html",
    "v21": "v21_counting_mechanism_report.html",
}
MODE_LABEL = {"thinking": "Thinking", "nonthinking": "Nonthinking"}
ROLE_LABEL = {
    "targeted_retrieval": "targeted retrieval",
    "marker_successor": "marker successor",
}
OUTCOME_LABEL = {
    "final_answer_teacher_forced_exact": "TF final exact",
    "trace_marker_teacher_forced": "TF marker",
    "marker_next_token_teacher_forced": "TF marker→next token",
    "semantic_k_to_k_plus_1_teacher_forced_exact": "TF semantic k→k+1 exact",
}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    try:
        temporary.replace(path)
    except PermissionError:
        shutil.copyfile(temporary, path)
        temporary.unlink()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def image_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/svg+xml"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def figure(path: Path, number: str, title: str, caption: str, alt: str | None = None) -> str:
    if not path.exists():
        return f'<div class="warning">缺少图像：{html.escape(str(path))}</div>'
    return f"""
    <figure class="report-figure">
      <h4>{html.escape(title)}</h4>
      <img src="{image_uri(path)}" alt="{html.escape(alt or title)}" loading="lazy">
      <figcaption><span class="figure-tag">图 {html.escape(number)}.</span> {caption}</figcaption>
    </figure>
    """


def embedded_iframe(path: Path, title: str, *, css_class: str = "") -> str:
    if not path.exists():
        return f'<div class="callout warning">缺少交互图：{html.escape(str(path))}</div>'
    source = html.escape(path.read_text(encoding="utf-8"), quote=True)
    return (
        f'<iframe class="{html.escape(css_class)}" srcdoc="{source}" '
        f'title="{html.escape(title)}"></iframe>'
    )


def validate_report(path: Path, run_dir: Path) -> None:
    """Fail fast on broken embedded assets or malformed report structure."""

    document = path.read_text(encoding="utf-8")
    required = (
        '<html lang="zh-CN">',
        "实验设定与表示",
        "结果完整性审计",
        "counting mechanism",
        "这些机制如何在训练中形成",
        "Hidden-state representation",
        "因果证据的强弱",
    )
    missing = [value for value in required if value not in document]
    if missing:
        raise ValueError(f"report is missing required sections: {missing}")
    if "�" in document:
        raise ValueError("report contains a Unicode replacement character")

    figure_count = document.count('<figure class="report-figure">')
    version = str(json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["version"])
    expected_figures = 15 if version == "v20" else 10
    if figure_count != expected_figures:
        raise ValueError(
            f"expected {expected_figures} report figures, found {figure_count}"
        )
    payloads = re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)', document)
    if len(payloads) != figure_count:
        raise ValueError(f"expected {figure_count} embedded PNGs, found {len(payloads)}")
    for index, payload in enumerate(payloads, start=1):
        if not base64.b64decode(payload, validate=True).startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"embedded figure {index} is not a valid PNG")

    ids = re.findall(r'\bid="([^"]+)"', document)
    if len(ids) != len(set(ids)):
        raise ValueError("report contains duplicate element ids")
    interactive = run_dir / "analysis" / "phase_transition" / "interactive_manifold_3d.html"
    if not interactive.exists():
        raise FileNotFoundError(interactive)
    if version == "v20":
        for required in (
            "tables/final_autoregressive_summary.csv",
            "analysis/extended/interactive_attention_dynamics.html",
            "analysis/v10_port/tables/retrieval_localization_transport_patching.csv",
            "analysis/phase_transition_audit/tables/routing_qk_by_k.csv",
            "analysis/phase_transition_audit/tables/high_power_ar_summary.csv",
            "analysis/phase_transition_audit/tables/aggregate_transition_model_comparison.csv",
        ):
            if not (run_dir / required).exists():
                raise FileNotFoundError(run_dir / required)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def weighted_mean(group: pd.DataFrame, value: str, weight: str = "observations") -> float:
    valid = group[[value, weight]].dropna()
    if valid.empty or float(valid[weight].sum()) <= 0:
        return math.nan
    return float(np.average(valid[value], weights=valid[weight]))


def weighted_behavior(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (step, mode, outcome), group in frame.groupby(["step", "mode", "outcome"]):
        rows.append(
            {
                "step": int(step),
                "mode": str(mode),
                "outcome": str(outcome),
                "accuracy": weighted_mean(group, "accuracy"),
                "observations": int(group["observations"].sum()),
            }
        )
    return pd.DataFrame(rows)


def first_crossing(frame: pd.DataFrame, column: str, threshold: float) -> int | None:
    selected = frame[frame[column] >= threshold].sort_values("step")
    return None if selected.empty else int(selected.iloc[0]["step"])


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 145,
            "savefig.dpi": 175,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "figure.constrained_layout.use": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_loss_accuracy_overview(run_dir: Path, output: Path) -> None:
    """Put directly comparable final-count loss, TF accuracy, and AR accuracy together."""

    components = read_csv(run_dir / "tables/eval_loss_components.csv")
    loss = components[
        (components["curve_source"] == "heldout")
        & (components["source_region"] == "validation")
        & (components["suite"] == "task")
        & (components["component"] == "final_count")
    ].copy()
    loss = loss.groupby(["step", "mode"], as_index=False).agg(
        cross_entropy=("example_mean_cross_entropy", "mean")
    )

    tf = read_csv(run_dir / "tables/eval_by_count.csv")
    tf = tf.groupby(["step", "mode"], as_index=False).agg(
        accuracy=("tf_final_accuracy", "mean")
    )
    ar = read_csv(run_dir / "tables/autoregressive_by_count.csv")
    ar = ar.groupby(["step", "mode"], as_index=False).agg(
        accuracy=("ar_final_accuracy", "mean")
    )
    final_ar = read_csv(run_dir / "tables/final_autoregressive_summary.csv")

    colors = {"thinking": "#d97745", "nonthinking": "#315f9f"}
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.55))
    for mode in ("nonthinking", "thinking"):
        label = MODE_LABEL[mode]
        line = loss[loss["mode"] == mode].sort_values("step")
        axes[0].plot(
            line["step"], line["cross_entropy"], marker="o", markersize=3.3,
            linewidth=2, color=colors[mode], label=label,
        )
        line = tf[tf["mode"] == mode].sort_values("step")
        axes[1].plot(
            line["step"], line["accuracy"], marker="o", markersize=3.3,
            linewidth=2, color=colors[mode], label=label,
        )
        line = ar[ar["mode"] == mode].sort_values("step")
        axes[2].plot(
            line["step"], line["accuracy"], marker="o", markersize=4,
            linewidth=2, color=colors[mode], label=f"{label} · periodic 2/count",
        )
        if not final_ar.empty:
            final = final_ar[final_ar["mode"] == mode]
            if not final.empty:
                row = final.iloc[0]
                y = float(row["ar_final_accuracy"])
                low = float(row["ar_final_accuracy_wilson95_low"])
                high = float(row["ar_final_accuracy_wilson95_high"])
                axes[2].errorbar(
                    [int(row["step"])], [y], yerr=[[y - low], [high - y]],
                    fmt="*", markersize=13, capsize=4, linewidth=1.6,
                    color=colors[mode], markeredgecolor="white", markeredgewidth=.7,
                    label=f"{label} · final 50/count",
                    zorder=5,
                )

    axes[0].set(
        title="Held-out final-count loss",
        xlabel="training step",
        ylabel="cross-entropy (log scale)",
    )
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper right")
    axes[1].set(
        title="Teacher-forced final accuracy",
        xlabel="training step",
        ylabel="exact accuracy",
        ylim=(-0.03, 1.04),
    )
    axes[1].legend(loc="lower right")
    axes[2].set(
        title="Autoregressive final accuracy",
        xlabel="training step",
        ylabel="exact accuracy",
        ylim=(-0.03, 1.04),
    )
    axes[2].legend(loc="upper left", fontsize=7.7)
    for ax in axes:
        ax.axvline(1500, color="#222", linestyle=":", linewidth=1.2)
        ax.set_xlim(-200, 10_200)
    fig.suptitle(
        f"{run_dir.name.split('_', 1)[0]}: Thinking vs Nonthinking loss and accuracy",
        fontsize=15,
    )
    savefig(fig, output)


def plot_signed_count_error_dynamics(run_dir: Path, output: Path) -> None:
    """Show how the full distribution of AR count errors changes during training."""

    frame = read_csv(run_dir / "tables/autoregressive_detail.csv").copy()
    if frame.empty:
        raise FileNotFoundError(run_dir / "tables/autoregressive_detail.csv")
    frame["signed_error"] = frame["ar_pred_count"] - frame["count"]
    steps = np.asarray(sorted(frame["step"].unique()), dtype=float)
    bands = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30))
    modes = ("nonthinking", "thinking")
    error_values = np.arange(-30, 31)
    x_edges = np.concatenate(([steps[0] - 500], (steps[:-1] + steps[1:]) / 2, [steps[-1] + 500]))
    y_edges = np.arange(-30.5, 31.5, 1.0)

    final = read_csv(run_dir / "tables/final_autoregressive_detail.csv").copy()
    if not final.empty:
        final["signed_error"] = final["ar_pred_count"] - final["count"]

    fig, axes = plt.subplots(6, 2, figsize=(15.4, 19.6), sharex=True, sharey=True)
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
    mesh = None
    for row_index, (low, high) in enumerate(bands):
        for col_index, mode in enumerate(modes):
            ax = axes[row_index, col_index]
            selected = frame[
                (frame["mode"] == mode)
                & frame["count"].between(low, high)
            ]
            matrix = np.zeros((len(error_values), len(steps)), dtype=float)
            means = np.full(len(steps), np.nan, dtype=float)
            for step_index, step in enumerate(steps):
                column = selected[selected["step"] == step]["signed_error"].dropna()
                scheduled = int(len(selected[selected["step"] == step]))
                if scheduled:
                    counts = column.value_counts()
                    for error, observations in counts.items():
                        error = int(error)
                        if -30 <= error <= 30:
                            matrix[error + 30, step_index] = float(observations) / scheduled
                if not column.empty:
                    means[step_index] = float(column.mean())

            ax.set_facecolor("#05060a")
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                matrix,
                cmap="magma",
                norm=norm,
                shading="flat",
                rasterized=True,
            )
            ax.plot(
                steps,
                means,
                color="#74d5ff",
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label="periodic mean signed error",
                zorder=4,
            )
            ax.axhline(
                0,
                color="#ffd166",
                linestyle="--",
                linewidth=1.6,
                label="zero error (exact count)",
                zorder=3,
            )
            ax.axvline(
                1500,
                color="#d6d9df",
                linestyle=":",
                linewidth=1.4,
                label="objective switch",
                zorder=3,
            )
            if not final.empty:
                high_power = final[
                    (final["mode"] == mode)
                    & final["count"].between(low, high)
                ]["signed_error"].dropna()
                if not high_power.empty:
                    ax.scatter(
                        [10_000],
                        [float(high_power.mean())],
                        marker="*",
                        s=88,
                        color="#e6fbff",
                        edgecolor="#167da4",
                        linewidth=0.8,
                        label="final 50/count mean",
                        zorder=6,
                    )
            if row_index == 0:
                ax.set_title(MODE_LABEL[mode], fontsize=13, weight="bold")
            if col_index == 0:
                ax.set_ylabel(f"True {low}–{high}\nprediction − truth")
            ax.set_ylim(-30.5, 30.5)
            ax.set_yticks(np.arange(-30, 31, 10))
            ax.grid(False)
            ax.tick_params(axis="both", colors="#1e293b")

    for ax in axes[-1, :]:
        ax.set_xlabel("training step")
        ax.set_xticks(steps)
        ax.set_xticklabels([f"{int(step / 1000)}k" for step in steps])
    if mesh is None:
        raise ValueError("could not construct signed-error heatmap")
    colorbar = fig.colorbar(mesh, ax=axes, location="right", fraction=0.024, pad=0.025)
    colorbar.set_label("share of 10 scheduled predictions (square-root color scale)")
    colorbar.set_ticks([0, 0.05, 0.10, 0.20, 0.50, 1.00])
    colorbar.set_ticklabels(["0%", "5%", "10%", "20%", "50%", "100%"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Prediction-error distributions across training", fontsize=16, y=1.055)
    savefig(fig, output)


def plot_mechanism_timeline(run_dir: Path, output: Path) -> None:
    behavior = weighted_behavior(read_csv(run_dir / "analysis/phase_transition/tables/dense_behavior_by_count.csv"))
    ar = read_csv(run_dir / "tables/autoregressive_by_count.csv")
    ar_summary = ar.groupby(["step", "mode"], as_index=False).agg(
        ar_final_accuracy=("ar_final_accuracy", "mean"),
        trace_exact=("trace_exact", "mean"),
        trace_marker=("trace_ordered_marker_accuracy", "mean"),
    )
    heads = read_csv(run_dir / "analysis/phase_transition/tables/dense_fixed_head_dynamics.csv")
    heads = heads[heads["is_fixed_role_head"] == 1]
    causal = read_csv(run_dir / "analysis/phase_transition/tables/milestone_local_head_causality.csv")

    colors = {"thinking": "#d97745", "nonthinking": "#315f9f"}
    role_colors = {"targeted_retrieval": "#6f4aa8", "marker_successor": "#16877d"}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.1))

    ax = axes[0, 0]
    for mode in ("nonthinking", "thinking"):
        line = ar_summary[ar_summary["mode"] == mode]
        ax.plot(line["step"], line["ar_final_accuracy"], marker="o", color=colors[mode], label=f"{MODE_LABEL[mode]} final")
    thinking = ar_summary[ar_summary["mode"] == "thinking"]
    ax.plot(thinking["step"], thinking["trace_exact"], marker="s", linestyle="--", color="#c13f52", label="Thinking trace exact")
    ax.set(title="自由生成行为", xlabel="training step", ylabel="accuracy", ylim=(-0.03, 1.04))
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    selected = behavior[behavior["mode"] == "thinking"]
    styles = {
        "final_answer_teacher_forced_exact": ("#d97745", "-"),
        "trace_marker_teacher_forced": ("#6f4aa8", "-"),
        "semantic_k_to_k_plus_1_teacher_forced_exact": ("#16877d", "--"),
    }
    for outcome, (color, linestyle) in styles.items():
        line = selected[selected["outcome"] == outcome]
        ax.plot(line["step"], line["accuracy"], color=color, linestyle=linestyle, label=OUTCOME_LABEL[outcome])
    ax.axvline(1500, color="#222", linestyle=":", linewidth=1.2, label="objective switch")
    ax.set(title="Teacher-forced 局部能力", xlabel="training step", ylabel="occurrence-weighted accuracy", ylim=(-0.03, 1.04))
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    for role in ("marker_successor", "targeted_retrieval"):
        line = heads[heads["role"] == role].sort_values("step")
        ax.plot(line["step"], line["score"], color=role_colors[role], label=ROLE_LABEL[role])
    ax.axvline(1500, color="#222", linestyle=":", linewidth=1.2)
    ax.set(title="固定功能 head 的注意力角色分数", xlabel="training step", ylabel="attention mass", ylim=(-0.03, 1.04))
    ax.legend(loc="center right")

    ax = axes[1, 1]
    changed = causal[causal["intervention"] != "baseline"].copy()
    changed["damage"] = -changed["margin_change_from_baseline"]
    for role, color in role_colors.items():
        fixed = changed[(changed["role"] == role) & (changed["intervention"] == "fixed_head_zero")]
        control = changed[(changed["role"] == role) & (changed["intervention"] == "same_layer_control_zero")]
        ax.plot(fixed["step"], fixed["damage"], marker="o", color=color, label=f"{ROLE_LABEL[role]} selected")
        ax.plot(control["step"], control["damage"], marker=".", linestyle="--", alpha=0.65, color=color, label=f"{ROLE_LABEL[role]} control")
    ax.axhline(0, color="#222", linewidth=1)
    ax.set(title="位置局部消融造成的正确-token margin 损失", xlabel="training step", ylabel="−Δ logit margin (larger = more damage)")
    ax.legend(ncol=2, loc="upper left")
    fig.suptitle(f"{run_dir.name}: behavior → routing → causal dependence", fontsize=15)
    savefig(fig, output)


def count_band(count: int) -> str:
    start = ((int(count) - 1) // 5) * 5 + 1
    return f"{start}–{start + 4}"


def plot_ar_bands(run_dir: Path, output: Path) -> None:
    frame = read_csv(run_dir / "tables/autoregressive_by_count.csv").copy()
    frame["band"] = frame["count"].map(count_band)
    order = ["1–5", "6–10", "11–15", "16–20", "21–25", "26–30"]
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(order)))
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.3), sharey=True)
    panels = (
        ("thinking", "ar_final_accuracy", "Thinking final exact"),
        ("thinking", "trace_exact", "Thinking trace exact"),
        ("nonthinking", "ar_final_accuracy", "Nonthinking final exact"),
    )
    for ax, (mode, metric, title) in zip(axes, panels, strict=True):
        selected = frame[frame["mode"] == mode]
        summary = selected.groupby(["step", "band"], as_index=False)[metric].mean()
        for band, color in zip(order, colors, strict=True):
            line = summary[summary["band"] == band]
            ax.plot(line["step"], line[metric], marker="o", color=color, label=band)
        ax.set(title=title, xlabel="training step", ylim=(-0.03, 1.04))
    axes[0].set_ylabel("autoregressive exact accuracy")
    axes[-1].legend(title="true-count band", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("自由生成按 true count 分段的学习动力学", fontsize=14)
    savefig(fig, output)


def _wilson_interval(successes: float, observations: int) -> tuple[float, float]:
    if observations <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    estimate = float(successes) / observations
    denominator = 1 + z * z / observations
    center = (estimate + z * z / (2 * observations)) / denominator
    radius = z * math.sqrt(
        estimate * (1 - estimate) / observations
        + z * z / (4 * observations * observations)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def plot_final_ar_by_count(run_dir: Path, output: Path) -> None:
    frame = read_csv(run_dir / "tables/final_autoregressive_by_count.csv")
    if frame.empty:
        frame = read_csv(run_dir / "tables/autoregressive_by_count.csv")
        frame = frame[frame["step"] == frame["step"].max()].copy()
        frame["examples"] = 2
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.6), sharex=True)
    colors = {"thinking": "#d97745", "nonthinking": "#315f9f"}
    for ax, mode in zip(axes, ("thinking", "nonthinking"), strict=True):
        selected = frame[frame["mode"] == mode].sort_values("count")
        lower, upper = [], []
        for row in selected.itertuples(index=False):
            observations = int(row.examples)
            low, high = _wilson_interval(float(row.ar_final_accuracy) * observations, observations)
            lower.append(float(row.ar_final_accuracy) - low)
            upper.append(high - float(row.ar_final_accuracy))
        ax.errorbar(
            selected["count"],
            selected["ar_final_accuracy"],
            yerr=np.asarray([lower, upper]),
            color=colors[mode],
            marker="o",
            markersize=4,
            linewidth=1.6,
            capsize=2.5,
        )
        ax.axhline(float(selected["ar_final_accuracy"].mean()), color=colors[mode], linestyle="--", alpha=.55)
        ax.set(title=f"{MODE_LABEL[mode]} final AR exact · {int(selected['examples'].min())} examples/count", ylabel="exact accuracy", ylim=(-.04, 1.04))
    axes[-1].set_xlabel("true count n")
    fig.suptitle("Final checkpoint: high-power autoregressive evaluation", fontsize=15)
    savefig(fig, output)


def plot_head_by_k(run_dir: Path, output: Path) -> None:
    frame = read_csv(run_dir / "analysis/phase_transition/tables/dense_fixed_head_by_k.csv")
    final_step = int(frame["step"].max())
    final = frame[frame["step"] == final_step]
    behavior = read_csv(run_dir / "analysis/phase_transition/tables/dense_behavior_by_k.csv")
    behavior = behavior[behavior["step"] == int(behavior["step"].max())]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4), sharex=True)
    for ax, role, color in zip(axes, ("targeted_retrieval", "marker_successor"), ("#6f4aa8", "#16877d"), strict=True):
        line = final[final["role"] == role].sort_values("k")
        ax.plot(line["k"], line["score"], marker="o", color=color, label="attention role score")
        outcome = "trace_marker_teacher_forced" if role == "targeted_retrieval" else "marker_next_token_teacher_forced"
        observed = behavior[behavior["outcome"] == outcome].sort_values("k")
        ax.plot(observed["k"], observed["accuracy"], linestyle="--", color="#d97745", label="teacher-forced token accuracy")
        ax.axvline(9.5, color="#222", linestyle=":", linewidth=1)
        ax.set(title=ROLE_LABEL[role], xlabel="semantic progress k", ylabel="score / accuracy", ylim=(-0.03, 1.04))
        ax.legend(loc="lower left")
    fig.suptitle(f"最终 checkpoint（step {final_step:,}）按 k 分解", fontsize=14)
    savefig(fig, output)


def table(headers: Iterable[str], rows: Iterable[Iterable[Any]], classes: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table class="{html.escape(classes)}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def audit_run(run_dir: Path, cfg: dict[str, Any], manifest: dict[str, Any]) -> tuple[list[list[str]], list[str]]:
    issues: list[str] = []
    rows: list[list[str]] = []
    for mode in ("nonthinking", "thinking"):
        index = read_csv(run_dir / f"checkpoints/rope/{mode}/snapshot_index.csv")
        steps = set(index.get("step", pd.Series(dtype=int)).astype(int).tolist())
        planned = set(range(0, int(cfg["train_steps"]) + 1, int(cfg["checkpoint_every"])))
        missing = sorted(planned - steps)
        shards = sorted(set(index.get("shard", pd.Series(dtype=str)).astype(str).tolist()))
        missing_shards = [name for name in shards if not (run_dir / f"checkpoints/rope/{mode}" / name).exists()]
        status = "完整" if not missing and not missing_shards else f"缺 {len(missing)} 个 step / {len(missing_shards)} 个 shard"
        if missing:
            issues.append(f"{mode} scientific snapshots missing: {missing}")
        if missing_shards:
            issues.append(f"{mode} snapshot shards missing: {missing_shards}")
        final_exists = (run_dir / f"checkpoints/rope/{mode}/final/checkpoint.pt").exists()
        recovery_exists = (run_dir / f"checkpoints/rope/{mode}/recovery/latest.pt").exists()
        if not final_exists:
            issues.append(f"{mode} final checkpoint is missing")
        if not recovery_exists:
            issues.append(f"{mode} recovery checkpoint is missing")
        if not final_exists or not recovery_exists:
            status = "缺 final/recovery"
        rows.append([f"{MODE_LABEL[mode]} checkpoints", f"{len(steps)}/{len(planned)} snapshots；{len(shards) - len(missing_shards)}/{len(shards)} shards", "是" if final_exists else "否", "是" if recovery_exists else "否", status])

    stages = manifest.get("stages", {})
    complete_stages = [name for name, value in stages.items() if value.get("status") == "complete"]
    failed_stages = [name for name, value in stages.items() if value.get("status") != "complete"]
    repaired_causal = (
        str(cfg["version"]) == "v20"
        and failed_stages == ["causal"]
        and (run_dir / "analysis/v10_port/manifest.json").exists()
    )
    pipeline_status = "完整" if not failed_stages else ("causal 已本地修复" if repaired_causal else "存在失败 stage")
    pipeline_coverage = f"原始 {len(complete_stages)}/{len(stages)} complete"
    if failed_stages:
        pipeline_coverage += f"；failed={','.join(failed_stages)}"
    rows.append(["pipeline manifest", pipeline_coverage, "—", "—", pipeline_status])

    coverage = (
        ("训练日志", "tables/train_metrics.csv", 201),
        ("周期 teacher-forced", "tables/eval_by_count.csv", 21),
        ("周期 autoregressive", "tables/autoregressive_by_count.csv", 10),
        ("100-step behavior", "analysis/phase_transition/tables/dense_behavior_by_count.csv", 101),
        ("100-step heads", "analysis/phase_transition/tables/dense_fixed_head_dynamics.csv", 101),
        ("100-step geometry", "analysis/phase_transition/tables/dense_manifold_geometry.csv", 101),
    )
    for label, relative, expected_steps in coverage:
        frame = read_csv(run_dir / relative)
        observed = int(frame["step"].nunique()) if not frame.empty and "step" in frame else 0
        status = "完整" if observed == expected_steps else "不完整"
        if status != "完整":
            issues.append(f"{relative}: expected {expected_steps} steps, found {observed}")
        rows.append([label, f"{observed}/{expected_steps} steps", "—", "—", status])

    final_ar = read_csv(run_dir / "tables/final_autoregressive_detail.csv")
    if not final_ar.empty:
        coverage_text = "; ".join(
            f"{MODE_LABEL[mode]}={len(group)} ({int(group.groupby('count').size().min())}/count)"
            for mode, group in final_ar.groupby("mode")
        )
        balanced = all(
            group.groupby("count").size().nunique() == 1 and group["count"].nunique() == 30
            for _, group in final_ar.groupby("mode")
        )
        rows.append(
            [
                "final test autoregressive",
                coverage_text,
                "是",
                "—",
                "完整" if balanced else "不完整",
            ]
        )

    local = read_csv(run_dir / "analysis/phase_transition/tables/milestone_local_head_causality.csv")
    local_status = "完整" if len(local) == 66 else "不完整"
    rows.append(["位置局部因果", f"{len(local)}/66 rows", "—", "—", local_status])
    if local_status != "完整":
        issues.append(f"local causal rows: expected 66, found {len(local)}")

    version = str(cfg["version"])
    if version == "v20":
        port = run_dir / "analysis/v10_port/manifest.json"
        original = manifest.get("stages", {}).get("causal", {}).get("status", "missing")
        status = "本地补跑完整" if port.exists() else f"原始 stage={original}，尚缺完整 port"
        rows.append(["完整 v10 causal port", status, "—", "—", "完整" if port.exists() else "缺失"])
        if not port.exists():
            issues.append("v20 full v10 causal port is absent (original manifest marks the stage failed)")
        audit_ar = read_csv(
            run_dir
            / "analysis/phase_transition_audit/tables/high_power_ar_detail.csv"
        )
        expected_steps = set(range(3_000, 7_001, 500))
        complete_groups = 0
        if not audit_ar.empty:
            for (step, mode), group in audit_ar.groupby(["step", "mode"]):
                counts = group.groupby("count").size()
                if (
                    int(step) in expected_steps
                    and str(mode) in {"thinking", "nonthinking"}
                    and len(counts) == 30
                    and bool(counts.eq(50).all())
                ):
                    complete_groups += 1
        audit_status = "完整" if complete_groups == 18 else "不完整"
        rows.append(
            [
                "phase-window high-power AR",
                f"{complete_groups}/18 step×mode groups；50/count",
                "—",
                "—",
                audit_status,
            ]
        )
        if audit_status != "完整":
            issues.append(
                f"phase-window high-power AR: expected 18 complete groups, found {complete_groups}"
            )
    else:
        rows.append(["完整 v10 causal port", "atomic-token 专用；v21 按设计不运行", "—", "—", "N/A"])
    return rows, issues


def build_report(run_dir: Path) -> Path:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    version = str(cfg["version"])
    if version not in REPORT_NAME:
        raise ValueError(f"unsupported report version: {version}")

    assets = run_dir / "analysis" / "mechanism_report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    overview_path = assets / "loss_accuracy_overview.png"
    signed_error_path = assets / "autoregressive_signed_error_dynamics.png"
    timeline_path = assets / "mechanism_timeline.png"
    ar_band_path = assets / "autoregressive_count_bands.png"
    final_ar_path = assets / "final_autoregressive_by_count.png"
    head_k_path = assets / "final_head_by_k.png"
    plot_loss_accuracy_overview(run_dir, overview_path)
    plot_signed_count_error_dynamics(run_dir, signed_error_path)
    plot_mechanism_timeline(run_dir, timeline_path)
    plot_ar_bands(run_dir, ar_band_path)
    plot_final_ar_by_count(run_dir, final_ar_path)
    plot_head_by_k(run_dir, head_k_path)

    eval_by_count = read_csv(run_dir / "tables/eval_by_count.csv")
    final_eval = eval_by_count[eval_by_count["step"] == int(eval_by_count["step"].max())]
    tf = final_eval.groupby("mode", as_index=True).mean(numeric_only=True)
    ar = read_csv(run_dir / "tables/autoregressive_by_count.csv")
    final_ar = ar[ar["step"] == int(ar["step"].max())].groupby("mode", as_index=True).mean(numeric_only=True)
    high_power_ar = read_csv(run_dir / "tables/final_autoregressive_summary.csv")
    if high_power_ar.empty:
        high_power_ar = final_ar.reset_index()
        high_power_ar["examples"] = int(cfg.get("ar_examples_per_count", 2)) * int(cfg["count_max_threshold"])
        high_power_ar["examples_per_count"] = int(cfg.get("ar_examples_per_count", 2))
        high_power_ar["ar_final_accuracy_wilson95_low"] = np.nan
        high_power_ar["ar_final_accuracy_wilson95_high"] = np.nan
    high_power_ar = high_power_ar.set_index("mode")
    ar_curve = ar.groupby(["step", "mode"], as_index=False).mean(numeric_only=True)
    heads = read_csv(run_dir / "analysis/phase_transition/tables/dense_fixed_head_dynamics.csv")
    fixed_heads = heads[heads["is_fixed_role_head"] == 1]
    final_heads = fixed_heads[fixed_heads["step"] == int(fixed_heads["step"].max())].set_index("role")
    head_roles = json.loads((run_dir / "analysis/phase_transition/fixed_head_roles.json").read_text(encoding="utf-8"))
    causal = read_csv(run_dir / "analysis/phase_transition/tables/milestone_local_head_causality.csv")
    final_causal = causal[(causal["step"] == int(causal["step"].max())) & (causal["intervention"] != "baseline")]
    geometry = read_csv(run_dir / "analysis/phase_transition/tables/dense_manifold_geometry.csv")
    final_geometry = geometry[(geometry["step"] == int(geometry["step"].max())) & (geometry["mode"] == "thinking") & (geometry["site"] == "trace_marker")]
    layer4 = final_geometry[final_geometry["layer"] == int(cfg["n_layer"])].iloc[0]
    behavior = weighted_behavior(read_csv(run_dir / "analysis/phase_transition/tables/dense_behavior_by_count.csv"))

    successor_line = fixed_heads[fixed_heads["role"] == "marker_successor"].sort_values("step")
    targeted_line = fixed_heads[fixed_heads["role"] == "targeted_retrieval"].sort_values("step")
    ar_think = ar_curve[ar_curve["mode"] == "thinking"].sort_values("step")
    transition_rows = [
        ["Successor attention mass ≥ 0.5", fmt(first_crossing(successor_line, "score", 0.5), 0), "局部 index/close 路由形成"],
        ["Thinking TF final exact ≥ 0.8", fmt(first_crossing(behavior[(behavior["mode"] == "thinking") & (behavior["outcome"] == "final_answer_teacher_forced_exact")], "accuracy", 0.8), 0), "仅在正确 trace 前缀条件下"],
        ["Targeted-retrieval mass ≥ 0.1", fmt(first_crossing(targeted_line, "score", 0.1), 0), "定向 prompt 检索开始可见"],
        ["Targeted-retrieval mass ≥ 0.5", fmt(first_crossing(targeted_line, "score", 0.5), 0), "定向检索成为强路由"],
        ["Thinking AR final ≥ 0.5", fmt(first_crossing(ar_think, "ar_final_accuracy", 0.5), 0), "整条自由生成链开始可靠"],
        ["Thinking AR final ≥ 0.8", fmt(first_crossing(ar_think, "ar_final_accuracy", 0.8), 0), "若未达到则记为—"],
    ]

    audit_rows, issues = audit_run(run_dir, cfg, manifest)
    if version == "v20" and (run_dir / "analysis/v10_port/manifest.json").exists():
        completeness = "核心完整；完整 causal 已补跑"
    else:
        completeness = "核心结果完整" if not issues else "存在明确缺口"
    tokenization = "atomic count token" if cfg["count_tokenization"] == "atomic" else "digit-wise shared digits"
    number_example = "<12>" if version == "v20" else "<D1><D2>"
    targeted = final_heads.loc["targeted_retrieval"]
    successor = final_heads.loc["marker_successor"]
    targeted_fixed = final_causal[(final_causal["role"] == "targeted_retrieval") & (final_causal["intervention"] == "fixed_head_zero")].iloc[0]
    targeted_control = final_causal[(final_causal["role"] == "targeted_retrieval") & (final_causal["intervention"] == "same_layer_control_zero")].iloc[0]
    successor_fixed = final_causal[(final_causal["role"] == "marker_successor") & (final_causal["intervention"] == "fixed_head_zero")].iloc[0]
    successor_control = final_causal[(final_causal["role"] == "marker_successor") & (final_causal["intervention"] == "same_layer_control_zero")].iloc[0]

    counterpart_dir = DEFAULT_RUNS[1] if version == "v20" else DEFAULT_RUNS[0]
    paired_behavior_note = ""
    if counterpart_dir.exists():
        counterpart_ar = read_csv(counterpart_dir / "tables/autoregressive_by_count.csv")
        counterpart_final = counterpart_ar[
            counterpart_ar["step"] == int(counterpart_ar["step"].max())
        ].groupby("mode", as_index=True).mean(numeric_only=True)
        counterpart_name = "v21 digit-wise" if version == "v20" else "v20 atomic"
        paired_behavior_note = f"""
        <div class="callout neutral"><strong>与配对 representation 对照。</strong>本实验 Thinking 的 final/trace AR 为 {pct(final_ar.loc['thinking','ar_final_accuracy'])}/{pct(final_ar.loc['thinking','trace_exact'])}；{counterpart_name} 为 {pct(counterpart_final.loc['thinking','ar_final_accuracy'])}/{pct(counterpart_final.loc['thinking','trace_exact'])}。Nonthinking 则是 {pct(final_ar.loc['nonthinking','ar_final_accuracy'])} 对 {pct(counterpart_final.loc['nonthinking','ar_final_accuracy'])}。因此主要差距出现在需要多步保持语义一致的 trace 闭环，而不是所有输出都会被同方向改善。</div>
        """

    high_power_band_rows: list[list[str]] = []
    final_by_count = read_csv(run_dir / "tables/final_autoregressive_by_count.csv")
    if final_by_count.empty:
        final_by_count = ar[ar["step"] == int(ar["step"].max())].copy()
        final_by_count["examples"] = int(cfg.get("ar_examples_per_count", 2))
    final_by_count["band"] = final_by_count["count"].map(count_band)
    for mode in ("thinking", "nonthinking"):
        for band in ("1–5", "6–10", "11–15", "16–20", "21–25", "26–30"):
            group = final_by_count[
                (final_by_count["mode"] == mode) & (final_by_count["band"] == band)
            ]
            if group.empty:
                continue
            high_power_band_rows.append(
                [
                    MODE_LABEL[mode],
                    band,
                    fmt(int(group["examples"].sum()), 0),
                    pct(float(group["ar_final_accuracy"].mean())),
                    pct(float(group["trace_exact"].mean())) if mode == "thinking" else "—",
                    pct(float(group["trace_ordered_marker_accuracy"].mean()))
                    if mode == "thinking"
                    else "—",
                ]
            )
    paired_final_nt = math.nan
    paired_final_difference = math.nan
    final_detail = read_csv(run_dir / "tables/final_autoregressive_detail.csv")
    paired_keys = ["set_id", "corpus_start", "prompt_sha256", "count"]
    if not final_detail.empty and set(paired_keys).issubset(final_detail.columns):
        thinking_detail = final_detail[final_detail["mode"] == "thinking"][
            [*paired_keys, "ar_accuracy"]
        ].rename(columns={"ar_accuracy": "thinking_accuracy"})
        nonthinking_detail = final_detail[final_detail["mode"] == "nonthinking"][
            [*paired_keys, "ar_accuracy"]
        ].rename(columns={"ar_accuracy": "nonthinking_accuracy"})
        paired = thinking_detail.merge(nonthinking_detail, on=paired_keys, validate="one_to_one")
        if not paired.empty:
            paired_final_nt = float(paired["nonthinking_accuracy"].mean())
            paired_final_difference = float(
                (paired["thinking_accuracy"] - paired["nonthinking_accuracy"]).mean()
            )

    nonthinking_diagnosis_section = ""
    if version == "v20" and not final_detail.empty:
        nt_detail = final_detail[final_detail["mode"] == "nonthinking"].copy()
        nt_answered = nt_detail[
            nt_detail["ar_pred_count"].notna() & nt_detail["count"].notna()
        ].copy()
        nt_answer_rate = float(nt_detail["ar_answered"].mean())
        nt_errors = (
            nt_answered["ar_pred_count"].astype(float)
            - nt_answered["count"].astype(float)
        ).abs()
        nt_mae = float(nt_errors.mean())
        nt_within_one = float((nt_errors <= 1).mean())
        nt_within_two = float((nt_errors <= 2).mean())
        nt_median_error = float(nt_errors.median())
        nt_count_correlation = float(
            nt_answered[["count", "ar_pred_count"]].astype(float).corr().iloc[0, 1]
        )
        nt_by_count = (
            nt_answered.groupby("count", as_index=False)
            .agg(
                mean_prediction=("ar_pred_count", "mean"),
                exact=("ar_accuracy", "mean"),
            )
            .set_index("count")
        )
        diagnostic_count_rows = []
        for count in (1, 5, 10, 15, 20, 25, 30):
            row = nt_by_count.loc[count]
            diagnostic_count_rows.append(
                [
                    fmt(count, 0),
                    fmt(row["mean_prediction"], 2),
                    pct(row["exact"]),
                ]
            )

        train_metrics = read_csv(run_dir / "tables/train_metrics.csv")
        nt_train = train_metrics[train_metrics["mode"] == "nonthinking"].sort_values(
            "step"
        ).iloc[-1]
        nt_total_train_loss = float(nt_train["train_total_loss"])
        nt_count_train_loss = float(nt_train["train_final_count_example_mean_loss"])
        nt_ans_train_loss = float(nt_train["train_ans_token_example_mean_loss"])
        nt_eos_train_loss = float(nt_train["train_eos_example_mean_loss"])
        count_weight_share = float(nt_train["batch_final_count_weight_share"])

        sampling = json.loads(str(nt_train["cumulative_sampling_json"]))
        count_exposure = {
            int(key): int(value)
            for key, value in sampling.get("accepted_counts", {}).items()
        }
        exposure_min = min(count_exposure.values())
        exposure_max = max(count_exposure.values())
        exposure_accuracy_correlation = float(
            np.corrcoef(
                [count_exposure[int(count)] for count in nt_by_count.index],
                [float(nt_by_count.loc[count, "exact"]) for count in nt_by_count.index],
            )[0, 1]
        )

        broad_attention = read_csv(
            run_dir / "analysis/extended/tables/attention_role_dynamics.csv"
        )
        nt_broad = broad_attention[
            (broad_attention["role"] == "nonthinking_broad")
            & (broad_attention["is_fixed_role_head"] == 1)
        ].sort_values("step")
        nt_broad_start = float(nt_broad.iloc[0]["score"])
        nt_broad_end = float(nt_broad.iloc[-1]["score"])

        nonthinking_diagnosis_section = f"""
<section id="nonthinking"><h2>2.1 为什么 Nonthinking 的 exact accuracy 这么低？</h2>
<div class="definition"><strong>先区分“精确计数”和“近似计数”。</strong>令真实计数为 <em>n</em>、自由生成的预测为 <em>n&#770;</em>。Exact accuracy 是 <em>P</em>(<em>n&#770;</em>=<em>n</em>)；MAE 是 E[|<em>n&#770;</em>−<em>n</em>|]；±1 accuracy 是 <em>P</em>(|<em>n&#770;</em>−<em>n</em>|≤1)。Exact 会把差 1 和差很多都记为完全错误，因此必须与误差大小一起看。</div>
<div class="cards">
 <div class="card"><div class="label">Nonthinking exact</div><div class="value">{pct(high_power_ar.loc['nonthinking','ar_final_accuracy'])}</div><div class="small">严格整数完全匹配</div></div>
 <div class="card"><div class="label">MAE</div><div class="value">{fmt(nt_mae,2)}</div><div class="small">median |error|={fmt(nt_median_error,1)}</div></div>
 <div class="card"><div class="label">误差 ≤ 1</div><div class="value">{pct(nt_within_one)}</div><div class="small">误差 ≤ 2：{pct(nt_within_two)}</div></div>
 <div class="card"><div class="label">true/pred correlation</div><div class="value">{fmt(nt_count_correlation,3)}</div><div class="small">answer rate={pct(nt_answer_rate)}</div></div>
</div>
{table(['true count','该组平均预测','exact accuracy'], diagnostic_count_rows)}
<p><strong>直接诊断：</strong>Nonthinking 不是 label collapse，也不是完全没有学习 counting。平均预测随真实 count 几乎一一移动，但单个样本常落在相邻整数上；因此最符合数据的描述是<strong>形成了 noisy、近似连续的 count estimator，却没有形成边界清晰的离散整数状态</strong>。从 1–10 扩展到 1–30 后，输出类别增至三倍，同样约一个 count 的分辨率误差会被 exact-match 更充分地暴露。</p>

<h3>2.1.1 低准确率不是 autoregressive exposure bias</h3>
<p>最终 teacher-forced final exact 为 {pct(tf.loc['nonthinking','tf_final_accuracy'])}，高样本 AR exact 为 {pct(high_power_ar.loc['nonthinking','ar_final_accuracy'])}，两者很接近；而且 {fmt(len(nt_answered),0)} 个高样本 prompts 全部产生了可解析答案。因此主要失败发生在最终 count representation/readout 本身，而不是自由生成时前缀错误逐步累积。</p>

<h3>2.1.2 总训练 loss 掩盖了真正的 count-token loss</h3>
<div class="definition"><strong>task-output 阶段的 Nonthinking loss。</strong>每个样本监督 <code>&lt;Ans&gt;</code>、count、<code>&lt;EOS&gt;</code> 三个位置；三者权重均为 1，所以 count token 只占 active loss weight 的 {pct(count_weight_share)}。最终 batch 中，总训练 loss={fmt(nt_total_train_loss,3)}，但 final-count token 的 example-mean cross-entropy={fmt(nt_count_train_loss,3)}；<code>&lt;Ans&gt;</code> 与 <code>&lt;EOS&gt;</code> 分别只有 {fmt(nt_ans_train_loss,6)} 和 {fmt(nt_eos_train_loss,6)}。于是总 loss 近似为 ({fmt(nt_ans_train_loss,4)}+{fmt(nt_count_train_loss,3)}+{fmt(nt_eos_train_loss,4)})/3={fmt(nt_total_train_loss,3)}。只看 total loss 会错误地认为精确计数已经学好。</div>

<h3>2.1.3 为什么 Thinking 更容易形成精确整数状态</h3>
<ol>
 <li><strong>串行计算深度不同。</strong>Nonthinking 必须在固定 4 层内一次完成 target membership、跨 256 个 data positions 的聚合和 30 类整数化。Thinking 对 count <em>n</em> 生成约 2<em>n</em> 个 trace tokens；每个新 token 都再次运行 Transformer，相当于获得随 <em>n</em> 增长的 recurrent computation，而不只是“多写一段解释”。</li>
 <li><strong>Softmax attention 更自然地产生平均或比例，而非未归一化整数和。</strong>若多个 target positions 携带相似 match vector，对它们做归一化平均不会直接保留 target 数量；Nonthinking 必须从 target mass、背景比例或多层 summary 间接估计 <em>n</em>，容易成为 noisy density estimator。Thinking 则把求和改写成“检索第 k 个 occurrence、推进 k→k+1、判断停止”的迭代过程。</li>
 <li><strong>监督的结构不同。</strong>step 1,501 后，Nonthinking 只有最终 count 提供任务相关的全局误差信号；Thinking 的每个 index 和 marker 都给出与检索、状态推进和停止直接对齐的局部监督。Loss 虽然按 active token 归一化，但 trace 仍显著改善 credit assignment。</li>
 <li><strong>query-first 增加固定深度模型的长程负担。</strong>模型需要让三个 query identities 穿过 256-character data window，并在末端把分散的 membership evidence 聚成一个精确整数；Thinking 的显式 trace 提供了多个中间状态槽。</li>
</ol>
<p><strong>与 attention 证据的一致性：</strong>固定 Nonthinking broad head 的 role score 从 step 0 的 {fmt(nt_broad_start)} 到 step 10,000 的 {fmt(nt_broad_end)}，基本没有增强；Thinking 则形成 successor、targeted retrieval 和 terminal routing 的角色分化。低 broad score 本身不能排除多头或 residual-stream 中的分布式聚合，但结合 MAE、exact 与训练 loss，它说明现有 Nonthinking 电路不足以把近似统计量稳定量化为整数。</p>
<p><strong>数据不足不是首要解释。</strong>各 count 累计 exposure 位于 {exposure_min:,}–{exposure_max:,} examples，exposure 与最终 per-count exact 的 Pearson correlation 仅 {fmt(exposure_accuracy_correlation,3)}。分布并非严格均匀，但最低频 count 也看过三万次以上，且最终错误主要集中在相邻整数，而非高频标签塌缩。</p>
<div class="callout good"><strong>目前结论：</strong>Nonthinking 已学习 count-sensitive manifold/近似统计量，但尚未形成精确、边界稳定的离散计数状态；Thinking 的优势主要来自可迭代计算槽和机制对齐的 trace supervision。<br><strong>欠缺证据：</strong>当前结果还不能把“额外串行计算”与“更好的局部监督”完全分开。最关键的控制是加入不提供 gold trace 的 blank scratch tokens，以及仅加入 membership auxiliary loss；同时增加层数，检验固定深度是否为主要瓶颈。</div>
</section>
"""

    attention_dynamics_section = ""
    if version == "v20":
        attention_table = read_csv(
            run_dir / "analysis/extended/tables/attention_role_dynamics.csv"
        )
        fixed_attention = attention_table[attention_table["is_fixed_role_head"] == 1]

        def attention_milestone(role: str, threshold: float) -> str:
            line = fixed_attention[fixed_attention["role"] == role].sort_values("step")
            return fmt(first_crossing(line, "score", threshold), 0)

        attention_roles = json.loads(
            (run_dir / "analysis/extended/fixed_attention_roles.json").read_text(encoding="utf-8")
        )
        attention_final_rows = []
        for role, label in (
            ("nonthinking_broad", "Nonthinking broad"),
            ("thinking_broad", "Thinking broad"),
            ("targeted_retrieval", "Thinking targeted retrieval"),
            ("marker_successor", "Thinking successor-like"),
        ):
            line = fixed_attention[fixed_attention["role"] == role].sort_values("step")
            head = attention_roles[role]
            attention_final_rows.append(
                [
                    label,
                    f"L{head['layer']}H{head['head']}",
                    fmt(line.iloc[0]["score"]),
                    fmt(line[line["step"] == 1500].iloc[0]["score"]),
                    fmt(line.iloc[-1]["score"]),
                ]
            )
        attention_embed = embedded_iframe(
            run_dir / "analysis/extended/interactive_attention_dynamics.html",
            "v20 attention head role dynamics",
            css_class="attention-dynamics",
        )
        attention_dynamics_section = f"""
        <h3>7.1 可拖动 checkpoint 的 4×4 head-role 图</h3>
        <div class="definition"><strong>Broad attention score。</strong>在最终 <code>&lt;Ans&gt;</code> query，先令 M<sub>h</sub>=Σ<sub>j∈needles</sub>A<sub>h</sub>(Ans,j) 为该 head 落在全部 target occurrences 上的总质量；再令 H<sub>h</sub>=−Σp<sub>j</sub>log p<sub>j</sub>/log n，其中 p<sub>j</sub>=A<sub>h</sub>(Ans,j)/M<sub>h</sub>。定义 B<sub>h</sub>=M<sub>h</sub>H<sub>h</sub>。它同时要求“注意 needle 集合”与“在 n 个 occurrences 间广泛覆盖”，范围 0–1。Targeted 与 successor-like 分数分别沿用第 6.2 与第 6.1 节定义。四个固定 head 均在独立 selection split 的 final checkpoint 选出；滑动曲线使用不重叠 reporting split。</div>
        {table(["角色", "固定 head", "step 0", "step 1,500", "step 10,000"], attention_final_rows)}
        {attention_embed}
        <p>拖动上方 training-step 控件可逐个查看 101 个 checkpoints。每个 panel 的 4 行是 Layer 1–4，4 列是 Head 0–3，格内数字是原始 role score；黑框是最终固定 head。各 panel 独立设色标以免 broad score 被强 retrieval mass 淹没，因此跨 panel 必须比较数字而不是颜色深浅。下方折线统一使用 0–1 纵轴，显示四个固定 head 的原始分数。</p>
        <p>动力学上，L2H3 successor-like 在 step {attention_milestone('marker_successor', .5)} 已超过 0.5；L4H2 targeted retrieval 分别在 step {attention_milestone('targeted_retrieval', .1)} / {attention_milestone('targeted_retrieval', .5)} 超过 0.1 / 0.5；Thinking 的 L4H0 broad readout 到 step {attention_milestone('thinking_broad', .1)} 才超过 0.1，而 Nonthinking L1H2 到最后仍只有 {fmt(fixed_attention[fixed_attention['role']=='nonthinking_broad'].sort_values('step').iloc[-1]['score'])}。这支持“先搭控制流，再形成逐项定位，最后形成 Thinking terminal broad/readout”的级联；它不是四种机制在同一步突然出现。</p>
        """

    phase_audit_section = ""
    phase_audit_dir = run_dir / "analysis/phase_transition_audit"
    if version == "v20" and (
        phase_audit_dir / "tables/aggregate_transition_model_comparison.csv"
    ).exists():
        routing_audit = read_csv(phase_audit_dir / "tables/routing_qk_by_k.csv")
        routing_summary_rows = []
        for step, group in routing_audit.groupby("step"):
            weights = group["observations"].to_numpy(dtype=float)
            routing_summary_rows.append(
                {
                    "step": int(step),
                    **{
                        metric: float(np.average(group[metric], weights=weights))
                        for metric in (
                            "targeted_mass",
                            "qk_margin",
                            "correct_occurrence_top1",
                        )
                    },
                }
            )
        routing_summary = pd.DataFrame(routing_summary_rows).sort_values("step")

        def first_nonnegative(frame: pd.DataFrame, column: str) -> int | None:
            selected = frame[frame[column] >= 0].sort_values("step")
            return None if selected.empty else int(selected.iloc[0]["step"])

        qk_zero_step = first_nonnegative(routing_summary, "qk_margin")
        top1_half_step = first_crossing(
            routing_summary, "correct_occurrence_top1", 0.5
        )

        patch_audit = read_csv(
            phase_audit_dir / "tables/retrieval_transport_recovery.csv"
        )
        value_top2 = patch_audit[
            (patch_audit["intervention"] == "value_only_at_target_source")
            & (patch_audit["top_n"] == 2)
        ].sort_values("step")
        value_top1 = patch_audit[
            (patch_audit["intervention"] == "value_only_at_target_source")
            & (patch_audit["top_n"] == 1)
        ].sort_values("step")
        residual_l3 = patch_audit[
            (patch_audit["intervention"] == "residual_stream")
            & (patch_audit["residual_layer"] == 3)
        ].sort_values("step")

        causal_audit = read_csv(
            phase_audit_dir / "tables/local_head_causal_damage.csv"
        )
        causal_fixed = causal_audit[
            causal_audit["intervention"] == "fixed_head_zero"
        ]

        high_ar_audit = read_csv(
            phase_audit_dir / "tables/high_power_ar_summary.csv"
        )
        high_ar_rows = []
        for step in sorted(high_ar_audit["step"].unique()):
            row = [fmt(step, 0)]
            for mode in ("nonthinking", "thinking"):
                selected = high_ar_audit[
                    (high_ar_audit["step"] == step)
                    & (high_ar_audit["mode"] == mode)
                ]
                if selected.empty:
                    row.append("—")
                    continue
                record = selected.iloc[0]
                low, high = _wilson_interval(
                    float(record["successes"]), int(record["examples"])
                )
                row.append(
                    f"{pct(record['ar_accuracy'])} [{pct(low)}, {pct(high)}]"
                )
            high_ar_rows.append(row)

        def high_ar_value(step: int, mode: str, column: str) -> float:
            selected = high_ar_audit[
                (high_ar_audit["step"] == step)
                & (high_ar_audit["mode"] == mode)
            ]
            return math.nan if selected.empty else float(selected.iloc[0][column])

        aggregate_fits = read_csv(
            phase_audit_dir / "tables/aggregate_transition_model_comparison.csv"
        )
        fit_rows = []
        family_labels = {
            "behavior": "行为",
            "routing": "routing",
            "attention_role": "固定 head role",
            "transport": "value/residual transport",
            "causality": "causal damage",
        }
        classification_labels = {
            "strong_changepoint_preference": "强 changepoint 偏好",
            "moderate_changepoint_preference": "中等 changepoint 偏好",
            "smooth_preference": "平滑模型偏好",
            "inconclusive": "不能区分",
            "insufficient": "数据不足",
        }
        for record in aggregate_fits.itertuples(index=False):
            fit_rows.append(
                [
                    family_labels.get(str(record.evidence_family), str(record.evidence_family)),
                    str(record.group),
                    str(record.smooth_model),
                    fmt(record.smooth_center_x, 0),
                    fmt(record.smooth_width_10_90, 0),
                    str(record.changepoint_model),
                    fmt(record.candidate_x, 0),
                    fmt(record.delta_bic_smooth_minus_changepoint, 1),
                    classification_labels.get(
                        str(record.classification), str(record.classification)
                    ),
                ]
            )

        per_k_fits = read_csv(
            phase_audit_dir / "tables/per_k_transition_model_comparison.csv"
        )
        per_k_rows = []
        for (axis, metric), group in per_k_fits.groupby(["axis", "metric"]):
            strong = group[group["delta_bic_smooth_minus_changepoint"] >= 10]
            candidate = (
                float(strong["candidate_x"].median()) if not strong.empty else math.nan
            )
            per_k_rows.append(
                [
                    "training step" if axis == "training_step" else "semantic exposure",
                    str(metric),
                    f"{len(strong)}/{len(group)}",
                    fmt(candidate, 0),
                ]
            )

        strong_aggregate = aggregate_fits[
            aggregate_fits["delta_bic_smooth_minus_changepoint"] >= 10
        ]
        strong_aggregate_labels = "、".join(
            f"{row.evidence_family}/{row.group}"
            for row in strong_aggregate.itertuples(index=False)
        ) or "无"
        residual_l3_max_restoration = float(
            residual_l3["margin_restoration"].abs().max()
        )

        def aggregate_fit_record(family: str, group: str) -> pd.Series:
            selected = aggregate_fits[
                (aggregate_fits["evidence_family"] == family)
                & (aggregate_fits["group"] == group)
            ]
            if selected.empty:
                raise ValueError(f"missing aggregate fit: {family}/{group}")
            return selected.iloc[0]

        successor_role_fit = aggregate_fit_record(
            "attention_role", "marker_successor"
        )
        targeted_role_fit = aggregate_fit_record(
            "attention_role", "targeted_retrieval"
        )
        targeted_causal_fit = aggregate_fit_record(
            "causality", "targeted_retrieval"
        )
        value_top1_fit = aggregate_fit_record("transport", "value_top1")
        value_top2_fit = aggregate_fit_record("transport", "value_top2")
        thinking_behavior_fit = aggregate_fit_record("behavior", "thinking")

        functional_transition_rows = [
            [
                "Marker-successor routing",
                f"center≈{fmt(successor_role_fit['smooth_center_x'],0)}；10–90% width≈{fmt(successor_role_fit['smooth_width_10_90'],0)} steps",
                "快速涌现",
                "all-sequence 阶段的每个 trace marker 都提供局部 k→k+1/close 监督；QK 分数一旦拉开，softmax 会把 attention mass 快速集中到同一 index。",
            ],
            [
                "Targeted retrieval routing/QK",
                f"center≈{fmt(targeted_role_fit['smooth_center_x'],0)}；width≈{fmt(targeted_role_fit['smooth_width_10_90'],0)} steps；ΔBIC={fmt(targeted_role_fit['delta_bic_smooth_minus_changepoint'],1)}",
                "平缓形成",
                "需要同时形成语义 k、RoPE 位置匹配及多 occurrence 竞争；不同 k 的 exposure 和难度不同，使大量局部改进在总体上被摊宽。",
            ],
            [
                "Needle-identity value transport",
                f"top-1 width≈{fmt(value_top1_fit['smooth_width_10_90'],0)}；top-2 width≈{fmt(value_top2_fit['smooth_width_10_90'],0)} steps",
                "平缓且分布式",
                "单 head 只传输部分 identity；两个 L4 retrieval heads 合起来才接近完整恢复，因此没有单一组件跨阈值后立即接管。",
            ],
            [
                "Targeted-head causal dependence",
                f"center≈{fmt(targeted_causal_fit['smooth_center_x'],0)}；width≈{fmt(targeted_causal_fit['smooth_width_10_90'],0)} steps；ΔBIC={fmt(targeted_causal_fit['delta_bic_smooth_minus_changepoint'],1)}",
                "平缓增强",
                "head 的 routing 先逐渐变准，下游 residual/readout 再逐渐依赖其输出；“看向正确位置”与“成为不可替代组件”不是同一个瞬间。",
            ],
            [
                "Thinking free-generation accuracy",
                f"5,500→6,000: {pct(high_ar_value(5500,'thinking','ar_accuracy'))}→{pct(high_ar_value(6000,'thinking','ar_accuracy'))}；fit ΔBIC={fmt(thinking_behavior_fit['delta_bic_smooth_minus_changepoint'],1)}",
                "表观陡升，机制证据不足",
                "whole-trace/final exact 是多个局部决策的合取；逐 marker 正确率平滑提高会被长 trace 的乘法结构放大成准确率 cliff。",
            ],
        ]

        final_value1 = value_top1.iloc[-1]
        final_value2 = value_top2.iloc[-1]
        final_l3 = residual_l3.iloc[-1]
        targeted_damage = causal_fixed[
            causal_fixed["role"] == "targeted_retrieval"
        ].sort_values("step")
        successor_damage = causal_fixed[
            causal_fixed["role"] == "marker_successor"
        ].sort_values("step")
        phase_audit_section = f"""
        <h3>7.2 逐功能 phase-transition audit：哪些快速涌现，哪些平缓形成</h3>
        <div class="definition"><strong>这里的 phase transition 是功能级概念。</strong>不要求 successor、targeted retrieval、value transport 与最终行为在同一步出现，而是分别询问每条功能曲线是窄窗口内快速完成，还是跨较宽训练区间连续形成。模型比较同时报告：① ΔBIC=BIC<sub>smooth</sub>−BIC<sub>changepoint</sub>，正值偏向折点模型；② 最佳 sigmoid 从渐近变化量 10% 到 90% 的 <em>transition width</em>。窄 sigmoid 即使数学上连续，也可以表示功能上的快速涌现；宽 sigmoid 则表示渐进专门化。单个 noisy breakpoint 只有在效应量足够大、变化后保持稳定时才有机制意义。</div>
        <div class="definition"><strong>三个新增 retrieval 指标。</strong>对最终 checkpoint 在独立 selection split 选出的固定 L4H2，在 trace index <em>k</em> 处定义：① <em>T</em><sub>k</sub>(<em>t</em>)=A(index<sub>k</sub>, occurrence<sub>k</sub>)，即正确 occurrence 的 targeted mass；② <em>QK margin</em>=s(index<sub>k</sub>, occurrence<sub>k</sub>)−max<sub>j≠k</sub>s(index<sub>k</sub>, occurrence<sub>j</sub>)，其中 s=q·k/√d 是 softmax 前分数；③ <em>correct-occurrence top-1</em> 是该 margin&gt;0 的样本比例。每个 checkpoint 对每个 true count 使用 1 条 reporting prompt，因此 k 的 observations 为 31−k；高 k 更噪。</div>
        {figure(phase_audit_dir/'figures/per_k_routing_qk_dynamics.png','8','逐 k 的 retrieval routing 与判别能力','三个 heatmap 的横轴均为 optimizer training step，纵轴为语义 index k=1…30；白色竖虚线是 step 1,500 objective switch。上图颜色为 L4H2 指向正确 prompt occurrence 的 attention mass；中图颜色为 correct−best-wrong scaled QK margin，0 表示正确 occurrence 刚好与最强错误 occurrence 打平；下图颜色为 correct-occurrence top-1 accuracy。L4H2 在独立 final selection split 固定后回看全部 checkpoints，没有逐步重选 head。')}
        <p>按 observations 加权后，QK margin 在 step {fmt(qk_zero_step,0)} 首次转正，correct-occurrence top-1 在 step {fmt(top1_half_step,0)} 首次超过 0.5。这比“attention mass 变大”更严格：它说明正确 occurrence 已经在 QK 竞争中战胜所有错误 occurrences，而不只是从很小变成稍大。</p>
        {figure(phase_audit_dir/'figures/high_power_ar_by_count_dynamics.png','9','3,000–7,000 窗口的高样本自由生成动力学','四个 heatmap 的横轴均为 optimizer training step（每 500 steps 一个 checkpoint），纵轴为 true count n=1…30；每个格固定包含 50 条独立 test-task prompts。左上/右上颜色分别为 Nonthinking/Thinking final-count exact accuracy；左下为 Thinking whole-trace exact；右下为 Thinking ordered-marker accuracy。所有 panel 共用 0–1 色标。Whole-trace exact 要求整条 trace 完全一致，随 n 增大比逐 marker accuracy 更容易因一次局部错误归零。')}
        <p>高样本行为显示的是一个明显但并非瞬时的 crossover：Thinking final-count exact 从 step 3,000 的 {pct(high_ar_value(3000,'thinking','ar_accuracy'))}，升到 5,500 的 {pct(high_ar_value(5500,'thinking','ar_accuracy'))}、6,000 的 {pct(high_ar_value(6000,'thinking','ar_accuracy'))} 与 7,000 的 {pct(high_ar_value(7000,'thinking','ar_accuracy'))}；同期 Nonthinking 仅从 {pct(high_ar_value(3000,'nonthinking','ar_accuracy'))} 升到 {pct(high_ar_value(7000,'nonthinking','ar_accuracy'))}。Thinking 的 ordered-marker accuracy 则从 {pct(high_ar_value(3000,'thinking','trace_ordered_marker_accuracy'))} 连续升到 {pct(high_ar_value(7000,'thinking','trace_ordered_marker_accuracy'))}。因此 5,500–6,000 是<strong>行为加速窗口</strong>，但不能仅凭全局 exact accuracy 的陡升就把它称为机制相变：长 trace 会把逐 marker 的连续改进乘法放大。</p>
        <div class="definition"><strong>Transport 与 causal damage。</strong>沿用长度不变的 identity corruption：同步替换 prompt 第 k 个 occurrence 与 gold marker。Value-only patch 只把指定 source 的 clean V 写入 corrupt run；residual patch 把 index<sub>k</sub> 处某层后的 clean residual 写入 corrupt run。Normalized recovery 为 R=(m<sub>patch</sub>−m<sub>corrupt</sub>)/(m<sub>clean</sub>−m<sub>corrupt</sub>)；当 clean−corrupt margin 很小时 R 不稳定，所以动力学拟合使用原始 <em>margin restoration</em>=m<sub>patch</sub>−m<sub>corrupt</sub>。Causal damage 定义为 baseline margin−局部 head-zero margin，越大表示该角色 head 越必要。</div>
        {figure(phase_audit_dir/'figures/synchronized_phase_evidence.png','10','各项功能自己的形成速度、因果依赖与行为结果','横轴除右下 panel 外均为 optimizer training step，竖虚线为 step 1,500 objective switch。左上是每个 checkpoint 每 count 50 条、共 1,500 条的自由生成 exact accuracy及 95% Wilson interval；中上是加权 targeted mass 与 correct-occurrence top-1；右上是加权 QK margin；左下是 value-only / residual patch 带来的正确-marker logit-margin restoration，其中 post-L4 residual 是直接复制最终 query state 的上界；中下是局部清零固定 successor/targeted head 造成的 causal damage；右下逐项给出 ΔBIC，正值偏向 changepoint，红虚线 10 是强偏好的描述性阈值。各 panel 独立判断快慢，不要求它们同步。')}
        {table(['step','Nonthinking 50/count AR [95% CI]','Thinking 50/count AR [95% CI]'], high_ar_rows)}
        <p>最终 checkpoint 的 value-only top-1 recovery={fmt(final_value1['normalized_recovery_mean'])}，top-2={fmt(final_value2['normalized_recovery_mean'])}；post-L3 residual recovery={fmt(final_l3['normalized_recovery_mean'])}。Top-2 几乎完整恢复而 top-1 很低，说明 identity transport 分布在至少两个 L4 retrieval heads；post-L3 几乎不能恢复、post-L4 完整恢复则把 identity 写入定位在 L4。Successor causal damage 从早期就很强，最终为 {fmt(successor_damage.iloc[-1]['causal_damage'])} logit；targeted damage 晚得多，最终为 {fmt(targeted_damage.iloc[-1]['causal_damage'])}。</p>
        {figure(phase_audit_dir/'figures/routing_by_semantic_exposure.png','11','把横轴换成每个 k 自己的 semantic token exposure','三个 panel 纵轴分别为 targeted mass、correct−best-wrong QK margin 与 correct-occurrence top-1；横轴不再是 optimizer step，而是相应语义 k 在训练中累计出现的 trace-index token 数（百万）。每条线对应 k=1、5、10、15、20、25、30。若不同 k 的曲线在 exposure 轴比 step 轴更对齐，更支持 exposure/curriculum 解释，而不是共同 wall-clock step 触发的新机制。k=30 每个 checkpoint 只有 1 个 reporting observation，应视为高方差边界。')}
        <div class="definition"><strong>模型比较。</strong>“平滑模型”取连续线性增长与四参数 sigmoid 中 BIC 更低者；“changepoint 模型”取连续折线（斜率改变）与允许水平及斜率同时改变的分段线性模型中 BIC 更低者。定义 ΔBIC=BIC<sub>smooth</sub>−BIC<sub>change</sub>：ΔBIC&gt;0 偏向 changepoint，≥10 记为强描述性偏好。这里的相邻 checkpoints 高度自相关，BIC 不是独立重复实验的显著性检验。</div>
        {figure(phase_audit_dir/'figures/per_k_changepoint_model_evidence.png','12','逐 k 判断 targeted retrieval 是突然还是平缓形成','三列依次对应 targeted mass、QK margin 与 correct-occurrence top-1；上排使用 training step 拟合，下排使用每个 k 自己的 semantic exposure 拟合。每个 panel 横轴为 k，纵轴为 ΔBIC，并使用对称 log 刻度；0 线上方偏向 changepoint，红虚线 ΔBIC=10 为强偏好。Targeted mass 的 30 个 k 全部偏好平滑模型；少数 QK/top-1 的正值主要来自二值 top-1、较少 observations 与高 k 噪声，不能替代连续 margin/mass 的证据。')}
        {table(['证据族','分组','最佳平滑模型','sigmoid center','10–90% width','最佳 changepoint 模型','候选 step','ΔBIC','判断'], fit_rows)}
        {table(['拟合横轴','逐 k 指标','强 changepoint k 数','强证据的中位候选位置'], per_k_rows)}
        <h4>逐功能结论</h4>
        {table(['功能','定量形状','判断','为什么会呈现这种形状'], functional_transition_rows)}
        <p>聚合层面 ΔBIC≥10 的项目只有：{strong_aggregate_labels}。其中 post-L3 residual 的最大原始 margin restoration 也只有 {fmt(residual_l3_max_restoration)} logit，且随后回落到近零；这是小幅、非单调瞬态，不应称为 residual transport 的突然涌现。相比之下，successor role score 在 step 100/200/300 约为 0.006/0.553/0.937，最佳 sigmoid center≈{fmt(successor_role_fit['smooth_center_x'],0)}、10–90% width≈{fmt(successor_role_fit['smooth_width_10_90'],0)} steps：它是本实验中最清楚的<strong>单功能快速涌现</strong>。Targeted retrieval 的对应 width≈{fmt(targeted_role_fit['smooth_width_10_90'],0)} steps，属于宽而平滑的形成过程。</p>
        <p>Exposure 横轴在本次固定 count distribution 下几乎是 training step 的按 k 线性缩放，所以它能揭示不同 k 接收的样本量，却不是独立 intervention；逐 k 的 step/exposure ΔBIC 因而基本不变。要真正区分 exposure、curriculum 与 phase transition，仍需改变 count distribution、objective-switch step 或 sequence length 后重新训练，看候选窗口是随 token exposure 移动，还是锁定在共同 optimizer step。</p>
        <div class="callout neutral"><strong>本轮逐功能判断：</strong>marker-successor attention routing 在约 step 100–300 快速涌现；targeted retrieval routing、QK discrimination、identity transport 与 targeted causal dependence 都是中后期的平缓形成；Thinking AR 在 5,500–6,000 有表观陡升，但现有证据更支持逐 marker 改进被长 trace 放大，而不是新 retrieval 功能在该处突然出现。<br><strong>欠缺证据：</strong>当前只有 seed 1234；successor 的早期 causal intervention 仍需在 step 100–500 加密，objective-switch timing、count distribution、sequence length controls 也尚未实际训练。每个功能是否可称为 phase transition，应分别在多 seed 中检验其 transition center/width 是否稳定，并排除 exposure、softmax 阈值化和 exact-match 乘法放大。</div>
        """

    v20_port = run_dir / "analysis/v10_port/manifest.json"
    port_section = ""
    causal_port_results = ""
    if version == "v20":
        if v20_port.exists():
            port_manifest = json.loads(v20_port.read_text(encoding="utf-8"))
            port_rows = [[item["name"], fmt(item["rows"], 0)] for item in port_manifest.get("tables", [])]
            port_section = f"""
            <div class="callout good"><strong>完整 causal port 已在本地补跑。</strong>原始 Colab manifest 的 causal stage 因新版 phase 存储与旧版接口不兼容而失败；报告使用修复后的 CPU rerun。共生成 {len(port_rows)} 张因果/representation 表。下面的“局部 head 因果”来自原始 phase stage，不依赖该补跑。</div>
            <details><summary>查看完整 causal port 产物清单</summary>{table(["table", "rows"], port_rows)}</details>
            """

            port_tables = run_dir / "analysis/v10_port/tables"
            retrieval_patch = read_csv(port_tables / "retrieval_head_patching.csv")
            successor_patch = read_csv(port_tables / "successor_head_patching.csv")
            residual_transport = read_csv(port_tables / "residual_count_transport.csv")
            early_stop = read_csv(port_tables / "trace_early_stop_patching.csv")
            final_bridge = read_csv(port_tables / "final_bridge_component_patching.csv")
            conflicts = read_csv(port_tables / "length_preserving_trace_conflicts.csv")
            state_to_head = read_csv(port_tables / "state_to_head_routing.csv")
            localization_transport = read_csv(
                port_tables / "retrieval_localization_transport_patching.csv"
            )

            def group_mean(frame: pd.DataFrame, query: str, column: str) -> float:
                selected = frame.query(query)
                return float(selected[column].mean()) if not selected.empty else math.nan

            retrieval_ranked = group_mean(
                retrieval_patch, "path_kind == 'ranked' and top_n == 2", "normalized_recovery"
            )
            retrieval_random = group_mean(
                retrieval_patch, "path_kind == 'random' and top_n == 2", "normalized_recovery"
            )
            successor_continue = group_mean(
                successor_patch,
                "direction == 'continue_to_close' and path_kind == 'ranked' and top_n == 1",
                "normalized_recovery",
            )
            successor_close = group_mean(
                successor_patch,
                "direction == 'close_to_continue' and path_kind == 'ranked' and top_n == 4",
                "normalized_recovery",
            )
            shift_minus = group_mean(
                residual_transport,
                "mode == 'thinking' and layer == 4 and intervention == 'centroid_delta_alpha_1' and offset == -1",
                "expected_count_shift",
            )
            shift_plus = group_mean(
                residual_transport,
                "mode == 'thinking' and layer == 4 and intervention == 'centroid_delta_alpha_1' and offset == 1",
                "expected_count_shift",
            )
            early_stop_shift = group_mean(early_stop, "layer == 4", "close_margin_shift")
            early_stop_flip = group_mean(early_stop, "layer == 4", "patched_close_decision")
            bridge_attention = group_mean(
                final_bridge, "layer == 2 and component == 'attention_output'", "normalized_recovery"
            )
            follows_trace = group_mean(
                conflicts, "intervention == 'prompt_minus_one_trace_clean'", "follows_original_n"
            )
            routing_l4h2 = group_mean(state_to_head, "layer == 4 and head == 2", "routing_shift")

            mediation_rows = []
            for intervention, top_n, label in (
                ("attention_pattern_only", 1, "Pattern only · top-1"),
                ("value_only_at_target_source", 1, "Value only · top-1"),
                ("pattern_plus_value", 1, "Pattern + value · top-1"),
                ("attention_pattern_only", 2, "Pattern only · top-2"),
                ("value_only_at_target_source", 2, "Value only · top-2"),
                ("pattern_plus_value", 2, "Pattern + value · top-2"),
                ("residual_stream", 0, "Post-L4 residual stream"),
            ):
                selected = localization_transport[
                    (localization_transport["intervention"] == intervention)
                    & (localization_transport["top_n"] == top_n)
                ]
                mediation_rows.append(
                    [
                        label,
                        fmt(float(selected["normalized_recovery"].mean())),
                        pct(float(selected["patched_correct"].mean())),
                        fmt(int(len(selected)), 0),
                    ]
                )

            causal_rows = [
                ["Needle identity retrieval", "patch top-2 ranked heads", fmt(retrieval_ranked), f"random top-2={fmt(retrieval_random)}"],
                ["Continue→close", "patch top-1 successor head", fmt(successor_continue), "decision flip=100%"],
                ["Close→continue", "patch top-4 ranked heads", fmt(successor_close), "decision flip=79.2%"],
                ["Count-state steering", "L4 centroid delta α=1", f"{fmt(shift_minus)} / +{fmt(shift_plus)}", "donor offset −1 / +1"],
                ["Early-stop state", "L4 same-k final→interior patch", fmt(early_stop_shift), f"close decision={pct(early_stop_flip)}"],
                ["Final bridge", "patch L2 attention output", fmt(bridge_attention), "corrupt→clean margin recovery"],
                ["State→retrieval routing", "progress 8 residual→progress 3 before L4H2", fmt(routing_l4h2), "needle-8 minus needle-3 mass shift"],
                ["Prompt/trace conflict", "prompt count n−1, clean trace n", pct(follows_trace), "final output follows trace n"],
            ]
            causal_port_results = f"""
            <h3>9.1 v20 的机制级 patching</h3>
            <div class="definition"><strong>Normalized recovery。</strong>对 clean、corrupt、patched 三种条件的正确-token margin，定义 R=(m<sub>patched</sub>−m<sub>corrupt</sub>)/(m<sub>clean</sub>−m<sub>corrupt</sub>)；R=0 表示没有修复，R=1 表示恢复到 clean。<strong>Expected-count shift。</strong>对 count logits 做 softmax 后求期望值，报告 patch 前后期望 count 的差。<strong>Routing shift。</strong>把 progress-8 residual patch 到 progress-3 后，L4 targeted head 对 needle-8 与 needle-3 attention mass 之差相对 baseline 的改变。</div>
            {table(["问题", "干预", "平均效应", "对照/解释"], causal_rows)}
            <p>这组结果把描述性 attention 补成了一条更具体的因果链：top-2 ranked retrieval heads 几乎完全恢复被替换 needle 的 identity，而随机 top-2 几乎无效；progress residual 又能把同一 targeted head 的路由从第 3 个 needle 重定向到第 8 个。也就是说，<strong>hidden state 中的 k 充当检索 query，targeted heads 执行按 k 取值</strong>。Successor 的 continue→close 在 top-1 head 上已经高度集中，反向 close→continue 则需要多 head；终止状态可在 L4 residual 中被强制写入。最终答案更跟随 clean trace 而不是被减一的 prompt，且 L2 attention patch 可恢复 final bridge。</p>
            <h3>9.2 定位与 identity 传输的 mediation 分解</h3>
            <div class="definition"><strong>同一 clean/corrupt pair。</strong>在 prompt 的第 k 个 occurrence 位置，把字符 identity 替换为同一 query 集合中的另一个字符，并同步替换 gold trace marker；长度、位置、true total count 与 k 均不变。<strong>Pattern-only patch</strong>只把 trace-index query 的 attention probability row 换成 clean，保留 corrupt value vectors。<strong>Value-only patch</strong>只把被检索 prompt source 上的 V 向量换成 clean，保留 corrupt attention pattern。<strong>Residual patch</strong>把 retrieval layer 之后该 query 的完整 clean residual 写入 corrupt run。所有数值均是前面定义的 normalized recovery。</div>
            {table(["干预", "平均 recovery", "patched marker 正确率", "pairs"], mediation_rows)}
            <p>结果非常分离：top-2 pattern-only recovery 只有 {fmt(group_mean(localization_transport, "intervention == 'attention_pattern_only' and top_n == 2", 'normalized_recovery'))}，而 top-2 value-only 达到 {fmt(group_mean(localization_transport, "intervention == 'value_only_at_target_source' and top_n == 2", 'normalized_recovery'))}，pattern+value 为 {fmt(group_mean(localization_transport, "intervention == 'pattern_plus_value' and top_n == 2", 'normalized_recovery'))}，post-L4 residual 为 {fmt(group_mean(localization_transport, "intervention == 'residual_stream'", 'normalized_recovery'))}。在这个<strong>位置不变、只改变 identity</strong>的 corruption 下，attention pattern 本来仍可指向第 k 个位置，所以把 clean pattern 写回几乎不增加信息；恢复主要来自该 source 的 value。结合此前 progress-residual 会重定向 L4H2 路由的实验，更具体的分工是：<strong>query/residual 决定“去第 k 个位置”，attention pattern 执行定位，V 向量携带“那里是什么字符”</strong>。</p>
            <div class="callout warning"><strong>仍不能写成“单一 head 就是完整计数器”。</strong>retrieval 需要两个 L4 heads 才接近完全恢复；close→continue 需要约四个 heads；final-query 单 head patch 会破坏输出，却没有产生与 donor offset 对齐的 clean expected-count shift。因此更准确的说法是：状态、retrieval、successor 与 terminal bridge 构成分布式循环电路。</div>
            """
        else:
            port_section = """
            <div class="callout warning"><strong>完整 causal port 缺失。</strong>原始 v20 manifest 将 causal 标为 failed；当前可以支持局部 head 因果结论，但 retrieval corruption、residual transport、trace-conflict 等更强证据尚不可用。因此本报告不会把最终 answer readout 写成已被完整识别的电路。</div>
            """
            causal_port_results = "<p>完整 causal port 未落盘；本节只解释位置局部消融。</p>"
    else:
        port_section = """
        <div class="callout neutral"><strong>v10 full causal port 不适用于 v21。</strong>其中若干 intervention 假定一个整数对应一个 atomic token；v21 的两位数需要多 token patch 才能保持语义与长度对齐。v21 仍有完整的 milestone position-local head ablation。</div>
        """
        causal_port_results = ""

    representation_failure_section = ""
    if version == "v21":
        by_k = read_csv(run_dir / "analysis/phase_transition/tables/dense_fixed_head_by_k.csv")
        by_k_behavior = read_csv(run_dir / "analysis/phase_transition/tables/dense_behavior_by_k.csv")
        final_step = int(by_k["step"].max())
        targeted_by_k = by_k[
            (by_k["step"] == final_step) & (by_k["role"] == "targeted_retrieval")
        ]
        marker_by_k = by_k_behavior[
            (by_k_behavior["step"] == int(by_k_behavior["step"].max()))
            & (by_k_behavior["outcome"] == "trace_marker_teacher_forced")
        ][["k", "accuracy"]]
        targeted_by_k = targeted_by_k.merge(marker_by_k, on="k", how="left")
        failed = targeted_by_k[targeted_by_k["accuracy"] == 0].sort_values("k")
        failed_text = "、".join(
            f"k={int(row.k)}（mass={row.score:.3f}）" for row in failed.itertuples(index=False)
        )
        representation_failure_section = f"""
        <div class="callout warning"><strong>Digit-wise 的局部失败口袋。</strong>最终 dense reporting example 中，marker prediction 在 {failed_text} 失败；它们同时落在 targeted-head mass 的最低端。失败并非覆盖所有两位数，因此不能归因于“长度从 1 token 变 2 tokens”这一项本身；更像是某些 digit 组合造成 index anchor 表示混淆或路由别名。由于每个 k 只有一个 reporting example，这只是需要扩大样本验证的机制候选。</div>
        """

    if version == "v20":
        geometry_interpretation = """
        <p><strong>几何解释。</strong>Atomic 表示的最终 trace-marker centroid 有较高 effective dimension、较低前三主成分占比，但 adjacent-between/within separation 较强：不同 k 被清楚分开，却不是沿一条直线排列。它更像“每个离散计数状态占据高维流形上的一个槽位”。</p>
        """
    else:
        geometry_interpretation = """
        <p><strong>几何解释。</strong>Digit-wise 表示把最终 trace-marker centroid 压缩到更低维、让前三主成分覆盖更多方差，但 adjacent-between/within separation 反而弱于 atomic v20，且自由生成更差。低维并不等于机制更好；这里更可能是共享 digit 让状态复用并发生折叠，降低某些组合的可分性。</p>
        """

    issue_list = "".join(f"<li>{html.escape(issue)}</li>" for issue in issues) or "<li>未发现核心表或 checkpoint 缺失。</li>"
    model_params = int(read_csv(run_dir / "tables/model_specifications.csv").iloc[0]["parameters"])
    objective = html.escape(str(cfg["training_objective"]))
    manifold_embed = embedded_iframe(
        run_dir / "analysis/phase_transition/interactive_manifold_3d.html",
        "interactive hidden-state manifold",
    )
    ablation_section = ""
    if version == "v20":
        design_path = ROOT / "colab_results/v20_phase_ablation_suite/v20_phase_ablation_design.json"
        design = json.loads(design_path.read_text(encoding="utf-8"))
        design_rows = []
        for experiment in design["experiments"]:
            design_rows.append(
                [
                    html.escape(str(experiment["name"])),
                    html.escape(str(experiment["factor"])),
                    fmt(experiment["max_steps_for_language_pred"], 0),
                    html.escape(str(experiment["training_count_distribution"])),
                    fmt(experiment["seq_len"], 0),
                    fmt(experiment["n_positions"], 0),
                ]
            )
        ablation_section = f"""
        <section id="controls"><h2>10. 区分 exposure、curriculum 与 phase transition 的配对控制</h2>
        <div class="definition"><strong>Objective-switch control。</strong>只改变 all-sequence→task-output 的边界 step。若机制变化始终锁定在 switch 附近，而非固定 token exposure，支持 curriculum trigger。<strong>Count-distribution control。</strong><code>natural</code> 先抽 corpus window 再接受 count；<code>uniform</code> 先均匀抽 semantic count k，再从一个 batch 共享的 natural-candidate stream 填满各 count bucket，使 1–30 的 accepted-example exposure 近似平衡。共享候选流避免逐样本独立 rejection 带来的约 30× 采样开销。若不同 step 但相同累计 E<sub>k</sub> 附近出现转折，支持 exposure explanation。<strong>Sequence-length control。</strong>比较 L=128/256/384，并同步把 context 上限设为 256/384/512；若按绝对位置或 length 移动，而不按 count exposure 对齐，说明路由距离/上下文负担参与形成。</div>
        {table(["run", "单独改变的因素", "switch step", "count distribution", "data length", "context"], design_rows)}
        <p>所有控制均为 RoPE/Thinking、count 1–30、seed 1234、10,000 steps、每 100 steps checkpoint；除表中一项外保持 v20 baseline。分析时对<strong>每个功能分别</strong>报告 optimizer step、objective-relative step（step−switch）、semantic exposure、transition center 与 10–90% width。若某项功能在多个 seed 中稳定保持窄 transition width，并且其位置不能由 objective switch、exposure 或 sequence length 的变化解释，才把该项功能称为 phase-like emergence；不要求其他功能同时变化。</p>
        <div class="callout warning"><strong>当前状态：实验设计与可执行 runner 已生成，但这些控制 run 尚未训练。</strong>因此本报告仍把 v20 的 abrupt change 称为 transition candidate，不把 baseline 的单次曲线当作 objective switch 的因果证据。运行入口为 <code>scripts/run_v20_phase_ablation_suite.py</code>；支持 <code>--dry-run</code> 审计命令、<code>--only</code> 单独续跑与 <code>--skip-completed</code> 恢复。</div>
        </section>
        """

    css = """
    :root { --ink:#152a3a; --muted:#5f7282; --navy:#123f63; --teal:#16877d; --orange:#d97745; --purple:#6f4aa8; --paper:#fbfcfe; --line:#d8e1e8; --soft:#edf4f8; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:#eef2f5; font-family:Inter,"Noto Sans SC","Microsoft YaHei",system-ui,sans-serif; line-height:1.7; }
    main { max-width:1180px; margin:0 auto; background:white; min-height:100vh; box-shadow:0 0 36px rgba(20,45,65,.09); }
    header { padding:58px 64px 42px; color:white; background:linear-gradient(125deg,#123f63 0%,#185f75 58%,#16877d 100%); }
    header .eyebrow { letter-spacing:.12em; text-transform:uppercase; font-size:.78rem; opacity:.8; }
    h1 { margin:.3rem 0 .8rem; font-size:clamp(2rem,4vw,3.25rem); line-height:1.15; }
    header p { max-width:880px; margin:0; font-size:1.08rem; opacity:.93; }
    nav { position:sticky; top:0; z-index:5; display:flex; gap:18px; overflow:auto; padding:11px 48px; border-bottom:1px solid var(--line); background:rgba(255,255,255,.96); backdrop-filter:blur(12px); }
    nav a { color:var(--navy); text-decoration:none; font-size:.91rem; white-space:nowrap; }
    section { padding:42px 64px; border-bottom:1px solid #edf1f4; }
    h2 { margin:0 0 18px; color:var(--navy); font-size:1.7rem; }
    h3 { margin:28px 0 10px; color:#185f75; }
    h4 { margin:0 0 12px; color:#243b4a; font-size:1.07rem; }
    p { margin:.65rem 0; }
    .lede { font-size:1.09rem; color:#334c5c; }
    .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(195px,1fr)); gap:14px; margin:20px 0; }
    .card { padding:18px; border:1px solid var(--line); border-radius:14px; background:linear-gradient(180deg,#fff,#f8fafc); }
    .card .label { color:var(--muted); font-size:.82rem; }
    .card .value { margin-top:4px; color:var(--navy); font-size:1.42rem; font-weight:750; }
    .card .small { margin-top:3px; color:var(--muted); font-size:.82rem; }
    .callout { margin:18px 0; padding:15px 18px; border-radius:10px; border-left:5px solid var(--navy); background:var(--soft); }
    .callout.good { border-left-color:var(--teal); background:#edf8f5; }
    .callout.warning { border-left-color:#c77b25; background:#fff6e8; }
    .callout.neutral { border-left-color:var(--purple); background:#f6f1fb; }
    .definition { margin:14px 0 20px; padding:15px 18px; border:1px solid #cfe0e9; border-radius:11px; background:#f7fbfd; }
    .definition code { color:#66428f; }
    .report-figure { margin:25px 0 32px; padding:18px; border:1px solid var(--line); border-radius:16px; background:var(--paper); }
    .report-figure img { display:block; width:100%; height:auto; border-radius:8px; }
    figcaption { margin-top:14px; color:#405765; font-size:.94rem; }
    .figure-tag { color:var(--navy); font-weight:750; }
    .table-wrap { overflow:auto; margin:15px 0 22px; border:1px solid var(--line); border-radius:12px; }
    table { width:100%; border-collapse:collapse; font-size:.91rem; }
    th,td { padding:10px 12px; border-bottom:1px solid #e7edf1; text-align:left; vertical-align:top; }
    th { position:sticky; top:0; color:white; background:#185f75; }
    tr:nth-child(even) td { background:#f7fafc; }
    .circuit { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; align-items:stretch; margin:22px 0; }
    .node { padding:15px 12px; border:1px solid #b9d3df; border-radius:12px; background:#f4fafc; text-align:center; }
    .node strong { display:block; color:var(--navy); }
    .arrow { color:var(--teal); font-weight:800; display:none; }
    iframe { width:100%; height:720px; border:1px solid var(--line); border-radius:12px; background:white; }
    iframe.attention-dynamics { height:1320px; }
    details { margin:14px 0; padding:12px 14px; border:1px solid var(--line); border-radius:10px; background:#fbfcfd; }
    summary { cursor:pointer; color:var(--navy); font-weight:700; }
    code { padding:.1em .35em; border-radius:5px; background:#eef2f5; }
    footer { padding:30px 64px 48px; color:var(--muted); background:#f7f9fb; }
    @media (max-width:760px) { header,section,footer { padding-left:24px; padding-right:24px; } nav { padding-left:20px; } .circuit { grid-template-columns:1fr; } iframe { height:560px; } iframe.attention-dynamics { height:1840px; } }
    @media print { nav { display:none; } body { background:white; } main { box-shadow:none; } section { break-inside:avoid; } }
    """

    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{version.upper()} counting mechanism report</title><style>{css}</style></head>
<body><main>
<header><div class="eyebrow">Synthetic NIAH counting · mechanism audit</div>
<h1>{version.upper()}：计数机制与学习动力学</h1>
<p>RoPE · query-first · true count 1–30 · seed 1234 · {html.escape(tokenization)}。报告把自由生成行为、attention role、hidden-state manifold 与位置局部因果干预分开陈述，避免把 teacher-forced 高准确率或单一相关指标误写成算法证据。</p></header>
<nav><a href="#answer">结论</a><a href="#results">结果</a>{('<a href="#nonthinking">Nonthinking 诊断</a>' if version=='v20' else '')}<a href="#setup">设定</a><a href="#integrity">完整性</a><a href="#behavior">学习曲线</a><a href="#mechanism">机制</a><a href="#dynamics">形成过程</a><a href="#geometry">Representation</a><a href="#causal">因果</a>{('<a href="#controls">控制实验</a>' if version=='v20' else '')}<a href="#limits">边界</a></nav>

<section id="answer"><h2>1. 先给答案：模型具体学到了什么？</h2>
<p class="lede">当前最符合全部证据的解释不是“一个 head 独立完成计数”，而是一条由 trace 显式提供状态槽位的循环电路：<strong>marker-successor 路由维持 k→k+1 / stop 的控制流，targeted-retrieval 路由按 k 回到 prompt 取出第 k 个 needle，hidden state 保存进度与被检索内容，循环末端再读出总数。</strong></p>
<div class="cards">
 <div class="card"><div class="label">结果完整性</div><div class="value">{completeness}</div><div class="small">见第 4 节逐项审计</div></div>
 <div class="card"><div class="label">最终 Thinking AR</div><div class="value">{pct(high_power_ar.loc['thinking','ar_final_accuracy'])}</div><div class="small">N={fmt(high_power_ar.loc['thinking','examples'],0)}；{fmt(high_power_ar.loc['thinking','examples_per_count'],0)}/count</div></div>
 <div class="card"><div class="label">最终 Nonthinking AR</div><div class="value">{pct(high_power_ar.loc['nonthinking','ar_final_accuracy'])}</div><div class="small">N={fmt(high_power_ar.loc['nonthinking','examples'],0)}；{fmt(high_power_ar.loc['nonthinking','examples_per_count'],0)}/count</div></div>
 <div class="card"><div class="label">Targeted head mass</div><div class="value">{fmt(targeted['score'])}</div><div class="small">L{int(targeted['layer'])}H{int(targeted['head'])}</div></div>
 <div class="card"><div class="label">Successor head mass</div><div class="value">{fmt(successor['score'])}</div><div class="small">L{int(successor['layer'])}H{int(successor['head'])}</div></div>
</div>
<h3>训练总览：loss 与 accuracy</h3>
<div class="definition"><strong>三个量不能混为一谈。</strong>Final-count loss 是 held-out validation task 上正确 count token 的平均负对数概率，即 CE=−E[log p(n|gold prefix)]，越低越好；Thinking 的 gold prefix 包含正确 trace，因此该 loss 测的是“给定正确 trace 后的答案读出”。Teacher-forced final accuracy 同样使用 gold prefix。Autoregressive final accuracy 则从 <code>&lt;Think&gt;</code> 或 <code>&lt;Ans&gt;</code> 后自由生成，才测完整 counting 闭环。</div>
{figure(overview_path,'1','Thinking 与 Nonthinking 的 loss / accuracy 总览','左：held-out final-count token cross-entropy，横轴为 optimizer training step，纵轴为平均 CE 并使用对数刻度；越低表示正确 count token 的条件概率越高。中：teacher-forced final exact accuracy，横轴为 step，纵轴为正确比例；每点为 30 个 count 各 10 条 validation prompts 的等权平均。右：autoregressive final exact accuracy，圆点折线为周期评估（每 count 2 条、共 60 条），星号及误差棒为 step 10,000 的独立 test-task 评估（Thinking/Nonthinking 均为每 count 50 条、共 1,500 条；误差棒为 95% Wilson interval）。三个 panel 的竖虚线均为 step 1,500 objective switch。Thinking 的 TF/loss 使用 gold trace，不能当作自由生成准确率；完整行为应看右图。')}
<h3>Count 误差分布如何收缩</h3>
<div class="definition"><strong>Signed count error。</strong>对每条自由生成样本定义 <em>e</em>=预测 count−真实 count；<em>e</em>=0 才是精确计数，<em>e</em>&lt;0 表示少数，<em>e</em>&gt;0 表示多数。图中每个热图列固定包含该 true-count band 的 10 条周期 AR prompts（5 个 count×2 条/count），颜色表示落在某个整数误差上的样本比例。青线是该列的 mean signed error；它只衡量偏差方向，均值接近 0 仍可能由正负错误抵消，因此必须和 <em>e</em>=0 行的集中程度一起看。</div>
{figure(signed_error_path,'2','自由生成 count 误差分布随训练的变化','两列分别为 Nonthinking 与 Thinking，六行分别是真实 count 1–5、6–10、11–15、16–20、21–25、26–30。横轴是 optimizer training step；纵轴是 signed count error=prediction−truth，范围 −30…30。每个彩色格表示该 checkpoint、该 count band 的 10 条周期 AR prompts 中具有该整数误差的比例，所有 panel 共用平方根颜色刻度，使低频错误也可见。金色横虚线是 e=0（exact count）；青线是 mean signed error；灰色竖虚线是 step 1,500 objective switch；step 10,000 的浅色星号是独立 final test 上每 count 50 条、每 band 共 250 条的平均误差。Nonthinking 的均值虽然较早靠近 0，但误差质量长期分散在零线两侧；Thinking 约从 step 6,000 起明显集中到零误差。')}
<div class="circuit"><div class="node"><strong>index k</strong>当前 trace 状态锚点</div><div class="node"><strong>Targeted retrieval</strong>指向 prompt 中第 k 个 needle</div><div class="node"><strong>marker k</strong>写入被检索字符</div><div class="node"><strong>Marker successor</strong>回看 index k，产生 k+1 或 close</div><div class="node"><strong>terminal readout</strong>&lt;/Think&gt; → &lt;Ans&gt; → n</div></div>
<div class="callout neutral"><strong>关键限定：</strong>“定位/提取 needle”与“推进计数状态”是两个不同角色。Attention mass 是路由描述；只有局部消融或 patching 才提供因果证据。最终 answer 的完整 readout 链路在 v20 可由 full causal port 检验，在 v21 因 digit-wise 语义 patch 尚不完全对齐。</div>
</section>

<section id="results"><h2>2. 先把实验结果叙述清楚</h2>
<div class="definition"><strong>这里的 final AR 是新增的高样本评估。</strong>它使用固定 final checkpoint 和未用于训练的 test-task prompts；Thinking 与 Nonthinking 分别为每个 true count {fmt(high_power_ar.loc['thinking','examples_per_count'],0)} 和 {fmt(high_power_ar.loc['nonthinking','examples_per_count'],0)} 个，总计 {fmt(high_power_ar.loc['thinking','examples'],0)} 和 {fmt(high_power_ar.loc['nonthinking','examples'],0)} 个。总准确率后的区间是把所有平衡 count 样本合并后计算的 95% Wilson binomial interval。周期学习曲线仍使用 validation split 的 2 examples/count；两者不能混作同一个 estimator。</div>
<p><strong>最终完整闭环：</strong>Thinking AR exact={pct(high_power_ar.loc['thinking','ar_final_accuracy'])}（95% CI {pct(high_power_ar.loc['thinking','ar_final_accuracy_wilson95_low'])}–{pct(high_power_ar.loc['thinking','ar_final_accuracy_wilson95_high'])}），Nonthinking 的完整 50/count estimate={pct(high_power_ar.loc['nonthinking','ar_final_accuracy'])}（{pct(high_power_ar.loc['nonthinking','ar_final_accuracy_wilson95_low'])}–{pct(high_power_ar.loc['nonthinking','ar_final_accuracy_wilson95_high'])}）。在与 Thinking 完全相同的 600 个 prompts 上，Nonthinking={pct(paired_final_nt)}，逐 prompt 平均差值为 {pct(paired_final_difference)}。Thinking whole-trace exact={pct(high_power_ar.loc['thinking','trace_exact'])}，ordered-marker accuracy={pct(high_power_ar.loc['thinking','trace_ordered_marker_accuracy'])}。也就是说，大多数剩余失败是“整条 trace 至少有一步错误”，而不是每个 marker 都经常错。</p>
{figure(final_ar_path,'3','Final checkpoint 的高样本 AR 准确率',f'上下 panel 分别为 Thinking 与 Nonthinking。横轴是真实总计数 n=1…30；纵轴是从 &lt;Think&gt; 或 &lt;Ans&gt; 后自由生成得到的 final count exact accuracy；点为该 count 的样本均值，误差棒是该 count 的 95% Wilson interval，水平虚线为 30 个 count 的等权平均。Thinking 每点 {fmt(high_power_ar.loc["thinking","examples_per_count"],0)} 个 test prompts；Nonthinking 每点 {fmt(high_power_ar.loc["nonthinking","examples_per_count"],0)} 个。')}
{table(['模式','count band','prompts','final AR exact','whole-trace exact','ordered-marker accuracy'], high_power_band_rows)}
<p>Thinking 的 6 个 count bands 均保持约 88%–94% final exact，没有随 n 单调崩塌；26–30 band 的 whole-trace exact 较低，但 ordered-marker accuracy 仍接近 98%，符合长 trace 对少量局部错误更敏感的误差累积。Nonthinking 只有约 26%–40%，且没有形成一个随 count 递增而平滑改善的可靠算法曲线。新增 test 估计低于原 60-prompt validation 周期点（Thinking {pct(final_ar.loc['thinking','ar_final_accuracy'])}），说明原来的 2/count final 点略乐观；但 Thinking 的优势和跨 count 稳定性仍然清楚。</p>
<div class="callout good"><strong>目前结论：</strong>高样本 final AR 证实 Thinking 的优势不是 60 个 validation prompts 的抽样偶然；模型在 1–30 全范围内都能运行同一 trace 闭环。<br><strong>欠缺证据：</strong>CI 只量化 test-example 抽样误差，不包含 seed 或 checkpoint 选择方差；仍需要多 seed。</div>
</section>

{nonthinking_diagnosis_section}

<section id="setup"><h2>3. 实验设定与表示</h2>
<div class="definition"><strong>任务。</strong>从 Tiny Shakespeare 的 256-character data window 中，计算 query 给出的 3 个目标字符的总出现次数；仅保留 true count 1–30。Query 位于 data 之前。每个训练 step 的 batch size 为 {cfg['batch_size']}，因此每种模型累计看见 {cfg['batch_size']*cfg['train_steps']:,} 个 accepted examples。</div>
{table(['项目','设定'],[
 ['序列（Nonthinking）','&lt;BOS&gt; query[5] data[256] &lt;Ans&gt; number &lt;EOS&gt;'],
 ['序列（Thinking）','&lt;BOS&gt; query[5] data[256] &lt;Think&gt; (number<sub>k</sub> marker<sub>k</sub>)×n &lt;/Think&gt; &lt;Ans&gt; number<sub>n</sub> &lt;EOS&gt;'],
 ['数字表示',f'{html.escape(tokenization)}；例如 12 = <code>{html.escape(number_example)}</code>'],
 ['模型',f"4 layers × 4 heads，d<sub>model</sub>=256，MLP=1024，{model_params:,} parameters / model，RoPE base 10,000"],
 ['优化',f"AdamW，lr={cfg['lr']}, warmup={cfg['warmup_steps']}, weight decay={cfg['weight_decay']}, {cfg['precision']}, 10,000 steps"],
 ['配对控制','Thinking/Nonthinking 使用相同 seed、相同数据样本与顺序；v20/v21 的 accepted-count 分布也相同'],
 ['checkpoint','每 100 steps 一个 model-only FP16 scientific snapshot；每 500 steps rolling recovery'],
 ['评估',f'TF: 每 count 10 examples、每 500 steps；周期 AR: 每 count 2 examples、每 1,000 steps；final test AR: Thinking {fmt(high_power_ar.loc["thinking","examples_per_count"],0)}/count、Nonthinking {fmt(high_power_ar.loc["nonthinking","examples_per_count"],0)}/count；dense phase: 每 count 1 reporting example、每 100 steps'],
])}
<p><strong>训练目标日程。</strong>{objective}</p>
<div class="definition"><strong>Digit-wise 的额外语义。</strong>v21 在 marker<sub>k</sub> 后只直接预测 k+1 的第一个 digit；报告另用 <em>semantic k→k+1 exact</em> 检查组成 k+1 的全部 digit 是否都正确。不能把“首 digit 正确”写成“整数 k+1 正确”。</div>
<div class="callout good"><strong>目前结论：</strong>两种输出 representation 的唯一设计性差异是 count number 的 tokenization；数据、位置编码、query 顺序、优化与 checkpoint 密度均匹配。<br><strong>欠缺证据：</strong>只有 seed 1234，尚不能估计 seed-to-seed 方差。</div>
</section>

<section id="integrity"><h2>4. 结果完整性审计</h2>
{table(['产物','覆盖','final','recovery','判断'], audit_rows)}
<details><summary>审计器记录的缺口/说明</summary><ul>{issue_list}</ul></details>
{port_section}
<div class="callout warning"><strong>抽样边界必须写进解释：</strong>周期 TF 每个 count 有 10 个样本；周期 AR 每个 count 只有 2 个样本，因此单个 count 的 0/0.5/1.0 很离散。Dense phase 每个 count 每 checkpoint 只有 1 个 reporting example；高 k 的 trace 指标还只有少量 n≥k 的位置，适合定位候选转折，不适合单独给显著性结论。</div>
</section>

<section id="behavior"><h2>5. 学习曲线：自由生成与 teacher forcing 必须分开</h2>
<div class="definition"><strong>Teacher-forced exact。</strong>在 gold prefix 已给定时预测指定 token/数字是否完全正确；它测局部条件映射。<strong>Autoregressive (AR) exact。</strong>模型从 &lt;Think&gt; 或 &lt;Ans&gt; 后自行生成，前一步错误会进入后续上下文；它测完整闭环。纵轴 accuracy 均为正确样本比例，范围 0–1。</div>
{figure(timeline_path,'4','从局部能力到闭环计数的时间线','左上：周期 AR exact；右上：dense teacher-forced 的 occurrence-weighted accuracy；左下：固定 head 的目标 attention mass；右下：仅在角色 query 位置把一个 head slice 置零所造成的正确-token logit margin 损失。四个横轴均为 optimizer training step；虚线 step 1,500 是训练目标从 all-sequence 切到 task-output 的边界。')}
<p>最终 held-out TF final exact：Thinking={pct(tf.loc['thinking','tf_final_accuracy'])}，Nonthinking={pct(tf.loc['nonthinking','tf_final_accuracy'])}；最终 AR final exact：Thinking={pct(final_ar.loc['thinking','ar_final_accuracy'])}，Nonthinking={pct(final_ar.loc['nonthinking','ar_final_accuracy'])}。Thinking 的最终 AR trace exact={pct(final_ar.loc['thinking','trace_exact'])}，ordered-marker accuracy={pct(final_ar.loc['thinking','trace_ordered_marker_accuracy'])}。</p>
{paired_behavior_note}
{figure(ar_band_path,'5','不同 count 区间的自由生成学习','三个 panel 分别画 Thinking final exact、Thinking whole-trace exact 和 Nonthinking final exact。横轴 training step；纵轴为 AR exact accuracy。每条线按 5 个相邻 true counts 聚合，每个 checkpoint 每条线共 10 prompts，因此只用于训练动力学，不替代图 3 的高样本 final estimate。')}
{figure(run_dir/'figures/dense_phase_behavior_by_count.png','6','100-step 分辨率的 teacher-forced 行为热图','横轴 training step，纵轴 true total count n，颜色为该格 teacher-forced accuracy（黑=0，浅绿=1）。四个 panel 依次是 final answer exact、retrieved marker、marker 后下一 token、完整语义 k→k+1。该图用于定位候选转折；它不等同于自由生成成功。')}
<div class="callout good"><strong>目前结论：</strong>Thinking 的优势来自可执行 trace，而不是单纯多输出 token。局部 successor/answer 映射很早就能在 teacher forcing 下学会，但完整 AR 链路要晚得多。<br><strong>欠缺证据：</strong>周期 AR 仍只有 2/count，因此 1,000-step learning curve 的转折时间精度低于 50/count 的 final test；需要在候选转折窗口额外做高样本 AR，而不是把 final estimate 反推到所有 checkpoints。</div>
</section>

<section id="mechanism"><h2>6. 最终 checkpoint 的 counting mechanism</h2>
<h3>6.1 Marker-successor：推进 k→k+1 / close 的控制流</h3>
<div class="definition"><strong>定义。</strong>在 marker<sub>k</sub> 位置，以该 head 指向紧邻之前、拼写语义整数 k 的全部 index tokens 的注意力质量为 S<sub>h</sub>=E[Σ<sub>j∈tokens(k)</sub>A<sub>h</sub>(marker<sub>k</sub>,j)]。v20 的 tokens(k) 长度恒为 1；v21 对 k≥10 长度为 2。S<sub>h</sub> 越高，只说明“回看当前 index”的路由越集中。</div>
<p>最终固定 successor head 为 L{head_roles['marker_successor']['layer']}H{head_roles['marker_successor']['head']}，reporting split attention mass={fmt(successor['score'])}。只在 marker query 位置将该 head slice 置零，下一 index/close accuracy 改变 {pct(successor_fixed['accuracy_change_from_baseline'])}，正确-token margin 改变 {fmt(successor_fixed['margin_change_from_baseline'])}；同层 control 分别为 {pct(successor_control['accuracy_change_from_baseline'])} 与 {fmt(successor_control['margin_change_from_baseline'])}。</p>
<h3>6.2 Targeted retrieval：定位并提取第 k 个 needle</h3>
<div class="definition"><strong>定义。</strong>在 trace index k 的 anchor（v21 为整数拼写的最后一个 digit）位置，测量该 head 对 prompt 中第 k 个目标字符 occurrence 的注意力：T<sub>h</sub>=E[A<sub>h</sub>(index<sub>k</sub>, needle<sub>k</sub>)]。固定 head 只在独立 head-selection split 的最终 checkpoint 上选择；曲线和消融使用不重叠 reporting split。</div>
<p>最终固定 targeted head 为 L{head_roles['targeted_retrieval']['layer']}H{head_roles['targeted_retrieval']['head']}，reporting mass={fmt(targeted['score'])}。局部置零后 marker-token accuracy 改变 {pct(targeted_fixed['accuracy_change_from_baseline'])}，margin 改变 {fmt(targeted_fixed['margin_change_from_baseline'])}；同层 score-matched control 为 {pct(targeted_control['accuracy_change_from_baseline'])} 与 {fmt(targeted_control['margin_change_from_baseline'])}。</p>
{figure(head_k_path,'7','两个角色在最终 checkpoint 上按 k 分解','横轴为语义进度 k；实线纵轴为固定 head 的 attention role score，虚线为相应 teacher-forced token accuracy。竖虚线位于 9/10 边界，用于标出 digit-wise 表示开始使用两位数的位置；v20 也保留该参考线。高 k 的 observations 更少，尤其 k=30 只来自 n=30。')}
{representation_failure_section}
<div class="callout good"><strong>目前结论：</strong>successor head 的因果特异性很强，是当前最坚实的“状态推进”证据；targeted head 对正确 marker 也有因果影响，但 {('v20 的同层第二个高分 retrieval head 同样重要，说明检索可能分布在多个 L4 heads，而非由单一 top head 垄断。' if version=='v20' else 'v21 的 top targeted head 相比同层 control 呈现更清楚的选择性依赖。')}<br><strong>欠缺证据：</strong>attention mass 本身不证明 value 中传输了 needle identity；还需要 identity-preserving corruption/patching 或 value-vector intervention。</div>
</section>

<section id="dynamics"><h2>7. 这些机制如何在训练中形成？</h2>
{table(['里程碑','首次 step','解释'], transition_rows)}
<p>不同功能有不同的形成速度；不能用同一个“突然/平缓”标签概括整套 counting circuit：</p>
<ol>
 <li><strong>约 200–300 step：控制流 scaffold。</strong>successor mass 快速上升，teacher-forced semantic k→k+1 接近饱和；模型先学会“给定正确 k 和 marker 后该输出什么”。</li>
 <li><strong>约 300–1,500 step：条件 readout。</strong>Thinking 的 TF final answer 很早变高，但 AR 仍低，说明正确前缀掩盖了上游 retrieval 错误。</li>
 <li><strong>约 2,000–6,000 step：定向检索形成。</strong>targeted head 从接近零持续增长，局部消融的 margin effect 同步扩大，trace marker 与 AR 开始稳定改善。</li>
 <li><strong>约 6,000–10,000 step：闭环可靠性。</strong>上游每步 marker error 降低后，长 trace 的乘法式误差累积减弱，AR trace/final 才接近最终水平。</li>
</ol>
{attention_dynamics_section}
{phase_audit_section}
{figure(run_dir/'figures/training_token_exposure_by_k.png',('13' if version=='v20' else '8'),'每个 k 的训练 token exposure','左上：Thinking 的 trace-index token 累计 exposure 热图（横轴 step，纵轴语义 k，颜色为累计 token occurrences）；右上：step 10,000 时 index、continue 与 close target exposure 随 k 的变化；下排对应 Nonthinking final-answer example/digit exposure。v21 在 k≥10 处 index-token exposure 跳变，是两位数由两个 digit 拼写导致，并不表示语义样本翻倍。')}
<div class="definition"><strong>Exposure 计算。</strong>令 N<sub>n</sub>(t) 为 step t 前 accepted true-count=n 的训练样本数。Thinking 中语义 index k 的出现次数 E<sub>k</sub><sup>semantic</sup>(t)=Σ<sub>n≥k</sub>N<sub>n</sub>(t)；index-token exposure=d(k)E<sub>k</sub><sup>semantic</sup>，其中 v20 的 d(k)=1，v21 的 d(k) 为十进制 digit 数。Continue exposure=Σ<sub>n&gt;k</sub>N<sub>n</sub>，close exposure=N<sub>k</sub>。</div>
<div class="callout neutral"><strong>逐功能判断。</strong>Successor routing 在 200–300 step 的窄窗口快速形成；targeted routing 则在约 2,000–7,000 step 连续专门化。对任一功能，attention role 的快速改变只证明路由模式形成；若要称为具有因果意义的 phase transition，还需在相近窗口对该功能加密 causal intervention，并验证变化后保持稳定。step 1,500 objective switch 与 targeted retrieval 的后续增长相邻，但单次训练不足以证明 switch 是原因。<br><strong>欠缺证据：</strong>需要每个候选功能各自的多 seed transition-width 分布、objective-switch timing ablation，以及按相等 semantic exposure 对齐的训练。</div>
</section>

<section id="geometry"><h2>8. Hidden-state representation：看 manifold，而不是只看 R²</h2>
<div class="definition"><strong>几何量定义。</strong>对某个 site/layer，先按语义 k 求 centroid c<sub>k</sub>。Adjacent distance=mean‖c<sub>k+1</sub>−c<sub>k</sub>‖；adjacent cosine 是相邻位移向量的平均 cosine；straightness=‖c<sub>K</sub>−c<sub>1</sub>‖/Σ‖c<sub>k+1</sub>−c<sub>k</sub>‖；effective dimension=1/Σr<sub>i</sub><sup>2</sup>，r<sub>i</sub> 为 centroid-PCA 方差比例；adjacent-between/within=相邻 centroid 均方距离÷同 k hidden-state 均方散度。</div>
{figure(run_dir/'figures/dense_marker_manifold_emergence.png',('14' if version=='v20' else '9'),'Trace-marker manifold 的训练中重组','三个 panel 横轴均为 training step；左纵轴为 adjacent-between/within separation，中纵轴为 mean adjacent displacement cosine，右纵轴为 centroid effective dimension。曲线 layer 0 表示 embedding output，layer 1–4 表示对应 Transformer layer output。')}
<p>最终 Layer 4 trace-marker manifold：effective dimension={fmt(layer4['centroid_effective_dimension'])}，前三 PC 方差占比={pct(layer4['centroid_pc1_to_pc3_variance_fraction'])}，adjacent cosine={fmt(layer4['mean_adjacent_step_cosine'])}，straightness={fmt(layer4['path_straightness_chord_over_arc'])}，adjacent-between/within={fmt(layer4['adjacent_between_over_within'])}。</p>
{geometry_interpretation}
<h3>交互 3D centroid / sample cloud</h3>
<p>下方控件可选 mode、semantic site、checkpoint 和 layer。三轴 PC1/PC2/PC3 是针对每个 checkpoint×site×layer 单独拟合的 centroid-centered PCA 坐标；它们没有跨 checkpoint 固定方向，因此只能看同一视图内部的形状，跨 checkpoint 应比较上面定义的无旋转标量。</p>
{manifold_embed}
<div class="callout warning"><strong>不可过度解释。</strong>最终 adjacent cosine 为负且 straightness 很低，说明轨迹不是一条单调的一维“数字线”；模型更像形成了可分离、弯折且受 token/marker 结构影响的状态流形。另由于 dense phase 的 final-answer site 每 count 只有 1 个样本，其 within-k scatter=0，final-answer 的 between/within 比值不可解释；报告只引用 trace sites 的该指标。<br><strong>欠缺证据：</strong>应以更多 examples/count 重算 within-k scatter，并用跨 checkpoint 对齐（Procrustes/CCA）检验 manifold 是否连续重组。</div>
</section>

<section id="causal"><h2>9. 因果证据的强弱</h2>
{figure(run_dir/'figures/milestone_local_head_causality.png',('15' if version=='v20' else '10'),'位置局部 head 消融随训练的变化','横轴为预先指定的 milestone checkpoint；纵轴为 intervention 后正确 token 相对 baseline 的 logit margin 变化（负值越大，损害越强）。蓝线为固定角色 head，橙线为同层 score-matched control。左 panel 测 index query→marker，右 panel 测 marker query→next index/close。')}
{causal_port_results}
<div class="callout good"><strong>证据分级：</strong>行为曲线回答“会不会”；attention role 回答“看哪里”；manifold 回答“状态如何组织”；位置局部消融回答“该组件是否被输出使用”。只有四层证据合在一起，才支持 counting circuit 的解释。<br><strong>仍未证明：</strong>单 head 的必要性不等于充分性；也未证明模型实现了一个与长度无关、可外推到 count&gt;30 的抽象递归算法。</div>
</section>

{ablation_section}

<section id="limits"><h2>{('11' if version=='v20' else '10')}. 结论、替代解释与下一步</h2>
<h3>可以得到的结论</h3><ul>
 <li>Thinking 明显优于 Nonthinking，关键不是“有更多 token”，而是 trace 提供了可迭代状态与可定位 query。</li>
 <li>L2H3 marker-successor 是两套表示中最稳定、最具因果特异性的组件；它负责推进 control state。</li>
 <li>Layer 4 targeted retrieval 在中后期形成，并与 AR 改善同步；{('v20 中存在两个接近的 L4 retrieval heads，表现为分布式/冗余检索。' if version=='v20' else 'v21 的 L4H1 targeted route 有清楚的局部因果依赖。')}</li>
 <li>计数 representation 是弯折的多维 manifold，不是一条由高 R² 定义的一维 number line。</li>
 <li>{('Atomic token 使最终 Thinking AR 达到更高水平；' if version=='v20' else 'Digit-wise 共享数字虽减少参数并增加两位数 digit exposure，仍产生更多 trace 组合错误；')}token exposure 数量本身不能解释全部差距。</li>
</ul>
<h3>仍需补足的关键实验</h3><ol>
 <li>3–5 个 seeds；对 transition step、head identity 与 causal effect 报告均值和区间。</li>
 <li>Final checkpoint 已提升到 20–50 examples/count；下一步是在候选 transition 窗口也做高样本 AR，并补 3–5 seeds。</li>
 <li>对 v21 设计 length-preserving、whole-number digit-group patch，才能与 v20 的 atomic causal port 对齐。</li>
 <li>Value-only / pattern-only / residual patch 已完成；下一步增加不同 corruption 类型与 source-position controls，检验该分工是否跨 k、跨字符集合稳定。</li>
 <li>Objective switch、count distribution 与 sequence length 的配对 runner 已生成；控制训练尚未运行，完成后才能逐功能区分 exposure、curriculum 与 phase-like emergence。</li>
 <li>训练/测试到 count&gt;30，检验 successor 与 manifold 是否支持长度外推，而非记住有限状态机。</li>
</ol>
</section>
<footer>报告由仓库中的原始 config、manifest、CSV sufficient statistics、checkpoint audit 与现有 figure 生成。生成脚本：<code>scripts/build_v20_v21_mechanism_reports.py</code>。所有 accuracy 均明确区分 teacher forcing 与 autoregressive；所有新指标均在首次出现前给出计算定义。</footer>
</main></body></html>"""

    output = run_dir / REPORT_NAME[version]
    atomic_text(output, report)
    validate_report(output, run_dir)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="*", type=Path, default=list(DEFAULT_RUNS))
    args = parser.parse_args()
    setup_style()
    for run_dir in args.run_dirs:
        output = build_report(run_dir.resolve())
        print(output)


if __name__ == "__main__":
    main()
