from __future__ import annotations

import argparse
import base64
import html
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V18 = ROOT / "colab_results" / "v18_main_all_seed1234_20260719_191912"
DEFAULT_V19 = ROOT / "colab_results" / "v19_main_all_seed1234_20260719_205527"

BANDS = ["count_1_32", "count_33_64", "count_65_96", "count_97_128"]
BAND_LABELS = {
    "count_1_32": "1-32",
    "count_33_64": "33-64",
    "count_65_96": "65-96",
    "count_97_128": "97-128",
}
DIST_ORDER = ["uniform", "power"]
MODE_ORDER = ["direct", "cot"]
COLORS = {
    "direct": "#2563eb",
    "cot": "#e76f51",
    "uniform": "#2563eb",
    "power": "#e76f51",
    "count_1_32": "#2563eb",
    "count_33_64": "#16a34a",
    "count_65_96": "#f59e0b",
    "count_97_128": "#dc2626",
    "enumeration_accuracy": "#2563eb",
    "trace_marker_accuracy": "#16a34a",
    "trace_exact_accuracy": "#7c3aed",
    "primary_accuracy": "#e76f51",
}


def load_tables(root: Path) -> dict[str, pd.DataFrame]:
    names = [
        "training_metrics",
        "dynamics_summary",
        "dynamics_by_band",
        "dynamics_detail",
        "final_summary",
        "final_by_band",
        "final_by_count",
        "final_detail",
        "attention_summary",
        "state_probe_summary",
        "state_pca_variance",
        "state_centroids_pca",
    ]
    tables: dict[str, pd.DataFrame] = {}
    for name in names:
        path = root / "tables" / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if "run_name" in frame.columns:
            if "distribution" not in frame.columns:
                frame["distribution"] = frame["run_name"].map(
                    lambda value: "power" if str(value).startswith("power_") else "uniform"
                )
            if "mode" not in frame.columns:
                frame["mode"] = frame["run_name"].map(
                    lambda value: "direct" if str(value).endswith("direct") else "cot"
                )
        tables[name] = frame
    return tables


def pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):.{digits}f}%"


def num(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def esc(value: object) -> str:
    return html.escape(str(value))


def image_data(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dbe4f0", linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)


def run_short(value: str) -> str:
    distribution = "Power" if value.startswith("power_") else "Uniform"
    mode = "CoT" if value.endswith("cot") else "Direct"
    return f"{distribution} / {mode}"


def plot_training_loss(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    frame = tables["training_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), sharey=True, constrained_layout=True)
    for axis, distribution in zip(axes, DIST_ORDER):
        subset = frame[frame.run_name.str.startswith(distribution)]
        for run_name, rows in subset.groupby("run_name"):
            mode = "cot" if run_name.endswith("cot") else "direct"
            axis.plot(rows.step, rows.loss, color=COLORS[mode], linewidth=2.2, label=mode.title())
        axis.set_title(f"{distribution.title()} count sampling")
        axis.set_xlabel("training step")
        axis.set_ylabel("completion-token cross-entropy")
        axis.set_yscale("symlog", linthresh=0.05)
        style_axis(axis)
        axis.legend(frameon=False)
    fig.suptitle("Training objective by count-sampling distribution", fontsize=15, fontweight="bold")
    return save_figure(fig, out / "learning_training_loss.png")


def plot_primary_dynamics(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    frame = tables["dynamics_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), sharey=True, constrained_layout=True)
    for axis, distribution in zip(axes, DIST_ORDER):
        subset = frame[frame.distribution == distribution]
        for mode in MODE_ORDER:
            rows = subset[subset["mode"] == mode].sort_values("step")
            axis.plot(rows.step, rows.primary_accuracy, marker="o", markersize=4, linewidth=2.2,
                      color=COLORS[mode], label=mode.title())
        axis.axhline(0.99, color="#64748b", linestyle="--", linewidth=1.1, label="99% threshold")
        axis.set_title(f"{distribution.title()} count sampling")
        axis.set_xlabel("training step")
        axis.set_ylabel("free-running exact final-count accuracy")
        axis.set_ylim(-0.03, 1.04)
        style_axis(axis)
        axis.legend(frameon=False, loc="lower right")
    fig.suptitle("Learning dynamics: exact scalar answer", fontsize=15, fontweight="bold")
    return save_figure(fig, out / "learning_primary_accuracy.png")


def plot_band_dynamics(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    frame = tables["dynamics_by_band"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True, sharey=True, constrained_layout=True)
    for row_index, distribution in enumerate(DIST_ORDER):
        for column_index, mode in enumerate(MODE_ORDER):
            axis = axes[row_index, column_index]
            subset = frame[(frame["distribution"] == distribution) & (frame["mode"] == mode)]
            for band in BANDS:
                rows = subset[subset.count_band == band].sort_values("step")
                axis.plot(rows.step, rows.primary_accuracy, marker="o", markersize=3.2,
                          linewidth=1.9, color=COLORS[band], label=BAND_LABELS[band])
            axis.axhline(0.99, color="#64748b", linestyle="--", linewidth=1)
            axis.set_title(f"{distribution.title()} / {mode.title()}")
            axis.set_xlabel("training step")
            axis.set_ylabel("exact final-count accuracy")
            axis.set_ylim(-0.03, 1.04)
            style_axis(axis)
            if row_index == 0 and column_index == 1:
                axis.legend(title="gold count", frameon=False, ncol=2, loc="lower right")
    fig.suptitle("Difficulty-resolved learning dynamics", fontsize=15, fontweight="bold")
    return save_figure(fig, out / "learning_by_count_band.png")


def plot_cot_decomposition(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    frame = tables["dynamics_summary"]
    metrics = ["primary_accuracy", "enumeration_accuracy", "trace_marker_accuracy", "trace_exact_accuracy"]
    labels = ["final count", "trace length", "marker identity", "entire trace exact"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), sharey=True, constrained_layout=True)
    for axis, distribution in zip(axes, DIST_ORDER):
        subset = frame[(frame["distribution"] == distribution) & (frame["mode"] == "cot")]
        for metric, label in zip(metrics, labels):
            rows = subset.sort_values("step")
            axis.plot(rows.step, rows[metric], marker="o", markersize=3.5, linewidth=2,
                      color=COLORS[metric], label=label)
        axis.axhline(0.1, color="#94a3b8", linestyle=":", linewidth=1.2, label="10-marker chance")
        axis.set_title(f"{distribution.title()} / CoT")
        axis.set_xlabel("training step")
        axis.set_ylabel("free-running accuracy")
        axis.set_ylim(-0.03, 1.04)
        style_axis(axis)
        axis.legend(frameon=False, loc="center right", fontsize=8.5)
    fig.suptitle("What CoT learned: length, identity, and final answer", fontsize=15, fontweight="bold")
    return save_figure(fig, out / "learning_cot_decomposition.png")


def plot_cot_validity_dynamics(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    detail = tables["dynamics_detail"]
    detail = detail[detail["mode"] == "cot"].copy()
    rows = []
    for (distribution, step), group in detail.groupby(["distribution", "step"]):
        valid = group.primary_abs_error.notna()
        rows.append(
            {
                "distribution": distribution,
                "step": int(step),
                "exact": float(group.primary_accuracy.mean()),
                "parsable": float(valid.mean()),
                "exact_given_parsable": float(group.loc[valid, "primary_accuracy"].mean()) if valid.any() else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), sharey=True, constrained_layout=True)
    specs = [
        ("exact overall", "exact", "#e76f51"),
        ("parsable scalar", "parsable", "#2563eb"),
        ("exact | parsable", "exact_given_parsable", "#16a34a"),
    ]
    for axis, distribution in zip(axes, DIST_ORDER):
        subset = frame[frame.distribution == distribution].sort_values("step")
        for label, metric, color in specs:
            axis.plot(subset.step, subset[metric], marker="o", markersize=4, linewidth=2.1, color=color, label=label)
        axis.set_title(f"{distribution.title()} / CoT")
        axis.set_xlabel("training step")
        axis.set_ylabel("free-running rate")
        axis.set_ylim(-0.03, 1.04)
        style_axis(axis)
        axis.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.suptitle("CoT scalar-answer accuracy separated into grammar validity and value", fontsize=15, fontweight="bold")
    return save_figure(fig, out / "learning_cot_scalar_validity.png")


def plot_representation_dynamics(
    v18_tables: dict[str, pd.DataFrame],
    v19_tables: dict[str, pd.DataFrame],
    out: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), sharex=True, sharey=True, constrained_layout=True)
    specs = [
        ("v18 atomic classes", v18_tables["dynamics_summary"], "#2563eb", "o"),
        ("v19 shared digits", v19_tables["dynamics_summary"], "#e76f51", "s"),
    ]
    for row, distribution in enumerate(DIST_ORDER):
        for column, mode in enumerate(MODE_ORDER):
            axis = axes[row, column]
            for label, frame, color, marker in specs:
                rows = frame[(frame["distribution"] == distribution) & (frame["mode"] == mode)].sort_values("step")
                axis.plot(
                    rows.step,
                    rows.primary_accuracy,
                    color=color,
                    marker=marker,
                    markersize=4,
                    linewidth=2.2,
                    label=label,
                )
            axis.axhline(0.99, color="#64748b", linestyle="--", linewidth=1)
            axis.set_title(f"{distribution.title()} / {'CoT' if mode == 'cot' else 'Direct'}")
            axis.set_xlabel("training step")
            axis.set_ylabel("free-running exact final-count accuracy")
            axis.set_ylim(-0.03, 1.04)
            style_axis(axis)
            if row == 0 and column == 0:
                axis.legend(frameon=False, loc="lower right")
    fig.suptitle("Paired learning dynamics: atomic classes versus shared decimal digits", fontsize=15, fontweight="bold")
    return save_figure(fig, out / "representation_learning_dynamics.png")


def plot_representation_delta(
    v18_tables: dict[str, pd.DataFrame],
    v19_tables: dict[str, pd.DataFrame],
    out: Path,
) -> Path:
    atomic = v18_tables["final_by_band"]
    digits = v19_tables["final_by_band"]
    keys = ["distribution", "mode", "count_band"]
    paired = atomic.merge(digits, on=keys, suffixes=("_atomic", "_digit"))
    paired["delta_pp"] = 100 * (paired.primary_accuracy_digit - paired.primary_accuracy_atomic)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True, constrained_layout=True)
    width = 0.36
    positions = np.arange(len(BANDS))
    for axis, distribution in zip(axes, DIST_ORDER):
        subset = paired[paired.distribution == distribution]
        for offset, mode in [(-width / 2, "direct"), (width / 2, "cot")]:
            rows = subset[subset["mode"] == mode].set_index("count_band").reindex(BANDS)
            axis.bar(
                positions + offset,
                rows.delta_pp,
                width=width,
                color=COLORS[mode],
                label="CoT" if mode == "cot" else "Direct",
            )
        axis.axhline(0, color="#334155", linewidth=1.1)
        axis.set_title(distribution.title())
        axis.set_xticks(positions, [BAND_LABELS[band] for band in BANDS])
        axis.set_xlabel("gold needle-count band")
        axis.set_ylabel("digit minus atomic accuracy (percentage points)")
        style_axis(axis)
        axis.legend(frameon=False)
    fig.suptitle("Final representation effect by count difficulty", fontsize=15, fontweight="bold")
    return save_figure(fig, out / "representation_final_delta.png")


def plot_final_count_bands(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    frame = tables["final_by_band"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5), sharey=True, constrained_layout=True)
    width = 0.36
    positions = np.arange(len(BANDS))
    for axis, distribution in zip(axes, DIST_ORDER):
        subset = frame[frame.distribution == distribution]
        for offset, mode in [(-width / 2, "direct"), (width / 2, "cot")]:
            rows = subset[subset["mode"] == mode].set_index("count_band").reindex(BANDS)
            axis.bar(positions + offset, rows.primary_accuracy, width=width, color=COLORS[mode],
                     label=mode.title())
        axis.set_title(distribution.title())
        axis.set_xticks(positions, [BAND_LABELS[band] for band in BANDS])
        axis.set_xlabel("gold needle-count band")
        axis.set_ylabel("free-running exact final-count accuracy")
        axis.set_ylim(0, 1.06)
        style_axis(axis)
        axis.legend(frameon=False)
    fig.suptitle("Final checkpoint: scalar counting behavior", fontsize=15, fontweight="bold")
    return save_figure(fig, out / "final_accuracy_by_band.png")


def plot_final_trace_bands(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    frame = tables["final_by_band"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5), sharey=True, constrained_layout=True)
    metrics = ["enumeration_accuracy", "trace_marker_accuracy", "trace_exact_accuracy"]
    labels = ["trace length", "marker identity", "entire trace exact"]
    width = 0.24
    positions = np.arange(len(BANDS))
    for axis, distribution in zip(axes, DIST_ORDER):
        subset = frame[(frame["distribution"] == distribution) & (frame["mode"] == "cot")].set_index("count_band").reindex(BANDS)
        for index, (metric, label) in enumerate(zip(metrics, labels)):
            offset = (index - 1) * width
            axis.bar(positions + offset, subset[metric], width=width, color=COLORS[metric], label=label)
        axis.axhline(0.1, color="#64748b", linestyle=":", linewidth=1.1)
        axis.set_title(distribution.title())
        axis.set_xticks(positions, [BAND_LABELS[band] for band in BANDS])
        axis.set_xlabel("gold needle-count band")
        axis.set_ylabel("free-running trace metric")
        axis.set_ylim(0, 1.06)
        style_axis(axis)
        axis.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Final checkpoint: CoT trace quality", fontsize=15, fontweight="bold")
    return save_figure(fig, out / "final_trace_by_band.png")


def head_matrix(frame: pd.DataFrame, metric: str) -> np.ndarray:
    pivot = frame.pivot_table(index="layer", columns="head", values=metric, aggfunc="mean")
    return pivot.reindex(index=[1, 2, 3, 4], columns=[0, 1, 2, 3]).to_numpy(dtype=float)


def draw_heatmap(axis: plt.Axes, values: np.ndarray, title: str, vmin: float, vmax: float) -> matplotlib.image.AxesImage:
    image = axis.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    axis.set_title(title, fontsize=9.4)
    axis.set_xticks(range(4), range(4))
    axis.set_yticks(range(4), range(1, 5))
    axis.set_xlabel("head (0-based)")
    axis.set_ylabel("layer")
    threshold = vmin + 0.58 * (vmax - vmin)
    for row in range(4):
        for column in range(4):
            value = values[row, column]
            text = "-" if np.isnan(value) else f"{value:.2f}"
            axis.text(column, row, text, ha="center", va="center",
                      color="black" if np.isfinite(value) and value > threshold else "white", fontsize=8)
    return image


def plot_attention_grid(
    tables: dict[str, pd.DataFrame],
    out: Path,
    *,
    metric: str,
    query_kind: str,
    mode: str,
    filename: str,
    title: str,
    fixed_scale: tuple[float, float] | None = None,
) -> Path:
    frame = tables["attention_summary"]
    frame = frame[(frame["mode"] == mode) & (frame["query_kind"] == query_kind)]
    if fixed_scale is None:
        maximum = float(frame[metric].max())
        scale = (0.0, max(maximum, 1e-4))
    else:
        scale = fixed_scale
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 7.1), constrained_layout=True)
    image = None
    for row, distribution in enumerate(DIST_ORDER):
        for column, band in enumerate(BANDS):
            subset = frame[(frame.distribution == distribution) & (frame.count_band == band)]
            image = draw_heatmap(
                axes[row, column],
                head_matrix(subset, metric),
                f"{distribution.title()} | count {BAND_LABELS[band]}",
                *scale,
            )
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.76, label=metric)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    return save_figure(fig, out / filename)


def plot_probe(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    frame = tables["state_probe_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True, constrained_layout=True)
    line_specs = [
        ("direct", "final_answer", "Direct final query", "#2563eb"),
        ("cot", "final_answer", "CoT final query", "#e76f51"),
        ("cot", "trace_index", "CoT trace index", "#7c3aed"),
        ("cot", "trace_marker", "CoT trace marker", "#16a34a"),
    ]
    for axis, distribution in zip(axes, DIST_ORDER):
        for mode, site, label, color in line_specs:
            rows = frame[(frame["distribution"] == distribution) & (frame["mode"] == mode) & (frame["site"] == site)].sort_values("layer")
            if rows.empty:
                continue
            axis.plot(rows.layer, rows.ridge_r2, marker="o", linewidth=2, color=color, label=label)
        axis.axhline(0, color="#64748b", linewidth=1)
        axis.set_xticks(range(5), ["Embed", "L1", "L2", "L3", "L4"])
        axis.set_title(distribution.title())
        axis.set_xlabel("hidden-state extraction depth")
        axis.set_ylabel("held-out ridge regression R-squared")
        axis.set_ylim(-0.15, 1.05)
        style_axis(axis)
        axis.legend(frameon=False, fontsize=8.3, loc="lower right")
    fig.suptitle("Linear readability of count/progress in residual states", fontsize=15, fontweight="bold")
    return save_figure(fig, out / "state_probe_ridge.png")


def plot_pca_coverage(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    frame = tables["state_pca_variance"]
    frame = frame[frame.component == 6]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True, constrained_layout=True)
    line_specs = [
        ("direct", "final_answer", "Direct final query", "#2563eb"),
        ("cot", "final_answer", "CoT final query", "#e76f51"),
        ("cot", "trace_index", "CoT trace index", "#7c3aed"),
        ("cot", "trace_marker", "CoT trace marker", "#16a34a"),
    ]
    for axis, distribution in zip(axes, DIST_ORDER):
        for mode, site, label, color in line_specs:
            rows = frame[(frame["distribution"] == distribution) & (frame["mode"] == mode) & (frame["site"] == site)].sort_values("layer")
            if rows.empty:
                continue
            axis.plot(rows.layer, rows.cumulative_explained_variance, marker="o", linewidth=2, color=color, label=label)
        axis.set_xticks(range(5), ["Embed", "L1", "L2", "L3", "L4"])
        axis.set_title(distribution.title())
        axis.set_xlabel("hidden-state extraction depth")
        axis.set_ylabel("cumulative variance explained by PC1-PC6")
        axis.set_ylim(0, 1.04)
        style_axis(axis)
        axis.legend(frameon=False, fontsize=8.3, loc="lower left")
    fig.suptitle("Dimensional concentration of exact-count centroids", fontsize=15, fontweight="bold")
    return save_figure(fig, out / "state_pca_coverage.png")


def plot_pc12_summary(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    frame = tables["state_centroids_pca"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), constrained_layout=True)
    panels = [
        ("uniform", "direct", "final_answer", "Uniform / Direct final query"),
        ("uniform", "cot", "trace_index", "Uniform / CoT trace index"),
        ("power", "direct", "final_answer", "Power / Direct final query"),
        ("power", "cot", "trace_index", "Power / CoT trace index"),
    ]
    for axis, (distribution, mode, site, title) in zip(axes.ravel(), panels):
        rows = frame[(frame["distribution"] == distribution) & (frame["mode"] == mode) & (frame["site"] == site) & (frame["layer"] == 4)].copy()
        rows = rows.sort_values("state_label")
        scatter = axis.scatter(rows.pc1, rows.pc2, c=rows.state_label, cmap="turbo", s=22, alpha=0.86)
        axis.plot(rows.pc1, rows.pc2, color="#94a3b8", linewidth=0.8, alpha=0.5)
        axis.set_title(title)
        axis.set_xlabel("PC1 coordinate")
        axis.set_ylabel("PC2 coordinate")
        axis.grid(color="#e2e8f0", linewidth=0.7)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.72, label="exact count / trace progress")
    fig.suptitle("Layer 4 count-centroid geometry (PC1-PC2)", fontsize=15, fontweight="bold")
    return save_figure(fig, out / "state_pc12_summary.png")


def plot_representation_pca(
    v18_tables: dict[str, pd.DataFrame],
    v19_tables: dict[str, pd.DataFrame],
    out: Path,
) -> Path:
    site_specs = [
        ("direct", "final_answer", "Direct final"),
        ("cot", "final_answer", "CoT final"),
        ("cot", "trace_index", "CoT index"),
        ("cot", "trace_marker", "CoT marker"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), sharey=True, constrained_layout=True)
    positions = np.arange(len(site_specs))
    width = 0.36
    for axis, distribution in zip(axes, DIST_ORDER):
        for offset, (label, tables, color) in zip(
            (-width / 2, width / 2),
            [
                ("v18 atomic", v18_tables, "#2563eb"),
                ("v19 digits", v19_tables, "#e76f51"),
            ],
        ):
            frame = tables["state_pca_variance"]
            values = []
            for mode, site, _ in site_specs:
                rows = frame[
                    (frame["distribution"] == distribution)
                    & (frame["mode"] == mode)
                    & (frame["site"] == site)
                    & (frame["layer"] == 4)
                    & (frame["component"] == 6)
                ]
                values.append(float(rows.cumulative_explained_variance.iloc[0]))
            axis.bar(positions + offset, values, width=width, label=label, color=color)
        axis.set_title(distribution.title())
        axis.set_xticks(positions, [label for _, _, label in site_specs], rotation=15, ha="right")
        axis.set_xlabel("semantic extraction site at Layer 4")
        axis.set_ylabel("variance explained by PC1-PC6")
        axis.set_ylim(0, 1.04)
        style_axis(axis)
        axis.legend(frameon=False, loc="lower right")
    fig.suptitle("Representation geometry: Layer-4 centroid dimensional concentration", fontsize=15, fontweight="bold")
    return save_figure(fig, out / "representation_pca_comparison.png")


def first_stable_step(group: pd.DataFrame, threshold: float, floor: float) -> int | str:
    ordered = group.sort_values("step").reset_index(drop=True)
    values = ordered.primary_accuracy.to_numpy(dtype=float)
    for index, value in enumerate(values):
        if value >= threshold and np.nanmin(values[index:]) >= floor:
            return int(ordered.loc[index, "step"])
    return "not reached"


def thresholds_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = tables["dynamics_by_band"]
    rows: list[dict[str, object]] = []
    for (run_name, band), group in frame.groupby(["run_name", "count_band"]):
        row: dict[str, object] = {"run": run_short(run_name), "count band": BAND_LABELS[band]}
        for threshold, floor in ((0.50, 0.45), (0.90, 0.85), (0.99, 0.95)):
            row[f"stable >= {int(threshold * 100)}% (floor {int(floor * 100)}%)"] = first_stable_step(
                group, threshold, floor
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["_run_order"] = result["run"].map({
        "Uniform / Direct": 0, "Uniform / CoT": 1, "Power / Direct": 2, "Power / CoT": 3,
    })
    result["_band_order"] = result["count band"].map({value: index for index, value in enumerate(BAND_LABELS.values())})
    return result.sort_values(["_run_order", "_band_order"]).drop(columns=["_run_order", "_band_order"])


def dataframe_html(frame: pd.DataFrame, *, percent_columns: Iterable[str] = ()) -> str:
    percent_columns = set(percent_columns)
    columns = list(frame.columns)
    head = "".join(f"<th>{esc(column)}</th>" for column in columns)
    body_rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column in percent_columns:
                rendered = pct(value)
            elif isinstance(value, (float, np.floating)):
                rendered = num(value)
            else:
                rendered = esc(value)
            cells.append(f"<td>{rendered}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def figure_html(path: Path, title: str, caption: str) -> str:
    return (
        '<figure class="figure-card">'
        f'<h3>{title}</h3><img src="{image_data(path)}" alt="{esc(title)}">'
        f'<figcaption>{caption}</figcaption></figure>'
    )


def top_head_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = tables["attention_summary"]
    rows: list[dict[str, object]] = []
    tasks = [
        ("Direct broad", "direct", "final_answer", "broad_attention_score"),
        ("CoT k-to-k raw mass", "cot", "trace_index", "correct_prompt_needle_mass"),
        ("CoT k-to-k top-1", "cot", "trace_index", "correct_top1"),
        ("CoT trace readout", "cot", "final_answer", "trace_markers_mass"),
    ]
    for distribution in DIST_ORDER:
        for label, mode, query, metric in tasks:
            subset = frame[(frame["distribution"] == distribution) & (frame["mode"] == mode) & (frame["query_kind"] == query)]
            values = subset.groupby(["layer", "head"])[metric].mean().sort_values(ascending=False)
            if values.empty:
                continue
            (layer, head), value = values.index[0], values.iloc[0]
            rows.append({
                "distribution": distribution,
                "candidate score": label,
                "best head": f"L{int(layer)}H{int(head)}",
                "mean score": float(value),
            })
    return pd.DataFrame(rows)


def final_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = tables["final_summary"].copy()
    detail = tables["final_detail"]
    validity_rows = []
    for run_name, group in detail.groupby("run_name"):
        valid = group.primary_abs_error.notna()
        validity_rows.append(
            {
                "run_name": run_name,
                "valid_scalar_rate": float(valid.mean()),
                "exact_given_valid": float(group.loc[valid, "primary_accuracy"].mean()) if valid.any() else np.nan,
            }
        )
    frame = frame.merge(pd.DataFrame(validity_rows), on="run_name", how="left")
    frame["run"] = frame.apply(
        lambda row: f"{str(row['distribution']).title()} / {'CoT' if row['mode'] == 'cot' else 'Direct'}",
        axis=1,
    )
    return frame[
        [
            "run",
            "primary_accuracy",
            "valid_scalar_rate",
            "exact_given_valid",
            "enumeration_accuracy",
            "trace_marker_accuracy",
            "trace_exact_accuracy",
            "primary_mae",
        ]
    ]


def representation_comparison_table(
    v18_tables: dict[str, pd.DataFrame],
    v19_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    atomic = final_table(v18_tables).set_index("run")
    digits = final_table(v19_tables).set_index("run")
    order = ["Uniform / Direct", "Uniform / CoT", "Power / Direct", "Power / CoT"]
    rows = []
    for run in order:
        atomic_value = float(atomic.loc[run, "primary_accuracy"])
        digit_value = float(digits.loc[run, "primary_accuracy"])
        rows.append(
            {
                "run": run,
                "v18 atomic": atomic_value,
                "v19 shared digits": digit_value,
                "digit - atomic": f"{100 * (digit_value - atomic_value):+.1f} pp",
            }
        )
    return pd.DataFrame(rows)


def completion_exposure_table() -> pd.DataFrame:
    counts = np.arange(1, 129)
    digit_width = np.array([len(str(value)) for value in counts])
    lengths = {
        ("v18 atomic", "Direct"): np.ones_like(counts, dtype=float),
        ("v18 atomic", "CoT"): (2 * counts + 2).astype(float),
        ("v19 shared digits", "Direct"): (digit_width + 2).astype(float),
        ("v19 shared digits", "CoT"): np.array(
            [
                sum(len(str(index)) + 2 for index in range(1, count + 1))
                + len(str(count))
                + 3
                for count in counts
            ],
            dtype=float,
        ),
    }
    rows: list[dict[str, object]] = []
    for distribution, weights in (
        ("Uniform", np.ones_like(counts, dtype=float)),
        ("Power α=1.5", counts.astype(float) ** -1.5),
    ):
        probabilities = weights / weights.sum()
        for (representation, mode), completion_length in lengths.items():
            expected_length = float(np.dot(probabilities, completion_length))
            high_band_token_share = float(
                np.dot(probabilities[96:], completion_length[96:]) / expected_length
            )
            rows.append(
                {
                    "representation": representation,
                    "mode": mode,
                    "sampling": distribution,
                    "E[supervised tokens/example]": expected_length,
                    "example share n=97-128": float(probabilities[96:].sum()),
                    "CE-token share n=97-128": high_band_token_share,
                }
            )
    return pd.DataFrame(rows)


def ktok_final_comparison_table(
    v18_tables: dict[str, pd.DataFrame],
    v19_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    counts = np.arange(1, 129)
    random_top1 = float(128 / counts.sum())
    for version, tables in (("v18 atomic", v18_tables), ("v19 digits", v19_tables)):
        attention = tables["attention_summary"]
        attention = attention[
            (attention["mode"] == "cot")
            & (attention["query_kind"] == "trace_index")
            & (attention["count_band"] == "all")
        ]
        final = tables["final_summary"]
        if version.startswith("v18"):
            denominators = [1024 + 2 * k for n in counts for k in range(1, n + 1)]
        else:
            denominators = []
            for n in counts:
                prefix = 0
                for k in range(1, n + 1):
                    denominators.append(1026 + prefix + len(str(k)))
                    prefix += len(str(k)) + 2
        uniform_position = float(np.mean(1 / np.asarray(denominators, dtype=float)))
        for distribution in DIST_ORDER:
            subset = attention[attention["distribution"] == distribution]
            best_mass = subset.loc[subset.correct_prompt_needle_mass.idxmax()]
            best_top1 = subset.loc[subset.correct_top1.idxmax()]
            marker = float(
                final[
                    (final["distribution"] == distribution)
                    & (final["mode"] == "cot")
                ].trace_marker_accuracy.iloc[0]
            )
            rows.append(
                {
                    "run": f"{version} / {distribution.title()}-CoT",
                    "final marker accuracy": marker,
                    "best exact-k raw mass": float(best_mass.correct_prompt_needle_mass),
                    "uniform-position baseline": uniform_position,
                    "raw-mass lift": float(best_mass.correct_prompt_needle_mass / uniform_position),
                    "best exact top-1": float(best_top1.correct_top1),
                    "random exact top-1": random_top1,
                }
            )
    return pd.DataFrame(rows)


def ignore_marker_floor_table(
    v18_tables: dict[str, pd.DataFrame],
    v19_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    counts = np.arange(1, 129)
    digit_lengths = np.array(
        [
            sum(len(str(k)) + 2 for k in range(1, n + 1)) + len(str(n)) + 3
            for n in counts
        ],
        dtype=float,
    )
    specs = [
        ("v18 atomic", v18_tables, (2 * counts + 2).astype(float)),
        ("v19 digits", v19_tables, digit_lengths),
    ]
    rows: list[dict[str, object]] = []
    for version, tables, lengths in specs:
        metrics = tables["training_metrics"]
        final = tables["final_summary"]
        for distribution, weights in (
            ("Uniform", np.ones_like(counts, dtype=float)),
            ("Power", counts.astype(float) ** -1.5),
        ):
            probabilities = weights / weights.sum()
            marker_fraction = float(np.dot(probabilities, counts) / np.dot(probabilities, lengths))
            run_metrics = metrics[
                (metrics["distribution"] == distribution.lower())
                & (metrics["mode"] == "cot")
            ].sort_values("step")
            run_final = final[
                (final["distribution"] == distribution.lower())
                & (final["mode"] == "cot")
            ].iloc[0]
            rows.append(
                {
                    "run": f"{version} / {distribution}-CoT",
                    "marker-token fraction": marker_fraction,
                    "ignore-marker CE floor": marker_fraction * math.log(10),
                    "logged loss at step 10000": float(run_metrics.iloc[-1].loss),
                    "final marker accuracy": float(run_final.trace_marker_accuracy),
                }
            )
    return pd.DataFrame(rows)


def high_k_exposure_table() -> pd.DataFrame:
    counts = np.arange(1, 129)
    rows = []
    for k in (1, 16, 32, 64, 96, 128):
        row: dict[str, object] = {"trace step k": k}
        for distribution, weights in (
            ("Uniform expected examples", np.ones_like(counts, dtype=float)),
            ("Power expected examples", counts.astype(float) ** -1.5),
        ):
            probabilities = weights / weights.sum()
            row[distribution] = float(320_000 * probabilities[k - 1 :].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def ktok_checkpoint_diagnostics_table() -> pd.DataFrame:
    rows = [
        ("v18 Uniform-CoT", 2000, .100, .0034, .0500, .0047),
        ("v18 Uniform-CoT", 6000, .825, .0128, .0375, .4648),
        ("v18 Uniform-CoT", 10000, .844, .0124, .0625, .3968),
        ("v19 Uniform-CoT", 2000, .125, .0018, .0312, .0040),
        ("v19 Uniform-CoT", 6000, .125, .0056, .0312, .0080),
        ("v19 Uniform-CoT", 10000, .131, .0190, .0312, .0137),
        ("v19 Power-CoT", 2000, .119, .0018, .0312, .0018),
        ("v19 Power-CoT", 6000, .062, .0040, .0437, .0227),
        ("v19 Power-CoT", 10000, .094, .0053, .0375, .0135),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "run",
            "checkpoint",
            "teacher-forced marker accuracy",
            "best exact-k mass",
            "best exact top-1",
            "best previous-marker relay mass",
        ],
    )


def ktok_key_mask_table() -> pd.DataFrame:
    rows = [
        ("clean", .825, .844, .131, .094),
        ("mask prompt needle keys", .106, .056, .019, .094),
        ("mask entire prompt", .006, .006, .000, .019),
        ("mask trace-marker keys", .475, .444, .087, .113),
        ("mask trace-number/index keys", .206, .181, .037, .081),
        ("mask complete trace prefix", .194, .194, .037, .087),
        ("mask START key", .812, .825, .131, .094),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "attention-key condition",
            "v18 Uniform step 6000",
            "v18 Uniform step 10000",
            "v19 Uniform step 10000",
            "v19 Power step 10000",
        ],
    )


def ktok_source_causal_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("replace one prompt marker token", -8.571, .017, -.010),
            ("block one prompt key position", -7.596, -.165, -.307),
        ],
        columns=[
            "intervention",
            "Δ log P(gold): exact kth occurrence",
            "Δ log P(gold): other same-ID occurrence",
            "Δ log P(gold): control occurrence",
        ],
    )


def layer4_state_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    probes = tables["state_probe_summary"]
    pca = tables["state_pca_variance"]
    probes = probes[probes.layer == 4].copy()
    pca = pca[(pca.layer == 4) & (pca.component == 6)][
        ["distribution", "mode", "site", "cumulative_explained_variance", "effective_dimension"]
    ]
    merged = probes.merge(pca, on=["distribution", "mode", "site"])
    merged["run/site"] = merged.apply(
        lambda row: f"{row['distribution'].title()} / {'CoT' if row['mode'] == 'cot' else 'Direct'} / {row['site']}",
        axis=1,
    )
    return merged[
        [
            "run/site",
            "position_only_accuracy",
            "nearest_centroid_accuracy",
            "ridge_r2",
            "ridge_mae",
            "cumulative_explained_variance",
            "effective_dimension",
        ]
    ]


def pca_widget_data(tables: dict[str, pd.DataFrame]) -> str:
    frame = tables["state_centroids_pca"].copy()
    records = []
    for row in frame.itertuples(index=False):
        records.append({
            "d": row.distribution,
            "m": row.mode,
            "s": row.site,
            "l": int(row.layer),
            "c": float(row.state_label),
            "p": [float(row.pc1), float(row.pc2), float(row.pc3), float(row.pc4), float(row.pc5), float(row.pc6)],
        })
    return json.dumps(records, separators=(",", ":"))


def peak_final_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = tables["dynamics_summary"]
    rows = []
    for run_name, group in frame.groupby("run_name"):
        group = group.sort_values("step")
        peak_index = group.primary_accuracy.idxmax()
        peak = group.loc[peak_index]
        final = group.iloc[-1]
        rows.append({
            "run": run_short(run_name),
            "peak accuracy": peak.primary_accuracy,
            "peak step": int(peak.step),
            "last dynamics accuracy": final.primary_accuracy,
            "peak-to-last change": final.primary_accuracy - peak.primary_accuracy,
        })
    return pd.DataFrame(rows).sort_values("run")


def report_text(
    *,
    version: str,
    root: Path,
    tables: dict[str, pd.DataFrame],
    other_tables: dict[str, pd.DataFrame],
    assets: dict[str, Path],
) -> str:
    is_v19 = version == "v19"
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    final = final_table(tables)
    peak = peak_final_summary(tables)
    heads = top_head_table(tables)
    thresholds = thresholds_table(tables)
    other_final = final_table(other_tables)
    v18_tables, v19_tables = (other_tables, tables) if is_v19 else (tables, other_tables)
    representation_final = representation_comparison_table(v18_tables, v19_tables)
    exposure = completion_exposure_table()
    state_l4 = layer4_state_table(tables)
    ktok_final = ktok_final_comparison_table(v18_tables, v19_tables)
    marker_floor = ignore_marker_floor_table(v18_tables, v19_tables)
    high_k_exposure = high_k_exposure_table()
    ktok_checkpoints = ktok_checkpoint_diagnostics_table()
    ktok_masks = ktok_key_mask_table()
    ktok_sources = ktok_source_causal_table()

    power_cot = final[(final["run"] == "Power / CoT")].iloc[0]
    uniform_cot = final[(final["run"] == "Uniform / CoT")].iloc[0]
    power_direct = final[(final["run"] == "Power / Direct")].iloc[0]
    uniform_direct = final[(final["run"] == "Uniform / Direct")].iloc[0]
    other_power_cot = other_final[(other_final["run"] == "Power / CoT")].iloc[0]
    other_uniform_cot = other_final[(other_final["run"] == "Uniform / CoT")].iloc[0]

    representation_name = "共享十进制 digit 表征" if is_v19 else "原子 ordinal/count 类 token 表征"
    grammar_direct = (
        "COUNT, digits(n), NUM_END" if is_v19 else "C_n"
    )
    grammar_cot = (
        "INDEX, digits(1), M_1, ..., INDEX, digits(n), M_n, END, COUNT, digits(n), NUM_END"
        if is_v19 else "I_1, M_1, ..., I_n, M_n, END, C_n"
    )
    token_note = (
        "I_k 与 C_n 是两套互不共享的 128 类 token；模型可以把 ordinal progress 与 scalar answer 分开记忆。"
        if not is_v19 else
        "ordinal index 与最终 count 共享 D0-D9；INDEX/COUNT 只标记数值角色。多位数必须由若干 digit 自回归组成。"
    )
    counterpart = "v18 原子 token" if is_v19 else "v19 digit token"
    main_interpretation = (
        f"Power-CoT 在 digit 表征下达到 {pct(power_cot.primary_accuracy)}，显著高于配对 v18 的 {pct(other_power_cot.primary_accuracy)}；"
        f"但 Uniform-CoT 从训练中峰值回落，最终仅 {pct(uniform_cot.primary_accuracy)}（配对 v18 为 {pct(other_uniform_cot.primary_accuracy)}）。"
        if is_v19 else
        f"Uniform 条件下 Direct/CoT 最终分别为 {pct(uniform_direct.primary_accuracy)} / {pct(uniform_cot.primary_accuracy)}；"
        f"Power 条件下则降至 {pct(power_direct.primary_accuracy)} / {pct(power_cot.primary_accuracy)}。配对 v19 digit CoT 在 Power 下为 {pct(other_power_cot.primary_accuracy)}。"
    )
    marker_interpretation = (
        f"Power/Uniform 的 marker identity 分别是 {pct(power_cot.trace_marker_accuracy)} / {pct(uniform_cot.trace_marker_accuracy)}，都接近十种 marker 的 10% 随机基线；"
        "因此 digit-CoT 的高 count accuracy 不能被解释成逐项复制了 marker identity。"
        if is_v19
        else
        f"Power-CoT 的 marker identity 只有 {pct(power_cot.trace_marker_accuracy)}，接近十种 marker 的 10% 随机基线；"
        f"Uniform-CoT 则达到 {pct(uniform_cot.trace_marker_accuracy)}，说明它学到了相当强的逐位置 marker 绑定，但仍没有形成高 exact-trace 成功率。"
    )
    attention_interpretation = (
        "v19 的最佳平均 k-to-k raw mass 在 Uniform/Power 下仅 0.0206/0.0051，最佳 trace-readout mass 也只有 0.136/0.260；"
        "它与 marker identity 近 chance 相互一致。Direct-Power 的 broad score 则达到 0.722，和较强 scalar counting 相容。"
        if is_v19
        else
        "v18 的最佳平均 k-to-k raw mass 在 Uniform/Power 下仅 0.0122/0.0077；但 final query 对 trace markers 的最佳 readout mass 达到 0.693/0.987。"
        "这更支持“从 trace 长度/累计状态读最终 count”，而不是一个强逐项 k-to-k copier；Uniform 条件的 80.1% marker identity 是值得进一步做 causal test 的例外。"
    )
    state_interpretation = (
        "v19 在 Layer 4 的 PC1-PC6 方差覆盖显著集中：Direct final 在 Uniform/Power 为 96.8%/95.5%，CoT trace sites 多在 97% 以上。"
        "这比 v18 的多数 final/marker sites 更低维，但共享 digit、role token 与绝对位置都可能贡献该结构，不能直接等同于更好的 counting algorithm。"
        if is_v19
        else
        "v18 Layer-4 Direct final 的 ridge R² 在 Uniform/Power 为 0.994/0.997，但 PC1-PC6 只覆盖 58.5%/66.1% 的 centroid 方差；"
        "数值线性可读，却不是简单的一条一维 number line。Power-CoT trace-index 是例外，其 PC1-PC6 覆盖 93.9%。"
    )
    validity_interpretation = (
        f"v19 Uniform-CoT 的最终 scalar 只有 {pct(uniform_cot.valid_scalar_rate)} 可被 grammar parser 成功解析；"
        f"但在可解析子集上，exact accuracy 是 {pct(uniform_cot.exact_given_valid)}。因此 45.4% overall accuracy 的主要损失不是“答案差几个”，而是长自回归 trace 未能稳定到达合法 final scalar。"
        if is_v19
        else
        f"v18 Uniform-CoT 的 scalar 可解析率为 {pct(uniform_cot.valid_scalar_rate)}，所以其错误主要是数值/trace-length 错误；"
        f"Power-CoT 仍有 {pct(power_cot.valid_scalar_rate)} 可解析，且可解析子集 exact accuracy 为 {pct(power_cot.exact_given_valid)}。"
    )
    trace_warning = (
        f"{marker_interpretation} 两种 v19 CoT 的整条 trace exact 都约 {pct(power_cot.trace_exact_accuracy)}；这与单个 marker 近 chance 及长序列联合要求一致。"
        if is_v19
        else
        f"{marker_interpretation} Uniform-v18 的整条 trace exact 仍仅 {pct(uniform_cot.trace_exact_accuracy)}：长 trace 要求长度和每个 marker 同时正确，单点 80.1% 不会转化为高全序列 exact。"
    )
    ktok_version_takeaway = (
        "对 v19 而言，核心问题不是 direct attention metric 漏掉了一个已经成功的 marker copier，而是 marker copier 基本没有被优化出来。"
        "Uniform/Power 的 marker accuracy 都约 chance，step-10000 loss 又贴近 ignore-marker floor；Power-CoT 的 98.9% final count 来自 progress/trace-length scaffold，而非 faithful retrieval。"
        if is_v19
        else
        "对 v18 必须分开解释：Uniform-CoT 的 exact kth source 具有很强因果必要性，但信息通过 prompt 与 trace 的多跳 relay 传播，所以单个 direct edge 很弱；"
        "Power-CoT 则和 v19 相似，loss 停在 ignore-marker floor，marker retrieval 本身没有学会。"
    )

    css = r"""
    :root{--ink:#0f172a;--muted:#526177;--line:#d9e2ee;--panel:#f8fafc;--blue:#2563eb;--green:#159947;--orange:#e76f51;--warn:#b45309}
    *{box-sizing:border-box} body{margin:0;background:#edf2f7;color:var(--ink);font-family:Inter,"Noto Sans SC","Microsoft YaHei",Arial,sans-serif;line-height:1.72}
    main{width:min(1120px,calc(100% - 32px));margin:26px auto;background:white;padding:42px 52px 70px;border:1px solid #dbe4ef;box-shadow:0 10px 32px rgba(15,23,42,.08)}
    h1{font-size:2.15rem;line-height:1.2;margin:.2rem 0 .4rem;letter-spacing:-.02em} h2{font-size:1.55rem;margin:2.7rem 0 1rem;padding-top:1.1rem;border-top:1px solid var(--line)} h3{font-size:1.06rem;margin:.2rem 0 .75rem}
    p{margin:.7rem 0}.subtitle{color:var(--muted);font-size:1.04rem}.kicker{color:var(--blue);font-weight:800;letter-spacing:.08em;text-transform:uppercase;font-size:.82rem}
    .thesis{margin:1.6rem 0;padding:20px 22px;border-left:5px solid var(--green);background:#effcf4;border-radius:6px}.warning{border-left-color:#f59e0b;background:#fff8e8}.info{border-left-color:var(--blue);background:#eff6ff}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:1.4rem 0}.card{border:1px solid var(--line);background:var(--panel);padding:15px;border-radius:8px}.card b{display:block;font-size:1.4rem;color:var(--blue)}.card span{font-size:.84rem;color:var(--muted)}
    .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}.mini{border:1px solid var(--line);border-radius:8px;padding:17px;background:var(--panel)}
    code{background:#eef3f9;border-radius:4px;padding:.12rem .36rem;font-family:"Cascadia Mono",Consolas,monospace;font-size:.9em}.formula{font-family:Cambria,"Times New Roman",serif;font-size:1.13rem;background:#fbfdff;border:1px solid var(--line);padding:12px 16px;margin:9px 0;text-align:center;border-radius:6px;overflow-x:auto}
    .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:7px;margin:1rem 0 1.4rem}table{border-collapse:collapse;width:100%;font-size:.88rem}th{background:#eaf0f7;text-align:left;padding:10px;border-bottom:1px solid var(--line);white-space:nowrap}td{padding:9px 10px;border-bottom:1px solid #e8edf4;vertical-align:top}tr:last-child td{border-bottom:0}
    .figure-card{margin:1.5rem 0 2rem;border:1px solid var(--line);border-radius:9px;padding:18px;background:white}.figure-card img{display:block;width:100%;height:auto;max-height:760px;object-fit:contain;margin:6px auto}.figure-card figcaption{color:var(--muted);font-size:.9rem;border-top:1px solid #e7edf4;padding-top:11px;margin-top:10px}
    .math-list li{margin:.55rem 0}.toc{columns:2;padding:14px 22px;background:var(--panel);border:1px solid var(--line);border-radius:8px}.toc a{color:#174ea6;text-decoration:none}
    .widget{border:1px solid var(--line);border-radius:9px;padding:16px;background:#fbfdff}.controls{display:flex;flex-wrap:wrap;gap:10px 14px;margin-bottom:10px}.controls label{font-size:.8rem;color:var(--muted);font-weight:700}.controls select,.controls input{display:block;margin-top:3px;padding:6px 8px;border:1px solid #b9c7d8;border-radius:5px;background:white}.widget canvas{width:100%;height:500px;background:white;border:1px solid #dce5ef;border-radius:6px}.widget-note{font-size:.86rem;color:var(--muted)}
    .footer{margin-top:3rem;color:var(--muted);font-size:.84rem;border-top:1px solid var(--line);padding-top:1rem}
    @media(max-width:820px){main{width:100%;margin:0;padding:25px 18px}.cards{grid-template-columns:1fr 1fr}.two-col{grid-template-columns:1fr}.toc{columns:1}.widget canvas{height:390px}}
    """

    widget_data = pca_widget_data(tables)
    widget = f"""
    <div class="widget">
      <div class="controls">
        <label>Distribution<select id="pca-dist"><option value="uniform">uniform</option><option value="power">power α=1.5</option></select></label>
        <label>Mode<select id="pca-mode"><option value="direct">direct</option><option value="cot">CoT</option></select></label>
        <label>Semantic site<select id="pca-site"></select></label>
        <label>Depth<select id="pca-layer"><option value="0">embedding</option><option value="1">Layer 1</option><option value="2">Layer 2</option><option value="3">Layer 3</option><option value="4" selected>Layer 4</option></select></label>
        <label>X axis<select id="pca-x"></select></label><label>Y axis<select id="pca-y"></select></label><label>Z axis<select id="pca-z"></select></label>
        <label>Yaw<input id="pca-yaw" type="range" min="-180" max="180" value="-35"></label>
        <label>Pitch<input id="pca-pitch" type="range" min="-80" max="80" value="20"></label>
      </div>
      <canvas id="pca-canvas" width="1040" height="500"></canvas>
      <div id="pca-status" class="widget-note"></div>
    </div>
    <script>
    (()=>{{
      const DATA={widget_data};
      const $=id=>document.getElementById(id), canvas=$('pca-canvas'), ctx=canvas.getContext('2d');
      const dist=$('pca-dist'),mode=$('pca-mode'),site=$('pca-site'),layer=$('pca-layer'),sx=$('pca-x'),sy=$('pca-y'),sz=$('pca-z'),yaw=$('pca-yaw'),pitch=$('pca-pitch'),status=$('pca-status');
      for(let i=1;i<=6;i++){{for(const s of [sx,sy,sz]){{const o=document.createElement('option');o.value=i-1;o.textContent='PC'+i;s.appendChild(o)}}}} sx.value=0;sy.value=1;sz.value=2;
      function sites(){{const vals=[...new Set(DATA.filter(r=>r.d===dist.value&&r.m===mode.value).map(r=>r.s))];const old=site.value;site.innerHTML='';vals.forEach(v=>{{const o=document.createElement('option');o.value=v;o.textContent=v.replaceAll('_',' ');site.appendChild(o)}});if(vals.includes(old))site.value=old}}
      function color(t){{const h=250-245*t;return `hsl(${{h}},78%,48%)`}}
      function draw(){{sites();const rows=DATA.filter(r=>r.d===dist.value&&r.m===mode.value&&r.s===site.value&&r.l===+layer.value);ctx.clearRect(0,0,canvas.width,canvas.height);if(!rows.length){{ctx.fillText('No data for this selection',30,40);return}}
        const xi=+sx.value,yi=+sy.value,zi=+sz.value, ya=+yaw.value*Math.PI/180,pi=+pitch.value*Math.PI/180;
        const raw=rows.map(r=>[r.p[xi],r.p[yi],r.p[zi],r.c]);const mins=[0,1,2].map(i=>Math.min(...raw.map(p=>p[i]))),maxs=[0,1,2].map(i=>Math.max(...raw.map(p=>p[i])));const center=mins.map((v,i)=>(v+maxs[i])/2),span=Math.max(...mins.map((v,i)=>maxs[i]-v),1e-9);
        const points=raw.map(p=>{{let x=(p[0]-center[0])/span,y=(p[1]-center[1])/span,z=(p[2]-center[2])/span;let x1=x*Math.cos(ya)-z*Math.sin(ya),z1=x*Math.sin(ya)+z*Math.cos(ya);let y1=y*Math.cos(pi)-z1*Math.sin(pi),z2=y*Math.sin(pi)+z1*Math.cos(pi);return [520+x1*760,245-y1*700,z2,p[3]]}}).sort((a,b)=>a[2]-b[2]);
        ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(60,440);ctx.lineTo(980,440);ctx.moveTo(60,440);ctx.lineTo(60,50);ctx.stroke();ctx.fillStyle='#334155';ctx.font='15px sans-serif';ctx.fillText('rotatable projection of '+('PC'+(xi+1))+', '+('PC'+(yi+1))+', '+('PC'+(zi+1)),70,35);
        points.forEach((p,i)=>{{ctx.beginPath();ctx.arc(p[0],p[1],4.2,0,Math.PI*2);ctx.fillStyle=color((p[3]-1)/127);ctx.globalAlpha=.82;ctx.fill();if(i>0){{const q=points[i-1];ctx.beginPath();ctx.moveTo(q[0],q[1]);ctx.lineTo(p[0],p[1]);ctx.strokeStyle='rgba(100,116,139,.18)';ctx.stroke()}}}});ctx.globalAlpha=1;
        const labels=[1,32,64,96,128];labels.forEach((v,i)=>{{ctx.fillStyle=color((v-1)/127);ctx.fillRect(800+i*42,462,34,10);ctx.fillStyle='#475569';ctx.font='11px sans-serif';ctx.fillText(v,805+i*42,492)}});
        status.textContent=`${{rows.length}} exact-count centroids | ${{dist.value}} / ${{mode.value}} / ${{site.value}} / depth ${{layer.value}}. Color encodes count/progress 1-128.`;
      }}
      [dist,mode,site,layer,sx,sy,sz,yaw,pitch].forEach(el=>el.addEventListener('input',draw));sites();draw();
    }})();
    </script>
    """

    html_doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{version.upper()} Synthetic Counting Report</title><style>{css}</style></head><body><main>
    <div class="kicker">Synthetic counting / {version.upper()}</div>
    <h1>{version.upper()}：{representation_name}下的长上下文计数</h1>
    <p class="subtitle">1024-token haystack，1-128 个 needle，Direct 与显式 CoT，Uniform 与 Power-law 训练分布。重点：学习动态、attention 表征与 hidden-state 几何。</p>
    <div class="thesis"><b>核心结论。</b> {main_interpretation} {marker_interpretation}</div>
    <div class="cards">
      <div class="card"><b>{pct(uniform_direct.primary_accuracy)}</b><span>Uniform / Direct final count</span></div>
      <div class="card"><b>{pct(uniform_cot.primary_accuracy)}</b><span>Uniform / CoT final count</span></div>
      <div class="card"><b>{pct(power_direct.primary_accuracy)}</b><span>Power / Direct final count</span></div>
      <div class="card"><b>{pct(power_cot.primary_accuracy)}</b><span>Power / CoT final count</span></div>
    </div>
    <nav class="toc"><b>目录</b><ol><li><a href="#question">研究对象</a></li><li><a href="#setting">实验设定</a></li><li><a href="#definitions">定义与计算</a></li><li><a href="#learning">学习动态</a></li><li><a href="#behavior">最终行为</a></li><li><a href="#attention">Attention 表征</a></li><li><a href="#ktok-diagnosis">为什么 k-to-k 很低</a></li><li><a href="#state">Hidden-state 表征</a></li><li><a href="#comparison">跨版本解释</a></li></ol></nav>

    <h2 id="question">1. 研究对象与两种计算假设</h2>
    <div class="two-col"><div class="mini"><h3>Direct / non-thinking</h3><p>模型在 prompt 结束后的单个答案 query 直接输出 count。候选机制是 <b>broad aggregation</b>：一组 head 把注意力分散到全部 needle，再由 residual/MLP 把集合统计压到 scalar count state。</p></div><div class="mini"><h3>Explicit CoT</h3><p>模型先生成逐步 trace，再生成最终 count。候选机制是 <b>targeted k-to-k retrieval</b>：第 k 个 index query 定位第 k 个 prompt needle，trace 逐项外化检索结果，最终答案从 trace 或其进度 state 读出。</p></div></div>
    <p>本报告同时检查两类证据：(1) 行为学习动态是否支持 CoT 的逐步算法优势；(2) attention 与 hidden state 是否出现与上述机制一致的描述性表征。没有 ablation/patching，因此这里不把描述性相关写成因果结论。</p>

    <h2 id="setting">2. 实验设定与受控表征变化</h2>
    <div class="table-wrap"><table><tbody>
      <tr><th>模型</th><td>Decoder-only pre-norm Transformer；4 layers × 4 heads；d_model=256；MLP=1024；RoPE base=10000；tied embedding/unembedding</td></tr>
      <tr><th>Prompt</th><td>长度 1024；256 种 noise token；10 种 marker/needle token；needle count n∈[1,128]</td></tr>
      <tr><th>两种分布</th><td>Uniform: p(n)=1/128；Power: p(n)∝n<sup>-1.5</sup>。评估始终按每个 exact count 平衡抽样，因此 Power 的高 count 是训练稀缺区。</td></tr>
      <tr><th>Direct completion</th><td><code>{esc(grammar_direct)}</code></td></tr>
      <tr><th>CoT completion</th><td><code>{esc(grammar_cot)}</code></td></tr>
      <tr><th>数值表征</th><td>{token_note}</td></tr>
      <tr><th>训练</th><td>completion-only teacher-forced CE；10000 steps × batch 32，共抽样 320000 个训练 examples；AdamW lr=3e-4，β=(0.9,0.95)，weight decay=.01；200-step warmup + cosine；clip=1；bf16；seed=1234</td></tr>
      <tr><th>动态评估</th><td>每 1000 step 评估一次；每个 run 的 dynamics checkpoint 使用 96 个平衡 validation prompts，因此曲线适合看阶段与回退，不适合把相邻点的小差异当成精确效应。</td></tr>
      <tr><th>最终评估</th><td>greedy free-running generation；每个 exact count 96 个独立样本，共 128×96=12288 个 prompts/run；四段为 1-32 / 33-64 / 65-96 / 97-128。</td></tr>
      <tr><th>表征分析</th><td>attention 每 count 1 个样本；state probe 训练/测试每 count 为 2/1。只有一个 seed，head 排名与 PCA 不能当作跨 seed 稳定性。</td></tr>
    </tbody></table></div>
    <div class="thesis info"><b>配对比较的准确表述。</b> v18 与 v19 固定了 prompt、架构、example-level count sampling、optimizer、step 数和评估；操纵的是 number representation。它不是“训练信号长度完全不变”的替换：shared digits 同时引入跨数值参数共享、多 token 数值、较长 CoT、不同的 per-example CE 权重。这些都是 representation 变化的机制性后果，解释跨版本结果时必须一起考虑。</div>
    <h3>Representation 如何改变监督长度与 token exposure</h3>
    <p>令 d(k) 为整数 k 的十进制位数。v18 的 Direct/CoT completion 长度分别是 L<sub>18,D</sub>(n)=1 与 L<sub>18,C</sub>(n)=2n+2；v19 分别是 L<sub>19,D</sub>(n)=d(n)+2 与 L<sub>19,C</sub>(n)=Σ<sub>k=1</sub><sup>n</sup>[d(k)+2]+d(n)+3。因为训练代码把一个 batch 内所有非 ignore 的 completion-token losses 取平均，count n 的期望 CE-token 权重为 q(n)=p(n)L(n)/Σ<sub>m=1</sub><sup>128</sup>p(m)L(m)。</p>
    {dataframe_html(exposure, percent_columns=['example share n=97-128','CE-token share n=97-128'])}
    <p class="subtitle">读表例：Uniform sampling 中 n=97-128 占 examples 的 25%，但在 v18/v19 CoT loss 中分别贡献约 43.3%/44.7% 的被监督 tokens；Power sampling 中它们只占 1.12% examples，却因长 trace 占约 12.9%/14.3% 的 CoT tokens。n=128 的 CoT completion 长度为 v18 258、v19 538 tokens。</p>

    <h2 id="definitions">3. 新术语、数据列与计算公式</h2>
    <ul class="math-list">
      <li><b>Free-running exact final-count accuracy</b>：模型从 prompt 自由生成完整 completion；解析最终 scalar answer ĉ，计算 1[ĉ=n]。Direct 与 CoT 的主指标相同。</li>
      <li><b>Valid-scalar rate 与 exact-given-valid</b>：前者是生成结果符合对应 grammar、能解析出 scalar ĉ 的比例；后者是在这个子集上计算 mean 1[ĉ=n]。overall exact = valid-scalar rate × exact-given-valid。表中的 MAE=mean |ĉ-n| 也只在可解析子集上计算。</li>
      <li><b>Enumeration accuracy</b>：CoT 生成的 trace 步数 n̂ 是否等于 n，即 1[n̂=n]。它只检验“数了几步”，不检验每步 marker 身份。</li>
      <li><b>Trace-marker accuracy</b>：把生成 marker 序列与 gold marker 序列按位置 zip，对前 n 个 gold 位置计分：Σ<sub>k=1</sub><sup>min(n,n̂)</sup>1[M̂<sub>k</sub>=M<sub>k</sub>] / n。十种 marker 的随机基线约 0.10。</li>
      <li><b>Trace-exact accuracy</b>：trace 长度正确且整个 marker 序列逐位完全正确。</li>
      <li><b>Stable acquisition step</b>：在观测 checkpoint t 首次达到阈值 τ，且从 t 到 10000 的所有已观测点都不低于容忍下界 f。本文用 (τ,f)=(50%,45%)、(90%,85%)、(99%,95%)。它过滤短暂越线，但由于每 1000 step 才评估一次，最后一个 checkpoint 达标只能表示“截至现有观测未再回落”。</li>
      <li><b>Attention mass</b>：对 query q 和位置集合 S，M(S|q)=Σ<sub>j∈S</sub>A[q,j]。</li>
      <li><b>Broad-attention score</b>：prompt_needles_mass × H<sub>normalized</sub>(needle attention)。第一项奖励 mass 落在 needle 集合，第二项奖励在多个 needles 间均匀覆盖。</li>
      <li><b>k-to-k raw mass</b>：CoT 第 k 个 index query 对 prompt 中第 k 个 needle 的 raw attention A[q<sub>k</sub>,needle<sub>k</sub>]。</li>
      <li><b>Correct top-1</b>：只在 prompt needle 子集中取 argmax，检查最大 attention 是否落到 matching needle。</li>
      <li><b>Diagonal dominance</b>：k-to-k raw mass / prompt_needles_mass。它是 needle 子集内部的相对占比；即使接近 1，raw mass 仍可能很低。</li>
      <li><b>Ridge R²</b>：用 held-out hidden state 线性回归 count/progress；R²=1-Σ(y-ŷ)²/Σ(y-ȳ)²。高 R² 只表示线性可读，不证明该方向被模型因果使用。</li>
      <li><b>PC1-PC6 cumulative explained variance</b>：对 exact-count centroid 做 PCA，报告前六主成分方差和 / 总方差。值越高，count centroid manifold 越集中在低维子空间。</li>
    </ul>

    <h2 id="learning">4. 学习动态</h2>
    {figure_html(assets['loss'], 'Figure 1. Completion-token training loss', '横轴是 optimizer step；纵轴是被监督 completion tokens 的平均 next-token cross-entropy，使用 symlog 轴保留接近 0 的差异。左/右分别是 Uniform/Power，颜色区分 Direct 与 CoT。该 loss 不包含 1024-token prompt。')}
    {figure_html(assets['primary'], 'Figure 2. Free-running exact final count', '横轴是训练 step；纵轴是在平衡 validation prompts 上自由生成后的 exact final-count accuracy。虚线是 99%。该图揭示“曾经学会后又退化”的情况，不能由最终 checkpoint 单点替代。')}
    {figure_html(assets['bands'], 'Figure 3. Count difficulty-resolved dynamics', '四个面板分别是 distribution × mode。横轴为 step，纵轴为 exact final-count accuracy；四条线分别对应 gold count 1-32、33-64、65-96、97-128。Power 训练中高 count 更稀缺，但评估四段等权。')}
    {figure_html(assets['cot_decomp'], 'Figure 4. CoT learning decomposed', '横轴为 step，纵轴为 free-running accuracy。final count 检验最终数值；trace length 检验生成步数；marker identity 检验逐步 marker 身份；entire trace exact 要求整条 marker trace 全对。灰色点线是 10-marker identity chance。')}
    {figure_html(assets['validity'], 'Figure 5. CoT grammar validity versus scalar correctness', '横轴是训练 step；纵轴是 free-running 样本比例。橙线为 overall exact final count；蓝线为 completion 中能解析出合法 final scalar 的比例；绿线为只在可解析子集上的 exact accuracy。三线分离意味着错误来自 grammar/termination，而不只是数值预测。')}
    <h3>稳定达到阈值的最早观测 checkpoint</h3>
    <p>每个单元格报告 stable acquisition step：先达到列标题中的阈值，并在所有后续已观测点保持在括号内 floor 以上；“not reached” 表示 10000 step 内未满足。它比“第一次越线”更保守，但 checkpoint 间隔仍是 1000 steps。</p>
    {dataframe_html(thresholds)}
    <h3>峰值与最终动态点</h3>
    {dataframe_html(peak, percent_columns=['peak accuracy','last dynamics accuracy','peak-to-last change'])}
    <div class="thesis {'warning' if is_v19 else ''}"><b>学习动态解读。</b> {('v19 Uniform-CoT 在 step 6000 达到 96.9% 峰值，随后降至最后 dynamics 点 37.5%，独立 final set 为 45.4%；它不是从未学会，而是发生了 late regression。Figure 5 进一步显示 step 9000-10000 时可解析率降到 46.9%/38.5%，而可解析子集仍为 100.0%/97.3% exact。' if is_v19 else 'v18 的曲线并非单调，但 Uniform-Direct/CoT 后期达到约 94%-96%；Power 分布呈明显 difficulty curriculum，低 count 先学，高 count 继续受 exposure 限制，最终 CoT 比 Direct 高约 24.6 percentage points。Uniform-CoT 的 marker identity 从 step 3000 的 41.0% 逐步升到 80.7%，而 Power-CoT 始终约 chance。')} {validity_interpretation}</div>

    <h2 id="behavior">5. 最终 checkpoint 行为</h2>
    {figure_html(assets['final_bands'], 'Figure 6. Final scalar count by difficulty', '横轴为四个平衡 count 区间；纵轴为自由生成的 exact final-count accuracy。每个分布面板内蓝/橙柱比较 Direct 与 CoT。')}
    {figure_html(assets['final_trace'], 'Figure 7. CoT trace quality by difficulty', '横轴为 count 区间；纵轴分别显示 trace 步数正确、逐位置 marker identity、整条 trace exact。10% 点线是单个 marker identity 的随机基线，不是整条 trace exact 的基线。')}
    {dataframe_html(final, percent_columns=['primary_accuracy','valid_scalar_rate','exact_given_valid','enumeration_accuracy','trace_marker_accuracy','trace_exact_accuracy'])}
    <p class="subtitle">valid_scalar_rate 是能够解析出合法 final scalar 的样本比例；exact_given_valid 只在这些可解析样本上计算。primary_mae 同样只对可解析样本取平均，因此必须与 valid_scalar_rate 联读。</p>
    <div class="thesis warning"><b>不要把 trace length 当作 faithful retrieval。</b> {trace_warning}</div>

    <h2 id="attention">6. 描述性 attention 表征</h2>
    <p>以下每个热图单元格是某 layer/head 在该分布与 count 段上的平均 score。纵轴 layer 1-4，横轴 head 0-3。注意：head score 是路由描述，不是 ablation 或 patching 后的因果效应。</p>
    {figure_html(assets['broad'], 'Figure 8. Direct broad-aggregation candidates', '每个小面板对应一个 distribution × count band；横轴是 0-based head 0-3，纵轴是 layer 1-4。颜色与格内数字都是 broad_attention_score = prompt_needles_mass × normalized needle entropy；高值表示 final-answer query 把较多 mass 放到 needles，并在多个 needles 间铺开。')}
    {figure_html(assets['ktok_raw'], 'Figure 9. CoT k-to-k raw attention mass', '每个小面板对应一个 distribution × count band；横轴是 head 0-3，纵轴是 layer 1-4。格内值是 trace index query k 直接投向 matching prompt needle k 的 raw mass；色条范围使用本报告实际最大值而非强制 0-1。')}
    {figure_html(assets['ktok_top1'], 'Figure 10. CoT correct top-1 within prompt needles', '每个小面板对应一个 distribution × count band；横轴是 head 0-3，纵轴是 layer 1-4。格内值是 matching needle 在 prompt needle 子集内获得最高 attention 的比例；它忽略 noise/BOS/trace mass，须与 Figure 9 和 Figure 11 联读。')}
    {figure_html(assets['ktok_diag'], 'Figure 11. CoT needle-conditional diagonal dominance', '每个小面板对应一个 distribution × count band；横轴是 head 0-3，纵轴是 layer 1-4。格内值为 matching-needle mass / all-prompt-needle mass；高值只说明 needle 子集内对角占优，不说明总 attention mass 很大。')}
    {figure_html(assets['readout'], 'Figure 12. CoT final-answer attention to trace markers', '每个小面板对应一个 distribution × count band；横轴是 head 0-3，纵轴是 layer 1-4。格内值是 final-answer query 投向所有已生成 trace-marker positions 的总 mass，用于寻找 trace-readout candidate heads。')}
    <h3>按描述性 score 排名的最佳 head</h3>{dataframe_html(heads)}
    <div class="thesis warning"><b>Attention 结论边界。</b> {attention_interpretation} 高 top-1 或 diagonal dominance 可能来自“在很少的 needle mass 内排序正确”；当前数据没有 head ablation/patching，不能单独证明 targeted retrieval circuit。</div>

    <h2 id="ktok-diagnosis">7. 为什么 k-to-k top-1 与 attention mass 很低</h2>
    <div class="thesis"><b>诊断结论。</b> {ktok_version_takeaway} 因此低 k-to-k 分数包含两种不同现象：<b>真实 retrieval 未形成</b>，以及<b>已经存在的 retrieval 采用多跳 relay、被 direct-edge 指标低估</b>。</div>

    <h3>7.1 正确基线：绝对值低不等于完全随机</h3>
    <p>当前 <code>correct_top1</code> 要在 n 个 prompt needles 中选中“第 k 个 exact occurrence”，单个 query 的随机基线是 1/n。attention CSV 对每个 trace query 各记一行，因此在 count 1-128 各一个 prompt 时，整体 query-weighted 随机基线为：</p>
    <div class="formula">Σ<sub>n=1</sub><sup>128</sup> n(1/n) / Σ<sub>n=1</sub><sup>128</sup>n = 128/8256 = 1.55%</div>
    <p>raw mass 的均匀位置基线则是 mean[1/T<sub>q</sub>]，其中 T<sub>q</sub> 是该 query 可见的 token 数；v18/v19 分别约 0.090%/0.085%。下表使用 <code>count_band=all</code> 的完整 attention queries；best raw-mass head 与 best top-1 head 可以不是同一个 head。</p>
    {dataframe_html(ktok_final, percent_columns=['final marker accuracy','best exact-k raw mass','uniform-position baseline','best exact top-1','random exact top-1'])}
    <p>例如 v18 Uniform-CoT 的最佳 exact-k mass 虽只有 0.93%，却是均匀单位置基线的 10.3 倍；其 exact top-1 为 3.63%，约为 1.55% 随机基线的 2.3 倍。另一方面，v19 Uniform 的 raw-mass lift 更高但 marker 仍近 chance，说明 attention mass 既不是 retrieval 成功的必要充分统计量，也不能单独解释行为。</p>

    <h3>7.2 Checkpoint 证据：v18 Uniform 出现的是 trace relay</h3>
    <p>下面是固定三个 prompts（count=16/48/96，sampling seed=100407）上的 teacher-forced checkpoint 诊断。每个 score 都先对这些 prompts 的全部 k 求平均，再在 4×4 heads 中取最大值；它用于辨认阶段变化，不是 prompt-level 置信区间。</p>
    {dataframe_html(ktok_checkpoints, percent_columns=['teacher-forced marker accuracy','best exact-k mass','best exact top-1','best previous-marker relay mass'])}
    <p>v18 Uniform 从 step 2000 到 6000 的 marker accuracy 由 10.0% 跳到 82.5%，但最佳 exact-position top-1 没有同步变成 one-hot；最明显的新结构是 <code>I_k</code> 对前一个 trace-marker position 的 relay mass 从 0.47% 增至 46.5%。v19 两个 run 的 marker accuracy 始终约 chance，即使 v19 Uniform 的 direct mass 后期上升，也没有形成可靠 marker copier。</p>

    <h3>7.3 Attention-key masking：prompt source 与 trace memory 都必要</h3>
    <p>下表在同一组 teacher-forced prompts 上，把指定位置设为所有 layer 都不可作为 attention key；数值是 marker-token accuracy。它测试某类 source position 的必要性，但一次屏蔽一整类 keys，不能单独确定具体 head 或唯一传播路径。</p>
    {dataframe_html(ktok_masks, percent_columns=['v18 Uniform step 6000','v18 Uniform step 10000','v19 Uniform step 10000','v19 Power step 10000'])}
    <p>v18 Uniform step 10000 从 clean 84.4% 降到：屏蔽 prompt needles 后 5.6%，屏蔽 trace markers 后 44.4%，屏蔽 trace number/index 后 18.1%；屏蔽 START 仍为 82.5%。因此它既依赖原始 needle 信息，也依赖 trace 中间状态，但不是把 START 当成唯一 summary bottleneck。与数据最相容的候选路线是：</p>
    <div class="formula">needle<sub>k</sub> → later prompt/residual proxy states → trace relay states → I<sub>k</sub> → M<sub>k</sub></div>

    <h3>7.4 Exact kth occurrence 有强因果作用，但 direct mass 仍可很小</h3>
    <p>在 v18 Uniform step 10000 的一个 n=96 prompt（sampling seed=78675）上，选取 k={8,16,24,32,48,64,80,96}。下表报告对 gold marker log-probability 的平均变化；“other same-ID”是另一个具有相同 marker token 的 occurrence。</p>
    {dataframe_html(ktok_sources)}
    <p>只替换第 k 个 source token 会使 gold log-probability 平均下降 8.57 nats；直接屏蔽该 key 也下降 7.60 nats。替换/屏蔽另一个 same-ID occurrence 或普通 control 的影响很小。这排除了“只需要在重复 marker identity 中任选一个”的解释，并证明 exact kth source 具有强因果必要性。</p>
    <p>之所以这个结果可以和低 direct mass 共存，是因为 Figure 9 只统计当前 query 对原始 source 的单层边权。source token 可以先改变后续 prompt token、trace position 或 residual state，再多跳到达 <code>I_k</code>。而真正写入 residual 的量是 <code>A[q,j]·V[j]·W<sub>O</sub></code>；attention weight 本身不等于 value-weighted contribution，更不等于所有路径的总因果贡献。</p>

    <h3>7.5 为什么 v18 Power 与 v19 停在“忽略 marker”的捷径</h3>
    <p>令 f<sub>M</sub> 是 completion 中 marker tokens 的期望比例。如果结构性 index/digit/END/count tokens 全部预测正确，而十种 marker 完全随机，则整体 token CE 的理论 floor 是：</p>
    <div class="formula">L<sub>ignore-marker</sub> = f<sub>M</sub> · ln(10)</div>
    {dataframe_html(marker_floor, percent_columns=['marker-token fraction','final marker accuracy'])}
    <p>step-10000 logged training loss 是单个记录 batch，存在采样噪声，但 v18 Power 与两个 v19-CoT 都几乎贴在这一理论 floor；v18 Uniform 则从理论 1.134 降到 0.134。这是很强的证据：前三者主要学会 deterministic scaffold，并把 marker identity 留在 chance 水平。</p>
    <p>Power sampling 还使高 k 的监督极少。下表按 10000 steps × batch 32=320000 个 examples 计算期望出现次数；第 k 个 trace target 只有在 n≥k 时才出现。</p>
    {dataframe_html(high_k_exposure)}
    <p>shared-digit v19 又把 marker token 占比从约 45%-49% 降至约 25%，并把 n=128 CoT completion 从 258 tokens 拉长到 538 tokens。easy INDEX/digit/count scaffold 在梯度中占比更高，绝对 trace position 又泄露 k/n，因而“长度和最终 count 正确、marker 忽略”的局部解尤其有吸引力。</p>

    <h3>7.6 对后续实验与指标的直接含义</h3>
    <ul><li>把 exact top-1 同时报告为 raw accuracy 与相对 1/n 的 lift；把 exact mass 报告为相对 1/T<sub>q</sub> 的 lift。</li><li>增加 attention rollout、value-weighted contribution、gradient attribution，以及 clean→corrupt activation patching；不要用单层 attention heatmap代替因果路线。</li><li>分别在 teacher-forced gold trace 与 free-running generated trace 上收集 attention。尤其 v19 Uniform 的生成 grammar 会后期失稳，gold-trace attention 不能代表实际生成轨迹。</li><li>按 token type 分解 CE；若目标是 faithful CoT，可提高 marker loss 权重、按 token family 分别归一化，或增加显式 pointer/alignment 辅助目标。</li><li>对 Power sampling 增加 high-k curriculum/oversampling，并在每个 checkpoint 跟踪 marker accuracy、relay mass 与 causal source effect，而不只看最终 count。</li></ul>

    <h2 id="state">8. Hidden-state count/progress 表征</h2>
    {figure_html(assets['probe'], 'Figure 13. Linear count/progress readability', '横轴是 hidden-state extraction depth：Embed 表示 token+position 输入，L1-L4 表示依次经过各 Transformer layer 后的 residual stream。纵轴是 held-out ridge R²。Direct/CoT final site 的标签是 gold final count；CoT trace-index/marker site 的标签是当前 trace progress k。')}
    {figure_html(assets['pca'], 'Figure 14. PC1-PC6 cumulative explained variance', '对每个 distribution × mode × semantic site × depth，先按 exact count/progress 求 hidden-state centroid，再对 128 个 centroids 做 PCA。纵轴是 PC1-PC6 累计解释的 centroid 方差比例；高值表示 count geometry 可被低维子空间概括。')}
    {figure_html(assets['pc12'], 'Figure 15. Layer-4 PC1-PC2 centroid trajectories', '四个面板展示 Layer 4 的 exact-count/progress centroids；横纵轴分别是该条件独立拟合 PCA 的 PC1/PC2 坐标，颜色从 count/progress 1 到 128。连线只帮助观察按标签排序的几何路径，不代表训练时间或真实连续动力学。各面板独立拟合 PCA，坐标符号与尺度不能跨面板直接比较。')}
    <h3>Layer 4 数值摘要</h3>
    {dataframe_html(state_l4, percent_columns=['position_only_accuracy','nearest_centroid_accuracy','cumulative_explained_variance'])}
    <h3>交互式 PC1-PC6 三维视图</h3>
    <p>可选择 distribution、mode、semantic site、深度以及任意三个 PC 轴。每个点是一个 exact count/progress centroid；颜色编码 1-128。该视图用于检查曲线、折叠和层间压缩，不提供因果证据。</p>
    {widget}
    <div class="thesis info"><b>Hidden-state 读法。</b> {state_interpretation} 所有 CoT sites（包括 final answer）的 position-only baseline 都可达 1，因为 completion 的绝对位置由 n 或 k 决定；因此 CoT 高 probe 不能单独证明内部加法。Direct final-answer 的 position-only baseline 才是 1/128≈0.78%，其深层高 R² 信息量更强。</div>

    <h2 id="comparison">9. 与配对 tokenization 的比较</h2>
    <p>配对报告 <b>{counterpart}</b> 固定架构、example-level count sampling、训练预算和评估，并改变 number representation。两套模型独立随机训练，所以最可靠的是同 distribution/mode 的行为对照与 family-level geometry；attention head 编号、PCA 轴方向和单个 checkpoint 噪声不能硬对齐。</p>
    {figure_html(assets['representation_dynamics'], 'Figure 16. Paired representation learning dynamics', '四个面板是 distribution × mode；横轴为 step，纵轴为 free-running exact final-count accuracy。蓝色圆点是 v18 atomic classes，橙色方块是 v19 shared digits；虚线为 99%。该图比较的是独立模型的观测轨迹，不是同一模型内的 intervention。')}
    {figure_html(assets['representation_delta'], 'Figure 17. Final digit-minus-atomic effect by difficulty', '横轴为 gold count band；纵轴为 v19 digit accuracy 减 v18 atomic accuracy，单位 percentage points。0 以上表示 digit 较高，0 以下表示 atomic 较高；蓝/橙柱分别为 Direct/CoT。')}
    {figure_html(assets['representation_pca'], 'Figure 18. Paired Layer-4 dimensional concentration', '横轴为 semantic extraction site，纵轴为前六个 PCs 对 128 个 exact-count/progress centroids 的累计解释方差。每个 distribution 面板比较 v18 atomic 与 v19 digits。PCA 在每个条件内独立拟合，因此这里只比较方差集中度，不比较 PC 坐标本身。')}
    {dataframe_html(representation_final, percent_columns=['v18 atomic','v19 shared digits'])}
    <div class="thesis"><b>综合解释。</b> {main_interpretation} Figure 17 显示效应具有强烈 distribution × mode × difficulty 交互：Power-CoT 的 digit 优势从 1-32 的 +0.3 pp 扩大到 97-128 的 +60.5 pp；Uniform-CoT 则从 -12.2 pp 恶化到 -76.1 pp。与此同时 v19 hidden centroids 更低维。shared digits 因而不是单调“更好”的编码，而是同时改变参数共享、生成稳定性和 token-level exposure；目前单 seed 不能区分这些成分的平均因果效应。</div>

    <h2>10. 限制与下一步</h2>
    <ul><li>只有 seed 1234，不能估计训练方差或 head 稳定性。</li><li>标准 Attention 分析每 exact count 只有 1 个 prompt；虽然一个 CoT prompt 产生多个 trace query，prompt-level 不确定性仍被低估。</li><li>Section 7 的 key masking 与 exact-source corruption 提供了 source-position 必要性证据，但诊断样本较小，也没有定位唯一 head/MLP 路径；仍需 activation patching 与 component-level ablation。</li><li>Power 与 Uniform 的训练 token exposure 不同；建议保存 peak checkpoint，并在多 seed 下复现 marker-loss floor 与 relay transition。</li><li>Trace marker identity 近 chance 时，应先确认任务是否只要求 count，还是要求 faithful retrieval；两种目标不能混为同一个“CoT 成功”。</li></ul>
    <div class="footer">Generated from <code>{esc(root)}</code>. All figures are embedded as base64; the report is self-contained. Configuration seed={config['seed']}, generated by scripts/build_v18_v19_reports.py.</div>
    </main></body></html>"""
    return html_doc


def build_one(version: str, root: Path, other_root: Path) -> Path:
    tables = load_tables(root)
    other_tables = load_tables(other_root)
    v18_tables, v19_tables = (other_tables, tables) if version == "v19" else (tables, other_tables)
    out = root / "report_assets"
    assets = {
        "loss": plot_training_loss(tables, out),
        "primary": plot_primary_dynamics(tables, out),
        "bands": plot_band_dynamics(tables, out),
        "cot_decomp": plot_cot_decomposition(tables, out),
        "validity": plot_cot_validity_dynamics(tables, out),
        "final_bands": plot_final_count_bands(tables, out),
        "final_trace": plot_final_trace_bands(tables, out),
        "broad": plot_attention_grid(tables, out, metric="broad_attention_score", query_kind="final_answer", mode="direct", filename="attention_direct_broad.png", title="Direct final-query broad-attention score by count band", fixed_scale=(0, 1)),
        "ktok_raw": plot_attention_grid(tables, out, metric="correct_prompt_needle_mass", query_kind="trace_index", mode="cot", filename="attention_cot_ktok_raw.png", title="CoT trace-index k-to-k raw attention mass by count band"),
        "ktok_top1": plot_attention_grid(tables, out, metric="correct_top1", query_kind="trace_index", mode="cot", filename="attention_cot_ktok_top1.png", title="CoT trace-index correct top-1 within prompt needles", fixed_scale=(0, 1)),
        "ktok_diag": plot_attention_grid(tables, out, metric="diagonal_dominance", query_kind="trace_index", mode="cot", filename="attention_cot_ktok_diagonal.png", title="CoT trace-index needle-conditional diagonal dominance", fixed_scale=(0, 1)),
        "readout": plot_attention_grid(tables, out, metric="trace_markers_mass", query_kind="final_answer", mode="cot", filename="attention_cot_trace_readout.png", title="CoT final-answer mass on generated trace markers", fixed_scale=(0, 1)),
        "probe": plot_probe(tables, out),
        "pca": plot_pca_coverage(tables, out),
        "pc12": plot_pc12_summary(tables, out),
        "representation_dynamics": plot_representation_dynamics(v18_tables, v19_tables, out),
        "representation_delta": plot_representation_delta(v18_tables, v19_tables, out),
        "representation_pca": plot_representation_pca(v18_tables, v19_tables, out),
    }
    document = report_text(version=version, root=root, tables=tables, other_tables=other_tables, assets=assets)
    target = root / f"syn_{version}_report.html"
    target.write_text(document, encoding="utf-8")
    print(f"Wrote {target} ({target.stat().st_size / 1024 / 1024:.2f} MiB)")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v18-root", type=Path, default=DEFAULT_V18)
    parser.add_argument("--v19-root", type=Path, default=DEFAULT_V19)
    args = parser.parse_args()
    v18 = args.v18_root.resolve()
    v19 = args.v19_root.resolve()
    build_one("v18", v18, v19)
    build_one("v19", v19, v18)


if __name__ == "__main__":
    main()
