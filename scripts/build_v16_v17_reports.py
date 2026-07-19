from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COUNT_BINS = ("1-10", "11-20", "21-30")
MODES = ("nonthinking", "thinking")
BLUE = "#2563eb"
ORANGE = "#f97316"
GREEN = "#16a34a"
PURPLE = "#7c3aed"
RED = "#dc2626"
GRAY = "#64748b"
GRID = "#dbe4f0"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def final_rows(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if "step" not in frame.columns:
        return frame.copy()
    indices = frame.groupby(keys, dropna=False)["step"].idxmax()
    return frame.loc[indices].sort_values(keys).reset_index(drop=True)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{100.0 * number:.{digits}f}%"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def table_html(frame: pd.DataFrame, *, float_format: str = "{:.3f}") -> str:
    view = frame.copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: "-" if pd.isna(value) else float_format.format(float(value))
            )
    return view.to_html(index=False, border=0, classes="data-table", escape=True)


def image_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def figure(path: Path, title: str, caption: str) -> str:
    return f"""
    <figure class="report-figure">
      <h4>{esc(title)}</h4>
      <img src="{image_uri(path)}" alt="{esc(title)}">
      <figcaption>{caption}</figcaption>
    </figure>
    """


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.2,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def draw_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    xlabels: list[str],
    ylabels: list[str],
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    digits: int = 2,
) -> Any:
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, pad=8)
    ax.set_xticks(np.arange(len(xlabels)), labels=xlabels)
    ax.set_yticks(np.arange(len(ylabels)), labels=ylabels)
    threshold = vmin + 0.53 * (vmax - vmin)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                ax.text(
                    column,
                    row,
                    f"{value:.{digits}f}",
                    ha="center",
                    va="center",
                    color="black" if value >= threshold else "white",
                    fontsize=8.5,
                )
    return image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_v17_exports(primary: Path, duplicate: Path | None) -> tuple[bool, str]:
    if duplicate is None or not duplicate.exists():
        return False, "未提供第二份导出目录。"
    relative = [
        Path("config.json"),
        Path("vocab.json"),
        Path("manifest.json"),
        Path("tables/eval_by_bin.csv"),
        Path("tables/autoregressive_by_bin.csv"),
        Path("tables/attention_summary.csv"),
        Path("tables/state_probe_summary.csv"),
        Path("checkpoints/rope/nonthinking/final/checkpoint.pt"),
        Path("checkpoints/rope/thinking/final/checkpoint.pt"),
    ]
    mismatches: list[str] = []
    for item in relative:
        left = primary / item
        right = duplicate / item
        if not left.exists() or not right.exists() or sha256(left) != sha256(right):
            mismatches.append(str(item))
    if mismatches:
        return False, "以下关键文件不同：" + ", ".join(mismatches)
    return True, "两份目录的配置、核心表格和两个最终 checkpoint 的 SHA-256 均一致。"


def audit_bundle(run_dir: Path, positions: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    required_tables = (
        "train_metrics.csv",
        "eval_losses.csv",
        "eval_by_bin.csv",
        "eval_by_count.csv",
        "autoregressive_by_bin.csv",
        "autoregressive_detail.csv",
        "attention_summary.csv",
        "state_probe_summary.csv",
        "state_pca_variance.csv",
        "state_centroids_pca.csv",
        "time_to_99.csv",
        "training_count_distribution.csv",
    )
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in required_tables:
        path = run_dir / "tables" / name
        if not path.exists():
            missing.append(str(path))
            rows.append({"artifact": name, "status": "missing", "rows": "-"})
        else:
            rows.append({"artifact": name, "status": "complete", "rows": len(read_csv(path))})
    for position in positions:
        for mode in MODES:
            root = run_dir / "checkpoints" / position / mode
            final_ok = (root / "final" / "checkpoint.pt").exists()
            snapshots = len(list(root.glob("step_*")))
            status = "complete" if final_ok and snapshots == 10 else "incomplete"
            rows.append(
                {
                    "artifact": f"checkpoint {position}/{mode}",
                    "status": status,
                    "rows": f"final + {snapshots} snapshots",
                }
            )
            if status != "complete":
                missing.append(str(root))
    for stage in ("train", "attention", "state", "plots"):
        status = manifest.get("stages", {}).get(stage, {}).get("status", "missing")
        rows.append({"artifact": f"stage {stage}", "status": status, "rows": "-"})
        if status != "complete":
            missing.append(f"manifest stage {stage}")
    if missing:
        raise RuntimeError("Incomplete bundle:\n" + "\n".join(missing))
    return pd.DataFrame(rows), config, manifest


def load_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    names = (
        "train_metrics",
        "eval_losses",
        "eval_by_bin",
        "eval_by_count",
        "autoregressive_by_bin",
        "autoregressive_detail",
        "attention_summary",
        "state_probe_summary",
        "state_pca_variance",
        "state_centroids_pca",
        "time_to_99",
        "training_count_distribution",
    )
    return {name: read_csv(run_dir / "tables" / f"{name}.csv") for name in names}


def subplot_grid(nrows: int, ncols: int, *, width: float = 6.2, height: float = 4.0, **kwargs: Any) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(7.5, width * ncols), max(4.0, height * nrows)),
        constrained_layout=True,
        **kwargs,
    )
    return fig, np.asarray(axes, dtype=object).reshape(nrows, ncols)


def plot_training_loss(train: pd.DataFrame, positions: tuple[str, ...], out_dir: Path, all_sequence: bool) -> Path:
    fig, axes = subplot_grid(1, len(positions), width=6.3, height=4.4, sharey=True)
    colors = {"nonthinking": BLUE, "thinking": ORANGE}
    for column, position in enumerate(positions):
        ax = axes[0, column]
        for mode in MODES:
            rows = train[(train.position_encoding == position) & (train["mode"] == mode)].sort_values("step")
            ax.plot(rows.step, rows.train_total_loss, color=colors[mode], linewidth=2.1, label=mode)
        ax.set_title(position.upper())
        ax.set_xlabel("training step")
        ax.set_ylabel("mean next-token cross-entropy")
        ax.grid(color=GRID, alpha=0.72)
        ax.set_xlim(0, float(train.step.max()))
        ax.legend(frameon=False)
    scope = "prompt + completion" if all_sequence else "completion tokens only"
    fig.suptitle(f"Training objective over {scope}")
    return save_figure(fig, out_dir / "01_training_loss.png")


def plot_segment_losses(losses: pd.DataFrame, positions: tuple[str, ...], out_dir: Path, all_sequence: bool) -> Path:
    metrics = {
        "eval_prompt_body_loss": ("prompt body", GRAY),
        "eval_final_count_loss": ("final count", BLUE),
        "eval_trace_marker_loss": ("trace marker", GREEN),
        "eval_trace_index_loss": ("trace index", PURPLE),
        "eval_think_close_loss": ("think close", RED),
    }
    fig, axes = subplot_grid(len(positions), 2, width=6.0, height=3.8)
    for row_index, position in enumerate(positions):
        for column_index, mode in enumerate(MODES):
            ax = axes[row_index, column_index]
            rows = losses[(losses.position_encoding == position) & (losses["mode"] == mode)].sort_values("step")
            for metric, (label, color) in metrics.items():
                if metric in rows.columns and rows[metric].notna().any():
                    ax.plot(rows.step, rows[metric], color=color, linewidth=1.9, label=label)
            ax.set_yscale("log")
            ax.set_title(f"{position.upper()} / {mode}")
            ax.set_xlabel("training step")
            ax.set_ylabel("segment CE (log scale)")
            ax.grid(color=GRID, alpha=0.65, which="both")
            ax.legend(frameon=False, fontsize=8.5)
    title = "Validation loss by semantic segment"
    if all_sequence:
        title += ": prompt modeling and counting are optimized together"
    fig.suptitle(title)
    return save_figure(fig, out_dir / "02_segment_losses.png")


def plot_final_learning(eval_bins: pd.DataFrame, ar_bins: pd.DataFrame, positions: tuple[str, ...], out_dir: Path) -> Path:
    colors = {"1-10": BLUE, "11-20": ORANGE, "21-30": GREEN}
    fig, axes = subplot_grid(len(positions), 2, width=6.2, height=3.8, sharex=True, sharey=True)
    for row_index, position in enumerate(positions):
        for column_index, mode in enumerate(MODES):
            ax = axes[row_index, column_index]
            for count_bin in COUNT_BINS:
                tf = eval_bins[
                    (eval_bins.position_encoding == position)
                    & (eval_bins["mode"] == mode)
                    & (eval_bins.count_bin.astype(str) == count_bin)
                ].sort_values("step")
                ar = ar_bins[
                    (ar_bins.position_encoding == position)
                    & (ar_bins["mode"] == mode)
                    & (ar_bins.count_bin.astype(str) == count_bin)
                ].sort_values("step")
                ax.plot(tf.step, tf.tf_final_accuracy, color=colors[count_bin], linewidth=2.0, label=count_bin)
                ax.plot(ar.step, ar.ar_final_accuracy, color=colors[count_bin], linewidth=1.6, linestyle="--")
            ax.set_title(f"{position.upper()} / {mode}")
            ax.set_xlabel("training step")
            ax.set_ylabel("final-count accuracy")
            ax.set_ylim(-0.03, 1.04)
            ax.grid(color=GRID, alpha=0.7)
            ax.legend(title="gold count bin", frameon=False, loc="lower right")
    fig.suptitle("Final-count dynamics: teacher-forced (solid) and autoregressive (dashed)")
    return save_figure(fig, out_dir / "03_final_count_learning.png")


def plot_trace_learning(eval_bins: pd.DataFrame, ar_bins: pd.DataFrame, positions: tuple[str, ...], out_dir: Path) -> Path:
    colors = {"1-10": BLUE, "11-20": ORANGE, "21-30": GREEN}
    fig, axes = subplot_grid(len(positions), 2, width=6.2, height=3.8, sharex=True, sharey=True)
    for row_index, position in enumerate(positions):
        for column_index, (source, metric, title) in enumerate(
            (
                (eval_bins, "tf_trace_index_accuracy", "teacher-forced trace-index accuracy"),
                (ar_bins, "trace_exact", "autoregressive trace exact match"),
            )
        ):
            ax = axes[row_index, column_index]
            for count_bin in COUNT_BINS:
                rows = source[
                    (source.position_encoding == position)
                    & (source["mode"] == "thinking")
                    & (source.count_bin.astype(str) == count_bin)
                ].sort_values("step")
                ax.plot(rows.step, rows[metric], color=colors[count_bin], linewidth=2.0, label=count_bin)
            ax.set_title(f"{position.upper()} / {title}")
            ax.set_xlabel("training step")
            ax.set_ylabel("trace metric")
            ax.set_ylim(-0.03, 1.04)
            ax.grid(color=GRID, alpha=0.7)
            ax.legend(title="gold count bin", frameon=False, loc="lower right")
    fig.suptitle("Thinking-trace learning: local teacher forcing versus free generation")
    return save_figure(fig, out_dir / "04_trace_learning.png")


def plot_final_by_count(eval_count: pd.DataFrame, ar_detail: pd.DataFrame, positions: tuple[str, ...], out_dir: Path) -> Path:
    tf = final_rows(eval_count, ["position_encoding", "mode", "count"])
    ar = ar_detail.copy()
    if "step" in ar.columns:
        final_step = ar.groupby(["position_encoding", "mode"], dropna=False)["step"].transform("max")
        ar = ar[ar["step"] == final_step]
    ar_mean = ar.groupby(["position_encoding", "mode", "count"], as_index=False).ar_accuracy.mean()
    colors = {"nonthinking": BLUE, "thinking": ORANGE}
    fig, axes = subplot_grid(1, len(positions), width=6.4, height=4.4, sharex=True, sharey=True)
    for column, position in enumerate(positions):
        ax = axes[0, column]
        for mode in MODES:
            tf_rows = tf[(tf.position_encoding == position) & (tf["mode"] == mode)].sort_values("count")
            ar_rows = ar_mean[(ar_mean.position_encoding == position) & (ar_mean["mode"] == mode)].sort_values("count")
            ax.plot(tf_rows["count"], tf_rows.tf_final_accuracy, color=colors[mode], linewidth=2.0, label=f"{mode} TF")
            ax.plot(ar_rows["count"], ar_rows.ar_accuracy, color=colors[mode], linewidth=1.7, linestyle="--", label=f"{mode} AR")
        ax.axvline(10.5, color="#94a3b8", linestyle=":")
        ax.axvline(20.5, color="#94a3b8", linestyle=":")
        ax.set_title(position.upper())
        ax.set_xlabel("gold count")
        ax.set_ylabel("final-count accuracy")
        ax.set_ylim(-0.03, 1.04)
        ax.set_xlim(1, 30)
        ax.grid(color=GRID, alpha=0.7)
        ax.legend(frameon=False, loc="lower right", ncol=2)
    fig.suptitle("Final-checkpoint accuracy for every exact count")
    return save_figure(fig, out_dir / "05_final_accuracy_by_count.png")


def plot_count_distribution(distribution: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.8, 4.2), constrained_layout=True)
    colors = [BLUE if c <= 10 else ORANGE if c <= 20 else GREEN for c in distribution["count"]]
    ax.bar(distribution["count"], distribution.probability, color=colors, width=0.82)
    ax.set_xlabel("sampled training count n")
    ax.set_ylabel("probability p(n)")
    ax.set_title("Training count distribution")
    ax.grid(axis="y", color=GRID, alpha=0.7)
    return save_figure(fig, out_dir / "06_count_distribution.png")


def pivot_heads(frame: pd.DataFrame, metric: str) -> np.ndarray:
    return frame.pivot_table(index="layer", columns="head", values=metric, aggfunc="mean").reindex(
        index=[1, 2, 3, 4], columns=[0, 1, 2, 3]
    ).to_numpy(dtype=float)


def plot_broad_attention(attention: pd.DataFrame, positions: tuple[str, ...], out_dir: Path) -> Path:
    fig, axes = subplot_grid(len(positions), 2, width=5.5, height=4.6)
    selected: list[np.ndarray] = []
    for position in positions:
        for mode in MODES:
            rows = attention[
                (attention.position_encoding == position)
                & (attention["mode"] == mode)
                & (attention.query_kind == "final_answer")
                & (attention.count_bin.astype(str) == "all")
            ]
            selected.append(pivot_heads(rows, "broad_attention_score"))
    vmax = max(0.2, max(float(np.nanmax(values)) for values in selected))
    cursor = 0
    for row_index, position in enumerate(positions):
        for column_index, mode in enumerate(MODES):
            ax = axes[row_index, column_index]
            values = selected[cursor]
            cursor += 1
            image = draw_heatmap(
                ax,
                values,
                title=f"{position.upper()} / {mode}",
                xlabels=["H0", "H1", "H2", "H3"],
                ylabels=["L1", "L2", "L3", "L4"],
                vmin=0,
                vmax=vmax,
            )
            ax.set_xlabel("attention head")
            ax.set_ylabel("Transformer layer")
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Broad prompt-needle aggregation score at the final-answer query")
    return save_figure(fig, out_dir / "07_broad_attention.png")


def plot_targeted_attention(attention: pd.DataFrame, positions: tuple[str, ...], out_dir: Path) -> Path:
    fig, axes = subplot_grid(len(positions), 3, width=4.5, height=4.2)
    subsets: list[np.ndarray] = []
    for position in positions:
        for count_bin in COUNT_BINS:
            rows = attention[
                (attention.position_encoding == position)
                & (attention["mode"] == "thinking")
                & (attention.query_kind == "trace_index")
                & (attention.count_bin.astype(str) == count_bin)
            ]
            subsets.append(pivot_heads(rows, "correct_prompt_needle_mass"))
    vmax = max(0.01, max(float(np.nanmax(values)) for values in subsets))
    cursor = 0
    for row_index, position in enumerate(positions):
        for column_index, count_bin in enumerate(COUNT_BINS):
            ax = axes[row_index, column_index]
            values = subsets[cursor]
            cursor += 1
            image = draw_heatmap(
                ax,
                values,
                title=f"{position.upper()} / count {count_bin}",
                xlabels=["H0", "H1", "H2", "H3"],
                ylabels=["L1", "L2", "L3", "L4"],
                vmin=0,
                vmax=vmax,
                digits=3,
            )
            ax.set_xlabel("attention head")
            ax.set_ylabel("Transformer layer")
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Raw k-to-k attention mass from trace index C_k to the kth prompt occurrence")
    return save_figure(fig, out_dir / "08_targeted_attention.png")


def plot_targeted_quality(attention: pd.DataFrame, positions: tuple[str, ...], out_dir: Path) -> Path:
    fig, axes = subplot_grid(len(positions), 2, width=5.7, height=4.5)
    metrics = (
        ("correct_top1", "correct top-1 among prompt occurrences"),
        ("diagonal_dominance", "matching share within occurrence mass"),
    )
    for row_index, position in enumerate(positions):
        for column_index, (metric, title) in enumerate(metrics):
            rows = attention[
                (attention.position_encoding == position)
                & (attention["mode"] == "thinking")
                & (attention.query_kind == "trace_index")
                & (attention.count_bin.astype(str) == "all")
            ]
            values = pivot_heads(rows, metric)
            ax = axes[row_index, column_index]
            image = draw_heatmap(
                ax,
                values,
                title=f"{position.upper()} / {title}",
                xlabels=["H0", "H1", "H2", "H3"],
                ylabels=["L1", "L2", "L3", "L4"],
                vmin=0,
                vmax=1,
            )
            ax.set_xlabel("attention head")
            ax.set_ylabel("Transformer layer")
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Relative retrieval quality must be interpreted together with raw k-to-k mass")
    return save_figure(fig, out_dir / "09_targeted_quality.png")


def plot_trace_readout(attention: pd.DataFrame, positions: tuple[str, ...], out_dir: Path) -> Path:
    fig, axes = subplot_grid(1, len(positions), width=5.7, height=4.5)
    subsets: list[np.ndarray] = []
    for position in positions:
        rows = attention[
            (attention.position_encoding == position)
            & (attention["mode"] == "thinking")
            & (attention.query_kind == "final_answer")
            & (attention.count_bin.astype(str) == "all")
        ]
        subsets.append(pivot_heads(rows, "trace_markers_mass"))
    vmax = max(0.2, max(float(np.nanmax(values)) for values in subsets))
    for column, (position, values) in enumerate(zip(positions, subsets)):
        ax = axes[0, column]
        image = draw_heatmap(
            ax,
            values,
            title=position.upper(),
            xlabels=["H0", "H1", "H2", "H3"],
            ylabels=["L1", "L2", "L3", "L4"],
            vmin=0,
            vmax=vmax,
        )
        ax.set_xlabel("attention head")
        ax.set_ylabel("Transformer layer")
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Final-answer attention mass on all thinking-trace marker tokens")
    return save_figure(fig, out_dir / "10_trace_readout.png")


def plot_state_probe(probes: pd.DataFrame, positions: tuple[str, ...], out_dir: Path) -> Path:
    row_order = [
        "nonthinking | final_answer",
        "thinking | final_answer",
        "thinking | trace_index",
        "thinking | trace_marker",
    ]
    fig, axes = subplot_grid(len(positions), 2, width=6.2, height=4.4)
    for row_index, position in enumerate(positions):
        rows = probes[probes.position_encoding == position].copy()
        rows["row_label"] = rows["mode"] + " | " + rows["site"]
        for column_index, (metric, title, vmin) in enumerate(
            (
                ("ridge_r2", "ridge count R-squared", -0.2),
                ("position_only_accuracy", "position-only baseline accuracy", 0.0),
            )
        ):
            pivot = rows.pivot_table(index="row_label", columns="layer", values=metric, aggfunc="mean").reindex(
                index=row_order, columns=[0, 1, 2, 3, 4]
            )
            values = pivot.to_numpy(dtype=float)
            ax = axes[row_index, column_index]
            image = draw_heatmap(
                ax,
                values,
                title=f"{position.upper()} / {title}",
                xlabels=["embed", "L1", "L2", "L3", "L4"],
                ylabels=row_order,
                vmin=vmin,
                vmax=1,
            )
            ax.set_xlabel("residual state")
            ax.set_ylabel("model mode | semantic site")
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Count decodability and the absolute-position confound")
    return save_figure(fig, out_dir / "11_state_probe.png")


def plot_pca_coverage(variance: pd.DataFrame, positions: tuple[str, ...], out_dir: Path) -> Path:
    rows = variance[variance.component == 6].copy()
    rows["row_label"] = rows["mode"] + " | " + rows["site"]
    row_order = [
        "nonthinking | final_answer",
        "thinking | final_answer",
        "thinking | trace_index",
        "thinking | trace_marker",
    ]
    fig, axes = subplot_grid(1, len(positions), width=6.2, height=4.6)
    for column, position in enumerate(positions):
        pivot = rows[rows.position_encoding == position].pivot_table(
            index="row_label", columns="layer", values="cumulative_explained_variance", aggfunc="mean"
        ).reindex(index=row_order, columns=[0, 1, 2, 3, 4])
        values = pivot.to_numpy(dtype=float)
        ax = axes[0, column]
        image = draw_heatmap(
            ax,
            values,
            title=position.upper(),
            xlabels=["embed", "L1", "L2", "L3", "L4"],
            ylabels=row_order,
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("residual state")
        ax.set_ylabel("model mode | semantic site")
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Cumulative centroid variance captured by PC1-PC6")
    return save_figure(fig, out_dir / "12_pca_coverage.png")


def pca_interactive_html(centroids: pd.DataFrame, positions: tuple[str, ...]) -> str:
    columns = ["position_encoding", "mode", "site", "layer", "state_label"] + [f"pc{i}" for i in range(1, 7)]
    records = centroids[columns].copy()
    records["layer"] = records["layer"].astype(int)
    records["state_label"] = records["state_label"].astype(int)
    payload = records.to_dict(orient="records")
    plotly_path = Path(r"C:\anaconda3\Lib\site-packages\plotly\package_data\plotly.min.js")
    if not plotly_path.exists():
        return '<div class="warning">本机未找到 Plotly JavaScript；静态 PCA 覆盖率图仍可使用。</div>'
    plotly_js = plotly_path.read_text(encoding="utf-8")
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    pos_options = "".join(f'<option value="{esc(position)}">{esc(position.upper())}</option>' for position in positions)
    axis_options = lambda selected: "".join(
        f'<option value="pc{i}" {"selected" if i == selected else ""}>PC{i}</option>' for i in range(1, 7)
    )
    return f"""
    <div class="interactive-card">
      <div class="controls">
        <label>Position encoding<select id="pca-pos">{pos_options}</select></label>
        <label>Model / semantic site<select id="pca-site"></select></label>
        <label>Residual state<select id="pca-layer"><option value="0">Embedding</option><option value="1">After L1</option><option value="2" selected>After L2</option><option value="3">After L3</option><option value="4">After L4</option></select></label>
        <label>X axis<select id="pca-x">{axis_options(1)}</select></label>
        <label>Y axis<select id="pca-y">{axis_options(2)}</select></label>
        <label>Z axis<select id="pca-z">{axis_options(3)}</select></label>
      </div>
      <div id="pca-plot" class="pca-plot"></div>
      <p class="muted">每个点是一个 exact-count hidden-state centroid。颜色和数字为 count 1-30；连线仅帮助观察 count 顺序，不表示单个样本真实沿直线演化。</p>
    </div>
    <script>{plotly_js}</script>
    <script>
      const pcaRows = {data_json};
      const posEl = document.getElementById('pca-pos');
      const siteEl = document.getElementById('pca-site');
      const layerEl = document.getElementById('pca-layer');
      const xEl = document.getElementById('pca-x');
      const yEl = document.getElementById('pca-y');
      const zEl = document.getElementById('pca-z');
      function siteKey(row) {{ return row.mode + ' | ' + row.site; }}
      function refreshSites() {{
        const previous = siteEl.value;
        const sites = [...new Set(pcaRows.filter(r => r.position_encoding === posEl.value).map(siteKey))];
        siteEl.innerHTML = sites.map(s => `<option value="${{s}}">${{s}}</option>`).join('');
        if (sites.includes(previous)) siteEl.value = previous;
      }}
      function renderPca() {{
        refreshSites();
        const rows = pcaRows.filter(r => r.position_encoding === posEl.value && siteKey(r) === siteEl.value && String(r.layer) === layerEl.value).sort((a,b) => a.state_label - b.state_label);
        Plotly.react('pca-plot', [{{
          type: 'scatter3d', mode: 'lines+markers+text',
          x: rows.map(r => r[xEl.value]), y: rows.map(r => r[yEl.value]), z: rows.map(r => r[zEl.value]),
          text: rows.map(r => String(r.state_label)), textposition: 'top center',
          marker: {{size: 5.5, color: rows.map(r => r.state_label), colorscale: 'Viridis', colorbar: {{title: 'count'}}, line: {{width: 0.3, color: '#ffffff'}}}},
          line: {{width: 2, color: '#94a3b8'}}, hovertemplate: 'count=%{{text}}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>'
        }}], {{
          title: `${{posEl.value.toUpperCase()}} | ${{siteEl.value}} | state ${{layerEl.value}}`,
          scene: {{xaxis: {{title: xEl.value.toUpperCase()}}, yaxis: {{title: yEl.value.toUpperCase()}}, zaxis: {{title: zEl.value.toUpperCase()}}}},
          margin: {{l: 0, r: 0, t: 55, b: 0}}, paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff'
        }}, {{responsive: true, displaylogo: false}});
      }}
      [posEl, siteEl, layerEl, xEl, yEl, zEl].forEach(el => el.addEventListener('change', renderPca));
      refreshSites(); renderPca();
    </script>
    """


def final_bin_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tf = final_rows(tables["eval_by_bin"], ["position_encoding", "mode", "count_bin"])
    ar = final_rows(tables["autoregressive_by_bin"], ["position_encoding", "mode", "count_bin"])
    merged = tf.merge(ar, on=["position_encoding", "mode", "count_bin", "step"], how="outer", suffixes=("_tf", "_ar"))
    columns = [
        "position_encoding",
        "mode",
        "count_bin",
        "tf_final_accuracy",
        "ar_final_accuracy",
        "ar_abs_error",
        "tf_trace_index_accuracy",
        "trace_exact",
        "trace_marker_recall",
    ]
    return merged[[column for column in columns if column in merged.columns]]


def final_loss_table(losses: pd.DataFrame) -> pd.DataFrame:
    rows = final_rows(losses, ["position_encoding", "mode"])
    rows = rows.copy()
    rows["eval_perplexity"] = np.exp(rows.eval_total_loss)
    columns = [
        "position_encoding",
        "mode",
        "eval_total_loss",
        "eval_perplexity",
        "eval_prompt_body_loss",
        "eval_final_count_loss",
        "eval_trace_index_loss",
        "eval_trace_marker_loss",
    ]
    return rows[[column for column in columns if column in rows.columns]]


def top_attention_table(attention: pd.DataFrame, metric: str, query_kind: str, mode: str, n: int = 8) -> pd.DataFrame:
    rows = attention[
        (attention["mode"] == mode)
        & (attention.query_kind == query_kind)
        & (attention.count_bin.astype(str) == "all")
    ].nlargest(n, metric)
    columns = ["position_encoding", "layer", "head", metric]
    for extra in ("correct_top1", "diagonal_dominance", "prompt_needles_mass", "needle_entropy_normalized"):
        if extra in rows.columns and extra not in columns:
            columns.append(extra)
    return rows[columns]


def report_css() -> str:
    return """
    :root { --ink:#13213a; --muted:#5f6f86; --line:#d9e3ef; --blue:#2563eb; --green:#16a34a; --orange:#f59e0b; --panel:#f7f9fc; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:#eef3f8; font-family:Inter,"Segoe UI","Microsoft YaHei",Arial,sans-serif; line-height:1.7; }
    main { width:min(1220px,calc(100% - 36px)); margin:24px auto 60px; background:white; border:1px solid var(--line); box-shadow:0 18px 44px rgba(25,50,80,.08); }
    header { padding:48px 58px 38px; background:linear-gradient(135deg,#10213b,#1d4d74); color:white; }
    header h1 { margin:0 0 12px; font-size:clamp(30px,4vw,48px); letter-spacing:0; line-height:1.15; }
    header p { max-width:940px; margin:8px 0 0; color:#d9e8f5; font-size:17px; }
    .content { padding:20px 58px 58px; }
    h2 { margin:44px 0 18px; padding-top:14px; border-top:1px solid var(--line); font-size:29px; letter-spacing:0; }
    h3 { margin:30px 0 12px; font-size:21px; letter-spacing:0; }
    h4 { margin:0 0 14px; font-size:18px; letter-spacing:0; }
    p { margin:10px 0; }
    code { padding:2px 6px; border-radius:4px; background:#eef3f9; color:#173a64; font-family:"Cascadia Mono",Consolas,monospace; font-size:.92em; }
    .lead { font-size:18px; color:#27384f; }
    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin:18px 0; }
    .card { border:1px solid var(--line); background:var(--panel); padding:18px 20px; border-radius:7px; }
    .card h3,.card h4 { margin-top:0; }
    .callout { border-left:5px solid var(--green); background:#effbf3; padding:16px 20px; margin:18px 0; border-radius:5px; }
    .warning { border-left:5px solid var(--orange); background:#fff8e7; padding:16px 20px; margin:18px 0; border-radius:5px; }
    .audit { border-left-color:var(--blue); background:#eef5ff; }
    .report-figure { margin:24px 0 32px; padding:20px; border:1px solid var(--line); border-radius:7px; background:white; }
    .report-figure img { display:block; width:100%; max-height:860px; object-fit:contain; margin:0 auto; }
    figcaption { margin-top:14px; color:var(--muted); font-size:14.5px; }
    .table-wrap { width:100%; overflow-x:auto; margin:16px 0 24px; }
    .data-table { width:100%; border-collapse:collapse; font-size:14px; }
    .data-table th { text-align:left; background:#eaf0f7; color:#14233b; }
    .data-table th,.data-table td { padding:9px 11px; border:1px solid #d7e1ed; white-space:nowrap; }
    .data-table tr:nth-child(even) td { background:#f9fbfd; }
    .formula { margin:12px 0; padding:14px 18px; border:1px solid var(--line); background:#fbfdff; text-align:center; font-family:Georgia,"Times New Roman",serif; font-size:20px; }
    .interactive-card { border:1px solid var(--line); border-radius:7px; padding:18px; margin:22px 0; background:white; }
    .controls { display:grid; grid-template-columns:repeat(3,minmax(170px,1fr)); gap:12px; padding:14px; background:#eef4fa; border-radius:6px; }
    .controls label { display:flex; flex-direction:column; gap:5px; font-size:13px; font-weight:650; }
    select { width:100%; padding:9px; border:1px solid #bccbdd; border-radius:5px; background:white; }
    .pca-plot { width:100%; height:650px; }
    .muted { color:var(--muted); font-size:14px; }
    .tag { display:inline-block; padding:3px 9px; margin:2px 4px 2px 0; background:#e8eff8; border-radius:999px; font-size:13px; }
    .toc { columns:2; padding:12px 18px 12px 36px; background:#f7f9fc; border:1px solid var(--line); }
    .toc a { color:#174c80; text-decoration:none; }
    @media (max-width:850px) { main{width:100%;margin:0;border:0}.content,header{padding-left:22px;padding-right:22px}.grid{grid-template-columns:1fr}.controls{grid-template-columns:1fr}.toc{columns:1}.pca-plot{height:520px} }
    """


def metric_sentence(final_bins: pd.DataFrame, position: str, mode: str) -> str:
    rows = final_bins[(final_bins.position_encoding == position) & (final_bins["mode"] == mode)].copy()
    rows["order"] = rows.count_bin.astype(str).map({name: index for index, name in enumerate(COUNT_BINS)})
    rows = rows.sort_values("order")
    tf = "/".join(pct(value) for value in rows.tf_final_accuracy)
    ar = "/".join(pct(value) for value in rows.ar_final_accuracy)
    return f"{position.upper()} {mode}: TF {tf}，AR {ar}（依次为 1-10 / 11-20 / 21-30）"


def build_report(
    run_dir: Path,
    out_dir: Path,
    *,
    version: str,
    duplicate_dir: Path | None = None,
) -> Path:
    set_plot_style()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    positions = tuple(config["position_encodings"])
    audit, config, manifest = audit_bundle(run_dir, positions)
    tables = load_tables(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = out_dir / "assets"
    assets.mkdir(exist_ok=True)

    all_sequence = config["loss_scope"] == "all_sequence"
    plots = {
        "training": plot_training_loss(tables["train_metrics"], positions, assets, all_sequence),
        "segments": plot_segment_losses(tables["eval_losses"], positions, assets, all_sequence),
        "final_learning": plot_final_learning(tables["eval_by_bin"], tables["autoregressive_by_bin"], positions, assets),
        "trace_learning": plot_trace_learning(tables["eval_by_bin"], tables["autoregressive_by_bin"], positions, assets),
        "by_count": plot_final_by_count(tables["eval_by_count"], tables["autoregressive_detail"], positions, assets),
        "distribution": plot_count_distribution(tables["training_count_distribution"], assets),
        "broad": plot_broad_attention(tables["attention_summary"], positions, assets),
        "targeted": plot_targeted_attention(tables["attention_summary"], positions, assets),
        "targeted_quality": plot_targeted_quality(tables["attention_summary"], positions, assets),
        "readout": plot_trace_readout(tables["attention_summary"], positions, assets),
        "probe": plot_state_probe(tables["state_probe_summary"], positions, assets),
        "pca": plot_pca_coverage(tables["state_pca_variance"], positions, assets),
    }

    final_bins = final_bin_table(tables)
    losses = final_loss_table(tables["eval_losses"])
    time99 = tables["time_to_99"]
    reached = time99[(time99.metric == "tf_final_accuracy")][
        ["position_encoding", "mode", "count_bin", "first_step_at_threshold", "reached_threshold"]
    ]
    broad = top_attention_table(tables["attention_summary"], "broad_attention_score", "final_answer", "nonthinking")
    targeted = top_attention_table(tables["attention_summary"], "correct_prompt_needle_mass", "trace_index", "thinking")
    readout = top_attention_table(tables["attention_summary"], "trace_markers_mass", "final_answer", "thinking")
    probes = tables["state_probe_summary"].copy()
    best_probes = probes.sort_values(["nearest_centroid_accuracy", "ridge_r2"], ascending=False).head(12)
    best_probes = best_probes[
        ["position_encoding", "mode", "site", "layer", "nearest_centroid_accuracy", "position_only_accuracy", "ridge_r2", "ridge_mae"]
    ]

    if version == "v16":
        title = "v16: Tiny Shakespeare 原生字符计数与 RoPE/RPE 对比"
        subtitle = "四个独立 Transformer；全序列 next-token loss；自然字符 haystack；描述性 attention 与 count-state geometry"
        task_description = "在标准 Tiny Shakespeare 的长度 256 字符窗口中，指定目标字符 S/H/A/K/E/R 或对应小写，要求模型输出该字符出现次数。"
        sequence_nt = "&lt;BOS&gt; &lt;CountChar&gt; target &lt;Sep&gt; prompt[256 chars] &lt;Ans&gt; C_n &lt;EOS&gt;"
        sequence_th = "&lt;BOS&gt; &lt;CountChar&gt; target &lt;Sep&gt; prompt &lt;Think&gt; C_1 char ... C_n char &lt;/Think&gt; &lt;Ans&gt; C_n &lt;EOS&gt;"
        duplicate_note = "不适用：v16 只有一份完整导出。"
        main_findings = (
            "CoT 在 RoPE/RPE 下都达到 100% teacher-forced final accuracy；自由生成仍在低 count 只有 74%。"
            "Non-thinking 未完全收敛，RoPE 明显优于 RPE，尤其 count 1-10（AR 66% 对 39%）。"
            "本任务的 k-to-k raw attention 很弱，说明自然字符计数没有复现 v2 插入 marker 的强逐项检索头。"
        )
        objective_detail = (
            "loss 覆盖 task prefix、256 个 Shakespeare prompt 字符和 completion；因此 total CE 主要受 prompt language modeling 支配，"
            "不能用 total CE 单独判断计数是否学会。"
        )
    else:
        title = "v17: RoPE completion-only counting with power-distributed counts"
        subtitle = "两个独立 Transformer；插入 marker；p(n) ∝ 1/n；学习动态、attention 与 count-state geometry"
        task_description = "长度 256 的均匀 noise 序列中插入 1-30 个 marker，模型输出 marker 数量；训练 count 按 p(n) ∝ 1/n 采样。"
        sequence_nt = "&lt;BOS&gt; prompt[256 tokens] &lt;Ans&gt; C_n &lt;EOS&gt;"
        sequence_th = "&lt;BOS&gt; prompt &lt;Think&gt; C_1 M_1 ... C_n M_n &lt;/Think&gt; &lt;Ans&gt; C_n &lt;EOS&gt;"
        same, comparison = compare_v17_exports(run_dir, duplicate_dir)
        duplicate_note = ("重复导出确认：" if same else "导出比较：") + comparison
        main_findings = (
            "Non-thinking 在三个 count 区间都达到 100% TF/AR；CoT teacher-forced 为 91.1%/99.4%/100%，"
            "AR 为 84%/95%/100%。低 count 反而最难，主要来自 trace 终止与短轨迹边界，而非样本频率不足。"
            "模型仍形成 broad aggregation 与 trace readout heads，但 k-to-k raw mass 比 v2 弱。"
        )
        objective_detail = (
            "prompt labels 全部 mask；non-thinking 只监督 final count 与 EOS，thinking 从第一个 trace token 一直监督到 EOS。"
            "因此 total CE 是 completion-token 平均 CE，两个 mode 因 completion 长度不同不可机械横比。"
        )

    settings = pd.DataFrame(
        [
            ("task", task_description),
            ("models", f"{len(positions)} position encodings × 2 modes = {2 * len(positions)} independent Transformers"),
            ("architecture", config["architecture"]),
            ("sequence length / count", f"prompt={config['seq_len']}; count={config['count_min']}-{config['count_max']}; context={config['n_positions']}"),
            ("training objective", config["training_objective"]),
            ("optimizer", config["optimizer"]),
            ("steps / batch / LR", f"{config['train_steps']} / {config['batch_size']} / {config['lr']}"),
            ("count sampling", config["count_sampling_definition"]),
            ("checkpointing", "step 1000-10000, every 1000 steps, plus final"),
            ("seed / precision", f"{config['seed']} / {config['precision_definition']}"),
        ],
        columns=["item", "setting"],
    )

    audit_summary = audit[audit.artifact.str.startswith("stage") | audit.artifact.str.startswith("checkpoint")]
    metric_lines = "<br>".join(metric_sentence(final_bins, position, mode) for position in positions for mode in MODES)
    perplexities = ", ".join(
        f"{row.position_encoding.upper()}/{row.mode}: CE={row.eval_total_loss:.3f}, PPL={row.eval_perplexity:.3f}"
        for row in losses.itertuples(index=False)
    )

    body = f"""
    <!doctype html>
    <html lang="zh-CN">
    <head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{report_css()}</style></head>
    <body><main>
      <header><h1>{esc(title)}</h1><p>{esc(subtitle)}</p><p>Canonical result bundle: <code>{esc(run_dir)}</code></p></header>
      <div class="content">
        <ol class="toc">
          <li><a href="#s1">研究对象与核心问题</a></li><li><a href="#s2">实验设置与完整性</a></li>
          <li><a href="#s3">术语与计算定义</a></li><li><a href="#s4">学习动态与行为结果</a></li>
          <li><a href="#s5">描述性 attention</a></li><li><a href="#s6">Hidden-state geometry</a></li>
        </ol>

        <section id="s1"><h2>1. 研究对象与核心问题</h2>
          <p class="lead">本报告比较两个从零训练、参数架构相同但输出格式不同的 causal Transformer。研究问题不是只看“谁更准”，而是观察显式 thinking trace 是否改变了计数的计算路径、attention routing 和 hidden-state 表征。</p>
          <div class="grid">
            <div class="card"><h3>Non-thinking / direct count</h3><p><code>{sequence_nt}</code></p><p>机制假设：最终答案位置通过一组对 prompt needle/目标字符集合进行 broad aggregation 的 heads，直接把集合统计写入 final-answer residual state。</p></div>
            <div class="card"><h3>Thinking / trace-mediated count</h3><p><code>{sequence_th}</code></p><p>机制假设：trace 逐步显式表示 count progress；中间 heads 负责检索或延续 trace，最终答案位置再从 trace 读取 scalar count。</p></div>
          </div>
          <div class="callout"><strong>本次最重要的观察：</strong>{esc(main_findings)}</div>
          <p class="warning"><strong>证据边界：</strong>本报告第 5-6 节是描述性 attention / probe / PCA，不把高 attention mass 或高线性可读性直接等同于因果必要性。要证明机制，需要后续 position-local ablation、activation patching 或 geometry steering。</p>
        </section>

        <section id="s2"><h2>2. 实验设置与结果完整性</h2>
          <div class="table-wrap">{table_html(settings)}</div>
          <h3>结果包审计</h3><div class="table-wrap">{table_html(audit_summary)}</div>
          <div class="callout audit"><strong>完整性结论：</strong>train、attention、state、plots 四阶段均标记 complete；每个模型有最终 checkpoint 和 10 个千步 checkpoint。{esc(duplicate_note)}</div>
          <p>{objective_detail}</p>
        </section>

        <section id="s3"><h2>3. 术语、数据列与计算定义</h2>
          <h3>3.1 行为指标</h3>
          <div class="grid">
            <div class="card"><h4>Teacher-forced final accuracy</h4><div class="formula">1[argmax p(C | gold prefix) = C<sub>gold</sub>]</div><p>给定真实 prompt 与此前真实 completion token，只检查最终 count token。它隔离局部读出能力，但不反映前面自由生成出错后的级联。</p></div>
            <div class="card"><h4>Autoregressive final accuracy</h4><div class="formula">1[generated final count = gold count]</div><p>模型从 prompt 开始自由生成完整 completion 后，解析最终 count。它包含 trace 错误、终止错误与最终读出错误。</p></div>
            <div class="card"><h4>Cross-entropy and perplexity</h4><div class="formula">CE = -(1/N) Σ log p(x<sub>t</sub>|x&lt;t), &nbsp; PPL = exp(CE)</div><p>N 只包括未 mask 的监督 token。PPL 是平均每个 token 的有效候选分支数；不同 loss scope 下不能直接比较。</p></div>
            <div class="card"><h4>Trace exact / marker recall</h4><div class="formula">trace exact = 1[generated trace = gold trace]</div><p>trace exact 要求整个 trace 完全一致；marker recall 是生成 trace 中正确 marker 的召回率，允许其他 token 出错。</p></div>
          </div>
          <h3>3.2 Attention scores</h3>
          <div class="grid">
            <div class="card"><h4>Broad aggregation score</h4><div class="formula">B = M<sub>needles</sub> × H<sub>normalized</sub></div><p>M 是 final-answer query 投向全部 prompt occurrences 的 attention mass；H 是该 mass 在 occurrences 间分布的归一化熵。B 同时奖励“看得多”和“分得广”。</p></div>
            <div class="card"><h4>Raw k-to-k mass</h4><div class="formula">K = A[q(C<sub>k</sub>), occurrence<sub>k</sub>]</div><p>thinking trace 的 C_k query 直接投向 prompt 中按位置排序的第 k 个目标 occurrence 的 attention 权重。</p></div>
            <div class="card"><h4>Correct top-1</h4><div class="formula">1[argmax<sub>j in occurrences</sub> A(q,j) = occurrence<sub>k</sub>]</div><p>只在目标 occurrence 子集内比较最大值；不要求该 occurrence 获得整个上下文的最大 attention。</p></div>
            <div class="card"><h4>Diagonal dominance</h4><div class="formula">D = A[q(C<sub>k</sub>), occurrence<sub>k</sub>] / M<sub>occurrences</sub></div><p>匹配 occurrence 占全部 occurrence mass 的比例。D 可以高而 raw K 很低，因此必须与 K 联合解释。</p></div>
          </div>
          <h3>3.3 Hidden-state metrics</h3>
          <p><strong>Residual state：</strong>embedding state（layer 0）或经过第 1-4 层 Transformer 后，在指定 token 位置的 d_model 维 residual vector。<strong>Nearest-centroid accuracy</strong> 用训练集各 count 的均值向量做类别中心；测试向量分配给欧氏距离最近的中心。<strong>Ridge R-squared</strong> 把 count 当连续数拟合；越接近 1 表示近似线性可读。<strong>Position-only baseline</strong> 只用绝对 token 位置预测 count，用于识别“trace 长度直接泄露 count”的混淆。</p>
        </section>

        <section id="s4"><h2>4. 学习动态与行为结果</h2>
          {figure(plots['training'], 'Figure 1. Training objective', '横轴为 optimizer step；纵轴为当前 loss scope 上的平均 next-token cross-entropy。v16 是全序列 loss，包含 prompt；v17 是 completion-only loss。')}
          {figure(plots['segments'], 'Figure 2. Validation CE by semantic segment', '横轴为训练步；纵轴为各语义片段的 teacher-forced CE（对数刻度）。分段 loss 回答模型是在学 prompt 建模、trace index、trace marker，还是最终 count。')}
          <div class="callout"><strong>最终 total CE / perplexity：</strong>{esc(perplexities)}。PPL=exp(CE)；低 PPL 只说明受监督 token 易预测，不自动等于自由生成计数正确。</div>
          {figure(plots['distribution'], 'Figure 3. Training count distribution', '横轴为训练时抽到的 gold count n；纵轴为 p(n)。颜色区分 1-10、11-20、21-30。评估仍对每个 exact count 平衡采样，因此测试结果不会被训练频率直接加权。')}
          {figure(plots['final_learning'], 'Figure 4. Final-count learning dynamics', '每个子图对应一个 position encoding 和 mode。横轴为训练步，纵轴为 final-count accuracy；实线是 teacher-forced，虚线是 autoregressive。颜色区分三个 count 区间。')}
          {figure(plots['trace_learning'], 'Figure 5. Thinking-trace learning dynamics', '左列为给定 gold prefix 时下一个 trace index 的准确率；右列为完整自由生成 trace 的 exact match。两者差距量化局部预测能力与序列级错误累积。')}
          {figure(plots['by_count'], 'Figure 6. Final accuracy by exact count', '横轴为 gold count 1-30，纵轴为最终 count 准确率；实线 TF、虚线 AR。竖虚线分隔三个预注册 count bins。')}
          <h3>最终行为表</h3><div class="table-wrap">{table_html(final_bins)}</div>
          <h3>达到 99% teacher-forced final accuracy 的时间</h3><div class="table-wrap">{table_html(reached)}</div>
          <div class="callout"><strong>读法：</strong>{metric_lines}</div>
        </section>

        <section id="s5"><h2>5. 描述性 attention：broad aggregation、targeted retrieval 与 trace readout</h2>
          {figure(plots['broad'], 'Figure 7. Broad prompt-occurrence aggregation', '横轴为 head 0-3，纵轴为 Transformer layer 1-4；单元格为 B=M_occurrences×normalized entropy，在 final-answer query 计算。高值表示 attention 既落在 occurrence 集合上，又在多个 occurrence 间广泛分布。')}
          <h3>最高 broad candidates</h3><div class="table-wrap">{table_html(broad)}</div>
          {figure(plots['targeted'], 'Figure 8. Raw k-to-k targeted mass by count bin', '横轴为 head，纵轴为 layer；单元格是 trace index C_k 对 prompt 中第 k 个目标 occurrence 的 raw attention mass，并对样本/k 平均。各行 position encoding、各列 count bin。')}
          {figure(plots['targeted_quality'], 'Figure 9. Relative targeted-retrieval quality', '左侧 correct top-1 只问在 occurrence 子集内最大 attention 是否落到第 k 个；右侧 diagonal dominance 是匹配 occurrence 占全部 occurrence mass 的比例。两者都必须与 Figure 8 的 raw mass 联合解释。')}
          <h3>最高 raw k-to-k candidates</h3><div class="table-wrap">{table_html(targeted)}</div>
          {figure(plots['readout'], 'Figure 10. CoT final-answer trace readout', '横轴为 head，纵轴为 layer；单元格是最终答案 query 投向全部 trace marker token 的总 attention mass。它寻找从已经给出的 trace 读取 count 的候选 heads，不等于 prompt retrieval。')}
          <h3>最高 trace-readout candidates</h3><div class="table-wrap">{table_html(readout)}</div>
          <div class="warning"><strong>解释限制：</strong>attention mass 是路由描述，不是因果贡献。尤其 v16 中每次 occurrence 都是同一个目标字符 token，marker identity 本身无法区分第 k 项；顺序只能由位置与 trace progress 提供。其极低 raw k-to-k mass 表明模型没有形成 v2 那种强、稀疏的逐项 retrieval signature。</div>
        </section>

        <section id="s6"><h2>6. Hidden-state count geometry</h2>
          {figure(plots['probe'], 'Figure 11. Count decodability and position confound', '横轴为 embedding 与经过 Layer 1-4 后的 residual state；纵轴为 mode 与语义位置。左侧 ridge R-squared 测线性 count 可读性；右侧 position-only baseline 只用 token 绝对位置预测 count。若两者同时接近 1，不能把可读性全归因于内容计算。')}
          <h3>最高 held-out count probes</h3><div class="table-wrap">{table_html(best_probes)}</div>
          {figure(plots['pca'], 'Figure 12. Variance captured by the first six centroid PCs', '横轴为 residual state，纵轴为 mode/site；单元格为 exact-count centroid 矩阵前六个主成分的累计 explained-variance ratio。它描述 count 类均值几何的低维程度，不描述单样本全部方差。')}
          {pca_interactive_html(tables['state_centroids_pca'], positions)}
          <div class="warning"><strong>Position confound：</strong>thinking 的 final-answer 与 trace token 位置随 trace 长度/count 改变，所以 position-only accuracy 常接近 100%。这类 probe 证明“状态可读”，但还不能证明 residual 中存在独立、可搬运的 scalar count；需要固定位置、截断 trace、state patching 或 geometry steering。</div>
        </section>

        <section><h2>结论与后续实验</h2>
          <div class="grid"><div class="card"><h3>当前支持</h3><p>{esc(main_findings)}</p></div><div class="card"><h3>仍未证明</h3><p>哪些 attention heads 对行为必要；count-state 是否可独立于 token 位置搬运；broad routing、trace retrieval 与最终 residual geometry 之间是否存在直接因果链。</p></div></div>
          <p>建议下一步优先做：在同一 count bin 内的 position-local head ablation；clean-to-corrupt head-output patch；固定 trace 长度的 hidden-state patch；以及沿 count centroid direction 的 dose-response steering。</p>
        </section>
      </div>
    </main></body></html>
    """
    report_name = "syn_v16_report.html" if version == "v16" else "syn_v17_report.html"
    report_path = out_dir / report_name
    report_path.write_text(body, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build self-contained v16 and v17 reports")
    parser.add_argument("--v16-dir", type=Path, required=True)
    parser.add_argument("--v17-dir", type=Path, required=True)
    parser.add_argument("--v17-duplicate-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    v16 = build_report(args.v16_dir, args.output_root / "v16", version="v16")
    v17 = build_report(
        args.v17_dir,
        args.output_root / "v17",
        version="v17",
        duplicate_dir=args.v17_duplicate_dir,
    )
    print(v16.resolve())
    print(v17.resolve())


if __name__ == "__main__":
    main()
