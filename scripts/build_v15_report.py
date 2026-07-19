from __future__ import annotations

import argparse
import base64
import html
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#16a34a"
PURPLE = "#7c3aed"
RED = "#dc2626"
GRAY = "#64748b"
INK = "#172033"
GRID = "#d7deea"
COUNT_BINS = ("1-10", "11-20", "21-30")
POSITIONS = ("rope", "rpe")
MODES = ("nonthinking", "thinking")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def final_rows(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    maximum = frame.groupby(keys)["step"].transform("max")
    return frame[frame["step"] == maximum].copy()


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    if number != 0 and abs(number) < 10 ** (-digits):
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{100 * number:.{digits}f}%"


def esc(value: Any) -> str:
    return html.escape(str(value))


def table_html(frame: pd.DataFrame, *, float_format: str = "{:.3f}") -> str:
    if frame.empty:
        return '<p class="muted">无可用数据。</p>'
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "NA" if pd.isna(value) else float_format.format(value)
            )
    return display.to_html(index=False, escape=False, classes="data-table", border=0)


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def figure(path: Path, title: str, caption: str, *, wide: bool = True) -> str:
    if not path.exists():
        return f'<div class="warning">缺少图片：{esc(path.name)}</div>'
    classes = "figure wide" if wide else "figure"
    return f"""
    <figure class="{classes}">
      <h4>{esc(title)}</h4>
      <img src="{image_uri(path)}" alt="{esc(title)}">
      <figcaption>{caption}</figcaption>
    </figure>
    """


def save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#94a3b8",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.titlesize": 17,
        }
    )


def draw_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    xlabels: list[str],
    ylabels: list[str],
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
    annotate: bool = True,
) -> Any:
    image = ax.imshow(values, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, pad=10)
    ax.set_xticks(np.arange(len(xlabels)), labels=xlabels)
    ax.set_yticks(np.arange(len(ylabels)), labels=ylabels)
    if annotate:
        midpoint = (vmin + vmax) / 2
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if not np.isfinite(value):
                    continue
                color = "white" if value < midpoint else "black"
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)
    return image


def audit_bundle(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    config_path = run_dir / "config.json"
    manifest_path = run_dir / "manifest.json"
    if not config_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("The result bundle must contain config.json and manifest.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_tables = (
        "train_metrics",
        "eval_losses",
        "eval_by_bin",
        "eval_by_count",
        "autoregressive_by_bin",
        "autoregressive_detail",
        "time_to_99",
        "attention_summary",
        "state_probe_summary",
        "state_pca_variance",
        "state_centroids_pca",
        "model_specifications",
    )
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in required_tables:
        path = run_dir / "tables" / f"{name}.csv"
        okay = path.exists() and path.stat().st_size > 0
        if not okay:
            missing.append(str(path))
        rows.append(
            {
                "artifact": f"tables/{name}.csv",
                "status": "complete" if okay else "missing",
                "rows": len(read_csv(path)) if okay else 0,
            }
        )
    for position in POSITIONS:
        for mode in MODES:
            root = run_dir / "checkpoints" / position / mode
            final_ok = (root / "final").exists()
            snapshots = len(list(root.glob("step_*"))) if root.exists() else 0
            rows.append(
                {
                    "artifact": f"checkpoint {position}/{mode}",
                    "status": "complete" if final_ok and snapshots == 10 else "incomplete",
                    "rows": f"final + {snapshots} snapshots",
                }
            )
            if not final_ok or snapshots != 10:
                missing.append(str(root))
    for stage in ("train", "attention", "state", "plots"):
        status = manifest.get("stages", {}).get(stage, {}).get("status", "missing")
        rows.append({"artifact": f"stage {stage}", "status": status, "rows": "-"})
        if status != "complete":
            missing.append(f"manifest stage {stage}")
    if missing:
        raise RuntimeError("Incomplete v15 bundle:\n" + "\n".join(missing))
    return pd.DataFrame(rows), config, manifest


def plot_training_loss(train: pd.DataFrame, out_dir: Path) -> Path:
    colors = {"nonthinking": BLUE, "thinking": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.7), sharey=True, constrained_layout=True)
    for ax, position in zip(axes, POSITIONS):
        for mode in MODES:
            rows = train[(train.position_encoding == position) & (train["mode"] == mode)].sort_values("step")
            ax.plot(rows.step, rows.train_total_loss, color=colors[mode], linewidth=2.1, label=mode)
        ax.set_title(position.upper())
        ax.set_xlabel("training step")
        ax.grid(color=GRID, alpha=0.72)
        ax.set_xlim(0, train.step.max())
    axes[0].set_ylabel("all-sequence next-token cross-entropy")
    axes[1].legend(title="model mode", frameon=False)
    fig.suptitle("Training loss over prompt + completion tokens")
    return save_figure(fig, out_dir / "01_training_loss.png")


def plot_segment_losses(losses: pd.DataFrame, out_dir: Path) -> Path:
    colors = {
        "eval_prompt_body_loss": GRAY,
        "eval_final_count_loss": BLUE,
        "eval_trace_marker_loss": GREEN,
        "eval_trace_index_loss": PURPLE,
    }
    labels = {
        "eval_prompt_body_loss": "prompt body",
        "eval_final_count_loss": "final count",
        "eval_trace_marker_loss": "trace marker",
        "eval_trace_index_loss": "trace index",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.5), sharex=True, constrained_layout=True)
    for row_index, position in enumerate(POSITIONS):
        for column_index, mode in enumerate(MODES):
            ax = axes[row_index, column_index]
            rows = losses[(losses.position_encoding == position) & (losses["mode"] == mode)].sort_values("step")
            metrics = ["eval_prompt_body_loss", "eval_final_count_loss"]
            if mode == "thinking":
                metrics += ["eval_trace_marker_loss", "eval_trace_index_loss"]
            for metric in metrics:
                if metric not in rows or rows[metric].isna().all():
                    continue
                ax.plot(rows.step, rows[metric], color=colors[metric], linewidth=2, label=labels[metric])
            ax.set_yscale("log")
            ax.set_title(f"{position.upper()} / {mode}")
            ax.set_xlabel("training step")
            ax.set_ylabel("teacher-forced segment CE (log scale)")
            ax.grid(color=GRID, alpha=0.65, which="both")
            if ax.lines:
                ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Validation loss decomposed by semantic sequence segment")
    return save_figure(fig, out_dir / "02_segment_losses.png")


def plot_final_learning(eval_bins: pd.DataFrame, ar_bins: pd.DataFrame, out_dir: Path) -> Path:
    colors = {"1-10": BLUE, "11-20": ORANGE, "21-30": GREEN}
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.0), sharex=True, sharey=True, constrained_layout=True)
    for row_index, position in enumerate(POSITIONS):
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
                ax.plot(tf.step, tf.tf_final_accuracy, color=colors[count_bin], linewidth=2.1, label=count_bin)
                ax.plot(ar.step, ar.ar_final_accuracy, color=colors[count_bin], linewidth=1.7, linestyle="--")
            ax.set_title(f"{position.upper()} / {mode}")
            ax.set_xlabel("training step")
            ax.set_ylabel("final-count accuracy")
            ax.set_ylim(-0.03, 1.04)
            ax.grid(color=GRID, alpha=0.7)
    axes[0, 1].legend(title="gold count bin", frameon=False, loc="lower right")
    fig.suptitle("Final-count learning dynamics: teacher-forced (solid) vs autoregressive (dashed)")
    return save_figure(fig, out_dir / "03_final_count_learning.png")


def plot_trace_learning(eval_bins: pd.DataFrame, ar_bins: pd.DataFrame, out_dir: Path) -> Path:
    colors = {"1-10": BLUE, "11-20": ORANGE, "21-30": GREEN}
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.0), sharex=True, sharey=True, constrained_layout=True)
    for row_index, position in enumerate(POSITIONS):
        for column_index, metric in enumerate(("tf_trace_marker_accuracy", "trace_marker_recall")):
            ax = axes[row_index, column_index]
            for count_bin in COUNT_BINS:
                source = eval_bins if metric.startswith("tf_") else ar_bins
                rows = source[
                    (source.position_encoding == position)
                    & (source["mode"] == "thinking")
                    & (source.count_bin.astype(str) == count_bin)
                ].sort_values("step")
                ax.plot(rows.step, rows[metric], color=colors[count_bin], linewidth=2.1, label=count_bin)
            title = "teacher-forced next-marker accuracy" if metric.startswith("tf_") else "autoregressive marker recall"
            ax.set_title(f"{position.upper()} / {title}")
            ax.set_xlabel("training step")
            ax.set_ylabel("trace-marker metric")
            ax.set_ylim(-0.03, 1.04)
            ax.grid(color=GRID, alpha=0.7)
    axes[0, 1].legend(title="gold count bin", frameon=False, loc="lower right")
    fig.suptitle("Thinking-trace learning remains harder than final-count readout")
    return save_figure(fig, out_dir / "04_trace_learning.png")


def plot_final_by_count(eval_count: pd.DataFrame, ar_detail: pd.DataFrame, out_dir: Path) -> Path:
    tf = final_rows(eval_count, ["position_encoding", "mode", "count"])
    ar = final_rows(ar_detail, ["position_encoding", "mode", "count"])
    ar_mean = ar.groupby(["position_encoding", "mode", "count"], as_index=False).ar_accuracy.mean()
    colors = {"nonthinking": BLUE, "thinking": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), sharex=True, sharey=True, constrained_layout=True)
    for ax, position in zip(axes, POSITIONS):
        for mode in MODES:
            tf_rows = tf[(tf.position_encoding == position) & (tf["mode"] == mode)].sort_values("count")
            ar_rows = ar_mean[(ar_mean.position_encoding == position) & (ar_mean["mode"] == mode)].sort_values("count")
            ax.plot(tf_rows["count"], tf_rows.tf_final_accuracy, color=colors[mode], linewidth=2.1, label=f"{mode} TF")
            ax.plot(ar_rows["count"], ar_rows.ar_accuracy, color=colors[mode], linewidth=1.7, linestyle="--", label=f"{mode} AR")
        ax.axvline(10.5, color="#94a3b8", linestyle=":")
        ax.axvline(20.5, color="#94a3b8", linestyle=":")
        ax.set_title(position.upper())
        ax.set_xlabel("gold needle count")
        ax.set_ylabel("final-count accuracy")
        ax.set_ylim(-0.03, 1.04)
        ax.set_xlim(1, 30)
        ax.grid(color=GRID, alpha=0.7)
    axes[1].legend(frameon=False, loc="lower left", ncol=2)
    fig.suptitle("Final checkpoint accuracy for every exact count")
    return save_figure(fig, out_dir / "05_final_accuracy_by_count.png")


def pivot_heads(frame: pd.DataFrame, metric: str) -> np.ndarray:
    return frame.pivot_table(index="layer", columns="head", values=metric, aggfunc="mean").reindex(
        index=[1, 2, 3, 4], columns=[0, 1, 2, 3]
    ).to_numpy(dtype=float)


def plot_broad_attention(attention: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0), constrained_layout=True)
    for row_index, position in enumerate(POSITIONS):
        for column_index, mode in enumerate(MODES):
            rows = attention[
                (attention.position_encoding == position)
                & (attention["mode"] == mode)
                & (attention.query_kind == "final_answer")
                & (attention.count_bin.astype(str) == "all")
            ]
            values = pivot_heads(rows, "broad_attention_score")
            ax = axes[row_index, column_index]
            image = draw_heatmap(
                ax,
                values,
                title=f"{position.upper()} / {mode}",
                xlabels=["H0", "H1", "H2", "H3"],
                ylabels=["L1", "L2", "L3", "L4"],
                vmin=0,
                vmax=max(0.36, float(np.nanmax(values))),
            )
            ax.set_xlabel("attention head")
            ax.set_ylabel("Transformer layer")
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Broad prompt-needle aggregation score at the final-answer query")
    return save_figure(fig, out_dir / "06_broad_attention.png")


def plot_targeted_attention(attention: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.5), constrained_layout=True)
    for row_index, position in enumerate(POSITIONS):
        for column_index, count_bin in enumerate(COUNT_BINS):
            rows = attention[
                (attention.position_encoding == position)
                & (attention["mode"] == "thinking")
                & (attention.query_kind == "trace_index")
                & (attention.count_bin.astype(str) == count_bin)
            ]
            values = pivot_heads(rows, "correct_prompt_needle_mass")
            ax = axes[row_index, column_index]
            image = draw_heatmap(
                ax,
                values,
                title=f"{position.upper()} / count {count_bin}",
                xlabels=["H0", "H1", "H2", "H3"],
                ylabels=["L1", "L2", "L3", "L4"],
                vmin=0,
                vmax=0.65,
            )
            ax.set_xlabel("attention head")
            ax.set_ylabel("Transformer layer")
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Raw k-to-k targeted-retrieval mass in the thinking trace")
    return save_figure(fig, out_dir / "07_targeted_attention.png")


def plot_targeted_quality(attention: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.0), constrained_layout=True)
    metrics = (("correct_top1", "Correct top-1 among prompt needles"), ("diagonal_dominance", "Diagonal dominance within needle mass"))
    for row_index, position in enumerate(POSITIONS):
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
    fig.suptitle("Relative targeted-retrieval quality is not the same as raw mass")
    return save_figure(fig, out_dir / "08_targeted_quality.png")


def plot_trace_readout(attention: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    for ax, position in zip(axes, POSITIONS):
        rows = attention[
            (attention.position_encoding == position)
            & (attention["mode"] == "thinking")
            & (attention.query_kind == "final_answer")
            & (attention.count_bin.astype(str) == "all")
        ]
        values = pivot_heads(rows, "trace_markers_mass")
        image = draw_heatmap(
            ax,
            values,
            title=position.upper(),
            xlabels=["H0", "H1", "H2", "H3"],
            ylabels=["L1", "L2", "L3", "L4"],
            vmin=0,
            vmax=0.6,
        )
        ax.set_xlabel("attention head")
        ax.set_ylabel("Transformer layer")
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Final-answer attention mass on all thinking-trace marker tokens")
    return save_figure(fig, out_dir / "09_trace_readout.png")


def plot_state_probe(probes: pd.DataFrame, out_dir: Path) -> Path:
    row_order = [
        "nonthinking | final_answer",
        "thinking | final_answer",
        "thinking | trace_index",
        "thinking | trace_marker",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6), constrained_layout=True)
    for row_index, position in enumerate(POSITIONS):
        rows = probes[probes.position_encoding == position].copy()
        rows["row_label"] = rows["mode"] + " | " + rows["site"]
        for column_index, (metric, title, vmin) in enumerate(
            (("ridge_r2", "ridge count R²", -0.2), ("position_only_accuracy", "position-only baseline accuracy", 0.0))
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
    return save_figure(fig, out_dir / "10_state_probe.png")


def plot_pca_coverage(variance: pd.DataFrame, out_dir: Path) -> Path:
    rows = variance[variance.component == 6].copy()
    rows["row_label"] = rows["mode"] + " | " + rows["site"]
    row_order = [
        "nonthinking | final_answer",
        "thinking | final_answer",
        "thinking | trace_index",
        "thinking | trace_marker",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    for ax, position in zip(axes, POSITIONS):
        pivot = rows[rows.position_encoding == position].pivot_table(
            index="row_label", columns="layer", values="cumulative_explained_variance", aggfunc="mean"
        ).reindex(index=row_order, columns=[0, 1, 2, 3, 4])
        values = pivot.to_numpy(dtype=float)
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
    fig.suptitle("Cumulative variance captured by the first six count-centroid PCs")
    return save_figure(fig, out_dir / "11_pca_coverage.png")


def pca_interactive_html(centroids: pd.DataFrame) -> str:
    columns = ["position_encoding", "mode", "site", "layer", "state_label"] + [f"pc{i}" for i in range(1, 7)]
    records = centroids[columns].copy()
    records["layer"] = records["layer"].astype(int)
    records["state_label"] = records["state_label"].astype(int)
    payload = records.to_dict(orient="records")
    plotly_path = Path(r"C:\anaconda3\Lib\site-packages\plotly\package_data\plotly.min.js")
    if not plotly_path.exists():
        try:
            import plotly

            plotly_path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        except Exception:
            plotly_path = Path()
    if not plotly_path.exists():
        return '<div class="warning">本机未找到 Plotly JavaScript；静态 PCA 图仍可使用。</div>'
    plotly_js = plotly_path.read_text(encoding="utf-8")
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""
    <div class="interactive-card">
      <div class="controls">
        <label>Position encoding<select id="pca-pos"><option value="rope">RoPE</option><option value="rpe">RPE</option></select></label>
        <label>Model / semantic site<select id="pca-site"></select></label>
        <label>Residual state<select id="pca-layer"><option value="0">Embedding</option><option value="1">After L1</option><option value="2" selected>After L2</option><option value="3">After L3</option><option value="4">After L4</option></select></label>
        <label>X axis<select id="pca-x">{''.join(f'<option value="pc{i}" {"selected" if i == 1 else ""}>PC{i}</option>' for i in range(1, 7))}</select></label>
        <label>Y axis<select id="pca-y">{''.join(f'<option value="pc{i}" {"selected" if i == 2 else ""}>PC{i}</option>' for i in range(1, 7))}</select></label>
        <label>Z axis<select id="pca-z">{''.join(f'<option value="pc{i}" {"selected" if i == 3 else ""}>PC{i}</option>' for i in range(1, 7))}</select></label>
      </div>
      <div id="pca-plot" class="pca-plot"></div>
      <p class="muted">每个点是一个 exact-count centroid；颜色和标签表示 count 1–30。连线仅帮助观察 centroid 顺序，不表示模型实际沿直线演化。</p>
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
          marker: {{size: 6, color: rows.map(r => r.state_label), colorscale: 'Viridis', colorbar: {{title: 'count'}}, line: {{width: 0.3, color: '#ffffff'}}}},
          line: {{width: 2, color: '#94a3b8'}}, hovertemplate: 'count=%{{text}}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>'
        }}], {{
          title: `${{posEl.value.toUpperCase()}} · ${{siteEl.value}} · residual ${{layerEl.value}}`,
          scene: {{xaxis: {{title: xEl.value.toUpperCase()}}, yaxis: {{title: yEl.value.toUpperCase()}}, zaxis: {{title: zEl.value.toUpperCase()}}}},
          margin: {{l: 0, r: 0, t: 55, b: 0}}, paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff'
        }}, {{responsive: true, displaylogo: false}});
      }}
      [posEl, siteEl, layerEl, xEl, yEl, zEl].forEach(el => el.addEventListener('change', renderPca));
      refreshSites(); renderPca();
    </script>
    """


def metric_value(frame: pd.DataFrame, position: str, mode: str, count_bin: str, metric: str) -> float:
    rows = frame[
        (frame.position_encoding == position)
        & (frame["mode"] == mode)
        & (frame.count_bin.astype(str) == count_bin)
    ]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[-1][metric])


def build_report(run_dir: Path) -> Path:
    set_plot_style()
    audit, config, manifest = audit_bundle(run_dir)
    tables_dir = run_dir / "tables"
    assets = run_dir / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    train = read_csv(tables_dir / "train_metrics.csv")
    losses = read_csv(tables_dir / "eval_losses.csv")
    eval_bins = read_csv(tables_dir / "eval_by_bin.csv")
    eval_count = read_csv(tables_dir / "eval_by_count.csv")
    ar_bins = read_csv(tables_dir / "autoregressive_by_bin.csv")
    ar_detail = read_csv(tables_dir / "autoregressive_detail.csv")
    time99 = read_csv(tables_dir / "time_to_99.csv")
    attention = read_csv(tables_dir / "attention_summary.csv")
    probes = read_csv(tables_dir / "state_probe_summary.csv")
    variance = read_csv(tables_dir / "state_pca_variance.csv")
    centroids = read_csv(tables_dir / "state_centroids_pca.csv")
    model_specs = read_csv(tables_dir / "model_specifications.csv")

    figures = {
        "loss": plot_training_loss(train, assets),
        "segments": plot_segment_losses(losses, assets),
        "final_learning": plot_final_learning(eval_bins, ar_bins, assets),
        "trace_learning": plot_trace_learning(eval_bins, ar_bins, assets),
        "by_count": plot_final_by_count(eval_count, ar_detail, assets),
        "broad": plot_broad_attention(attention, assets),
        "targeted": plot_targeted_attention(attention, assets),
        "targeted_quality": plot_targeted_quality(attention, assets),
        "readout": plot_trace_readout(attention, assets),
        "probe": plot_state_probe(probes, assets),
        "pca": plot_pca_coverage(variance, assets),
    }

    final_eval = final_rows(eval_bins, ["position_encoding", "mode", "count_bin"])
    final_ar = final_rows(ar_bins, ["position_encoding", "mode", "count_bin"])
    final_losses = final_rows(losses, ["position_encoding", "mode"])

    performance_rows: list[dict[str, Any]] = []
    for position in POSITIONS:
        for mode in MODES:
            for count_bin in COUNT_BINS:
                performance_rows.append(
                    {
                        "position": position.upper(),
                        "mode": mode,
                        "count bin": count_bin,
                        "TF final": metric_value(final_eval, position, mode, count_bin, "tf_final_accuracy"),
                        "AR final": metric_value(final_ar, position, mode, count_bin, "ar_final_accuracy"),
                        "TF marker": metric_value(final_eval, position, mode, count_bin, "tf_trace_marker_accuracy") if mode == "thinking" else np.nan,
                        "AR trace exact": metric_value(final_ar, position, mode, count_bin, "trace_exact") if mode == "thinking" else np.nan,
                        "AR marker recall": metric_value(final_ar, position, mode, count_bin, "trace_marker_recall") if mode == "thinking" else np.nan,
                    }
                )
    performance = pd.DataFrame(performance_rows)

    display_perf = performance.copy()
    for column in ("TF final", "AR final", "TF marker", "AR trace exact", "AR marker recall"):
        display_perf[column] = display_perf[column].map(lambda value: "-" if pd.isna(value) else pct(value))

    loss_columns = [
        "position_encoding",
        "mode",
        "step",
        "eval_total_loss",
        "eval_prompt_body_loss",
        "eval_final_count_loss",
        "eval_trace_marker_loss",
        "eval_trace_index_loss",
    ]
    display_losses = final_losses[[column for column in loss_columns if column in final_losses.columns]].copy()
    display_losses.columns = [
        "position",
        "mode",
        "step",
        "total CE",
        "prompt CE",
        "final-count CE",
        "trace-marker CE",
        "trace-index CE",
    ][: len(display_losses.columns)]

    timing = time99.copy()
    timing = timing[["position_encoding", "mode", "count_bin", "metric", "first_step_at_threshold"]]
    timing["first_step_at_threshold"] = timing["first_step_at_threshold"].map(
        lambda value: "未达到" if pd.isna(value) else f"{int(value):,}"
    )
    timing.columns = ["position", "mode", "count bin", "metric", "first step ≥99%"]

    top_attention_rows: list[dict[str, Any]] = []
    definitions = (
        ("nonthinking", "final_answer", "broad_attention_score", "broad aggregation"),
        ("thinking", "trace_index", "correct_prompt_needle_mass", "raw k-to-k retrieval"),
        ("thinking", "final_answer", "trace_markers_mass", "trace readout"),
    )
    for position in POSITIONS:
        for mode, query, metric, mechanism in definitions:
            rows = attention[
                (attention.position_encoding == position)
                & (attention["mode"] == mode)
                & (attention.query_kind == query)
                & (attention.count_bin.astype(str) == "all")
            ].dropna(subset=[metric])
            if rows.empty:
                continue
            best = rows.sort_values(metric, ascending=False).iloc[0]
            top_attention_rows.append(
                {
                    "position": position.upper(),
                    "mechanism": mechanism,
                    "best head": f"L{int(best.layer)}H{int(best['head'])}",
                    "score": float(best[metric]),
                    "prompt needle mass": float(best.prompt_needles_mass),
                    "correct top-1": float(best.correct_top1),
                    "trace marker mass": float(best.trace_markers_mass),
                }
            )
    top_attention = pd.DataFrame(top_attention_rows)

    best_probe_rows: list[dict[str, Any]] = []
    for position in POSITIONS:
        for mode, site in (
            ("nonthinking", "final_answer"),
            ("thinking", "final_answer"),
            ("thinking", "trace_index"),
            ("thinking", "trace_marker"),
        ):
            rows = probes[
                (probes.position_encoding == position)
                & (probes["mode"] == mode)
                & (probes.site == site)
            ]
            if rows.empty:
                continue
            best = rows.sort_values("ridge_r2", ascending=False).iloc[0]
            best_probe_rows.append(
                {
                    "position": position.upper(),
                    "mode / site": f"{mode} / {site}",
                    "best residual": "embedding" if int(best.layer) == 0 else f"after L{int(best.layer)}",
                    "ridge R²": float(best.ridge_r2),
                    "ridge MAE": float(best.ridge_mae),
                    "nearest-centroid acc.": float(best.nearest_centroid_accuracy),
                    "position-only acc.": float(best.position_only_accuracy),
                }
            )
    best_probes = pd.DataFrame(best_probe_rows)

    all_complete = (audit.status == "complete").all()
    rope_trace = performance[(performance.position == "ROPE") & (performance["mode"] == "thinking")]
    rpe_trace = performance[(performance.position == "RPE") & (performance["mode"] == "thinking")]
    rope_trace_exact = rope_trace["AR trace exact"].tolist()
    rpe_trace_exact = rpe_trace["AR trace exact"].tolist()
    rope_marker = rope_trace["TF marker"].tolist()
    rpe_marker = rpe_trace["TF marker"].tolist()

    settings = pd.DataFrame(
        [
            ("研究变量", "RoPE vs learned relative-position bias (RPE); each has separate thinking and non-thinking Transformers"),
            ("模型", f"4 layers × 4 heads; d_model={config['n_embd']}; MLP={config['n_inner']}; tied embedding/unembedding"),
            ("数据", "Standard Tiny Shakespeare character windows; insert 1–30 synthetic marker tokens; online streaming; uniform count distribution"),
            ("序列", f"prompt length={config['seq_len']}; context window={config['n_positions']}; count bins=1–10, 11–20, 21–30"),
            ("训练", f"{config['train_steps']} steps; batch={config['batch_size']}; AdamW lr={config['lr']}; warmup={config['warmup_steps']}; seed={config['seed']}"),
            ("目标", "All-sequence teacher-forced causal next-token cross-entropy over every non-padding prompt and completion token"),
            ("评估", f"TF={config['eval_examples_per_count']} examples/count; AR={config['ar_examples_per_count']} examples/count; checkpoints every {config['checkpoint_every']} steps"),
        ],
        columns=["item", "setting"],
    )

    css = """
    :root { --ink:#172033; --muted:#5b677a; --line:#dbe3ee; --soft:#f6f8fb; --blue:#2563eb; --green:#16a34a; --amber:#f59e0b; }
    * { box-sizing:border-box; }
    body { margin:0; background:#edf2f7; color:var(--ink); font-family:Inter,"Segoe UI","Microsoft YaHei",Arial,sans-serif; font-size:17px; line-height:1.72; }
    main { max-width:1480px; margin:0 auto; background:#fff; padding:44px 64px 80px; box-shadow:0 0 30px rgba(15,23,42,.08); }
    h1 { font-size:38px; line-height:1.2; margin:0 0 12px; letter-spacing:0; }
    h2 { font-size:29px; margin:58px 0 22px; border-top:1px solid var(--line); padding-top:30px; letter-spacing:0; }
    h3 { font-size:23px; margin:30px 0 14px; letter-spacing:0; }
    h4 { font-size:20px; margin:0 0 18px; letter-spacing:0; }
    p { margin:10px 0 16px; }
    code { background:#edf3fa; color:#17365d; border-radius:4px; padding:2px 6px; font-family:Consolas,monospace; }
    footer code { white-space:normal; overflow-wrap:anywhere; word-break:break-word; }
    .subtitle { color:var(--muted); font-size:19px; margin-bottom:24px; }
    .meta { display:flex; gap:12px; flex-wrap:wrap; margin:20px 0 28px; }
    .chip { background:#eaf2ff; color:#174ea6; border:1px solid #c8daf8; padding:6px 11px; border-radius:5px; font-weight:650; }
    .callout { border-left:5px solid var(--blue); background:#eef5ff; padding:17px 22px; margin:22px 0; }
    .callout.green { border-color:var(--green); background:#effbf3; }
    .callout.amber { border-color:var(--amber); background:#fff9e9; }
    .warning { border-left:5px solid #dc2626; background:#fff1f2; padding:16px 20px; }
    .muted { color:var(--muted); font-size:15px; }
    .cards { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
    .card { border:1px solid var(--line); border-radius:7px; padding:20px; background:#fff; }
    .data-table { width:100%; border-collapse:collapse; margin:18px 0 26px; font-size:14px; }
    .data-table th { background:#e9eff7; text-align:left; font-weight:700; }
    .data-table th,.data-table td { border:1px solid #d4deea; padding:9px 11px; vertical-align:top; }
    .table-wrap { overflow-x:auto; }
    figure { margin:24px 0 34px; border:1px solid var(--line); border-radius:7px; padding:20px; background:#fff; }
    figure img { display:block; width:100%; height:auto; max-height:780px; object-fit:contain; margin:0 auto; }
    figcaption { color:#4b5b70; font-size:15px; line-height:1.65; border-top:1px solid #e5eaf0; margin-top:16px; padding-top:13px; }
    .equation { display:block; overflow-x:auto; margin:12px 0; padding:12px 16px; background:#f8fafc; border:1px solid var(--line); text-align:center; font-family:"Cambria Math","Times New Roman",serif; font-size:22px; }
    .definition-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
    .definition { border:1px solid var(--line); border-radius:7px; padding:16px 18px; }
    .definition strong { display:block; margin-bottom:7px; }
    .interactive-card { border:1px solid var(--line); border-radius:7px; padding:18px; margin:22px 0; }
    .controls { display:grid; grid-template-columns:repeat(3,minmax(180px,1fr)); gap:12px; margin-bottom:12px; }
    .controls label { font-size:14px; font-weight:700; color:#3b4a60; }
    .controls select { display:block; width:100%; padding:8px 9px; margin-top:4px; border:1px solid #bcc9d9; border-radius:4px; background:white; }
    .pca-plot { width:100%; height:650px; }
    nav { background:#f8fafc; border:1px solid var(--line); border-radius:7px; padding:16px 22px; }
    nav a { color:#2459a9; text-decoration:none; margin-right:18px; white-space:nowrap; }
    @media(max-width:900px) { main{padding:28px 20px}.cards,.definition-grid{grid-template-columns:1fr}.controls{grid-template-columns:1fr}.pca-plot{height:500px}h1{font-size:31px} }
    """

    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v15: Shakespeare inserted-needle counting with RoPE/RPE</title><style>{css}</style></head>
<body><main>
<h1>v15：Shakespeare haystack 中的合成 needle counting</h1>
<p class="subtitle">RoPE 与 learned relative-position bias 下，non-thinking 直接计数和 thinking trace 计数的学习、attention 与 count-state 表征</p>
<div class="meta"><span class="chip">seed 1234</span><span class="chip">4 models</span><span class="chip">10,000 steps</span><span class="chip">count 1–30</span><span class="chip">all-sequence causal loss</span></div>
<nav><a href="#s1">1. 研究问题</a><a href="#s2">2. 实验设置</a><a href="#s3">3. 定义</a><a href="#s4">4. 学习动态</a><a href="#s5">5. Attention</a><a href="#s6">6. Hidden state</a></nav>

<section id="s1"><h2>1. 研究对象、目标与两种机制假设</h2>
<p>v15 不再使用均匀独立的 noise token 作为 haystack，而是从标准 Tiny Shakespeare 字符语料中截取长度 256 的自然字符窗口，再插入 1–30 个合成 marker。它研究的是<strong>插入 marker 的数量</strong>，不是 Shakespeare 中某个原生字母的频数。</p>
<div class="cards">
  <div class="card"><h3>Non-thinking：prompt-wide set aggregation</h3><p>模型直接在 <code>&lt;Ans&gt;</code> 位置输出 count。工作假设是：若干 early broad heads 将 attention 同时分配到多个 prompt marker，再由 residual/MLP 把集合统计写入可解码的 count state。这里没有逐项 trace 作为外部工作区。</p></div>
  <div class="card"><h3>Thinking：retrieval → trace → readout</h3><p>模型先生成共享数字 token 与 marker token 组成的 trace，再输出最终 count。工作假设是：trace 数字 query 逐项定位 prompt needle（k-to-k retrieval），trace marker 保存被取回的 identity，最后 <code>&lt;Ans&gt;</code> query 从整段 trace 读出 count。</p></div>
</div>
<div class="callout green"><strong>核心结果。</strong> 四个模型最终 count 都接近饱和，但内部过程并不等价。RoPE thinking 的 trace 明显优于 RPE thinking；与此同时，两种位置编码都没有重现 v2 中接近 1.0 的尖锐 k-to-k head。最稳定的 thinking attention 现象反而是 Layer 4 对 trace markers 的最终 readout。</div>
<div class="callout amber"><strong>证据等级。</strong> 本报告 1–6 节是学习动态、attention 与 hidden-state 的<strong>描述性</strong>分析。高 attention mass、高 probe R² 或清晰 PCA 几何都不自动等于因果机制；还需要 ablation/patching 才能确认必要性与充分性。</div>
</section>

<section id="s2"><h2>2. 模型、数据、训练与完整性</h2>
<h3>2.1 四个独立模型</h3>
<p>实验是 2×2 设计：位置编码为 RoPE 或 RPE；输出模式为 non-thinking 或 thinking。四者是独立随机初始化、独立训练的 Transformer，不是同一个模型的开关模式。</p>
<div class="table-wrap">{table_html(settings)}</div>
<p><strong>RoPE</strong> 在每层 attention 的 query/key 上施加旋转位置变换（base=10,000）；<strong>RPE</strong> 在每层每头的 causal attention logits 上加入可学习的相对距离 bias，最大相对距离 256。两者都没有 learned absolute-position embedding。</p>
<h3>2.2 序列与 loss mask</h3>
<div class="cards">
  <div class="card"><strong>Non-thinking</strong><div class="equation">&lt;BOS&gt; prompt &lt;Think&gt; &lt;/Think&gt; &lt;Ans&gt; C<sub>n</sub> &lt;EOS&gt;</div></div>
  <div class="card"><strong>Thinking</strong><div class="equation">&lt;BOS&gt; prompt &lt;Think&gt; C<sub>1</sub> M<sub>1</sub> … C<sub>n</sub> M<sub>n</sub> &lt;/Think&gt; &lt;Ans&gt; C<sub>n</sub> &lt;EOS&gt;</div></div>
</div>
<p>数字 token 在 trace index 与最终答案之间共享。训练目标覆盖<strong>全部非 padding token</strong>：prompt 中的 Shakespeare 字符、插入 marker、控制 token 与 completion 全部进入 teacher-forced next-token cross-entropy。</p>
<div class="equation">L<sub>all</sub> = − (1 / Σ<sub>t</sub> m<sub>t</sub>) Σ<sub>t</sub> m<sub>t</sub> log p(x<sub>t</sub> | x<sub>&lt;t</sub>)</div>
<div class="callout amber"><strong>重要解释。</strong> prompt 有 256 个 token，而 count 答案只有 1 个 token，因此 total CE 主要由 Shakespeare prompt 建模决定。total CE 下降或接近 0 不等价于 counting 成功；必须单独看 final-count、trace segment 与 autoregressive 指标。</div>
<h3>2.3 下载结果完整性</h3>
<p>审计结果：<strong>{'全部完整' if all_complete else '存在缺失'}</strong>。四个 pipeline stage 均标记 complete；四个模型各含 final checkpoint 和 step 1,000–10,000 的 10 个中间 checkpoint；训练、teacher-forced、autoregressive、attention、probe 与 PCA 表均可读。</p>
<div class="table-wrap">{table_html(audit)}</div>
<h3>2.4 参数量</h3><div class="table-wrap">{table_html(model_specs)}</div>
</section>

<section id="s3"><h2>3. 新术语、数据列与计算定义</h2>
<div class="definition-grid">
  <div class="definition"><strong>Teacher-forced (TF) final-count accuracy</strong>给模型完整 gold prefix（thinking 时包括 gold trace），在最终 count 位置检查 argmax 是否等于真实 count。它测局部 readout，不测模型能否自己生成正确 trace。</div>
  <div class="definition"><strong>Autoregressive (AR) final-count accuracy</strong>只给 prompt，让模型逐 token 自由生成 completion，再检查最终 count。它包含所有上游生成错误，是更严格的端到端指标。</div>
  <div class="definition"><strong>Trace-marker accuracy / recall</strong>TF accuracy 是在每个 gold trace-marker query 预测正确 marker identity 的比例；AR recall 是自由生成 trace 与 gold marker 多重集合的召回率。<code>trace_exact</code> 要求整段生成 trace 完全匹配。</div>
  <div class="definition"><strong>Segment cross-entropy</strong>在同一次 teacher-forced 前向中，只对某个语义位置集合（prompt、trace index、trace marker、final count 等）的 token loss 求均值；用于避免 total CE 被长 prompt 淹没。</div>
  <div class="definition"><strong>Attention mass</strong>对某个 query q 和位置集合 S，把该 head 指向 S 的 softmax 权重相加：<span class="equation">M(S|q) = Σ<sub>j∈S</sub> A(q,j)</span>它说明 attention 放在哪里，不直接说明因果贡献。</div>
  <div class="definition"><strong>Broad aggregation score</strong><span class="equation">S<sub>broad</sub> = M(N|q) × H<sub>N</sub>(q)</span>N 是 prompt needle 集合；H<sub>N</sub> 是 needle 子集内归一化熵。该分数同时奖励“总 mass 在 needles 上”和“在多个 needles 间广泛分布”。</div>
  <div class="definition"><strong>Raw k-to-k mass</strong>在第 k 个 trace 数字 query q<sub>k</sub>，直接读取 matching prompt needle n<sub>k</sub> 的权重：<span class="equation">S<sub>target</sub>(k) = A(q<sub>k</sub>, n<sub>k</sub>)</span>这是全上下文中的绝对 mass。</div>
  <div class="definition"><strong>Correct top-1 与 diagonal dominance</strong>top-1 只在 prompt needle 子集内判断最大权重是否落在 n<sub>k</sub>；diagonal dominance 是 A(q<sub>k</sub>,n<sub>k</sub>)/M(N|q<sub>k</sub>)。两者可很高而 raw mass 仍很低。</div>
  <div class="definition"><strong>Trace-readout score</strong>最终 <code>&lt;Ans&gt;</code> query 指向所有 trace marker 位置的 attention mass，用来寻找从已给 trace 读取 count/进度的候选 heads。</div>
  <div class="definition"><strong>Count-state probe</strong>在指定 token 位置提取 residual vector。Nearest-centroid 做 30 类 exact-count 分类；ridge 把 count 当连续变量并报告 held-out R²/MAE。高可读性不等于该方向被模型因果使用。</div>
</div>
</section>

<section id="s4"><h2>4. 学习动态：最终答案很快，trace identity 较慢</h2>
{figure(figures['loss'], 'Figure 1. 全序列训练 loss', '<strong>横轴</strong>是 optimizer step；<strong>纵轴</strong>是 prompt + completion 所有非 padding token 的平均 next-token CE。左右分别是 RoPE/RPE，颜色区分 non-thinking/thinking。RoPE 最终约 0.1、RPE 约 0.3，但该差异主要反映 Shakespeare prompt 建模，不可直接解释为计数差异。')}
{figure(figures['segments'], 'Figure 2. 按语义 segment 分解的 validation CE', '<strong>横轴</strong>是 step；<strong>纵轴</strong>是各 segment teacher-forced CE（对数刻度）。灰色 prompt CE 与蓝色 final-count CE 分离；thinking 另画绿色 trace-marker 和紫色 trace-index。RoPE thinking 的 marker CE 明显低于 RPE，解释了两者自由生成 trace 的差距。')}
<div class="table-wrap">{table_html(display_losses)}</div>
{figure(figures['final_learning'], 'Figure 3. 三个 count 区间的最终答案学习曲线', '<strong>横轴</strong>是 step；<strong>纵轴</strong>是 final-count exact accuracy。颜色表示 count 1–10、11–20、21–30；实线是 teacher-forced，虚线是 autoregressive。Thinking 的 TF final count 很早达到 99%，因为 gold trace 已经把进度显式写在 prefix 中；这不意味着 trace 已学会。')}
{figure(figures['trace_learning'], 'Figure 4. Thinking trace-marker 的学习曲线', '<strong>横轴</strong>是 step；<strong>纵轴</strong>分别为 teacher-forced next-marker accuracy（左列）和自由生成 marker recall（右列）。RoPE 在三个难度区间维持约 0.94 TF marker accuracy；RPE 随 count 增长由约 0.70 降至 0.39。')}
{figure(figures['by_count'], 'Figure 5. Final checkpoint 的 exact-count 曲线', '<strong>横轴</strong>是 gold needle count 1–30；<strong>纵轴</strong>是 final-count accuracy。实线为 TF，虚线为 AR；两条竖虚线划分三个 count bins。最终答案几乎饱和，因此 v15 的关键差异不在 final count，而在 trace quality 与内部路由。')}
<h3>4.1 Final checkpoint 数值</h3><div class="table-wrap">{table_html(display_perf)}</div>
<h3>4.2 首次达到 99% 的 step</h3><div class="table-wrap">{table_html(timing)}</div>
<div class="callout green"><strong>为什么 thinking 的 count 学得快、trace 却慢？</strong> TF final-count query 已经看到完整 gold trace，最简单策略是从 trace 长度、最后数字或位置线索读出 C<sub>n</sub>；marker identity 预测则必须解决“第 k 个 prompt needle 是什么”的逐项 retrieval。RPE thinking 因此可以做到 final count 100%，同时 AR trace exact 在 11–30 上为 0%。</div>
</section>

<section id="s5"><h2>5. 描述性 attention：RoPE trace 路由更强，但没有 v2 式尖锐 k-to-k head</h2>
<h3>5.1 Broad prompt aggregation</h3>
{figure(figures['broad'], 'Figure 6. Final-answer query 的 broad aggregation score', '<strong>横轴</strong>是 head H0–H3；<strong>纵轴</strong>是 Layer 1–4；单元格为 prompt needle mass × needle-subset normalized entropy。Non-thinking 的 Layer 1 最突出，符合“直接从 prompt 集合聚合”的候选机制；thinking 的 broad score 较弱。')}
<h3>5.2 Thinking trace 的 k-to-k targeted retrieval</h3>
{figure(figures['targeted'], 'Figure 7. 按 count 区间分解的 raw k-to-k mass', '<strong>横轴</strong>是 head；<strong>纵轴</strong>是 layer；每列是一个 count bin，每行是一种位置编码。单元格是第 k 个 trace 数字 query 对 matching 第 k 个 prompt needle 的原始 attention 权重。RoPE 的 L1H2 最强但全区间均值仅约 0.13；RPE 峰值约 0.04。')}
{figure(figures['targeted_quality'], 'Figure 8. Needle 子集内部的相对 retrieval 质量', '<strong>左列</strong> correct top-1：matching needle 是否是 prompt needle 子集内权重最大者；<strong>右列</strong> diagonal dominance：needle 总 mass 中 matching needle 的占比。横轴=head，纵轴=layer。相对指标也不高，说明低 raw mass 不只是 BOS/noise 稀释。')}
<div class="callout amber"><strong>与 v2 的差异。</strong> v15 的 count 扩至 30、haystack 改成自然字符、且 loss 覆盖整个 prompt。模型仍能输出 count，但 attention 不再收敛到单个几乎确定性的 k-to-k head。RoPE 比 RPE 更接近 targeted routing，同时也生成更好的 trace；这是一致的相关证据，但尚非因果证明。</div>
<h3>5.3 Thinking final-answer 的 trace readout</h3>
{figure(figures['readout'], 'Figure 9. 最终答案 query 指向 trace markers 的 mass', '<strong>横轴</strong>是 head；<strong>纵轴</strong>是 layer；单元格为最终 <Ans> query 对全部 trace-marker positions 的 attention mass。两种位置编码都在 Layer 4 出现强 readout heads（RoPE L4H2/L4H0；RPE 四个 L4 heads），比 k-to-k retrieval 更稳定。')}
<h3>5.4 各机制的最强描述性 head</h3><div class="table-wrap">{table_html(top_attention)}</div>
</section>

<section id="s6"><h2>6. 描述性 hidden-state geometry：count 可读，但 trace 位置存在完全混淆</h2>
<p><strong>Residual state 0</strong> 是 token embedding；state 1–4 分别是经过对应 Transformer layer 后的 residual stream。对每个 model/site/layer，我们在 held-out examples 上做 nearest-centroid exact-count 分类和 ridge count 回归。</p>
{figure(figures['probe'], 'Figure 10. Count probe 与 position-only control', '<strong>横轴</strong>是 embedding 及经过 Layer 1–4 后的 residual state；<strong>纵轴</strong>是 model mode 与 semantic site。左列是 ridge R²，右列是只用绝对 token position 的 baseline accuracy。Non-thinking final-answer 的 position baseline 接近 chance，而 thinking trace anchors 为 1.0，说明后者的 count 与 trace token 位置完全共线。')}
<div class="table-wrap">{table_html(best_probes)}</div>
<div class="callout amber"><strong>不能过度解释 trace probe。</strong> 第 k 个 trace index/marker 本来就位于由 k 决定的绝对位置，因此 position-only baseline=100%。即使 residual ridge R²≈1，也无法区分“语义 count state”与“位置编码”。Non-thinking final-answer 的 token 位置固定，position baseline≈3.3%，其高 R² 更能支持真正的 count 可读性。</div>
<h3>6.1 PCA centroid geometry</h3>
{figure(figures['pca'], 'Figure 11. PC1–PC6 对 exact-count centroid 几何的累计覆盖', '<strong>横轴</strong>是 residual state；<strong>纵轴</strong>是 model/site；单元格是前 6 个主成分对 30 个 count centroid 之间方差的累计解释比例。浅层常接近低维曲线，深层在部分 thinking sites 变得更分散；这描述 centroid 几何，不等于单样本隐状态只占 6 维。')}
<h3>6.2 可交互 3D centroid manifold</h3>
<p>下面可以选择 RoPE/RPE、semantic site、layer，以及 PC1–PC6 中任意三轴。所有坐标都来自同一 model/site/layer 的 30 个 exact-count centroid 上单独拟合的 PCA。</p>
{pca_interactive_html(centroids)}
<div class="callout green"><strong>综合结论。</strong> v15 证明了两种位置编码都能学会最终 count；RoPE 在 trace identity 与 targeted routing 上显著优于 RPE，但两者的最终答案都可通过强 Layer-4 trace readout 饱和。Non-thinking 的 Layer-1 broad attention 与低 position-baseline count geometry 支持 prompt-wide aggregation 假设；thinking 的 trace position probes 则高度混淆。下一步若要确认机制，需要在同一批 clean/corrupt pairs 上对候选 broad、targeted、readout heads 和 residual states 做 position-local ablation 与 activation patching。</div>
</section>

<footer><p class="muted">Generated from <code>{esc(run_dir)}</code>. Manifest updated {esc(manifest.get('updated_at_utc','NA'))}. All figures and Plotly JavaScript are embedded; this HTML can be moved or shared independently.</p></footer>
</main></body></html>"""

    report_path = run_dir / "syn_v15_report.html"
    report_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Completeness: {all_complete}; size={report_path.stat().st_size:,} bytes")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the self-contained v15 result report")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    build_report(args.run_dir.resolve())


if __name__ == "__main__":
    main()
