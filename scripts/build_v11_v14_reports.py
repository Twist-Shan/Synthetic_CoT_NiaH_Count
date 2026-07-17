from __future__ import annotations

import argparse
import base64
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VERSIONS = ("v11", "v12", "v13", "v14")


@dataclass
class RunBundle:
    version: str
    path: Path
    config: dict[str, Any]
    tables: dict[str, pd.DataFrame]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def discover_runs(results_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for version in VERSIONS:
        candidates = sorted(results_root.glob(f"{version}_main_seed1234_*"))
        candidates = [path for path in candidates if path.is_dir() and (path / "config.json").exists()]
        if not candidates:
            raise FileNotFoundError(f"No formal {version} run found under {results_root}")
        found[version] = candidates[-1]
    return found


def load_bundle(version: str, path: Path) -> RunBundle:
    table_names = (
        "attention_summary",
        "autoregressive_by_bin",
        "eval_by_bin",
        "eval_by_count",
        "eval_losses",
        "model_specifications",
        "state_pca_variance",
        "state_centroids_pca",
        "state_probe_summary",
        "time_to_99",
        "train_metrics",
    )
    tables = {name: read_csv(path / "tables" / f"{name}.csv") for name in table_names}
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    return RunBundle(version, path, config, tables)


def esc(value: Any) -> str:
    return html.escape(str(value))


def pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if pd.isna(number):
        return "NA"
    return f"{100.0 * number:.{digits}f}%"


def num(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if pd.isna(number):
        return "NA"
    return f"{number:.{digits}f}"


def int_or_dash(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "未达到"
    if pd.isna(number):
        return "未达到"
    return f"{int(number):,}"


def table_html(frame: pd.DataFrame, *, classes: str = "data-table") -> str:
    if frame.empty:
        return '<p class="muted">没有可用数据。</p>'
    display = frame.copy()
    display.columns = [str(column) for column in display.columns]
    return display.to_html(index=False, escape=False, classes=classes, border=0)


def image_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def figure(path: Path, title: str, caption: str, *, compact: bool = False) -> str:
    if not path.exists():
        return f'<div class="warning"><strong>{esc(title)}</strong><br>缺少图像：{esc(path.name)}</div>'
    cls = "figure compact" if compact else "figure"
    return (
        f'<figure class="{cls}">'
        f'<h4>{esc(title)}</h4>'
        f'<img src="{image_uri(path)}" alt="{esc(title)}">'
        f'<figcaption>{caption}</figcaption>'
        "</figure>"
    )


def figure_grid(items: Iterable[str], *, columns: int = 2) -> str:
    return f'<div class="figure-grid cols-{columns}">' + "".join(items) + "</div>"


def clean_state_probe_figure(bundle: RunBundle, position: str) -> Path:
    """Redraw the state-probe table with one shared, non-overlapping row axis."""
    out_dir = bundle.path / "report_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"state_probe_clean_{position}.png"

    frame = bundle.tables["state_probe_summary"]
    frame = frame[frame["position_encoding"] == position].copy()
    frame["row_label"] = frame["mode"] + " | " + frame["site"]
    row_order = [
        "nonthinking | final_answer",
        "thinking | final_answer",
        "thinking | trace_index",
        "thinking | trace_marker",
    ]
    layers = [0, 1, 2, 3, 4]
    metrics = [
        ("nearest_centroid_accuracy", "Nearest-centroid accuracy", 0.0, 1.0),
        ("position_only_accuracy", "Absolute-position baseline", 0.0, 1.0),
        ("ridge_r2", "Ridge count R²", -0.2, 1.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), constrained_layout=True)
    fig.suptitle(f"{position.upper()} count-state decodability and position control", fontsize=17)
    for axis_index, (ax, (metric, title, vmin, vmax)) in enumerate(zip(axes, metrics)):
        pivot = (
            frame.pivot_table(index="row_label", columns="layer", values=metric, aggfunc="mean")
            .reindex(index=row_order, columns=layers)
        )
        values = pivot.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(len(layers)), [str(layer) for layer in layers])
        ax.set_xlabel("residual state: 0=embedding, 1–4=after Layer")
        ax.set_yticks(range(len(row_order)))
        if axis_index == 0:
            ax.set_yticklabels(row_order)
            ax.set_ylabel("model | semantic site")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if np.isnan(value):
                    continue
                midpoint = (vmin + vmax) / 2
                color = "white" if value < midpoint else "black"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def final_group_rows(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    maximum = frame.groupby(keys)["step"].transform("max")
    return frame[frame["step"] == maximum].copy()


def head_label(row: pd.Series) -> str:
    return f"L{int(row['layer'])}H{int(row['head'])}"


def top_attention_head(
    attention: pd.DataFrame,
    position: str,
    mode: str,
    query_kind: str,
    metric: str,
) -> dict[str, Any]:
    rows = attention[
        (attention["position_encoding"] == position)
        & (attention["mode"] == mode)
        & (attention["query_kind"] == query_kind)
        & (attention["count_bin"].astype(str) == "all")
    ].copy()
    rows = rows.dropna(subset=[metric])
    if rows.empty:
        return {"head": "NA", metric: float("nan")}
    row = rows.sort_values(metric, ascending=False).iloc[0]
    result = {"head": head_label(row), metric: float(row[metric])}
    for column in (
        "correct_prompt_needle_mass",
        "correct_top1",
        "diagonal_dominance",
        "broad_attention_score",
        "prompt_needles_mass",
        "needle_entropy_normalized",
        "trace_markers_mass",
    ):
        if column in row:
            result[column] = float(row[column])
    return result


def variant_summary(bundle: RunBundle, position: str) -> dict[str, Any]:
    eval_rows = final_group_rows(
        bundle.tables["eval_by_bin"], ["position_encoding", "mode", "count_bin"]
    )
    eval_rows = eval_rows[eval_rows["position_encoding"] == position]
    ar_rows = final_group_rows(
        bundle.tables["autoregressive_by_bin"], ["position_encoding", "mode", "count_bin"]
    )
    ar_rows = ar_rows[ar_rows["position_encoding"] == position]
    attention = bundle.tables["attention_summary"]
    probes = bundle.tables["state_probe_summary"]
    probes = probes[probes["position_encoding"] == position]

    def mean_metric(frame: pd.DataFrame, mode: str, metric: str) -> float:
        rows = frame[frame["mode"] == mode]
        return float(rows[metric].mean()) if not rows.empty else float("nan")

    best_state: dict[str, dict[str, Any]] = {}
    for mode, site in (
        ("nonthinking", "final_answer"),
        ("thinking", "final_answer"),
        ("thinking", "trace_index"),
        ("thinking", "trace_marker"),
    ):
        rows = probes[(probes["mode"] == mode) & (probes["site"] == site)]
        key = f"{mode}:{site}"
        if rows.empty:
            best_state[key] = {"layer": "NA", "ridge_r2": float("nan"), "nearest": float("nan")}
        else:
            row = rows.sort_values("ridge_r2", ascending=False).iloc[0]
            best_state[key] = {
                "layer": int(row["layer"]),
                "ridge_r2": float(row["ridge_r2"]),
                "nearest": float(row["nearest_centroid_accuracy"]),
                "position_only": float(row["position_only_accuracy"]),
            }

    return {
        "version": bundle.version,
        "position": position,
        "tf_nonthinking": mean_metric(eval_rows, "nonthinking", "tf_final_accuracy"),
        "tf_thinking": mean_metric(eval_rows, "thinking", "tf_final_accuracy"),
        "tf_trace_marker": mean_metric(eval_rows, "thinking", "tf_trace_marker_accuracy"),
        "tf_trace_index": mean_metric(eval_rows, "thinking", "tf_trace_index_accuracy"),
        "ar_nonthinking": mean_metric(ar_rows, "nonthinking", "ar_final_accuracy"),
        "ar_thinking": mean_metric(ar_rows, "thinking", "ar_final_accuracy"),
        "ar_trace_exact": mean_metric(ar_rows, "thinking", "trace_exact"),
        "ar_trace_marker_recall": mean_metric(ar_rows, "thinking", "trace_marker_recall"),
        "broad_nonthinking": top_attention_head(
            attention, position, "nonthinking", "final_answer", "broad_attention_score"
        ),
        "broad_thinking": top_attention_head(
            attention, position, "thinking", "final_answer", "broad_attention_score"
        ),
        "targeted": top_attention_head(
            attention, position, "thinking", "trace_index", "correct_prompt_needle_mass"
        ),
        "readout": top_attention_head(
            attention, position, "thinking", "final_answer", "trace_markers_mass"
        ),
        "states": best_state,
    }


def config_row(bundle: RunBundle, *, current: str) -> dict[str, str]:
    cfg = bundle.config
    change = {
        "v11": "位置编码对照：APE / RoPE / learned RPE",
        "v12": "长度与 count 负载：512 token，1–50 needles",
        "v13": "训练分布：固定 15,360 条训练集，而非 streaming",
        "v14": "haystack：Tiny Shakespeare 字符流，而非 uniform noise",
    }[bundle.version]
    return {
        "版本": f"<strong>{bundle.version}</strong>" if bundle.version == current else bundle.version,
        "核心改动": change,
        "长度": str(cfg["seq_len"]),
        "count": f"{cfg['count_min']}–{cfg['count_max']}",
        "位置编码": ", ".join(value.upper() for value in cfg["position_encodings"]),
        "训练数据": cfg["training_data_mode"],
        "haystack": "Tiny Shakespeare chars" if cfg["noise_source"] == "shakespeare_char" else "uniform synthetic noise",
        "模型": f"4L × 4H, d={cfg['n_embd']}, MLP={cfg['n_inner']}",
    }


def comparison_frame(
    bundles: dict[str, RunBundle], summaries: dict[tuple[str, str], dict[str, Any]]
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for version in VERSIONS:
        bundle = bundles[version]
        for position in bundle.config["position_encodings"]:
            summary = summaries[(version, position)]
            time_rows = bundle.tables["time_to_99"]
            time_rows = time_rows[
                (time_rows["position_encoding"] == position)
                & (time_rows["metric"] == "tf_final_accuracy")
            ]

            def last_time(mode: str) -> str:
                rows_mode = time_rows[time_rows["mode"] == mode]
                reached = rows_mode[rows_mode["reached_threshold"].astype(str).str.lower() == "true"]
                if len(reached) != len(rows_mode) or reached.empty:
                    return "未全部达到"
                return f"{int(reached['first_step_at_threshold'].max()):,}"

            rows.append(
                {
                    "variant": f"{version}-{position.upper()}",
                    "non-thinking 全区间 ≥99%": last_time("nonthinking"),
                    "thinking final 全区间 ≥99%": last_time("thinking"),
                    "AR final (non / CoT)": f"{pct(summary['ar_nonthinking'])} / {pct(summary['ar_thinking'])}",
                    "AR trace exact": pct(summary["ar_trace_exact"]),
                    "最强 k-to-k head": summary["targeted"]["head"],
                    "k-to-k raw mass": num(summary["targeted"].get("correct_prompt_needle_mass")),
                    "top-1 within needles": pct(summary["targeted"].get("correct_top1")),
                }
            )
    return pd.DataFrame(rows)


def time_to_99_table(bundle: RunBundle, position: str) -> pd.DataFrame:
    rows = bundle.tables["time_to_99"]
    rows = rows[rows["position_encoding"] == position].copy()
    keep = rows[rows["metric"].isin(
        ["tf_final_accuracy", "tf_trace_index_accuracy", "tf_trace_marker_accuracy"]
    )]
    keep["metric"] = keep["metric"].map(
        {
            "tf_final_accuracy": "teacher-forced final count",
            "tf_trace_index_accuracy": "teacher-forced trace index",
            "tf_trace_marker_accuracy": "teacher-forced trace marker",
        }
    )
    keep["首次 ≥99% step"] = keep["first_step_at_threshold"].map(int_or_dash)
    return keep[["mode", "count_bin", "metric", "首次 ≥99% step"]].rename(
        columns={"mode": "模型", "count_bin": "count 区间", "metric": "指标"}
    )


def final_eval_table(bundle: RunBundle, position: str) -> pd.DataFrame:
    tf = final_group_rows(bundle.tables["eval_by_bin"], ["position_encoding", "mode", "count_bin"])
    tf = tf[tf["position_encoding"] == position]
    ar = final_group_rows(
        bundle.tables["autoregressive_by_bin"], ["position_encoding", "mode", "count_bin"]
    )
    ar = ar[ar["position_encoding"] == position]
    keys = ["position_encoding", "mode", "count_bin"]
    merged = tf.merge(ar, on=keys, how="outer")
    result = pd.DataFrame(
        {
            "模型": merged["mode"],
            "count 区间": merged["count_bin"],
            "TF final": merged["tf_final_accuracy"].map(pct),
            "AR final": merged["ar_final_accuracy"].map(pct),
            "AR |error|": merged["ar_abs_error"].map(lambda value: num(value, 2)),
            "AR trace exact": merged["trace_exact"].map(pct),
            "AR marker recall": merged["trace_marker_recall"].map(pct),
        }
    )
    return result


def attention_table(bundle: RunBundle, position: str) -> pd.DataFrame:
    summary = variant_summary(bundle, position)
    rows = []
    for label, key, metric in (
        ("Non-thinking broad aggregation", "broad_nonthinking", "broad_attention_score"),
        ("CoT final-query broad aggregation", "broad_thinking", "broad_attention_score"),
        ("CoT targeted k-to-k retrieval", "targeted", "correct_prompt_needle_mass"),
        ("CoT trace-marker readout", "readout", "trace_markers_mass"),
    ):
        item = summary[key]
        rows.append(
            {
                "候选机制": label,
                "最强 head": item["head"],
                "排名分数": num(item.get(metric)),
                "prompt needle mass": num(item.get("prompt_needles_mass")),
                "needle entropy": num(item.get("needle_entropy_normalized")),
                "k-to-k top-1": pct(item.get("correct_top1")),
                "diagonal dominance": num(item.get("diagonal_dominance")),
            }
        )
    return pd.DataFrame(rows)


def state_table(bundle: RunBundle, position: str) -> pd.DataFrame:
    probes = bundle.tables["state_probe_summary"]
    probes = probes[probes["position_encoding"] == position]
    variance = bundle.tables["state_pca_variance"]
    variance = variance[variance["position_encoding"] == position]
    rows: list[dict[str, str]] = []
    for (mode, site), group in probes.groupby(["mode", "site"], sort=False):
        best = group.sort_values("ridge_r2", ascending=False).iloc[0]
        component = variance[
            (variance["mode"] == mode)
            & (variance["site"] == site)
            & (variance["layer"] == best["layer"])
        ]
        pc6 = component.sort_values("component").tail(1)
        rows.append(
            {
                "模型 / site": f"{mode} / {site}",
                "最佳 state": "embedding" if int(best["layer"]) == 0 else f"after Layer {int(best['layer'])}",
                "ridge R²": num(best["ridge_r2"]),
                "ridge MAE": num(best["ridge_mae"], 2),
                "nearest-centroid acc": pct(best["nearest_centroid_accuracy"]),
                "position-only acc": pct(best["position_only_accuracy"]),
                "PC1–PC6 累计覆盖": pct(pc6.iloc[0]["cumulative_explained_variance"]) if not pc6.empty else "NA",
            }
        )
    return pd.DataFrame(rows)


def current_question(version: str) -> tuple[str, str]:
    return {
        "v11": (
            "位置编码会不会改变 targeted retrieval 的可学习性？",
            "同一数据、同一 64 维架构下，对比 learned absolute position embedding (APE)、RoPE 与 learned relative-position bias (RPE)。",
        ),
        "v12": (
            "小模型在更长上下文与 50 个 needles 下是否仍能形成同类机制？",
            "相对 v11-APE，把 prompt 从 256 延长到 512，并把 count 从 1–30 扩到 1–50；其余训练方式保持 streaming。",
        ),
        "v13": (
            "streaming 新样本是否是学会 counting 的必要条件？",
            "相对 v11-APE，只把训练数据改为固定数据集；模型会反复看到每个 count 512 条、共 15,360 条训练样本。",
        ),
        "v14": (
            "uniform synthetic noise 是否掩盖了结构化 haystack 下的行为？",
            "相对 v11-APE，只把 haystack 改为标准 Tiny Shakespeare character corpus 的字符 token 流，marker 仍随机插入。",
        ),
    }[version]


def version_specific_analysis(
    version: str,
    bundle: RunBundle,
    summaries: dict[tuple[str, str], dict[str, Any]],
) -> str:
    if version == "v11":
        parts = []
        for position in bundle.config["position_encodings"]:
            s = summaries[(version, position)]
            parts.append(
                f"<li><strong>{position.upper()}</strong>：AR final 为 non-thinking {pct(s['ar_nonthinking'])}、CoT {pct(s['ar_thinking'])}；"
                f"最强 k-to-k 候选为 {s['targeted']['head']}，raw mass={num(s['targeted'].get('correct_prompt_needle_mass'))}、"
                f"needle-subset top-1={pct(s['targeted'].get('correct_top1'))}。</li>"
            )
        return (
            "<ul>" + "".join(parts) + "</ul>"
            '<p class="analysis"><strong>解释。</strong>三种位置编码都能把最终 count 学到饱和，但学习速度与 trace-marker retrieval 不同。'
            "因此“最终分类做对”不能推出内部使用了同一种 retrieval circuit；位置编码比较必须同时看 AR trace、raw k-to-k mass 和 marker accuracy。</p>"
        )
    current = summaries[(version, "ape")]
    baseline = summaries[("v11", "ape")]
    if version == "v12":
        return (
            f'<p><strong>结果。</strong>扩到 512 / count 1–50 后，最终 AR accuracy 为 non-thinking {pct(current["ar_nonthinking"])}、'
            f'CoT {pct(current["ar_thinking"])}，CoT trace exact={pct(current["ar_trace_exact"])}。'
            f'最强 k-to-k raw mass={num(current["targeted"].get("correct_prompt_needle_mass"))}（v11-APE baseline '
            f'{num(baseline["targeted"].get("correct_prompt_needle_mass"))}）。</p>'
            '<p class="analysis"><strong>解释。</strong>这是 capacity/load stress test。若最终 count 仍高而 trace exact 或 k-to-k 下降，说明低容量模型可借助 teacher-forced progression 或别的聚合路径得到答案，不能据此宣称 targeted retrieval 保持不变。</p>'
        )
    if version == "v13":
        return (
            f'<p><strong>结果。</strong>固定训练集下，最终 AR accuracy 为 non-thinking {pct(current["ar_nonthinking"])}、CoT {pct(current["ar_thinking"])}；'
            f'最强 k-to-k raw mass={num(current["targeted"].get("correct_prompt_needle_mass"))}，v11-APE streaming baseline 为 '
            f'{num(baseline["targeted"].get("correct_prompt_needle_mass"))}。</p>'
            '<p class="analysis"><strong>解释。</strong>最终验证准确率只说明固定样本训练没有阻止任务求解；若 attention / state geometry 与 streaming 不同，则固定集可能改变的是实现算法而非最终函数。这里使用独立 balanced validation，不把训练集记忆当成泛化证据。</p>'
        )
    return (
        f'<p><strong>结果。</strong>Tiny Shakespeare haystack 下，最终 AR accuracy 为 non-thinking {pct(current["ar_nonthinking"])}、'
        f'CoT {pct(current["ar_thinking"])}，CoT trace exact={pct(current["ar_trace_exact"])}；最强 k-to-k raw mass='
        f'{num(current["targeted"].get("correct_prompt_needle_mass"))}，v11-APE uniform baseline 为 '
        f'{num(baseline["targeted"].get("correct_prompt_needle_mass"))}。</p>'
        '<p class="analysis"><strong>解释。</strong>字符 haystack 带有强重复与局部统计，和 uniform token noise 的“每个位置同分布”不同。更快学习或更强 marker retrieval 可能来自 marker/字符的可分性与较低背景熵，不能直接解释成模型理解了 Shakespeare 语义。</p>'
    )


def geometry_specific_analysis(bundle: RunBundle) -> str:
    """Summarize state geometry without reusing behavioral conclusions."""
    probe = bundle.tables["state_probe_summary"].copy()
    variance = bundle.tables["state_pca_variance"].copy()
    items: list[str] = []

    for position in bundle.config["position_encodings"]:
        subset = probe[
            (probe["position_encoding"] == position)
            & (probe["mode"] == "nonthinking")
            & (probe["site"] == "final_answer")
        ]
        if subset.empty:
            continue
        best = subset.sort_values("ridge_r2", ascending=False).iloc[0]
        pca = variance[
            (variance["position_encoding"] == position)
            & (variance["mode"] == "nonthinking")
            & (variance["site"] == "final_answer")
            & (variance["layer"] == best["layer"])
            & (variance["component"] == 6)
        ]
        coverage = (
            pct(pca.iloc[0]["cumulative_explained_variance"])
            if not pca.empty
            else "NA"
        )
        layer_label = (
            "embedding state"
            if int(best["layer"]) == 0
            else f"Layer {int(best['layer'])} 后"
        )
        items.append(
            f"<li><strong>{position.upper()}</strong>：non-thinking 最终答案 query 在 {layer_label} "
            f"达到最佳 ridge R²={num(best['ridge_r2'])}、MAE={num(best['ridge_mae'], 3)}；"
            f"nearest-centroid accuracy={pct(best['nearest_centroid_accuracy'])}，而 position-only baseline="
            f"{pct(best['position_only_accuracy'])}；同层 count centroids 的 PC1–PC6 累计覆盖为 {coverage}。</li>"
        )

    thinking = probe[probe["mode"] == "thinking"]
    confounded = thinking[thinking["position_only_accuracy"] >= 0.99]
    confound_text = (
        f"本次 CoT state 表中有 {len(confounded)}/{len(thinking)} 个 position-encoding × site × layer 条目 "
        "的 position-only accuracy ≥99%。"
        if not thinking.empty
        else "本次没有可用的 CoT state probe 条目。"
    )
    return (
        "<ul>" + "".join(items) + "</ul>"
        '<p class="analysis"><strong>最稳健的描述性结果。</strong>non-thinking 的 <code>&lt;Ans&gt;</code> 位置固定，'
        "position-only baseline 接近随机水平；因此其深层 residual state 对 count 的高可读性不能由答案 token 的绝对位置解释。"
        "这支持模型从 prompt 内容形成了低维 count representation，但仍未证明该 representation 对输出有因果作用。</p>"
        f'<p class="warning"><strong>CoT 几何的关键混杂。</strong>{confound_text}'
        "thinking final-answer、trace-index 与 trace-marker anchor 的位置随 count/progress 系统变化；此外 trace index token 本身也直接编码 k。"
        "因此这些 site 上接近 1 的 probe R² 或紧致 PCA 曲线，可能主要反映 token identity/sequence position，而不是独立的隐式计数器。"
        "后续必须用 fixed-position controls、state patching 或 steering 才能把可读性提升为因果证据。</p>"
    )


def report_asset_dir(bundle: RunBundle) -> Path:
    path = bundle.path / "report_assets_v2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _final_rows(frame: pd.DataFrame, position: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rows = final_group_rows(frame, ["position_encoding", "mode", "count_bin"])
    return rows[rows["position_encoding"] == position].copy()


def _last_train_losses(bundle: RunBundle, position: str) -> dict[str, float]:
    frame = bundle.tables["train_metrics"]
    frame = frame[frame["position_encoding"] == position]
    result: dict[str, float] = {}
    for mode in ("nonthinking", "thinking"):
        rows = frame[frame["mode"] == mode].sort_values("step")
        result[mode] = float(rows.iloc[-1]["train_total_loss"]) if not rows.empty else float("nan")
    return result


def convergence_frame(bundle: RunBundle, position: str) -> pd.DataFrame:
    tf = _final_rows(bundle.tables["eval_by_bin"], position)
    ar = _final_rows(bundle.tables["autoregressive_by_bin"], position)
    losses = _last_train_losses(bundle, position)

    def minimum(frame: pd.DataFrame, mode: str, metric: str) -> float:
        rows = frame[frame["mode"] == mode]
        return float(rows[metric].min()) if not rows.empty else float("nan")

    rows = [
        {
            "子任务": "Non-thinking final count",
            "训练末 loss": losses["nonthinking"],
            "最弱 count-bin TF accuracy": minimum(tf, "nonthinking", "tf_final_accuracy"),
            "最弱 count-bin AR accuracy": minimum(ar, "nonthinking", "ar_final_accuracy"),
            "最弱 count-bin trace marker": np.nan,
            "最弱 count-bin trace exact": np.nan,
        },
        {
            "子任务": "CoT final count",
            "训练末 loss": losses["thinking"],
            "最弱 count-bin TF accuracy": minimum(tf, "thinking", "tf_final_accuracy"),
            "最弱 count-bin AR accuracy": minimum(ar, "thinking", "ar_final_accuracy"),
            "最弱 count-bin trace marker": minimum(tf, "thinking", "tf_trace_marker_accuracy"),
            "最弱 count-bin trace exact": minimum(ar, "thinking", "trace_exact"),
        },
    ]
    return pd.DataFrame(rows)


def _style_axis(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)


def clean_learning_figures(bundle: RunBundle, position: str) -> dict[str, Path]:
    out = report_asset_dir(bundle)
    train = bundle.tables["train_metrics"]
    train = train[train["position_encoding"] == position].copy()
    eval_rows = bundle.tables["eval_by_bin"]
    eval_rows = eval_rows[eval_rows["position_encoding"] == position].copy()
    ar_rows = bundle.tables["autoregressive_by_bin"]
    ar_rows = ar_rows[ar_rows["position_encoding"] == position].copy()
    palette = plt.get_cmap("viridis")
    bins = list(dict.fromkeys(eval_rows["count_bin"].astype(str).tolist()))
    colors = {name: palette(i / max(1, len(bins) - 1)) for i, name in enumerate(bins)}

    loss_path = out / f"learning_loss_clean_{position}.png"
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    for mode, color in (("nonthinking", "#2563eb"), ("thinking", "#ea580c")):
        rows = train[train["mode"] == mode].sort_values("step")
        if not rows.empty:
            ax.plot(rows["step"], rows["train_total_loss"].clip(lower=1e-5), label=mode, color=color, linewidth=2.1)
    ax.set_yscale("log")
    _style_axis(ax, f"{position.upper()}: training loss", "optimizer step", "next-token cross-entropy (log scale)")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(loss_path, dpi=170, facecolor="white")
    plt.close(fig)

    final_path = out / f"learning_final_clean_{position}.png"
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.7), constrained_layout=True, sharey=True)
    for ax, mode in zip(axes, ("nonthinking", "thinking")):
        for count_bin in bins:
            tf_part = eval_rows[(eval_rows["mode"] == mode) & (eval_rows["count_bin"].astype(str) == count_bin)].sort_values("step")
            ar_part = ar_rows[(ar_rows["mode"] == mode) & (ar_rows["count_bin"].astype(str) == count_bin)].sort_values("step")
            if not tf_part.empty:
                ax.plot(tf_part["step"], tf_part["tf_final_accuracy"], color=colors[count_bin], linewidth=2, label=f"{count_bin} TF")
            if not ar_part.empty:
                ax.plot(ar_part["step"], ar_part["ar_final_accuracy"], color=colors[count_bin], linewidth=1.7, linestyle="--", label=f"{count_bin} AR")
        _style_axis(ax, mode, "training step", "final-count accuracy")
        ax.set_ylim(-0.03, 1.04)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(6, max(1, len(labels))), frameon=False)
    fig.suptitle(f"{position.upper()}: final-count learning dynamics", fontsize=16)
    fig.savefig(final_path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    trace_path = out / f"learning_trace_clean_{position}.png"
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.7), constrained_layout=True, sharey=True)
    for count_bin in bins:
        tf_part = eval_rows[(eval_rows["mode"] == "thinking") & (eval_rows["count_bin"].astype(str) == count_bin)].sort_values("step")
        ar_part = ar_rows[(ar_rows["mode"] == "thinking") & (ar_rows["count_bin"].astype(str) == count_bin)].sort_values("step")
        if not tf_part.empty:
            axes[0].plot(tf_part["step"], tf_part["tf_trace_marker_accuracy"], color=colors[count_bin], linewidth=2, label=count_bin)
        if not ar_part.empty:
            axes[1].plot(ar_part["step"], ar_part["trace_marker_recall"], color=colors[count_bin], linewidth=2, label=f"{count_bin} recall")
            axes[1].plot(ar_part["step"], ar_part["trace_exact"], color=colors[count_bin], linewidth=1.5, linestyle="--", label=f"{count_bin} exact")
    _style_axis(axes[0], "Teacher-forced local marker prediction", "training step", "marker-token accuracy")
    _style_axis(axes[1], "Autoregressive trace generation", "training step", "sequence metric")
    for ax in axes:
        ax.set_ylim(-0.03, 1.04)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(6, max(1, len(labels))), frameon=False)
    fig.suptitle(f"{position.upper()}: CoT trace convergence", fontsize=16)
    fig.savefig(trace_path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {"loss": loss_path, "final": final_path, "trace": trace_path}


def _head_matrix(frame: pd.DataFrame, metric: str) -> np.ndarray:
    layers = [0, 1, 2, 3]
    heads = [0, 1, 2, 3]
    normalized = frame.copy()
    numeric_layers = pd.to_numeric(normalized["layer"], errors="coerce")
    # Saved v11-v14 tables use human-readable Layer 1..4, while some older
    # diagnostics use internal layer indices 0..3. Normalize before reindexing
    # so Layer 4 is not silently dropped and Layer 1 is not rendered blank.
    if numeric_layers.notna().any() and numeric_layers.min() >= 1:
        numeric_layers = numeric_layers - 1
    normalized["layer"] = numeric_layers
    pivot = normalized.pivot_table(index="layer", columns="head", values=metric, aggfunc="mean")
    return pivot.reindex(index=layers, columns=heads).to_numpy(dtype=float)


def _draw_head_heatmap(ax: Any, values: np.ndarray, title: str, *, vmin: float = 0.0, vmax: float = 1.0) -> Any:
    image = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=12.5, pad=9)
    ax.set_xticks(range(4), range(4))
    ax.set_yticks(range(4), range(1, 5))
    ax.set_xlabel("head (0-based)")
    ax.set_ylabel("Layer (1-based)")
    for row in range(4):
        for col in range(4):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=9.5)
    return image


def clean_attention_figures(bundle: RunBundle, position: str) -> dict[str, Path]:
    out = report_asset_dir(bundle)
    attention = bundle.tables["attention_summary"]
    attention = attention[(attention["position_encoding"] == position) & (attention["count_bin"].astype(str) == "all")].copy()
    non_final = attention[(attention["mode"] == "nonthinking") & (attention["query_kind"] == "final_answer")]
    cot_final = attention[(attention["mode"] == "thinking") & (attention["query_kind"] == "final_answer")]
    trace = attention[(attention["mode"] == "thinking") & (attention["query_kind"] == "trace_index")]

    broad_path = out / f"attention_broad_clean_{position}.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), constrained_layout=True)
    for ax, frame, title in ((axes[0], non_final, "Non-thinking final query"), (axes[1], cot_final, "CoT final query")):
        image = _draw_head_heatmap(ax, _head_matrix(frame, "broad_attention_score"), title)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025, label="broad-attention score")
    fig.suptitle(f"{position.upper()}: broad prompt-needle aggregation candidates", fontsize=15)
    fig.savefig(broad_path, dpi=175, facecolor="white")
    plt.close(fig)

    targeted_path = out / f"attention_targeted_clean_{position}.png"
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("correct_prompt_needle_mass", "correct_top1", "diagonal_dominance"),
        ("raw k-to-k mass", "needle-subset correct top-1", "diagonal dominance"),
    ):
        image = _draw_head_heatmap(ax, _head_matrix(trace, metric), title)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    fig.suptitle(f"{position.upper()}: targeted k-to-k retrieval diagnostics", fontsize=15)
    fig.savefig(targeted_path, dpi=175, facecolor="white")
    plt.close(fig)

    readout_path = out / f"attention_readout_clean_{position}.png"
    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    image = _draw_head_heatmap(ax, _head_matrix(cot_final, "trace_markers_mass"), "CoT final-answer query")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025, label="mass on all trace-marker tokens")
    fig.suptitle(f"{position.upper()}: trace-marker readout candidates", fontsize=15)
    fig.savefig(readout_path, dpi=175, facecolor="white")
    plt.close(fig)
    return {"broad": broad_path, "targeted": targeted_path, "readout": readout_path}


def clean_probe_figure(bundle: RunBundle, position: str) -> Path:
    out = report_asset_dir(bundle)
    path = out / f"state_probe_clean_v2_{position}.png"
    frame = bundle.tables["state_probe_summary"]
    frame = frame[frame["position_encoding"] == position].copy()
    frame["site_label"] = frame["mode"] + " | " + frame["site"]
    order = list(dict.fromkeys(frame["site_label"].tolist()))
    metrics = (("ridge_r2", "Ridge R2", -0.2, 1.0), ("nearest_centroid_accuracy", "Nearest-centroid accuracy", 0.0, 1.0), ("position_only_accuracy", "Position-only baseline", 0.0, 1.0))
    fig, axes = plt.subplots(1, 3, figsize=(16.0, max(4.5, 0.48 * len(order) + 2.0)), constrained_layout=True)
    for index, (ax, (metric, title, vmin, vmax)) in enumerate(zip(axes, metrics)):
        pivot = frame.pivot_table(index="site_label", columns="layer", values=metric, aggfunc="mean").reindex(index=order, columns=[0, 1, 2, 3, 4])
        values = pivot.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=12.5)
        ax.set_xticks(range(5), ["embed", "L1", "L2", "L3", "L4"])
        ax.set_xlabel("residual-stream checkpoint")
        ax.set_yticks(range(len(order)), order if index == 0 else [""] * len(order))
        if index == 0:
            ax.set_ylabel("mode | semantic site")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if np.isfinite(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if value < (vmin + vmax) / 2 else "black", fontsize=8.8)
        fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f"{position.upper()}: count/progress state decodability", fontsize=15)
    fig.savefig(path, dpi=175, facecolor="white")
    plt.close(fig)
    return path


def clean_pca_variance_figure(bundle: RunBundle, position: str) -> Path:
    """Render cumulative centroid-PCA variance without a crowded shared axis."""
    out = report_asset_dir(bundle)
    path = out / f"state_pca_variance_clean_{position}.png"
    frame = bundle.tables["state_pca_variance"]
    frame = frame[frame["position_encoding"] == position].copy()
    frame["site_label"] = frame["mode"] + " | " + frame["site"]
    site_order = list(dict.fromkeys(frame["site_label"].tolist()))

    ncols = 2
    nrows = max(1, int(np.ceil(len(site_order) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 4.1 * nrows), sharex=True, sharey=True, constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, 5))
    layer_labels = {0: "embedding", 1: "after L1", 2: "after L2", 3: "after L3", 4: "after L4"}
    for ax, site_label in zip(axes_array, site_order):
        rows = frame[frame["site_label"] == site_label]
        for layer, color in zip(range(5), colors):
            part = rows[rows["layer"] == layer].sort_values("component")
            if part.empty:
                continue
            ax.plot(
                part["component"],
                part["cumulative_explained_variance"],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=color,
                label=layer_labels[layer],
            )
        ax.set_title(site_label, fontsize=12.5)
        ax.set_xticks(range(1, 7))
        ax.set_ylim(0.0, 1.03)
        ax.grid(alpha=0.22)
        ax.set_xlabel("number of centroid PCs retained")
        ax.set_ylabel("cumulative explained variance")
    for ax in axes_array[len(site_order):]:
        ax.set_visible(False)
    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle(f"{position.upper()}: cumulative PCA variance of exact-count/progress centroids", fontsize=15)
    fig.savefig(path, dpi=175, facecolor="white")
    plt.close(fig)
    return path


def _plotly_source() -> str:
    candidates = [
        Path(r"C:\anaconda3\Lib\site-packages\plotly\package_data\plotly.min.js"),
        Path(r"C:\Users\HP\anaconda3\Lib\site-packages\plotly\package_data\plotly.min.js"),
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError("Could not locate a local plotly.min.js for standalone interactive PCA plots")


def interactive_pca_html(bundle: RunBundle) -> str:
    frame = bundle.tables["state_centroids_pca"].copy()
    if frame.empty:
        return '<div class="warning">Missing state_centroids_pca.csv; interactive PCA is unavailable.</div>'
    columns = ["position_encoding", "mode", "site", "layer", "state_label", "pc1", "pc2", "pc3", "pc4", "pc5", "pc6"]
    rows = frame[columns].replace({np.nan: None}).to_dict(orient="records")
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    plotly = _plotly_source()
    uid = f"pca-{bundle.version}"
    return f"""
<div class="interactive-card" id="{uid}-card">
  <h3>Interactive 3D PCA centroid manifold</h3>
  <p>每个点是一个 exact-count 或 trace-progress label 的平均 hidden-state centroid，不是单个样本。使用下方控件切换位置编码、mode、语义位置、Layer，以及 PC1-PC6 中任意三个不同坐标轴。颜色表示 <code>state_label</code> 的数值顺序；鼠标悬停可查看标签和坐标。</p>
  <div class="controls">
    <label>Model / position encoding<select id="{uid}-pe"></select></label>
    <label>Mode<select id="{uid}-mode"></select></label>
    <label>Semantic site<select id="{uid}-site"></select></label>
    <label>Layer<select id="{uid}-layer"></select></label>
    <label>X axis<select id="{uid}-x"></select></label>
    <label>Y axis<select id="{uid}-y"></select></label>
    <label>Z axis<select id="{uid}-z"></select></label>
  </div>
  <div id="{uid}-message" class="muted"></div>
  <div id="{uid}-plot" class="pca-plot"></div>
</div>
<script>{plotly}</script>
<script>
(() => {{
  const rows = {payload};
  const id = "{uid}";
  const el = name => document.getElementById(id + "-" + name);
  const unique = values => [...new Set(values.map(String))];
  const setOptions = (select, values, preferred) => {{
    const prior = select.value;
    select.innerHTML = "";
    values.forEach(value => {{ const option = document.createElement("option"); option.value = value; option.textContent = value; select.appendChild(option); }});
    select.value = values.includes(prior) ? prior : (values.includes(preferred) ? preferred : values[0]);
  }};
  setOptions(el("pe"), unique(rows.map(r => r.position_encoding)), "ape");
  ["pc1","pc2","pc3","pc4","pc5","pc6"].forEach(pc => {{
    ["x","y","z"].forEach(axis => {{ if (![...el(axis).options].some(o => o.value === pc)) {{ const option = document.createElement("option"); option.value = pc; option.textContent = pc.toUpperCase(); el(axis).appendChild(option); }} }});
  }});
  el("x").value = "pc1"; el("y").value = "pc2"; el("z").value = "pc3";
  function refreshSelectors() {{
    const peRows = rows.filter(r => String(r.position_encoding) === el("pe").value);
    setOptions(el("mode"), unique(peRows.map(r => r.mode)), "nonthinking");
    const modeRows = peRows.filter(r => String(r.mode) === el("mode").value);
    setOptions(el("site"), unique(modeRows.map(r => r.site)), "final_answer");
    const siteRows = modeRows.filter(r => String(r.site) === el("site").value);
    setOptions(el("layer"), unique(siteRows.map(r => r.layer)).sort((a,b) => Number(a)-Number(b)), "4");
  }}
  function draw() {{
    refreshSelectors();
    const axes = [el("x").value, el("y").value, el("z").value];
    if (new Set(axes).size < 3) {{ el("message").textContent = "X/Y/Z 必须选择三个不同的 principal components。"; return; }}
    el("message").textContent = "";
    const chosen = rows.filter(r => String(r.position_encoding) === el("pe").value && String(r.mode) === el("mode").value && String(r.site) === el("site").value && String(r.layer) === el("layer").value);
    const labels = chosen.map(r => String(r.state_label));
    const numeric = labels.map((label, index) => Number.isFinite(Number(label)) ? Number(label) : index);
    const trace = {{
      type: "scatter3d", mode: "markers+text",
      x: chosen.map(r => r[axes[0]]), y: chosen.map(r => r[axes[1]]), z: chosen.map(r => r[axes[2]]),
      text: labels, textposition: "top center", hovertext: labels,
      hovertemplate: "%{{hovertext}}<br>" + axes[0].toUpperCase() + "=%{{x:.3f}}<br>" + axes[1].toUpperCase() + "=%{{y:.3f}}<br>" + axes[2].toUpperCase() + "=%{{z:.3f}}<extra></extra>",
      marker: {{size: 6, color: numeric, colorscale: "Viridis", showscale: true, colorbar: {{title: "state label"}}}}
    }};
    const title = [el("pe").value.toUpperCase(), el("mode").value, el("site").value, "Layer " + el("layer").value].join(" | ");
    Plotly.react(el("plot"), [trace], {{title, margin: {{l:0,r:0,b:0,t:55}}, scene: {{xaxis:{{title:axes[0].toUpperCase()}}, yaxis:{{title:axes[1].toUpperCase()}}, zaxis:{{title:axes[2].toUpperCase()}}}}, paper_bgcolor:"white"}}, {{responsive:true, displaylogo:false}});
  }}
  ["pe","mode","site","layer","x","y","z"].forEach(name => el(name).addEventListener("change", draw));
  refreshSelectors(); draw();
}})();
</script>
"""


def convergence_analysis(bundle: RunBundle, position: str) -> str:
    frame = convergence_frame(bundle, position)
    non = frame.iloc[0]
    cot = frame.iloc[1]
    non_ok = bool(non["最弱 count-bin AR accuracy"] >= 0.99)
    cot_final_ok = bool(cot["最弱 count-bin AR accuracy"] >= 0.99)
    marker = float(cot["最弱 count-bin trace marker"])
    exact = float(cot["最弱 count-bin trace exact"])
    if marker >= 0.99 and exact >= 0.99:
        trace_text = (
            "CoT 的局部 marker prediction 与整条 autoregressive trace 都达到 99%，"
            "因此该变体在本报告测量范围内可以视为完整收敛。"
        )
    elif marker >= 0.95:
        trace_text = (
            f"CoT 的最弱区间局部 marker accuracy 已到 {pct(marker)}，但整条 trace exact 只有 {pct(exact)}。"
            "这表示局部错误率虽低，长 trace 中任一位置出错都会让 sequence-exact 归零；"
            "不能把 final-count 饱和写成完整 CoT 已收敛。"
        )
    else:
        trace_text = (
            f"CoT 的最弱区间局部 marker accuracy 只有 {pct(marker)}，trace exact 为 {pct(exact)}。"
            "这里不只是 sequence-exact 的长度惩罚，content-addressed marker retrieval 本身仍未学稳。"
        )
    final_text = (
        f"Non-thinking final count {'已' if non_ok else '未'}在所有 count 区间达到 99%；"
        f"CoT final count {'已' if cot_final_ok else '未'}在所有区间达到 99%。"
    )
    return f'<div class="analysis"><strong>收敛审计。</strong>{final_text}{trace_text}</div>'


def architecture_or_data_analysis(
    bundle: RunBundle,
    bundles: dict[str, RunBundle],
    summaries: dict[tuple[str, str], dict[str, Any]],
) -> str:
    version = bundle.version
    if version == "v11":
        rows = []
        for position in bundle.config["position_encodings"]:
            summary = summaries[(version, position)]
            convergence = convergence_frame(bundle, position).iloc[1]
            rows.append(
                {
                    "位置编码": position.upper(),
                    "CoT 末 loss": num(convergence["训练末 loss"]),
                    "最弱区间 TF marker": pct(convergence["最弱 count-bin trace marker"]),
                    "最弱区间 AR trace exact": pct(convergence["最弱 count-bin trace exact"]),
                    "最强 k-to-k raw mass": num(summary["targeted"].get("correct_prompt_needle_mass"), 4),
                    "needle-subset top-1": pct(summary["targeted"].get("correct_top1")),
                    "diagonal dominance": num(summary["targeted"].get("diagonal_dominance"), 4),
                }
            )
        return (
            "<h3>v11 核心比较：位置编码如何改变 targeted retrieval</h3>"
            + table_html(pd.DataFrame(rows))
            + '<div class="analysis"><strong>结果。</strong>APE 与 RoPE 都能把 final count 学到近乎饱和，'
            "但显式 trace retrieval 仍明显欠拟合：APE 的 raw k-to-k mass 极低，RoPE 虽提高绝对 mass，"
            "仍未形成稳定的整条 trace。learned RPE 的 matching-needle mass、needle-subset top-1、"
            "局部 marker accuracy 与 trace exact 同时显著更高。<br><strong>解释。</strong>"
            "这说明在 d=64 的低容量模型中，位置表示不是无关紧要的实现细节。"
            "learned relative-position bias 给 attention logits 提供了直接的相对位置信号，更容易形成"
            "“第 k 个 trace query 对齐第 k 个 needle”的路由；APE/RoPE 则仍可用其他聚合/分类路径完成 final count。"
            "这些是描述性关联，尚不能单独证明 RPE head 对输出的因果必要性。</div>"
        )

    baseline = bundles["v11"]
    base_summary = summaries[("v11", "ape")]
    current = summaries[(version, "ape")]
    base_conv = convergence_frame(baseline, "ape").iloc[1]
    cur_conv = convergence_frame(bundle, "ape").iloc[1]

    def state_readout(summary: dict[str, Any], key: str) -> str:
        state = summary["states"].get(key, {})
        return f"{num(state.get('ridge_r2'))} / {pct(state.get('position_only'))}"

    rows = pd.DataFrame(
        [
            {
                "训练条件": "v11 APE streaming uniform baseline",
                "有效训练集": "每 step 重新采样",
                "haystack": "uniform synthetic tokens",
                "最弱 TF marker": pct(base_conv["最弱 count-bin trace marker"]),
                "最弱 AR trace exact": pct(base_conv["最弱 count-bin trace exact"]),
                "k-to-k raw mass": num(base_summary["targeted"].get("correct_prompt_needle_mass"), 4),
                "Non-thinking final R² / pos baseline": state_readout(base_summary, "nonthinking:final_answer"),
                "CoT trace-marker R² / pos baseline": state_readout(base_summary, "thinking:trace_marker"),
            },
            {
                "训练条件": version,
                "有效训练集": (
                    "固定 15,360 样本，反复约 83.3 epochs"
                    if version == "v13"
                    else "每 step 重新采样"
                ),
                "haystack": (
                    "standard Tiny Shakespeare characters"
                    if version == "v14"
                    else "uniform synthetic tokens"
                ),
                "最弱 TF marker": pct(cur_conv["最弱 count-bin trace marker"]),
                "最弱 AR trace exact": pct(cur_conv["最弱 count-bin trace exact"]),
                "k-to-k raw mass": num(current["targeted"].get("correct_prompt_needle_mass"), 4),
                "Non-thinking final R² / pos baseline": state_readout(current, "nonthinking:final_answer"),
                "CoT trace-marker R² / pos baseline": state_readout(current, "thinking:trace_marker"),
            },
        ]
    )
    if version == "v13":
        text = (
            "固定数据训练仍把 final count 学到 100%，但 trace exact 保持为 0，局部 marker 与 raw k-to-k mass 也弱于"
            "能够形成显式 routing 的变体。模型反复看到有限 prompt 后，可以利用训练集特有的位置/共现捷径或记忆"
            "来完成低熵 final classification，而不必形成可泛化的逐项 retrieval circuit。"
            "验证集是独立生成的，所以 final count 的成功不是直接复述训练标签；但 attention 与 state geometry 的差异"
            "说明 fixed-dataset training 改变了模型采用的内部算法。表中的 state 列同时报告最佳 ridge R² 与"
            "position-only accuracy；只有 R² 高而位置基线低时，才更像内容驱动的 count representation。"
        )
    elif version == "v14":
        text = (
            "Tiny Shakespeare 字符 haystack 下，final count、局部 marker、整条 trace 与 k-to-k routing 同时收敛。"
            "与 i.i.d. uniform noise 相比，字符流有重复、局部统计和较小的有效背景支持集，而人工 marker 与普通字符"
            "高度可分；这降低了“从背景中识别并路由 marker”的难度。"
            "因此结果支持数据分布会显著改变机制可学习性，并会改变 final/trace anchor 上 count state 的可读性；"
            "但 CoT anchor 的绝对位置随 count 变化，表中 position baseline 很高时仍不能把高 R² 当成独立隐式计数器，"
            "更不能解释为模型理解 Shakespeare 语义。"
        )
    else:
        text = (
            "v12 同时把上下文扩到 512、count 扩到 50，因此不是纯数据分布对照。局部 marker accuracy 已较高，"
            "但长 trace 的 sequence-exact 仍为 0；主要瓶颈是更长检索链上的误差累积与有限 d=64 容量。"
        )
    return "<h3>相对 v11-APE 的受控比较</h3>" + table_html(rows) + f'<div class="analysis"><strong>结果与解释。</strong>{text}</div>'


def build_report(
    bundle: RunBundle,
    bundles: dict[str, RunBundle],
    summaries: dict[tuple[str, str], dict[str, Any]],
    comparison: pd.DataFrame,
) -> str:
    version = bundle.version
    cfg = bundle.config
    question, question_detail = current_question(version)
    settings = pd.DataFrame([config_row(bundles[item], current=version) for item in VERSIONS])
    model_specs = bundle.tables["model_specifications"].copy()
    model_specs = model_specs.rename(
        columns={
            "position_encoding": "位置编码",
            "mode": "模型",
            "parameters": "参数量",
            "n_layer": "layers",
            "n_head": "heads/layer",
            "n_embd": "d_model",
            "n_inner": "MLP",
        }
    )

    learning_assets: dict[str, dict[str, Path]] = {}
    attention_assets: dict[str, dict[str, Path]] = {}
    position_sections = []
    attention_sections = []
    state_sections = []
    for position in cfg["position_encodings"]:
        label = position.upper()
        learning_assets[position] = clean_learning_figures(bundle, position)
        attention_assets[position] = clean_attention_figures(bundle, position)
        position_sections.append(
            f"<h3>{label}</h3>"
            + table_html(convergence_frame(bundle, position).assign(
                **{
                    "训练末 loss": lambda x: x["训练末 loss"].map(lambda value: num(value, 4)),
                    "最弱 count-bin TF accuracy": lambda x: x["最弱 count-bin TF accuracy"].map(pct),
                    "最弱 count-bin AR accuracy": lambda x: x["最弱 count-bin AR accuracy"].map(pct),
                    "最弱 count-bin trace marker": lambda x: x["最弱 count-bin trace marker"].map(pct),
                    "最弱 count-bin trace exact": lambda x: x["最弱 count-bin trace exact"].map(pct),
                }
            ))
            + convergence_analysis(bundle, position)
            + figure_grid(
                [
                    figure(
                        learning_assets[position]["loss"],
                        f"{label}：训练 loss",
                        "横轴是 optimizer step；纵轴是受监督 completion token 的平均 next-token cross-entropy，使用对数刻度。蓝线为 non-thinking，橙线为 CoT。CoT completion 更长且包含随机 marker retrieval，不能只用两条 loss 的绝对值判断两种模型谁更好。",
                        compact=True,
                    ),
                    figure(
                        learning_assets[position]["final"],
                        f"{label}：final-count 学习动态",
                        "左右面板分别是 non-thinking 与 CoT。横轴为 training step，纵轴为 final-count accuracy；实线是 teacher forcing，虚线是完整 autoregressive generation。颜色区分 count 区间。",
                        compact=True,
                    ),
                ],
                columns=2,
            )
            + figure(
                learning_assets[position]["trace"],
                f"{label}：CoT trace 收敛",
                "左图是给定 gold prefix 时的局部 marker-token accuracy；右图是自由生成时的 marker recall（实线）和整条 trace exact（虚线）。横轴为 training step，纵轴为 [0,1] 指标值。局部预测接近 1 仍可能因为长 trace 的误差累积而得到很低的 sequence exact。",
                compact=True,
            )
            + table_html(time_to_99_table(bundle, position))
            + "<h4>最终 checkpoint：teacher forcing 与自由生成</h4>"
            + table_html(final_eval_table(bundle, position))
        )
        attention_sections.append(
            f"<h3>{label}</h3>"
            + table_html(attention_table(bundle, position))
            + figure_grid(
                [
                    figure(
                        attention_assets[position]["broad"],
                        f"{label}：broad prompt-needle aggregation candidates",
                        "左图在 non-thinking 最终答案 query、右图在 CoT 最终答案 query 计算 broad score。横轴为 head 0–3，纵轴为 Layer 1–4；单元格是该 head 的 prompt-needle 总 mass 乘以 needle 子集归一化熵。高分要求既看向 needles，又在多个 needles 间分布。",
                        compact=True,
                    ),
                    figure(
                        attention_assets[position]["readout"],
                        f"{label}：CoT trace-marker readout candidates",
                        "横轴为 head，纵轴为 Layer；每格是最终答案 query 投向所有 trace-marker token 的 attention mass。它寻找可能从已经给出的 trace 读取 count/progress 的 heads，不等同于 prompt retrieval。",
                        compact=True,
                    ),
                ],
                columns=2,
            )
            + figure(
                attention_assets[position]["targeted"],
                f"{label}：targeted k-to-k retrieval 的三个互补指标",
                "三幅热图横轴都是 head 0–3，纵轴都是 Layer 1–4。左为 matching prompt needle 获得的绝对 raw mass；中为只在 needle 子集内判断 matching needle 是否 top-1；右为 matching mass 占全部 needle mass 的比例。只有三者同时较高，才能称为强而且专一的 targeted retrieval。",
                compact=True,
            )
        )
        state_sections.append(
            f"<h3>{label}</h3>"
            + table_html(state_table(bundle, position))
            + figure(
                clean_probe_figure(bundle, position),
                f"{label}：count/progress 可读性与 position baseline",
                "三张热图横轴都是 state index：0 是 token+position embedding，1–4 是经过对应 Layer 后的 residual stream；纵轴为模型模式与 semantic site。左为 held-out nearest-centroid exact-label accuracy，中为只用绝对 token position 的 baseline，右为 ridge regression R²。",
                compact=True,
            )
            + figure(
                clean_pca_variance_figure(bundle, position),
                f"{label}：centroid PCA 累计解释方差",
                "四个面板分别对应 non-thinking final answer、CoT final answer、CoT trace index 与 CoT trace marker。横轴是保留的 centroid principal components 数量（1–6），纵轴是累计 explained-variance ratio；颜色区分 embedding 与经过 Layer 1–4 后的 residual state。这里只分析不同 count/progress 类均值之间的几何，不是所有单样本方差。",
                compact=True,
            )
        )

    common_params = (
        f"seed={cfg['seed']}；10,000 steps；batch={cfg['batch_size']}；AdamW lr={cfg['lr']}；"
        f"warmup={cfg['warmup_steps']}；weight decay={cfg['weight_decay']}；每 {cfg['eval_every']} steps 做 teacher-forced eval，"
        f"每 {cfg['ar_eval_every']} steps 做 autoregressive eval。"
    )

    css = """
    :root { --ink:#172033; --muted:#5e6b80; --line:#dbe3ef; --blue:#2563eb; --green:#159957; --amber:#d97706; --paper:#ffffff; --bg:#f3f6fa; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--bg); font:16px/1.72 Inter, "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif; }
    main { max-width:1380px; margin:0 auto; background:var(--paper); padding:42px 56px 80px; box-shadow:0 0 28px rgba(25,45,75,.08); }
    h1 { font-size:2.2rem; line-height:1.2; margin:0 0 10px; letter-spacing:0; }
    h2 { font-size:1.58rem; margin:54px 0 20px; padding-top:18px; border-top:1px solid var(--line); letter-spacing:0; }
    h3 { font-size:1.2rem; margin:30px 0 14px; }
    h4 { font-size:1rem; margin:0 0 12px; }
    p { margin:10px 0 16px; }
    code { background:#edf3fb; padding:.1em .38em; border-radius:4px; font-family:Consolas, monospace; }
    .eyebrow { color:var(--blue); font-weight:750; text-transform:uppercase; font-size:.8rem; letter-spacing:.08em; }
    .subtitle { color:var(--muted); font-size:1.05rem; max-width:1000px; }
    .hero { border-left:5px solid var(--blue); padding:10px 0 10px 22px; margin-bottom:24px; }
    .cards { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
    .card, .note, .analysis, .warning { border:1px solid var(--line); border-radius:7px; padding:18px 20px; background:#fbfdff; }
    .card h3 { margin-top:0; }
    .analysis { border-left:5px solid var(--green); background:#f1fbf5; }
    .warning { border-left:5px solid var(--amber); background:#fff9ea; }
    .muted { color:var(--muted); }
    .sequence { font:600 15px/1.7 Consolas, monospace; padding:14px 18px; background:#f4f7fb; border:1px solid var(--line); border-radius:6px; overflow-wrap:anywhere; }
    .formula-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
    .formula { border:1px solid var(--line); border-radius:7px; padding:16px 18px; background:#fff; }
    .formula .math { display:block; text-align:center; font-family:Cambria Math, STIX Two Math, serif; font-size:1.18rem; margin:8px 0; }
    table.data-table { width:100%; border-collapse:collapse; margin:12px 0 22px; font-size:.91rem; }
    table.data-table th { background:#eaf0f8; text-align:left; font-weight:750; }
    table.data-table th, table.data-table td { border:1px solid #d7e0ec; padding:9px 10px; vertical-align:top; }
    table.data-table tr:nth-child(even) td { background:#f9fbfd; }
    .figure { margin:22px 0 30px; border:1px solid var(--line); border-radius:8px; padding:18px; background:white; overflow:hidden; }
    .figure img { display:block; width:100%; height:auto; max-height:820px; object-fit:contain; margin:0 auto; }
    .figure.compact img { max-height:570px; }
    figcaption { color:var(--muted); margin-top:13px; font-size:.92rem; }
    .figure-grid { display:grid; gap:16px; align-items:start; }
    .figure-grid.cols-2 { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .figure-grid.cols-3 { grid-template-columns:repeat(3,minmax(0,1fr)); }
    .interactive-card { margin:24px 0; border:1px solid var(--line); border-radius:8px; padding:20px; background:#fff; }
    .interactive-card h3 { margin-top:0; }
    .controls { display:grid; grid-template-columns:repeat(4,minmax(150px,1fr)); gap:12px; margin:16px 0; }
    .controls label { color:var(--muted); font-size:.84rem; font-weight:700; }
    .controls select { display:block; width:100%; margin-top:5px; padding:8px; border:1px solid #cbd5e1; border-radius:5px; background:#fff; color:var(--ink); }
    .pca-plot { width:100%; height:680px; min-height:520px; }
    details { border:1px solid var(--line); border-radius:7px; padding:12px 15px; margin:18px 0; }
    summary { cursor:pointer; font-weight:750; }
    .toc { columns:2; padding:18px 26px; background:#f7f9fc; border:1px solid var(--line); border-radius:7px; }
    .toc a { color:#244b84; text-decoration:none; }
    @media (max-width:900px) { main{padding:28px 18px 60px}.cards,.formula-grid,.figure-grid.cols-2,.figure-grid.cols-3,.controls{grid-template-columns:1fr}.toc{columns:1}.pca-plot{height:560px} }
    @media print { body{background:#fff}main{box-shadow:none;max-width:none}.figure,table{break-inside:avoid}details{display:block}details>*{display:block} }
    """

    version_title = {
        "v11": "Position Encoding Comparison at Small Capacity",
        "v12": "Longer Context and Count-50 Capacity Stress",
        "v13": "Fixed-Dataset Training versus Streaming",
        "v14": "Tiny Shakespeare Character Haystack",
    }[version]

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trace Count {version} Report</title><style>{css}</style></head><body><main>
<header class="hero"><div class="eyebrow">Synthetic NIAH Counting · {version}</div>
<h1>Trace Count {version}：{version_title}</h1>
<p class="subtitle">参考 v10 第 1–6 节的独立审计报告。所有结论来自 <code>{esc(bundle.path.name)}</code> 保存的正式 seed-1234 结果；报告内图片使用 base64 嵌入，可脱离原目录单独发送。</p></header>

<nav class="toc"><strong>目录</strong><br>
<a href="#s1">1. 研究对象与版本问题</a><br><a href="#s2">2. 模型、数据与训练</a><br>
<a href="#s3">3. 指标与计算定义</a><br><a href="#s4">4. 学习动态与最终行为</a><br>
<a href="#s5">5. 描述性 attention</a><br><a href="#s6">6. 描述性 hidden-state geometry</a></nav>

<section id="s1"><h2>1. 研究对象、两个工作机制与 {version} 的核心问题</h2>
<div class="cards"><div class="card"><h3>Non-thinking：直接集合聚合</h3>
<p>模型只看到 prompt 后的 <code>&lt;Ans&gt;</code>，需要把分散在 haystack 中的 marker/needle 集合直接压缩成 scalar count。工作假设是：早期 broad heads 在多个 needles 间分配 attention，后续 residual/MLP 将集合证据写成可读 count state。</p></div>
<div class="card"><h3>Thinking：逐项 targeted retrieval</h3>
<p>模型在第 <code>&lt;k&gt;</code> 个 trace index 位置，定位 prompt 中按位置排序的第 k 个 needle，复制其 marker，并在最后从 trace/progress state 读出 count。工作假设是 targeted k-to-k routing 与 final trace readout 分阶段实现。</p></div></div>
<div class="warning"><strong>证据边界。</strong>本报告只覆盖 v10 的前六节，因此 attention mass、probe 与 PCA 都是描述性证据，不证明 head/state 对输出有因果必要性或充分性；这需要后续 ablation、patching 与 steering。</div>
<h3>{esc(question)}</h3><p>{esc(question_detail)}</p>
<h3>四个版本的受控差异</h3>{table_html(settings)}
<h3>横向结果摘要</h3>{table_html(comparison)}
{architecture_or_data_analysis(bundle, bundles, summaries)}
</section>

<section id="s2"><h2>2. 模型、数据生成、训练目标与 sequence</h2>
<p><strong>共同模型。</strong>随机初始化、decoder-only、pre-LN causal Transformer；4 Layers × 4 heads，d_model=64，MLP=256，token embedding 与 unembedding tied。Non-thinking 与 thinking 是两个独立 Transformer，不共享参数。{esc(common_params)}</p>
{table_html(model_specs)}
<div class="sequence">non-thinking: &lt;BOS&gt; prompt[0:{cfg['seq_len']}] &lt;Ans&gt; &lt;C_n&gt; &lt;EOS&gt;</div>
<div class="sequence">thinking: &lt;BOS&gt; prompt[0:{cfg['seq_len']}] &lt;Think&gt; &lt;1&gt; M₁ … &lt;n&gt; Mₙ &lt;/Think&gt; &lt;Ans&gt; &lt;C_n&gt; &lt;EOS&gt;</div>
<p>两个模式共享同一套数字 token：trace index <code>&lt;k&gt;</code> 与最终 count <code>&lt;C_k&gt;</code> 是同一个 vocabulary item。Non-thinking loss 只监督 final count 与 EOS；thinking loss 监督完整 completion（trace indices、trace markers、关闭 token、final count 与 EOS）。marker identity 从 {cfg['marker_vocab_size']} 类中有放回采样，needle positions 在 prompt 内无放回均匀采样。</p>
<h3>{version} 特有 setting</h3><p>{esc(question_detail)}</p>
</section>

<section id="s3"><h2>3. 新术语、数据列与计算公式</h2>
<div class="formula-grid">
<div class="formula"><strong>Teacher-forced final accuracy</strong><span class="math">Acc<sub>TF</sub> = mean[ argmax p(C | gold prefix) = C<sub>gold</sub> ]</span><p>每个待预测 token 前都提供 gold prefix。它测局部 next-token readout，不等同于自由生成。</p></div>
<div class="formula"><strong>Autoregressive final accuracy</strong><span class="math">Acc<sub>AR</sub> = mean[ Ĉ<sub>generated</sub> = C<sub>gold</sub> ]</span><p>模型从 prompt 开始自己生成 completion；早期 trace 错误会向后传播。</p></div>
<div class="formula"><strong>Broad attention score</strong><span class="math">B = M(N | q) × H(A<sub>q,N</sub>) / log |N|</span><p>M(N|q) 是 query 投向全部 prompt needles 的总 mass；第二项是 needle 子集内归一化熵。高分同时要求“看 needles”且“在多个 needles 间广泛分布”。</p></div>
<div class="formula"><strong>k-to-k raw mass</strong><span class="math">K<sub>raw</sub>(k) = A[q=&lt;k&gt;, needle<sub>k</sub>]</span><p>CoT 第 k 个 index query 对 matching prompt needle 的绝对 attention mass。</p></div>
<div class="formula"><strong>Diagonal dominance</strong><span class="math">D(k) = A[q<sub>k</sub>, needle<sub>k</sub>] / Σ<sub>j∈N</sub>A[q<sub>k</sub>,j]</span><p>只在 needle 子集内部归一化；D 高但 raw mass 低时，head 可能仍把绝大多数 attention 放在 noise/BOS。</p></div>
<div class="formula"><strong>Needle-subset correct top-1</strong><span class="math">Top1(k)=1[argmax<sub>j∈N</sub>A[q<sub>k</sub>,j]=needle<sub>k</sub>]</span><p>只比较 prompt needles，不要求 matching needle 是整个上下文 top-1。</p></div>
<div class="formula"><strong>Trace exact / marker recall</strong><span class="math">Exact = 1[generated trace = gold trace]</span><p>trace exact 要整条 index-marker 序列完全一致；marker recall 是 gold marker positions 中生成正确的比例。</p></div>
<div class="formula"><strong>State probe 与 PCA</strong><span class="math">R² = 1 − SSE / SST；EVR<sub>r</sub> = λ<sub>r</sub> / Σλ</span><p>ridge 在 held-out states 上回归 count/progress；PCA 只作用于每个 label 的 mean hidden-state centroid，描述类均值几何。</p></div>
</div></section>

<section id="s4"><h2>4. 学习动态：什么时候学会 final count，什么时候学会完整 trace？</h2>
<p>训练曲线的横轴均为 optimizer step。loss 图纵轴是当前 completion 中受监督 token 的平均 next-token cross-entropy；accuracy 图按 count bin 分组。达到 99% 的时间以每 500-step 的离散 eval checkpoint 记录，因此只能解释为“最迟在该 checkpoint 已达到”。</p>
{''.join(position_sections)}
<h3>结果与版本差异</h3>{version_specific_analysis(version, bundle, summaries)}
<p class="analysis"><strong>为什么 thinking final count 通常学得比 trace marker 快？</strong>最终 count 是低熵的平衡分类，并且 teacher forcing 给出了完整 gold trace；模型可以从 trace 长度、最后 index 或 final marker state 读取 count。trace index 又是确定的 1→2→… progression。相比之下，trace marker 必须从长 prompt 的正确 needle 位置检索随机 marker identity，是更难的 content-addressed retrieval。因此“final count 先饱和”不等于模型已经能自由生成完整 CoT。</p>
</section>

<section id="s5"><h2>5. 描述性 attention：broad aggregation 与 targeted retrieval</h2>
<p>所有 attention 量都来自 final checkpoint 的 eager-attention forward。每个 count 取 {cfg['attention_examples_per_count']} 个平衡样本；Layer 为 1-based，head 为 0-based。排名分数只用于寻找候选 circuit，不是因果效应。</p>
{''.join(attention_sections)}
<h3>跨版本解读</h3><p>如果某版本 final accuracy=100% 但 raw k-to-k mass、top-1 或 AR trace exact 显著变化，最保守的结论是：模型实现同一输入输出函数时采用了不同程度的显式 targeted routing。尤其 diagonal dominance 只在 needle 子集内部归一化，不能替代 raw mass。</p>
</section>

<section id="s6"><h2>6. Hidden-state 描述性现象：count/progress manifold</h2>
<p>在每个 semantic site 提取 state index 0（token embedding + positional representation）以及经过 Layer 1–4 后的 residual stream。每个 state 是 64 维。Probe 使用独立 balanced train/eval states；position-only baseline 只根据 absolute token position 分类，用于排除“trace 越长，位置越靠后”的捷径。</p>
{''.join(state_sections)}
<div class="analysis"><strong>如何解读。</strong>高 ridge R² 或 nearest-centroid accuracy 表示 count/progress 对线性或类中心解码器可读；它不说明模型输出必须使用该方向。PC1–PC6 coverage 高表示 count centroid 主要落在低维子空间，但也不等于一条可 steering 的标量轴。只有后续 state patching / steering 才能检验因果充分性。</div>
<h3>本节结论与证据边界</h3>{geometry_specific_analysis(bundle)}
{interactive_pca_html(bundle)}
<p class="muted">生成脚本：<code>scripts/build_v11_v14_reports.py</code>。结果目录：<code>{esc(bundle.path)}</code>。</p>
</section>
</main></body></html>"""


def audit_frame(bundle: RunBundle, summaries: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for position in bundle.config["position_encodings"]:
        item = summaries[(bundle.version, position)]
        rows.append(
            {
                "version": bundle.version,
                "position_encoding": position,
                "tf_nonthinking": item["tf_nonthinking"],
                "tf_thinking": item["tf_thinking"],
                "tf_trace_marker": item["tf_trace_marker"],
                "tf_trace_index": item["tf_trace_index"],
                "ar_nonthinking": item["ar_nonthinking"],
                "ar_thinking": item["ar_thinking"],
                "ar_trace_exact": item["ar_trace_exact"],
                "ar_trace_marker_recall": item["ar_trace_marker_recall"],
                "best_broad_nonthinking_head": item["broad_nonthinking"]["head"],
                "best_broad_nonthinking_score": item["broad_nonthinking"].get("broad_attention_score"),
                "best_targeted_head": item["targeted"]["head"],
                "best_targeted_raw_mass": item["targeted"].get("correct_prompt_needle_mass"),
                "best_targeted_top1": item["targeted"].get("correct_top1"),
                "best_targeted_diagonal": item["targeted"].get("diagonal_dominance"),
                "best_readout_head": item["readout"]["head"],
                "best_readout_trace_mass": item["readout"].get("trace_markers_mass"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build self-contained Chinese v11-v14 reports")
    parser.add_argument("--results-root", type=Path, default=Path("colab_results"))
    args = parser.parse_args()
    runs = discover_runs(args.results_root)
    bundles = {version: load_bundle(version, path) for version, path in runs.items()}
    summaries = {
        (version, position): variant_summary(bundle, position)
        for version, bundle in bundles.items()
        for position in bundle.config["position_encodings"]
    }
    comparison = comparison_frame(bundles, summaries)
    for version, bundle in bundles.items():
        report = build_report(bundle, bundles, summaries, comparison)
        output = bundle.path / f"syn_{version}_report.html"
        output.write_text(report, encoding="utf-8")
        audit_frame(bundle, summaries).to_csv(bundle.path / "report_metrics_summary.csv", index=False)
        print(f"Wrote {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
