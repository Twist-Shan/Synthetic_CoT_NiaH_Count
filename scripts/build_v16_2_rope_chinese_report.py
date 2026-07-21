#!/usr/bin/env python3
"""Build the reorganized Chinese v16.2 RoPE report.

The report keeps every result family and visualization from the audited English
report, reorganizes the narrative, and adds v10-style attention-routing and
residual-representation analyses from the saved v16.2 checkpoint states.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def validate_reference_run(run_dir: Path) -> tuple[str, dict]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    expected = {
        "version": "v16_2",
        "seq_len": 256,
        "count_max_threshold": 10,
        "train_steps": 10000,
        "checkpoint_every": 500,
        "max_steps_for_language_pred": 1500,
        "position_encodings": ["rope"],
        "enabled_model_variants": ["rope/nonthinking", "rope/thinking"],
        "seed": 1234,
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"report builder received a non-reference run: {mismatches}")
    run_id_path = run_dir / "source_run_id.txt"
    if not run_id_path.exists():
        raise FileNotFoundError(
            "source_run_id.txt is required so the report cannot silently claim another run identity"
        )
    return run_id_path.read_text(encoding="utf-8").strip(), config


def table(headers: list[str], rows: list[list[str]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(
        f'<th class="num">{item}</th>' if i in numeric else f"<th>{item}</th>"
        for i, item in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="num">{item}</td>' if i in numeric else f"<td>{item}</td>"
            for i, item in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def extract_numbered_images(source: str) -> dict[int, str]:
    figures: dict[int, str] = {}
    for block in re.findall(r"<figure(?:\s[^>]*)?>.*?</figure>", source, flags=re.S):
        match = re.search(r'<span class="figure-tag">Figure\s+(\d+)\.</span>', block)
        image = re.search(r"<img\b[^>]+>", block, flags=re.S)
        if match and image:
            figures[int(match.group(1))] = image.group(0)
    missing = sorted(set(range(1, 16)) - set(figures))
    if missing:
        raise ValueError(f"missing embedded figures from English report: {missing}")
    return figures


def replace_tag_text(block: str, tag: str, value: str) -> str:
    return re.sub(
        rf"(<{tag}(?:\s[^>]*)?>).*?(</{tag}>)",
        lambda match: match.group(1) + value + match.group(2),
        block,
        count=1,
        flags=re.S,
    )


def translated_hypothesis_figure(source: str) -> str:
    match = re.search(
        r'<figure id="report-retrieval-strategy-dynamics">.*?</figure>', source, flags=re.S
    )
    if not match:
        raise ValueError("retrieval-strategy figure not found")
    block = match.group(0)
    translations = {
        "Retrieval strategy dynamics over ten thousand training steps": "一万步训练中的检索策略动态",
        "Three aligned line charts show thinking correct-k retrieval and accuracy, nonthinking broad coverage and accuracy, and the divergence of normalized needle-attention entropy.": "三幅对齐曲线依次显示 thinking 的正确 k 检索与准确率、nonthinking 的广覆盖与准确率，以及两者目标注意力归一化熵的分化。",
        "Thinking: one-at-a-time targeted retrieval": "Thinking：逐个目标的定向检索",
        "Correct-k above chance": "正确 k 的超随机优势",
        "AR count accuracy": "AR 计数准确率",
        "Nonthinking: broad occurrence coverage": "Nonthinking：多目标广覆盖",
        "Top-n needle recall": "Top-n 目标召回率",
        "Distribution across target occurrences": "目标出现位置之间的注意力分布",
        "Thinking entropy (concentrates)": "Thinking 熵（逐步集中）",
        "Nonthinking entropy (stays broad)": "Nonthinking 熵（保持广覆盖）",
        "loss scope switches after step 1,500": "第 1,500 步后切换 loss 范围",
        "Training step": "训练步数",
    }
    for old, new in translations.items():
        block = block.replace(old, new)
    block = re.sub(
        r"<figcaption>.*?</figcaption>",
        '<figcaption><span class="figure-tag">假设图 H1。</span>'
        "Thinking 的 correct-k 超随机优势与 nonthinking 的 top-n 广覆盖在不同时间表上涌现。"
        "下图给出关键分布差异：thinking 的目标注意力熵下降，表示对单个正确出现位置的集中；"
        "nonthinking 在第 5,000 步后熵反而上升，表示成熟电路覆盖更多目标位置。</figcaption>",
        block,
        count=1,
        flags=re.S,
    )
    return block


def translated_attention_maps(source: str) -> str:
    match = re.search(
        r'<div id="report-attention-checkpoint-headmaps">.*?</script>\s*</div>',
        source,
        flags=re.S,
    )
    if not match:
        raise ValueError("attention checkpoint headmaps not found")
    block = match.group(0)
    definitions = {
        "broad": (
            "注意力图 A1：Nonthinking 的 prompt 多目标广聚合",
            "每格是 broad score，即 <code>prompt needle mass × 目标内归一化熵</code>，"
            "先逐样本计算再取平均。高分同时要求把较多注意力放到目标位置、并分散覆盖多个出现位置。"
            "该图到第 5,000 步仍近乎平坦，第 7,000 步转为第 4 层主导；最终最大值集中于 L4H2、L4H0 与 L4H3。",
        ),
        "raw": (
            "注意力图 A2：Thinking 的原始 k-to-k 定向质量",
            "每格是 trace-index query 对正确第 k 个 prompt 目标位置的原始注意力质量。"
            "第 5,000 步时第 3 层已出现明显目标质量；随后 L4H2 成为后期主导头，"
            "由 0.130 上升至最终的 0.339。必须同时报告原始质量，因为仅看相对 top-1 可能在几乎没有质量到达目标集合时仍显得很强。",
        ),
        "relative": (
            "注意力图 A3：Thinking 的 correct-k 相对检索质量",
            "每格为 <code>correct-k top-1 − 1/count</code>。正值表示在真实目标位置内部，"
            "第 k 个位置被选为 top-1 的概率超过随机基线。第 5,000 步第 3 层已普遍高于随机，"
            "L4H2 随后专门化，最终达到 +0.467。",
        ),
        "readout": (
            "注意力图 A4：Thinking 的最终答案对 trace 的读取",
            "每格是最终答案 query 对 trace index 与 marker token 的总注意力质量。"
            "L2H2 在第 1,500 步已是主导 trace reader（0.895），之后稳定在约 0.85–0.87；"
            "L2H0 提供约 0.78–0.80 的第二条持续路径。该路由早于可靠 k-to-k 检索，说明模型先学会“答案应从哪里读”，再学会“如何写出可靠 trace”。",
        ),
    }
    for key, (title, caption) in definitions.items():
        pattern = rf'(<figure data-attention-metric="{key}">)(.*?)(</figure>)'
        found = re.search(pattern, block, flags=re.S)
        if not found:
            raise ValueError(f"attention map {key} not found")
        inner = replace_tag_text(found.group(2), "h4", title)
        inner = replace_tag_text(inner, "figcaption", caption)
        block = block[: found.start()] + found.group(1) + inner + found.group(3) + block[found.end() :]
    block = block.replace('`step ${step.toLocaleString()}`', '`步骤 ${step.toLocaleString()}`')
    block = block.replace(
        '`Shared scale: ${metric.min ?? 0} to ${metric.max}; outlined cell = fixed final-selected head.`',
        '`统一色标：${metric.min ?? 0} 至 ${metric.max}；描边单元格为按最终 checkpoint 固定选择的头。`',
    )
    return block


def centroids(npz_path: Path, site: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    states = np.load(npz_path)
    x = states[f"{site}__{layer}__x"]
    y = states[f"{site}__{layer}__y"]
    labels = np.unique(y)
    return labels, np.stack([x[y == label].mean(axis=0) for label in labels])


def pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    variances = singular**2
    ratios = variances / variances.sum()
    return centered @ vh[:2].T, ratios


def adjacent_cosines(points: np.ndarray) -> np.ndarray:
    delta = np.diff(points, axis=0)
    return np.sum(delta[:-1] * delta[1:], axis=1) / (
        np.linalg.norm(delta[:-1], axis=1) * np.linalg.norm(delta[1:], axis=1)
    )


def build_representation_figure(run_dir: Path) -> tuple[Path, pd.DataFrame]:
    part_root = run_dir / "analysis" / "checkpoint_dynamics" / "parts"
    nt_npz = part_root / "rope_nonthinking_step_010000" / "heldout_states.npz"
    th_npz = part_root / "rope_thinking_step_010000" / "heldout_states.npz"
    paths = [
        ("Nonthinking <Ans>, L4", nt_npz, "final_answer", 4, "#8a4f96"),
        ("Thinking <Ans>, L2", th_npz, "final_answer", 2, "#2065a8"),
        ("Thinking index k, L3", th_npz, "trace_index", 3, "#087f79"),
        ("Thinking marker M_k, L3", th_npz, "trace_marker", 3, "#b36f16"),
    ]
    values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = {}
    rows = []
    for name, npz_path, site, layer, color in paths:
        labels, means = centroids(npz_path, site, layer)
        coords, ratios = pca(means)
        cosines = adjacent_cosines(means)
        effective_dimension = 1.0 / float(np.square(ratios).sum())
        values[name] = (labels, means, coords, ratios, color)
        rows.append(
            {
                "representation": name,
                "site": site,
                "layer": layer,
                "centroids": len(labels),
                "pc1_variance": ratios[0],
                "pc1_to_pc2_variance": ratios[:2].sum(),
                "pc1_to_pc6_variance": ratios[:6].sum(),
                "effective_dimension": effective_dimension,
                "mean_adjacent_displacement_cosine": cosines.mean(),
            }
        )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.6), layout="constrained")
    for ax, name, title in [
        (axes[0, 0], "Nonthinking <Ans>, L4", "A  Nonthinking 最终答案状态：L4 count centroids"),
        (axes[0, 1], "Thinking <Ans>, L2", "B  Thinking 最终答案状态：L2 count centroids"),
    ]:
        labels, _, coords, ratios, color = values[name]
        ax.plot(coords[:, 0], coords[:, 1], "-", color=color, alpha=0.65, lw=2)
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", s=72, zorder=3)
        for label, (x, y) in zip(labels, coords):
            ax.annotate(str(int(label)), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=9)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(f"PC1（{100 * ratios[0]:.1f}%）")
        ax.set_ylabel(f"PC2（{100 * ratios[1]:.1f}%）")
        ax.grid(alpha=0.22)

    ax = axes[1, 0]
    index_labels, index_means, _, _, _ = values["Thinking index k, L3"]
    marker_labels, marker_means, _, _, _ = values["Thinking marker M_k, L3"]
    joint = np.vstack([index_means, marker_means])
    joint_coords, joint_ratios = pca(joint)
    idx_xy, marker_xy = joint_coords[:10], joint_coords[10:]
    ax.plot(idx_xy[:, 0], idx_xy[:, 1], "-o", color="#087f79", lw=2, ms=5, label="index k")
    ax.plot(marker_xy[:, 0], marker_xy[:, 1], "-o", color="#b36f16", lw=2, ms=5, label="marker M_k")
    for i, label in enumerate(index_labels):
        ax.annotate(str(int(label)), idx_xy[i], xytext=(4, 3), textcoords="offset points", fontsize=8, color="#087f79")
        ax.annotate(str(int(marker_labels[i])), marker_xy[i], xytext=(4, 3), textcoords="offset points", fontsize=8, color="#8a4d0d")
        ax.annotate("", xy=marker_xy[i], xytext=idx_xy[i], arrowprops={"arrowstyle": "->", "color": "#9aa7b5", "lw": 0.8, "alpha": 0.55})
    ax.set_title("C  Thinking trace：index 与 marker 的联合 progress geometry", loc="left", fontweight="bold")
    ax.set_xlabel(f"joint PC1（{100 * joint_ratios[0]:.1f}%）")
    ax.set_ylabel(f"joint PC2（{100 * joint_ratios[1]:.1f}%）")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)

    ax = axes[1, 1]
    x = np.arange(2, 10)
    for name in values:
        _, means, _, _, color = values[name]
        ax.plot(x, adjacent_cosines(means), "-o", lw=2, ms=4, color=color, label=name)
    ax.axhline(1.0, color="#5d6a7e", ls="--", lw=1, label="单一全局 +1 方向")
    ax.axhline(0.0, color="#aab4bf", lw=0.8)
    ax.set_ylim(-0.72, 1.08)
    ax.set_title("D  相邻两次 progress 位移的方向余弦", loc="left", fontweight="bold")
    ax.set_xlabel("中间 progress k")
    ax.set_ylabel("cos(delta_k, delta_(k+1))")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.22)
    fig.suptitle("v16.2 RoPE：v10 风格的 mean-first residual representation 分析", fontsize=16, fontweight="bold")
    out = run_dir / "figures" / "v16_2_representation_geometry_v10_style.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "tables" / "v16_2_representation_geometry_summary.csv", index=False)
    return out, summary


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_corrected_metrics(run_dir: Path, train_steps: int) -> dict[str, object]:
    tables = run_dir / "tables"
    permutation = pd.read_csv(tables / "prefix_permutation_consistency.csv")
    required_permutation = {
        "tf_permutation_accuracy",
        "ar_permutation_accuracy",
        "ar_prediction_agreement",
        "ar_all_permutations_correct",
    }
    if not required_permutation.issubset(permutation.columns):
        raise ValueError(
            "prefix permutation table predates the TF/AR semantic fix; refresh final evaluation"
        )
    permutation_summary = permutation.groupby("mode").agg(
        tf_accuracy=("tf_permutation_accuracy", "mean"),
        ar_accuracy=("ar_permutation_accuracy", "mean"),
        ar_agreement=("ar_prediction_agreement", "mean"),
        ar_all_correct=("ar_all_permutations_correct", "mean"),
    )

    probe = pd.read_csv(tables / "checkpoint_state_probe_summary.csv")
    trace_probe = probe[
        (probe["step"] == train_steps)
        & (probe["mode"] == "thinking")
        & (probe["context"] == "teacher_forced")
        & (probe["site"] == "trace_marker")
        & (probe["layer"] == 3)
    ].iloc[0]
    cross_site = pd.read_csv(tables / "checkpoint_state_cross_site.csv")
    cross_l2 = cross_site[
        (cross_site["step"] == train_steps)
        & (cross_site["layer"] == 2)
    ].set_index("direction")
    if set(cross_site["direction_coordinate_system"].dropna()) != {"raw_hidden_space"}:
        raise ValueError("cross-site direction cosine is not in raw hidden coordinates")

    behavior = pd.read_csv(tables / "checkpoint_fixed_head_behavior_link.csv")
    nt = behavior[
        (behavior["step"] == train_steps)
        & (behavior["mode"] == "nonthinking")
        & (behavior["head_selection_role"] == "needle_retrieval")
    ]
    if nt.empty or set(nt["diagnostic_split"]) != {"heldout_reporting"}:
        raise ValueError("fixed-head behavior link is missing heldout-reporting prompts")
    behavior_summary = nt.groupby("ar_accuracy").agg(
        enrichment=("needle_attention_enrichment", "mean"),
        recall=("top_n_needle_recall", "mean"),
        examples=("prompt_sha256", "nunique"),
    )

    audit = pd.read_csv(tables / "checkpoint_state_sampling_audit.csv")
    trace_audit = audit[audit["sampling_suite"] == "fixed_total_count_trace_progress"]
    if set(trace_audit["total_count"].astype(int)) != {10}:
        raise ValueError("trace progress states are still confounded with total count")
    balance = trace_audit.groupby(["mode", "data_split", "site", "progress_label"]).size()
    if balance.groupby(level=[0, 1, 2]).nunique().max() != 1:
        raise ValueError("trace progress state sampling is not balanced by progress label")
    return {
        "permutation": permutation_summary,
        "trace_probe": trace_probe,
        "cross_l2": cross_l2,
        "behavior": behavior_summary,
        "fixed_head": f"L{int(nt.layer.iloc[0])}H{int(nt['head'].iloc[0])}",
        "trace_audit_examples": int(trace_audit.prompt_sha256.nunique()),
    }


def build_fixed_head_behavior_figure(run_dir: Path) -> Path:
    frame = pd.read_csv(run_dir / "tables" / "checkpoint_fixed_head_behavior_link.csv")
    frame = frame[
        (frame["mode"] == "nonthinking")
        & (frame["head_selection_role"] == "needle_retrieval")
    ]
    summary = frame.groupby(["step", "ar_accuracy"], as_index=False).agg(
        enrichment=("needle_attention_enrichment", "mean"),
        recall=("top_n_needle_recall", "mean"),
        examples=("prompt_sha256", "nunique"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), layout="constrained")
    labels = {0.0: "AR incorrect", 1.0: "AR correct"}
    colors = {0.0: "#b45309", 1.0: "#2563eb"}
    for accuracy, group in summary.groupby("ar_accuracy"):
        group = group.sort_values("step")
        axes[0].plot(
            group["step"], group["enrichment"], "-o", ms=3.5,
            color=colors[float(accuracy)], label=labels[float(accuracy)],
        )
        axes[1].plot(
            group["step"], group["recall"], "-o", ms=3.5,
            color=colors[float(accuracy)], label=labels[float(accuracy)],
        )
    axes[0].set(xlabel="Training step", ylabel="Needle attention enrichment (×)")
    axes[1].set(xlabel="Training step", ylabel="Top-n needle recall", ylim=(-0.03, 1.03))
    for axis in axes:
        axis.axvline(1500, color="#475569", linestyle=":", linewidth=1.2)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.suptitle(
        "Nonthinking fixed-head retrieval on disjoint heldout-reporting prompts",
        fontsize=14,
        fontweight="bold",
    )
    output = run_dir / "figures" / "checkpoint_fixed_head_behavior_link.png"
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def figure(image_tag: str, number: int, caption: str, alt: str) -> str:
    image_tag = re.sub(r'alt="[^"]*"', f'alt="{html.escape(alt)}"', image_tag, count=1)
    return (
        "<figure>"
        + image_tag
        + f'<figcaption><span class="figure-tag">图 {number}。</span>{caption}</figcaption></figure>'
    )


def build(run_dir: Path) -> tuple[Path, Path]:
    RUN_ID, config = validate_reference_run(run_dir)
    metrics = load_corrected_metrics(run_dir, int(config["train_steps"]))
    complete = run_dir / "v16_2_rope_complete_report.html"
    english = run_dir / "v16_2_rope_complete_report_en.html"
    if not english.exists():
        if not complete.exists():
            raise FileNotFoundError(complete)
        shutil.copy2(complete, english)
    source = english.read_text(encoding="utf-8")
    images = extract_numbered_images(source)
    h1 = translated_hypothesis_figure(source)
    attention_maps = translated_attention_maps(source)
    rep_path, rep_summary = build_representation_figure(run_dir)
    corrected_figure_files = {
        1: "checkpoint_mechanism_overview.png",
        3: "checkpoint_attention_retrieval_emergence.png",
        4: build_fixed_head_behavior_figure(run_dir).name,
        5: "checkpoint_answer_routing.png",
        6: "checkpoint_ordered_trace_retrieval.png",
        7: "checkpoint_final_count_probe_heatmap.png",
        8: "checkpoint_trace_progress_probe_heatmap.png",
        9: "checkpoint_cross_site_counter_transfer.png",
        10: "checkpoint_state_geometry_emergence.png",
        11: "checkpoint_representation_stability.png",
        12: "checkpoint_counterfactual_trace_readout.png",
        15: "training_structure_and_noise_effects.png",
    }
    for number, filename in corrected_figure_files.items():
        path = run_dir / "figures" / filename
        if not path.exists():
            raise FileNotFoundError(path)
        images[number] = f'<img src="{data_uri(path)}" alt="corrected figure {number}">'
    trace_probe = metrics["trace_probe"]
    cross_l2 = metrics["cross_l2"]
    behavior_summary = metrics["behavior"]
    nt_correct = behavior_summary.loc[1.0]
    nt_incorrect = behavior_summary.loc[0.0]
    permutation = metrics["permutation"]
    perm_nt = permutation.loc["nonthinking"]
    perm_th = permutation.loc["thinking"]
    rep_lookup = rep_summary.set_index("representation")
    nt_rep = rep_lookup.loc["Nonthinking <Ans>, L4"]
    th_rep = rep_lookup.loc["Thinking <Ans>, L2"]
    index_rep = rep_lookup.loc["Thinking index k, L3"]
    marker_rep = rep_lookup.loc["Thinking marker M_k, L3"]

    head = re.search(r"<head>.*?</head>", source, flags=re.S).group(0)
    head = re.sub(r"<title>.*?</title>", "<title>NIAH 合成计数：v16.2 RoPE 机制与表征报告</title>", head)
    extra_css = """
  <style>
    body { font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Segoe UI", sans-serif; }
    .section-kicker { margin-bottom: -42px; color: var(--teal); font-size: .78rem; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; }
    .equation { margin: 13px 0; padding: 13px 16px; border: 1px solid #cad8e5; border-radius: 8px; background: #f7fafc; font: .94rem/1.65 Cambria, "Times New Roman", serif; overflow-x: auto; }
    .evidence-ladder { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin: 18px 0 28px; }
    .evidence-ladder > div { padding: 15px 17px; border: 1px solid var(--line); border-radius: 10px; background: #fbfcfe; }
    .evidence-ladder b { display: block; color: var(--navy); margin-bottom: 5px; }
    .preserved { border-left: 5px solid var(--green); }
    .small { font-size: .88rem; }
    @media (max-width: 760px) { .evidence-ladder { grid-template-columns: 1fr; } }
  </style>
"""
    head = head.replace("</head>", extra_css + "</head>")

    captions = {
        1: "21 个 checkpoint 上的行为与机制总览。横轴为训练步数；行为曲线纵轴为准确率，注意力曲线纵轴为对应质量或召回率。Thinking 更早提升并依赖 trace 路由；nonthinking 更晚提升并伴随直接 prompt 检索。竖虚线标记第 1,500 步 loss-scope 切换。",
        2: "100 个三字符 needle set 在允许的总语料频率区间中按设计近似均匀：20 个频率 bin，每 bin 5 个集合。横轴为三字符在训练语料中的频率和，纵轴为集合数量。这保证 pool 覆盖，却不保证经过窗口过滤后 count 分布均匀。",
        3: "跨 checkpoint 的检索强度。横轴为训练步数；纵轴为选定头的目标注意力 enrichment。Nonthinking 在后期形成强 L4 答案到 prompt 电路；thinking 在最终答案位置对 prompt 的直接检索始终弱得多。",
        4: f"Nonthinking 固定头（{metrics['fixed_head']}）的目标覆盖与行为关联。横轴为 checkpoint；纵轴分别为 enrichment 与 top-n recall。所有曲线只用与选头样本不重叠的 heldout-reporting prompts。最终预测正确样本的平均 enrichment（{nt_correct.enrichment:.2f}×）和召回（{100*nt_correct.recall:.1f}%）与错误样本（{nt_incorrect.enrichment:.2f}×、{100*nt_incorrect.recall:.1f}%）分开报告。",
        5: "Thinking 的答案位置路由。横轴为 checkpoint；纵轴为注意力质量。固定 L2H2 很早便将大部分质量放到 trace token，而答案位置的直接 prompt mass 仍较小。",
        6: "训练中的有序 trace-to-prompt 检索。横轴为第 k 次 trace 查询；纵轴为 correct-k top-1 或其随机基线。边界出现位置在全头均值中最容易；固定专门化头显著超过随机。",
        7: "各 checkpoint、各层的最终 count probe 准确率。横轴为训练步数，纵轴为层号/准确率编码。Thinking 很早在中间层形成 count representation；nonthinking 的可用 count state 到后期才在最终层出现。",
        8: f"Trace-marker progress 的去混淆可解码性。横轴为训练步数，纵轴为 probe 指标/层。每个 progress k 都来自 total count 固定为 10 的相同样本套件；最终 L3 marker state 的 nearest-centroid accuracy 为 {100*trace_probe.nearest_centroid_accuracy:.1f}%，ridge R² 为 {trace_probe.ridge_r2:.3f}。",
        9: "跨位置线性迁移。横轴为 checkpoint/层，纵轴为跨位置预测性能。Trace-progress 样本固定 total count=10；方向余弦先把标准化 ridge 系数除以各维 scale，转换回 raw hidden coordinates 后计算。迁移仍是描述性证据。",
        10: "Nonthinking count-centroid geometry 的涌现。横轴为 checkpoint，纵轴为 PC1-count R²、有效维度等几何量。随着模型成熟，几何更结构化且不再局限于单一轴。",
        11: "相对最终 checkpoint 的 linear CKA。横轴为训练步数，纵轴为 CKA。Thinking 的答案状态很早接近最终几何；nonthinking 在后期才发生明显重组。",
        12: "删除 gold trace 最后一对 <n> marker 后的因果效应。横轴为 checkpoint，纵轴为输出概率/隐藏状态解码 count 的变化。到第 500 步，decoded-count 已下降 −0.975；最终接近精确减一。",
        13: "训练目标切换后的 loss 动态。横轴为训练步数，纵轴为交叉熵或 AR accuracy。第 1,500 步后大部分语言 token 不再优化，因此 full-sequence loss 变差是预期现象；此后应看 task component loss 与 AR 准确率。",
        14: "运行时间分解。纵轴为累计分钟，颜色区分 optimizer、周期性 TF evaluation、checkpoint dynamics 等阶段；全部 523 个 timing event 成功完成。训练加 dynamics 约 90 分钟，不含最终 bundle 复制。",
        15: "训练结构与观察性 noise 效应。A：横轴是真实 union count n，纵轴为每模型训练概率，并与均衡评估的 0.10 比较。B：横轴为 count band，纵轴为最终 AR accuracy。C：横轴为控制 count 后的 needle-set 语料频率四分位，纵轴为最终 AR accuracy。D：横轴为控制 count 后的最长非目标 run 四分位，纵轴相同。C–D 每个 representation 各汇总 100 个 prompt，相关关系不是随机干预。",
    }
    alts = {
        1: "行为与机制学习动态总览", 2: "needle set 频率分布", 3: "检索强度动态", 4: "nonthinking 目标覆盖",
        5: "thinking 答案到 trace 的注意力", 6: "thinking k-to-k 检索", 7: "最终 count probe", 8: "trace marker progress probe",
        9: "跨位置线性迁移", 10: "count centroid geometry", 11: "状态 CKA 稳定性", 12: "trace 删除反事实",
        13: "loss 动态", 14: "运行时间分解", 15: "训练结构与 noise 关联",
    }
    figs = {n: figure(images[n], n, captions[n], alts[n]) for n in range(1, 16)}
    rep_img = f'<img src="{data_uri(rep_path)}" alt="v10 风格 residual representation 几何分析">'
    figs[16] = figure(
        rep_img,
        16,
        "Mean-first residual representation。每个点先对同一语义标签的 15 个 held-out state 取 256 维均值，再对这些 centroids 做 PCA；trace 的所有 progress 标签均来自 total count 固定为 10 的样本。A、B 分别单独拟合 PCA，C 对 index 与 marker 的 20 个 centroids 联合拟合。坐标轴为 centroid PCA 的 PC1/PC2，括号为解释方差。D 中 Δₖ=μₖ₊₁−μₖ，纵轴是 cos(Δₖ,Δₖ₊₁)：若模型沿单一全局“+1”方向更新，应接近 1。PCA 只显示均值几何，不显示单样本 cloud，也不能单独证明模型执行加法。",
        "v10 风格 residual representation 几何分析",
    )

    ar = pd.read_csv(run_dir / "tables" / "autoregressive_by_count.csv")
    final_ar = ar[ar["step"].eq(10000)]
    tf = pd.read_csv(run_dir / "tables" / "eval_by_count.csv")
    final_tf = tf[tf["step"].eq(10000)]
    by_count_rows = []
    for count in range(1, 11):
        nt = final_ar[(final_ar["mode"] == "nonthinking") & (final_ar["count"] == count)].iloc[0]
        th = final_ar[(final_ar["mode"] == "thinking") & (final_ar["count"] == count)].iloc[0]
        nt_tf = final_tf[(final_tf["mode"] == "nonthinking") & (final_tf["count"] == count)].iloc[0]
        th_tf = final_tf[(final_tf["mode"] == "thinking") & (final_tf["count"] == count)].iloc[0]
        by_count_rows.append(
            [str(count), f'{100*nt_tf.tf_final_accuracy:.0f}%', f'{100*nt.ar_final_accuracy:.0f}%', f'{nt.ar_abs_error:.2f}', f'{100*th_tf.tf_final_accuracy:.0f}%', f'{100*th.ar_final_accuracy:.0f}%', f'{th.ar_abs_error:.2f}', f'{100*th.trace_exact:.0f}%', f'{100*th.trace_marker_recall:.1f}%']
        )
    by_count_table = table(
        ["真实 count", "NT TF", "NT AR", "NT MAE", "Thinking TF", "Thinking AR", "Thinking MAE", "Trace exact", "Ordered-marker acc."],
        by_count_rows,
        {0, 1, 2, 3, 4, 5, 6, 7, 8},
    )

    geom_rows = []
    for row in rep_summary.itertuples(index=False):
        geom_rows.append(
            [
                html.escape(row.representation),
                f"L{row.layer}",
                f"{100*row.pc1_variance:.1f}%",
                f"{100*row.pc1_to_pc2_variance:.1f}%",
                f"{100*row.pc1_to_pc6_variance:.1f}%",
                f"{row.effective_dimension:.2f}",
                f"{row.mean_adjacent_displacement_cosine:.3f}",
            ]
        )
    geom_table = table(
        ["Representation / site", "层", "PC1 方差", "PC1–2 方差", "PC1–6 方差", "有效维度", "相邻位移余弦均值"],
        geom_rows,
        {2, 3, 4, 5, 6},
    )

    body = f"""<!doctype html>
<html lang="zh-CN">
{head}
<body>
<main>
  <header>
    <p class="eyebrow">NIAH SYNTHETIC COUNTING · v16.2 · RoPE · 完整中文机制报告</p>
    <h1>两种表示，两条计数路径</h1>
    <p class="subtitle">在同一 RoPE 架构、初始化、数据流和优化预算下，比较 nonthinking 的答案时多目标聚合与 thinking 的显式顺序 trace。报告以 learning dynamics 为主线，并按 v10 第五、六节重新加入 attention routing 与 residual representation 的分层分析。</p>
    <div class="run-id">{RUN_ID}</div>
  </header>
  <div class="content">
    <div class="callout success preserved"><strong>内容保留与重排。</strong> 本中文版保留原完整报告的全部 15 幅编号图、H1 假设图、A1–A4 四组 attention headmaps、全部实验/运行/复现结论；新增图 16 与相应几何表。原英文扩展版保存在 <a href="v16_2_rope_complete_report_en.html">v16_2_rope_complete_report_en.html</a>，原始参考 HTML 保存在 <a href="v16_2_rope_reference_report.html">v16_2_rope_reference_report.html</a>。</div>

    <nav class="toc"><strong>逻辑顺序</strong><ol>
      <li><a href="#questions">研究问题与结论</a></li>
      <li><a href="#setup">实验设定与序列格式</a></li>
      <li><a href="#definitions">指标与计算定义</a></li>
      <li><a href="#learning">行为与 learning dynamics</a></li>
      <li><a href="#attention-representation">Attention representation</a></li>
      <li><a href="#residual-representation">Residual representation</a></li>
      <li><a href="#causal">从描述性表征到因果证据</a></li>
      <li><a href="#data-noise">训练数据结构与 noise</a></li>
      <li><a href="#runtime-repro">运行、产物与复现审计</a></li>
      <li><a href="#limits">证据边界与后续实验</a></li>
    </ol></nav>

    <section id="questions">
      <p class="section-kicker">Question → representation → mechanism</p>
      <h2>1. 研究问题与核心结论</h2>
      <p class="lede">核心问题不是“哪一个模型最终分数更高”，而是：在相同 RoPE transformer 中，是否因为输出格式不同而涌现出两种不同的内部表示和信息流？这里把表示分成两个层级：<strong>attention representation</strong> 描述 query 把权重分配到哪里；<strong>residual representation</strong> 描述读取后写入 residual stream 的 256 维内容。</p>
      <div class="stat-grid">
        <div class="stat blue"><span class="value">84%</span><span class="label">Nonthinking 最终 AR accuracy；MAE 0.21</span></div>
        <div class="stat green"><span class="value">98%</span><span class="label">Thinking 最终 AR accuracy；MAE 0.02</span></div>
        <div class="stat gold"><span class="value">64.9%</span><span class="label">Thinking L4H2 correct-k top-1；随机基线 18.2%</span></div>
        <div class="stat"><span class="value">−1.016</span><span class="label">删除最后一个 trace pair 后隐藏状态解码 count 的变化</span></div>
      </div>
      <ol class="finding-list">
        <li><strong>Thinking 先学路由、后学检索。</strong> L2H2 在第 500–1,500 步已把约 84–90% 的答案注意力放到 trace，但 reliable correct-k prompt retrieval 约到第 4,500–5,000 步才明显涌现。</li>
        <li><strong>Nonthinking 后期才形成直接聚合。</strong> L4H2 的 top-n target coverage 与 AR accuracy 均在约第 7,000 步越过 persistent half-rise；它在答案位置广泛覆盖目标出现位置，而不是定义一个 k-to-k interface。</li>
        <li><strong>两种 residual geometry 都含 count，但形状不同。</strong> Thinking 在 L2 的最终答案 state 几乎完美线性可解码（R²=0.998），却不是单一 PC1 或单一“+1”轴；nonthinking 最终 L4 的 count probe R²=0.957、有效维度 3.01。</li>
        <li><strong>“可解码”与“注意到”都不是充分因果证据。</strong> Trace deletion 使输出从 n 几乎精确翻转到 n−1，并令隐藏状态解码值下降约 1，才把 thinking 的外部计数器解释提升到更强的因果层级。</li>
      </ol>
      {figs[1]}
    </section>

    <section id="setup">
      <p class="section-kicker">Matched comparison</p>
      <h2>2. 实验设定、数据构造与序列格式</h2>
      <h3>2.1 数据与任务</h3>
      <p>噪声源为 Tiny Shakespeare 字符流，采用有间隔保护的 80%/10%/10% train/validation/test 切分；本地记录的三段长度分别为 891,907、111,488、111,489 字符。每个样本截取长度 256 的 prompt，从 100 个预先构造的三字符集合中采样一个 query；仅接受三字符在窗口中的 union count 为 1–10 的窗口。三个字符互异，query 顺序每次随机打乱。</p>
      <p>Needle pool 由 20 个训练语料频率 bin 构成，每 bin 5 个集合；三字符频率和不超过 0.04。<code>task_occurrence_ratio=1.0</code>，因此两个模型的 1.28M 个训练样本全部是 counting task，不混入 raw-language 样本。</p>
      {figs[2]}
      <h3>2.2 两种输出 representation</h3>
      <div class="card-grid">
        <div class="card"><h4>Nonthinking（query-first）</h4><div class="format-box">&lt;BOS&gt; · query[&lt;CountChar&gt;, q₁, q₂, q₃, &lt;Sep&gt;] · data/prompt[256] · &lt;Ans&gt; · count · &lt;EOS&gt;</div><p>任务 query（待计数的三个字符）位于数据之前；<code>&lt;Ans&gt;</code> 是最终答案的 readout query。Nonthinking 没有语义明确的第 k 次检索位置，因此严格 k-to-k 指标不适用。</p></div>
        <div class="card"><h4>Thinking（query-first）</h4><div class="format-box">&lt;BOS&gt; · query[&lt;CountChar&gt;, q₁, q₂, q₃, &lt;Sep&gt;] · data/prompt[256] · &lt;Think&gt; · 1 · marker₁ · … · n · markerₙ · &lt;/Think&gt; · &lt;Ans&gt; · count · &lt;EOS&gt;</div><p>任务 query 同样位于数据之前。Trace 按数据中目标出现的从左到右顺序，依次输出 occurrence index 与实际字符 marker，提供可对齐的 k-to-k trace query。</p></div>
        <div class="card"><h4>架构</h4><p>4 层、4 头、d<sub>model</sub>=256、MLP=1024、context=384、无 dropout；两者均为 RoPE（base 10,000），各 3,180,800 参数。</p></div>
        <div class="card"><h4>严格匹配</h4><p>同一 seed=1234、相同初始化、同一训练样本及顺序、相同 AdamW 更新；唯一系统差别是输出序列及其监督。</p></div>
      </div>
      <h3>2.3 优化与两阶段 loss</h3>
      <p>AdamW，learning rate 3×10<sup>−4</sup>，warmup 500 步后 cosine decay，weight decay 0.01，batch size 128，gradient clip 1.0，float32 CUDA，共 10,000 步；每 500 步保存 checkpoint 与 TF evaluation。最终均衡 AR 评估每个 count 使用 10 个样本，共 100 个；TF 主评估每个 count 使用 50 个样本。</p>
      <div class="phase-row"><div class="phase language">1–1,500：all-sequence loss</div><div class="phase task">1,501–10,000：task-output-only loss</div></div>
      <p>前 1,500 步优化整个序列；之后 nonthinking 从 <code>&lt;Ans&gt;</code> 开始、thinking 从 <code>&lt;Think&gt;</code> 开始，仅优化任务输出。这个边界必须进入 learning-dynamics 解释：后期 prompt/prefix CE 上升并不等于计数能力退化。</p>
    </section>

    <section id="definitions">
      <p class="section-kicker">Operational definitions</p>
      <h2>3. 评估指标与新概念如何计算</h2>
      <div class="metric-grid">
        <div class="metric"><strong>TF final accuracy</strong>给定 gold prefix 后，最终 count token 是否正确。Thinking 早期可能受 gold trace 帮助，因此会高估 free-running 能力。</div>
        <div class="metric"><strong>AR final accuracy / answer rate / MAE</strong>从模型自己生成的 prefix/trace 到最终 count。若没有生成可解析答案，accuracy 记 0、answer rate 记 0；answered-only MAE 不把缺失答案偷算成 0，另报以 count 上限为缺失惩罚的 MAE。</div>
        <div class="metric"><strong>Trace exact / ordered-marker accuracy</strong>完整 trace 是否逐 token 正确；以及第 k 个生成 marker 是否等于第 k 个 gold marker 的平均值。后者是有序位置准确率，不是集合召回率。</div>
        <div class="metric"><strong>Nearest-centroid</strong>在训练 state 上按标签求 centroid，held-out state 归到欧氏距离最近者；用于 count 或 progress。</div>
        <div class="metric"><strong>Ridge probe</strong>在 train state 上拟合线性 ridge 回归预测标签，报告 held-out MAE 与 R²。</div>
        <div class="metric"><strong>Linear CKA</strong>比较同一批样本在两个 checkpoint 的 centered Gram geometry；高值表示群体几何相似，而非参数相同。</div>
      </div>
      <h3>3.1 Attention：广覆盖与定向检索必须分开</h3>
      <p>对某个 query q，令 N={{p₁,…,pₙ}} 为 prompt 中 n 个真实目标位置，a(q,p) 为该头的 attention weight：</p>
      <div class="equation">目标总质量 M<sub>N</sub>=Σ<sub>p∈N</sub>a(q,p)；目标内分布 π<sub>p</sub>=a(q,p)/M<sub>N</sub>；归一化熵 H̄<sub>N</sub>=−Σπ<sub>p</sub>logπ<sub>p</sub>/log n；有效覆盖数 N<sub>eff</sub>=exp(−Σπ<sub>p</sub>logπ<sub>p</sub>)；Broad score B=M<sub>N</sub>·H̄<sub>N</sub>。</div>
      <p>Broad score 低有两种完全不同的原因：模型可能忽略所有目标（M<sub>N</sub> 小），也可能强烈聚焦少数目标（熵小）。所以必须同时报告 mass、entropy、effective count 与 top-n recall。n=1 时目标内没有分散问题，归一化熵按实现约定处理，不用它单独解释广覆盖。</p>
      <p>Thinking 的第 k 个 trace-index token 提供 q<sub>k</sub>，其正确目标是第 k 个 prompt occurrence p<sub>k</sub>。原始 correct-k mass 是 a(q<sub>k</sub>,p<sub>k</sub>)；correct-k top-1 是 p<sub>k</sub> 在真实目标集合中获得最大 attention 的比例；随机基线为 1/n；图中 margin 为 <code>top-1−1/n</code>。Diagonal dominance 衡量正确对角权重相对其他目标位置的优势。Nonthinking 没有 q<sub>k</sub>，因此其 strict correct-k 是<strong>未定义</strong>，不能把 answer query 的广覆盖包装成 k-to-k。</p>
      <p><strong>State 采样去混淆。</strong>最终答案 count state 按 total count=1…10 分层，各标签 train/heldout 分别最多 40/15 个样本；trace progress state 则只取 total count=10 的样本，并从每个样本读取 k=1…10。这样 progress=k 不再与 total count=k 完全共线。逐条审计见 <code>checkpoint_state_sampling_audit.csv</code>。</p>
      <h3>3.2 Residual state：mean-first centroid geometry</h3>
      <p>令 h<sub>i</sub><sup>(ℓ,s)</sup>∈R<sup>256</sup> 为第 i 个 held-out 样本在层 ℓ、语义位置 s 的 residual state。对标签 c 先求 μ<sub>c</sub>=mean(h<sub>i</sub>|y<sub>i</sub>=c)，再对 10 个 centroids 去总均值并做 PCA；这叫 <strong>mean-first PCA</strong>，显示语义均值几何而不是 150 个单样本 cloud。</p>
      <div class="equation">PC 方差比 r<sub>j</sub>=λ<sub>j</sub>/Σλ；有效维度 d<sub>eff</sub>=1/Σr<sub>j</sub><sup>2</sup>；相邻位移 Δ<sub>k</sub>=μ<sub>k+1</sub>−μ<sub>k</sub>；方向一致性 c<sub>k</sub>=cos(Δ<sub>k</sub>,Δ<sub>k+1</sub>)。</div>
      <p>若所有 progress 都沿一个固定“+1”向量更新，c<sub>k</sub> 应接近 1、PC1 应占主要方差；偏低或为负表示路径弯曲/折返，但不能仅凭几何断言模型执行某个算法。</p>
      <div class="evidence-ladder"><div><b>描述性</b>Attention map、PCA、probe、CKA：说明信息在哪里与何时可读。</div><div><b>关联性</b>机制指标与 AR accuracy 同步/相关：支持共同涌现，但可能有第三变量。</div><div><b>干预性</b>删除/patch/steer 后输出按预测变化：才直接检验该表征是否被模型使用。</div></div>
    </section>

    <section id="learning">
      <p class="section-kicker">Behavior first, mechanism aligned in time</p>
      <h2>4. 行为表现与 learning dynamics</h2>
      <h3>4.1 最终行为、错误与 permutation robustness</h3>
      {table(["模式", "TF final acc.", "AR final acc.", "AR MAE", "Trace exact", "Ordered-marker acc."], [["Nonthinking", "79.4%", "84.0%", "0.210", "—", "—"], ["Thinking", "100.0%", "98.0%", "0.020", "81.0%", "93.8%"]], {1,2,3,4,5})}
      <p>100 个最终 AR 样本中，thinking 仅有 2 个 count error（2→3 与 7→6），nonthinking 有 16 个。置换鲁棒性现在严格区分两种上下文：<strong>TF</strong> 在 thinking 中允许答案 logit 看到 gold trace，只是条件诊断；<strong>AR</strong> 从 prompt 自由生成完整 trace/答案，才是主要结论。Thinking 六排列的 AR 平均准确率为 {100*perm_th.ar_accuracy:.1f}%，预测一致率为 {100*perm_th.ar_agreement:.1f}%，六排列全对样本为 {100*perm_th.ar_all_correct:.1f}%；nonthinking 分别为 {100*perm_nt.ar_accuracy:.1f}%、{100*perm_nt.ar_agreement:.1f}% 与 {100*perm_nt.ar_all_correct:.1f}%。对应 TF permutation accuracy 为 thinking {100*perm_th.tf_accuracy:.1f}%、nonthinking {100*perm_nt.tf_accuracy:.1f}%，不得与 AR 混用。</p>
      {by_count_table}
      <h3>4.2 准确率门槛与 TF–AR 暴露偏差</h3>
      {table(["AR 门槛", "Thinking 首次持续达到", "Nonthinking 首次持续达到"], [["50%", "step 4,000", "step 7,000"], ["75%", "step 5,000", "step 8,000"], ["90%", "step 7,000", "未达到"]], {1,2})}
      <p>第 500 步 thinking 的 TF final accuracy 已为 100%，但 AR 只有 12%；这是 gold trace 给最终答案提供了正确上下文。到后期 trace 自生成稳定，TF 与 AR 才收敛。故 training dynamics 中不能用 TF final accuracy 代替可部署能力。</p>
      {table(["Step", "Nonthinking AR", "Thinking AR", "Thinking TF", "Thinking exact trace", "Thinking marker recall"], [["500", "10%", "12%", "100%", "1%", "12.4%"], ["1,500", "10%", "21%", "100%", "11%", "72.2%"], ["3,000", "24%", "36%", "100%", "21%", "80.9%"], ["5,000", "32%", "76%", "100%", "49%", "84.5%"], ["7,500", "63%", "91%", "100%", "71%", "91.3%"], ["10,000", "84%", "98%", "100%", "81%", "93.8%"]], {0,1,2,3,4,5})}
      {h1}
      <h3>4.3 机制涌现的时间顺序</h3>
      <p class="small"><strong>诊断套件说明：</strong>跨 checkpoint 的总体 AR 曲线保留每个 count 10 个样本的固定平衡套件；固定头与行为的直接关联则只使用与 head-selection 不重叠的 heldout-reporting prompts，并在产物的 <code>diagnostic_split</code> 字段中区分。两者不能逐行混合。</p>
      {table(["模式", "指标（固定头）", "step 0", "step 10,000", "Persistent half-rise", "与 AR 的 Pearson r"], [["Thinking", "Trace routing（L2H2）", "0.040", "0.873", "500", "0.412"], ["Thinking", "Ordered correct-k margin（L4H2）", "−0.009", "+0.467", "4,500", "0.983"], ["Thinking", "Needle enrichment（L3H0）", "0.978×", "2.917×", "4,500", "0.973"], ["Nonthinking", "Top-n coverage（L4H2）", "0.024", "0.842", "7,000", "0.943"], ["Nonthinking", "Needle enrichment（L4H2）", "1.013×", "21.058×", "7,500", "0.943"]], {2,3,4,5})}
      <p>Persistent half-rise 定义为：从某 checkpoint 起，该指标在此后所有 checkpoint 都不低于从 step 0 到 step 10,000 总增量的一半。相关系数按 21 个 checkpoint 计算，只是共同时间趋势下的描述性同步，不能当作因果效应。</p>
      <h3>4.4 Loss-scope 切换与语言能力遗忘</h3>
      <p>Full-sequence validation CE 在 step 1,000 最低（nonthinking 1.492、thinking 1.437），之后最终升至 7.573 与 4.822；这与 step 1,500 后不再优化 prompt/prefix token 一致。最终 test task token-weighted CE 分别为 7.669 与 4.860，但它仍被大量 prompt token 主导，且 test suite 没有单独 AR generation，因此最终 84%/98% 是 balanced validation AR 估计。</p>
      {table(["Step 10,000 validation component", "Nonthinking CE", "Thinking CE"], [["Final count", "0.596", "0.0002"], ["Trace index", "—", "0.003"], ["Trace marker", "—", "0.044"], ["Prompt", "7.642", "5.058"], ["Task prefix", "8.442", "7.999"]], {1,2})}
      {table(["Step 10,000 protected test suite", "Nonthinking token-weighted CE", "Thinking token-weighted CE"], [["Mixture（task ratio=1，等同 task）", "7.669", "4.860"], ["Task", "7.669", "4.860"], ["Raw language", "9.224", "4.895"]], {1,2})}
      {figs[13]}
    </section>

    <section id="attention-representation">
      <p class="section-kicker">v10 §5 adapted to v16.2</p>
      <h2>5. Attention representation：广聚合、定向检索与路由涌现</h2>
      <p class="lede">本节只回答“每个 query 从哪些 token 读取”。Attention weight 是候选路由，不等于 value 向量所携带的内容，也不等于该头对输出的因果贡献。</p>
      <h3>5.1 最终 checkpoint 的两种 representation</h3>
      {table(["同位点、同层头比较", "Thinking：trace index 的 L4H2", "Nonthinking：<Ans> 的 L4H2"], [["Prompt 目标总质量", "61.3%", "38.1%"], ["Prompt 非目标质量", "37.8%", "60.5%"], ["目标 enrichment", "25.0×", "21.1×"], ["目标内归一化熵", "0.382（集中）", "0.813（广覆盖）"], ["Top-n 目标 recall", "38.7%", "84.2%"], ["Correct-k top-1", "64.9%（chance 18.2%）", "未定义"], ["正确目标原始 mass", "33.9%", "未定义"]], {1,2})}
      <div class="callout"><strong>为什么 k-to-k mass 看起来不大、但检索仍有意义？</strong> 正确目标的 33.9% 是对整个 causal prefix 的绝对 softmax 质量，不是目标集合内的条件概率；同时 37.8% 仍落在 prompt 非目标字符上。可是目标内部 correct-k top-1 达 64.9%，比 18.2% 随机基线高 46.7 个百分点，说明“质量不占多数”和“相对身份选择很弱”不是同一件事。反过来，若只看 top-1 而不看 raw mass，也可能误判一个几乎不读取 prompt 的头。</div>
      <h3>5.2 层×头地图：电路如何按时间组装</h3>
      <p>下列 A1–A4 在 step 1,500、5,000、7,000、10,000 使用跨 checkpoint 统一色标。行是 transformer 层 L1–L4，列是 H0–H3；描边格为按最终 checkpoint 固定选择的头，因此避免每一步重新挑最好头造成 selection bias。</p>
      {attention_maps}
      {table(["指标最大值", "step 1,500", "step 5,000", "step 7,000", "step 10,000"], [["NT broad score", "L4H0 · 0.022", "L1H0 · 0.040", "L4H2 · 0.127", "L4H2 · 0.342"], ["Thinking raw correct-k mass", "L2H1 · 0.012", "L3H3 · 0.211", "L4H2 · 0.261", "L4H2 · 0.339"], ["Thinking correct-k margin", "L2H1 · +0.102", "L4H2 · +0.267", "L4H2 · +0.404", "L4H2 · +0.467"], ["Thinking trace-readout mass", "L2H2 · 0.895", "L2H2 · 0.846", "L2H2 · 0.874", "L2H2 · 0.873"]], {1,2,3,4})}
      <p><strong>组装顺序：</strong>第 1,500 步 L2 已形成答案对 trace 的读取；第 5,000 步第 3 层开始分布式检索目标集合；第 7,000–10,000 步 L4H2 固化为 ordered retrieval 专门头。Nonthinking 的答案时广聚合更晚，主要在第 4 层出现。</p>
      {table(["Step", "Thinking correct-k margin", "正确目标 raw mass", "Thinking 目标熵", "NT top-n recall", "NT 目标 mass", "NT 目标熵", "Thinking / NT AR"], [["0", "−0.9 pp", "0.4%", "0.981", "2.4%", "2.1%", "0.899", "0% / 1%"], ["1,500", "−1.6 pp", "0.3%", "0.856", "3.1%", "2.3%", "0.827", "21% / 10%"], ["3,000", "+11.3 pp", "2.9%", "0.695", "3.4%", "2.9%", "0.706", "36% / 24%"], ["5,000", "+26.7 pp", "13.0%", "0.510", "5.1%", "3.0%", "0.680", "76% / 32%"], ["7,000", "+40.4 pp", "26.1%", "0.403", "55.9%", "16.6%", "0.733", "90% / 56%"], ["8,500", "+44.9 pp", "32.2%", "0.382", "83.6%", "35.0%", "0.804", "98% / 80%"], ["10,000", "+46.7 pp", "33.9%", "0.382", "84.2%", "38.1%", "0.813", "98% / 84%"]], {0,1,2,3,4,5,6,7})}
      <h3>5.3 Nonthinking：后期直接从 prompt 广聚合</h3>
      <p>最终 checkpoint 的行为关联严格使用由 head-selection prompts 选出的固定 {metrics['fixed_head']}，再只在不重叠的 heldout-reporting prompts 上统计。正确样本（n={int(nt_correct.examples)}）的 enrichment/top-n recall 为 {nt_correct.enrichment:.2f}×/{100*nt_correct.recall:.1f}%，错误样本（n={int(nt_incorrect.examples)}）为 {nt_incorrect.enrichment:.2f}×/{100*nt_incorrect.recall:.1f}%。错误样本也并非完全没有检索，因此失败还可能发生在 value aggregation 或 readout；该分组是观察性关联，不是 head 的因果效应。</p>
      <div class="two-up">{figs[3]}{figs[4]}</div>
      <h3>5.4 Thinking：先将答案路由到 trace，再形成有序 prompt 检索</h3>
      <p>L2H2 在 step 500 已有 83.8% answer-to-trace mass，最终为 87.3%；这是一个早期稳定的 readout scaffold。与之相对，L4H2 的 ordered retrieval 到中后期才成熟。最终第 3 层四个头的 enrichment 为 24.8×–34.4×、top-n recall 为 56.5%–77.8%，说明目标集合过滤先以分布式形式出现。按 k 分解时，最终 fixed-head top-1 为 k=1: 56.0%（chance 29.3%）、k=2: 30.1%（21.4%）、k=5: 16.2%（14.1%）、k=9: 13.8%（10.6%）、k=10: 31.3%（10.0%）；边界较易、内部位置较难。</p>
      {figs[5]}
      {figs[6]}
      <div class="callout caution"><strong>证据边界。</strong> 这些图说明 attention routing 与行为同步涌现，但没有证明 L4H2 或 L2H2 是必要且充分的。需要 head ablation、attention/value patching、对正确与错误 occurrence 的定点交换，才能把“看向哪里”升级为因果电路结论。Nonthinking 没有显式 k interface，也不能据此排除其在隐藏状态中存在串行计算。</div>
    </section>

    <section id="residual-representation">
      <p class="section-kicker">v10 §6 adapted to v16.2</p>
      <h2>6. Residual representation：count manifold 与 CoT trace progress</h2>
      <p class="lede">Attention 描述“从哪里读”；本节问“读完以后，模型在 256 维 residual stream 中写入了什么”。所有最终答案 state 都取自 <code>&lt;Ans&gt;</code> 位置，而不是答案数字 token；因此 causal decoder 尚未看到正确答案本身。</p>
      <h3>6.1 最终答案 count representation</h3>
      {table(["模式 / context", "最佳层", "Nearest-centroid acc.", "Ridge MAE", "Ridge R²"], [["Nonthinking · teacher-forced", "L4", "73.3%", "0.411", "0.957"], ["Nonthinking · generated prefix", "L4", "75.0%", "0.418", "0.954"], ["Thinking · teacher-forced", "L2", "100.0%", "0.096", "0.998"], ["Thinking · generated prefix", "L2", "98.0%", "0.107", "0.996"]], {2,3,4})}
      <p>Generated-prefix probe 与 teacher-forced 接近，说明 thinking 的高可解码性不是只靠 gold trace 注入。Thinking 的 count state 在中间层较早出现；nonthinking 的 count state 则到后期才在 L4 成形。</p>
      {figs[7]}
      <h3>6.2 Trace 内部的 progress representation</h3>
      <p>Thinking 额外在第 k 个 index token 与 marker M<sub>k</sub> 位置抽取 state。去混淆后所有 k 都来自 total count=10；最终 L3 marker state 达 {100*trace_probe.nearest_centroid_accuracy:.1f}% centroid accuracy、ridge R²={trace_probe.ridge_r2:.3f}。因此旧的“progress=k 与 total count=k 共线”解释被排除。trace index 的标签在输入 token 已显式给出，所以低层高可解码仍不能当作“模型完成计数”的证据；更重要的是 marker/answer state 及跨位置转移。</p>
      {figs[8]}
      <h3>6.3 v10 风格 mean-first manifold：本次新增分析</h3>
      {figs[16]}
      {geom_table}
      <p>Nonthinking L4 的 PC1 解释 {100*nt_rep.pc1_variance:.1f}% centroid 方差、前两维 {100*nt_rep.pc1_to_pc2_variance:.1f}%，平均相邻位移余弦 {nt_rep.mean_adjacent_displacement_cosine:+.3f}。Thinking L2 答案 state 的 PC1 解释 {100*th_rep.pc1_variance:.1f}%、有效维度 {th_rep.effective_dimension:.2f}，平均相邻位移余弦 {th_rep.mean_adjacent_displacement_cosine:+.3f}。这些 mean-first 几何与 256 维 ridge 的高可解码性并不矛盾：<strong>linearly decodable count ≠ globally straight +1 trajectory</strong>。</p>
      <p>在 total count 固定为 10 的去混淆样本上，trace index 与 marker 在 L3 的联合 PCA 中仍按 token role 与 progress 分离；两条轨迹的相邻更新方向平均为 {index_rep.mean_adjacent_displacement_cosine:+.3f} 与 {marker_rep.mean_adjacent_displacement_cosine:+.3f}。它支持阶段特异的 progress code，但不支持“每一步在 residual stream 中加同一个固定向量”的简单模型。</p>
      <h3>6.4 Trace-to-answer 的跨位置转移</h3>
      <p>在同层训练 trace-progress ridge readout，再用于预测 answer count：L2 trace-to-answer 的 R²={cross_l2.loc['trace_to_answer'].r2:.3f}、MAE={cross_l2.loc['trace_to_answer'].mae:.3f}；反向 answer-to-trace 的 R²={cross_l2.loc['answer_to_trace'].r2:.3f}。方向余弦为 {cross_l2.loc['trace_to_answer'].direction_cosine:.3f}，其计算使用 raw-hidden-space 方向 <code>β_std / scale</code>，而不是直接比较各自标准化坐标中的 β。它说明两个位置共享可迁移信息，但不能据此断言同一根全局轴或因果流向。</p>
      {figs[9]}
      <h3>6.5 表示何时稳定：CKA 与几何重组</h3>
      <p>Thinking L4 answer geometry 相对最终形态的 CKA 在 step 500 已为 0.882、step 1,500 为 0.910、step 5,000 为 0.969；nonthinking 到 step 5,000 仅 0.307，step 7,500 才升至 0.761。Nonthinking 的 PC1-count R² 从 step 5,000 的 0.948 降到最终 0.708，同时有效维度从 1.34 升到 3.01、ridge R² 反而改善：这不是 count 信息消失，而是从早期近单轴编码扩展为更高维的成熟表示。</p>
      <div class="two-up">{figs[10]}{figs[11]}</div>
      <div class="callout warning"><strong>与 v10 的对应与差异。</strong> 分析逻辑与 v10 第六节一致：答案 centroid、trace token role×progress、mean-first PCA、解码与因果分离。v16.2 没有 v10 的 fixed-15 counterfactual state，因此本报告不虚构该对照；取而代之的是 generated-prefix probes、跨位置 transfer、CKA 和实际 trace deletion。</div>
    </section>

    <section id="causal">
      <p class="section-kicker">Intervention</p>
      <h2>7. 从“表征相关”升级到因果证据：删除一个 trace pair</h2>
      <p>对真实 n≥2 的 gold thinking trace，删除最后一对 <code>&lt;n&gt; marker</code>，然后比较最终答案分布与 L4 state 的 count decoder。完整 trace 时 P(n)=0.99981、P(n−1)=0.00003、logit margin=+11.15、decoded count=5.990；删除后 P(n)=0.00001、P(n−1)=0.99984、margin=−11.77、decoded count=4.974。变化分别为 −0.99980、+0.99981、−22.91 与 −1.016。</p>
      {figs[12]}
      <p>这是当前最强证据：最终答案真正读取了外部 trace 中的进度，而不是 trace 仅与内部答案副本相关。但删除两个 token 同时改变 <code>&lt;Ans&gt;</code> 的绝对/相对位置，仍可能有长度混杂。更干净的后续实验应做 length-preserving pair replacement、marker/index 分开 patch，并在同一位置直接 steer progress direction。</p>
    </section>

    <section id="data-noise">
      <p class="section-kicker">Structure, imbalance, observational noise</p>
      <h2>8. 训练数据结构与 noise 的影响</h2>
      <h3>8.1 接受过滤造成的 count 不均衡</h3>
      <p>每个模型共见到 1,280,000 个训练样本；窗口候选的接受率为 57.4%，拒绝中 34.6% 因 count=0，8.0% 因 count&gt;10。接受样本平均 count=4.794，平均 98.1% prompt token 是非目标 noise。虽然 needle pool 的频率 bin 均衡，接受后的 count 明显偏向小值：count 1 占 16.16%，count 10 仅 5.75%，相对均衡目标的 KL divergence 为 0.035。</p>
      {table(["count", "训练样本数", "训练概率", "相对均衡评估比例"], [["1", "206,790", "16.16%", "1.616×"], ["2", "152,572", "11.92%", "1.192×"], ["3", "134,554", "10.51%", "1.051×"], ["4", "132,699", "10.37%", "1.037×"], ["5", "131,861", "10.30%", "1.030×"], ["6", "130,564", "10.20%", "1.020×"], ["7", "118,935", "9.29%", "0.929×"], ["8", "107,381", "8.39%", "0.839×"], ["9", "91,036", "7.11%", "0.711×"], ["10", "73,608", "5.75%", "0.575×"]], {0,1,2,3})}
      <p>两种模型使用完全相同的数据流，因此不均衡不能解释 thinking–nonthinking 的差距；最终 evaluation 按 count 均衡，避免总准确率被高频小 count 主导。</p>
      <h3>8.2 Noise/structure 特征如何定义</h3>
      <div class="equation">目标密度=n/256；noise fraction=1−n/256；字符平衡熵=−Σ<sub>j=1..3</sub>p<sub>j</sub>log p<sub>j</sub>/log 3；occurrence span=(max position−min position)/255；最长 noise run=最长连续非目标长度/256；prompt entropy=−Σ<sub>char</sub>p(char)log₂p(char)。</div>
      <p>此外计算 target-character switch rate。所有结构关联先在每个 count 内对特征和正确性去均值，再汇总 Pearson correlation；95% CI 来自 2,000 次按 count 分层 bootstrap。因此这些数值控制了一阶 count 混杂，但仍是自然数据上的观察性关系。</p>
      {figs[15]}
      {table(["模式", "结构特征", "控制 count 的 r", "bootstrap 95% CI"], [["Nonthinking", "needle-set 语料频率", "+0.168", "[−0.096, +0.421]"], ["Nonthinking", "三字符平衡熵", "−0.098", "[−0.291, +0.108]"], ["Nonthinking", "occurrence span", "−0.157", "[−0.383, +0.093]"], ["Nonthinking", "最长 noise run", "+0.068", "[−0.107, +0.245]"], ["Nonthinking", "prompt 字符熵", "−0.108", "[−0.302, +0.102]"], ["Nonthinking", "target switch rate", "−0.053", "[−0.248, +0.162]"], ["Thinking", "needle-set 语料频率", "−0.186", "[−0.346, −0.067]"], ["Thinking", "三字符平衡熵", "−0.026", "[−0.265, +0.216]"], ["Thinking", "occurrence span", "−0.085", "[−0.218, +0.053]"], ["Thinking", "最长 noise run", "+0.252", "[+0.135, +0.420]"], ["Thinking", "prompt 字符熵", "+0.153", "[+0.070, +0.280]"], ["Thinking", "target switch rate", "−0.099", "[−0.424, +0.168]"]], {2,3})}
      <p>Nonthinking 的所有 CI 均跨 0，样本量只有每 representation 100，无法得出稳定结构效应。Thinking 因 98% ceiling 只剩两个错误，出现显著相关也极其脆弱；例如较长 noise run 与更高准确率不是“noise 有益”的因果结论，更可能反映两个错误样本在条件空间中的位置。</p>
    </section>

    <section id="runtime-repro">
      <p class="section-kicker">Audit trail</p>
      <h2>9. 运行成本、产物与复现审计</h2>
      <h3>9.1 Runtime</h3>
      <p>训练 pipeline 总计 79.4 分钟：optimizer loop 51.6 分钟、周期性 TF evaluation 17.8 分钟、checkpoint dynamics 9.8 分钟；TF evaluation 约占 22.5%，序列化仅 7.6 秒。Thinking optimizer 约 28.7 分钟，nonthinking 22.9 分钟；显式 trace 也使 AR generation 更慢。全部 523 个 timing event 完成，训练与完整 dynamics 合计约 90 分钟。</p>
      {figs[14]}
      <h3>9.2 Artifact 完整性</h3>
      <p>本地包含 42 个 RoPE checkpoint（2 模式×21 steps，0–10,000 每 500 步），总计 1,555,295,236 bytes，42 个 SHA256 均唯一；step 0 与 step 10,000 的两模式 boundary checkpoint 均可用本地代码加载，每个模型 3,180,800 参数。配置审计 35/35 个关键字段匹配；四个边界加载均通过；机制表包含约 504k attention-detail rows、4,200 AR rows、625 state-probe summary rows、28,350 counterfactual rows、820 similarity rows。</p>
      {table(["审计项", "结果"], [["Run ID", RUN_ID], ["Position encoding", "仅 RoPE；不存在 RPE run 混入"], ["Checkpoint inventory", "42/42，尺寸与来源 manifest 一致"], ["边界 checkpoint load", "4/4 PASS"], ["本地测试", "本轮完整 pytest 与新增回归测试均通过（见交付说明）"], ["State 采样审计", "trace progress 的 total count 固定为 10，逐标签平衡"], ["原始参考 HTML SHA256", "A957428E1B06358EEAEB83CF807E2C18F70D56E4B957FB5936403D582DE1F3E8"]])}
      <h3>9.3 关键产物</h3>
      <ul class="artifact-list">
        <li><a href="config.json">config.json</a>；<a href="manifest.json">manifest.json</a>；<a href="checkpoint_sources.tsv">checkpoint_sources.tsv</a></li>
        <li><a href="tables/checkpoint_attention_summary.csv">attention dynamics summary</a>；<a href="tables/checkpoint_attention_by_k.csv">k-to-k detail</a>；<a href="tables/checkpoint_state_probe_summary.csv">state probes</a></li>
        <li><a href="tables/checkpoint_state_geometry.csv">state geometry</a>；<a href="tables/checkpoint_state_cross_site.csv">cross-site transfer</a>；<a href="tables/checkpoint_state_similarity.csv">CKA</a></li>
        <li><a href="tables/checkpoint_counterfactual_trace_readout.csv">trace deletion counterfactual</a>；<a href="tables/data_structure_accuracy_associations.csv">data/noise associations</a></li>
        <li><a href="tables/v16_2_representation_geometry_summary.csv">新增 v10-style geometry summary</a>；<a href="figures/v16_2_representation_geometry_v10_style.png">新增图 16 PNG</a></li>
      </ul>
      <p>审计确保本地默认值、notebook、result artifacts 与指定 RoPE run 对齐；但 bitwise retraining 仍取决于原 Colab CUDA/cuDNN、kernel 与确定性设置，不能仅凭 checkpoint 完整性承诺逐 bit 相同。</p>
    </section>

    <section id="limits">
      <p class="section-kicker">What is supported</p>
      <h2>10. 综合解释、证据边界与下一步</h2>
      <h3>10.1 当前证据支持什么</h3>
      <ul>
        <li>Thinking 最终采用“prompt 的逐个有序检索 → trace 中显式 progress → 答案从 trace 读取”的路径；其行为与 ordered-retrieval 指标共同在中期涌现。</li>
        <li>Nonthinking 最终采用“答案位置第 4 层对多个目标出现位置广覆盖 → 在 L4 residual 中形成 count state”的直接路径，并明显晚于 thinking 成熟。</li>
        <li>两种 representation 的最终 count 均可线性解码，但 thinking 更早、更准确且更高维；PCA 的连续轨迹与 probe 的高 R² 都不能单独证明算术算法。</li>
        <li>Trace deletion 证明 thinking 的最终输出实际使用 trace progress；这是报告中最强的干预证据。</li>
      </ul>
      <h3>10.2 尚未建立什么</h3>
      <ul>
        <li>没有证明某个单独 attention head 必要或充分；也没有将 attention weight 与 value/output contribution 分解。</li>
        <li>没有证明 nonthinking 完全没有隐式串行步骤；“没有显式 k token”只说明 strict k-to-k 指标无定义。</li>
        <li>Trace deletion 仍有长度/位置混杂；自然 data/noise 相关也不是随机因果效应。</li>
        <li>单 seed、单 corpus、单模型尺度限制了泛化；最终 test bundle 缺少独立 AR generation。</li>
      </ul>
      <h3>10.3 信息增益最高的后续实验</h3>
      <ol>
        <li><strong>Head-level causal tests：</strong>对 L4H2 ordered retrieval、L2H2 trace readout、nonthinking L4H2 broad aggregation 做 ablation 与 activation/value patch。</li>
        <li><strong>Length-preserving trace intervention：</strong>把最后 pair 替换成同长度错误 pair，分别 patch index 与 marker，排除位置变化。</li>
        <li><strong>Representation steering：</strong>使用跨验证得到的 progress/count direction 做小幅 steer，检验输出是否按 ±1 改变；不要用同一数据拟合与验证。</li>
        <li><strong>Randomized structure/noise sweep：</strong>固定 count，独立操纵 occurrence span、最长 noise run、字符 balance 与集合频率。</li>
        <li><strong>Seeds/scales/corpora：</strong>重复至少 3–5 seeds，并改变层数、context 与 corpus，检验“routing→retrieval→readout”的顺序是否稳定。</li>
        <li><strong>Independent test AR：</strong>在受保护 test 区域生成完整 trace/answer，复制所有 permutation、probe 与 counterfactual 分析。</li>
      </ol>
      <div class="callout success"><strong>整体结论。</strong> v16.2 RoPE 最清楚的故事不是“CoT 让同一个计数器更准”，而是监督格式改变了表示的接口与学习顺序：thinking 先建立答案到 trace 的读取接口，再学会 ordered retrieval 并把 progress 写进高维 residual geometry；nonthinking 则更晚地在答案位置形成多目标广聚合和直接 count state。两者都能编码 count，但它们不是同一种 representation 的简单强弱版本。</div>
    </section>

    <footer>生成依据：本地 v16.2 RoPE checkpoints、完整机制表与已审计英文报告。中文版新增 mean-first residual geometry，但不改动原始 checkpoint 或原始参考 HTML。报告生成脚本：<a href="../../scripts/build_v16_2_rope_chinese_report.py">scripts/build_v16_2_rope_chinese_report.py</a>。</footer>
  </div>
</main>
</body>
</html>
"""

    chinese = run_dir / "v16_2_rope_complete_report_zh.html"
    core = run_dir / "v16_2_rope_core_report_zh.html"
    chinese.write_text(body, encoding="utf-8", newline="\n")
    core.write_text(body, encoding="utf-8", newline="\n")
    complete.write_text(body, encoding="utf-8", newline="\n")
    return chinese, complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("colab_results/v16_2_main_rope_seed1234"),
    )
    args = parser.parse_args()
    chinese, complete = build(args.run_dir.resolve())
    print(chinese)
    print(complete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
