from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    value = float(value)
    if value != 0 and abs(value) < 10 ** (-digits):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def code(value: object) -> str:
    return f"<code>{html.escape(str(value))}</code>"


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def figure(path: Path, title: str, caption: str, *, wide: bool = False) -> str:
    if not path.exists():
        return ""
    cls = "figure wide" if wide else "figure"
    return f"""
    <figure class="{cls}">
      <h3>{html.escape(title)}</h3>
      <img src="{image_uri(path)}" alt="{html.escape(title)}">
      <figcaption>{caption}</figcaption>
    </figure>"""


def table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{row.get(key, '')}</td>" for key, _ in columns) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def weighted(df: pd.DataFrame, metric: str) -> float:
    weights = df.get("n_examples", pd.Series(np.ones(len(df)), index=df.index)).astype(float)
    values = df[metric].astype(float)
    ok = values.notna() & weights.notna()
    return float(np.average(values[ok], weights=weights[ok]))


def save_attention_panel(attn: pd.DataFrame, path: Path) -> None:
    think = attn.query("mode == 'thinking'").copy()
    metrics = [
        ("correct_top1", "Correct top-1 retrieval"),
        ("diagonal_dominance", "Needle-conditional diagonal dominance"),
        ("needle_mass", "Raw mass on all prompt needles"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        pivot = think.pivot(index="layer", columns="head", values=metric).sort_index()
        sns.heatmap(pivot, vmin=0, vmax=1, cmap="viridis", annot=True, fmt=".2f", ax=ax, cbar=metric == "needle_mass")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("head (0-based)")
        ax.set_ylabel("layer (0-based)")
    fig.suptitle("Thinking attention at <I_k>: query predicts M_k", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_switch_probability(switch: pd.DataFrame, path: Path) -> None:
    grouped = switch.groupby(["step", "mode"], as_index=False).apply(
        lambda g: pd.Series({"p_desired": weighted(g, "p_desired_next_token")}),
        include_groups=False,
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    for mode, group in grouped.groupby("mode"):
        ax.plot(group.step, group.p_desired, marker="o", ms=3.5, label=mode)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean P(desired first token)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="requested mode")
    ax.grid(alpha=.25)
    ax.set_title("Explicit switch routing confidence")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_probe_panel(probes: pd.DataFrame, path: Path) -> None:
    subset = probes.query("target == 'final_count' and leakage_prone == False").copy()
    anchors = ["mode_pos", "think_open_pos", "pre_count_pos", "think_close_pos"]
    subset = subset[subset.anchor_name.isin(anchors)]
    best = subset.groupby(["mode", "anchor_name", "layer"], as_index=False).accuracy.max()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, ["nonthinking", "thinking"]):
        data = best[best["mode"] == mode].pivot(index="anchor_name", columns="layer", values="accuracy").reindex(anchors)
        sns.heatmap(data, vmin=0, vmax=1, cmap="Blues", annot=True, fmt=".2f", ax=ax, cbar=mode == "thinking")
        ax.set_title(mode)
        ax.set_xlabel("hidden-state index (0=embedding, 1-4=blocks)")
        ax.set_ylabel("anchor")
    fig.suptitle("Linear classification accuracy for final count", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def first_perfect_step(eval_df: pd.DataFrame, mode: str) -> int | None:
    for step in sorted(eval_df.step.unique()):
        rows = eval_df[(eval_df.step == step) & (eval_df["mode"] == mode)]
        if weighted(rows, "final_accuracy") < 1:
            continue
        if mode == "thinking" and weighted(rows, "trace_exact") < 1:
            continue
        return int(step)
    return None


def build_report(run_dir: Path) -> str:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    tables = run_dir / "tables"
    source_figures = run_dir / "figures"
    report_figures = run_dir / "report_figures"
    report_figures.mkdir(exist_ok=True)

    train_log = pd.read_csv(tables / "train_log.csv")
    eval_df = pd.read_csv(tables / "eval_by_step.csv")
    switch = pd.read_csv(tables / "mode_switch.csv")
    attn = pd.read_csv(tables / "attention_metrics.csv")
    probes = pd.read_csv(tables / "probe_results.csv")
    similarity = pd.read_csv(tables / "mode_hidden_similarity.csv")

    save_attention_panel(attn, report_figures / "indexed_retrieval_metrics.png")
    save_switch_probability(switch, report_figures / "switch_probability.png")
    save_probe_panel(probes, report_figures / "probe_accuracy.png")

    final_step = int(eval_df.step.max())
    final_eval = eval_df[eval_df.step == final_step]
    final_switch = switch[switch.step == switch.step.max()]
    modes = ["nonthinking", "thinking"]
    summary = {}
    for mode in modes:
        e = final_eval[final_eval["mode"] == mode]
        s = final_switch[final_switch["mode"] == mode]
        summary[mode] = {
            "n": int(e.n_examples.sum()),
            "final": weighted(e, "final_accuracy"),
            "mae": weighted(e, "final_mae"),
            "trace": weighted(e, "trace_exact"),
            "precision": weighted(e, "trace_marker_precision"),
            "recall": weighted(e, "trace_marker_recall"),
            "empty": weighted(e, "empty_trace_rate"),
            "switch": weighted(s, "argmax_is_desired"),
            "p_desired": weighted(s, "p_desired_next_token"),
        }

    think_attn = attn[(attn["mode"] == "thinking") & (attn.query_anchor == "index_token_k_predicts_marker_k")].sort_values(
        ["correct_top1", "needle_mass"], ascending=False
    )
    best = think_attn.iloc[0]
    perfect = think_attn[(think_attn.correct_top1 >= .999) & (think_attn.needle_mass >= .99)]
    random_top1 = float(
        np.mean(
            [
                1.0 / count
                for count in range(
                    int(config["train"]["count_min"]),
                    int(config["train"]["count_max"]) + 1,
                )
            ]
        )
    )

    mode_similarity = similarity.groupby("layer").cosine_similarity.agg(["mean", "std"]).reset_index()
    sim_embedding = float(mode_similarity.sort_values("layer").iloc[0]["mean"])
    sim_final = float(mode_similarity.sort_values("layer").iloc[-1]["mean"])
    sim_rows = [
        {"state": "embedding" if int(r.layer) == 0 else f"after block {int(r.layer)}", "mean": fmt(r["mean"]), "std": fmt(r["std"])}
        for _, r in mode_similarity.iterrows()
    ]
    head_rows = [
        {
            "head": f"L{int(r['layer'])}H{int(r['head'])} (block {int(r['layer'])+1})",
            "top1": fmt(r.correct_top1),
            "diag": fmt(r.diagonal_dominance),
            "mass": fmt(r.needle_mass),
            "entropy": fmt(r.entropy),
        }
        for _, r in think_attn.head(8).iterrows()
    ]
    behavior_rows = [
        {
            "mode": "THINK_OFF / non-thinking",
            "n": f"{summary['nonthinking']['n']:,}",
            "acc": pct(summary["nonthinking"]["final"]),
            "trace": "empty by design",
            "switch": pct(summary["nonthinking"]["switch"]),
            "p": fmt(summary["nonthinking"]["p_desired"], 6),
        },
        {
            "mode": "THINK_ON / thinking",
            "n": f"{summary['thinking']['n']:,}",
            "acc": pct(summary["thinking"]["final"]),
            "trace": pct(summary["thinking"]["trace"]),
            "switch": pct(summary["thinking"]["switch"]),
            "p": fmt(summary["thinking"]["p_desired"], 6),
        },
    ]
    train = config["train"]
    model = config["model"]
    trace_indices = bool(config.get("trace_indices"))
    trace_format = "<I1> M1 <I2> M2 ... <In> Mn" if trace_indices else "M1 M2 ... Mn"
    vocab_size = int(model["vocab_size"])
    query_count = int(sum(range(train["count_min"], train["count_max"] + 1)) * train["attention_examples_per_count"])
    loss_labels = [
        ("loss_total", "全部有效 label tokens"),
        ("loss_thinking_trace", "THINK_ON trace tokens + </Think>"),
        ("loss_thinking_final_count", "THINK_ON <C_n>"),
        ("loss_nonthinking_close", "THINK_OFF </Think>"),
        ("loss_nonthinking_final_count", "THINK_OFF <C_n>"),
    ]
    loss_rows = [
        {
            "component": label,
            "tokens": tokens,
            "first": fmt(train_log.iloc[0][label]),
            "final": fmt(train_log.iloc[-1][label]),
        }
        for label, tokens in loss_labels
    ]
    probe_summary_rows: list[dict[str, object]] = []
    probe_best_map: dict[tuple[str, str], pd.Series] = {}
    probe_subset = probes[(probes.target == "final_count") & (~probes.leakage_prone.astype(bool))]
    for mode in ("nonthinking", "thinking"):
        for anchor in ("mode_pos", "think_open_pos", "think_close_pos", "pre_count_pos"):
            part = probe_subset[(probe_subset["mode"] == mode) & (probe_subset.anchor_name == anchor)]
            if part.empty:
                continue
            row = part.sort_values("accuracy", ascending=False).iloc[0]
            probe_best_map[(mode, anchor)] = row
            probe_summary_rows.append(
                {
                    "mode": mode,
                    "anchor": code(anchor),
                    "layer": int(row.layer),
                    "accuracy": fmt(row.accuracy),
                    "r2": fmt(row.r2),
                    "mae": fmt(row.mae),
                    "position": fmt(row.position_baseline_acc),
                    "trace_len": fmt(row.trace_len_baseline_acc),
                }
            )

    v53_dir = run_dir / "v5_3_mechanism_causal"
    v53_tables = v53_dir / "tables"
    v53_figures = v53_dir / "figures"
    v53_section = ""
    if (v53_tables / "attention_head_summary.csv").exists():
        v53_config = json.loads((v53_dir / "run_config.json").read_text(encoding="utf-8"))
        head_groups = json.loads((v53_dir / "head_groups.json").read_text(encoding="utf-8"))
        v53_attention = pd.read_csv(v53_tables / "attention_head_summary.csv")
        v53_ablation = pd.read_csv(v53_tables / "head_ablation_summary.csv")
        v53_behavior = pd.read_csv(v53_tables / "behavioral_ablation_summary.csv")
        v53_patching = pd.read_csv(v53_tables / "patching_summary.csv")
        v53_conflict = pd.read_csv(v53_tables / "trace_conflict_rows.csv")
        v53_progress = pd.read_csv(v53_tables / "progress_transplant_summary.csv")

        def select_one(frame: pd.DataFrame, **filters: object) -> pd.Series:
            selected = frame
            for key, value in filters.items():
                selected = selected[selected[key] == value]
            if selected.empty:
                return pd.Series(dtype=object)
            return selected.iloc[0]

        def value(row: pd.Series, key: str) -> float:
            return float(row[key]) if key in row and pd.notna(row[key]) else math.nan

        def heads_text(name: str) -> str:
            return " ".join(f"L{int(layer)}H{int(head)}" for layer, head in head_groups.get(name, []))

        direct_signature = select_one(v53_attention, mode="nonthinking", query_kind="final_count_query", layer=0, head=1)
        targeted_signature_h0 = select_one(v53_attention, mode="thinking", query_kind="trace_marker_query", layer=1, head=0)
        targeted_signature_h1 = select_one(v53_attention, mode="thinking", query_kind="trace_marker_query", layer=1, head=1)
        targeted_signature_h2 = select_one(v53_attention, mode="thinking", query_kind="trace_marker_query", layer=1, head=2)
        trace_readout_signature = select_one(v53_attention, mode="thinking", query_kind="final_count_query", layer=3, head=1)
        direct_top1 = select_one(v53_ablation, mode="nonthinking", group_name="direct_broad_top1")
        direct_top2 = select_one(v53_ablation, mode="nonthinking", group_name="direct_broad_top2")
        targeted_top1 = select_one(v53_ablation, mode="thinking", group_name="targeted_top1")
        targeted_top2 = select_one(v53_ablation, mode="thinking", group_name="targeted_top2")
        targeted_top4 = select_one(v53_ablation, mode="thinking", group_name="targeted_top4")
        targeted_behavior2 = select_one(v53_behavior, mode="thinking", group_name="targeted_top2")
        targeted_behavior4 = select_one(v53_behavior, mode="thinking", group_name="targeted_top4")
        direct_behavior2 = select_one(v53_behavior, mode="nonthinking", group_name="direct_broad_top2")
        direct_behavior4 = select_one(v53_behavior, mode="nonthinking", group_name="direct_broad_top4")
        thinking_direct2 = select_one(v53_ablation, mode="thinking", group_name="direct_broad_top2")
        thinking_direct_behavior2 = select_one(v53_behavior, mode="thinking", group_name="direct_broad_top2")
        thinking_trace4 = select_one(v53_ablation, mode="thinking", group_name="trace_readout_top4")
        thinking_trace_behavior4 = select_one(v53_behavior, mode="thinking", group_name="trace_readout_top4")

        retrieval_patch1 = select_one(v53_patching, experiment="retrieval_identity", group_name="targeted_top1")
        retrieval_patch2 = select_one(v53_patching, experiment="retrieval_identity", group_name="targeted_top2")
        retrieval_patch4 = select_one(v53_patching, experiment="retrieval_identity", group_name="targeted_top4")
        retrieval_wrong_pos = select_one(
            v53_patching,
            experiment="retrieval_identity",
            group_name="targeted_wrong_donor_position",
        )
        direct_patch1 = select_one(v53_patching, experiment="nonthinking_count_readout", group_name="direct_broad_top1")
        direct_patch2 = select_one(v53_patching, experiment="nonthinking_count_readout", group_name="direct_broad_top2")
        direct_patch4 = select_one(v53_patching, experiment="nonthinking_count_readout", group_name="direct_broad_top4")
        think_direct_patch2 = select_one(v53_patching, experiment="thinking_count_readout", group_name="direct_broad_top2")
        think_direct_patch4 = select_one(v53_patching, experiment="thinking_count_readout", group_name="direct_broad_top4")
        think_trace_patch4 = select_one(v53_patching, experiment="thinking_count_readout", group_name="trace_readout_top4")

        off_diagonal = v53_conflict[v53_conflict.prompt_count != v53_conflict.forced_trace_count]
        follows_prompt = float(off_diagonal.follows_prompt.mean())
        follows_trace = float(off_diagonal.follows_trace.mean())
        follows_other = 1.0 - follows_prompt - follows_trace
        progress_shift = v53_progress[v53_progress.donor_kind == "same_example_progress_shift"]
        progress_success = float(progress_shift.patched_predicts_donor_next.mean())
        progress_control = v53_progress[v53_progress.donor_kind == "different_prompt_same_progress_control"]
        progress_control_success = float(progress_control.patched_predicts_receiver_next.mean())

        group_rows = [
            {
                "role": "non-thinking broad aggregation",
                "top1": code(heads_text("direct_broad_top1")),
                "top2": code(heads_text("direct_broad_top2")),
                "top4": code(heads_text("direct_broad_top4")),
                "selection": "final-count query 上的 broad_aggregation_score 从高到低",
            },
            {
                "role": "thinking targeted retrieval",
                "top1": code(heads_text("targeted_top1")),
                "top2": code(heads_text("targeted_top2")),
                "top4": code(heads_text("targeted_top4")),
                "selection": "trace-marker query 上正确 prompt needle 的 raw attention mass 从高到低",
            },
            {
                "role": "thinking trace readout",
                "top1": code(heads_text("trace_readout_top1")),
                "top2": code(heads_text("trace_readout_top2")),
                "top4": code(heads_text("trace_readout_top4")),
                "selection": "final-count query 投向全部 trace marker 的 attention mass 从高到低",
            },
        ]
        ablation_rows = [
            {
                "test": "non-thinking / direct top-1",
                "target": "count",
                "heads": code(heads_text("direct_broad_top1")),
                "tf": fmt(value(direct_top1, "drop_count_accuracy")),
                "margin": fmt(value(direct_top1, "drop_count_margin")),
                "free": "n/a",
                "trace": "n/a",
            },
            {
                "test": "non-thinking / direct top-2",
                "target": "count",
                "heads": code(heads_text("direct_broad_top2")),
                "tf": fmt(value(direct_top2, "drop_count_accuracy")),
                "margin": fmt(value(direct_top2, "drop_count_margin")),
                "free": pct(value(direct_behavior2, "final_accuracy")),
                "trace": "n/a",
            },
            {
                "test": "non-thinking / direct top-4",
                "target": "count",
                "heads": code(heads_text("direct_broad_top4")),
                "tf": fmt(value(select_one(v53_ablation, mode="nonthinking", group_name="direct_broad_top4"), "drop_count_accuracy")),
                "margin": fmt(value(select_one(v53_ablation, mode="nonthinking", group_name="direct_broad_top4"), "drop_count_margin")),
                "free": pct(value(direct_behavior4, "final_accuracy")),
                "trace": "n/a",
            },
            {
                "test": "thinking / targeted top-1",
                "target": "marker",
                "heads": code(heads_text("targeted_top1")),
                "tf": fmt(value(targeted_top1, "drop_trace_marker_accuracy")),
                "margin": fmt(value(targeted_top1, "drop_trace_marker_margin")),
                "free": "n/a",
                "trace": "n/a",
            },
            {
                "test": "thinking / targeted top-2",
                "target": "marker",
                "heads": code(heads_text("targeted_top2")),
                "tf": fmt(value(targeted_top2, "drop_trace_marker_accuracy")),
                "margin": fmt(value(targeted_top2, "drop_trace_marker_margin")),
                "free": pct(value(targeted_behavior2, "final_accuracy")),
                "trace": pct(value(targeted_behavior2, "trace_exact")),
            },
            {
                "test": "thinking / targeted top-4 (= all block-2 heads)",
                "target": "marker",
                "heads": code(heads_text("targeted_top4")),
                "tf": fmt(value(targeted_top4, "drop_trace_marker_accuracy")),
                "margin": fmt(value(targeted_top4, "drop_trace_marker_margin")),
                "free": pct(value(targeted_behavior4, "final_accuracy")),
                "trace": pct(value(targeted_behavior4, "trace_exact")),
            },
            {
                "test": "thinking / direct-broad top-2",
                "target": "count",
                "heads": code(heads_text("direct_broad_top2")),
                "tf": fmt(value(thinking_direct2, "drop_count_accuracy")),
                "margin": fmt(value(thinking_direct2, "drop_count_margin")),
                "free": pct(value(thinking_direct_behavior2, "final_accuracy")),
                "trace": pct(value(thinking_direct_behavior2, "trace_exact")),
            },
            {
                "test": "thinking / trace-readout top-4",
                "target": "count",
                "heads": code(heads_text("trace_readout_top4")),
                "tf": fmt(value(thinking_trace4, "drop_count_accuracy")),
                "margin": fmt(value(thinking_trace4, "drop_count_margin")),
                "free": pct(value(thinking_trace_behavior4, "final_accuracy")),
                "trace": pct(value(thinking_trace_behavior4, "trace_exact")),
            },
        ]
        patch_rows = [
            {
                "path": "retrieval identity",
                "intervention": "targeted top-1 / top-2 / top-4",
                "recovery": f"{fmt(value(retrieval_patch1, 'normalized_recovery'))} / {fmt(value(retrieval_patch2, 'normalized_recovery'))} / {fmt(value(retrieval_patch4, 'normalized_recovery'))}",
                "control": f"wrong donor position = {fmt(value(retrieval_wrong_pos, 'normalized_recovery'))}",
            },
            {
                "path": "non-thinking count readout",
                "intervention": "direct broad top-1 / top-2 / top-4",
                "recovery": f"{fmt(value(direct_patch1, 'normalized_recovery'))} / {fmt(value(direct_patch2, 'normalized_recovery'))} / {fmt(value(direct_patch4, 'normalized_recovery'))}",
                "control": "targeted top-4 ≈ 0；路径选择具有特异性",
            },
            {
                "path": "thinking count readout",
                "intervention": "direct broad top-2 / top-4",
                "recovery": f"{fmt(value(think_direct_patch2, 'normalized_recovery'))} / {fmt(value(think_direct_patch4, 'normalized_recovery'))}",
                "control": f"trace-readout top-4 = {fmt(value(think_trace_patch4, 'normalized_recovery'))}",
            },
        ]
        v53_setting_rows = [
            {
                "item": "checkpoint",
                "setting": f"直接加载同一个 v5 final checkpoint；不重新训练；device={v53_config['device']}",
            },
            {
                "item": "attention signature",
                "setting": f"每个 count {v53_config['attention_examples_per_count']} 个样本，共 {10 * v53_config['attention_examples_per_count']} 个 prompts；对 non-thinking final query、thinking 每个 trace query/successor query 和 final query 读取全部 16 heads；共 62,400 个 query-head rows",
            },
            {
                "item": "teacher-forced head ablation",
                "setting": f"每个 count {v53_config['ablation_examples_per_count']} 个样本，共 {10 * v53_config['ablation_examples_per_count']} prompts；逐组将 GPT-2 head_mask 设为 0，测 count/marker/successor logit margin 与 subset accuracy；共 6,000 mode×example×mask rows",
            },
            {
                "item": "autoregressive ablation",
                "setting": f"每个 count {v53_config['generation_examples_per_count']} 个样本，共 {10 * v53_config['generation_examples_per_count']} prompts；greedy 生成最多 26 tokens，测 final count 与 marker trace；共 1,020 mode×example×mask generations",
            },
            {
                "item": "activation patching",
                "setting": f"count 2–10 每个 count {v53_config['patch_examples_per_count']} 个 clean/corrupt pairs；patch clean head output 或 residual 到 corrupt run 的同一语义 query；共 6,300 pair×target×intervention rows",
            },
            {
                "item": "trace conflict / progress transplant",
                "setting": f"每个 prompt count {v53_config['conflict_examples_per_count']} 个样本；trace conflict 穷举 forced trace count 1–10（1,000 rows）；progress transplant 在 trace progress k=2 处替换 k−1/k+1 donor residual（840 rows）",
            },
        ]
        retrieval_ablation_caption = (
            "横轴是 gold marker 相对其他 marker 的平均 logit-margin drop；纵轴为被 mask 的 head group。"
            "最强两个 targeted heads 只降低 margin 而不改 argmax；mask 四个 L1 heads 后 "
            f"teacher-forced marker accuracy 平均下降 {pct(value(targeted_top4, 'drop_trace_marker_accuracy'))}，"
            f"free-running trace exact 降到 {pct(value(targeted_behavior4, 'trace_exact'))}，显示明显冗余。"
        )
        retrieval_patch_caption = (
            "左图横轴为 head-output patch 的 normalized recovery，纵轴为 patched group；右图为 patch 某层 block 输出 residual 的 recovery。"
            f"targeted top-2 已恢复 {pct(value(retrieval_patch2, 'normalized_recovery'))}，top-4 恢复约 100%；"
            f"把同一 heads 从错误 donor token position 搬来只恢复 {pct(value(retrieval_wrong_pos, 'normalized_recovery'))}，"
            "支持“正确 head + 正确 query 位置”的特异性。"
        )
        nonthinking_patch_caption = (
            "横轴为 normalized recovery，纵轴为 patched head group 或 residual layer。"
            f"Direct broad top-2/top-4 分别恢复 {pct(value(direct_patch2, 'normalized_recovery'))}/"
            f"{pct(value(direct_patch4, 'normalized_recovery'))}，而 targeted heads 约为 0，"
            "说明 non-thinking count 的直接 aggregation 路径具有功能特异性。"
        )
        thinking_patch_caption = (
            "定义同 Figure 15，但 clean 与 corrupt trace 长度不同，因此 donor/receiver 绝对位置不相同。"
            f"Direct broad top-4 恢复 {pct(value(think_direct_patch4, 'normalized_recovery'))}；"
            f"按 trace attention 选出的 top-4 只恢复 {pct(value(think_trace_patch4, 'normalized_recovery'))}。"
            "全 residual patch 接近完全恢复，说明 count 状态在 residual 中存在，但当前结果不支持由少数 trace-attending heads 单独读出。"
        )
        conflict_caption = (
            "横轴是人工 trace 长度 t，纵轴是 prompt needle count p；每个格子的颜色与数字是平均 predicted count。"
            f"对角线是无冲突 control。只统计 p≠t 的 900 个样本时，模型 {pct(follows_prompt)} 跟随 prompt、"
            f"{pct(follows_trace)} 跟随 trace、{pct(follows_other)} 输出第三个 count。"
            "模型总体更依赖 prompt，但冲突会显著扰动答案，因此 trace 既不是唯一读出源，也不是完全被忽略。"
        )
        progress_caption = (
            "横轴是被替换的 block 输出层 0–3；纵轴是 donor-next-index logit margin 相对未 patch receiver 的变化；"
            f"颜色区分 donor progress k−1 与 k+1。所有层 patch 后 donor-next prediction rate 均为 {pct(progress_success)}，"
            f"而不同 prompt、相同 progress control 保持 receiver-next 的比例为 {pct(progress_control_success)}。"
            "这说明 trace marker position 的 residual 携带足以决定下一 index 的 progress state。"
            "由于 learned absolute position 与 progress 同时变化，该实验尚不能把“抽象计数状态”与“绝对位置/局部 token pattern”完全分离。"
        )

        v53_section = f"""
    <h2>6. v5.3 因果机制实验：从 attention 相关性走向 intervention</h2>
    <div class="callout good"><b>因果结论摘要。</b>结果支持三点：第一，non-thinking 的 count 直接依赖 block 1 的 broad-aggregation heads；第二，thinking 的 block 2 retrieval circuit 对正确 marker trace 有因果作用，但至少有四头协同/冗余，mask 最强两个头仍不够破坏行为；第三，最终 count 并不完全从显式 trace 读取，模型保留了强 prompt-direct 旁路。也就是说，当前 CoT 是一个真实运行的检索过程，但不是最终答案唯一的信息通道。</div>

    <h3>6.1 统一设置、样本规模与 head group 选择</h3>
    {table(v53_setting_rows, [('item','实验'),('setting','精确定义')])}
    {table(group_rows,[('role','机制角色'),('top1','top-1'),('top2','top-2'),('top4','top-4'),('selection','选择规则')])}
    <p class="small">注意：所有层/头均为 0-based。<code>L1</code> 是第 2 个 Transformer block。<code>targeted_top4</code> 在本次 checkpoint 中恰好等于整个 L1 的四个 heads；因此 top-4 ablation 不能把“retrieval head 数量效应”和“整层被 mask”完全分开。随机组也允许与候选 heads 重叠，主要用于观察背景波动，不应视为严格 matched control。</p>

    <div class="protocol"><p><b>Sequence。</b>对同一批 base prompts 同时渲染完整 gold THINK_OFF 与 indexed THINK_ON sequence，并在 eager-attention 模式下一次读取全部 4×4 heads。</p><p><b>三个 query。</b>① non-thinking <code>final_count_query</code>：读取 <code>&lt;/Think&gt;</code> 的 attention row，该位置预测 <code>&lt;C_n&gt;</code>；② thinking <code>trace_marker_query</code>：读取每个 <code>&lt;I_k&gt;</code> row，该位置预测 <code>M_k</code>；③ thinking <code>successor_query</code>：读取每个 <code>M_k</code> row，该位置预测 <code>&lt;I_(k+1)&gt;</code> 或末尾 <code>&lt;/Think&gt;</code>。另外也读取 thinking close row 作为 final-count query。</p><p><b>Attention categories。</b>对每个 row 分别求 BOS、switch、prompt needles、prompt noise、Think-open、trace indices、trace markers、last trace marker 的 raw mass；这些互斥位置集合的 mass 都相对于整条 causal context。Targeted score 采用正确第 k 个 prompt needle 的 raw mass。Broad score 定义为 <code>prompt_needles_mass × H_normalized(needle weights)</code>，其中 entropy 只在 n 个 needle positions 内归一化并除以 <code>log(n)</code>；因此它偏好“总 mass 高且覆盖多个 needles”的 heads。</p><p><b>聚合。</b>Attention detail 共 62,400 行；每个 mode×query×layer×head 对对应 rows 直接求均值，得到 64 行 head summary。Trace query 的每个 k 是独立一行，因此高-count 样本贡献更多 query rows；这是 query-weighted 而不是 example-weighted 汇总。候选 top-1/2/4 groups 由该 summary 排序确定。</p></div>

    {figure(v53_figures/'attention_mechanism_comparison.png','Figure 9. 三类候选 attention signature','三幅热图横轴都是 head 0–3，纵轴都是 layer 0–3。左图仅用 non-thinking final-count query：单元格为 broad_aggregation_score = prompt_needles_mass × 针对 needle 子集的归一化 attention entropy；它同时奖励“看向 needles”与“在多个 needles 间广泛分布”。中图用 thinking 的 &lt;I_k&gt; query：单元格为投向正确第 k 个 prompt needle 的 raw attention mass；L1H0/H1/H2 接近 1。右图用 thinking 的 final-count query：单元格为投向全部 trace-marker positions 的 mass，L3H0/H1 与 L2H3 较高。三图只用于定义候选组，因果性由后续 mask/patch 决定。',wide=True)}
    <div class="callout result-line"><b>Signature 结果。</b>Non-thinking 的最高 broad score 是 L0H1={fmt(value(direct_signature, 'broad_aggregation_score'))}。Thinking retrieval 中 L1H0/H1/H2 对正确第 k 个 prompt needle 的 raw mass 分别为 {fmt(value(targeted_signature_h0, 'correct_prompt_needle_mass'))}/{fmt(value(targeted_signature_h1, 'correct_prompt_needle_mass'))}/{fmt(value(targeted_signature_h2, 'correct_prompt_needle_mass'))}。Final-count query 对 trace markers 的最高单头是 L3H1，mass={fmt(value(trace_readout_signature, 'trace_markers_mass'))}。这些数值用于提出候选 circuit，不作为因果结论。</div>

    <h3>6.2 必要性：head mask 是否破坏原行为？</h3>
    <div class="protocol"><p><b>Teacher-forced sequence 与位置。</b>输入完整 gold sequence。Count metric 读取 close token 的 logits；marker metric 读取每个 <code>&lt;I_k&gt;</code> 的 logits；successor metric 读取每个 <code>M_k</code> 的 logits。每个 count 10 个样本，共 100 prompts×2 modes。</p><p><b>干预。</b>使用 Hugging Face GPT-2 的 <code>head_mask[layer,head]=0</code>。这是<b>全序列、全 token positions</b> 的 head ablation：被选 head 在该 forward/generation 的每个 query 位置都被清零，不是只 mask 图中所分析的某一个 token。分别测试候选 top-1/2/4、整层、全部 heads 和随机 groups。</p><p><b>Teacher-forced readout。</b>Count 只在 10 个 count tokens 内取 argmax；marker 只在 10 个 marker tokens 内取 argmax；successor 只在 10 个 index tokens 与 close 中取 argmax。记录 clean−masked 的 accuracy 与 logit margin。</p><p><b>自由生成。</b>另取每 count 3 个 prompts，从 switch+prompt+Think-open prefix 开始 greedy 生成，不给 gold trace；同一 mask 在每个 autoregressive step 都生效。记录 final count、marker-trace exact 和格式错误。</p></div>
    <div class="formula">logit margin(target) = logit(target) − max logit(other valid tokens)</div>
    <div class="formula">drop_margin = clean_margin − masked_margin；drop_accuracy = clean_subset_accuracy − masked_subset_accuracy</div>
    <p>Teacher-forced marker accuracy 只在 10 个 marker tokens 内取 argmax；successor accuracy 只在 <code>&lt;I1&gt;...&lt;I10&gt;</code> 与 <code>&lt;/Think&gt;</code> 内取 argmax；count accuracy 只在 10 个 count tokens 内取 argmax。Free-running 指标则从开关后的 prefix 开始真正 greedy 生成，不给 gold trace。</p>
    {table(ablation_rows,[('test','路径 / mask'),('target','teacher-forced target'),('heads','被 mask heads'),('tf','teacher-forced accuracy drop'),('margin','mean margin drop'),('free','free-running final accuracy'),('trace','free-running marker-trace exact')])}
    <div class="figure-grid">
      {figure(v53_figures/'nonthinking_ablation.png','Figure 10. Non-thinking teacher-forced count-margin drop','横轴是 clean count margin 减去 masked count margin，越大表示该 mask 越破坏正确 count；纵轴为不同候选、整层和随机 head groups。L0H1 单头和 L0 全层效果最大，说明 direct broad 路径集中在第 1 个 block。')}
      {figure(v53_figures/'thinking_retrieval_ablation.png','Figure 11. Thinking marker-retrieval margin drop',retrieval_ablation_caption)}
      {figure(v53_figures/'thinking_readout_ablation.png','Figure 12. Thinking final-count margin drop','横轴是 thinking 条件下 final count margin 的下降；纵轴为 mask group。最大的 effect 来自 direct-broad/L0 groups，而仅按 final query 的 trace-marker attention 选出的 heads 对 final count 影响较小。这提示“高 attention 到 trace”不等于“trace 是最终读出的唯一因果来源”。')}
      {figure(v53_figures/'behavioral_head_ablation.png','Figure 13. Free-running head ablation','左右分别是 THINK_OFF 与 THINK_ON；横轴为 autoregressive final-count accuracy，纵轴为 mask group。右图文本额外给 trace exact。THINK_ON targeted top-4 让 trace exact 从 1.00 降至 0.07，但 final accuracy 仍为 1.00；这是“trace 生成 circuit 被破坏，而 final answer 由旁路保住”的直接证据。',wide=True)}
    </div>
    <div class="callout result-line"><b>必要性结果。</b>Non-thinking 中，L0H1 单头使 teacher-forced count accuracy 平均下降 {pct(value(direct_top1, 'drop_count_accuracy'))}，mask direct top-2 后 free-running final accuracy 只剩 {pct(value(direct_behavior2, 'final_accuracy'))}。Thinking 中，targeted top-2 仍保留 100% marker-trace 与 final count，说明高-attention heads 之间有冗余；只有加入第四个 L1H3、等价于 mask 整个 block-2 attention 层时 marker-trace exact 才降到 {pct(value(targeted_behavior4, 'trace_exact'))}。与此同时 final count 保持 {pct(value(targeted_behavior4, 'final_accuracy'))}。相反，mask 早期 direct-broad top-2 会把 thinking final accuracy 降到 {pct(value(thinking_direct_behavior2, 'final_accuracy'))}。因此 retrieval circuit 对 trace 必要，但 final answer 主要由共享的早期 direct path 保底。</div>

    <h3>6.3 充分性：clean-to-corrupt activation patching</h3>
    <div class="protocol"><p><b>Patched tensor。</b>GPT-2 每层四个 head outputs 在 attention <code>c_proj</code> 前拼成 256 维向量；每头占连续 64 维。Head patch 先缓存 clean run 的 pre-<code>c_proj</code> 输入，只在指定 donor/query token 位置复制候选 head 的 64 维 slice 到 corrupt run，再让 <code>c_proj</code>、residual addition、后续 blocks 与 unembedding 正常运行。Residual patch 则在指定 block 输出后，把该 token 的完整 256 维 residual 替换为 clean donor。</p><p><b>Retrieval corruption。</b>对 count 2–10 每 count 10 个样本，令 k=n，只把 prompt 中最后一个 needle 的 marker identity 从原 token 换成词表中第一个不同 marker；needle 位置、count、sequence 长度、<code>&lt;I_k&gt;</code> 绝对位置完全不变。读取 <code>&lt;I_k&gt;</code> logits，比较 clean marker 与 corrupt marker。错误位置 control 使用相同 targeted heads，但 donor 取 <code>&lt;I_k&gt;</code> 前一个 token。</p><p><b>Count corruption。</b>把 prompt 最后一个 needle 替换为确定性的 noise token，使 n→n−1。THINK_OFF 的 close 绝对位置不变；THINK_ON 因 gold trace 少一对 <code>&lt;I&gt;,M</code> 而缩短 2 tokens，所以 clean donor close 与 corrupt receiver close 位置不同。读取 close logits，比较 <code>&lt;C_n&gt;</code> 与 <code>&lt;C_(n−1)&gt;</code>。</p><p><b>计算。</b>每个 pair 分别算 clean、corrupt、patched target margin，再逐 pair 计算 normalized recovery，最后对 90 pairs 平均。Recovery=1 表示恢复到 clean margin，0 表示没有超过 corrupt，负值表示反向，超过 1 表示过度恢复。</p></div>
    <div class="formula">normalized recovery = (patched_margin − corrupt_margin) / (clean_margin − corrupt_margin)</div>
    <p><b>Retrieval identity corruption：</b>固定 count、needle 位置和 trace 长度，只把最后一个 prompt needle 的 marker identity 换成另一类。clean 与 corrupt 的 &lt;I_k&gt; query 绝对位置相同；把 clean run 指定 heads 在该 query 的 pre-<code>c_proj</code> head slice patch 到 corrupt run，测 clean marker 相对 corrupt marker 的 logit margin。<b>Count-readout corruption：</b>删除 prompt 最后一个 needle，使 count 从 n 变成 n−1；在 <code>&lt;/Think&gt;</code> 的 count-prediction state patch clean head output/residual，测 &lt;C_n&gt; 相对 &lt;C_n−1&gt; 的 margin。</p>
    {table(patch_rows,[('path','因果路径'),('intervention','patched component'),('recovery','normalized recovery'),('control','关键对照')])}
    <div class="figure-grid">
      {figure(v53_figures/'patching_retrieval.png','Figure 14. Retrieval identity patching',retrieval_patch_caption,wide=True)}
      {figure(v53_figures/'patching_nonthinking_readout.png','Figure 15. Non-thinking count-readout patching',nonthinking_patch_caption,wide=True)}
      {figure(v53_figures/'patching_thinking_readout.png','Figure 16. Thinking count-readout patching',thinking_patch_caption,wide=True)}
    </div>
    <div class="callout result-line"><b>充分性结果。</b>Retrieval corruption 把 clean-marker margin 从 {fmt(value(retrieval_patch2, 'clean_margin'))} 翻到 corrupt 的 {fmt(value(retrieval_patch2, 'corrupt_margin'))}；只 patch {code(heads_text('targeted_top2'))} 就恢复 {pct(value(retrieval_patch2, 'normalized_recovery'))}，四个 L1 heads 恢复约 100%，错误 donor 位置只有 {pct(value(retrieval_wrong_pos, 'normalized_recovery'))}。Non-thinking count corruption 中，direct top-2/top-4 分别恢复 {pct(value(direct_patch2, 'normalized_recovery'))}/{pct(value(direct_patch4, 'normalized_recovery'))}，targeted top-4 约为 0。Thinking count readout 中，direct top-4 恢复 {pct(value(think_direct_patch4, 'normalized_recovery'))}，trace-readout top-4 只有 {pct(value(think_trace_patch4, 'normalized_recovery'))}；这与 mask 结果共同表明最终 count 不是由“final query attention 最大的 trace heads”单独决定。</div>

    <h3>6.4 Prompt–trace conflict：最终答案到底跟谁？</h3>
    <div class="protocol"><p><b>Counterfactual sequence。</b>固定真实 prompt needle count 为 p，构造 teacher-forced prefix <code>&lt;BOS&gt; &lt;THINK_ON&gt; prompt &lt;Think/&gt; &lt;I1&gt; M1 ... &lt;I_t&gt; M_t &lt;/Think&gt;</code>，其中 forced trace count t 穷举 1–10。若 t&gt;p，marker identity 按原 prompt marker 列表循环使用；因此只强制改变 trace 长度/进度，不引入词表外 marker。</p><p><b>读取位置。</b>只做一次 forward，读取最后一个 <code>&lt;/Think&gt;</code> hidden state 对应的 logits；在 10 个 count tokens 子集中取 argmax 得到 predicted count。没有继续自由生成。</p><p><b>样本与计算。</b>每个 p 有 10 个 base prompts，每个 prompt 配 10 个 t，共 1,000 rows。热图每格对 10 个样本的 predicted count 取均值。另在 900 个 p≠t rows 上分别统计 prediction=p、prediction=t 或其他 count 的比例。</p><p><b>结果。</b>无冲突对角线均正确；冲突时 {pct(follows_prompt)} 严格跟随 prompt、{pct(follows_trace)} 严格跟随 forced trace、{pct(follows_other)} 输出第三个 count。说明 final readout 以 prompt 为主，但被人工 trace 明显调制。</p></div>
    {figure(v53_figures/'trace_conflict_pred_count.png','Figure 17. Prompt count 与 forced trace count 冲突时的预测',conflict_caption,wide=True)}

    <h3>6.5 Trace progress state transplant</h3>
    <div class="protocol"><p><b>Sequence 与 receiver。</b>仅保留 count≥4 的 gold THINK_ON sequences。固定 receiver 为 trace 中第 k=2 个 marker token <code>M2</code>；正常情况下该 token 的 hidden state 应预测下一个 index <code>&lt;I3&gt;</code>。</p><p><b>Donor。</b>同一 sequence 内选择 k=1 的 <code>M1</code> 或 k=3 的 <code>M3</code> 作为 donor。对每个 block layer=0–3，提取 <code>hidden_states[layer+1]</code> 中 donor token 的完整 256 维 block-output residual，并在该层 forward hook 中替换 receiver <code>M2</code> residual。这样 k=1 donor 希望把下一 token 推向 <code>&lt;I2&gt;</code>，k=3 donor 推向 <code>&lt;I4&gt;</code>。</p><p><b>Readout。</b>在 receiver token 的 logits 上，计算 donor-next token 相对 receiver-next token 的 pairwise margin；同时在 10 个 index tokens+close 的候选集合内判断 argmax。<code>margin_shift_toward_donor = patched pairwise margin − unpatched pairwise margin</code>。</p><p><b>Control。</b>从另一条 prompt 的相同 k=2 marker position提取 residual，patch 到 receiver 的同一 absolute position；该 donor 的 marker identity/context 不同，但 progress 相同。观察是否仍预测 receiver-next <code>&lt;I3&gt;</code>。</p><p><b>结果与限制。</b>所有四层的 k−1/k+1 donor patch 都使 donor-next argmax rate 达到 {pct(progress_success)}，same-progress control 保持 receiver-next 也为 {pct(progress_control_success)}。但 donor progress 与绝对 token position相差两个位置，且 residual 已包含 learned absolute-position 信息；因此结果证明“该 residual 足以携带下一步状态”，尚不能证明这是与位置无关的抽象 counter。</p></div>
    {figure(v53_figures/'progress_state_transplant.png','Figure 18. Progress-state residual transplant',progress_caption,wide=True)}
    """

        successor_heads = v53_attention[
            (v53_attention["mode"] == "thinking")
            & (v53_attention["query_kind"] == "successor_query")
        ].sort_values("next_prompt_needle_mass", ascending=False)
        best_successor = successor_heads.iloc[0]
        core_questions_section = f"""
    <h2>6.6 两个核心算法问题：successor transition 与 count arithmetic</h2>
    <div class="callout warn"><b>先区分两个不同问题。</b><code>&lt;I_k&gt; → M_k</code> 回答“第 k 个 needle 是什么”，而 <code>M_k → &lt;I_(k+1)&gt;</code> 回答“检索完第 k 个以后，如何进入第 k+1 步”。最终 <code>&lt;/Think&gt; → &lt;C_n&gt;</code> 又是第三个 readout 问题。把三种 query 混在一起，会把 targeted retrieval、状态递增和最终计数误写成同一个机制。</div>

    <h3>问题 1：输入/输出第 k 个 needle 后，模型如何进入第 k+1 步？</h3>
    <div class="protocol"><p><b>Sequence 与 token 位置。</b>在完整 teacher-forced THINK_ON sequence <code>... &lt;I_k&gt; M_k &lt;I_(k+1)&gt; M_(k+1) ...</code> 中，<code>&lt;I_k&gt;</code> 位置的 attention/hidden state 预测 <code>M_k</code>；紧随其后的 <code>M_k</code> 位置才预测 <code>&lt;I_(k+1)&gt;</code>。因此 successor 实验读取的是每个 <code>M_k</code> 的 attention row 与 logits，而不是 <code>&lt;I_k&gt;</code> row。</p><p><b>Attention 计算。</b>对每个 <code>M_k</code> query，将 attention 分别累加到下一 prompt needle、全部 prompt needles、prompt noise、既有 trace markers 与 trace indices。当前最高的 next-needle head 是 L{int(best_successor['layer'])}H{int(best_successor['head'])}：投向第 k+1 个 prompt needle 的平均 raw mass 为 {fmt(float(best_successor['next_prompt_needle_mass']))}，投向全部 prompt needles 为 {fmt(float(best_successor['prompt_needles_mass']))}，投向已有 trace markers 为 {fmt(float(best_successor['trace_markers_mass']))}。这说明 successor query 中确实存在朝下一个 needle 的 attention，但 attention 本身还不是因果证据。</p><p><b>Residual transplant。</b>receiver 固定为 <code>M2</code>，其正常下一个 token 是 <code>&lt;I3&gt;</code>。从同一 sequence 的 <code>M1</code> 或 <code>M3</code> 取 <code>hidden_states[layer+1]</code> 的完整 256 维 residual，替换 receiver 在同一层的 residual。四层中 donor 都使输出 100% 改成 donor 对应的 <code>&lt;I2&gt;</code> 或 <code>&lt;I4&gt;</code>；不同 prompt 的同 progress <code>M2</code> donor 则 100% 保持 <code>&lt;I3&gt;</code>。所以 marker-position residual 足以决定下一 index。</p><p><b>目前不能区分什么。</b><code>M1/M2/M3</code> 位于不同 absolute positions，而本模型使用 learned absolute position embeddings；相邻 progress 也恰好相差两个 token。因此 donor residual 同时携带“k 的语义状态”“绝对位置”和局部 <code>&lt;I_k&gt;,M_k</code> pattern。当前实验不能断言模型实现了抽象的 <code>k+1</code> 运算。</p></div>
    <div class="formula">successor target at query M_k = I_(k+1)；terminal target at M_n = &lt;/Think&gt;</div>
    <div class="callout good"><b>现有最稳妥结论。</b>模型内部存在一个可移植的 progress state，它在 marker token 的 residual 中足以控制下一 index；但这个 state 可能是 learned absolute-position lookup，也可能是由前一个 index token 驱动的有限状态机，还没有被证明为位置无关的 successor counter。</div>

    <h3>问题 2：count 数值如何做加减法，存在哪里？</h3>
    <div class="protocol"><p><b>输出表示。</b>模型不输出十进制字符并执行显式算术，而是在十个独立类别 token <code>&lt;C1&gt;...&lt;C10&gt;</code> 中选择一个。最终 readout 读取 <code>&lt;/Think&gt;</code> 位置的 hidden state，经共享 unembedding 得到十个 count logits。因此“准确输出 n+1”既可能来自连续 count direction，也可能只是十类 lookup。</p><p><b>Probe 已经说明什么。</b>在固定 absolute position 的 THINK_OFF close token，提取 hidden-state index 0–4 的 256 维 residual；用训练集标准化后做 nearest-centroid 分类与 ridge count regression。第一个 block 后 count 已可被线性/质心 probe 高精度读出，而 position baseline 很低。这说明 count 信息存在于 residual population code 中，不等于存在某一个 neuron 或某一个专用标量槽。</p><p><b>Causal evidence。</b>把 prompt 的一个 needle 替换为 noise，使 count 从 n 变成 n−1，但 sequence 长度不变。将 clean n run 的 direct-broad head outputs patch 到 corrupt n−1 run 的固定 THINK_OFF close query，可恢复大部分 <code>C_n</code> 相对 <code>C_(n−1)</code> 的 logit margin；全 residual patch 近乎完全恢复。这定位了“一个 needle 的差分信息”流经早期 broad-aggregation heads 并写入 residual，但仍没有证明该 residual 采用线性加法编码。</p><p><b>CoT 条件的额外混淆。</b>THINK_ON close 的 absolute position 是 <code>259+2n</code>，所以仅凭 close-state probe 可从位置直接读出 n；其 100% probe 不能用来证明模型在 trace 中累加了一个抽象 counter。要研究 arithmetic，应优先在固定位置的 <code>&lt;Think/&gt;</code>、THINK_OFF close，或经过位置匹配的 counterfactual anchors 上进行。</p></div>
    <div class="callout warn"><b>当前没有证据证明“做了加减法”。</b>我们已经知道 count 可从 hidden state 读出，也知道相关 heads/residual 对答案有因果作用；但还没有证明相邻 count states 满足 <code>h_(n+1) ≈ h_n + d</code>，也没有证明对 hidden state 加/减同一方向会稳定令答案变成 n±1。</div>

    <h3>能够真正回答这两个问题的下一组实验</h3>
    <ol>
      <li><b>Same-position successor counterfactual。</b>保持 prompt、sequence 长度与 <code>M_k</code> absolute position 不变，只把前一 token 从 <code>&lt;I_k&gt;</code> 改为 <code>&lt;I_j&gt;</code>，分别读取每层 <code>M_k</code> residual 和 index-subset logits。若输出变为 <code>&lt;I_(j+1)&gt;</code>，支持 token-driven successor；若仍为 <code>&lt;I_(k+1)&gt;</code>，更支持 position/context-driven transition。</li>
      <li><b>Position-controlled residual patch。</b>构造两条等长 counterfactual prefix，使 donor 与 receiver query 位于同一 absolute position、但其显式 progress token 不同；逐层 patch 256 维 residual，并加入“同 progress、不同 marker”和“同 marker、不同 progress”两个 controls。这样才能把 progress 从 position 与 marker identity 中分离。</li>
      <li><b>Adjacent-difference geometry。</b>在固定 anchor 和每个 layer 计算各 count 的均值状态 <code>mu_n</code>，令 <code>d_n=mu_(n+1)-mu_n</code>；报告所有 <code>cos(d_n,d_m)</code>、方向长度及 held-out split。只有相邻差分近似平行，才有资格讨论统一 count direction。</li>
      <li><b>Arithmetic steering。</b>只用训练样本估计 <code>d=mean_n d_n</code>，在 held-out 样本同一 layer/position 注入 <code>h'=h+alpha*d</code>，继续运行剩余 blocks；测试 alpha=−2...2 时 count-subset 期望值和 exact prediction 是否单调移动，并重点检验 <code>n→n±1</code>。随机正交方向、shuffled-label direction 和等范数 direction 是必要 controls。</li>
      <li><b>Needle add/delete trajectory。</b>在不改变 sequence 长度的前提下将一个 noise token 与 needle token互换，形成 n 与 n±1 配对；逐层比较固定 query residual 差异并做 clean-to-corrupt head/residual patch，定位“+1 信息”第一次出现在哪层、由哪些 heads 写入，以及后续层是否只负责分类 readout。</li>
    </ol>
    """

    styles = """
    :root{--ink:#172033;--muted:#5b6678;--line:#dce3ee;--soft:#f6f8fb;--blue:#2563eb;--green:#15803d;--amber:#a16207}
    *{box-sizing:border-box}body{margin:0;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","Microsoft YaHei",Arial,sans-serif;line-height:1.7;background:#fff}
    main{max-width:1240px;margin:auto;padding:38px 28px 80px}h1{font-size:34px;margin:0}h2{font-size:25px;margin:46px 0 18px;padding-top:20px;border-top:1px solid var(--line)}h3{font-size:17px;margin:0 0 10px}p,li{font-size:15.5px}.subtitle{color:var(--muted);margin:5px 0 24px}
    code{background:#edf1f7;border-radius:4px;padding:1px 5px;font-family:Consolas,monospace}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:14px}.label{color:var(--muted);font-size:12px}.value{font-size:22px;font-weight:750}.callout{border-left:5px solid var(--blue);background:#eef4ff;padding:14px 18px;border-radius:7px;margin:20px 0}.good{border-left-color:var(--green);background:#edf9f0}.warn{border-left-color:var(--amber);background:#fff7e6}
    .format-grid,.figure-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.format,.figure{border:1px solid var(--line);border-radius:8px;padding:15px;background:#fff}.sequence{font:14px Consolas,monospace;overflow-wrap:anywhere}.figure{margin:0 0 18px}.figure.wide{grid-column:1/-1}.figure img{display:block;width:100%;height:390px;object-fit:contain}.figure.wide img{height:500px}figcaption{font-size:13.5px;color:var(--muted);margin-top:10px}.on{color:#08783e;font-weight:700}.off{color:#b45309;font-weight:700}.target{color:#b42318;font-weight:700}
    .table-wrap{overflow-x:auto;margin:14px 0 24px}table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}th{background:#eef2f8}.formula{font:13.5px Consolas,monospace;background:#f7f8fb;border:1px solid var(--line);border-radius:7px;padding:12px 14px;margin:10px 0}.diagram{border:1px solid var(--line);border-radius:8px;padding:18px;background:#fbfcfe}.arrow{color:var(--blue);font-weight:800}.small{font-size:13px;color:var(--muted)}.protocol{border:1px solid var(--line);border-left:4px solid #64748b;border-radius:8px;padding:14px 16px;margin:14px 0 20px;background:#fbfcfe}.protocol p{margin:5px 0}.protocol b{display:inline-block;min-width:112px;color:#334155}.result-line{border-left-color:var(--green);background:#f3faf5}
    @media(max-width:900px){main{padding:24px 14px}.cards,.format-grid,.figure-grid{grid-template-columns:1fr}.figure.wide{grid-column:auto}.figure img,.figure.wide img{height:auto}}
    """

    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v5 Explicit Switch Report</title><style>{styles}</style></head><body><main>
    <h1>v5 显式 Thinking 开关：结果与机制诊断</h1>
    <div class="subtitle">单模型混合训练；修正版 indexed trace；结果目录：{code(run_dir)}</div>
    <h2 id="glossary">0. 术语、位置约定与指标定义</h2>
    <div class="callout warn"><b>最重要的区别。</b><code>teacher-forced accuracy</code> 不是自由生成准确率。Teacher forcing 会把 gold sequence 中此前所有正确 token 都提供给模型，再检查某个位置对“下一个 token”的预测；autoregressive/free-running evaluation 则只给 prompt prefix，后续 token 全由模型自己逐步生成，前一步错误会进入下一步 context。因此 teacher-forced accuracy 可以很高，而自由生成仍可能因误差累积失败。</div>
    <div class="protocol">
      <p><b>Causal-LM 位置。</b>本报告说“读取 token t 的 hidden state、attention row 或 logits”，都指该位置的表示用于预测紧随其后的 token t+1。例如 <code>&lt;I_k&gt;</code> 位置预测 <code>M_k</code>，<code>M_k</code> 位置预测 <code>&lt;I_(k+1)&gt;</code>，<code>&lt;/Think&gt;</code> 位置预测 <code>&lt;C_n&gt;</code>。</p>
      <p><b>Teacher-forced forward。</b>把包含 gold trace、gold close 和 gold answer 的完整正确序列一次性输入模型。每个 query 都能看到它之前的真实 token，而不是模型自己生成的 token。本报告的 teacher-forced count/marker/successor accuracy 分别只在预先规定的合法候选子集中取 argmax；它们是局部组件诊断，不是端到端生成成功率。</p>
      <p><b>Candidate-subset accuracy。</b>Count accuracy 只比较十个 <code>&lt;C1&gt;...&lt;C10&gt;</code> logits；marker accuracy 只比较十个 marker-token logits；successor accuracy 只比较十个 <code>&lt;I1&gt;...&lt;I10&gt;</code> 与 <code>&lt;/Think&gt;</code> logits。若正确 token 在该子集内的 logit 最大，该 query 记为 1，否则为 0，再对 queries/examples 求均值。它比“全 vocabulary argmax 正确”更宽松。</p>
      <p><b>Autoregressive / free-running。</b>只输入 <code>&lt;BOS&gt; + switch + prompt + &lt;Think/&gt;</code>，之后每一步把模型的 greedy argmax 追加回输入，直到 <code>&lt;EOS&gt;</code> 或达到长度上限。<code>final-count exact accuracy</code> 是最终解析出的 count 是否严格等于 gold count；这是本报告最接近真实使用方式的端到端准确率。</p>
      <p><b>Trace exact。</b>从自由生成文本中抽取 marker-token 序列，并与 prompt 中按从左到右排列的 gold marker 序列逐项比较；完全相同记为 1。当前实现不逐项检查每个 <code>&lt;I_k&gt;</code> 是否正确，因此它准确地说是 <i>marker-trace exact</i>。</p>
      <p><b>Hidden state / residual。</b>这里的 residual 不是预测误差，而是 <i>residual stream</i>：每个 token 在模型内部携带的一条 256 维“工作向量”。<code>hidden-state index 0</code> 是 token embedding 与 learned absolute-position embedding 的和。每个 GPT-2 block 先让 attention 从先前 tokens 读信息并把增量加回该向量，再让 MLP 做逐 token 变换并再次加回；所以同一个 token position 的向量会沿层数不断更新。index 1/2/3 分别是 block 0/1/2 后的 residual；index 4 是 block 3 后再经过 GPT-2 最终 LayerNorm <code>ln_f</code> 的 hidden state。说“count 可从 hidden state 读出”只表示 probe 能利用该向量预测 count，不自动意味着模型在生成时因果使用同一方向。</p>
      <div class="formula">u_l = x_l + Attention_l(LN₁(x_l))；x_(l+1) = u_l + MLP_l(LN₂(u_l))</div>
      <p><b>Attention row 与 mass。</b>某 query token 的 attention row 是一个对所有先前可见 token positions 的概率分布，和为 1。某类位置的 <code>attention mass</code> 是这些位置权重之和。例如 <code>prompt_needles_mass</code> 是投向所有 prompt needle positions 的权重总和；raw mass 的分母始终是全部可见 context。</p>
      <p><b>Correct top-1 与 diagonal dominance。</b>对 <code>&lt;I_k&gt;</code> query，仅在该样本的 n 个 prompt needle positions 中寻找 attention 最大值；最大值位于第 k 个 needle 时 <code>correct_top1=1</code>。<code>diagonal_dominance=A(I_k,needle_k)/sum_j A(I_k,needle_j)</code>，只描述 needle 子集内部的集中程度，必须与 raw needle mass 一起看。</p>
      <p><b>Logit margin。</b><code>margin = logit(correct target) − max logit(other legal candidates)</code>。正值表示正确 target 在候选子集中领先，负值表示已有其他候选超过它。Margin 比 0/1 accuracy 能显示干预在尚未翻转 argmax 时造成的连续变化。</p>
      <p><b>Head mask / ablation。</b>将指定 attention head 的输出乘以 0，再测性能下降。本报告使用的 Hugging Face <code>head_mask</code> 在整个 sequence 的所有 query positions、以及自由生成的每一步都生效，因此是全局必要性实验，不是只删除图中某一个 token 上的 attention。</p>
      <p><b>Activation patching。</b>先在 clean input 上缓存指定 token/层的 head output 或 residual，再把它替换到 corrupt input 的指定位置，观察正确 logit 是否恢复。它测试该 activation 对恢复行为是否具有局部充分性。<code>normalized recovery=(patched−corrupt)/(clean−corrupt)</code>；1 表示恢复到 clean，0 表示没有改善，负值表示进一步恶化。</p>
      <p><b>Probe。</b>冻结模型，只把某 token/层的 hidden vector 当作特征训练简单的 nearest-centroid 或 ridge predictor。Probe accuracy/R² 衡量信息是否可线性读取，不等于该信息是模型实际使用的因果变量；position baseline 与 shuffled/control probe 用于检查泄漏。</p>
      <p><b>编号。</b>全文 layer/head 均采用代码中的 0-based 编号：L0 是第一个 Transformer block，H0 是该层第一个 head。</p>
    </div>
    <div class="cards">
      <div class="card"><div class="label">THINK_OFF final accuracy</div><div class="value">{pct(summary['nonthinking']['final'])}</div></div>
      <div class="card"><div class="label">THINK_ON final accuracy</div><div class="value">{pct(summary['thinking']['final'])}</div></div>
      <div class="card"><div class="label">THINK_ON marker-trace exact</div><div class="value">{pct(summary['thinking']['trace'])}</div></div>
      <div class="card"><div class="label">Perfect indexed-retrieval heads</div><div class="value">{len(perfect)}</div></div>
    </div>
    <div class="callout good"><b>核心结论。</b>同一个 Transformer 已经可靠学会显式模式路由：<code>THINK_OFF</code> 立即关闭 trace，<code>THINK_ON</code> 以 indexed 格式生成正确 marker 序列，再输出 count；两条路径在当前 ID 测试上均为 100%。v5.3 的 mask 与 activation patching 进一步表明：block 2 的 k-to-k retrieval heads 确实因果控制 marker trace，但最终 count 仍保留一条强 prompt-direct aggregation 旁路。因此，显式 CoT 在本模型中是真实执行的检索过程，却不是最终答案唯一的信息来源。</div>

    <h2>1. 实验设定</h2>
    {table([
      {'item':'模型','setting':f"随机初始化 GPT-2；4 blocks × 4 heads；d_model={model['n_embd']}；MLP={model['n_inner']}；learned absolute position embedding；context={model['n_positions']}；dropout=0"},
      {'item':'词表','setting':f"vocab={vocab_size}；包含 64 noise、10 marker、10 count、10 trace-index token 及 special/switch tokens"},
      {'item':'数据','setting':f"prompt body 长度 {train['seq_len']}；count 均匀取 {train['count_min']}–{train['count_max']}；needle 位置无放回均匀采样；marker identity 从 10 类中有放回采样"},
      {'item':'单模型混合训练','setting':f"thinking_fraction={train['thinking_fraction']}；每个 batch 中 THINK_ON/OFF 样本随机混合；不是两个独立模型"},
      {'item':'优化','setting':f"{train['train_steps']} steps；batch={train['batch_size']}；AdamW lr={train['lr']}；warmup={train['warmup_steps']}；weight decay={train['weight_decay']}；seed={train['seed']}"},
      {'item':'评估','setting':f"每个 mode × count 使用 {train['eval_examples_per_count']} 个样本；greedy autoregressive generation；最终每个 mode 共 10,000 个 ID 样本"},
      {'item':'attention 诊断','setting':f"每个 count {train['attention_examples_per_count']} 个样本；总计 {query_count:,} 个 trace-index queries × 16 heads；所有 layer/head 编号均从 0 开始"},
    ], [('item','项目'),('setting','具体设置')])}
    <div class="format-grid">
      <div class="format"><b>THINK_ON</b><div class="sequence">&lt;BOS&gt; <span class="on">&lt;THINK_ON&gt;</span> prompt &lt;Think/&gt; <span class="target">{html.escape(trace_format)} &lt;/Think&gt; &lt;Cn&gt; &lt;EOS&gt;</span></div><p>所有 trace index、trace marker、关闭符、最终 count 与 EOS 都进入 next-token cross-entropy。</p></div>
      <div class="format"><b>THINK_OFF</b><div class="sequence">&lt;BOS&gt; <span class="off">&lt;THINK_OFF&gt;</span> prompt &lt;Think/&gt; <span class="target">&lt;/Think&gt; &lt;Cn&gt; &lt;EOS&gt;</span></div><p>模型被监督立即关闭 thinking block，然后输出 count；没有伪造空 trace token。</p></div>
    </div>
    <div class="callout warn"><b>任务边界。</b>这是固定长度 256、count 1–10 的 ID 实验。最终饱和不能证明 length/count OOD，也不能证明 thinking 比 non-thinking 更准确；本实验主要检验显式开关是否可学、两条路径是否共存，以及 indexed trace 是否形成可解释 retrieval。</div>

    <h2>2. 指标定义与 retrieval query</h2>
    <div class="formula">final_accuracy = mean[1(predicted count token == gold &lt;C_n&gt;)]</div>
    <div class="formula">marker_trace_exact = mean[1(extract_markers(generated trace) == [M1,...,Mn])]</div>
    <div class="formula">correct_top1 = mean[1(argmax over prompt-needle positions at query &lt;I_k&gt; is needle k)]</div>
    <div class="formula">diagonal_dominance = mean[A(&lt;I_k&gt;, needle_k) / sum_j A(&lt;I_k&gt;, needle_j)]</div>
    <div class="formula">needle_mass = mean[sum_j A(&lt;I_k&gt;, needle_j)]</div>
    <p><b>为什么 query 是 <code>&lt;I_k&gt;</code>？</b>在 causal LM 中，位置 <code>&lt;I_k&gt;</code> 的 hidden state/attention 用来预测下一个 token <code>M_k</code>。因此 k-to-k retrieval 应读取该 attention row，并在 prompt 中按位置从左到右排列的 needles 之间判断第 k 个 needle 是否获得最大 attention。读取 <code>M_k</code> 自身的 row 会变成“生成 marker 后模型看哪里”，不是生成该 marker 的 retrieval 证据。随机在当前样本的 needles 中选一个的加权 top-1 基线约为 {fmt(random_top1)}。</p>

    <h2>3. 行为结果与训练过程</h2>
    <h3>3.1 实验 A：混合模式 next-token 训练</h3>
    <div class="protocol"><p><b>Sequence。</b>THINK_ON 使用完整序列 <code>&lt;BOS&gt; &lt;THINK_ON&gt; prompt[256] &lt;Think/&gt; &lt;I1&gt; M1 ... &lt;In&gt; Mn &lt;/Think&gt; &lt;C_n&gt; &lt;EOS&gt;</code>；THINK_OFF 使用 <code>&lt;BOS&gt; &lt;THINK_OFF&gt; prompt[256] &lt;Think/&gt; &lt;/Think&gt; &lt;C_n&gt; &lt;EOS&gt;</code>。每个 batch 的 128 个样本独立以 0.5 概率选择一种格式。</p><p><b>监督位置。</b>THINK_ON 只监督全部 trace index/marker、<code>&lt;/Think&gt;</code>、<code>&lt;C_n&gt;</code>、<code>&lt;EOS&gt;</code>；THINK_OFF 只监督 <code>&lt;/Think&gt;</code>、<code>&lt;C_n&gt;</code>、<code>&lt;EOS&gt;</code>。BOS、switch token、256 个 prompt tokens 和 <code>&lt;Think/&gt;</code> 的 label 均为 −100，不直接计算 loss，但作为 causal context 参与所有后续预测。</p><p><b>计算。</b>将 logits 左移一位与 labels 对齐；对所有非 −100 target positions 计算 cross-entropy，再按有效 token 数求平均。因此 THINK_ON 每个样本有 <code>2n+3</code> 个有效 targets，THINK_OFF 只有 3 个，主 loss 是 token-weighted 而不是 mode-balanced。图中的 component loss 是从同一个 CE tensor 重新选取相应 target positions 后分别求平均。</p><p><b>结果。</b>10,000 steps 后全部 component loss 均接近 0；模型能够同时拟合两种格式。这个结果只说明训练集分布上的序列规则可学，不单独证明两种推理机制不同。</p></div>
    {table(loss_rows,[('component','记录字段'),('tokens','纳入该分项的 target tokens'),('first','首次记录 loss'),('final','最终记录 loss')])}
    {figure(source_figures/'train_loss_by_step_and_mode.png','Figure 1. 训练 loss 分解','横轴为 optimizer step；每 50 steps 记录一次，另含 step 1。纵轴为该批次指定 target positions 上的平均 next-token cross-entropy。蓝/橙等曲线不是额外加权项，而是同一次训练 CE 的诊断性切片；不同分项包含的 token 数不同，不能用曲线高度直接推断其对总梯度的贡献相同。',wide=True)}

    <h3>3.2 实验 B：各 checkpoint 的自由生成行为</h3>
    <div class="protocol"><p><b>输入 prefix。</b>对同一批 balanced base prompts 分别提供 <code>&lt;BOS&gt; &lt;THINK_OFF&gt; prompt &lt;Think/&gt;</code> 或 <code>&lt;BOS&gt; &lt;THINK_ON&gt; prompt &lt;Think/&gt;</code>，不提供 gold trace、close 或 count。每个 count 1–10 各 1,000 个样本；同一 base prompt 同时用于两种模式。</p><p><b>生成。</b>使用 greedy argmax。THINK_OFF 最多生成 14 tokens，THINK_ON 最多生成 26 tokens；模型遇到 <code>&lt;/Think&gt;</code> 后继续生成 count 和后续 token，遇到 EOS 停止。</p><p><b>解析与计算。</b><code>final_accuracy</code> 检查 close 后第一个可解析 count token 是否等于 <code>&lt;C_n&gt;</code>。<code>trace_exact</code> 比较从生成 trace 中抽取出的 marker token 序列是否严格等于 gold marker 序列；它<b>不检查 trace index tokens 是否逐个正确</b>。Precision/recall 使用生成 marker 序列与 gold marker 序列的最长公共子序列长度。每个 checkpoint 先按 mode×count 聚合，再在图中对 count 等权平均。</p><p><b>结果。</b>最终 checkpoint 两种模式的 final accuracy 都是 100%；THINK_ON 的 marker-trace exact/precision/recall 也是 100%。首次同时达到这些条件的 checkpoint：THINK_OFF step {first_perfect_step(eval_df,'nonthinking')}，THINK_ON step {first_perfect_step(eval_df,'thinking')}。由于两种模式都饱和，该实验不能提供 thinking accuracy 优势。</p></div>
    {table(behavior_rows,[('mode','请求模式'),('n','最终评估样本'),('acc','final accuracy'),('trace','marker-trace exact'),('switch','首 token 路由准确率'),('p','P(desired token)')])}
    <div class="figure-grid">
      {figure(source_figures/'final_accuracy_by_step_mode.png','Figure 2. Final-count accuracy 随训练变化','横轴为保存的 checkpoint step（评估间隔 500）；纵轴为 count 1–10 上等权平均的 autoregressive final-count exact accuracy；颜色区分请求模式。中期的非单调波动说明单个早期 checkpoint 不能代表稳定性能。')}
      {figure(source_figures/'final_accuracy_by_count_mode.png','Figure 3. 最终 accuracy 按 gold count 分解','横轴为 gold needle count 1–10；纵轴为最终 checkpoint 上对应 count 的 autoregressive exact-count accuracy；颜色区分 THINK_OFF/THINK_ON。20 个 mode×count 条件均为 1.0。')}
      {figure(source_figures/'trace_metrics_by_count.png','Figure 4. THINK_ON marker-trace 质量','横轴为 gold count；纵轴为 marker-sequence 指标或格式错误率。Trace exact 只检查抽取后的 marker 序列；precision/recall 基于 marker LCS；premature close/missing close 检查关闭位置。所有 count 的 marker trace 均正确，但该图本身不验证每个 &lt;I_k&gt; index token。')}
    </div>

    <h3>3.3 实验 C：显式 switch 的第一步路由</h3>
    <div class="protocol"><p><b>Sequence 与 query。</b>仍使用两种 prefix，但只做一次 batched forward；读取最后一个输入 token <code>&lt;Think/&gt;</code> 位置的 logits，因为该 hidden state 预测 thinking block 的第一个 token。</p><p><b>目标。</b>THINK_ON 的 desired next token 是 <code>&lt;I1&gt;</code>；THINK_OFF 的 desired next token 是 <code>&lt;/Think&gt;</code>。对完整 100-token vocabulary 做 softmax，记录 desired token 概率与全词表 argmax 是否等于 desired token。</p><p><b>结果。</b>最终两种模式的 argmax routing accuracy 都为 {pct(summary['thinking']['switch'])}/{pct(summary['nonthinking']['switch'])}，平均 desired-token probability 分别为 {fmt(summary['thinking']['p_desired'],6)}/{fmt(summary['nonthinking']['p_desired'],6)}。这证明 switch token 能控制 trace 是否开启，但尚未说明 trace 内部如何完成检索。</p></div>
    {figure(report_figures/'switch_probability.png','Figure 5. 显式开关的路由置信度','横轴为 checkpoint step；纵轴为完整 prompt 后 &lt;Think/&gt; query 对 desired first token 的平均 softmax probability。THINK_OFF 的目标是 &lt;/Think&gt;，THINK_ON 的目标是 &lt;I1&gt;；颜色区分请求模式。该指标只测第一步路由，不等同于整段 trace 或最终 count 正确。',wide=True)}

    <h2>4. Indexed trace attention：三个近乎完美的 retrieval heads</h2>
    <div class="protocol"><p><b>Sequence。</b>加载最终 checkpoint，对完整 gold THINK_ON sequence 做 teacher-forced forward：<code>&lt;BOS&gt; &lt;THINK_ON&gt; prompt[256] &lt;Think/&gt; &lt;I1&gt; M1 ... &lt;In&gt; Mn &lt;/Think&gt; &lt;C_n&gt; &lt;EOS&gt;</code>。每个 count 1–10 各 100 个独立样本，共 1,000 个 prompts。</p><p><b>Attention row。</b>对每个 k，读取每层每头在 <code>&lt;I_k&gt;</code> token 位置的 causal attention row。这个位置的 residual/logits 用来预测紧随其后的 <code>M_k</code>；读取 <code>M_k</code> 自己的 row 会回答“生成 M_k 后看哪里”，不是“如何预测 M_k”。Prompt needle positions 由原始 256-token prompt 中所有 marker 位置按从左到右排序得到。</p><p><b>计算。</b><code>correct_top1</code>：只在该样本的 n 个 prompt needle positions 中取 attention argmax，检查是否为第 k 个；<code>diagonal_dominance = A(I_k,needle_k)/Σ_j A(I_k,needle_j)</code>；<code>needle_mass = Σ_j A(I_k,needle_j)</code> 是相对全部可见 context 的 raw mass。先在同一 example 的所有 k queries 上平均，再在 1,000 个 examples 上平均。Count 均匀时，随机在 n 个 needles 中选择的期望 top-1 baseline 为 <code>mean_n(1/n)={fmt(random_top1)}</code>。</p><p><b>结果。</b>L1H0、L1H1、L1H2（0-based；第 2 个 block）均达到约 1.0 correct top-1 和约 0.99–1.00 raw correct-needle mass。这意味着这些 heads 不是只在 needle 子集内部排序正确，而是把几乎全部 attention 集中到正确第 k 个 needle。第 6 节再检验它们是否具有因果作用。</p></div>
    <div class="callout good"><b>最强结果。</b>{code(f"L{int(best['layer'])}H{int(best['head'])}")}（第 {int(best['layer'])+1} 个 block）在 <code>&lt;I_k&gt; → M_k</code> query 上的 correct top-1={fmt(best.correct_top1)}、diagonal dominance={fmt(best.diagonal_dominance)}、all-needle mass={fmt(best.needle_mass)}。共有 {len(perfect)} 个 heads 同时满足 top-1≥0.999 且 needle mass≥0.99：{', '.join(f'L{int(r["layer"])}H{int(r["head"])}' for _,r in perfect.iterrows())}。</div>
    {figure(report_figures/'indexed_retrieval_metrics.png','Figure 6. 16 个 heads 的 indexed retrieval 三种指标','三幅热图的横轴均为 head 0–3，纵轴均为 layer 0–3（均为代码中的 0-based 编号；layer 1 即第 2 个 Transformer block）。左：第 k 个 prompt needle 是否在 needle 子集中获得 top-1；中：投向所有 prompt needles 的质量中，正确第 k 个所占比例；右：投向全部 prompt needles 的原始 attention mass。block 2 的 H0/H1/H2 在三项上都接近 1，说明它们不是只在 needle 子集中勉强排序，而是几乎把全部 attention 放到正确 needle。',wide=True)}
    {table(head_rows,[('head','head'),('top1','correct top-1'),('diag','diagonal dominance'),('mass','all-needle mass'),('entropy','attention entropy')])}
    <div class="diagram"><b>由 v5.3 修正后的机制图景</b><p><b>THINK_ON trace：</b><code>&lt;I_k&gt;</code> <span class="arrow">→</span> block-2 多头 retrieval circuit 定位 prompt 第 k 个 needle <span class="arrow">→</span> marker identity 写入 residual <span class="arrow">→</span> 输出 <code>M_k</code>。<br><b>最终 count：</b>prompt needles <span class="arrow">→</span> block-1 broad aggregation <span class="arrow">→</span> count state；显式 trace 会扰动并可能辅助该状态，但 conflict 与 ablation 显示它不是唯一读出路径。</p></div>
    <p><b>证据强度。</b>Figure 6 本身仍只是相关性；attention weight 高不自动等于该 head output 对 logit 必要。对应的 global head-mask 必要性和 clean-to-corrupt activation-patching 充分性检验在第 6.2–6.3 节给出。</p>

    <h2>5. 两种模式的 hidden state 与线性 probe</h2>
    <h3>5.1 实验 D：THINK_ON/OFF close-state cosine similarity</h3>
    <div class="protocol"><p><b>配对 sequence。</b>对同一个 base prompt 分别渲染 THINK_ON 与 THINK_OFF。两个输入都截断在各自 <code>&lt;/Think&gt;</code>（包含 close token），不把 <code>&lt;C_n&gt;</code> 放入模型；因此比较的是“即将预测 count”的最后一个 hidden state。每个 count 1–10 各 500 个 prompts，共 5,000 个配对。</p><p><b>提取位置。</b>每个模式都取序列最后一个 token，即 <code>think_close_pos/pre_count_pos</code>。提取 hidden-state index 0（token embedding + learned absolute-position embedding）、index 1–3（分别经过 block 0–2 后的 residual）和 index 4（block 3 后再经过最终 <code>ln_f</code> 的 hidden state）。</p><p><b>计算。</b>对每个样本、每个 hidden-state index 计算 <code>cos(h_THINK_ON(close), h_THINK_OFF(close))</code>，再对 5,000 个配对求均值和标准差。</p><p><b>结果。</b>平均 cosine 从 embedding 层的 {fmt(sim_embedding)} 上升到最终 <code>ln_f</code> 后的 {fmt(sim_final)}；两种模式在最终 count readout 前不是完全分离的表示，反而经过 blocks 后更接近。</p><p><b>解释边界。</b>THINK_OFF close 固定在绝对位置 259；THINK_ON close 在 259+2n，因为 trace 含 n 对 index/marker。因此 cosine 同时混合了 switch 模式、trace context 和 learned absolute position 差异，不能单独解释成“推理机制距离”。</p></div>
    {figure(source_figures/'mode_hidden_similarity.png','Figure 7. THINK_ON/OFF 在 close anchor 的 hidden cosine','横轴是 hidden-state index：0=embedding+position，1–3=block 0–2 后的 residual，4=block 3 后再经过最终 ln_f 的 hidden state；纵轴是同一 prompt 两种模式在各自 close token hidden state 的平均 cosine similarity。点越低表示表示方向越不同，但差异包含 trace 与绝对位置两种混淆。',wide=True)}
    {table(sim_rows,[('state','表示位置'),('mean','平均 cosine'),('std','标准差')])}

    <h3>5.2 实验 E：count 信息何时出现、被写到哪个 token？</h3>
    <div class="callout good"><b>实验目的。</b>这不是再次评估模型答题是否正确，也不是继续训练模型；它是一个冻结模型后的“信息定位”实验。我们想回答：①模型尚未读 prompt 时，hidden state 中应当没有 count；②读完整个 prompt 后、生成 trace 前，是否已经能从某个固定位置读出 count；③到最终答案前，哪个 block 把 count 写成可直接分类的表示；④ THINK_ON 与 THINK_OFF 是否在相同阶段形成 count 信息。</div>
    <div class="protocol"><p><b>为什么比较 token anchors。</b>Transformer 在每个 token 位置都有一个 256 维 residual。由于 causal mask，一个位置只能包含它之前的上下文信息。比较 sequence 中不同位置，就像沿计算流程放置几个测量点：在 prompt 前测一次、读完整个 prompt 后测一次、即将输出答案前再测一次。</p><p><b>Sequence 与样本。</b>对每个 count 1–10 生成 500 个 prompts，共 5,000 个 base examples；同一 prompt 分别渲染 THINK_ON 与 THINK_OFF 的完整 gold sequence，并做一次 teacher-forced forward。虽然完整 sequence 被送入模型，但 causal mask 保证每个 anchor 看不到位于它之后的 gold tokens。模型参数全程冻结。</p></div>
    {table([
      {'anchor':'mode_pos','token / absolute position':'&lt;THINK_ON&gt; 或 &lt;THINK_OFF&gt;；position=1','此时看到了什么':'只有 BOS 和 mode token，prompt 尚未出现','为什么测':'负对照；理论上不应知道 needle 总数'},
      {'anchor':'prompt_marker_k','token / absolute position':'prompt 内第 k 个 marker；位置随 prompt 随机变化','此时看到了什么':'只看到 prompt 前缀与前 k 个 needles，尚未看到后续 prompt','为什么测':'观察 prefix 中局部累计信息；但存在条件选择和位置相关，不能当主要 final-count 证据'},
      {'anchor':'think_open_pos','token / absolute position':'&lt;Think/&gt;；position=258，两种模式固定','此时看到了什么':'完整 256-token prompt，但尚未看到任何 trace token','为什么测':'最干净的 pre-trace anchor；能检验模型是否在生成 CoT 前已从 prompt 得到 count'},
      {'anchor':'think_close_pos / pre_count_pos','token / absolute position':'&lt;/Think&gt;；THINK_OFF=259，THINK_ON=259+2n','此时看到了什么':'完整 prompt；THINK_ON 还看到了完整 indexed trace','为什么测':'它的 logits 直接预测 &lt;C_n&gt;，是 answer-ready state；但 THINK_ON 的位置本身泄漏 n'},
      {'anchor':'count token','token / absolute position':'gold &lt;C_n&gt; 本身','此时看到了什么':'token identity 已直接给出答案','为什么测':'严重 label leakage，标记为 leakage-prone 并从主图排除'}
    ], [('anchor','anchor 名称'),('token / absolute position','实际 token 与位置'),('此时看到了什么','可见上下文'),('为什么测','实验角色')])}
    <div class="protocol"><p><b>提取什么。</b>在每个 anchor 的同一个 token position 沿模型深度提取五次：index 0 是尚未经过 Transformer 的 embedding+position 向量；index 1、2、3 是分别经过 block 0、1、2 后的 residual；index 4 是经过 block 3 与最终 <code>ln_f</code> 后的向量。每次都是 256 维。以固定位置 <code>&lt;Think/&gt;</code> 为例：index 0 只含该 token identity 与 position 258；到 index 1 时，该位置的 attention 已经可以从前面的 256-token prompt 读入信息；后续 blocks 再继续更新同一条 residual stream。</p><p><b>Probe 做什么。</b>按缓存顺序用前 70% 样本训练、后 30% 测试。先用训练集 mean/std 标准化每一维；对十个 count 类分别计算训练向量 centroid。测试时，将一个 hidden vector 分给欧氏距离最近的 centroid。若分到真实 count，记为正确。图中的 accuracy 是 held-out nearest-centroid exact-count accuracy；平衡十分类的直观 chance 约为 10%。</p><p><b>为什么再做 ridge。</b>Nearest-centroid 要求精确分成十类；ridge regression（α=1）则把 count 当连续数值，测 hidden state 是否沿某个近似连续方向排列。报告 R² 与 MAE：R² 越接近 1 表示 count 的连续变化越可线性解释，MAE 是预测 count 与真实 count 的平均绝对差。</p><p><b>为什么需要 baseline。</b>Position baseline 只用 absolute token position 查训练集中的 majority count。如果 probe 很高、position baseline 也同样高，hidden vector 可能只编码了位置。Trace-length baseline 直接使用缓存中的 gold <code>trace_len=count</code>，即使 THINK_OFF 也如此，因此它只是 oracle sanity check，不是模型真正可见的输入。</p></div>
    <div class="formula">centroid_c = mean(z-scored hidden vectors with count=c)；prediction = argmin_c ||z(h_test) − centroid_c||₂</div>
    <div class="formula">ridge: count_hat = wᵀ z(h) + b；R²=1−SSE/SST；MAE=mean(|count_hat−count|)</div>
    <div class="callout result-line"><b>结果 1：负对照正常。</b><code>mode_pos</code> 位于 prompt 之前；non-thinking 的最佳 centroid accuracy 只有 {pct(float(probe_best_map[('nonthinking','mode_pos')].accuracy))}，接近十分类 chance，R² 约为 {fmt(float(probe_best_map[('nonthinking','mode_pos')].r2))}。这说明 probe 没有凭空从样本编号或 mode token 解码出 count。</div>
    <div class="callout result-line"><b>结果 2：模型在生成 trace 前已经具有 count 信息。</b>固定位置 <code>&lt;Think/&gt;</code> 已经看完整个 prompt、但还没有看到 trace。其最佳 exact centroid accuracy 为 THINK_OFF {pct(float(probe_best_map[('nonthinking','think_open_pos')].accuracy))}、THINK_ON {pct(float(probe_best_map[('thinking','think_open_pos')].accuracy))}；更重要的是 ridge R² 分别为 {fmt(float(probe_best_map[('nonthinking','think_open_pos')].r2))}/{fmt(float(probe_best_map[('thinking','think_open_pos')].r2))}，MAE 约 {fmt(float(probe_best_map[('nonthinking','think_open_pos')].mae))}/{fmt(float(probe_best_map[('thinking','think_open_pos')].mae))} 个 count。也就是说，count 在 trace 开始前已形成很强的连续可读表示；显式 CoT 不是模型第一次获得 count 的唯一途径。</div>
    <div class="callout result-line"><b>结果 3：THINK_OFF 的第一个 block 形成 answer-ready state。</b>固定位置的 non-thinking <code>&lt;/Think&gt;</code> 在 embedding 层只有 {pct(float(probe_subset[(probe_subset['mode']=='nonthinking') & (probe_subset.anchor_name=='think_close_pos') & (probe_subset.layer==0)].iloc[0].accuracy))}；经过第一个 block（hidden-state index 1）后立即达到 100% exact accuracy、ridge R²={fmt(float(probe_subset[(probe_subset['mode']=='nonthinking') & (probe_subset.anchor_name=='think_close_pos') & (probe_subset.layer==1)].iloc[0].r2))}，而 position baseline 只有 {pct(float(probe_best_map[('nonthinking','think_close_pos')].position_baseline_acc))}。因此 count 不是 close token embedding 自带的，也不是由绝对位置给出的；第一个 block 将 prompt 中的计数结果写入了用于预测 <code>&lt;C_n&gt;</code> 的 residual。这与后文 L0 broad-head ablation/patching 的因果结果一致。</div>
    <div class="callout warn"><b>结果 4：THINK_ON close 的 100% probe 不能证明“CoT 算出了 count”。</b>THINK_ON trace 每个 needle 恰好增加两个 token <code>&lt;I_k&gt;,M_k</code>，所以 close 的 absolute position 是 <code>259+2n</code>。在任何 Transformer block 之前，embedding+position state 就已经达到 100%，position baseline 也是 100%。这主要是实验格式直接把 n 编进了位置；因此该 anchor 不能用于判断 count arithmetic，应优先看固定位置的 Think-open，或做 position-matched counterfactual。</div>
    {figure(report_figures/'probe_accuracy.png','Figure 8. Count 信息在不同 token/层的可读性','左右分别为 THINK_OFF 与 THINK_ON。横轴 hidden-state index：0=token+position embedding，1–3=block 0–2 后的 residual，4=block 3 后再经过最终 ln_f 的 hidden state；纵轴为上表定义的 token anchor；每格是 30% held-out test 的 nearest-centroid exact-count accuracy。读图时重点看三类对照：mode_pos 应接近 chance；固定位置 think_open 显示 pre-trace count 信息；close 显示 answer-ready state。prompt_marker_k 行受“只有 count≥k 才存在该 anchor”及随机 absolute position影响，不应与固定 anchors 等价解释。',wide=True)}
    {table(probe_summary_rows,[('mode','模式'),('anchor','anchor'),('layer','最佳 hidden-state index'),('accuracy','centroid accuracy'),('r2','ridge R²'),('mae','ridge MAE'),('position','position baseline'),('trace_len','trace-length baseline')])}
    <div class="callout good"><b>这项实验最终说明什么。</b>在 THINK_OFF 中，模型读完整个 prompt 后已经形成近似连续的 count representation，并在第一个 block 后把它写成可直接选择 <code>&lt;C_n&gt;</code> 的 answer-ready residual。THINK_ON 在 trace 开始前也已有类似 count 信息，因此显式 trace 更像额外执行的检索过程，而不是获得 count 的唯一计算链。Probe 只说明“信息在哪里可读”；count-readout head mask 和 clean-to-corrupt patching 才说明后续报告中的 early broad heads 对该表示具有因果作用。</div>
    <div class="callout warn"><b>这项实验没有说明什么。</b>它没有证明存在单一 count neuron，没有证明 residual 使用严格线性加法，也没有证明 THINK_ON close 的表示来自逐步累加。尤其是 THINK_ON close 的绝对位置泄漏，使其高 probe accuracy 本身几乎没有机制解释力。</div>

    {v53_section}
    {core_questions_section}

    <h2>7. 综合机制解释</h2>
    <div class="format-grid">
      <div class="format"><b>THINK_OFF：direct broad aggregation</b><p>最终 query 的候选 signature 是对多个 prompt needles 的宽分布 attention。L0H1 单头 mask 使 teacher-forced count accuracy 平均下降 70 个百分点；L0H1+L0H3 clean-to-corrupt patch 恢复约 79% count margin。证据共同支持第一个 block 形成直接 count aggregation，再由后层稳定读出。</p></div>
      <div class="format"><b>THINK_ON：targeted retrieval + redundant final route</b><p>block 2 的 L1H0/H1/H2 在 <code>&lt;I_k&gt;</code> 处执行 k-to-k retrieval。正确位置的 targeted top-2 patch 恢复约 82% marker identity；mask 整个 L1 会摧毁 trace，但 final count 仍可正确，说明 final answer 主要由与 non-thinking 共用的 direct prompt path 保底，trace 是可执行但非唯一的中间计算。</p></div>
    </div>
    <div class="callout warn"><b>不要过度解读。</b>“trace 被破坏而 final 仍正确”不等于 trace 永远无用：forced-trace conflict 会让约 44% 的非对角样本偏离 prompt count。更准确的描述是：prompt-direct 路径占主导，trace/position-progress 状态会调制最终读出，两条信息源在正常数据上始终一致，因此训练没有迫使模型只选其中一条。</div>

    <h2>8. 结论：支持什么，不支持什么</h2>
    <div class="callout good"><b>当前结果支持：</b><ul><li>一个单模型可被显式 THINK_ON/OFF token 稳定路由到“生成 indexed trace”或“跳过 trace”两条路径。</li><li>block-2 多头 circuit 对 indexed marker retrieval 具有必要性和位置特异的充分性；top-2 patch 恢复约 82%，top-4 接近 100%。</li><li>non-thinking count 主要依赖 block-1 broad aggregation heads；对应 top-2 patch 恢复约 79%。</li><li>trace progress residual 足以因果改变下一 index token，说明 trace 中存在显式进度状态。</li><li>最终 count 保留 prompt-direct 旁路；CoT trace 是实际执行的检索，不是唯一 answer channel。</li></ul></div>
    <div class="callout warn"><b>当前结果不支持：</b><ul><li>不支持 thinking 提升当前 ID accuracy，因为两种模式都饱和。</li><li>不支持“每个 targeted head 单独必要”；top-1/top-2 mask 不改变 argmax，存在明显冗余，top-4 又与整层 ablation 混淆。</li><li>不支持最终 count 完全由 trace 决定；冲突样本只有 {pct(follows_trace)} 严格跟随 forced trace。</li><li>不支持 progress state 已与 learned absolute position 解耦；当前 transplant 同时改变语义进度和 donor 绝对位置。</li><li>不支持 count/length OOD，也不支持直接外推到真实语言模型。</li></ul></div>

    <h2>9. 下一步最有价值的实验</h2>
    <ol><li><b>Combinatorial retrieval ablation：</b>对 L1H0/H1/H2/H3 的全部非空子集做 15 组 mask，定位冗余结构，避免把 targeted top-4 与 whole-layer effect 混为一谈。</li><li><b>Position-controlled progress patch：</b>构造相同绝对位置、不同 progress token 的 donor，或显式交换 index token embedding，区分抽象计数状态和 learned position cue。</li><li><b>Path-specific final readout：</b>在 prompt count 与 trace count 冲突时分别 patch L0 broad heads、trace-attending heads 与 residual，测哪条路径把 final logit 拉向 prompt 或 trace。</li><li><b>Hard regime：</b>增加长度、needle 数量或减小模型，使 direct path 不再饱和，再比较显式 retrieval trace 是否提高样本效率、鲁棒性或 OOD。</li></ol>

    <h2>10. 文件与复现</h2>
    {table([
      {'item':'训练配置','path':code(run_dir/'config.json')}, {'item':'v5 原始表','path':code(tables)}, {'item':'checkpoint','path':code(run_dir/'checkpoints'/'final.pt')}, {'item':'v5.3 配置','path':code(v53_dir/'run_config.json')}, {'item':'v5.3 原始表','path':code(v53_tables)}, {'item':'报告生成器','path':code('scripts/build_v5_explicit_switch_report.py')}
    ], [('item','内容'),('path','路径')])}
    <p class="small">本 HTML 中所有 PNG 均以 base64 内嵌，可作为单文件离线发送。所有核心数字从 CSV 重新聚合；层和 head 的表格编号严格采用结果文件中的 0-based 编号。</p>
    </main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.run_dir / "syn_v5_report.html"
    output.write_text(build_report(args.run_dir), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
