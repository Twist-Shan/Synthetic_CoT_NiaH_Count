#!/usr/bin/env python
"""Build the self-contained v22 synthetic NiaH mechanism report.

Primary comparison:
    v22 separator/no-index Thinking vs the registered matched v20 Non-thinking
    baseline.

The report separates clean representation, post-ablation mediation,
free-running necessity/sufficiency, and training dynamics.  It consumes only
archived CSV/JSON artifacts and therefore does not rerun model inference.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
V20_RUN = ROOT / "colab_results" / "v20_main_RoPE_count1-30_seed1234"
ASSET_DIR = ROOT / "reports" / "NiaH_Synthetic_report_assets"

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


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D8DEE7", linewidth=0.65, alpha=0.72)
    ax.spines[["top", "right"]].set_visible(False)


def data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def figure(path: Path, caption: str, alt: str) -> str:
    return (
        f'<figure><img src="{data_uri(path)}" alt="{html.escape(alt)}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{100 * number:.{digits}f}%"


def html_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{item}</th>" for item in headers)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{item}</td>" for item in row)
            + "</tr>"
        )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def selected_clean_rows(clean: pd.DataFrame, selected: dict) -> pd.DataFrame:
    rows = []
    for mode, endpoints in selected.items():
        for endpoint, layer in endpoints.items():
            match = clean.loc[
                clean["comparison_mode"].eq(mode)
                & clean["endpoint"].eq(endpoint)
                & clean["layer"].astype(int).eq(int(layer))
            ]
            if len(match) != 1:
                raise RuntimeError(f"clean selection mismatch: {mode}/{endpoint}/L{layer}")
            rows.append(match.iloc[0])
    return pd.DataFrame(rows)


def plot_clean_geometry(selected: pd.DataFrame, path: Path) -> None:
    order = [
        ("nonthinking", "nonthinking_prompt_occurrence", "NT running\n(prompt occurrence)"),
        ("thinking", "thinking_item_end", "Thinking running\n(trace item end)"),
        ("nonthinking", "nonthinking_answer_query", "NT final count\n(answer query)"),
        ("thinking", "thinking_answer_query", "Thinking final count\n(answer query)"),
    ]
    frame_rows = []
    for mode, endpoint, label in order:
        row = selected.loc[
            selected["comparison_mode"].eq(mode) & selected["endpoint"].eq(endpoint)
        ].iloc[0]
        frame_rows.append((label, row))
    x = np.arange(len(frame_rows))
    width = 0.34
    logistic = [float(row["confirmation_logistic_balanced_accuracy"]) for _, row in frame_rows]
    ncc = [float(row["confirmation_ncc_balanced_accuracy"]) for _, row in frame_rows]
    colors = [BLUE, ORANGE, BLUE_DARK, PURPLE]
    fig, ax = plt.subplots(figsize=(11.5, 5.3))
    ax.bar(x - width / 2, logistic, width, color=colors, alpha=0.93, label="L2-logistic BA")
    ax.bar(
        x + width / 2,
        ncc,
        width,
        color="white",
        edgecolor=colors,
        linewidth=2.0,
        hatch="///",
        label="Nearest-centroid BA",
    )
    ax.axhline(1 / 30, color="#555", linestyle=":", linewidth=1.3, label="Chance = 1/30")
    ax.set_xticks(x, [label for label, _ in frame_rows])
    ax.set_ylabel("Held-out balanced accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Clean count geometry at discovery-selected layers")
    for index, (_, row) in enumerate(frame_rows):
        layer = int(row["layer"])
        ax.text(index, max(logistic[index], ncc[index]) + 0.035, f"L{layer}", ha="center", fontsize=9)
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    style_axis(ax)
    savefig(fig, path)


def condition_series(
    frame: pd.DataFrame,
    *,
    metric: str,
    endpoint: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    data = frame.copy()
    if endpoint is not None:
        data = data.loc[data["endpoint"].eq(endpoint)]
    ranked = (
        data.loc[data["path_kind"].isin(["clean", "ranked"])]
        .groupby("top_k")[metric]
        .mean()
        .sort_index()
    )
    controls = (
        data.loc[data["path_kind"].eq("layer_matched_control")]
        .groupby("top_k")[metric]
        .agg(["mean", "min", "max", "count"])
        .sort_index()
    )
    return ranked, controls


def draw_ranked_control(
    ax: plt.Axes,
    ranked: pd.Series,
    controls: pd.DataFrame,
    *,
    color: str,
    ylabel: str,
    title: str,
    chance: float | None = None,
) -> None:
    ax.plot(ranked.index, ranked.values, marker="o", linewidth=2.3, color=color, label="Ranked Top-K")
    if len(controls):
        lower = controls["mean"] - controls["min"]
        upper = controls["max"] - controls["mean"]
        ax.errorbar(
            controls.index,
            controls["mean"],
            yerr=np.vstack([lower, upper]),
            marker="s",
            linestyle="--",
            capsize=4,
            linewidth=1.7,
            color=GREY,
            label="Layer-matched controls (mean/range)",
        )
    if chance is not None:
        ax.axhline(chance, color="#555", linestyle=":", linewidth=1.1)
    ax.set_xticks([0, 1, 2, 4])
    ax.set_xlabel("Cumulative K heads ablated")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax)


def plot_teacher_forced_topk(ncc: pd.DataFrame, behavior: pd.DataFrame, path: Path) -> None:
    ncc = ncc.loc[ncc["scope"].eq("role_query_local")]
    behavior = behavior.loc[behavior["scope"].eq("role_query_local")]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.3))

    nt = ncc.loc[ncc["comparison_mode"].eq("nonthinking")]
    ranked, controls = condition_series(
        nt, metric="confirmation_ncc_balanced_accuracy", endpoint="nonthinking_answer_query"
    )
    draw_ranked_control(
        axes[0, 0], ranked, controls, color=BLUE_DARK,
        ylabel="Frozen NCC balanced accuracy", title="A · NT final-count geometry", chance=1 / 30,
    )

    nt_b = behavior.loc[behavior["comparison_mode"].eq("nonthinking")]
    ranked, controls = condition_series(nt_b, metric="teacher_forced_final_accuracy")
    draw_ranked_control(
        axes[0, 1], ranked, controls, color=BLUE,
        ylabel="Teacher-forced final accuracy", title="B · NT final behavior",
    )

    th_b = behavior.loc[behavior["comparison_mode"].eq("thinking")]
    ranked, controls = condition_series(th_b, metric="teacher_forced_trace_accuracy")
    draw_ranked_control(
        axes[1, 0], ranked, controls, color=ORANGE,
        ylabel="Teacher-forced marker accuracy", title="C · Thinking targeted retrieval",
    )

    th = ncc.loc[ncc["comparison_mode"].eq("thinking")]
    ranked, controls = condition_series(
        th, metric="confirmation_ncc_balanced_accuracy", endpoint="thinking_answer_query"
    )
    draw_ranked_control(
        axes[1, 1], ranked, controls, color=PURPLE,
        ylabel="Frozen NCC balanced accuracy", title="D · Thinking final-count geometry", chance=1 / 30,
    )
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Role-query-local Top-K necessity: immediate damage versus downstream repair", fontsize=15)
    fig.tight_layout()
    savefig(fig, path)


def plot_free_running_topk(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.3))
    panels = [
        ("nonthinking", "ar_final_accuracy", "A · NT final answer", BLUE),
        ("thinking", "ar_final_accuracy", "B · Thinking final answer", PURPLE),
        ("thinking", "trace_exact", "C · Thinking exact trace", ORANGE),
        (
            "thinking",
            "trace_ordered_marker_accuracy",
            "D · Thinking ordered-marker accuracy",
            GREEN,
        ),
    ]
    for ax, (mode, metric, title, color) in zip(axes.flat, panels, strict=True):
        data = summary.loc[summary["comparison_mode"].eq(mode)]
        ranked, controls = condition_series(data, metric=metric)
        draw_ranked_control(
            ax,
            ranked,
            controls,
            color=color,
            ylabel="Held-out free-running accuracy",
            title=title,
        )
        ax.set_ylim(0, 1.04)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Free-running Top-K necessity on the same reporting examples", fontsize=15)
    fig.tight_layout()
    savefig(fig, path)


def plot_answer_sufficiency(answer: pd.DataFrame, path: Path) -> None:
    chosen = answer.loc[answer["is_discovery_selected_layer"].eq(1)].copy()
    conditions = ["clean", "same_count_context_control", "adjacent_count_donor"]
    labels = ["Clean", "Same-count\ncontext control", "Adjacent-count\ndonor"]
    colors = [LIGHT_GREY, GREY, PURPLE]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0))
    for column, mode in enumerate(("nonthinking", "thinking")):
        data = chosen.loc[chosen["mode"].eq(mode)]
        adoption = [float(data.loc[data["condition"].eq(c), "donor_adoption"].mean()) for c in conditions]
        shift = [float(data.loc[data["condition"].eq(c), "donor_margin_shift"].mean()) for c in conditions]
        axes[0, column].bar(range(3), adoption, color=colors, edgecolor="#52606D")
        axes[1, column].bar(range(3), shift, color=colors, edgecolor="#52606D")
        axes[0, column].set_title(f"{'Non-thinking' if mode == 'nonthinking' else 'Thinking'} donor adoption")
        axes[1, column].set_title(f"{'Non-thinking' if mode == 'nonthinking' else 'Thinking'} margin transport")
        axes[0, column].set_ylabel("Greedy donor-count adoption")
        axes[1, column].set_ylabel("Δ donor-minus-receiver logit margin")
        for ax in axes[:, column]:
            ax.set_xticks(range(3), labels)
            style_axis(ax)
    axes[0, 0].set_ylim(0, 0.75)
    axes[0, 1].set_ylim(0, 0.75)
    axes[1, 0].axhline(0, color="#333", linewidth=1)
    axes[1, 1].axhline(0, color="#333", linewidth=1)
    fig.suptitle("Free-running answer-query residual transplantation (56 held-out adjacent pairs)", fontsize=15)
    fig.tight_layout()
    savefig(fig, path)


def plot_progress_sufficiency(progress: pd.DataFrame, path: Path) -> None:
    order = [
        "centroid_shift",
        "orthogonal_control",
        "natural_marker_cross_position",
        "natural_item_span_cross_position",
    ]
    labels = ["Centroid\nshift", "Equal-norm\northogonal", "Natural marker\n(cross-position)", "Natural span\n(cross-position)"]
    summary = progress.groupby("condition").agg(
        route_margin=("forced_donor_minus_natural_margin", "mean"),
        donor_adoption=("donor_first_adoption", "mean"),
    )
    baseline = float(summary.loc["clean", "route_margin"])
    margin_shift = [float(summary.loc[item, "route_margin"] - baseline) for item in order]
    adoption = [float(summary.loc[item, "donor_adoption"]) for item in order]
    colors = [PURPLE, GREY, ORANGE, GREEN]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    bars = axes[0].bar(range(4), margin_shift, color=colors, edgecolor="#52606D")
    bars[2].set_hatch("///")
    bars[3].set_hatch("///")
    axes[0].axhline(0, color="#333", linewidth=1)
    axes[0].set_ylabel("Δ donor-minus-natural successor margin")
    axes[0].set_title("A · Continuous route shift")
    bars = axes[1].bar(range(4), adoption, color=colors, edgecolor="#52606D")
    bars[2].set_hatch("///")
    bars[3].set_hatch("///")
    axes[1].set_ylabel("Greedy donor-successor adoption")
    axes[1].set_ylim(0, 0.5)
    axes[1].set_title("B · Behavioral continuation transfer")
    for ax in axes:
        ax.set_xticks(range(4), labels)
        style_axis(ax)
    fig.suptitle("Free-running progress-state intervention at frozen L1 (11 eligible held-out cells)", fontsize=15)
    fig.tight_layout()
    savefig(fig, path)


def bank_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for step, group in frame.groupby("step"):
        values = np.clip(group["score"].to_numpy(dtype=float), 0, None)
        if values.sum() <= 0:
            shares = np.full(len(values), 1 / len(values))
        else:
            shares = values / values.sum()
        entropy = -float(np.sum(shares * np.log(shares + 1e-30)))
        rows.append(
            {
                "step": int(step),
                "effective_heads": float(np.exp(entropy)),
                "top2_share": float(np.sort(shares)[-2:].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("step")


def plot_training_dynamics(
    v20_behavior: pd.DataFrame,
    v22_behavior: pd.DataFrame,
    nt_broad: pd.DataFrame,
    th_targeted: pd.DataFrame,
    rankings: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0))
    ax = axes[0, 0]
    nt = (
        v20_behavior.loc[
            v20_behavior["mode"].eq("nonthinking")
            & v20_behavior["outcome"].eq("final_answer_teacher_forced_exact")
        ]
        .groupby("step")["accuracy"]
        .mean()
    )
    th = (
        v22_behavior.loc[
            v22_behavior["mode"].eq("thinking")
            & v22_behavior["outcome"].eq("final_answer_teacher_forced_exact")
        ]
        .groupby("step")["accuracy"]
        .mean()
    )
    ax.plot(nt.index, nt.values, color=BLUE, linewidth=2, label="Matched NT (v20)")
    ax.plot(th.index, th.values, color=PURPLE, linewidth=2, label="No-index Thinking (v22)")
    ax.set_title("A · Macro teacher-forced final-count behavior")
    ax.set_ylabel("Accuracy across counts 1–30")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8)
    style_axis(ax)

    final_nt = rankings.loc[rankings["comparison_mode"].eq("nonthinking")].sort_values("rank")
    top_nt = {(int(row.layer), int(row["head"])) for _, row in final_nt.head(4).iterrows()}
    ax = axes[0, 1]
    for (layer, head), group in nt_broad.groupby(["layer", "head"]):
        key = (int(layer), int(head))
        selected = key in top_nt
        ax.plot(
            group["step"], group["score"],
            color=BLUE if selected else LIGHT_GREY,
            linewidth=2.0 if selected else 0.8,
            alpha=0.95 if selected else 0.65,
            label=f"L{layer}H{head}" if selected else None,
        )
    ax.set_title("B · Non-thinking broad-bank specialization")
    ax.set_ylabel("Broad score  M·exp(H)/N")
    ax.legend(fontsize=7, ncol=2)
    style_axis(ax)

    final_th = rankings.loc[rankings["comparison_mode"].eq("thinking")].sort_values("rank")
    top_th = {(int(row.layer), int(row["head"])) for _, row in final_th.head(2).iterrows()}
    ax = axes[1, 0]
    for (layer, head), group in th_targeted.groupby(["layer", "head"]):
        key = (int(layer), int(head))
        selected = key in top_th
        ax.plot(
            group["step"], group["score"],
            color=ORANGE if selected else LIGHT_GREY,
            linewidth=2.4 if selected else 0.8,
            alpha=0.98 if selected else 0.65,
            label=f"L{layer}H{head}" if selected else None,
        )
    ax.set_title("C · Thinking targeted-bank specialization")
    ax.set_ylabel("Correct kth-occurrence attention mass")
    ax.legend(fontsize=8)
    style_axis(ax)

    ax = axes[1, 1]
    nt_stats = bank_statistics(nt_broad)
    th_stats = bank_statistics(th_targeted)
    ax.plot(nt_stats["step"], nt_stats["effective_heads"], color=BLUE, linewidth=2, label="NT broad: effective heads")
    ax.plot(th_stats["step"], th_stats["effective_heads"], color=PURPLE, linewidth=2, label="Thinking targeted: effective heads")
    ax.set_ylabel("Entropy effective number of heads")
    ax.set_ylim(0, 17)
    twin = ax.twinx()
    twin.plot(nt_stats["step"], nt_stats["top2_share"], color=BLUE, linestyle="--", linewidth=1.8, label="NT broad: Top-2 share")
    twin.plot(th_stats["step"], th_stats["top2_share"], color=PURPLE, linestyle="--", linewidth=1.8, label="Thinking targeted: Top-2 share")
    twin.set_ylabel("Top-2 share of role mass")
    twin.set_ylim(0, 1)
    ax.set_title("D · Head-bank differentiation")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=7, loc="center right")
    style_axis(ax)

    for ax in axes.flat:
        ax.axvline(1500, color="#555", linestyle=":", linewidth=1.1)
        ax.set_xlabel("Optimizer step (linear axis)")
    fig.suptitle("Aligned training dynamics: matched Non-thinking broad vs v22 Thinking targeted retrieval", fontsize=15)
    fig.tight_layout()
    savefig(fig, path)


def build_report(output: Path) -> None:
    clean_root = WORK / "ncc_v22_thinking_vs_nonthinking"
    topk_root = WORK / "v22_topk_ncc"
    fr_topk_root = WORK / "v22_free_running_topk"
    suff_root = WORK / "v22_free_running_sufficiency"
    dyn_root = WORK / "v22_training_dynamics"

    clean = read_csv(clean_root / "geometry_site_layer_metrics.csv")
    selected_map = json.loads((clean_root / "selected_layers.json").read_text(encoding="utf-8"))
    selected = selected_clean_rows(clean, selected_map)
    ncc = read_csv(topk_root / "post_ablation_ncc_selected.csv")
    behavior = read_csv(topk_root / "post_ablation_behavior.csv")
    rankings = read_csv(topk_root / "head_rankings.csv")
    fr_topk = read_csv(fr_topk_root / "free_running_summary.csv")
    answer = read_csv(suff_root / "answer_transplant_confirmation.csv")
    progress = read_csv(suff_root / "progress_rollout_confirmation.csv")
    selected_suff = json.loads((suff_root / "selected_layers.json").read_text(encoding="utf-8"))
    v22_cfg = json.loads((dyn_root / "config.json").read_text(encoding="utf-8"))
    v22_ar = read_csv(dyn_root / "tables__final_autoregressive_summary.csv").iloc[0]
    v20_ar = read_csv(V20_RUN / "tables" / "final_autoregressive_summary.csv")
    v20_nt_ar = v20_ar.loc[v20_ar["mode"].eq("nonthinking")].iloc[0]
    v20_behavior = read_csv(V20_RUN / "analysis" / "phase_transition" / "tables" / "dense_behavior_by_count.csv")
    v22_behavior = read_csv(dyn_root / "analysis__phase_transition__tables__dense_behavior_by_count.csv")
    v20_roles = read_csv(V20_RUN / "analysis" / "extended" / "tables" / "attention_role_dynamics.csv")
    nt_broad = v20_roles.loc[v20_roles["role"].eq("nonthinking_broad")].copy()
    v22_roles = read_csv(dyn_root / "analysis__phase_transition__tables__dense_fixed_head_dynamics.csv")
    th_targeted = v22_roles.loc[v22_roles["role"].eq("targeted_retrieval")].copy()

    figures = {
        "clean": ASSET_DIR / "clean_ncc_comparison.png",
        "tf_topk": ASSET_DIR / "teacher_forced_topk_ncc.png",
        "fr_topk": ASSET_DIR / "free_running_topk.png",
        "answer": ASSET_DIR / "free_running_answer_transplant.png",
        "progress": ASSET_DIR / "free_running_progress_sufficiency.png",
        "dynamics": ASSET_DIR / "aligned_head_bank_training_dynamics.png",
        "v22_roles": dyn_root / "figures__dense_fixed_head_emergence.png",
        "v22_causal": dyn_root / "figures__milestone_local_head_causality.png",
    }
    plot_clean_geometry(selected, figures["clean"])
    plot_teacher_forced_topk(ncc, behavior, figures["tf_topk"])
    plot_free_running_topk(fr_topk, figures["fr_topk"])
    plot_answer_sufficiency(answer, figures["answer"])
    plot_progress_sufficiency(progress, figures["progress"])
    plot_training_dynamics(
        v20_behavior, v22_behavior, nt_broad, th_targeted, rankings, figures["dynamics"]
    )

    endpoint_labels = {
        "nonthinking_prompt_occurrence": "Non-thinking running index",
        "thinking_item_end": "Thinking running index",
        "nonthinking_answer_query": "Non-thinking final count",
        "thinking_answer_query": "Thinking final count",
    }
    geometry_rows = []
    for _, row in selected.sort_values(["endpoint"]).iterrows():
        geometry_rows.append(
            [
                "v22 Thinking" if row["comparison_mode"] == "thinking" else "matched v20 Non-thinking",
                endpoint_labels[str(row["endpoint"])],
                f"L{int(row['layer'])}",
                pct(row["confirmation_logistic_balanced_accuracy"]),
                pct(row["confirmation_ncc_balanced_accuracy"]),
                fmt(row["confirmation_isotropic_snr_db"], 2) + " dB",
                fmt(row["confirmation_ordinal_rsa"], 3),
            ]
        )
    geometry_table = html_table(
        ["Model", "Endpoint / label", "Frozen layer", "Logistic BA", "NCC BA", "Isotropic SNR", "Ordinal RSA"],
        geometry_rows,
    )

    local_ncc = ncc.loc[ncc["scope"].eq("role_query_local")]
    local_beh = behavior.loc[behavior["scope"].eq("role_query_local")]

    def arm_value(frame: pd.DataFrame, mode: str, kind: str, k: int, metric: str, endpoint: str | None = None) -> float:
        subset = frame.loc[
            frame["comparison_mode"].eq(mode)
            & frame["path_kind"].eq(kind)
            & frame["top_k"].astype(int).eq(k)
        ]
        if endpoint is not None:
            subset = subset.loc[subset["endpoint"].eq(endpoint)]
        return float(subset[metric].mean())

    topk_rows = [
        [
            "NT final NCC",
            pct(arm_value(local_ncc, "nonthinking", "clean", 0, "confirmation_ncc_balanced_accuracy", "nonthinking_answer_query")),
            pct(arm_value(local_ncc, "nonthinking", "ranked", 1, "confirmation_ncc_balanced_accuracy", "nonthinking_answer_query")),
            pct(arm_value(local_ncc, "nonthinking", "ranked", 2, "confirmation_ncc_balanced_accuracy", "nonthinking_answer_query")),
            pct(arm_value(local_ncc, "nonthinking", "ranked", 4, "confirmation_ncc_balanced_accuracy", "nonthinking_answer_query")),
            pct(arm_value(local_ncc, "nonthinking", "layer_matched_control", 2, "confirmation_ncc_balanced_accuracy", "nonthinking_answer_query")),
        ],
        [
            "NT final accuracy (one-token free run)",
            pct(arm_value(fr_topk, "nonthinking", "clean", 0, "ar_final_accuracy")),
            pct(arm_value(fr_topk, "nonthinking", "ranked", 1, "ar_final_accuracy")),
            pct(arm_value(fr_topk, "nonthinking", "ranked", 2, "ar_final_accuracy")),
            pct(arm_value(fr_topk, "nonthinking", "ranked", 4, "ar_final_accuracy")),
            pct(arm_value(fr_topk, "nonthinking", "layer_matched_control", 2, "ar_final_accuracy")),
        ],
        [
            "Thinking trace marker TF accuracy",
            pct(arm_value(local_beh, "thinking", "clean", 0, "teacher_forced_trace_accuracy")),
            pct(arm_value(local_beh, "thinking", "ranked", 1, "teacher_forced_trace_accuracy")),
            pct(arm_value(local_beh, "thinking", "ranked", 2, "teacher_forced_trace_accuracy")),
            pct(arm_value(local_beh, "thinking", "ranked", 4, "teacher_forced_trace_accuracy")),
            pct(arm_value(local_beh, "thinking", "layer_matched_control", 2, "teacher_forced_trace_accuracy")),
        ],
        [
            "Thinking final NCC after local ablation",
            pct(arm_value(local_ncc, "thinking", "clean", 0, "confirmation_ncc_balanced_accuracy", "thinking_answer_query")),
            pct(arm_value(local_ncc, "thinking", "ranked", 1, "confirmation_ncc_balanced_accuracy", "thinking_answer_query")),
            pct(arm_value(local_ncc, "thinking", "ranked", 2, "confirmation_ncc_balanced_accuracy", "thinking_answer_query")),
            pct(arm_value(local_ncc, "thinking", "ranked", 4, "confirmation_ncc_balanced_accuracy", "thinking_answer_query")),
            pct(arm_value(local_ncc, "thinking", "layer_matched_control", 2, "confirmation_ncc_balanced_accuracy", "thinking_answer_query")),
        ],
        [
            "Thinking final free-running accuracy",
            pct(arm_value(fr_topk, "thinking", "clean", 0, "ar_final_accuracy")),
            pct(arm_value(fr_topk, "thinking", "ranked", 1, "ar_final_accuracy")),
            pct(arm_value(fr_topk, "thinking", "ranked", 2, "ar_final_accuracy")),
            pct(arm_value(fr_topk, "thinking", "ranked", 4, "ar_final_accuracy")),
            pct(arm_value(fr_topk, "thinking", "layer_matched_control", 2, "ar_final_accuracy")),
        ],
        [
            "Thinking exact free-running trace",
            pct(arm_value(fr_topk, "thinking", "clean", 0, "trace_exact")),
            pct(arm_value(fr_topk, "thinking", "ranked", 1, "trace_exact")),
            pct(arm_value(fr_topk, "thinking", "ranked", 2, "trace_exact")),
            pct(arm_value(fr_topk, "thinking", "ranked", 4, "trace_exact")),
            pct(arm_value(fr_topk, "thinking", "layer_matched_control", 2, "trace_exact")),
        ],
    ]
    topk_table = html_table(
        ["Outcome", "Clean", "Ranked K=1", "Ranked K=2", "Ranked K=4", "Matched control K=2"],
        topk_rows,
    )

    answer_selected = answer.loc[answer["is_discovery_selected_layer"].eq(1)]
    answer_rows = []
    for mode in ("nonthinking", "thinking"):
        for condition in ("clean", "same_count_context_control", "adjacent_count_donor"):
            part = answer_selected.loc[
                answer_selected["mode"].eq(mode) & answer_selected["condition"].eq(condition)
            ]
            answer_rows.append(
                [
                    "Non-thinking" if mode == "nonthinking" else "Thinking",
                    condition.replace("_", " "),
                    f"L{selected_suff['answer_transplant'][mode]}",
                    len(part),
                    pct(part["donor_adoption"].mean()),
                    pct(part["receiver_retention"].mean()),
                    fmt(part["donor_minus_receiver_margin"].mean(), 3),
                    fmt(part["donor_margin_shift"].mean(), 3),
                ]
            )
    answer_table = html_table(
        ["Mode", "Condition", "Layer", "Pairs", "Donor adoption", "Receiver retention", "Donor−receiver margin", "Shift from clean"],
        answer_rows,
    )

    progress_summary = progress.groupby("condition").agg(
        n=("donor_first_adoption", "size"),
        route_margin=("forced_donor_minus_natural_margin", "mean"),
        donor_adoption=("donor_first_adoption", "mean"),
        natural_retention=("natural_first_retention", "mean"),
        match_length=("donor_continuation_match_length", "mean"),
        first3=("donor_first_three_exact", "mean"),
        final_correct=("generated_final_correct", "mean"),
    )
    progress_rows = []
    for condition in (
        "clean",
        "self_patch",
        "centroid_shift",
        "orthogonal_control",
        "natural_marker_cross_position",
        "natural_item_span_cross_position",
    ):
        row = progress_summary.loc[condition]
        progress_rows.append(
            [
                condition.replace("_", " "),
                int(row["n"]),
                fmt(row["route_margin"], 3),
                pct(row["donor_adoption"]),
                pct(row["natural_retention"]),
                fmt(row["match_length"], 2),
                pct(row["first3"]),
                pct(row["final_correct"]),
            ]
        )
    progress_table = html_table(
        ["Condition", "Eligible cells", "Donor−natural margin", "Donor successor", "Natural successor", "Mean donor match", "First 3 exact", "Final answer correct"],
        progress_rows,
    )

    nt_final_ncc = arm_value(local_ncc, "nonthinking", "clean", 0, "confirmation_ncc_balanced_accuracy", "nonthinking_answer_query")
    th_final_ncc = arm_value(local_ncc, "thinking", "clean", 0, "confirmation_ncc_balanced_accuracy", "thinking_answer_query")
    nt_running_ncc = arm_value(local_ncc, "nonthinking", "clean", 0, "confirmation_ncc_balanced_accuracy", "nonthinking_prompt_occurrence")
    th_running_ncc = arm_value(local_ncc, "thinking", "clean", 0, "confirmation_ncc_balanced_accuracy", "thinking_item_end")
    th_fr_clean = arm_value(fr_topk, "thinking", "clean", 0, "ar_final_accuracy")
    th_fr_k2 = arm_value(fr_topk, "thinking", "ranked", 2, "ar_final_accuracy")
    th_fr_k2_control = arm_value(fr_topk, "thinking", "layer_matched_control", 2, "ar_final_accuracy")
    th_trace_clean = arm_value(fr_topk, "thinking", "clean", 0, "trace_exact")
    th_trace_k2 = arm_value(fr_topk, "thinking", "ranked", 2, "trace_exact")
    th_trace_k2_control = arm_value(fr_topk, "thinking", "layer_matched_control", 2, "trace_exact")

    top_nt_heads = ", ".join(
        f"L{int(row.layer)}H{int(row['head'])}"
        for _, row in rankings.loc[rankings["comparison_mode"].eq("nonthinking")].sort_values("rank").head(4).iterrows()
    )
    top_th_heads = ", ".join(
        f"L{int(row.layer)}H{int(row['head'])}"
        for _, row in rankings.loc[rankings["comparison_mode"].eq("thinking")].sort_values("rank").head(4).iterrows()
    )
    nt_stats = bank_statistics(nt_broad)
    th_stats = bank_statistics(th_targeted)
    nt_final_stats = nt_stats.iloc[-1]
    th_final_stats = th_stats.iloc[-1]

    setup_table = html_table(
        ["Component", "Setting", "Why it is controlled"],
        [
            ["Task", "256 Shakespeare characters; query-first; 3 target characters; total count 1–30; natural count distribution", "Same underlying task examples in the two modes"],
            ["Model", "3,185,920 parameters; 4 layers × 4 heads; d_model=256; MLP=1024; RoPE; atomic answer tokens", "Architecture and final-answer tokenization matched"],
            ["Thinking trace", "v22: &lt;Think&gt; (&lt;Sep&gt; marker)<sup>N</sup> &lt;/Think&gt; &lt;Ans&gt; count", "No numeric running index appears in the trace"],
            ["Non-thinking", "matched v20: prompt → &lt;Ans&gt; count", "v22 intentionally trained no separate Non-thinking model"],
            ["Optimization", "seed 1234; batch 128; 10,000 steps; Adam β=(0.9,0.999); lr=3×10⁻⁴; warmup 500; weight decay 0.01; grad clip 1", "Only one training seed: descriptive case study"],
            ["Loss schedule", "steps 1–1500 all-sequence LM; steps 1501–10,000 task-output only; final-count weight=1; trace weight=1", "Vertical line at 1,500 in dynamics plots"],
            ["Runtime", "CUDA + BF16 configured; dense scientific snapshots every 100 steps; recovery state every 500", "Archived metadata does not record the exact GPU SKU"],
        ],
    )

    alignment_table = html_table(
        ["Large-model experiment / claim", "Synthetic aligned result", "Status"],
        [
            ["Separate prompt running-index from answer-query final-count geometry", f"Running NCC: NT {pct(nt_running_ncc)}, Thinking {pct(th_running_ncc)}; final NCC: NT {pct(nt_final_ncc)}, Thinking {pct(th_final_ncc)}", '<span class="status partial">final aligned; running gap</span>'],
            ["Selected broad/targeted Top-K vs matched controls", f"NT broad is a distributed L1 bank; Thinking K=2 targeted removal cuts free-run trace exact {pct(th_trace_clean)}→{pct(th_trace_k2)} while K=2 control is {pct(th_trace_k2_control)}", '<span class="status yes">aligned positive</span>'],
            ["Post-ablation representation readout", "Clean-frozen NCC is reported after both global and role-local ablation; local targeted damage leaves final NCC unchanged under gold-trace teacher forcing", '<span class="status yes">aligned + masking diagnosed</span>'],
            ["Free-running final-state sufficiency", "Thinking answer-query donor patch adopts adjacent donor count on 66.1% of held-out pairs versus 14.3% same-count control; NT has margin movement but no categorical gain", '<span class="status yes">Thinking positive</span>'],
            ["Free-running progress-state sufficiency", "Same-position centroid shift moves the successor margin by +0.706 but changes 0/11 greedy successors; natural cross-position state changes 4/11 but is position-confounded", '<span class="status no">clean test negative</span>'],
            ["Universal final broad aggregator in Thinking", "Not tested as a required universal module; trace-to-answer state is behaviorally executable and is sufficient for the current mechanism claim", '<span class="status partial">not required</span>'],
            ["Serial mediation in one damaged baseline", "Top-K damage, carrier restoration, commit, and free rollout are not yet closed in one arm", '<span class="status no">open gap</span>'],
        ],
    )

    css = """
:root{--ink:#16202A;--muted:#52606D;--line:#D6DEE8;--paper:#FFFFFF;--wash:#F3F6F9;--blue:#2563A6;--orange:#D97706;--purple:#7158A6;--green:#23856D;--red:#B94444}
*{box-sizing:border-box} body{margin:0;background:#E9EEF3;color:var(--ink);font-family:Inter,"Segoe UI","Noto Sans SC",Arial,sans-serif;line-height:1.62}
main{max-width:1160px;margin:28px auto;background:var(--paper);padding:55px 66px 70px;box-shadow:0 12px 38px rgba(24,39,56,.12)}
h1{font-size:2.25rem;line-height:1.16;margin:0 0 12px;letter-spacing:-.025em} h2{font-size:1.55rem;margin:52px 0 14px;padding-top:10px;border-top:2px solid var(--ink)} h3{font-size:1.14rem;margin:30px 0 9px} h4{margin:22px 0 6px}
p{margin:9px 0 13px}.dek{font-size:1.08rem;color:var(--muted);max-width:920px}.meta{font-size:.88rem;color:var(--muted);margin-bottom:25px}.abstract,.conclusion,.warning,.example,.formula{padding:15px 18px;margin:16px 0;border-left:4px solid var(--blue);background:#F4F8FC}.conclusion{border-left-color:var(--green);background:#F2F8F6}.warning{border-left-color:var(--red);background:#FFF5F4}.example{border-left-color:var(--orange);background:#FFF8EC}.formula{border-left-color:var(--purple);background:#F7F5FB}.label{font-weight:750;margin-right:5px}
.toc{background:var(--wash);padding:17px 22px;border:1px solid var(--line);margin:24px 0}.toc ol{columns:2;margin:8px 0 0;padding-left:23px}.toc a{color:var(--blue);text-decoration:none}
.chain{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}.chain>div{padding:14px;border:1px solid var(--line);background:#FAFBFC}.chain b{display:block;color:var(--blue);margin-bottom:4px}
figure{margin:24px 0 31px}figure img{display:block;width:100%;height:auto;border:1px solid var(--line);background:white}figcaption{font-size:.9rem;color:#46525E;margin-top:8px;line-height:1.5}
.table-wrap{overflow-x:auto;margin:15px 0 24px;border:1px solid var(--line)}table{width:100%;border-collapse:collapse;font-size:.88rem}th{background:#EDF2F7;text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}td{padding:9px 10px;border-bottom:1px solid #E8EDF2;vertical-align:top}tr:last-child td{border-bottom:0}
code{font-family:"Cascadia Mono",Consolas,monospace;font-size:.88em;background:#EEF2F5;padding:2px 5px;border-radius:3px}.status{display:inline-block;padding:2px 7px;border-radius:12px;font-size:.78rem;font-weight:700;white-space:nowrap}.status.yes{color:#12664F;background:#DDF3EA}.status.partial{color:#855600;background:#FFF0C9}.status.no{color:#8F2F2F;background:#FBE2E2}
ul{padding-left:22px}.small{font-size:.86rem;color:var(--muted)}a{color:var(--blue)}
@media(max-width:800px){main{margin:0;padding:30px 20px}.chain{grid-template-columns:1fr}.toc ol{columns:1}h1{font-size:1.8rem}}
@media print{body{background:white}main{box-shadow:none;margin:0;max-width:none}figure{break-inside:avoid}}
"""

    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NiaH Synthetic v22: aligned geometry, causal experiments, and training dynamics</title><style>{css}</style></head>
<body><main>
<h1>NiaH Synthetic v22：No-index Thinking vs Matched Non-thinking</h1>
<p class="dek">Geometry Comparison、Top-K causal necessity、free-running causal sufficiency 与 Anthropic-style training dynamics 的统一对齐报告</p>
<p class="meta">生成日期：2026-08-28 · 主模型：v22 separator/no-index Thinking · 对照：matched v20 Non-thinking · count classes：1–30 · seed：1234</p>

<div class="abstract"><span class="label">核心结论。</span>v22 Thinking 相对 matched Non-thinking 的清楚优势出现在<strong>answer-query final-count compression</strong>（NCC {pct(th_final_ncc)} vs {pct(nt_final_ncc)}），而不在 running-index state（{pct(th_running_ncc)} vs {pct(nt_running_ncc)}，chance={pct(1/30)}）。Thinking 的 targeted Top-2 bank 是强且特异的：free-running exact trace 从 {pct(th_trace_clean)} 降到 {pct(th_trace_k2)}，matched K=2 control 仍为 {pct(th_trace_k2_control)}；最终答案从 {pct(th_fr_clean)} 降到 {pct(th_fr_k2)}，control 为 {pct(th_fr_k2_control)}。同时，Thinking 的 free-running trace 结束后，answer-query full-state transplant 可将相邻 donor count 的 greedy adoption 提高到 66.1%，证明 trace-to-answer state 可执行。相反，低维 running-centroid shift 没有改变 11 个 eligible continuation 中的任何一个。因此目前最稳健的论文主张是：<strong>Thinking 形成 concentrated targeted-retrieval bank，并把 trace 结果压缩为可执行 final-count state；不需要额外假设一个 universal final broad aggregator。</strong></div>

<div class="toc"><b>目录</b><ol><li><a href="#logic">问题与证据层级</a></li><li><a href="#setup">完整训练与比较设定</a></li><li><a href="#behavior">行为基线</a></li><li><a href="#geometry">Clean Geometry Comparison</a></li><li><a href="#topk">Top-K necessity 与 post-ablation NCC</a></li><li><a href="#sufficiency">Free-running causal sufficiency</a></li><li><a href="#dynamics">Training dynamics</a></li><li><a href="#alignment">与大模型实验对齐及 gaps</a></li><li><a href="#limits">结论、限制与下一步</a></li><li><a href="#artifacts">复现产物</a></li></ol></div>

<h2 id="logic">1. 问题与证据层级</h2>
<p>报告不把 probe、attention 与 causality 混成一个结论，而是按下列四层依次回答。Clean NCC 问“count 是否以紧致、可泛化的 cloud 存在”；Top-K ablation 问“被定位的 head bank 是否必要”；post-ablation frozen NCC 问“这条 bank 是否中介了下游 representation”；free-running patch 问“写入候选状态后能否真正改变后续生成”。</p>
<div class="chain"><div><b>1 · Represent</b>分别测 running index 与 final count；Thinking/Non-thinking 不交叉借标签。</div><div><b>2 · Retrieve</b>Non-thinking 测 broad coverage；Thinking 测第 k 个 trace query 对第 k 个 prompt occurrence 的 targeted mass。</div><div><b>3 · Necessity</b>按 discovery ranking 删除 Top-K，并与 layer/count-matched disjoint controls 比较。</div><div><b>4 · Sufficiency</b>在 free rollout 中 transplant answer state 或 progress state，看 greedy continuation 是否转向 donor。</div></div>
<div class="example"><span class="label">直观例子。</span>如果 L4 的状态能以 80% NCC 区分 count，只说明 count cloud 清楚；删掉 targeted heads 后 trace token 预测崩溃，才说明这些 heads 被自然使用；再把 count=8 的 answer state 写入 count=7 的 receiver 并让输出转向 8，才说明该 state 足以驱动答案。</div>
<div class="conclusion"><span class="label">本节结论。</span>Clean geometry、ablation necessity、representation mediation 与 free-running sufficiency 是不同问题，必须分别报告；任何单项都不能替代整条证据链。</div>

<h2 id="setup">2. 完整训练与比较设定</h2>
{setup_table}
<h3>2.1 为什么是 v22 Thinking 对 matched v20 Non-thinking</h3>
<p>v22 的注册设计只重新训练了 no-index Thinking；它没有另训一个 Non-thinking。两者的 standardized config 除 enabled variant、trace format 与版本标签外一致。训练与 held-out task examples 逐字段核验完全相同；corpus split 相同；needle pool 的字符内容相同。v22 的 vocabulary fingerprint 不同，是因为删掉了 trace 中的数字 index tokens，而不是因为换了任务。</p>
<p><code>Non-thinking: &lt;BOS&gt; query[5] data[256] &lt;Ans&gt; &lt;N&gt; &lt;EOS&gt;</code><br><code>Thinking v22: &lt;BOS&gt; query[5] data[256] &lt;Think&gt; (&lt;Sep&gt; marker)<sup>N</sup> &lt;/Think&gt; &lt;Ans&gt; &lt;N&gt; &lt;EOS&gt;</code></p>
<div class="warning"><span class="label">可比性边界。</span>这不是一个“同一 run 同时训练两种 mode”的 paired seed；它是两个相同架构、相同 task rows、相同 seed 与 schedule 的独立模型。可以比较功能 profile 与 intervention direction，不能把 LxHy identity 当成共享神经元。</div>
<div class="conclusion"><span class="label">本节结论。</span>v22 是当前最合适的主 setting：它去除了显式数字 running index，同时保留固定 separator scaffold；matched v20 Non-thinking 是正确对照，v23 weight=8 不进入主比较。</div>

<h2 id="behavior">3. 行为基线</h2>
{html_table(["Model", "Examples", "Free-running final accuracy", "95% Wilson interval", "Trace exact", "Ordered-marker accuracy"], [["matched v20 Non-thinking", int(v20_nt_ar['examples']), pct(v20_nt_ar['ar_final_accuracy']), f"[{pct(v20_nt_ar['ar_final_accuracy_wilson95_low'])}, {pct(v20_nt_ar['ar_final_accuracy_wilson95_high'])}]", "—", "—"], ["v22 no-index Thinking", int(v22_ar['examples']), pct(v22_ar['ar_final_accuracy']), f"[{pct(v22_ar['ar_final_accuracy_wilson95_low'])}, {pct(v22_ar['ar_final_accuracy_wilson95_high'])}]", pct(v22_ar['trace_exact']), pct(v22_ar['trace_ordered_marker_accuracy'])]])}
<p>官方 final evaluation 每个 count 有 50 个 held-out examples，共 1,500 条。v22 Thinking 的 final accuracy 为 {pct(v22_ar['ar_final_accuracy'])}，高于 Non-thinking 的 {pct(v20_nt_ar['ar_final_accuracy'])}，但低于旧 v20 indexed Thinking 的 91.2%。这正是 no-index control 的价值：移除显式数字 index 后性能下降，却没有消除 targeted retrieval 或 final-state compression。</p>
<div class="example"><span class="label">例子。</span>v20 indexed trace 类似 “1:a, 2:b, 3:a”；v22 只输出 “&lt;Sep&gt;a &lt;Sep&gt;b &lt;Sep&gt;a”。后者必须从 separator 次数、prefix 状态与已输出 marker 自行维持进度，不能直接复制 gold 数字 index。</div>
<div class="conclusion"><span class="label">本节结论。</span>v22 保留了 Thinking 的行为优势，但它是更困难且更接近 natural no-index reasoning 的版本；后续机制分析解释的是 59.6% 的真实 free-running system，而不是旧 indexed 模型的 91.2%。</div>

<h2 id="geometry">4. Clean Geometry Comparison：running 与 final 分开</h2>
<h3>4.1 目的与计算方法</h3>
<p>每个 endpoint 用 discovery 10 states/class（300 rows）与 disjoint confirmation 8/class（240 rows）。每层只在 discovery 中做 grouped 5-fold selection，criterion 为 mean(Logistic BA, NCC BA)；选层后用全部 discovery 拟合 <code>StandardScaler → whitened PCA≤16</code>、L2 logistic 与 class centroids，并冻结到 confirmation。</p>
<div class="formula"><span class="label">Nearest-centroid classifier (NCC)。</span>对 discovery 中 count class <i>c</i> 的投影状态求 centroid μ<sub>c</sub>；held-out state z 的预测为 argmin<sub>c</sub>‖z−μ<sub>c</sub>‖²。Balanced accuracy 是 30 个 class recall 的平均，chance=1/30=3.33%。它同时受 within-class scatter 和 between-class separation 影响，因此本报告称它为“紧致度/可分性 operational readout”，不把它解释为纯 cluster radius。</div>
{figure(figures['clean'], "图 1｜Clean geometry。横轴依次为两种 mode 的 running endpoint 与 final answer-query endpoint；纵轴为从未参与层选择的 confirmation balanced accuracy。实心柱是 L2 logistic，斜线空心柱是 NCC；水平点线是 30-class chance。柱顶 Lx 为 discovery-only 选出的 residual layer（L0 embedding，L1–L4 block outputs）。", "clean running-index and final-count NCC comparison")}
{geometry_table}
<p>正确比较不是 v22 vs v23，而是 v22 Thinking vs matched Non-thinking，并且要分别回答两个标签。Running-index NCC 只从 {pct(nt_running_ncc)} 增到 {pct(th_running_ncc)}，两者都很弱；Thinking 的显著优势在 answer query：NCC 从 {pct(nt_final_ncc)} 增到 {pct(th_final_ncc)}（+{100*(th_final_ncc-nt_final_ncc):.1f} percentage points）。Thinking running endpoint 的 ordinal RSA 较高，说明 class centroids 随 count gap 有序，但低 NCC 表明单例 clouds 并不紧致；“平均轨迹有序”不等于“autonomous compact counter”。</p>
<div class="example"><span class="label">例子。</span>如果 count=8 与 count=9 的平均 centroid 按顺序排列，RSA 可以很高；但每个 count 的八个 held-out states 若互相重叠，NCC 仍会接近 chance。本结果正是这种“有序平均路径、弱单例分类”的情形。</div>
<div class="conclusion"><span class="label">本节结论。</span>v22 没有复现大模型中清楚的 natural running-index manifold；它复现的是 Thinking 在 answer query 形成更紧致 final-count state。后文的 causal test 因而应重点检验 targeted retrieval 与 trace-to-answer readout，而不能先验声称存在 compact running counter。</div>

<h2 id="topk">5. Top-K causal necessity 与 post-ablation NCC</h2>
<h3>5.1 目的、ranking 与 controls</h3>
<p>Non-thinking broad score 定义为每个 answer query 对全部 target occurrences 的总 mass <i>M</i>，乘以其归一化 occurrence distribution 的 entropy effective number exp(H)，再除以 target 数 N：<code>broad=M·exp(H)/N</code>。它同时奖励“看得多”和“覆盖得广”。Thinking targeted score 是每个第 k 个 separator query 指向 prompt 第 k 个 target occurrence 的 attention mass。所有 ranking 都只用独立 heldout head-selection rows。</p>
<p>最终 bank 为：Non-thinking broad Top-4 = <code>{top_nt_heads}</code>；Thinking targeted Top-4 = <code>{top_th_heads}</code>。K=1/2 的 control 穷举不与 ranked heads 重叠、且每层 head 数相同的集合；4×4 inventory 下 K=4 通常没有 disjoint matched set，因此只作为 dose endpoint。</p>
<div class="formula"><span class="label">Post-ablation frozen NCC。</span>Clean discovery 决定 layer、scaler、PCA 与 centroids；只对 confirmation forward 删除 heads，并用 clean fit 直接分类。于是 ΔNCC 问的是“删除 bank 后，held-out state 是否离开原来的 clean count geometry”，不是“受损状态能否重新训练一个新 decoder”。</div>
{figure(figures['tf_topk'], "图 2｜Role-query-local Top-K。横轴 K 是累计删除的 discovery-ranked heads；纵轴分别是 clean-frozen NCC 或 teacher-forced token accuracy。实线为 ranked bank，灰色虚线为 layer-matched control 的均值，误差线是所有可用 disjoint controls 的 min–max（不是置信区间）。A/B 显示 Non-thinking broad removal 会降低 final geometry/答案，但 K=2 specificity 较弱；C 显示 Thinking targeted removal 对 trace marker 有巨大、特异损害；D 显示 gold trace teacher forcing 会使 answer-query NCC 恢复。", "teacher-forced Top-K ablation and frozen NCC")}
{topk_table}
<h3>5.2 为什么 Thinking 的 final NCC 在 local Top-K 后不降</h3>
<p>这不是“targeted heads 不重要”。Role-local intervention 只在 separator query 破坏 marker retrieval；teacher-forced forward 随即在下一个 position 输入正确 gold marker，因此后续 item-end state 和 answer query 又看到正确 trace。直接证据是：K=2 marker accuracy 从 99.62% 降到 59.41%，而 final NCC 仍为 79.17%。这是一种实验协议造成的 repair，不是机制没有 effect。</p>
{figure(figures['fr_topk'], "图 3｜同一 reporting rows 的 free-running Top-K necessity。横轴与 controls 同图 2，但生成从 Thinking 的 &lt;Think&gt; 或 Non-thinking 的 &lt;Ans&gt; 开始，之后不再提供 gold trace token。A/B 为最终答案 exact accuracy；C 为整条 separator-marker trace exact；D 为按 gold 顺序的 marker accuracy。Thinking ranked K=2 把 final accuracy 64.58%→27.92%、exact trace 93.33%→12.08%，而 K=2 matched control 分别为 61.25% 与 85.83%。", "free-running cumulative Top-K ablation")}
<p>Free-running 关闭了 teacher-forcing repair：Thinking K=2 ranked ablation 将 final accuracy 从 {pct(th_fr_clean)} 降到 {pct(th_fr_k2)}，比 matched K=2 的 {pct(th_fr_k2_control)} 明显更严重；exact trace 从 {pct(th_trace_clean)} 降到 {pct(th_trace_k2)}，matched control 为 {pct(th_trace_k2_control)}。Non-thinking K=2 final accuracy 35.0%→20.8%，但 matched K=2 也为 22.1%，说明它更像整个 L1 的 distributed broad bank，而不是两只 sharply unique heads。</p>
<div class="example"><span class="label">例子。</span>把一本书每页的正确页码强行写回，前一页导航器坏了也可能在最后一页读到正确总页数；让模型自己翻页时，导航错误会累积。图 2 的 Thinking final NCC 是前一种测量，图 3 是后一种。</div>
<div class="conclusion"><span class="label">本节结论。</span>Thinking targeted bank 有明确、matched-control-specific 的自然使用证据；Non-thinking broad retrieval 有剂量效应但 bank 分布在全部 L1 heads，K=2 ranking specificity 弱。Post-ablation NCC 应报告，但必须与 immediate token damage 和 free-running behavior 一起解释。</div>

<h2 id="sufficiency">6. Free-running causal sufficiency</h2>
<h3>6.1 Experiment A：answer-query state 是否足以改写最终答案</h3>
<p>对 receiver count N∈{{2,…,29}} 构造 N−1 与 N+1 donor，共 56 对。Non-thinking 在 <code>&lt;Ans&gt;</code> 开始；Thinking 先从 <code>&lt;Think&gt;</code> 自己生成 trace，直到第一次产生 <code>&lt;Ans&gt;</code>，再 patch answer-query full residual。Discovery 按 donor-margin shift 与 donor adoption 选层（NT L{selected_suff['answer_transplant']['nonthinking']}，Thinking L{selected_suff['answer_transplant']['thinking']}），confirmation 固定层。Same-count context donor 控制 context identity，self patch 检查 intervention plumbing。</p>
{figure(figures['answer'], "图 4｜Free-running answer-state transplant。每个 panel 的横轴是 clean、same-count context control 与 adjacent-count donor；上排纵轴是 greedy 输出 donor count 的比例，下排是 donor-minus-receiver logit margin 相对 clean 的变化。Thinking 的相邻 count donor 在生成自己的 trace 后仍产生 66.1% donor adoption 与 +8.36 margin shift；same-count control 仅 14.3% / +0.033。", "free-running answer-query residual sufficiency")}
{answer_table}
<p>Thinking 有强 categorical sufficiency：adjacent donor adoption=66.07%，clean=12.50%，same-count control=14.29%。Non-thinking 的连续 margin shift为 +0.585（control +0.028），但 donor adoption 28.57% 不高于 clean 的 30.36%，因此只能称“局部候选 margin 被移动”，不能称 categorical free-running transfer。</p>
<div class="warning"><span class="label">边界。</span>最终答案是 atomic single token；Thinking 的 trace 确实是 free-running，但 patch 后的 answer decision 只有一步，没有 multi-token self-correction。这个实验支持 executable trace-to-answer state，不证明该 state 是唯一 mediator。</div>
<div class="conclusion"><span class="label">Experiment A 结论。</span>v22 Thinking 的 answer-query state 对最终 count 具有 held-out、control-specific、free-running sufficiency；Non-thinking 在同协议下只有连续 logit effect，没有 categorical sufficiency。</div>

<h3>6.2 Experiment B：running/progress state 是否足以改写下一步 continuation</h3>
<p>固定 N=10 prompts，在第 k∈{{4,6,8}} 个 trace item 后暂停。用 discovery count centroids 构造同一 position 的 Δ<sub>k→k±1</sub>，写入 receiver L1 marker state后恢复 greedy generation；control 是与 centroid span 正交且 realized norm 相同的方向。另做 natural donor marker 与完整 two-token item-span patch，作为上界。</p>
<p>Eligible cell 要求 donor successor marker 与 natural successor marker 不同，否则“adoption”不可识别。三字符 target set 使大量相邻 successor 相同，confirmation 最终只有 11 cells；这个实验统计功效很低。</p>
{figure(figures['progress'], "图 5｜Free-running progress-state sufficiency。横轴是同位置 centroid shift、等范数正交 control，以及两个 natural cross-position donor；A 的纵轴是 donor successor 相对 natural successor 的 logit margin shift，B 是 greedy donor-successor adoption。斜线柱是位置混杂的 natural upper bound。Centroid shift 比正交 control 多移动约 +0.528 logit，但两者均为 0/11 behavioral adoption；natural state 为 4/11。", "free-running progress state sufficiency")}
{progress_table}
<p>Centroid shift 的 route margin 从 clean −8.043 改为 −7.337（+0.706），orthogonal control 为 −7.865（+0.179），但两者都保持 100% natural successor、0% donor adoption。Natural marker/span 都达到 36.36% adoption，但 fixed two-token grammar 使 donor k 与 receiver k 不能位于相同 absolute position，因此混入 RoPE position 与具体 marker content，不能作为 clean natural-counter proof。所有 11 个 eligible cells 的 final answer 都不正确，所以 progress patch 也没有救回最终 count。</p>
<div class="example"><span class="label">例子。</span>在生成第 6 个 marker 后，把 hidden state 沿“centroid 6→7”的方向移动。如果模型把这条低维方向当作真实进度，下一 marker 应更像第 8 个；实际 margin 略微朝 donor 移动，但 greedy token 从未改变。</div>
<div class="conclusion"><span class="label">Experiment B 结论。</span>当前证据否定“简单低维 centroid direction 已足以控制 v22 progress rollout”。完整 natural state 有部分上界效应，但位置混杂与 n=11 使其只能保留为 qualified result。这与 clean running NCC 很弱是一致的。</div>

<h2 id="dynamics">7. Training dynamics：role specialization 与 head-bank differentiation</h2>
<h3>7.1 为什么主图用 linear optimizer steps</h3>
<p>主横轴用 0–10,000 的线性 optimizer step，因为目标是比较 behavior、attention role 与 causal damage 的出现时序，并准确显示 step 1,500 objective switch。<code>log(1+step)</code> 只会展开前期、压缩后期；它可以作为 early-emergence 辅助图，但不会让一个平滑变化自动变成“突变”。Scaling-law 工作常用 log-log 是因为拟合跨数量级的 power law；这里是一条固定训练 run 的机制形成问题。</p>
{figure(figures['dynamics'], "图 6｜对齐后的 mode-specific training dynamics。所有横轴是线性 optimizer step，竖点线是 step 1,500 loss-scope switch。A 为 count 1–30 的 macro teacher-forced final accuracy（每 count 的 dense probe 很小，曲线只作 descriptive timing）；B/C 分别画 matched Non-thinking broad score 与 v22 Thinking targeted correct-occurrence mass 的 16-head trajectories，粗线是 final heldout ranking 的 bank；D 左轴为 entropy effective head number，右轴虚线为 Top-2 role-mass share。两个 role score 的绝对单位不同，只比较各自 specialization profile。", "aligned broad versus targeted head bank training dynamics")}
<p>到 step 10,000，Non-thinking broad bank 的 effective head number 仍为 {nt_final_stats['effective_heads']:.2f}、Top-2 share={nt_final_stats['top2_share']:.3f}，且 Top-4 全部落在 L1；Thinking targeted bank 则收缩到 {th_final_stats['effective_heads']:.2f} effective heads、Top-2 share={th_final_stats['top2_share']:.3f}，主要是 L4H2/L4H1。换言之，broad bank 是 layer-local but head-distributed，targeted bank 是 late-layer and head-concentrated。</p>
{figure(figures['v22_roles'], "图 7｜v22 内部两个不同 role 的 fixed-head emergence。左图 L4H2 targeted retrieval 在约 step 2,900 超过 0.1，之后缓慢上升；右图 L2H1 marker-successor/control-flow preference 在 step 400 已超过 0.5。纵轴均为各自 role score，不能把 0.5 跨 panel 当成相同 attention event。", "v22 targeted retrieval and marker successor emergence")}
{figure(figures['v22_causal'], "图 8｜v22 role-local causal effect 随训练变化。横轴为预注册 milestones；纵轴是删除 head slice 后 correct-token logit margin 相对 baseline 的变化，越负表示损害越大。蓝线为 final fixed role head，橙线为同层 control。Targeted L4H2 的 ranking signal约在 2.9k 后出现，control-specific causal damage 在 3.5k–5k 后才清楚扩大；marker-successor head 更早形成并持续必要。", "v22 causal role emergence over training")}
<p>这个次序与 Anthropic induction-head 风格的分析逻辑一致：先把 macro behavior、head score 和 ablation effect 放在同一训练坐标上，再谈 emergence。这里最清楚的是<strong>早期 control-flow head → 后期 targeted retrieval concentration → 更晚且逐步增强的 causal use</strong>；数据不像单一步骤的突然 phase transition，因此不应仅靠 log-x 视觉宣称“突变”。</p>
<div class="conclusion"><span class="label">本节结论。</span>训练过程中确实出现 role specialization 与 head-bank differentiation：v22 marker-successor 早成型，targeted L4 bank 约 2.9k 后持续集中并获得越来越强的局部因果作用；matched Non-thinking broad role 最终仍分散在 L1 全 bank。</div>

<h2 id="alignment">8. 与大模型实验的对齐及 gaps</h2>
{alignment_table}
<h3>8.1 是否必须证明 universal final aggregator？</h3>
<p>不需要。论文若主张的是 broad retrieval vs targeted retrieval 的计算差异，Thinking 侧只需要证明：targeted bank 自然参与逐项 retrieval，trace-derived state 能传到 answer query，并且该 answer state 可执行。v22 已有前两者的强证据：Top-K free-running necessity 与 answer-query donor sufficiency。除非正文明确主张“所有 Thinking 模型都必须在最后位置用 prompt-wide broad heads 再聚合一次”，否则没有必要把一个未观察到的 universal final aggregator 当作成功标准。</p>
<p>更准确的机制句是：<strong>Thinking uses a concentrated targeted-retrieval bank to construct a trace; the resulting trace-to-answer state is compressed and behaviorally executable.</strong> 这比“Thinking 最后也有 broad aggregator”更贴合现有证据。</p>
<h3>8.2 当前相对大模型最重要的 gaps</h3>
<ul><li><b>Running representation gap：</b>大模型能看到更清楚的 natural running-index structure；v22 item-end NCC 只有 {pct(th_running_ncc)}。</li><li><b>Progress sufficiency gap：</b>大模型 natural no-index state transplant 可改写 continuation；v22 同位置 centroid shift 是 negative，natural donor 又有 absolute-position confound。</li><li><b>Non-thinking gap：</b>synthetic prompt running-index NCC 也很弱；final count 可读但不紧致，answer state categorical transplant 不成立。</li><li><b>Serial mediation gap：</b>还没有在同一 Top-K-damaged free rollout 中依次 restore carrier、commit 与 final answer，因而不能声称 targeted bank→carrier→commit→answer 是完整唯一 circuit。</li><li><b>Scale/tokenization gap：</b>3.19M、4×4 heads、atomic answers、256-char context，与 4B/8B natural-language model 的绝对 accuracy、head 数和 formation step 不可直接比较。</li></ul>
<div class="conclusion"><span class="label">本节结论。</span>Synthetic 已与大模型对齐到 representation/necessity/sufficiency 的实验层级，并复现 targeted retrieval 与 executable final state；没有复现 compact natural running counter，也尚未闭合完整 serial mediation。应如实报告这些正负结果，而不是换一个更好看的 setting。</div>

<h2 id="limits">9. 最终结论、限制与下一步</h2>
<div class="abstract"><span class="label">可以写进正文的结论。</span>(1) v22 no-index Thinking 相对 matched Non-thinking 的主要 representation advantage 是 final-count compression，不是 running-index NCC；(2) Thinking targeted bank 比 matched controls 更集中、更必要，且其 free-running removal 同时破坏 trace 与最终答案；(3) Thinking free-generated trace 后的 answer state 可被 donor-transfer，支持 trace-to-answer executable readout；(4) Non-thinking broad retrieval 是 distributed L1 bank，存在剂量效应但 Top-2 specificity 弱；(5) low-dimensional progress centroid 目前不具备 free-running behavioral sufficiency。</div>
<p><b>必须同时写出的限制：</b></p>
<ul><li>所有结论来自单一 training seed 1234；token-level observations 不是独立 training replicates。</li><li>v22 Thinking 与 v20 Non-thinking 是匹配的独立模型，不是一个 checkpoint 的 mode toggle。</li><li>Top-K K=4 无 disjoint matched control；K=1 control pool包含第二 ranked targeted head，因此 K=2 是更干净的 bank-specific contrast。</li><li>Progress confirmation 只有 11 eligible cells；自然 state patch跨 absolute positions，不能排除 RoPE/position content。</li><li>Answer transplant 是 atomic one-token decision；充分性不等于唯一性。</li><li>Clean-frozen NCC 是 mediator readout；teacher-forced downstream recovery不能替代 free-running evaluation。</li></ul>
<p><b>下一步优先级：</b>(1) 3–5 additional seeds，预注册同一层/heads selection rule；(2) 训练 padded separator 或 variable-gap no-index trace，使不同 k 的 donor/receiver marker 可在同一 absolute position 对齐；(3) 在 ranked Top-K damaged free rollout 中逐层 patch targeted-head output、marker carrier、answer state，做完整 serial mediation；(4) 增加 multi-token answer 与超出训练 count/context length 的 OOD evaluation；(5) 如要检验 Thinking broad readout，仅将其作为 competing terminal pathway，而非先验 universal requirement。</p>
<div class="conclusion"><span class="label">最终判断。</span>保留 v22 是正确选择。它虽然比 indexed v20 更难、accuracy 更低，却给出更有价值的机制分解：强 targeted-bank necessity + 强 final-state sufficiency + 弱 running-centroid sufficiency。这个组合比只汇报高 accuracy 更接近我们真正要区分的 broad retrieval 与 targeted retrieval。</div>

<h2 id="artifacts">10. 复现产物</h2>
{html_table(["Artifact", "Path / role"], [["Clean comparison runner", html.escape(str((ROOT/'scripts/compare_v22_modes_ncc.py').resolve()))], ["Post-ablation NCC runner", html.escape(str((ROOT/'scripts/run_v22_topk_ncc.py').resolve()))], ["Free-running Top-K runner", html.escape(str((ROOT/'scripts/run_v22_free_running_topk.py').resolve()))], ["Free-running sufficiency runner", html.escape(str((ROOT/'scripts/run_v22_free_running_sufficiency.py').resolve()))], ["Report builder", html.escape(str((ROOT/'scripts/build_v22_synthetic_report.py').resolve()))], ["Clean NCC tables", html.escape(str(clean_root.resolve()))], ["Top-K tables", html.escape(str(topk_root.resolve()))], ["Free-running Top-K tables", html.escape(str(fr_topk_root.resolve()))], ["Sufficiency tables", html.escape(str(suff_root.resolve()))], ["v22 archived dynamics subset", html.escape(str(dyn_root.resolve()))], ["Final self-contained report", html.escape(str(output.resolve()))]])}
<p class="small">所有图均由表格重新生成或从 v22 archived phase-transition artifacts 内嵌为 base64；最终 HTML 不依赖外部图片路径。旧的 v20-only report 叙述已由本版本取代；v20 indexed causal results 只作为历史辅助，不再冒充 v22 no-index evidence。</p>
</main></body></html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    manifest = {
        "report": str(output.resolve()),
        "primary_comparison": "v22 separator/no-index Thinking vs matched v20 Non-thinking",
        "figures": {key: str(value.resolve()) for key, value in figures.items()},
        "source_roots": {
            "clean_ncc": str(clean_root.resolve()),
            "topk_ncc": str(topk_root.resolve()),
            "free_running_topk": str(fr_topk_root.resolve()),
            "free_running_sufficiency": str(suff_root.resolve()),
            "v22_training_dynamics": str(dyn_root.resolve()),
        },
    }
    (output.parent / "NiaH_Synthetic_report_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "NiaH_Synthetic_report.html",
    )
    args = parser.parse_args()
    build_report(args.output.resolve())


if __name__ == "__main__":
    main()
