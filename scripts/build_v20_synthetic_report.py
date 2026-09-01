#!/usr/bin/env python
"""Build the self-contained synthetic-v20 geometry/causal/dynamics report."""

from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "colab_results" / "v20_main_RoPE_count1-30_seed1234"

COLORS = {
    "nonthinking_prompt_occurrence": "#4C78A8",
    "nonthinking_answer_query": "#1F4E79",
    "thinking_item_end": "#7B61A8",
    "thinking_answer_query": "#F28E2B",
    "targeted": "#7B61A8",
    "successor": "#2A9D8F",
    "control": "#7A7A7A",
    "danger": "#D95F5F",
}

ENDPOINT_SHORT = {
    "nonthinking_prompt_occurrence": "NT prompt occurrence",
    "nonthinking_answer_query": "NT answer query",
    "thinking_item_end": "Thinking item end",
    "thinking_answer_query": "Thinking answer query",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def plot_geometry_layerwise(metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, endpoint in zip(axes.flat, ENDPOINT_SHORT, strict=True):
        frame = metrics.loc[metrics["endpoint"].eq(endpoint)].sort_values("layer")
        ax.plot(
            frame["layer"],
            frame["confirmation_logistic_balanced_accuracy"],
            marker="o",
            linewidth=2.2,
            color=COLORS[endpoint],
            label="Logistic BA",
        )
        ax.plot(
            frame["layer"],
            frame["confirmation_ncc_balanced_accuracy"],
            marker="s",
            linewidth=1.8,
            linestyle="--",
            color=COLORS[endpoint],
            alpha=0.72,
            label="Nearest-centroid BA",
        )
        ax.axhline(1 / 30, color="#777777", linestyle=":", linewidth=1.2, label="Chance 1/30")
        winner = frame.sort_values(
            ["discovery_selection_score", "layer"], ascending=[False, True]
        ).iloc[0]
        ax.axvline(int(winner["layer"]), color="#222222", linestyle=":", linewidth=1)
        ax.set_title(f"{ENDPOINT_SHORT[endpoint]} · selected L{int(winner['layer'])}")
        ax.set_xticks(range(5))
        ax.set_ylim(0, 1.04)
        _style_axis(ax)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Residual layer (0 = token embedding)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Confirmation balanced accuracy")
    fig.suptitle("Layerwise held-out count decodability (discovery-selected protocol)", fontsize=15)
    _savefig(fig, path)


def _common_selected(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, frame in metrics.groupby("endpoint", sort=False):
        selected = frame.sort_values(
            ["discovery_selection_score", "layer"], ascending=[False, True]
        ).iloc[0]
        rows.append(selected)
    return pd.DataFrame(rows)


def plot_geometry_metrics(metrics: pd.DataFrame, path: Path) -> None:
    selected = _common_selected(metrics)
    order = list(ENDPOINT_SHORT)
    selected = selected.set_index("endpoint").loc[order].reset_index()
    panels = [
        ("confirmation_logistic_balanced_accuracy", "Logistic balanced accuracy", lambda x: x),
        ("confirmation_ncc_balanced_accuracy", "Nearest-centroid balanced accuracy", lambda x: x),
        ("confirmation_isotropic_snr_db", "Isotropic SNR (dB)", lambda x: x),
        ("confirmation_fisher_trace_frozen", "log10(1 + frozen Fisher trace)", lambda x: np.log10(1 + x)),
        ("confirmation_mahalanobis_silhouette", "Mahalanobis silhouette", lambda x: x),
        ("confirmation_ordinal_rsa", "Ordinal RSA (Spearman rho)", lambda x: x),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = np.arange(len(order))
    colors = [COLORS[item] for item in order]
    for ax, (column, title, transform) in zip(axes.flat, panels, strict=True):
        values = transform(selected[column].to_numpy(dtype=float))
        bars = ax.bar(x, values, color=colors, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(x, [ENDPOINT_SHORT[item] for item in order], rotation=20, ha="right")
        ax.tick_params(axis="x", labelsize=8)
        _style_axis(ax)
        for bar, value in zip(bars, values, strict=True):
            if np.isfinite(value):
                offset = 0.02 * max(1.0, float(np.nanmax(np.abs(values))))
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (offset if value >= 0 else -offset),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                )
    fig.subplots_adjust(top=0.88, hspace=0.58, wspace=0.24)
    fig.suptitle("Confirmation geometry at each endpoint's common discovery-selected layer", fontsize=15)
    _savefig(fig, path)


def plot_geometry_pca(coordinates: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    scatter = None
    for ax, endpoint in zip(axes.flat, ENDPOINT_SHORT, strict=True):
        frame = coordinates.loc[coordinates["endpoint"].eq(endpoint)]
        scatter = ax.scatter(
            frame["pc1"],
            frame["pc2"],
            c=frame["occurrence"],
            cmap="viridis",
            s=16,
            alpha=0.48,
            linewidths=0,
            vmin=1,
            vmax=30,
        )
        centroids = frame.groupby("occurrence")[["pc1", "pc2"]].mean().reset_index()
        ax.plot(centroids["pc1"], centroids["pc2"], color="#222222", linewidth=1.2, alpha=0.72)
        ax.scatter(centroids["pc1"], centroids["pc2"], c=centroids["occurrence"], cmap="viridis", s=28, vmin=1, vmax=30, edgecolors="#222222", linewidths=0.35)
        layer = int(frame["selected_layer"].iloc[0])
        var1 = float(frame["pc1_variance_ratio"].iloc[0]) * 100
        var2 = float(frame["pc2_variance_ratio"].iloc[0]) * 100
        ax.set_title(f"{ENDPOINT_SHORT[endpoint]} · L{layer}")
        ax.set_xlabel(f"Discovery-fitted PC1 ({var1:.1f}% variance)")
        ax.set_ylabel(f"Discovery-fitted PC2 ({var2:.1f}% variance)")
        _style_axis(ax)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.84, pad=0.02)
        cbar.set_label("Count / occurrence label (1–30)")
    fig.suptitle("Confirmation states in discovery-fitted PCA coordinates", fontsize=15)
    _savefig(fig, path)


def plot_local_head_ablation(frame: pd.DataFrame, path: Path) -> None:
    roles = ["nonthinking_broad", "thinking_targeted", "thinking_readout"]
    titles = ["Non-thinking broad bank", "Thinking targeted bank", "Thinking trace readout bank"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    for ax, role, title in zip(axes, roles, titles, strict=True):
        data = frame.loc[frame["role"].eq(role) & frame["top_n"].isin([0, 1, 2, 4])]
        ranked = data.loc[data["path_kind"].eq("ranked")].groupby("top_n")["margin_drop"].mean()
        random = data.loc[data["path_kind"].eq("random")].groupby("top_n")["margin_drop"].agg(["mean", "sem"])
        xs = np.asarray(ranked.index, dtype=float)
        ax.plot(xs, ranked.values, marker="o", linewidth=2.2, color=COLORS["targeted"], label="Discovery-ranked bank")
        ax.plot(random.index, random["mean"], marker="s", linestyle="--", color=COLORS["control"], label="Matched random banks")
        ax.fill_between(random.index, random["mean"] - random["sem"].fillna(0), random["mean"] + random["sem"].fillna(0), color=COLORS["control"], alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("Number of locally ablated heads (Top-K)")
        ax.set_ylabel("Drop in correct-token logit margin")
        ax.set_xticks([0, 1, 2, 4])
        _style_axis(ax)
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Position-local head-bank necessity: ranked versus matched-random ablation", fontsize=15)
    _savefig(fig, path)


def plot_retrieval_transport(mediation: pd.DataFrame, routing: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    left = axes[0]
    interventions = ["attention_pattern_only", "value_only_at_target_source", "pattern_plus_value"]
    labels = ["Pattern only", "Target-source value only", "Pattern + value"]
    width = 0.36
    x = np.arange(len(interventions))
    for offset, top_n, color in [(-width / 2, 1, "#9C89B8"), (width / 2, 2, COLORS["targeted"])]:
        values = [
            mediation.loc[(mediation["intervention"].eq(intervention)) & (mediation["top_n"].eq(top_n)), "normalized_recovery"].mean()
            for intervention in interventions
        ]
        left.bar(x + offset, values, width, label=f"Top-{top_n}", color=color)
    residual = mediation.loc[mediation["intervention"].eq("residual_stream"), "normalized_recovery"].mean()
    left.axhline(residual, color="#222222", linestyle=":", linewidth=1.3, label=f"Residual patch = {residual:.2f}")
    left.set_xticks(x, labels, rotation=18, ha="right")
    left.set_ylabel("Normalized recovery of identity margin")
    left.set_title("Localization versus value transport")
    left.set_ylim(-0.08, 1.12)
    left.legend(fontsize=8)
    _style_axis(left)

    right = axes[1]
    summary = routing.groupby(["layer", "head"])["routing_shift"].agg(["mean", "sem"]).reset_index()
    labels = [f"L{int(row.layer)}H{int(row.head)}" for row in summary.itertuples()]
    right.bar(labels, summary["mean"], yerr=summary["sem"], capsize=3, color=[COLORS["targeted"] if int(row.head) in (1, 2) else "#B8B8B8" for row in summary.itertuples()])
    right.axhline(0, color="#222222", linewidth=0.8)
    right.set_ylabel("Attention shift: donor occurrence minus receiver occurrence")
    right.set_title("Progress-state patch redirects target routing")
    _style_axis(right)
    fig.suptitle("Thinking targeted retrieval: identity transport and state-conditioned routing", fontsize=15)
    _savefig(fig, path)


def plot_evidence_progress(prompt: pd.DataFrame, scope: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    prompt_order = [
        (-1, "corrupt_baseline", "Deleted\n(no patch)"),
        (0, "ordinary_embedding_at_deleted_target_control", "Ordinary\nembedding"),
        (0, "target_embedding_at_deleted_target", "Target @\ndeleted site"),
        (0, "target_embedding_at_ordinary_location_control", "Target @\nnew location"),
        (1, "target_state_at_deleted_target", "Target state\npost-L1"),
    ]
    medians = []
    accuracies = []
    for layer, intervention, _ in prompt_order:
        values = prompt.loc[prompt["layer"].eq(layer) & prompt["intervention"].eq(intervention)]
        medians.append(values["normalized_recovery"].median())
        accuracies.append(values["patched_correct_pairwise"].mean())
    x = np.arange(len(prompt_order))
    axes[0].bar(x, medians, color=["#B8B8B8", "#B8B8B8", COLORS["nonthinking_prompt_occurrence"], "#7AA6C2", "#B8B8B8"])
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].axhline(1, color="#222222", linewidth=0.8, linestyle=":")
    axes[0].set_xticks(x, [item[2] for item in prompt_order], rotation=0, ha="center")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Median normalized recovery")
    axes[0].set_title("Non-thinking: evidence identity and patch timing")
    _style_axis(axes[0])
    for index, accuracy in enumerate(accuracies):
        axes[0].text(index, medians[index] + 0.04, f"acc {accuracy:.2f}", ha="center", fontsize=8)

    terminal = scope.loc[scope["donor_kind"].eq("terminal_total_equals_k") & scope["layer"].gt(0)]
    styles = {
        "index_only": ("Index only", "--", "#7A7A7A"),
        "marker_only": ("Marker only", "-", COLORS["successor"]),
        "index_plus_marker": ("Index + marker", "-", COLORS["targeted"]),
    }
    for scope_name, (label, linestyle, color) in styles.items():
        data = terminal.loc[terminal["scope"].eq(scope_name)].groupby("layer")["close_margin_shift"].mean()
        axes[1].plot(data.index, data.values, marker="o", linestyle=linestyle, color=color, linewidth=2, label=label)
    control = scope.loc[scope["donor_kind"].eq("continuing_same_total_control")].groupby("layer")["close_margin_shift"].mean()
    axes[1].plot(control.index, control.values, marker="s", linestyle=":", color=COLORS["control"], label="Continuing item control")
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_xlabel("Patched residual layer")
    axes[1].set_ylabel("Shift in close minus continue logit margin")
    axes[1].set_title("Thinking: terminal progress scope")
    axes[1].legend(fontsize=8)
    _style_axis(axes[1])
    fig.suptitle("Early evidence intake versus late progress execution", fontsize=15)
    _savefig(fig, path)


def plot_terminal_readout(residual: pd.DataFrame, bridge: pd.DataFrame, conflicts: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    residual = residual.copy()
    residual["signed_shift"] = residual["expected_count_shift"] * residual["offset"]
    for mode, color, label in [("nonthinking", COLORS["nonthinking_answer_query"], "Non-thinking"), ("thinking", COLORS["thinking_answer_query"], "Thinking")]:
        data = residual.loc[residual["mode"].eq(mode) & residual["intervention"].eq("natural_donor")].groupby("layer")["signed_shift"].mean()
        axes[0].plot(data.index, data.values, marker="o", linewidth=2.1, color=color, label=label)
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].set_xlabel("Answer-query residual layer")
    axes[0].set_ylabel("Signed expected-count shift toward donor")
    axes[0].set_title("Answer-state donor transport")
    axes[0].legend(fontsize=8)
    _style_axis(axes[0])

    styles = {
        "attention_output": ("Attention output", COLORS["targeted"], "-"),
        "mlp_output": ("MLP output", COLORS["thinking_answer_query"], "--"),
        "post_layer_residual": ("Post-layer residual", COLORS["successor"], "-"),
    }
    for component, (label, color, linestyle) in styles.items():
        data = bridge.loc[bridge["component"].eq(component)].groupby("layer")["normalized_recovery"].mean()
        axes[1].plot(data.index, data.values, marker="o", color=color, linestyle=linestyle, linewidth=2, label=label)
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_xlabel("Patched component layer")
    axes[1].set_ylabel("Normalized answer-margin recovery")
    axes[1].set_title("Thinking final bridge rescue")
    axes[1].legend(fontsize=8)
    _style_axis(axes[1])

    conflict_order = [
        "prompt_minus_one_trace_clean",
        "trace_index_minus_one",
        "trace_pair_copy_previous",
        "marker_identity_control",
        "trace_tail_neutral_control",
        "shortened_trace_position_shifted",
    ]
    conflict_labels = ["Prompt −1", "Final index −1", "Copy previous pair", "Marker identity", "Neutral tail", "Shortened trace"]
    values = [conflicts.loc[conflicts["intervention"].eq(name), "follows_original_n"].mean() for name in conflict_order]
    axes[2].bar(np.arange(len(values)), values, color=["#7AA6C2"] * 5 + [COLORS["danger"]])
    axes[2].set_xticks(np.arange(len(values)), conflict_labels, rotation=25, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Fraction predicting original N")
    axes[2].set_title("Length-preserving trace conflicts")
    _style_axis(axes[2])
    fig.suptitle("Terminal answer state: transport, bridge, and structural conflict tests", fontsize=15)
    _savefig(fig, path)


def plot_geometry_dynamics(dynamics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    plot_order = [
        "nonthinking_prompt_occurrence",
        "nonthinking_answer_query",
        "thinking_answer_query",
        "thinking_item_end",
    ]
    styles = {
        "nonthinking_prompt_occurrence": ("o", "--", 5),
        "nonthinking_answer_query": ("o", "-", 5),
        "thinking_answer_query": ("D", "-", 5),
        "thinking_item_end": ("^", "--", 7),
    }
    for endpoint in plot_order:
        frame = dynamics.loc[dynamics["endpoint"].eq(endpoint)].sort_values("step")
        marker, linestyle, marker_size = styles[endpoint]
        ba = 0.5 * (
            frame["confirmation_logistic_balanced_accuracy"]
            + frame["confirmation_ncc_balanced_accuracy"]
        )
        axes[0].plot(frame["step"], ba, marker=marker, markersize=marker_size, linestyle=linestyle, linewidth=2, color=COLORS[endpoint], label=ENDPOINT_SHORT[endpoint])
        axes[1].plot(frame["step"], frame["confirmation_isotropic_snr_db"], marker=marker, markersize=marker_size, linestyle=linestyle, linewidth=2, color=COLORS[endpoint])
        axes[2].plot(frame["step"], frame["confirmation_ordinal_rsa"], marker=marker, markersize=marker_size, linestyle=linestyle, linewidth=2, color=COLORS[endpoint])
    titles = ["Confirmation decoder BA", "Confirmation isotropic SNR", "Confirmation ordinal RSA"]
    ylabels = ["Mean(Logistic BA, NCC BA)", "SNR (dB)", "Spearman rho"]
    for ax, title, ylabel in zip(axes, titles, ylabels, strict=True):
        ax.axvline(1500, color="#555555", linestyle=":", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Optimizer step (linear scale)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 1500, 3000, 5000, 8000, 10000], ["0", "1.5k", "3k", "5k", "8k", "10k"])
        _style_axis(ax)
    axes[0].axhline(1 / 30, color="#777777", linestyle="--", linewidth=1)
    axes[0].legend(fontsize=7, loc="center right")
    fig.suptitle("Aligned representation geometry over training (fixed final-selected layer)", fontsize=15)
    _savefig(fig, path)


def _resize_reference(source: Path, destination: Path, max_width: int = 1800) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        image.save(destination, format="PNG", optimize=True)


def _data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def _table(frame: pd.DataFrame, *, index: bool = False) -> str:
    return frame.to_html(index=index, border=0, classes="data-table", escape=False)


def _figure(path: Path, caption: str, alt: str) -> str:
    return f'<figure><img src="{_data_uri(path)}" alt="{alt}"><figcaption>{caption}</figcaption></figure>'


def build_report(run_dir: Path, output_path: Path) -> None:
    analysis = run_dir / "analysis"
    synth = analysis / "synthetic_report"
    tables = synth / "tables"
    figures = synth / "figures"
    v10 = analysis / "v10_port" / "tables"
    dynamics_root = analysis / "training_dynamics_anthropic_style"

    geometry = _read(tables / "geometry_site_layer_metrics.csv")
    geometry_selection = _read(tables / "geometry_discovery_selected_metrics.csv")
    coordinates = _read(tables / "geometry_confirmation_pca_coordinates.csv")
    geometry_dynamics = _read(tables / "geometry_training_dynamics.csv")
    local_ablation = _read(v10 / "position_local_head_ablation.csv")
    mediation = _read(v10 / "retrieval_localization_transport_patching.csv")
    routing = _read(v10 / "state_to_head_routing.csv")
    residual = _read(v10 / "residual_count_transport.csv")
    bridge = _read(v10 / "final_bridge_component_patching.csv")
    conflicts = _read(v10 / "length_preserving_trace_conflicts.csv")
    prompt_restore = _read(tables / "nonthinking_prompt_evidence_restoration.csv")
    scope_restore = _read(tables / "thinking_trace_scope_restoration.csv")
    behavior = _read(run_dir / "tables" / "final_autoregressive_summary.csv")
    model_specs = _read(run_dir / "tables" / "model_specifications.csv")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    corpus_split = json.loads(
        (run_dir / "data" / "corpus_split.json").read_text(encoding="utf-8")
    )
    needle_pool = json.loads(
        (run_dir / "data" / "needle_pool.json").read_text(encoding="utf-8")
    )
    vocab = json.loads((run_dir / "vocab.json").read_text(encoding="utf-8"))
    suite_manifest = _read(run_dir / "tables" / "loss_suite_manifest_summary.csv")
    sampling = _read(run_dir / "tables" / "training_sampling_distribution.csv")
    formation = _read(dynamics_root / "tables" / "formation_windows.csv")
    banks = _read(dynamics_root / "tables" / "head_bank_differentiation.csv")
    js = _read(dynamics_root / "tables" / "head_role_js_divergence.csv")
    attention_roles = _read(analysis / "extended" / "tables" / "attention_role_dynamics.csv")

    generated = {
        "geometry_layerwise": figures / "geometry_layerwise_decoding.png",
        "geometry_metrics": figures / "geometry_selected_metrics.png",
        "geometry_pca": figures / "geometry_confirmation_pca.png",
        "local_ablation": figures / "causal_local_head_ablation.png",
        "transport": figures / "causal_retrieval_transport.png",
        "evidence_progress": figures / "causal_evidence_progress.png",
        "terminal": figures / "causal_terminal_readout.png",
        "geometry_dynamics": figures / "geometry_training_dynamics.png",
    }
    plot_geometry_layerwise(geometry, generated["geometry_layerwise"])
    plot_geometry_metrics(geometry, generated["geometry_metrics"])
    plot_geometry_pca(coordinates, generated["geometry_pca"])
    plot_local_head_ablation(local_ablation, generated["local_ablation"])
    plot_retrieval_transport(mediation, routing, generated["transport"])
    plot_evidence_progress(prompt_restore, scope_restore, generated["evidence_progress"])
    plot_terminal_readout(residual, bridge, conflicts, generated["terminal"])
    plot_geometry_dynamics(geometry_dynamics, generated["geometry_dynamics"])

    references = {
        "training_overview": figures / "reference_training_overview.png",
        "bank_differentiation": figures / "reference_head_bank_differentiation.png",
        "role_specialization": figures / "reference_broad_targeted_role_specialization.png",
        "linear_log": figures / "reference_linear_vs_log.png",
    }
    sources = {
        "training_overview": dynamics_root / "figures" / "v20_training_dynamics_overview.png",
        "bank_differentiation": dynamics_root / "figures" / "v20_head_bank_differentiation.png",
        "role_specialization": dynamics_root / "figures" / "v20_broad_targeted_role_specialization.png",
        "linear_log": dynamics_root / "figures" / "v20_broad_targeted_linear_vs_log.png",
    }
    for name in references:
        _resize_reference(sources[name], references[name])

    selected = _common_selected(geometry).set_index("endpoint")
    selected_table = pd.DataFrame(
        [
            {
                "Endpoint": ENDPOINT_SHORT[endpoint],
                "Layer": f"L{int(row['layer'])}",
                "Logistic BA": _fmt(row["confirmation_logistic_balanced_accuracy"]),
                "NCC BA": _fmt(row["confirmation_ncc_balanced_accuracy"]),
                "SNR (dB)": _fmt(row["confirmation_isotropic_snr_db"], 2),
                "Frozen Fisher": _fmt(row["confirmation_fisher_trace_frozen"], 1),
                "Silhouette": _fmt(row["confirmation_mahalanobis_silhouette"]),
                "Ordinal RSA": _fmt(row["confirmation_ordinal_rsa"]),
            }
            for endpoint, row in selected.iterrows()
        ]
    )
    ar_nt = behavior.loc[behavior["mode"].eq("nonthinking")].iloc[0]
    ar_th = behavior.loc[behavior["mode"].eq("thinking")].iloc[0]
    model = model_specs.iloc[0]
    target_window = formation.loc[formation["role"].eq("targeted_retrieval")].iloc[0]
    successor_window = formation.loc[formation["role"].eq("marker_successor")].iloc[0]
    final_bank = banks.loc[banks["step"].eq(banks["step"].max())].set_index("role")
    final_js = js.loc[js["step"].eq(js["step"].max()) & js["pair"].eq("thinking_broad__targeted_retrieval"), "normalized_js_divergence"].iloc[0]
    final_roles = attention_roles.loc[(attention_roles["step"].eq(10000)) & attention_roles["is_fixed_role_head"].eq(1)].set_index("role")

    accepted_sampling = sampling.loc[sampling["dimension"].eq("accepted_counts")].copy()
    sampling_summary = (
        accepted_sampling.groupby("mode", as_index=False)
        .agg(
            total_training_examples=("examples", "sum"),
            minimum_examples_in_a_count_bin=("examples", "min"),
            maximum_examples_in_a_count_bin=("examples", "max"),
        )
        .set_index("mode")
    )

    med_summary = mediation.groupby(["intervention", "top_n"])["normalized_recovery"].mean()
    routing_summary = routing.groupby("head")["routing_shift"].mean()
    scope_l4 = scope_restore.loc[scope_restore["layer"].eq(4)]
    terminal_close = scope_l4.loc[(scope_l4["scope"].eq("marker_only")) & (scope_l4["donor_kind"].eq("terminal_total_equals_k")), "patched_close_decision"].mean()
    control_close = scope_l4.loc[scope_l4["donor_kind"].eq("continuing_same_total_control"), "patched_close_decision"].mean()
    prompt_target = prompt_restore.loc[(prompt_restore["layer"].eq(0)) & prompt_restore["intervention"].eq("target_embedding_at_deleted_target")]
    prompt_transfer = prompt_restore.loc[(prompt_restore["layer"].eq(0)) & prompt_restore["intervention"].eq("target_embedding_at_ordinary_location_control")]
    prompt_l1 = prompt_restore.loc[(prompt_restore["layer"].eq(1)) & prompt_restore["intervention"].eq("target_state_at_deleted_target")]
    bridge_l2 = bridge.loc[(bridge["layer"].eq(2)) & bridge["component"].eq("attention_output"), "normalized_recovery"].mean()
    shortened_follow = conflicts.loc[conflicts["intervention"].eq("shortened_trace_position_shifted"), "follows_n_minus_1"].mean()

    endpoint_table = pd.DataFrame(
        [
            ["Non-thinking prompt occurrence", "第 k 个 target character 的位置", "k", "prompt 内局部 evidence 是否形成 running index"],
            ["Thinking item end", "第 k 个 trace marker（紧随显式 index）", "k", "trace progress state"],
            ["Non-thinking answer query", "&lt;Ans&gt; token", "总数 N", "broad retrieval 后的答案状态"],
            ["Thinking answer query", "&lt;Ans&gt; token", "总数 N", "trace-to-answer terminal bridge"],
        ],
        columns=["Endpoint", "Token site", "Label", "Interpretation"],
    )
    special_token_count = 8
    number_token_count = len(vocab["numbers"])
    character_token_count = len(vocab["id_to_token"]) - special_token_count - number_token_count
    task_test_size = int(
        suite_manifest.loc[
            suite_manifest["curve_source"].eq("test") & suite_manifest["suite"].eq("task"),
            "num_examples",
        ].iloc[0]
    )
    heldout_task_size = int(
        suite_manifest.loc[
            suite_manifest["curve_source"].eq("heldout") & suite_manifest["suite"].eq("task"),
            "num_examples",
        ].iloc[0]
    )
    setup_table = pd.DataFrame(
        [
            ["Run identity", f"v20 main · seed {int(config['seed'])} · RoPE only"],
            ["Task", "Count every occurrence of any member of a queried 3-character set"],
            ["Input window", f"{int(config['seq_len'])} Tiny-Shakespeare characters; query precedes data"],
            ["Accepted count", f"1–{int(config['count_max_threshold'])}; zero-count and >{int(config['count_max_threshold'])} windows are rejected and resampled"],
            ["Training mixture", f"task_occurrence_ratio={config['task_occurrence_ratio']:.1f}: every minibatch example is a counting task"],
            ["Paired modes", "Same architecture, exact name-seeded initialization and matched sampled examples; separate optimizers and different output formats"],
        ],
        columns=["Item", "Exact setting"],
    )
    data_table = pd.DataFrame(
        [
            ["Corpus", "Tiny Shakespeare", f"{int(corpus_split['corpus_length']):,} Unicode characters"],
            ["Train region", f"[{int(corpus_split['train']['start']):,}, {int(corpus_split['train']['end']):,})", f"{int(corpus_split['train']['end']) - int(corpus_split['train']['start']):,} characters"],
            ["Validation / confirmation", f"[{int(corpus_split['validation']['start']):,}, {int(corpus_split['validation']['end']):,})", f"{int(corpus_split['validation']['end']) - int(corpus_split['validation']['start']):,} characters"],
            ["Test", f"[{int(corpus_split['test']['start']):,}, {int(corpus_split['test']['end']):,})", f"{int(corpus_split['test']['end']) - int(corpus_split['test']['start']):,} characters"],
            ["Leakage guard", "Two omitted boundaries", f"{int(config['seq_len']) - 1} characters each, so no length-{int(config['seq_len'])} window crosses a split"],
            ["Needle pool", f"{len(needle_pool['sets'])} frozen sets × {int(config['needle_set_size'])} distinct characters", f"Train-frequency sum ≤ {config['needle_pool_frequency_threshold']}; {int(config['needle_pool_frequency_bins'])} stratification bins; pool seed {int(config['effective_needle_pool_seed'])}"],
            ["Training count distribution", "Natural after rejection sampling", f"{int(sampling_summary.loc['nonthinking','minimum_examples_in_a_count_bin']):,}–{int(sampling_summary.loc['nonthinking','maximum_examples_in_a_count_bin']):,} examples per N; not balanced"],
        ],
        columns=["Object", "Construction", "Recorded value"],
    )
    sequence_table = pd.DataFrame(
        [
            [
                "Non-thinking",
                "&lt;BOS&gt; &lt;CountChar&gt; c₁ c₂ c₃ &lt;Sep&gt; data[256] &lt;Ans&gt; N &lt;EOS&gt;",
                "265 tokens",
                "step 1–1500: every next-token target; step 1501–10000: &lt;Ans&gt;, N and &lt;EOS&gt;",
            ],
            [
                "Thinking",
                "&lt;BOS&gt; &lt;CountChar&gt; c₁ c₂ c₃ &lt;Sep&gt; data[256] &lt;Think&gt; (k, mₖ)ₖ₌₁ᴺ &lt;/Think&gt; &lt;Ans&gt; N &lt;EOS&gt;",
                "267 + 2N tokens; maximum 327",
                "step 1–1500: every next-token target; step 1501–10000: &lt;Think&gt; through &lt;EOS&gt;",
            ],
        ],
        columns=["Mode", "Exact serialized sequence", "Length", "Active loss targets"],
    )
    model_table = pd.DataFrame(
        [
            ["Vocabulary", f"{len(vocab['id_to_token'])} tokens", f"{special_token_count} special + {character_token_count} atomic characters + {number_token_count} atomic numbers (1–30)"],
            ["Backbone", f"{int(model['n_layer'])} pre-LN decoder blocks", f"d_model={int(model['n_embd'])}, MLP={int(model['n_inner'])}, GELU(tanh), no dropout"],
            ["Self-attention", f"{int(model['n_head'])} heads × {int(model['n_embd'] // model['n_head'])} dimensions", f"Causal attention; RoPE base {int(config['rope_base'])}; SDPA/Flash fast path in ordinary training"],
            ["Context", f"n_positions={int(config['n_positions'])}", "Long enough for the 327-token maximum thinking sequence"],
            ["Readout", "Final LayerNorm + tied token-embedding matrix", "No separate LM-head weight matrix"],
            ["Parameters", f"{int(model['parameters']):,} per mode", "Two independently optimized models; parameter counts are identical"],
            ["Initialization", "Name-seeded Gaussian weights σ=0.02", "Biases 0; LayerNorm scale 1 / bias 0; exact shared step-0 weights across modes"],
        ],
        columns=["Component", "Setting", "Implementation detail"],
    )
    optimization_table = pd.DataFrame(
        [
            ["Updates / batch", f"{int(config['train_steps']):,} optimizer steps × {int(config['batch_size'])} examples", f"{int(sampling_summary.loc['nonthinking','total_training_examples']):,} task examples per model; no gradient accumulation"],
            ["Optimizer", "AdamW", f"β₁={config['adam_beta1']}, β₂={config['adam_beta2']}, ε=1e-8 (PyTorch default), weight decay={config['weight_decay']}"],
            ["Learning rate", f"peak {config['lr']:.1e}", f"linear warmup for {int(config['warmup_steps'])} steps, then cosine decay to 0 at step {int(config['train_steps']):,}"],
            ["Objective phase A", f"steps 1–{int(config['max_steps_for_language_pred']):,}", "teacher-forced weighted next-token cross-entropy over every non-padding target after the first token"],
            ["Objective phase B", f"steps {int(config['max_steps_for_language_pred']) + 1:,}–{int(config['train_steps']):,}", "task-output-only mask; all final-count and trace weights equal 1.0"],
            ["Numerics", f"CUDA, configured {config['precision']}", f"autocast BF16 when hardware supports it; global gradient-norm clip {config['grad_clip']}; otherwise FP32 fallback"],
        ],
        columns=["Training item", "Value", "Exact meaning"],
    )
    evaluation_table = pd.DataFrame(
        [
            ["Logging", f"every {int(config['log_every'])} steps", "stochastic minibatch loss, active-token accounting and gradient norm"],
            ["Teacher-forced curves", f"every {int(config['eval_every'])} steps", f"train and held-out frozen suites; {heldout_task_size} held-out task examples = 10/count"],
            ["Autoregressive curves", f"every {int(config['ar_eval_every'])} steps", f"{int(config['ar_examples_per_count'])} examples/count"],
            ["Final behavior", "step 10,000 test split", f"{task_test_size} balanced task examples = {int(config['final_examples_per_count'])}/count"],
            ["State snapshots", f"every {int(config['checkpoint_every'])} steps including step 0", "101 FP16 analysis snapshots; full optimizer/RNG recovery checkpoints every 500 steps"],
            ["Final geometry", "discovery 10/class; confirmation 8/class", "Train-region discovery selects layers; validation-region confirmation is frozen and never selects"],
            ["Geometry dynamics", "13 milestones: 0, 500, 1k, 1.5k, then every 1k to 10k", "5/class discovery + 4/class confirmation; always track the final discovery-selected physical layer"],
        ],
        columns=["Protocol", "Cadence / size", "Use"],
    )
    causal_ladder = pd.DataFrame(
        [
            ["Non-thinking evidence intake", "Delete one target; restore input embedding", f"Exact-site median recovery {_fmt(prompt_target.normalized_recovery.median())}; location-transfer {_fmt(prompt_transfer.normalized_recovery.median())}", "Target identity is usable before L1 and is largely location-invariant"],
            ["Non-thinking broad bank", "Position-local ranked Top-K ablation vs random", "Ranked Top-2 margin drop 1.054 vs random 0.846", "Broad retrieval is distributed; selected bank is only moderately more necessary than random"],
            ["Thinking targeted bank", "Target-source corruption + pattern/value patch", f"Top-2 value recovery {_fmt(med_summary.loc[('value_only_at_target_source', 2)])}; pattern-only {_fmt(med_summary.loc[('attention_pattern_only', 2)])}", "Value transport, not attention pattern alone, carries occurrence identity"],
            ["Progress → routing", "Patch k=8 progress state into k=3 query", f"Routing shift L4H2 {_fmt(routing_summary.loc[2])}; L4H1 {_fmt(routing_summary.loc[1])}", "Progress state causally redirects targeted heads"],
            ["Stop execution", "Terminal-total-k donor into continuing total-k+2 trace", f"L4 marker patch closes {_pct(terminal_close)} vs control {_pct(control_close)}", "Late marker state executes stop/continue decision"],
            ["Terminal answer bridge", "Shorten trace; restore component at answer query", f"L2 attention-output recovery {_fmt(bridge_l2)}; shortened trace follows N−1 {_pct(shortened_follow)}", "Synthetic readout is a structural trace bridge, not final-position prompt-wide aggregation"],
        ],
        columns=["Stage", "Intervention", "Result", "Supported claim"],
    )

    geometry_rows = []
    for row in geometry_selection.itertuples():
        geometry_rows.append(
            {
                "Endpoint": ENDPOINT_SHORT[row.endpoint],
                "Metric selector": str(row.selector).replace("_", " "),
                "Selected layer": f"L{int(row.selected_layer)}",
                "Discovery": _fmt(row.discovery_value),
                "Frozen confirmation": _fmt(row.confirmation_value),
            }
        )
    geometry_selector_table = pd.DataFrame(geometry_rows)

    large_model_reference_table = pd.DataFrame(
        [
            ["Models", "Qwen3-8B (L0–L35) and Gemma4-E4B (L0–L41)", "Frozen pretrained large models; the reports analyze existing forward/generation behavior rather than training them on NiaH"],
            ["Task", "Natural city-score records embedded in roughly 10k-token passages", "Counts 1–10; records are multi-token semantic spans rather than atomic characters"],
            ["Non-thinking panel", "canonical seeds 1234–1263 × N=1…10", "300 clean natural forwards/model, plus discovery/confirmation interventions"],
            ["Geometry panel", "20 discovery seeds + 10 confirmation seeds", "Native running states are parser-observed and ragged; confirmation never selects the displayed layer"],
            ["Native targeted banks", "Qwen frozen Top-128; Gemma frozen Top-6", "Bank sizes and raw attention mass are model-specific and cannot be compared numerically to v20 Top-2"],
            ["Strongest natural counter result", "Qwen3-8B, N=10, first-pass natural no-index traces", "Gemma does not yet have the corresponding natural no-index causal chain; its evidence is a cross-model reference under more controlled formats"],
        ],
        columns=["Reference dimension", "Large-model report setting", "Boundary for comparison"],
    )

    mechanism_gap_table = pd.DataFrame(
        [
            [
                "Geometry direction",
                "Native-thinking is more decodable in all 8 frozen held-out running/final comparisons; covariance metrics are mixed.",
                "Thinking item-end and answer-query are nearly perfectly decoded; non-thinking endpoints are weak.",
                '<span class="status aligned">direction aligned</span>',
                "The qualitative mode ordering reproduces, but v20's magnitude is inflated by explicit trace tokens.",
            ],
            [
                "Non-thinking prompt running state",
                "A noisy but readable occurrence-order signal forms across full needle spans/endpoints before broad retrieval.",
                f"Prompt-occurrence Logistic BA is only {_fmt(selected.loc['nonthinking_prompt_occurrence','confirmation_logistic_balanced_accuracy'])} (chance 0.033) with SNR {_fmt(selected.loc['nonthinking_prompt_occurrence','confirmation_isotropic_snr_db'],2)} dB.",
                '<span class="status open">mechanism gap</span>',
                "Synthetic v20 does not reproduce the large-model prompt running-index manifold; it looks more like local identity intake.",
            ],
            [
                "Non-thinking broad retrieval",
                "Answer query broadly retrieves distributed record-span evidence and later consolidates an executable count state.",
                "Deleting/restoring a target embedding identifies Layer-1 content-addressed intake; ranked Top-2 ablation is only moderately stronger than matched random (1.054 vs 0.846 margin drop).",
                '<span class="status partial">partial alignment</span>',
                "Broad aggregation exists functionally, but source granularity and bank identity specificity are weaker than in the large-model chain.",
            ],
            [
                "Thinking targeted retrieval",
                "Frozen targeted banks select the next record; selected-vs-random ablation and downstream carrier effects support causal use.",
                f"Top-2 target-source value patch recovers {_fmt(med_summary.loc[('value_only_at_target_source', 2)])}; progress patch redirects L4H2/L4H1.",
                '<span class="status aligned">strong functional alignment</span>',
                "Both settings support state-conditioned targeted reads, though head counts and absolute attention mass are incomparable.",
            ],
            [
                "Internal counter / progress state",
                "Qwen natural no-index item-state transplant changes successor likelihood, targeted attention, candidate argmax and free continuation.",
                "Every trace step is directly supervised with atomic k; the main routing patch is teacher-forced k=8→k=3.",
                '<span class="status open">major gap</span>',
                "v20 establishes a progress-conditioned state, but not a natural implicit counter independent of visible index tokens.",
            ],
            [
                "Retrieval → carrier → commit mediation",
                "Selected-bank damage deforms a downstream grammar carrier; restoring clean carrier partially rescues a later commit in the same damaged arm.",
                "Value transport, routing and marker-stop patches are each positive, but they are not yet one registered serial mediation experiment with a shared damaged baseline.",
                '<span class="status partial">edgewise only</span>',
                "The synthetic chain is plausible but not as tightly closed as the large-model carrier damage/rescue experiment.",
            ],
            [
                "Terminal readout",
                "Native trace content is the main natural source; prompt-broad readout is not ruled out, and terminal grammar-state restoration is model/site specific.",
                f"Shortening the trace makes {_pct(shortened_follow)} follow N−1; L2 answer-query attention-output recovery is {_fmt(bridge_l2)}, while length-preserving index conflicts leave N unchanged.",
                '<span class="status open">different implementation</span>',
                "v20 readout is dominated by trace length/absolute query position, a likely shortcut rather than the same semantic terminal bridge.",
            ],
            [
                "Thinking-mode broad retrieval",
                "Prompt-broad retrieval remains an allowed parallel path, but the large report does not make it the universal final aggregator.",
                "Final-query prompt coverage is measured only as a descriptive role; no causal source-composition test shows that it drives the answer.",
                '<span class="status partial">unresolved in both</span>',
                "The correct claim is 'not established', not 'absent in all large models' or 'required in synthetic'.",
            ],
            [
                "Free-running causal sufficiency",
                "Qwen includes a natural no-index continuation transfer after an internal-state intervention.",
                "Unintervened AR behavior is measured, but most head/state interventions read a teacher-forced next-token margin.",
                '<span class="status open">missing in v20</span>',
                "Local causal effects have not yet been shown to survive compounding generation errors.",
            ],
            [
                "Training emergence",
                "The pretrained large-model reports are static mechanistic analyses and have no original training checkpoints.",
                "v20 has 101 snapshots and shows scaffold by step 500, successor first, targeted bank later.",
                '<span class="status incomparable">synthetic-only evidence</span>',
                "This is a new controlled result, not yet evidence that large-model circuits emerged in the same order.",
            ],
        ],
        columns=["Question", "Large-model evidence", "Synthetic v20 evidence", "Status", "What the gap means"],
    )

    design_gap_table = pd.DataFrame(
        [
            ["Scale and prior training", "3.19M parameters, 4 layers × 4 heads, trained from scratch on v20", "Qwen3-8B / Gemma4-E4B pretrained models", "Circuit capacity, depth and feature reuse differ; layer/head counts cannot be mapped."],
            ["Input semantics", "256 character tokens; target is membership in a 3-character set", "Roughly 10k-token natural passages; needles are semantic record spans", "v20 can solve with atomic identity matching and avoids span binding/entity semantics."],
            ["Reasoning supervision", "Gold index and target marker are teacher-forced at every thinking step", "Native traces are naturally generated and grammar/parser dependent", "v20's early perfect geometry may encode supervised surface structure rather than an autonomous counter."],
            ["Number representation", "Each N=1…30 and each trace index is one atomic token", "Natural numerals may be multi-token and embedded in language", "v20 removes digit composition, carry and tokenizer-boundary problems."],
            ["Training distribution", "One fixed pool, natural in-range N distribution, no OOD task family", "Frozen pretrained knowledge plus multiple natural templates/domains in the reference suite", "No evidence yet for new needle sets, longer contexts or counts beyond 30."],
            ["Replication", "One architecture and one training seed (1234)", "Two model families and 30 stimulus seeds, but the strongest no-index causal result is Qwen-only", "Neither side alone establishes universality; v20 formation times especially need multiple training seeds."],
            ["Intervention unit", "Atomic character embedding/head slice/single trace token", "Whole semantic spans, parser-registered items and large distributed banks", "A successful v20 patch can be much easier and more localized than its large-model analogue."],
            ["Terminal confound", "Trace length moves the answer-query absolute position under RoPE", "Natural traces vary in wording/length and terminal sites are parser aligned", "The current shortened-trace effect does not separate count from position; a length-matched counterfactual is required."],
        ],
        columns=["Design axis", "Synthetic v20", "Large-model reports", "Scientific consequence"],
    )

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,">
<title>NiaH Synthetic v20: Geometry, Causal Experiments, and Training Dynamics</title>
<style>
:root{{--ink:#17212b;--muted:#5b6773;--paper:#ffffff;--soft:#f4f7fa;--line:#d9e1e8;--blue:#315f86;--purple:#6f568f;--teal:#187f73;--orange:#b86818;--red:#a63d40}}
*{{box-sizing:border-box}} body{{margin:0;background:#edf1f4;color:var(--ink);font-family:Inter,"Segoe UI","Microsoft YaHei",Arial,sans-serif;line-height:1.68}}
main{{max-width:1180px;margin:24px auto;background:var(--paper);padding:44px 58px 80px;box-shadow:0 8px 30px rgba(30,45,60,.10)}}
h1{{font-size:2.25rem;line-height:1.18;margin:0 0 10px}} h2{{font-size:1.55rem;border-bottom:2px solid var(--line);padding-bottom:8px;margin-top:52px}} h3{{font-size:1.18rem;margin-top:34px;color:#24384a}} h4{{font-size:1rem;margin-top:25px}}
p{{margin:10px 0}} .subtitle{{font-size:1.1rem;color:var(--muted);margin-bottom:22px}} .meta{{font-size:.88rem;color:var(--muted)}}
.abstract{{background:linear-gradient(135deg,#eef5fb,#f7f3fb);border-left:5px solid var(--blue);padding:18px 22px;margin:25px 0}}
.purpose,.example,.conclusion,.warning{{padding:12px 16px;margin:13px 0;border-left:4px solid var(--blue);background:#f3f7fb}} .example{{border-color:var(--teal);background:#f0f8f6}} .conclusion{{border-color:var(--purple);background:#f6f3f9;font-weight:500}} .warning{{border-color:var(--orange);background:#fff7ec}}
.label{{font-weight:700;margin-right:5px}} figure{{margin:24px 0 31px}} figure img{{display:block;width:100%;height:auto;border:1px solid var(--line)}} figcaption{{font-size:.9rem;color:#46525e;margin-top:8px;line-height:1.5}}
.data-table{{border-collapse:collapse;width:100%;max-width:100%;font-size:.88rem;margin:16px 0 24px}} .data-table th,.data-table td{{border-bottom:1px solid var(--line);padding:8px 9px;text-align:left;vertical-align:top}} .data-table th{{background:var(--soft);font-weight:650}} code{{background:#eef2f5;padding:1px 4px;border-radius:3px}} .formula{{font-family:"Cambria Math",Georgia,serif;background:#f8fafc;border:1px solid var(--line);padding:10px 13px;overflow-x:auto}}
.toc{{columns:2;background:var(--soft);padding:15px 22px;margin:20px 0}} .toc a{{color:var(--blue);text-decoration:none}} a{{color:#245b86}} .small{{font-size:.86rem;color:var(--muted)}} .tag{{display:inline-block;padding:2px 8px;border-radius:999px;background:#e9eff5;font-size:.78rem;margin-right:5px}}
.status{{display:inline-block;white-space:nowrap;padding:2px 7px;border-radius:999px;font-size:.76rem;font-weight:700}} .status.aligned{{background:#dff3ee;color:#11695f}} .status.partial{{background:#fff0d9;color:#8b500f}} .status.open{{background:#f8dfe1;color:#8d2f34}} .status.incomparable{{background:#e7ebef;color:#4b5661}}
@media(max-width:760px){{main{{margin:0;padding:24px 18px}}h1{{font-size:1.75rem}}.toc{{columns:1}}.data-table{{display:block;overflow-x:auto;font-size:.78rem}}}}
@media print{{body{{background:white}}main{{box-shadow:none;margin:0;max-width:none}}figure{{break-inside:avoid}}h2{{break-before:auto}}}}
</style></head><body><main>
<div class="meta">Synthetic NiaH v20 · RoPE · count 1–30 · seed 1234 · generated 2026-08-28</div>
<h1>NiaH Synthetic v20：Geometry Comparison、Causal Experiments 与 Training Dynamics</h1>
<div class="subtitle">对齐大模型 Non-thinking / Native-thinking 报告的受控机制研究</div>
<div class="abstract"><span class="label">核心结论。</span>在这个 3.19M 参数的受控模型中，thinking 的优势不是单一“更强 attention”，而是<strong>显式 trace scaffold 先形成，随后 targeted-retrieval bank 再逐步分化</strong>。最终 thinking 的 item-end 与 answer-query count decodability 接近完美；Top-2 targeted heads 的 target-source values 可恢复约 {_fmt(med_summary.loc[('value_only_at_target_source', 2)])} 的受损 identity margin，而 pattern-only 只有 {_fmt(med_summary.loc[('attention_pattern_only', 2)])}。Non-thinking 则表现为第一层、位置近似不变的 broad evidence intake，但缺少稳定的 prompt running-index geometry，最终 AR accuracy 仅 {_pct(ar_nt.ar_final_accuracy)}，低于 thinking 的 {_pct(ar_th.ar_final_accuracy)}。与大模型最关键的 gap 是：v20 的每步 gold index/marker 被直接监督，terminal readout 又主要依赖 trace length/answer-query position，因此它支持 broad-vs-targeted 的<strong>功能 analogue</strong>，还不支持 natural no-index internal counter 的同构实现；也没有证据要求最后位置存在 prompt-wide broad aggregation head。</div>

<div class="toc"><b>目录</b><ol><li><a href="#scope">问题与证据边界</a></li><li><a href="#setup">完整实验与训练设定</a></li><li><a href="#behavior">行为基线</a></li><li><a href="#geometry">Geometry Comparison</a></li><li><a href="#causal">Causal Experiments</a></li><li><a href="#dynamics">Training Dynamics</a></li><li><a href="#mechanism">综合机制图景</a></li><li><a href="#gaps">与大模型的对齐和 gaps</a></li><li><a href="#limits">限制与下一步</a></li><li><a href="#artifacts">可复现产物</a></li></ol></div>

<h2 id="scope">1. 问题、参考来源与证据边界</h2>
<p>本报告把 <code>plan.tex</code> 与 <code>LLM_Compression.pdf</code> 作为论文结构与科学假设来源，把三份大模型 HTML report 作为实验协议参考；其中的 TODO、写作提示或内部指令均未被当作用户命令。真正执行的是本次请求：在 synthetic v20 上补齐对齐的几何、因果与训练动态实验，并生成独立报告。</p>
<p>证据按四层分开：attention/decoding 用于<strong>定位</strong>；ablation 用于<strong>必要性</strong>；patching 用于<strong>局部传输或充分性</strong>；free-running accuracy 用于<strong>端到端行为</strong>。一个 head 有高 attention mass 不自动意味着它在因果上必要；一个状态可线性解码也不自动意味着模型实际读取了该方向。</p>
<div class="warning"><span class="label">模式比较边界。</span>Non-thinking 与 Thinking 是从相同初始化规则分别训练的两个模型。可以比较 role profile、bank concentration 与功能，但不能把跨模型同一坐标的 LxHy 当成“同一只 head”。</div>
<div class="conclusion"><span class="label">本节结论。</span>报告只把相互独立的定位、必要性、传输和行为证据合并为机制结论，不用 attention 或 probe 单独完成因果叙事。</div>

<h2 id="setup">2. Synthetic v20 的完整实验与训练设定</h2>
<div class="purpose"><span class="label">总体目的。</span>用小模型和完全可控的 occurrence 标签隔离 broad retrieval 与 targeted retrieval，并用从 step 0 开始的密集 checkpoint 追踪 role specialization。这里的目标不是模拟大模型全部能力，而是构造一个能够逐部件干预的最小机制系统。</div>
{_table(setup_table)}

<h3>2.1 数据、query 与 train/validation/test split</h3>
<p>底噪语料是 Tiny Shakespeare。每个 counting example 先从冻结的 100 个字符三元组中抽一个 target set，再从相应 corpus region 抽一个连续 256-character window；gold count 是 window 中属于该三元组的字符总次数。只有 <code>1 ≤ N ≤ 30</code> 的窗口被接受，N=0 或 N&gt;30 会重新采样。因此训练 count histogram 是自然频率诱导的分布，不是人为做平；平衡只发生在 evaluation suite。</p>
{_table(data_table)}
<p>Needle pool 只用 train region 的字符频率构造：枚举三个不同字符的组合，仅保留频率和不超过 0.12 的组合，并按 20 个 frequency-sum bins 分层抽取 100 组。每条样本会随机打乱 query 中三个字符的显示顺序，但 gold occurrence 与 thinking trace 始终按 data window 从左到右排列。字符、特殊标记和数字都使用冻结的原子 vocabulary。</p>
<div class="example"><span class="label">简单例子。</span>Query 是字符集合 {{a,b,c}}，data window 从左到右命中 <code>a, c, b, a</code>，则 N=4。Query 中可能显示为 <code>c a b</code>，但这不改变集合语义或 occurrence 顺序。Non-thinking 只需在末尾回答 4；Thinking 则逐项输出 <code>1,a; 2,c; 3,b; 4,a</code> 后再回答 4。</div>
<div class="conclusion"><span class="label">数据设定结论。</span>训练样本在 corpus split、needle pool 和 count filter 上完全冻结且可复现；但任务仍是原子字符集合匹配，不包含大模型中的多-token span binding 与自然语义。</div>

<h3>2.2 两种序列格式与 loss mask</h3>
{_table(sequence_table)}
<p>两种模式都做标准 causal next-token prediction，labels 与输入序列相同并右移一位计算交叉熵。step 1–1,500 时，除首 token 外的所有非 padding targets 都有权重 1；step 1,501–10,000 时，Non-thinking 只保留从 <code>&lt;Ans&gt;</code> 到 <code>&lt;EOS&gt;</code> 的 loss，Thinking 则保留从 <code>&lt;Think&gt;</code> 到 <code>&lt;EOS&gt;</code> 的整段 trace 和 final answer。<code>final_count_loss_weight</code> 与 <code>cot_trace_loss_weight</code> 都是 1.0，没有额外放大某类 token。</p>
<div class="example"><span class="label">Loss-mask 例子。</span>在 step 2,000，Non-thinking 的 256 个 data characters 不再直接贡献 loss，只训练模型从完整 prompt 生成 <code>&lt;Ans&gt; 4 &lt;EOS&gt;</code>；Thinking 同一步仍训练 <code>&lt;Think&gt; 1 a 2 c 3 b 4 a &lt;/Think&gt; &lt;Ans&gt; 4 &lt;EOS&gt;</code> 的每个输出 token。因而 thinking 的 index/marker geometry 得到直接监督，这是解释早期 perfect decodability 时必须保留的 confound。</div>
<div class="warning"><span class="label">Objective switch 的意义。</span>step 1,500 是预先写入训练代码的外生切换点；所有图中的竖虚线都标记该 schedule event，不是根据曲线拟合得到的 phase transition。</div>
<div class="conclusion"><span class="label">序列与目标结论。</span>两模型看到相同 query 和 data，但输出监督量不同：Thinking 额外获得逐 occurrence 的 gold index 与 identity sequence，因此它不是“只多给计算时间”的纯长度对照。</div>

<h3>2.3 模型架构与 paired-control 训练</h3>
{_table(model_table)}
<p>两个模式不是同一个模型切换 decoding flag，而是两个参数独立更新的 causal LM。构造每个模型时，linear、embedding 与 LayerNorm 按模块名和 seed 1234 做 deterministic initialization，所以 step 0 权重逐项相同；每个模式又把 Python sampler 重置到 seed 1234，因此两边实际接受的 1,280,000 个训练窗口及 N histogram 也完全一致。两边各自拥有独立 AdamW state，差异从序列后缀与 loss targets 开始产生。</p>
<div class="example"><span class="label">配对设计例子。</span>第 t 个 minibatch 在两种模式中包含同一组 target sets、同一批 256-character windows 和同一组 gold N。Non-thinking 把每条 window 序列化成短答案；Thinking 在同一 window 后插入 gold trace。这样可以比较 mode-level specialization profile，但训练后 L4H2 与另一模型的 L4H2 仍不是同一 causal object。</div>
<div class="conclusion"><span class="label">模型设定结论。</span>相同初始化和 matched examples 排除了架构、初始权重及样本顺序这三类混杂；无法排除的是两种 supervision 本身带来的直接表征捷径，而这正是设计的一部分。</div>

<h3>2.4 优化器、学习率、精度与训练量</h3>
{_table(optimization_table)}
<p>每一步先生成一个 128-example minibatch，执行一次 forward/backward，按全局 L2 norm 1.0 clip gradient，再把该 step 的 learning rate 写入 AdamW param groups 后更新；没有 gradient accumulation。普通训练用 PyTorch scaled-dot-product attention，只有需要显式 attention weights 或 head intervention 的分析 forward 才切换到 exact manual attention path。</p>
<div class="example"><span class="label">Learning-rate 例子。</span>step 500 达到峰值 3×10<sup>−4</sup>；之后按照 half-cosine 平滑下降，到 step 10,000 为 0。因而后期 targeted-bank 继续变化时，发生在越来越小的 update size 下，不能把横轴简单等同于等量参数移动。</div>
<div class="conclusion"><span class="label">训练方法结论。</span>每个模型训练 10,000 次更新、见到 128 万条 matched task examples；优化 schedule 和数值路径完全相同，唯一系统性处理差异是 output format / loss mask。</div>

<h3>2.5 Checkpoint、evaluation 与分析 split</h3>
{_table(evaluation_table)}
<p>Behavior 主结论使用从未参与训练或层选择的 test region；Geometry 的 layer/probe selection 只读取 discovery，confirmation 只在冻结 preprocessing、classifier 和 layer 后评价。Training-dynamics geometry 不在每个 checkpoint 重新挑最好层，而是向前追踪 final-checkpoint discovery winner，因此曲线表示同一物理 site 如何形成，而不是 checkpoint-wise oracle 上界。</p>
<div class="example"><span class="label">Split 例子。</span>若 L2 在 final discovery 上赢得 Thinking answer-query selector，则 step 0、500、…、10,000 都读取 L2；即使某个早期 checkpoint 的 L1 看起来更好也不改层。这样避免把训练噪声当成 mechanism migration。</div>
<div class="conclusion"><span class="label">完整设定结论。</span>v20 是一个可复现的 matched-pair training study：数据、初始化、optimizer 与 checkpoint coordinate 对齐；它清楚地区分并行 broad aggregation 和逐项 targeted retrieval，但显式 trace supervision 使它只能作为大模型机制的受控 analogue，不能视为缩小版等价物。</div>

<h2 id="behavior">3. 行为基线</h2>
<div class="purpose"><span class="label">实验目的。</span>先确认 v20 确实产生 thinking / non-thinking 的端到端差异，再分析该差异对应的内部机制。</div>
<p>在 1,500 个 test examples（每个 count 50 个）上，Non-thinking 自回归 final-count accuracy 为 <strong>{_pct(ar_nt.ar_final_accuracy)}</strong>（Wilson 95% CI {_pct(ar_nt.ar_final_accuracy_wilson95_low)}–{_pct(ar_nt.ar_final_accuracy_wilson95_high)}）；Thinking 为 <strong>{_pct(ar_th.ar_final_accuracy)}</strong>（{_pct(ar_th.ar_final_accuracy_wilson95_low)}–{_pct(ar_th.ar_final_accuracy_wilson95_high)}），trace exact accuracy 为 {_pct(ar_th.trace_exact)}。</p>
{_figure(references['training_overview'], "图 1｜宏观行为—attention role—routing—local causal use 的同步训练曲线。所有横轴均为线性 optimizer step；虚线为 step 1,500 objective switch。A 的纵轴为 exact accuracy；B 为固定 final-selected head 的 role score；C 左轴为 correct-occurrence attention mass / top-1 probability，右轴为 QK correct-minus-best-wrong margin；D 为在相应 query position 清零 head slice 后的正确 token logit-margin damage。阴影是 sigmoid 10–90% formation window，不是置信区间。", "v20 behavior attention routing and causal-use training dynamics")}
<div class="conclusion"><span class="label">本节结论。</span>thinking 的行为优势真实存在且在 free-running evaluation 中仍保留；后续机制实验需要解释约 57.7 percentage-point 的最终 accuracy gap，而不是只解释 teacher-forced token prediction。</div>

<h2 id="geometry">4. Geometry Comparison</h2>
<h3>4.1 端点、目的与 discovery/confirmation protocol</h3>
<div class="purpose"><span class="label">实验目的。</span>比较两种模式在“running progress”和“terminal count”上的可分性、信噪比、协方差结构与序关系；同时防止用 held-out confirmation 选择最好看的层。</div>
{_table(endpoint_table)}
<div class="example"><span class="label">简单例子。</span>对于 N=5 的样本，running endpoint 会产生标签 1,2,3,4,5；answer-query endpoint 只产生一个标签 5。每个 endpoint 在 discovery 中独立对所有层做 grouped CV，选层后才读取 confirmation 值。</div>
<p>预处理和选择：每个 discovery fold 内拟合 <code>StandardScaler → PCA≤16</code>（decoder 使用 whitened PCA），训练 balanced multinomial logistic probe 与 nearest-centroid classifier；共同层选择分数为两者 balanced accuracy 的平均值。相同 prompt/example 的多个 running states 作为一个 group，避免跨 fold 泄漏。</p>
<p class="formula"><b>六个指标。</b> (1) Logistic BA 与 (2) NCC BA：30 个 count class 的 macro recall；chance = 1/30。 (3) Isotropic SNR：10 log₁₀[tr(Σ<sub>B</sub>)/tr(Σ<sub>W</sub>)]。 (4) Frozen Fisher trace：tr[(Σ<sub>W,disc</sub>+λI)<sup>−1</sup>Σ<sub>B,conf</sub>]。 (5) Mahalanobis silhouette：用 discovery within-class covariance whitening 后的 class-balanced silhouette。 (6) Ordinal RSA：class centroid distance 与 |i−j| 的 Spearman ρ。</p>
<div class="warning"><span class="label">解释限制。</span>Thinking item-end 紧随显式数字 index，answer query 又位于完整 trace 之后；所以高 decodability 可能混合“真正的内部 counter”与“显式 token/trace-length scaffold”。因果 conflict test 将专门检验模型实际读取什么。</div>
<div class="conclusion"><span class="label">本小节结论。</span>协议与大模型报告对齐：discovery 负责选择，confirmation 只负责验证；六个指标分别回答可解码、紧致性、anisotropic separation、样本聚类和 ordinal organization，不能互相替代。</div>

<h3>4.2 Layerwise decodability</h3>
{_figure(generated['geometry_layerwise'], "图 2｜逐层 confirmation count decodability。横轴为 residual layer（L0 是 token embedding，L1–L4 是每个 block 之后）；纵轴为 30-class balanced accuracy。实线圆点为 logistic probe，虚线方点为 nearest-centroid classifier；水平点线为 chance=1/30；竖点线为只用 discovery 选出的共同层。", "layerwise held-out count decodability for four aligned endpoints")}
<p>Non-thinking prompt occurrence 的最佳 confirmation BA 仅约 0.08–0.10，说明局部 prompt evidence state 没有形成稳定的 occurrence-index manifold；Non-thinking answer query 到 L4 才升到 Logistic {_fmt(selected.loc['nonthinking_answer_query','confirmation_logistic_balanced_accuracy'])} / NCC {_fmt(selected.loc['nonthinking_answer_query','confirmation_ncc_balanced_accuracy'])}。相反，Thinking item-end 在 L3–L4 达到近完美，Thinking answer query 从 L2 起为 1.0。</p>
<div class="conclusion"><span class="label">本实验结论。</span>稳定优势是 thinking 的 progress/answer states 更易解码；synthetic non-thinking 没有大模型报告中那种清晰的 prompt running-index geometry。该结论是 representation localization，不是因果使用证明。</div>

<h3>4.3 Covariance-aware geometry</h3>
{_table(selected_table)}
{_figure(generated['geometry_metrics'], "图 3｜每个 endpoint 在共同 discovery-selected layer 上的六项 confirmation 指标。横轴为四个对齐端点；柱高分别为 raw BA、SNR dB、log₁₀(1+frozen Fisher)、silhouette 与 ordinal RSA。不同 panel 的纵轴单位不同，不能跨 panel 比较柱高。", "six confirmation geometry metrics at discovery selected layers")}
<p>Thinking 的 decodability、SNR、Fisher 与 silhouette 明显更高，但“更有序”不是单调的：Thinking item-end 在共同 L4 的 ordinal RSA 为 {_fmt(selected.loc['thinking_item_end','confirmation_ordinal_rsa'])}，而该 endpoint 按 RSA 单独在 discovery 选 L2 后，confirmation RSA 为 {_fmt(geometry_selection.loc[(geometry_selection.endpoint.eq('thinking_item_end')) & (geometry_selection.selector.eq('ordinal_rsa')),'confirmation_value'].iloc[0])}。因此“compact/separable”与“沿 count 排成单一一维轨迹”是不同命题。</p>
{_figure(generated['geometry_pca'], "图 4｜confirmation states 在 discovery-fitted PCA 的 PC1–PC2 投影。每个点是一个 held-out state，颜色表示 count/occurrence 1–30；黑线连接各 class centroid。横纵轴括号给出 discovery explained-variance ratio。二维投影仅用于可视化，正式结论来自 16D frozen metrics。", "confirmation PCA scatter and centroid paths for four endpoints")}
<div class="conclusion"><span class="label">本实验结论。</span>Thinking states 的主要优势是 class separation 与低相对噪声；不能概括成“所有层都更一维”。Non-thinking answer query 仍有较强 ordinal structure（RSA {_fmt(selected.loc['nonthinking_answer_query','confirmation_ordinal_rsa'])}），但 class overlap 大、实际解码率低。</div>

<h2 id="causal">5. Causal Experiments</h2>
<h3>5.1 Non-thinking：输入 evidence 与 broad bank</h3>
<div class="purpose"><span class="label">实验目的。</span>检验 broad retrieval 是否读取 target identity、是否依赖固定位置，以及这些 evidence 在哪一层进入 answer computation。</div>
<div class="example"><span class="label">简单例子。</span>原 prompt 有 N=8 次 target；删除最后一次得到 N−1 evidence，但保持答案 query 不变。把 clean target embedding 恢复到被删位置，或移动到一个原本普通字符的位置，再观察 N vs N−1 logit margin。</div>
{_figure(generated['evidence_progress'], "图 5｜A：Non-thinking evidence restoration。横轴为 patch 类型，纵轴为 (patched−corrupt)/(clean−corrupt) 的中位数；柱上 acc 是 N vs N−1 pairwise accuracy。L0 表示进入 Layer 1 前的 embedding patch，post-L1 表示第一层之后再 patch。B：Thinking terminal-progress scope；横轴为 patched layer，纵轴为 close-minus-continue logit margin shift；不同线表示 index、marker、两者和 continuing donor control。", "causal patch timing for nonthinking evidence and thinking progress scope")}
<p>Clean target embedding 在原位置给出中位 recovery {_fmt(prompt_target.normalized_recovery.median())}；放到普通位置仍为 {_fmt(prompt_transfer.normalized_recovery.median())}。普通 embedding 不救回，而第一层之后再 patch clean target state 的 recovery 约 {_fmt(prompt_l1.normalized_recovery.median())}。这说明模型在 Layer 1 broad intake 时按 target identity 读取 evidence，且 source position 近似可交换；读完以后修复局部 state 已经太晚。</p>
{_figure(generated['local_ablation'], "图 6｜三类 head bank 的 position-local Top-K ablation。横轴为只在角色 query positions 清零的 head 数；纵轴为正确 token logit margin 相对 baseline 的下降，各 panel 为突出角色内部差异而独立 autoscale。实线是 discovery-ranked bank，虚线是 layer/count-matched random banks 的均值，灰带为 random-path SEM。A 对应 final answer broad retrieval，B 对应 trace-index targeted retrieval，C 对应 trace readout。", "ranked versus random local head bank ablation")}
<div class="conclusion"><span class="label">本实验结论。</span>Non-thinking 是“输入 identity → 第一层 broad intake → 后层 answer state”，不是在每个 prompt occurrence 维护可修复的 running counter。位置转移仍能救回，直接支持 broad、content-addressed retrieval。</div>

<h3>5.2 Thinking：targeted retrieval 的定位、传输与 progress-conditioned routing</h3>
<div class="purpose"><span class="label">实验目的。</span>把“attention 指向第 k 个 occurrence”拆成 routing pattern 与 value identity transport，并检验 progress state 是否真的控制下一次检索位置。</div>
<div class="example"><span class="label">简单例子。</span>在 k=2 query 处，把第二个 target source 改坏。只恢复 attention pattern、只恢复 source value、或同时恢复；另把 k=8 progress state patch 到 k=3 query，看 head 是否从 occurrence 3 转向 occurrence 8。</div>
{_figure(generated['transport'], "图 7｜A：target-source corruption 的 normalized recovery。横轴为 pattern-only、target-source value-only、二者共同 patch；颜色表示 Top-1/Top-2 targeted heads；水平点线为完整 query residual patch。纵轴 1 表示恢复到 clean identity margin。B：progress k=8 state patch 到 k=3 后，各 Layer-4 head 对 occurrence 8 相对 occurrence 3 的 attention shift；柱为均值，误差线为 SEM。", "targeted retrieval pattern value mediation and state conditioned routing")}
<p>Top-2 pattern-only recovery 为 {_fmt(med_summary.loc[('attention_pattern_only', 2)])}，value-only 为 {_fmt(med_summary.loc[('value_only_at_target_source', 2)])}，pattern+value 为 {_fmt(med_summary.loc[('pattern_plus_value', 2)])}。因此 attention map 定位了路径，但 identity 主要由两只 head 的 target-source value vectors 携带。Progress patch 对 L4H2 与 L4H1 的 routing shift 分别为 {_fmt(routing_summary.loc[2])} 与 {_fmt(routing_summary.loc[1])}，同层其他 heads 明显较小。</p>
<div class="conclusion"><span class="label">本实验结论。</span>Thinking 的 targeted bank 不只是“看对位置”：它把对应 occurrence 的 value identity 传入 trace query；而 progress residual 又因果控制该 bank 下一步看哪一次 occurrence，形成 state → routing → identity transport 的闭环。</div>

<h3>5.3 Thinking：progress commit、stop execution 与 scope</h3>
<div class="purpose"><span class="label">实验目的。</span>区分显式 index token、item marker endpoint 和整个 item span 对 stop/continue 决策的贡献。</div>
<p>把 total=k 的 terminal trace state patch 到 total=k+2 的 continuing trace，在 L4 仅 patch marker 就让 {_pct(terminal_close)} 的样本选择 <code>&lt;/Think&gt;</code>；index-only 不翻转，index+marker 与 marker-only 相同；同 total 的 continuing donor control 为 {_pct(control_close)}。Layer 3 已出现约 12 logits 的 terminal margin shift，Layer 4 达约 19.9。</p>
<div class="example"><span class="label">简单例子。</span>Receiver 总数为 10，当前刚数到 k=8，本应继续输出 9；donor 总数正好为 8，本应 close。把 donor 的第 8 个 marker state 移到 receiver 的第 8 个 marker，即测试“这个局部 progress state 是否足以把 continue 改成 stop”。</div>
<div class="conclusion"><span class="label">本实验结论。</span>在 synthetic thinking 中，可执行的 terminal bit 最终集中在 marker endpoint，而不是显式 index 本身；因此“count 可解码”与“stop decision 实际读取哪个 token site”必须分开。</div>

<h3>5.4 Terminal readout：answer state、trace conflicts 与 bridge</h3>
<div class="purpose"><span class="label">实验目的。</span>确定最终答案来自 prompt-wide broad aggregation、显式 final index、trace length/position，还是一个可定位的 trace-to-answer component。</div>
<div class="example"><span class="label">简单例子。</span>把 N=8 trace 缩短成 7 项，answer query 位置随之左移；再把 clean N=8 在对应 answer-query 的 attention output 或 residual patch 回去，看 N vs 7 margin 能否恢复。</div>
{_figure(generated['terminal'], "图 8｜A：在 answer query patch ±1 count donor state；纵轴为 expected-count shift×donor offset，正值表示输出向 donor count 移动。B：shortened-trace corruption 后 patch clean component；纵轴为 normalized answer-margin recovery。C：trace conflict 行为；纵轴为仍预测原始 N 的比例，前五项保持长度，最后一项缩短 trace 并移动 answer-query 位置。", "terminal answer state donor transport bridge rescue and trace conflicts")}
<p>Thinking 的 answer state 从 L2 起几乎按 donor count 平移（signed shift≈1）；shortened trace 后，L2 attention output 单独恢复 {_fmt(bridge_l2)}，L2 post-layer residual 也恢复约 0.993。相反，保持长度的 final-index−1、copy-previous-pair、neutral-tail 等操作仍 100% 预测原始 N；只有真正缩短 trace 时 {_pct(shortened_follow)} 跟随 N−1。</p>
<div class="warning"><span class="label">关键差异。</span>这与“大模型最后位置存在 prompt-wide broad aggregation/readout”不是同一个机制。v20 的证据更符合<strong>trace length/position-sensitive terminal bridge</strong>：显式 trace scaffold 决定 answer-query 结构，Layer-2 attention 把它写入 answer state。</div>
<div class="conclusion"><span class="label">本实验结论。</span>我们没有在 synthetic thinking 中发现必须由最后位置 broad retrieval 聚合 prompt 的证据；终端读出主要依赖 trace structure 和 L2 answer-query attention bridge。</div>

<h3>5.5 对齐后的因果证据梯</h3>
{_table(causal_ladder)}
<div class="conclusion"><span class="label">本节总论。</span>因果链已经覆盖 evidence intake、head-bank necessity、identity transport、state-conditioned routing、stop execution 和 terminal bridge；每一步都有 matched control 或 corruption/restoration，不再依赖单独的 attention 可视化。</div>

<h2 id="dynamics">6. Training Dynamics</h2>
<h3>6.1 横轴：linear step 为主，log step 为补充</h3>
<p><a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html">Olsson et al. 的 induction-head 工作</a>核心不是“必须使用 log 横轴”，而是把宏观 behavior/loss 与微观 head formation、ablation effect 放在同一 training coordinate 上，并标出 phase-change window。<a href="https://arxiv.org/abs/2001.08361">Kaplan et al. 的 scaling laws</a>研究跨多个数量级的 power law，log-log 坐标适合估计斜率；这与单次 0–10k steps 的机制 emergence 问题不同。</p>
<p>因此本报告的主图使用<strong>线性 optimizer step</strong>：它保留 step 1,500 的真实距离、能判断角色形成是否同步，并避免 log 轴把早期噪声视觉放大成“突变”。log(1+step) 版本仅作 early-training 放大镜；是否发生突变由 changepoint/sigmoid width 和同步因果指标决定，不由坐标变换决定。</p>
{_figure(references['linear_log'], "图 9｜同一 broad/targeted role 曲线的线性横轴与 log(1+step) 横轴对照。纵轴完全相同；log 轴只展开早期 steps，不改变数据。虚线标出 step 1,500 objective switch。", "linear versus log optimizer step comparison")}
<div class="conclusion"><span class="label">本小节结论。</span>linear 是机制主图，log 是补充诊断。log 可以让早期变化更可见，但不能单独证明 phase transition。</div>

<h3>6.2 Role specialization 与 head-bank differentiation</h3>
<div class="purpose"><span class="label">实验目的。</span>检验 broad retrieval、targeted retrieval 与 marker successor 是否从初始近均匀的 16-head 分布分化成不同 bank，并确定分化的时间顺序。</div>
<div class="example"><span class="label">简单例子。</span>如果 16 只 heads 对 targeted role 得分相近，effective head count 接近 16；若两只 heads 承担绝大多数得分，effective count 接近 2，Top-2 share 接近 1。</div>
{_figure(references['role_specialization'], "图 10｜固定 final-selected heads 与所有 peer heads 的 role-score 轨迹。横轴为线性 optimizer step；纵轴为各 panel 定义的 attention role score。粗实线为 final frozen head，粗虚线为 final 排名第二，细线为其余 heads；颜色按 layer。Non-thinking 与 Thinking 是不同模型，只比较 specialization profile。", "per-head broad and targeted retrieval role specialization")}
{_figure(references['bank_differentiation'], "图 11｜Head-bank specialization。A 横轴 step、纵轴为 role-score distribution 的 entropy effective number；B 为 Top-2 share；C 为两种 normalized head-role map 的 Jensen–Shannon divergence（0 相同、1 不重叠）；D 为 step 10k 的 4×16 normalized role map，白框标出各 role Top-2。Thinking broad 在这里仅表示 final-query prompt coverage，不等同于已证明的 terminal readout。", "head bank effective size top two share and role map differentiation")}
<p>Final targeted bank 的 effective heads 为 {_fmt(final_bank.loc['targeted_retrieval','effective_heads'],2)}、Top-2 share {_fmt(final_bank.loc['targeted_retrieval','top2_share'])}；successor 为 {_fmt(final_bank.loc['marker_successor','effective_heads'],2)} / {_fmt(final_bank.loc['marker_successor','top2_share'])}。Thinking broad 与 targeted 的 normalized JS divergence 最终为 {_fmt(final_js)}，说明部分分化但并非完全 disjoint。Targeted role 的 sigmoid center 为 step {_fmt(target_window.smooth_center_step,0)}，10–90% width {_fmt(target_window.smooth_width_10_90_steps,0)}；successor center 仅 step {_fmt(successor_window.smooth_center_step,0)}。</p>
<div class="conclusion"><span class="label">本实验结论。</span>最明显的 role specialization 是 successor 先快速集中、targeted bank 后来在 L4 缓慢形成；broad 与 targeted 的 bank 有实质分化但仍共享部分 heads，因此应写成 functional differentiation，而不是完全模块化。</div>

<h3>6.3 Geometry emergence：scaffold 早于 targeted retrieval</h3>
<div class="purpose"><span class="label">实验目的。</span>在最终 discovery-selected 的同一物理层上向前追踪 geometry，判断 representation scaffold、retrieval routing 与行为 improvement 的先后关系。</div>
{_figure(generated['geometry_dynamics'], "图 12｜四个 endpoint 的 milestone geometry。横轴为线性 optimizer step；竖线为 step 1,500 objective switch。A 为 confirmation Logistic/NCC BA 均值；B 为 isotropic SNR dB；C 为 ordinal RSA。每条曲线固定追踪 final checkpoint 在 discovery 上选出的同一物理层，因此是 emergence 描述，不是每个 checkpoint 重新挑层的上界。", "geometry dynamics over optimizer steps")}
<p>Thinking item-end 与 answer-query 的 BA 在 step 500 已达到 1.0，而 targeted head 的形成中心在约 step 4,864；Non-thinking answer-query 的 geometry 则在 step 2k 后缓慢改善，step 10k 的小样本 dynamics BA 约 0.28。由此可见：显式 trace token/position scaffold 先变得可读，真正按 progress 定位 occurrence 的 retrieval circuit 晚约数千 steps 才形成。</p>
<div class="warning"><span class="label">不应过度解释。</span>早期 perfect geometry 不等价于早期就会数。v20 的 all-sequence loss 直接监督 trace index 与格式，模型可以先学会复制/编码显式 trace 结构，再学会把每个 index 与 prompt 中正确 occurrence 绑定。</div>
<div class="conclusion"><span class="label">本实验结论。</span>训练动态支持“scaffold-first, retrieval-later”：representation 与 routing 不是同一 phase；targeted retrieval 的 emergence 是渐进形成（宽窗口），而非 log 轴制造的尖锐突变。</div>

<h2 id="mechanism">7. 综合机制图景</h2>
<h3>7.1 Non-thinking</h3>
<p><span class="tag">Form</span>Target identity 在进入 Layer 1 前即可被 broad intake 读取，位置转移仍有效；局部 prompt state 不形成稳定 occurrence-index manifold。<span class="tag">Retrieve</span>Answer query 使用分布式 broad head bank 汇总 target evidence，ranked ablation 比随机更有害但集中度有限。<span class="tag">Consolidate</span>L2–L4 answer residual 可被 count donor/centroid steering，但最终 geometry 和行为都明显弱于 thinking。</p>
<div class="conclusion"><span class="label">Non-thinking 结论。</span>机制是“content-addressed broad evidence intake + late noisy consolidation”，而不是逐 occurrence 的显式 running counter；这解释了其低 AR accuracy 与弱 final separation。</div>

<h3>7.2 Thinking</h3>
<p><span class="tag">Scaffold</span>显式 trace index/marker geometry 很早形成。<span class="tag">Target</span>L4 targeted bank 随训练分化，progress state 控制 QK routing，Top-2 value vectors 携带 occurrence identity。<span class="tag">Commit</span>Marker endpoint 写入可执行的 stop/continue bit。<span class="tag">Readout</span>Layer-2 answer-query attention 把 trace length/position structure写入 terminal answer state。</p>
<div class="conclusion"><span class="label">Thinking 结论。</span>优势来自 temporal decomposition：每一步只检索一个 occurrence、更新一个 progress state，再通过 trace bridge 读出。它不是“最后再 broad sweep 一遍 prompt”。</div>

<h3>7.3 Broad retrieval 是否仍需关注？</h3>
<p>需要，但角色不同。Non-thinking broad retrieval 是主机制，必须作为核心对照；Thinking final-query prompt coverage 可作为描述性 control，帮助量化 bank differentiation，但目前没有因果证据把它称为 universal terminal aggregator。报告因此把 broad-vs-targeted 作为 routing 对比，把 thinking terminal readout 单独交给 trace-conflict 与 bridge experiments。</p>
<div class="conclusion"><span class="label">综合结论。</span>主叙事应聚焦 broad retrieval vs targeted retrieval，同时保留 successor/progress 与 terminal bridge 作为解释 targeted retrieval 如何变成可执行 counting 的必要下游环节。</div>

<h2 id="gaps">8. 与大模型结果的对齐和 gaps</h2>
<div class="purpose"><span class="label">比较目的。</span>判断 synthetic v20 到底复现了大模型报告中的哪一层主张：只是相似的 attention 图，还是相同的因果角色；并把任务/尺度造成的不可比性与真正的机制不一致分开。</div>

<h3>8.1 大模型 reference 的实际范围</h3>
{_table(large_model_reference_table)}
<p>这里“大模型结果”不是单一 homogeneous baseline。Non-thinking 的 form→broad retrieve→consolidate 链在 Qwen 与 Gemma 都有多项 matched-control 证据；Native-thinking 最强的自然 no-index counter-state transfer 则只在 Qwen3-8B、N=10 的特定 cohort 闭合。Gemma 的 Top-6 targeted/carrier 结果是重要跨模型参照，但不能写成它也已完成同一条自然 no-index 行为链。</p>
<div class="conclusion"><span class="label">Reference 边界结论。</span>对齐目标是大模型报告中已被对应实验支持的 claim，不是把所有 Qwen 结果、Gemma 结果和示意图合并成一个“普遍大模型机制”。</div>

<h3>8.2 逐机制比较：哪些对齐，哪些没有</h3>
<div class="warning"><span class="label">先给最重要的答案。</span><strong>最接近大模型的部分</strong>是 broad-vs-targeted 的功能分工，以及 progress state 对 targeted routing 的控制；<strong>最大的两个机制 gap</strong>是：(1) v20 Non-thinking 没有大模型式 prompt running-index representation；(2) v20 Thinking 的 progress/terminal signal 可由显式 index 与 trace length 解释，尚不是大模型 Qwen 的 natural no-index internal counter。</div>
{_table(mechanism_gap_table)}
<div class="example"><span class="label">如何区分“对齐”和“长得像”。</span>Thinking Top-2 heads 看向第 k 个 occurrence 只是定位；在 source corruption 下恢复这两只 heads 的 target values 能救回 identity margin，再用 progress-state patch 改变它们的目标，才构成功能对齐。相反，answer-query 的 prompt attention coverage 目前没有 source ablation/patch，因此只能叫描述性 broad score。</div>
<div class="conclusion"><span class="label">机制对齐结论。</span>当前证据支持“相同计算角色的受控 analogue”，不支持“相同微电路”。Targeted retrieval 的因果角色最接近；Non-thinking running formation、implicit counter、完整 serial mediation、free-running sufficiency 与 terminal source composition 仍有实质 gap。</div>

<h3>8.3 任务、模型与统计设计造成的不可比 gap</h3>
{_table(design_gap_table)}
<p>这些差异多数不是反例。例如 Qwen Top-128、Gemma Top-6 与 v20 Top-2 的 bank size 取决于总 head 数、层数、筛选分数尺度和功能分布，不能用“2 比 128 更集中”作跨模型结论。同样，v20 的 91.2% 与 33.5% AR accuracy 也不能直接与 counts 1–10 的大模型 accuracy 相减，因为输入长度、输出 grammar、count range 与训练状态都不同。</p>
<div class="conclusion"><span class="label">设计 gap 结论。</span>尺度、语义、tokenization 与 supervision 使绝对数值不可比；真正可比的是 intervention 定义后的方向性角色，例如“selected bank 是否比 matched random 更必要”或“修复 mediator 是否在同一 damaged arm 中救回 downstream state”。</div>

<h3>8.4 当前还不能写进论文主结论的强命题</h3>
<ul><li><b>不能写：</b>Thinking 普遍形成一个与可见数字无关的隐式 counter。当前 v20 直接监督 atomic index；只有大模型 Qwen 的自然 no-index cohort支持更强版本。</li><li><b>不能写：</b>Thinking 最终答案必然由最后位置的 prompt-wide broad head 聚合。Synthetic 没有该因果证据，大模型 Native report 也只把 prompt broad path 保留为未排除的并行路径。</li><li><b>不能写：</b>head-bank 分化在规模化模型训练中也约在相同步数或以相同顺序发生。大模型没有对应训练 checkpoints。</li><li><b>不能写：</b>v20 的 Top-2 是唯一 targeted circuit。Value recovery 很高，但 full bank/random necessity、downstream carrier mediation与自由生成仍允许冗余路径。</li><li><b>不能写：</b>Non-thinking 与 Thinking 的同坐标 head 是同一 head 发生角色转换。它们只在 step 0 同权重，随后是两个独立参数系统。</li></ul>
<div class="conclusion"><span class="label">Claim 边界结论。</span>最稳妥的论文表述是：v20 在配对训练中产生了 broad-vs-targeted 的功能分化，并显示 scaffold-first/retrieval-later；它揭示一种足以产生 thinking advantage 的机制，但还没有证明这是大模型自然 reasoning 的唯一或同构实现。</div>

<h2 id="limits">9. 限制与下一步</h2>
<ul><li><b>单 seed。</b>所有动态与因果估计来自 seed 1234；formation time 与 bank identity 目前是 descriptive case study。</li><li><b>显式 trace。</b>v20 每步包含数字 index，geometry 受到直接 token supervision；需要用 implicit-index 或 marker-only ablation 检验真正内部 counter。</li><li><b>Teacher-forced 局部实验。</b>多数 patch 只证明下一 token 的局部必要性/充分性；free-running global sufficiency 仍需 rollout intervention。</li><li><b>模型尺度。</b>3.19M 参数、4×4 heads 的 bank 很小；大模型可能使用更分布式或不同 terminal readout。</li><li><b>几何不是用途。</b>probe、SNR 与 RSA 只描述 residual states；真正的使用由 ablation/patching 支持。</li><li><b>跨模式 head identity。</b>两个模型独立训练，不比较 LxHy 的身份，只比较 bank profile。</li></ul>
<p><b>按 gap 优先级安排下一步：</b>(1) 训练 marker-only、shuffled-index 与 index-dropout controls，并用 padding 保持 answer-query 绝对位置，区分 counter content 与 trace length；(2) 把 selected targeted-bank damage → clean carrier restoration → later commit → free-running continuation 做成同一配对 serial mediation；(3) 对 Thinking answer query 做 prompt-source blank、trace-source blank、selected-head ablation 和 state restoration，直接测 broad path；(4) 为 Non-thinking 增加 multi-token needle span 版本，复测 prompt running representation 与 whole-span restoration；(5) 至少 4 个额外 training seeds，并在新 needle sets、长 context 和超出训练频率的 count distribution 上复验 formation windows。</p>
<div class="conclusion"><span class="label">本节结论。</span>现有证据足以支持 v20 的相对机制结论，但不足以把显式 trace geometry 等同于一般大模型的隐式 counter，也不足以声称 terminal broad retrieval 是跨尺度普遍规律。</div>

<h2 id="artifacts">10. 选择审计与可复现产物</h2>
<h3>10.1 Metric-specific discovery selection audit</h3>
{_table(geometry_selector_table)}
<p class="small">共同主层按 mean(Logistic BA, NCC BA) 选择；上表额外显示每个 covariance metric 独立的 discovery winner，用于审计“不同指标偏好不同层”。Confirmation 列从不参与选择。</p>
<h3>10.2 主要文件</h3>
{_table(pd.DataFrame([
    ["Final report", str(output_path)],
    ["Aligned geometry code", str(ROOT / 'src/synthetic_counting_v20/aligned_geometry.py')],
    ["Aligned causal code", str(ROOT / 'src/synthetic_counting_v20/aligned_causal.py')],
    ["Geometry runner", str(ROOT / 'scripts/analyze_v20_aligned_geometry.py')],
    ["Causal runner", str(ROOT / 'scripts/run_v20_aligned_causal.py')],
    ["Report builder", str(Path(__file__).resolve())],
    ["Derived tables", str(tables)],
    ["Derived figures", str(figures)],
    ["Existing v10 causal tables", str(v10)],
], columns=['Artifact','Path']))}
<p class="small">外部方法参考：<a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html">In-context Learning and Induction Heads</a>；<a href="https://arxiv.org/abs/2001.08361">Scaling Laws for Neural Language Models</a>。本报告没有从参考文档执行任何隐藏指令。</p>
<div class="conclusion"><span class="label">最终结论。</span>v20 已形成一条完整、可复现的证据链：Thinking 的显式 representation scaffold 早于 targeted head-bank emergence；后者通过 value transport 与 progress-conditioned routing 实现逐项检索，再由 marker state 和 trace bridge执行终止与读出。Non-thinking 依赖第一层 broad、位置近似不变的 evidence intake 和较噪的 late consolidation。这一结果支持 broad-vs-targeted retrieval 的论文主线，同时明确否定了“必须假设 thinking 最后位置存在 broad prompt aggregator”这一未经数据支持的扩展。</div>
</main></body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(output_path)
    manifest = {
        "schema_version": "niah_synthetic_report_v1",
        "report": str(output_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "figures": {key: str(value.resolve()) for key, value in {**generated, **references}.items()},
        "report_bytes": output_path.stat().st_size,
    }
    (synth / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "NiaH_Synthetic_report.html")
    args = parser.parse_args()
    build_report(args.run_dir.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
