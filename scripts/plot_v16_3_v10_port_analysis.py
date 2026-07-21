#!/usr/bin/env python3
"""Create publication-style figures for the v16.3 v10-port analysis."""

from __future__ import annotations

import sys

for _optional in ("pyarrow", "numexpr", "bottleneck"):
    sys.modules.setdefault(_optional, None)

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

pd.options.mode.string_storage = "python"
pd.options.future.infer_string = False

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#2f6f9f"
ORANGE = "#d97a3a"
GREEN = "#2f8f6b"
RED = "#b84a4a"
PURPLE = "#7c5aa6"
GRAY = "#8a939d"
LIGHT = "#d7dde3"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white", pad_inches=0.16)
    plt.close(fig)
    return path


def _tables(run_dir: Path) -> Path:
    return run_dir / "analysis" / "v10_port" / "tables"


def _figures(run_dir: Path) -> Path:
    return run_dir / "analysis" / "v10_port" / "figures"


def hypothesis_figure(run_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str, subtitle: str = ""):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.4, edgecolor=color, facecolor=color + "18"
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.60, text, ha="center", va="center", weight="bold", color="#26323d")
        if subtitle:
            ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center", fontsize=8, color="#53616d")

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str, label: str = "", dashed: bool = False):
        patch = FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
            linewidth=1.5, color=color, linestyle="--" if dashed else "-"
        )
        ax.add_patch(patch)
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center", fontsize=8, color=color)

    ax.text(0.2, 5.45, "H-NT: direct aggregation", fontsize=12, weight="bold", color=BLUE)
    box(0.2, 3.75, 2.1, 1.0, "Prompt targets", "#5b7184", "unordered occurrences")
    box(3.15, 3.75, 2.1, 1.0, "Broad heads", BLUE, "final <Ans> query")
    box(6.1, 3.75, 2.1, 1.0, "Count state", BLUE, "late Layer 4 residual")
    box(9.05, 3.75, 2.1, 1.0, "Answer", BLUE, "restricted count logits")
    for x1, x2 in ((2.3, 3.15), (5.25, 6.1), (8.2, 9.05)):
        arrow(x1, 4.25, x2, 4.25, BLUE)

    ax.text(0.2, 2.75, "H-CoT: externalized serial retrieval", fontsize=12, weight="bold", color=ORANGE)
    box(0.2, 1.05, 1.75, 1.0, "Prompt targets", "#5b7184", "ordered by position")
    box(2.45, 1.05, 1.75, 1.0, "k-to-k heads", ORANGE, "retrieve kth character")
    box(4.7, 1.05, 1.75, 1.0, "Trace progress", ORANGE, "index/marker state")
    box(6.95, 1.05, 1.75, 1.0, "Trace readout", PURPLE, "final <Ans> heads")
    box(9.2, 1.05, 1.75, 1.0, "Answer", ORANGE, "count from trace span")
    for x1, x2, color in ((1.95, 2.45, ORANGE), (4.2, 4.7, ORANGE), (6.45, 6.95, PURPLE), (8.7, 9.2, ORANGE)):
        arrow(x1, 1.55, x2, 1.55, color)
    arrow(5.58, 2.05, 5.58, 3.70, GRAY, "shared decodable count?", dashed=True)
    ax.text(
        11.65, 2.9,
        "Tests\n\nnecessity:\nablation\n\nsufficiency:\npatching\n\ngeometry:\nsteering",
        ha="center", va="center", fontsize=8.5, color="#46535f",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f3f5f7", ec="#aab2ba"),
    )
    ax.text(
        6, 0.28,
        "Solid arrows are tested by query-local intervention; the dashed cross-route is tested by cross-site decoding and state transport.",
        ha="center", va="center", fontsize=8, color="#5f6c77"
    )
    return _save(fig, _figures(run_dir) / "hypothesis_causal_map.png")


def _state_archive(run_dir: Path, mode: str):
    return np.load(
        run_dir
        / "analysis"
        / "checkpoint_dynamics"
        / "parts"
        / f"rope_{mode}_step_010000"
        / "heldout_states.npz"
    )


def _centroid_basis(values: np.ndarray, labels: np.ndarray, components: int = 3):
    unique = np.unique(labels)
    means = np.stack([values[labels == label].mean(axis=0) for label in unique])
    center = means.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(means - center, full_matrices=False)
    basis = vt[:components]
    return unique, means, center, basis, singular


def representation_2d(run_dir: Path) -> Path:
    panels = [
        ("nonthinking", "final_answer", 4, "Nonthinking answer / L4"),
        ("thinking", "final_answer", 2, "Thinking answer / L2"),
        ("thinking", "final_answer", 4, "Thinking answer / L4"),
        ("thinking", "trace_index", 3, "Trace index / L3"),
        ("thinking", "trace_marker", 3, "Trace marker / L3"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    cmap = plt.get_cmap("viridis")
    for ax, (mode, site, layer, title) in zip(axes.flat, panels, strict=False):
        archive = _state_archive(run_dir, mode)
        values = archive[f"{site}__{layer}__x"].astype(float)
        labels = archive[f"{site}__{layer}__y"].astype(int)
        unique, means, center, basis, singular = _centroid_basis(values, labels, 3)
        cloud = (values - center) @ basis[:2].T
        coords = (means - center) @ basis[:2].T
        for label in unique:
            mask = labels == label
            ax.scatter(cloud[mask, 0], cloud[mask, 1], s=8, alpha=0.10, color=cmap((label - 1) / 9))
        ax.plot(coords[:, 0], coords[:, 1], color="#59636c", lw=1.2, alpha=0.8)
        ax.scatter(coords[:, 0], coords[:, 1], c=unique, cmap="viridis", s=44, edgecolor="white", linewidth=0.6)
        for label, point in zip(unique, coords, strict=True):
            ax.text(point[0], point[1], str(int(label)), fontsize=7, ha="center", va="center")
        variance = singular**2 / np.maximum((singular**2).sum(), 1e-12)
        ax.set_title(title)
        ax.set_xlabel(f"Centroid PC1 ({100 * variance[0]:.1f}%)")
        ax.set_ylabel(f"Centroid PC2 ({100 * variance[1]:.1f}%)")
        ax.grid(alpha=0.18)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.05, 0.82, "Mean-first PCA", fontsize=13, weight="bold", transform=axes.flat[-1].transAxes
    )
    axes.flat[-1].text(
        0.05,
        0.68,
        "PCA is fitted to the ten class centroids.\nHeld-out examples are projected into that same basis.\n\nNumbers label count or progress k.\nPale points show within-class dispersion.\nAxes are panel-specific and are not comparable across panels.",
        fontsize=9,
        va="top",
        transform=axes.flat[-1].transAxes,
    )
    fig.suptitle("v16.3 mean-first residual manifolds: centroid path plus held-out sample cloud", y=1.01, fontsize=13)
    fig.tight_layout()
    return _save(fig, _figures(run_dir) / "representation_manifolds_2d.png")


def representation_3d(run_dir: Path) -> Path:
    panels = [
        ("nonthinking", "final_answer", 4, "Nonthinking answer / L4"),
        ("thinking", "final_answer", 2, "Thinking answer / L2"),
        ("thinking", "final_answer", 4, "Thinking answer / L4"),
        ("thinking", "trace_index", 3, "Trace index / L3"),
        ("thinking", "trace_marker", 3, "Trace marker / L3"),
    ]
    fig = plt.figure(figsize=(16, 10.2))
    cmap = plt.get_cmap("viridis")
    for panel_index, (mode, site, layer, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, panel_index, projection="3d")
        archive = _state_archive(run_dir, mode)
        values = archive[f"{site}__{layer}__x"].astype(float)
        labels = archive[f"{site}__{layer}__y"].astype(int)
        unique, means, center, basis, singular = _centroid_basis(values, labels, 3)
        cloud = (values - center) @ basis.T
        coords = (means - center) @ basis.T
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=labels, cmap="viridis", s=5, alpha=0.08)
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color="#4f5962", lw=1.5)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=unique, cmap="viridis", s=42, edgecolor="white")
        for label, point in zip(unique, coords, strict=True):
            if int(label) in {1, 5, 10}:
                ax.text(point[0], point[1], point[2], str(int(label)), fontsize=7)
        variance = singular**2 / np.maximum((singular**2).sum(), 1e-12)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 {100 * variance[0]:.0f}%", labelpad=3)
        ax.set_ylabel(f"PC2 {100 * variance[1]:.0f}%", labelpad=3)
        ax.set_zlabel(f"PC3 {100 * variance[2]:.0f}%", labelpad=3)
        ax.tick_params(labelsize=7, pad=0)
        ax.view_init(elev=24, azim=-55)
        ax.grid(alpha=0.15)
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    archive = _state_archive(run_dir, "thinking")
    centroids = []
    roles = []
    labels = []
    for site in ("trace_index", "trace_marker"):
        values = archive[f"{site}__3__x"].astype(float)
        y = archive[f"{site}__3__y"].astype(int)
        for label in range(1, 11):
            centroids.append(values[y == label].mean(axis=0))
            roles.append(site)
            labels.append(label)
    centroids = np.asarray(centroids)
    center = centroids.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centroids - center, full_matrices=False)
    coords = (centroids - center) @ vt[:3].T
    for site, marker in (("trace_index", "o"), ("trace_marker", "s")):
        idx = [i for i, value in enumerate(roles) if value == site]
        points = coords[idx]
        colors = np.asarray(labels)[idx]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=BLUE if site == "trace_index" else ORANGE, alpha=0.7)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, cmap="viridis", marker=marker, s=48, label=site.replace("trace_", ""))
    variance = singular**2 / np.maximum((singular**2).sum(), 1e-12)
    ax.set_title("Joint trace index/marker / L3")
    ax.set_xlabel(f"joint PC1 {100 * variance[0]:.0f}%")
    ax.set_ylabel(f"joint PC2 {100 * variance[1]:.0f}%")
    ax.set_zlabel(f"joint PC3 {100 * variance[2]:.0f}%")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.view_init(elev=24, azim=-55)
    ax.tick_params(labelsize=7, pad=0)
    fig.suptitle("Three-dimensional count/progress representations (mean-first PCA)", y=0.965, fontsize=13)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.045, top=0.91, wspace=0.08, hspace=0.16)
    return _save(fig, _figures(run_dir) / "representation_manifolds_3d.png")


def representation_geometry(run_dir: Path) -> Path:
    frame = pd.read_csv(_tables(run_dir) / "representation_geometry.csv")
    selections = [
        ("nonthinking", "final_answer", "NT answer"),
        ("thinking", "final_answer", "Thinking answer"),
        ("thinking", "trace_index", "Trace index"),
        ("thinking", "trace_marker", "Trace marker"),
    ]
    metrics = [
        ("pc1_to_pc3_variance", "Variance in PC1-PC3", (0, 1.05)),
        ("effective_dimension", "Effective dimension", None),
        ("mean_adjacent_displacement_cosine", "Adjacent-step cosine", (-1, 1)),
        ("path_straightness_chord_over_arc", "Chord / path length", (0, 1.05)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), sharex=True)
    for ax, (metric, title, limits) in zip(axes.flat, metrics, strict=True):
        for mode, site, label in selections:
            subset = frame[(frame["mode"] == mode) & (frame["site"] == site) & (frame["layer"] > 0)].sort_values("layer")
            color = BLUE if label == "NT answer" else (ORANGE if label == "Thinking answer" else (GREEN if label == "Trace index" else PURPLE))
            linestyle = "-" if "answer" in label else "--"
            ax.plot(subset["layer"], subset[metric], marker="o", color=color, linestyle=linestyle, label=label)
        ax.set_title(title)
        ax.set_xlabel("Layer output")
        ax.set_xticks([1, 2, 3, 4])
        if limits:
            ax.set_ylim(*limits)
        ax.axhline(0, color="#b4bbc2", lw=0.8)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Residual geometry is decodable but not globally one-dimensional", y=1.01, fontsize=13)
    fig.tight_layout()
    return _save(fig, _figures(run_dir) / "representation_geometry_summary.png")


def representation_dynamics(run_dir: Path) -> Path:
    geometry = pd.read_csv(run_dir / "tables" / "checkpoint_state_geometry.csv")
    probe = pd.read_csv(run_dir / "tables" / "checkpoint_state_probe_summary.csv")
    selections = [
        ("nonthinking", "final_answer", 4, BLUE, "NT answer L4"),
        ("thinking", "final_answer", 2, ORANGE, "Thinking answer L2"),
        ("thinking", "trace_marker", 3, GREEN, "Trace marker L3"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
    for mode, site, layer, color, label in selections:
        g = geometry[(geometry["mode"] == mode) & (geometry["site"] == site) & (geometry["layer"] == layer) & (geometry["context"] == "teacher_forced")].sort_values("step")
        p = probe[(probe["mode"] == mode) & (probe["site"] == site) & (probe["layer"] == layer) & (probe["context"] == "teacher_forced")].sort_values("step")
        axes[0, 0].plot(p["step"], p["ridge_r2"], color=color, label=label)
        axes[0, 1].plot(g["step"], g["pc1_variance"], color=color, label=label)
        axes[1, 0].plot(g["step"], g["effective_dimension"], color=color, label=label)
        axes[1, 1].plot(g["step"], g["pc1_adjacent_consistency"], color=color, label=label)
    titles = ["Held-out ridge R2", "Centroid PC1 variance", "Effective dimension", "Adjacent-direction consistency"]
    for ax, title in zip(axes.flat, titles, strict=True):
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.axvline(1500, color=GRAY, ls="--", lw=1)
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylim(-0.1, 1.05)
    axes[0, 1].set_ylim(0, 1)
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Representation learning dynamics across all 21 checkpoints", y=1.01, fontsize=13)
    fig.tight_layout()
    return _save(fig, _figures(run_dir) / "representation_learning_dynamics.png")


def head_ablation(run_dir: Path) -> Path:
    global_frame = pd.read_csv(_tables(run_dir) / "global_head_ablation.csv")
    local_frame = pd.read_csv(_tables(run_dir) / "position_local_head_ablation.csv")
    roles = [
        ("nonthinking_broad", "NT broad -> final count", BLUE),
        ("thinking_targeted", "Thinking targeted -> marker", ORANGE),
        ("thinking_readout", "Thinking readout -> final count", PURPLE),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11.8, 10), sharex=True, sharey="row")
    for row, (role, title, color) in enumerate(roles):
        for column, (frame, kind) in enumerate(((global_frame, "Global mask"), (local_frame, "Query-local mask"))):
            ax = axes[row, column]
            subset = frame[frame["role"] == role]
            ranked = subset[subset["path_kind"] == "ranked"].groupby("top_n")["accuracy"].mean()
            random_paths = subset[subset["path_kind"] == "random"].groupby(["path_id", "top_n"])["accuracy"].mean().unstack(0)
            x = ranked.index.to_numpy()
            ax.fill_between(x, random_paths.min(axis=1).reindex(x), random_paths.max(axis=1).reindex(x), color=LIGHT, alpha=0.8, label="random min-max")
            ax.plot(x, random_paths.mean(axis=1).reindex(x), color=GRAY, lw=1.3, label="random mean")
            ax.plot(x, ranked.to_numpy(), color=color, marker="o", lw=2.0, label="mechanism-ranked")
            ax.set_title(f"{title} | {kind}")
            ax.set_xlabel("Number of masked heads")
            ax.set_ylabel("Remaining accuracy")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Head necessity: ranked cumulative ablation versus deterministic random controls", y=1.005, fontsize=13)
    fig.tight_layout()
    return _save(fig, _figures(run_dir) / "head_ablation_global_local.png")


def retrieval_patching(run_dir: Path) -> Path:
    frame = pd.read_csv(_tables(run_dir) / "retrieval_head_patching.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, metric, title in (
        (axes[0], "normalized_recovery", "Clean marker-margin recovery"),
        (axes[1], "patched_correct", "Correct marker decision rate"),
    ):
        ranked = frame[frame["path_kind"] == "ranked"].groupby("top_n")[metric].mean()
        random_paths = frame[frame["path_kind"] == "random"].groupby(["path_id", "top_n"])[metric].mean().unstack(0)
        x = ranked.index.to_numpy()
        ax.fill_between(x, random_paths.min(axis=1).reindex(x), random_paths.max(axis=1).reindex(x), color=LIGHT, alpha=0.8)
        ax.plot(x, random_paths.mean(axis=1).reindex(x), color=GRAY, label="random mean")
        ax.plot(x, ranked, color=ORANGE, marker="o", lw=2, label="targeted ranking")
        ax.axhline(0 if metric == "normalized_recovery" else 1, color="#aeb5bc", ls="--", lw=0.9)
        if metric == "normalized_recovery":
            ax.axhline(1, color="#69737c", ls=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("Patched clean head slices")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Normalized recovery")
    axes[1].set_ylabel("Fraction")
    axes[0].legend(frameon=False)
    fig.suptitle("Count-preserving character corruption: k-to-k head-output sufficiency", y=1.02, fontsize=13)
    fig.tight_layout()
    return _save(fig, _figures(run_dir) / "retrieval_head_patching.png")


def successor_patching(run_dir: Path) -> Path:
    frame = pd.read_csv(_tables(run_dir) / "successor_head_patching.csv")
    titles = {
        "continue_to_close": "Continue donor -> close receiver",
        "close_to_continue": "Close donor -> continue receiver",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, direction in zip(axes, titles, strict=True):
        subset = frame[frame["direction"] == direction]
        ranked = subset[subset["path_kind"] == "ranked"].groupby("top_n")["normalized_recovery"].mean()
        random_paths = subset[subset["path_kind"] == "random"].groupby(["path_id", "top_n"])["normalized_recovery"].mean().unstack(0)
        x = ranked.index.to_numpy()
        ax.fill_between(x, random_paths.min(axis=1).reindex(x), random_paths.max(axis=1).reindex(x), color=LIGHT, alpha=0.8)
        ax.plot(x, random_paths.mean(axis=1).reindex(x), color=GRAY, label="random mean")
        ax.plot(x, ranked, color=GREEN if direction == "continue_to_close" else RED, marker="o", lw=2, label="causal ranking")
        ax.axhline(0, color="#aeb5bc", lw=0.8)
        ax.axhline(1, color="#69737c", ls=":", lw=1)
        ax.set_title(titles[direction])
        ax.set_xlabel("Patched head slices")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Normalized decision-margin recovery")
    axes[0].legend(frameon=False)
    fig.suptitle("Bidirectional successor/stop control at the same trace-marker query position", y=1.02, fontsize=13)
    fig.tight_layout()
    return _save(fig, _figures(run_dir) / "successor_head_patching.png")


def successor_conversion(run_dir: Path) -> Path:
    lens = pd.read_csv(_tables(run_dir) / "successor_residual_logit_lens.csv")
    component = pd.read_csv(_tables(run_dir) / "successor_component_evidence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    order = [(layer, stage) for layer in range(1, 5) for stage in ("pre", "post_attn", "post_mlp")]
    means = lens.groupby(["layer", "stage"])["clean_minus_short"].mean()
    x = np.arange(len(order))
    axes[0].plot(x, [means.loc[key] for key in order], color=GREEN, marker="o")
    axes[0].axhline(0, color=GRAY, lw=0.8)
    stage_labels = {"pre": "pre", "post_attn": "+attn", "post_mlp": "+MLP"}
    axes[0].set_xticks(x, [f"L{layer}\n{stage_labels[stage]}" for layer, stage in order], fontsize=8)
    axes[0].set_ylabel("Clean - short continue margin")
    axes[0].set_title("Residual logit lens")
    axes[0].grid(alpha=0.2, axis="y")
    summary = component.groupby(["layer", "component"])["clean_minus_short"].mean().unstack()
    width = 0.34
    layers = np.arange(1, 5)
    axes[1].bar(layers - width / 2, summary["attn_component"], width, label="attention output", color=ORANGE)
    axes[1].bar(layers + width / 2, summary["mlp_component"], width, label="MLP output", color=PURPLE)
    axes[1].axhline(0, color=GRAY, lw=0.8)
    axes[1].set_xticks(layers)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Direct-unembedding evidence difference")
    axes[1].set_title("Additive component evidence")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8)
    axes[1].grid(alpha=0.2, axis="y")
    fig.suptitle("Where continue-versus-close evidence enters and is converted", y=1.02, fontsize=13)
    fig.tight_layout()
    return _save(fig, _figures(run_dir) / "successor_logit_lens_components.png")


def mlp_features(run_dir: Path) -> Path:
    concentration = pd.read_csv(_tables(run_dir) / "successor_mlp_feature_concentration.csv")
    patching = pd.read_csv(_tables(run_dir) / "successor_mlp_feature_patching.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    for layer, color in ((3, GREEN), (4, PURPLE)):
        subset = concentration[concentration["layer"] == layer]
        axes[0, 0].plot(subset["support"], subset["positive_evidence_fraction"], marker="o", color=color, label=f"L{layer} positive")
        axes[0, 0].plot(subset["support"], subset["absolute_evidence_fraction"], marker="s", ls="--", color=color, alpha=0.75, label=f"L{layer} absolute")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xlabel("Ranked feature support")
    axes[0, 0].set_ylabel("Cumulative evidence fraction")
    axes[0, 0].set_title("Evidence concentration")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].grid(alpha=0.2)
    for column, direction in enumerate(("continue_to_close", "close_to_continue")):
        ax = axes[0, 1] if column == 0 else axes[1, 1]
        for layer, color in ((3, GREEN), (4, PURPLE)):
            subset = patching[(patching["direction"] == direction) & (patching["layer"] == layer)]
            ranked = subset[subset["path_kind"] == "ranked"].groupby("support")["normalized_recovery"].mean()
            random_mean = subset[subset["path_kind"] == "random"].groupby("support")["normalized_recovery"].mean()
            ax.plot(ranked.index + 1, ranked, marker="o", color=color, label=f"L{layer} ranked")
            ax.plot(random_mean.index + 1, random_mean, ls="--", color=color, alpha=0.45, label=f"L{layer} random")
        ax.set_xscale("log", base=2)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.axhline(1, color=GRAY, ls=":", lw=0.8)
        ax.set_xlabel("Patched features (+1 display offset)")
        ax.set_ylabel("Normalized recovery")
        ax.set_title(direction.replace("_", " "))
        ax.grid(alpha=0.2)
    axes[0, 1].legend(frameon=False, fontsize=8)
    ranked = patching[patching["path_kind"] == "ranked"].groupby(["layer", "support"])["decision_flipped"].mean().unstack(0)
    for layer, color in ((3, GREEN), (4, PURPLE)):
        axes[1, 0].plot(ranked.index + 1, ranked[layer], marker="o", color=color, label=f"L{layer}")
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xlabel("Patched features (+1 display offset)")
    axes[1, 0].set_ylabel("Decision-flip fraction")
    axes[1, 0].set_title("Both directions pooled")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.2)
    fig.suptitle("Layer-3/4 MLP evidence is distributed rather than single-neuron", y=1.005, fontsize=13)
    fig.tight_layout()
    return _save(fig, _figures(run_dir) / "successor_mlp_features.png")


def _origin_slope(frame: pd.DataFrame) -> float:
    x = frame["offset"].to_numpy(float)
    y = frame["expected_count_shift"].to_numpy(float)
    return float(x @ y / max(x @ x, 1e-12))


def head_transport(run_dir: Path) -> Path:
    frame = pd.read_csv(_tables(run_dir) / "final_query_head_transport.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, (mode, color) in zip(axes, (("nonthinking", BLUE), ("thinking", ORANGE)), strict=True):
        subset = frame[frame["mode"] == mode]
        ranked = subset[subset["path_kind"] == "ranked"].groupby("top_n").apply(_origin_slope, include_groups=False)
        random_paths = subset[subset["path_kind"] == "random"].groupby(["path_id", "top_n"]).apply(_origin_slope, include_groups=False).unstack(0)
        x = ranked.index.to_numpy()
        ax.fill_between(x, random_paths.min(axis=1).reindex(x), random_paths.max(axis=1).reindex(x), color=LIGHT, alpha=0.8)
        ax.plot(x, random_paths.mean(axis=1).reindex(x), color=GRAY, label="random mean")
        ax.plot(x, ranked, color=color, marker="o", lw=2, label="mechanism-ranked")
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.axhline(1, color=GRAY, ls=":", lw=1)
        ax.set_title(mode.capitalize())
        ax.set_xlabel("Patched donor head slices")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Expected-count transport slope")
    axes[0].legend(frameon=False)
    fig.suptitle("Do final-query head outputs carry donor count?", y=1.02, fontsize=13)
    fig.tight_layout()
    return _save(fig, _figures(run_dir) / "final_query_head_transport.png")


def trace_conflicts(run_dir: Path) -> Path:
    frame = pd.read_csv(_tables(run_dir) / "length_preserving_trace_conflicts.csv")
    order = [
        "prompt_minus_one_trace_clean",
        "trace_index_minus_one",
        "trace_pair_copy_previous",
        "marker_identity_control",
        "trace_tail_neutral_control",
        "shortened_trace_position_shifted",
    ]
    summary = frame.groupby(["intervention", "length_preserved"])[["follows_original_n", "follows_n_minus_1", "margin_change"]].mean().reset_index().set_index("intervention").reindex(order)
    labels = ["prompt n-1\ntrace n", "last index n-1", "copy previous pair", "marker identity", "neutral final pair", "remove final pair\n(position shifts)"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    width = 0.36
    axes[0].bar(x - width / 2, summary["follows_original_n"], width, color=ORANGE, label="argmax follows n")
    axes[0].bar(x + width / 2, summary["follows_n_minus_1"], width, color=BLUE, label="argmax follows n-1")
    axes[0].set_xticks(x, labels, rotation=24, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Fraction of examples")
    axes[0].set_title("Behavioral source attribution")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    colors = [GREEN if bool(value) else RED for value in summary["length_preserved"]]
    axes[1].bar(x, summary["margin_change"], color=colors)
    axes[1].axhline(0, color=GRAY, lw=0.8)
    axes[1].set_xticks(x, labels, rotation=24, ha="right")
    axes[1].set_ylabel("Change in margin z(n) - z(n-1)")
    axes[1].set_title("Green = length/Ans position preserved; red = shifted")
    for ax in axes:
        ax.grid(alpha=0.18, axis="y")
    fig.suptitle("Length-preserving conflicts isolate token identity from trace span", y=1.02, fontsize=13)
    fig.tight_layout()
    return _save(fig, _figures(run_dir) / "length_preserving_trace_conflicts.png")


def final_bridge(run_dir: Path) -> Path:
    frame = pd.read_csv(_tables(run_dir) / "final_bridge_component_patching.csv")
    summary = frame.groupby(["layer", "component"])["normalized_recovery"].agg(["mean", "sem"]).reset_index()
    components = ["attention_output", "mlp_output", "post_layer_residual"]
    colors = [ORANGE, PURPLE, GREEN]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    x = np.arange(1, 5)
    width = 0.24
    for index, (component, color) in enumerate(zip(components, colors, strict=True)):
        subset = summary[summary["component"] == component].set_index("layer").reindex(x)
        ax.bar(x + (index - 1) * width, subset["mean"], width, yerr=subset["sem"], color=color, label=component.replace("_", " "), capsize=2)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.axhline(1, color=GRAY, ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xlabel("Patched layer")
    ax.set_ylabel("Clean-to-shortened normalized recovery")
    ax.set_title("Final <Ans> bridge after removing the last trace pair")
    ax.legend(
        frameon=False,
        ncol=1,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    ax.grid(alpha=0.2, axis="y")
    return _save(fig, _figures(run_dir) / "final_bridge_component_recovery.png")


def residual_transport(run_dir: Path) -> Path:
    frame = pd.read_csv(_tables(run_dir) / "residual_count_transport.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    styles = [
        ("natural_donor", "Natural donor state", BLUE, "-"),
        ("centroid_delta_alpha_0.5", "Centroid delta, alpha=.5", ORANGE, "--"),
        ("centroid_delta_alpha_1", "Centroid delta, alpha=1", GREEN, "-"),
    ]
    for ax, mode in zip(axes, ("nonthinking", "thinking"), strict=True):
        for intervention, label, color, linestyle in styles:
            subset = frame[(frame["mode"] == mode) & (frame["intervention"] == intervention)]
            slopes = subset.groupby("layer").apply(_origin_slope, include_groups=False)
            ax.plot(slopes.index, slopes, marker="o", color=color, ls=linestyle, label=label)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.axhline(1, color=GRAY, ls=":", lw=1)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Residual patched after layer")
        ax.set_title(mode.capitalize())
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Expected-count transport slope")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Count-state causality: natural transplant and train-centroid steering", y=1.02, fontsize=13)
    fig.tight_layout()
    return _save(fig, _figures(run_dir) / "residual_count_transport.png")


def early_stop(run_dir: Path) -> Path:
    frame = pd.read_csv(_tables(run_dir) / "trace_early_stop_patching.csv")
    summary = frame.groupby("layer")[["close_margin_shift", "patched_close_decision"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    axes[0].bar(summary.index, summary["close_margin_shift"], color=GREEN)
    axes[0].axhline(0, color=GRAY, lw=0.8)
    axes[0].set_ylabel("Shift in z(close) - z(next index)")
    axes[0].set_title("Close-margin causal shift")
    axes[1].bar(summary.index, summary["patched_close_decision"], color=ORANGE)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Fraction predicting close")
    axes[1].set_title("Decision flip rate")
    for ax in axes:
        ax.set_xlabel("Donor final-marker state patched after layer")
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("Position-matched early stop: donor total=k into receiver interior marker k", y=1.02, fontsize=13)
    fig.tight_layout()
    return _save(fig, _figures(run_dir) / "trace_early_stop_patching.png")


def head_state(run_dir: Path) -> Path:
    head_to_state = pd.read_csv(_tables(run_dir) / "head_to_state_geometry.csv")
    state_to_head = pd.read_csv(_tables(run_dir) / "state_to_head_routing.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for mode, color in (("nonthinking", BLUE), ("thinking", ORANGE)):
        subset = head_to_state[head_to_state["mode"] == mode].groupby("top_n")[["state_accuracy", "output_accuracy"]].mean()
        axes[0].plot(subset.index, subset["state_accuracy"], color=color, marker="o", label=f"{mode} state")
        axes[0].plot(subset.index, subset["output_accuracy"], color=color, marker="s", ls="--", label=f"{mode} output")
    axes[0].set_xlabel("Query-local masked mechanism heads")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Head -> later count state and output")
    axes[0].legend(frameon=False, fontsize=8)
    routing = state_to_head.groupby("head")["routing_shift"].agg(["mean", "sem"])
    axes[1].bar([f"L3H{head}" for head in routing.index], routing["mean"], yerr=routing["sem"], color=GREEN, capsize=3)
    axes[1].axhline(0, color=GRAY, lw=0.8)
    axes[1].set_ylabel("Donor-k minus receiver-k attention shift")
    axes[1].set_title("Progress state -> k-to-k routing")
    axes[1].grid(alpha=0.2, axis="y")
    axes[0].grid(alpha=0.2)
    fig.suptitle("Bidirectional head-state causal relationship", y=1.02, fontsize=13)
    fig.tight_layout()
    return _save(fig, _figures(run_dir) / "head_state_bidirectional.png")


def plot_all(run_dir: Path) -> list[Path]:
    _style()
    builders = [
        hypothesis_figure,
        representation_2d,
        representation_3d,
        representation_geometry,
        representation_dynamics,
        head_ablation,
        retrieval_patching,
        successor_patching,
        successor_conversion,
        mlp_features,
        head_transport,
        trace_conflicts,
        final_bridge,
        residual_transport,
        early_stop,
        head_state,
    ]
    return [builder(run_dir) for builder in builders]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=ROOT / "colab_results" / "v16_3_main_rope_seed1234",
    )
    args = parser.parse_args()
    for path in plot_all(args.run_dir.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
