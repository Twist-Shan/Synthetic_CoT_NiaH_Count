from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _placeholder(path: Path, title: str) -> None:
    plt.figure(figsize=(6, 3.5))
    plt.text(0.5, 0.5, "No data", ha="center", va="center")
    plt.axis("off")
    plt.title(title)
    _save(path)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def make_probe_plots(run_dir: Path) -> None:
    figures = run_dir / "figures"
    probe = _read(run_dir / "tables" / "probe_results.csv")
    if probe.empty:
        _placeholder(figures / "probe_acc_by_layer_anchor.png", "Probe accuracy by layer/anchor")
        _placeholder(figures / "probe_r2_by_layer_anchor.png", "Probe R2 by layer/anchor")
        _placeholder(figures / "probe_minus_baseline_heatmap.png", "Probe minus baseline")
        return
    raw = probe[probe["raw_or_residualized"].eq("raw")]
    acc = raw[raw["probe_type"].eq("multiclass_logistic")]
    if acc.empty:
        acc = raw
    plt.figure(figsize=(10, 5))
    sns.barplot(data=acc, x="anchor_name", y="accuracy", hue="layer", errorbar=None)
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.title("v4 probe accuracy by layer and anchor")
    _save(figures / "probe_acc_by_layer_anchor.png")

    ridge = raw[raw["probe_type"].eq("ridge_scalar")]
    if ridge.empty:
        ridge = raw
    plt.figure(figsize=(10, 5))
    sns.barplot(data=ridge, x="anchor_name", y="r2", hue="layer", errorbar=None)
    plt.xticks(rotation=35, ha="right")
    plt.title("v4 ridge R2 by layer and anchor")
    _save(figures / "probe_r2_by_layer_anchor.png")

    acc = acc.copy()
    acc["minus_position"] = acc["accuracy"] - acc["position_baseline_acc"]
    mat = acc.pivot_table(index="anchor_name", columns="layer", values="minus_position", aggfunc="mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(mat, annot=True, fmt=".2f", center=0.0, cmap="vlag")
    plt.title("Probe accuracy minus position baseline")
    _save(figures / "probe_minus_baseline_heatmap.png")


def make_direction_plots(run_dir: Path) -> None:
    figures = run_dir / "figures"
    directions = _read(run_dir / "tables" / "direction_metrics.csv")
    if directions.empty:
        _placeholder(figures / "direction_cosine_heatmap.png", "Direction cosines")
        _placeholder(figures / "projection_by_count.png", "Projection by count")
        _placeholder(figures / "input_geometry_projection_trajectories.png", "Input geometry")
        return
    mat = directions.pivot_table(index="direction_type", columns="anchor_name", values="cosine_with_unembedding", aggfunc="mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="vlag", center=0.0)
    plt.title("Cosine with unembedding adjacent direction")
    _save(figures / "direction_cosine_heatmap.png")

    top = directions.sort_values("projection_r2", ascending=False).head(24)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top, x="anchor_name", y="projection_slope", hue="direction_type", errorbar=None)
    plt.xticks(rotation=35, ha="right")
    plt.title("Projection slope by anchor/direction")
    _save(figures / "projection_by_count.png")

    geom = _read(run_dir / "tables" / "input_geometry_results.csv")
    if geom.empty:
        _placeholder(figures / "input_geometry_projection_trajectories.png", "Input geometry")
    else:
        plt.figure(figsize=(8, 4.5))
        sns.barplot(data=geom, x="anchor_name", y="projection_slope", hue="direction_type", errorbar=None)
        plt.xticks(rotation=35, ha="right")
        plt.title("Input geometry projection summary")
        _save(figures / "input_geometry_projection_trajectories.png")


def make_steering_plots(run_dir: Path) -> None:
    figures = run_dir / "figures"
    steering = _read(run_dir / "tables" / "steering_results.csv")
    if steering.empty:
        _placeholder(figures / "steering_heatmap_anchor_layer.png", "Steering heatmap")
        _placeholder(figures / "steering_dose_response_top_configs.png", "Steering dose response")
        _placeholder(figures / "steering_controls.png", "Steering controls")
        return
    main = steering[steering["control_type"].eq("none")]
    mat = main.pivot_table(index="anchor_name", columns="layer", values="mean_count_shift", aggfunc="mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="vlag", center=0.0)
    plt.title("Mean count shift by anchor/layer")
    _save(figures / "steering_heatmap_anchor_layer.png")

    top_keys = (
        main.groupby(["model_type", "anchor_name", "layer", "direction_type"], as_index=False)["mean_count_shift"]
        .apply(lambda s: s.abs().max())
        .rename(columns={"mean_count_shift": "abs_shift"})
        .sort_values("abs_shift", ascending=False)
        .head(6)
    )
    merged = main.merge(top_keys[["model_type", "anchor_name", "layer", "direction_type"]])
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=merged, x="alpha", y="mean_pred_steered", hue="anchor_name", style="direction_type", marker="o", errorbar=None)
    plt.title("Top steering dose response")
    _save(figures / "steering_dose_response_top_configs.png")

    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=steering, x="control_type", y="mean_count_shift", hue="direction_type", errorbar=None)
    plt.xticks(rotation=20, ha="right")
    plt.title("Steering controls")
    _save(figures / "steering_controls.png")


def make_patching_plots(run_dir: Path) -> None:
    figures = run_dir / "figures"
    patch = _read(run_dir / "tables" / "interchange_patching_results.csv")
    if patch.empty:
        _placeholder(figures / "interchange_patch_matrix.png", "Interchange patch matrix")
        return
    mat = patch.pivot_table(index="receiver_count", columns="donor_count", values="causal_effect_size", aggfunc="mean")
    plt.figure(figsize=(6, 5))
    sns.heatmap(mat, annot=True, fmt=".1f", cmap="vlag", center=0.0)
    plt.title("Patched pred shift by receiver/donor count")
    _save(figures / "interchange_patch_matrix.png")


def make_all_plots(run_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    make_probe_plots(run_dir)
    make_direction_plots(run_dir)
    make_steering_plots(run_dir)
    make_patching_plots(run_dir)
