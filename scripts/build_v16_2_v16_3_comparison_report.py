#!/usr/bin/env python3
"""Build a self-contained Chinese report comparing v16.2 query-first and v16.3 query-last."""

from __future__ import annotations

import sys

for _optional in ("pyarrow", "numexpr", "bottleneck"):
    sys.modules.setdefault(_optional, None)

import argparse
import base64
import hashlib
import html
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

pd.options.mode.string_storage = "python"
pd.options.future.infer_string = False

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V16_2 = ROOT / "colab_results" / "v16_2_main_rope_seed1234"
DEFAULT_V16_3 = ROOT / "colab_results" / "v16_3_main_data-query_seed1234_20260721"
REPORT_NAME = "v16_2_vs_v16_3_query_order_report.html"

VERSION_ORDER = ("v16.2", "v16.3")
VERSION_LABEL = {
    "v16.2": "v16.2 · query-first",
    "v16.3": "v16.3 · query-last",
}
VERSION_COLOR = {"v16.2": "#3b6fb6", "v16.3": "#df7b3f"}
MODE_LABEL = {"nonthinking": "Nonthinking", "thinking": "Thinking"}


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    try:
        temporary.replace(path)
    except PermissionError:
        shutil.copyfile(temporary, path)
        temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    value = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _read(root: Path, filename: str) -> pd.DataFrame:
    path = root / "tables" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _final(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["step"] == int(frame["step"].max())].copy()


def _pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def _pp(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):+.{digits}f} pp"


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _mcnemar_exact(fixed: int, regressed: int) -> float:
    discordant = int(fixed + regressed)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(fixed, regressed) + 1))
    return min(1.0, 2.0 * tail / (2.0**discordant))


def _paired_bootstrap_ci(delta: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float]:
    values = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    chunk = 1_000
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _persistent_crossing(steps: np.ndarray, values: np.ndarray, threshold: float) -> int | None:
    order = np.argsort(steps)
    x = np.asarray(steps)[order]
    y = np.asarray(values, dtype=float)[order]
    for index in range(len(x)):
        if np.all(y[index:] >= threshold):
            return int(x[index])
    return None


def _aulc(steps: np.ndarray, values: np.ndarray) -> float:
    order = np.argsort(steps)
    x = np.asarray(steps, dtype=float)[order]
    y = np.asarray(values, dtype=float)[order]
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y, x) / (x[-1] - x[0]))


def _table(headers: Iterable[str], rows: Iterable[Iterable[object]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(
        f'<th class="num">{html.escape(str(value))}</th>' if index in numeric
        else f"<th>{html.escape(str(value))}</th>"
        for index, value in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="num">{value}</td>' if index in numeric else f"<td>{value}</td>"
            for index, value in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return '<div class="table-wrap"><table><thead><tr>' + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def _figure(path: Path, tag: str, title: str, caption: str, alt: str) -> str:
    return f"""
      <figure class="report-figure">
        <h3>{html.escape(title)}</h3>
        <img src="{_image_uri(path)}" alt="{html.escape(alt)}" loading="lazy">
        <figcaption><span class="figure-tag">{html.escape(tag)}</span>{caption}</figcaption>
      </figure>
    """


def _evidence(conclusions: list[str], gaps: list[str]) -> str:
    conclusion_items = "".join(f"<li>{item}</li>" for item in conclusions)
    gap_items = "".join(f"<li>{item}</li>" for item in gaps)
    return f"""
      <aside class="section-evidence-summary">
        <h3>本节小结：目前结论与证据缺口</h3>
        <div class="section-evidence-grid">
          <div class="section-evidence-column supported"><h4>目前可以得到的结论</h4><ul>{conclusion_items}</ul></div>
          <div class="section-evidence-column missing"><h4>欠缺的证据</h4><ul>{gap_items}</ul></div>
        </div>
      </aside>
    """


def _audit_initial_states(v16_2: Path, v16_3: Path) -> dict[str, object]:
    result: dict[str, object] = {"modes": {}}
    for mode in ("nonthinking", "thinking"):
        left_path = v16_2 / "checkpoints" / "rope" / mode / "step_000000" / "checkpoint.pt"
        right_path = v16_3 / "checkpoints" / "rope" / mode / "step_000000" / "checkpoint.pt"
        left = torch.load(left_path, map_location="cpu", weights_only=False)
        right = torch.load(right_path, map_location="cpu", weights_only=False)
        left_state = left["model_state_dict"]
        right_state = right["model_state_dict"]
        if list(left_state) != list(right_state):
            raise AssertionError(f"initial state keys differ for {mode}")
        maximum = max(
            float((left_state[key] - right_state[key]).abs().max().item())
            for key in left_state
        )
        result["modes"][mode] = {
            "state_entries": len(left_state),
            "max_abs_difference": maximum,
            "torch_rng_equal": bool(torch.equal(left["torch_rng_state"], right["torch_rng_state"])),
            "split_fingerprint_equal": left["split_fingerprint"] == right["split_fingerprint"],
        }
        del left, right, left_state, right_state
    return result


def _paired_effects(v16_2: Path, v16_3: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    specifications = {
        "TF final count": ("eval_detail.csv", "example_id", "tf_final_accuracy", 500),
        "AR final count": ("autoregressive_detail.csv", "row_id", "ar_accuracy", 100),
    }
    rows: list[dict[str, object]] = []
    paired_frames: dict[str, pd.DataFrame] = {}
    for protocol, (filename, id_column, metric, expected_n) in specifications.items():
        left = _final(_read(v16_2, filename))
        right = _final(_read(v16_3, filename))
        keys = ["mode", id_column, "set_id", "count", "corpus_start"]
        left_keys = left[keys].reset_index(drop=True)
        right_keys = right[keys].reset_index(drop=True)
        if not left_keys.equals(right_keys):
            raise AssertionError(f"paired metadata differs for {filename}")
        paired = left[keys + [metric]].merge(
            right[keys + [metric]], on=keys, validate="one_to_one", suffixes=("_v16_2", "_v16_3")
        )
        paired["protocol"] = protocol
        paired_frames[protocol] = paired
        for mode_index, mode in enumerate(("nonthinking", "thinking")):
            subset = paired[paired["mode"] == mode].copy()
            if len(subset) != expected_n:
                raise AssertionError(f"unexpected {protocol} sample count for {mode}: {len(subset)}")
            left_values = subset[f"{metric}_v16_2"].to_numpy(float)
            right_values = subset[f"{metric}_v16_3"].to_numpy(float)
            delta = right_values - left_values
            low, high = _paired_bootstrap_ci(delta, seed=16_230 + mode_index + 10 * len(rows))
            fixed = int(((left_values == 0) & (right_values == 1)).sum())
            regressed = int(((left_values == 1) & (right_values == 0)).sum())
            rows.append(
                {
                    "protocol": protocol,
                    "mode": mode,
                    "examples": len(subset),
                    "v16_2_accuracy": float(left_values.mean()),
                    "v16_3_accuracy": float(right_values.mean()),
                    "paired_gain": float(delta.mean()),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "fixed_errors": fixed,
                    "regressed_examples": regressed,
                    "mcnemar_exact_p": _mcnemar_exact(fixed, regressed),
                }
            )
    return pd.DataFrame(rows), paired_frames


def _learning_dynamics(roots: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for version, root in roots.items():
        detail = _read(root, "checkpoint_dynamics_autoregressive.csv")
        grouped = (
            detail.groupby(["mode", "step"], as_index=False)
            .agg(
                ar_accuracy=("ar_accuracy", "mean"),
                answer_rate=("ar_answered", "mean"),
                penalized_mae=("ar_abs_error_with_missing_penalty", "mean"),
                trace_exact=("trace_exact", "mean"),
                ordered_marker_accuracy=("trace_ordered_marker_accuracy", "mean"),
                marker_recall=("trace_marker_recall", "mean"),
                examples=("ar_accuracy", "size"),
            )
        )
        grouped["version"] = version
        curves.append(grouped)
        for mode in ("nonthinking", "thinking"):
            subset = grouped[grouped["mode"] == mode].sort_values("step")
            summaries.append(
                {
                    "version": version,
                    "mode": mode,
                    "diagnostic_examples_per_checkpoint": int(subset["examples"].iloc[0]),
                    "final_accuracy": float(subset["ar_accuracy"].iloc[-1]),
                    "accuracy_aulc": _aulc(subset["step"].to_numpy(), subset["ar_accuracy"].to_numpy()),
                    "persistent_50_step": _persistent_crossing(subset["step"].to_numpy(), subset["ar_accuracy"].to_numpy(), 0.5),
                    "persistent_80_step": _persistent_crossing(subset["step"].to_numpy(), subset["ar_accuracy"].to_numpy(), 0.8),
                    "persistent_90_step": _persistent_crossing(subset["step"].to_numpy(), subset["ar_accuracy"].to_numpy(), 0.9),
                }
            )
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summaries)


def _query_route_curves(roots: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    for version, root in roots.items():
        detail = _read(root, "checkpoint_attention_summary.csv").copy()
        source_column = "task_prefix_mass" if version == "v16.2" else "task_query_mass"
        detail["query_mass"] = detail[source_column]
        candidates = detail[
            (detail["step"] == int(detail["step"].max()))
            & (detail["diagnostic_split"] == "head_selection")
            & (detail["mode"] == "nonthinking")
            & (detail["query_kind"] == "final_answer")
        ]
        selected = candidates.sort_values("query_mass", ascending=False).iloc[0]
        layer, head = int(selected["layer"]), int(selected["head"])
        heldout = detail[
            (detail["diagnostic_split"] == "heldout_reporting")
            & (detail["mode"] == "nonthinking")
            & (detail["query_kind"] == "final_answer")
            & (detail["layer"] == layer)
            & (detail["head"] == head)
        ][["step", "query_mass"]].copy()
        heldout["version"] = version
        heldout["layer"] = layer
        heldout["head"] = head
        curves.append(heldout)
        selections.append(
            {
                "version": version,
                "layer": layer,
                "head": head,
                "selection_split_value": float(selected["query_mass"]),
                "heldout_final_value": float(heldout.loc[heldout["step"].idxmax(), "query_mass"]),
            }
        )
    return pd.concat(curves, ignore_index=True), pd.DataFrame(selections)


def _attention_routes(roots: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stability_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    query_curves, query_selection = _query_route_curves(roots)
    for version, root in roots.items():
        stability = _read(root, "checkpoint_head_stability.csv").copy()
        stability["version"] = version
        stability_parts.append(stability)
        final = stability[stability["step"] == int(stability["step"].max())]
        for role in ("needle_coverage", "needle_retrieval", "ordered_trace", "trace_routing"):
            row = final[final["role"] == role]
            if role in {"needle_coverage", "needle_retrieval"}:
                row = row[row["mode"] == "nonthinking"]
            if len(row) != 1:
                raise AssertionError(f"expected one final row for {version}/{role}, found {len(row)}")
            item = row.iloc[0]
            summary_rows.append(
                {
                    "version": version,
                    "route": role,
                    "mode": str(item["mode"]),
                    "metric": str(item["metric"]),
                    "layer": int(item["layer"]),
                    "head": int(item["head"]),
                    "heldout_value": float(item["heldout_value"]),
                    "heldout_rank": int(item["heldout_rank"]),
                }
            )
        query_item = query_selection[query_selection["version"] == version].iloc[0]
        summary_rows.append(
            {
                "version": version,
                "route": "answer_to_query",
                "mode": "nonthinking",
                "metric": "task query/prefix attention mass",
                "layer": int(query_item["layer"]),
                "head": int(query_item["head"]),
                "heldout_value": float(query_item["heldout_final_value"]),
                "heldout_rank": 1,
            }
        )
    return pd.concat(stability_parts, ignore_index=True), query_curves, pd.DataFrame(summary_rows)


def _representation_dynamics(roots: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_parts: list[pd.DataFrame] = []
    best_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for version, root in roots.items():
        frame = _read(root, "checkpoint_state_probe_summary.csv").copy()
        frame["version"] = version
        full_parts.append(frame)
        answer = frame[(frame["context"] == "teacher_forced") & (frame["site"] == "final_answer")]
        best = answer.groupby(["mode", "step"], as_index=False).agg(
            best_ridge_r2=("ridge_r2", "max"),
            best_centroid_accuracy=("nearest_centroid_accuracy", "max"),
        )
        best["version"] = version
        best_parts.append(best)
        final = answer[answer["step"] == int(answer["step"].max())]
        for mode in ("nonthinking", "thinking"):
            for layer in range(5):
                row = final[(final["mode"] == mode) & (final["layer"] == layer)].iloc[0]
                summary_rows.append(
                    {
                        "version": version,
                        "mode": mode,
                        "site": "final_answer",
                        "layer": layer,
                        "ridge_r2": float(row["ridge_r2"]),
                        "nearest_centroid_accuracy": float(row["nearest_centroid_accuracy"]),
                        "ridge_mae": float(row["ridge_mae"]),
                    }
                )
    return pd.concat(full_parts, ignore_index=True), pd.concat(best_parts, ignore_index=True), pd.DataFrame(summary_rows)


def _final_attention_head_maps(roots: dict[str, Path]) -> pd.DataFrame:
    """Collect the same four final attention roles for every layer/head."""

    role_specs = (
        ("answer_query_mass", "nonthinking", "final_answer"),
        ("direct_target_recall", "nonthinking", "final_answer"),
        ("ordered_retrieval_margin", "thinking", "trace_index"),
        ("trace_readout_mass", "thinking", "final_answer"),
    )
    rows: list[dict[str, object]] = []
    for version, root in roots.items():
        frame = _read(root, "checkpoint_attention_summary.csv")
        frame = frame[
            (frame["step"] == int(frame["step"].max()))
            & (frame["diagnostic_split"] == "heldout_reporting")
        ]
        query_column = "task_prefix_mass" if version == "v16.2" else "task_query_mass"
        metric_by_role = {
            "answer_query_mass": query_column,
            "direct_target_recall": "top_n_needle_recall",
            "ordered_retrieval_margin": "correct_top1_minus_chance",
            "trace_readout_mass": "trace_readout_mass",
        }
        for role, mode, query_kind in role_specs:
            subset = frame[(frame["mode"] == mode) & (frame["query_kind"] == query_kind)]
            if len(subset) != 16:
                raise AssertionError(
                    f"expected 16 final heads for {version}/{role}, found {len(subset)}"
                )
            metric = metric_by_role[role]
            for item in subset.itertuples(index=False):
                value = float(getattr(item, metric))
                if not np.isfinite(value):
                    raise AssertionError(f"non-finite {metric} for {version}/{role}")
                rows.append(
                    {
                        "version": version,
                        "role": role,
                        "mode": mode,
                        "query_kind": query_kind,
                        "layer": int(item.layer),
                        "head": int(item.head),
                        "value": value,
                        "source_metric": metric,
                    }
                )
    return pd.DataFrame(rows)


def _final_representation_geometry(roots: dict[str, Path]) -> pd.DataFrame:
    """Load final checkpoint centroid-geometry metrics shared by both runs."""

    parts: list[pd.DataFrame] = []
    sites = {
        ("nonthinking", "final_answer"),
        ("thinking", "final_answer"),
        ("thinking", "trace_marker"),
    }
    for version, root in roots.items():
        frame = _read(root, "checkpoint_state_geometry.csv")
        frame = frame[
            (frame["step"] == int(frame["step"].max()))
            & (frame["context"] == "teacher_forced")
            & (frame["layer"].between(1, 4))
        ].copy()
        frame = frame[
            frame.apply(lambda row: (str(row["mode"]), str(row["site"])) in sites, axis=1)
        ]
        expected = 3 * 4
        if len(frame) != expected:
            raise AssertionError(f"expected {expected} final geometry rows for {version}, found {len(frame)}")
        frame["version"] = version
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _mean_first_pca(values: np.ndarray, labels: np.ndarray, components: int = 6) -> dict[str, np.ndarray | float]:
    """PCA of class centroids, matching the v16 interactive geometry analysis."""

    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    unique = np.unique(labels)
    if values.ndim != 2 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("values must be [examples, hidden] and labels must align")
    means = np.stack([values[labels == label].mean(axis=0) for label in unique])
    centered = means - means.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    available = min(components, len(vt))
    basis = vt[:available].copy()
    coordinates = centered @ basis.T
    for axis in range(available):
        if axis == 0 and np.std(coordinates[:, axis]) > 1e-12:
            correlation = np.corrcoef(unique.astype(float), coordinates[:, axis])[0, 1]
            flip = bool(np.isfinite(correlation) and correlation < 0)
        else:
            pivot = int(np.argmax(np.abs(basis[axis])))
            flip = bool(basis[axis, pivot] < 0)
        if flip:
            basis[axis] *= -1
            coordinates[:, axis] *= -1
    if available < components:
        coordinates = np.pad(coordinates, ((0, 0), (0, components - available)))
    variance = singular**2
    total = float(variance.sum())
    full_ratio = variance / total if total > 1e-12 else np.zeros_like(variance)
    ratios = np.pad(full_ratio[:components], (0, max(0, components - len(full_ratio))))
    effective_dimension = float(1.0 / np.square(full_ratio).sum()) if total > 1e-12 else 0.0
    return {
        "labels": unique,
        "coordinates": coordinates,
        "variance": ratios,
        "effective_dimension": effective_dimension,
    }


def _hidden_state_centroid_pca(roots: dict[str, Path]) -> pd.DataFrame:
    """Build auditable real-state 3D centroid coordinates for selected sites."""

    selections = (
        ("nonthinking", "final_answer", 2, "NT answer / L2"),
        ("nonthinking", "final_answer", 4, "NT answer / L4"),
        ("thinking", "trace_marker", 3, "Thinking trace marker / L3"),
    )
    rows: list[dict[str, object]] = []
    for version, root in roots.items():
        archives: dict[str, np.lib.npyio.NpzFile] = {}
        try:
            for mode in {selection[0] for selection in selections}:
                state_path = (
                    root
                    / "analysis"
                    / "checkpoint_dynamics"
                    / "parts"
                    / f"rope_{mode}_step_010000"
                    / "heldout_states.npz"
                )
                if not state_path.is_file():
                    raise FileNotFoundError(state_path)
                archives[mode] = np.load(state_path, allow_pickle=False)
            for mode, site, layer, panel in selections:
                archive = archives[mode]
                prefix = f"{site}__{layer}"
                values = archive[f"{prefix}__x"]
                labels = archive[f"{prefix}__y"]
                geometry = _mean_first_pca(values, labels)
                coordinates = np.asarray(geometry["coordinates"])
                variance = np.asarray(geometry["variance"])
                for index, label in enumerate(np.asarray(geometry["labels"])):
                    rows.append(
                        {
                            "version": version,
                            "mode": mode,
                            "site": site,
                            "layer": layer,
                            "panel": panel,
                            "label": int(label),
                            "pc1": float(coordinates[index, 0]),
                            "pc2": float(coordinates[index, 1]),
                            "pc3": float(coordinates[index, 2]),
                            "pc1_variance": float(variance[0]),
                            "pc2_variance": float(variance[1]),
                            "pc3_variance": float(variance[2]),
                            "effective_dimension": float(geometry["effective_dimension"]),
                            "examples": int(np.sum(labels == label)),
                        }
                    )
        finally:
            for archive in archives.values():
                archive.close()
    return pd.DataFrame(rows)


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
        }
    )


def _mark_schedule(ax: plt.Axes) -> None:
    ax.axvline(1500, color="#202938", linestyle=":", linewidth=1.4, alpha=0.85)
    ax.text(1540, 0.035, "loss-scope switch", rotation=90, va="bottom", fontsize=8.5, color="#4b5563")


def _plot_behavior(
    roots: dict[str, Path], dynamics: pd.DataFrame, output: Path
) -> pd.DataFrame:
    per_count_parts: list[pd.DataFrame] = []
    for version, root in roots.items():
        frame = _final(_read(root, "autoregressive_detail.csv"))
        grouped = frame.groupby(["mode", "count"], as_index=False).agg(
            ar_accuracy=("ar_accuracy", "mean"), examples=("ar_accuracy", "size")
        )
        grouped["version"] = version
        per_count_parts.append(grouped)
    per_count = pd.concat(per_count_parts, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2), constrained_layout=True)
    for column, mode in enumerate(("nonthinking", "thinking")):
        ax = axes[0, column]
        for version in VERSION_ORDER:
            subset = dynamics[(dynamics["version"] == version) & (dynamics["mode"] == mode)].sort_values("step")
            ax.plot(
                subset["step"], subset["ar_accuracy"], marker="o", markersize=4.2,
                linewidth=2.2, color=VERSION_COLOR[version], label=VERSION_LABEL[version]
            )
        _mark_schedule(ax)
        ax.set(title=f"{MODE_LABEL[mode]}: diagnostic AR learning curve", xlabel="training step", ylabel="AR final-count accuracy")
        ax.set_ylim(-0.03, 1.04)
        ax.legend(loc="lower right", fontsize=9)

        ax = axes[1, column]
        for version in VERSION_ORDER:
            subset = per_count[(per_count["version"] == version) & (per_count["mode"] == mode)].sort_values("count")
            ax.plot(
                subset["count"], subset["ar_accuracy"], marker="o", markersize=5.2,
                linewidth=2.2, color=VERSION_COLOR[version], label=VERSION_LABEL[version]
            )
        ax.set(title=f"{MODE_LABEL[mode]}: final AR accuracy by true count", xlabel="true count n", ylabel="AR final-count accuracy")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(-0.03, 1.04)
        ax.legend(loc="lower right", fontsize=9)
    _save_figure(fig, output)
    return per_count


def _trace_final_metrics(roots: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for version, root in roots.items():
        frame = _final(_read(root, "autoregressive_detail.csv"))
        thinking = frame[frame["mode"] == "thinking"]
        for metric in ("trace_exact", "trace_marker_recall"):
            rows.append(
                {
                    "version": version,
                    "metric": metric,
                    "value": float(thinking[metric].mean()),
                    "examples": len(thinking),
                }
            )
    return pd.DataFrame(rows)


def _plot_loss_and_effects(
    roots: dict[str, Path], effects: pd.DataFrame, trace_metrics: pd.DataFrame, output: Path
) -> pd.DataFrame:
    component_parts: list[pd.DataFrame] = []
    for version, root in roots.items():
        frame = _read(root, "eval_loss_components.csv")
        frame = frame[
            (frame["curve_source"] == "heldout")
            & (frame["suite"] == "task")
            & (frame["component"].isin(["final_count", "trace_marker"]))
        ].copy()
        frame["version"] = version
        component_parts.append(frame)
    components = pd.concat(component_parts, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2), constrained_layout=True)
    for column, mode in enumerate(("nonthinking", "thinking")):
        ax = axes[0, column]
        for version in VERSION_ORDER:
            subset = components[
                (components["version"] == version)
                & (components["mode"] == mode)
                & (components["component"] == "final_count")
            ].sort_values("step")
            ax.semilogy(
                subset["step"], np.maximum(subset["example_mean_cross_entropy"], 1e-5),
                marker="o", markersize=4, linewidth=2.1, color=VERSION_COLOR[version], label=VERSION_LABEL[version]
            )
        ax.axvline(1500, color="#202938", linestyle=":", linewidth=1.4, alpha=0.85)
        ax.set(title=f"{MODE_LABEL[mode]}: held-out final-count loss", xlabel="training step", ylabel="final-count cross-entropy (log scale)")
        ax.legend(fontsize=9)

    ax = axes[1, 0]
    order = [
        ("TF final count", "nonthinking"),
        ("AR final count", "nonthinking"),
        ("TF final count", "thinking"),
        ("AR final count", "thinking"),
    ]
    selected = pd.concat(
        [effects[(effects["protocol"] == protocol) & (effects["mode"] == mode)] for protocol, mode in order],
        ignore_index=True,
    )
    positions = np.arange(len(selected))
    gains = selected["paired_gain"].to_numpy(float)
    low = selected["bootstrap_ci_low"].to_numpy(float)
    high = selected["bootstrap_ci_high"].to_numpy(float)
    colors = ["#4c78a8", "#2f5f95", "#f2a65a", "#d77b28"]
    ax.bar(positions, gains, color=colors, width=0.68)
    ax.errorbar(positions, gains, yerr=np.vstack([gains - low, high - gains]), fmt="none", ecolor="#172235", capsize=4, linewidth=1.3)
    ax.axhline(0, color="#172235", linewidth=1)
    ax.set_xticks(positions, ["NT TF", "NT AR", "T TF", "T AR"])
    ax.set(title="Paired query-last accuracy gain", xlabel="mode / evaluation protocol", ylabel="v16.3 − v16.2 accuracy")
    for x, value in zip(positions, gains):
        ax.text(x, value + 0.012, f"{100*value:+.1f} pp", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    width = 0.34
    metrics = ["trace_exact", "trace_marker_recall"]
    labels = ["trace exact", "marker recall"]
    x = np.arange(len(metrics))
    for offset, version in zip((-width / 2, width / 2), VERSION_ORDER):
        values = [float(trace_metrics[(trace_metrics["version"] == version) & (trace_metrics["metric"] == metric)]["value"].iloc[0]) for metric in metrics]
        ax.bar(x + offset, values, width=width, color=VERSION_COLOR[version], label=version)
        for xpos, value in zip(x + offset, values):
            ax.text(xpos, value + 0.015, f"{100*value:.1f}%", ha="center", va="bottom", fontsize=8.7)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set(title="Thinking final AR trace quality", xlabel="trace metric", ylabel="mean score")
    ax.legend(fontsize=9, loc="lower center")
    _save_figure(fig, output)
    return components


def _plot_attention_routes(
    stability: pd.DataFrame, query_curves: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2), constrained_layout=True)
    panels = [
        (axes[0, 0], "needle_coverage", "nonthinking", "NT fixed-head direct data coverage", "top-N target-occurrence recall"),
        (axes[1, 0], "ordered_trace", "thinking", "Thinking fixed-head ordered retrieval", "correct-top1 − chance"),
        (axes[1, 1], "trace_routing", "thinking", "Thinking fixed-head trace readout", "attention mass on trace span"),
    ]
    for ax, role, mode, title, ylabel in panels:
        for version in VERSION_ORDER:
            subset = stability[
                (stability["version"] == version)
                & (stability["role"] == role)
                & (stability["mode"] == mode)
            ].sort_values("step")
            ax.plot(
                subset["step"], subset["heldout_value"], marker="o", markersize=4.2,
                linewidth=2.2, color=VERSION_COLOR[version], label=VERSION_LABEL[version]
            )
        _mark_schedule(ax)
        ax.set(title=title, xlabel="training step", ylabel=ylabel)
        ax.legend(fontsize=9)

    ax = axes[0, 1]
    for version in VERSION_ORDER:
        subset = query_curves[query_curves["version"] == version].sort_values("step")
        label = f"{VERSION_LABEL[version]} · L{int(subset['layer'].iloc[0])}H{int(subset['head'].iloc[0])}"
        ax.plot(
            subset["step"], subset["query_mass"], marker="o", markersize=4.2,
            linewidth=2.2, color=VERSION_COLOR[version], label=label
        )
    _mark_schedule(ax)
    ax.set(title="NT fixed-head answer → query routing", xlabel="training step", ylabel="attention mass on five query tokens")
    ax.set_ylim(-0.03, 1.04)
    ax.legend(fontsize=9, loc="lower right")
    _save_figure(fig, output)


def _plot_representations(
    full: pd.DataFrame, best: pd.DataFrame, summary: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2), constrained_layout=True)
    for ax, metric, ylabel, title in (
        (axes[0, 0], "best_ridge_r2", "max held-out ridge R²", "NT best-layer count decodability"),
        (axes[0, 1], "best_centroid_accuracy", "max nearest-centroid accuracy", "NT best-layer discrete count separation"),
    ):
        for version in VERSION_ORDER:
            subset = best[(best["version"] == version) & (best["mode"] == "nonthinking")].sort_values("step")
            ax.plot(
                subset["step"], subset[metric], marker="o", markersize=4.2,
                linewidth=2.2, color=VERSION_COLOR[version], label=VERSION_LABEL[version]
            )
        _mark_schedule(ax)
        ax.set(title=title, xlabel="training step", ylabel=ylabel)
        ax.set_ylim(-0.15 if metric == "best_ridge_r2" else -0.03, 1.05)
        ax.legend(fontsize=9, loc="lower right")

    for column, mode in enumerate(("nonthinking", "thinking")):
        ax = axes[1, column]
        for version in VERSION_ORDER:
            subset = summary[(summary["version"] == version) & (summary["mode"] == mode)].sort_values("layer")
            ax.plot(
                subset["layer"], subset["ridge_r2"], marker="o", markersize=5,
                linewidth=2.2, color=VERSION_COLOR[version], label=version
            )
        ax.axhline(0, color="#6b7280", linewidth=0.9)
        ax.set_xticks(range(5), ["Emb", "L1", "L2", "L3", "L4"])
        ax.set(title=f"Final {MODE_LABEL[mode]} answer-state ridge R²", xlabel="residual depth", ylabel="held-out ridge R²")
        ax.set_ylim(-0.18, 1.05)
        ax.legend(fontsize=9, loc="lower right")
    _save_figure(fig, output)


def _plot_attention_head_maps(frame: pd.DataFrame, output: Path) -> None:
    """Show all 16 heads for four attention roles at the final checkpoint."""

    specs = (
        ("answer_query_mass", "Answer → query mass", "mass", "Blues", 0.0, 1.0),
        ("direct_target_recall", "Direct target recall", "top-N recall", "Blues", 0.0, 1.0),
        ("ordered_retrieval_margin", "Ordered retrieval", "top1 − chance", "coolwarm", -0.10, 0.70),
        ("trace_readout_mass", "Trace readout", "mass", "Blues", 0.0, 1.0),
    )
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.6), constrained_layout=True)
    for column, (role, title, colorbar_label, cmap, vmin, vmax) in enumerate(specs):
        images = []
        for row_index, version in enumerate(VERSION_ORDER):
            ax = axes[row_index, column]
            subset = frame[(frame["version"] == version) & (frame["role"] == role)]
            matrix = (
                subset.pivot(index="layer", columns="head", values="value")
                .reindex(index=range(1, 5), columns=range(4))
                .to_numpy(dtype=float)
            )
            if np.isnan(matrix).any():
                raise AssertionError(f"incomplete head matrix for {version}/{role}")
            image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            images.append(image)
            for layer_index in range(4):
                for head_index in range(4):
                    value = matrix[layer_index, head_index]
                    normalized = (value - vmin) / max(vmax - vmin, 1e-12)
                    text_color = "white" if normalized > 0.62 else "#182334"
                    ax.text(
                        head_index,
                        layer_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8.2,
                        color=text_color,
                        fontweight="bold",
                    )
            ax.set_xticks(range(4), [f"H{head}" for head in range(4)])
            ax.set_yticks(range(4), [f"L{layer}" for layer in range(1, 5)])
            ax.set_xlabel("head")
            if column == 0:
                ax.set_ylabel(f"{version}\nlayer")
            if row_index == 0:
                ax.set_title(title)
        fig.colorbar(images[-1], ax=axes[:, column], shrink=0.82, pad=0.025, label=colorbar_label)
    _save_figure(fig, output)


def _plot_representation_geometry(frame: pd.DataFrame, output: Path) -> None:
    """Compare the layerwise shape of count-centroid manifolds."""

    rows = (
        ("nonthinking", "final_answer", "NT final-answer state"),
        ("thinking", "final_answer", "Thinking final-answer state"),
        ("thinking", "trace_marker", "Thinking trace-marker state"),
    )
    columns = (
        ("effective_dimension", "effective dimension", (0.0, 8.0)),
        ("pc1_label_r2", "PC1 label R²", (-0.05, 1.05)),
        ("pc1_to_pc6_variance", "variance in PC1–PC6", (0.78, 1.01)),
    )
    fig, axes = plt.subplots(3, 3, figsize=(14.7, 10.4), sharex=True, constrained_layout=True)
    for row_index, (mode, site, row_title) in enumerate(rows):
        for column, (metric, ylabel, ylim) in enumerate(columns):
            ax = axes[row_index, column]
            for version in VERSION_ORDER:
                subset = frame[
                    (frame["version"] == version)
                    & (frame["mode"] == mode)
                    & (frame["site"] == site)
                ].sort_values("layer")
                ax.plot(
                    subset["layer"],
                    subset[metric],
                    marker="o",
                    markersize=5.2,
                    linewidth=2.2,
                    color=VERSION_COLOR[version],
                    label=VERSION_LABEL[version],
                )
            ax.set_title(f"{row_title}\n{ylabel}", fontsize=11.2)
            ax.set_ylim(*ylim)
            ax.set_xticks(range(1, 5), ["L1", "L2", "L3", "L4"])
            ax.set_xlabel("residual depth")
            ax.set_ylabel(ylabel)
    axes[0, 0].legend(loc="upper right", fontsize=8.8)
    _save_figure(fig, output)


def _plot_hidden_state_manifolds(frame: pd.DataFrame, output: Path) -> None:
    """Plot real checkpoint count centroids in independently fitted 3D PCA bases."""

    panels = ("NT answer / L2", "NT answer / L4", "Thinking trace marker / L3")
    fig = plt.figure(figsize=(15.4, 8.2), constrained_layout=True)
    cmap = plt.get_cmap("viridis", 10)
    for row_index, version in enumerate(VERSION_ORDER):
        for column, panel in enumerate(panels):
            ax = fig.add_subplot(2, 3, row_index * 3 + column + 1, projection="3d")
            subset = frame[(frame["version"] == version) & (frame["panel"] == panel)].sort_values("label")
            if len(subset) != 10:
                raise AssertionError(f"expected ten centroids for {version}/{panel}, found {len(subset)}")
            x = subset["pc1"].to_numpy()
            y = subset["pc2"].to_numpy()
            z = subset["pc3"].to_numpy()
            labels = subset["label"].to_numpy(dtype=int)
            ax.plot(x, y, z, color="#7a8796", linewidth=1.2, alpha=0.75)
            ax.scatter(
                x,
                y,
                z,
                c=labels,
                cmap=cmap,
                vmin=1,
                vmax=10,
                s=43,
                edgecolors="white",
                linewidths=0.6,
                depthshade=False,
            )
            for x_value, y_value, z_value, label in zip(x, y, z, labels, strict=True):
                ax.text(x_value, y_value, z_value, str(label), fontsize=7.4, ha="left", va="bottom")
            variance = subset.iloc[0]
            ax.set_xlabel(f"PC1 ({100 * variance.pc1_variance:.1f}%)", labelpad=6)
            ax.set_ylabel(f"PC2 ({100 * variance.pc2_variance:.1f}%)", labelpad=6)
            ax.set_zlabel(f"PC3 ({100 * variance.pc3_variance:.1f}%)", labelpad=6)
            ax.set_title(f"{version} · {panel}", pad=10)
            ax.view_init(elev=22, azim=-56)
            ax.set_box_aspect((1.0, 1.0, 0.82))
            ax.grid(True, alpha=0.25)
    _save_figure(fig, output)


def _value(frame: pd.DataFrame, **conditions: object) -> pd.Series:
    subset = frame
    for column, expected in conditions.items():
        subset = subset[subset[column] == expected]
    if len(subset) != 1:
        raise AssertionError(f"expected one row for {conditions}, found {len(subset)}")
    return subset.iloc[0]


def _comparison_audit(
    v16_2: Path,
    v16_3: Path,
    initial_audit: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    identical_files = (
        "tables/corpus_split.csv",
        "tables/needle_pool.csv",
        "tables/training_sampling_distribution.csv",
        "tables/model_specifications.csv",
    )
    rows: list[dict[str, object]] = []
    hashes: dict[str, dict[str, str | bool]] = {}
    for relative in identical_files:
        left_hash = _sha256(v16_2 / relative)
        right_hash = _sha256(v16_3 / relative)
        left_normalized = _normalized_text_sha256(v16_2 / relative)
        right_normalized = _normalized_text_sha256(v16_3 / relative)
        hashes[relative] = {
            "v16_2_raw": left_hash,
            "v16_3_raw": right_hash,
            "raw_equal": left_hash == right_hash,
            "normalized_v16_2": left_normalized,
            "normalized_v16_3": right_normalized,
            "normalized_equal": left_normalized == right_normalized,
        }
        rows.append(
            {
                "audit_item": relative,
                "v16_2": left_hash,
                "v16_3": right_hash,
                "equal": left_normalized == right_normalized,
            }
        )
    for mode, values in initial_audit["modes"].items():
        rows.append(
            {
                "audit_item": f"initial model state / {mode}",
                "v16_2": f"{values['state_entries']} entries",
                "v16_3": f"max |Δ|={values['max_abs_difference']:.1f}",
                "equal": bool(values["max_abs_difference"] == 0 and values["torch_rng_equal"]),
            }
        )
    if not all(bool(row["equal"]) for row in rows):
        failures = [str(row["audit_item"]) for row in rows if not row["equal"]]
        raise AssertionError(f"comparison controls failed: {failures}")
    return pd.DataFrame(rows), {"file_hashes": hashes, "initial_states": initial_audit}


def _counterfactual_summary(roots: dict[str, Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for version, root in roots.items():
        frame = _final(_read(root, "checkpoint_counterfactual_trace_readout.csv"))
        frame = frame[frame["condition"] == "remove_final_pair"]
        grouped = frame.groupby("layer", as_index=False).agg(
            delta_gold_probability=("delta_gold_probability", "mean"),
            delta_gold_logit_margin=("delta_gold_logit_margin_vs_count_minus_one", "mean"),
            delta_ridge_count_prediction=("delta_ridge_count_prediction", "mean"),
            examples=("prompt_sha256", "size"),
        )
        grouped["version"] = version
        parts.append(grouped)
    return pd.concat(parts, ignore_index=True)


def _validate_report(output: Path, output_dir: Path, manifest: dict[str, object]) -> None:
    document = output.read_text(encoding="utf-8")
    checks = {
        "sections": document.count("<section id=") == 11,
        "figures": document.count("<figure class=\"report-figure\">") == 7,
        "images": document.count("data:image/png;base64,") == 7,
        "captions": document.count("<figcaption>") == 7,
        "evidence_blocks": document.count("class=\"section-evidence-summary\"") == 11,
        "unicode": "\ufffd" not in document,
        "query_first_definition": "query → data → output" in document,
        "query_last_definition": "data → query → output" in document,
        "metric_definitions": all(
            phrase in document
            for phrase in (
                "Paired gain 与区间",
                "Accuracy AULC",
                "Attention route 指标",
                "Representation 指标",
                "Head map 与 centroid geometry",
                "3D centroid-PCA 图",
            )
        ),
        "report_hash": _sha256(output) == manifest["report_sha256"],
    }
    for filename, expected_hash in manifest["figures"].items():
        checks[f"figure:{filename}"] = _sha256(output_dir / "figures" / filename) == expected_hash
    for filename, expected_hash in manifest["tables"].items():
        checks[f"table:{filename}"] = _sha256(output_dir / "tables" / filename) == expected_hash
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise AssertionError(f"comparison report validation failed: {failures}")


def build(v16_2: Path, v16_3: Path) -> Path:
    v16_2 = v16_2.resolve()
    v16_3 = v16_3.resolve()
    roots = {"v16.2": v16_2, "v16.3": v16_3}
    for root in roots.values():
        if not (root / "config.json").is_file():
            raise FileNotFoundError(root / "config.json")

    output_dir = v16_3 / "analysis" / "v16_2_vs_v16_3"
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        version: json.loads((root / "config.json").read_text(encoding="utf-8"))
        for version, root in roots.items()
    }
    if configs["v16.2"]["seed"] != configs["v16.3"]["seed"]:
        raise AssertionError("seeds differ")
    initial_audit = _audit_initial_states(v16_2, v16_3)
    audit_table, audit_payload = _comparison_audit(v16_2, v16_3, initial_audit)
    effects, _ = _paired_effects(v16_2, v16_3)
    dynamics, learning_summary = _learning_dynamics(roots)
    stability, query_curves, route_summary = _attention_routes(roots)
    state_full, state_best, representation_summary = _representation_dynamics(roots)
    attention_head_maps = _final_attention_head_maps(roots)
    representation_geometry = _final_representation_geometry(roots)
    hidden_state_centroids = _hidden_state_centroid_pca(roots)
    trace_metrics = _trace_final_metrics(roots)
    counterfactual = _counterfactual_summary(roots)

    behavior_figure = figure_dir / "behavior_learning_dynamics.png"
    effect_figure = figure_dir / "loss_and_paired_effects.png"
    attention_figure = figure_dir / "attention_route_dynamics.png"
    representation_figure = figure_dir / "representation_dynamics.png"
    attention_heads_figure = figure_dir / "attention_head_maps.png"
    representation_geometry_figure = figure_dir / "representation_geometry_comparison.png"
    hidden_state_manifold_figure = figure_dir / "hidden_state_manifolds_3d.png"
    _plot_style()
    per_count = _plot_behavior(roots, dynamics, behavior_figure)
    loss_components = _plot_loss_and_effects(roots, effects, trace_metrics, effect_figure)
    _plot_attention_routes(stability, query_curves, attention_figure)
    _plot_representations(state_full, state_best, representation_summary, representation_figure)
    _plot_attention_head_maps(attention_head_maps, attention_heads_figure)
    _plot_representation_geometry(representation_geometry, representation_geometry_figure)
    _plot_hidden_state_manifolds(hidden_state_centroids, hidden_state_manifold_figure)

    tables = {
        "setting_and_identity_audit.csv": audit_table,
        "paired_final_behavior.csv": effects,
        "checkpoint_learning_summary.csv": learning_summary,
        "checkpoint_ar_curves.csv": dynamics,
        "final_ar_by_count.csv": per_count,
        "attention_route_summary.csv": route_summary,
        "answer_query_routing_curves.csv": query_curves,
        "attention_head_maps.csv": attention_head_maps,
        "representation_summary.csv": representation_summary,
        "representation_best_layer_curves.csv": state_best,
        "representation_geometry_final.csv": representation_geometry,
        "hidden_state_centroid_pca.csv": hidden_state_centroids,
        "thinking_trace_final_metrics.csv": trace_metrics,
        "trace_remove_final_pair_summary.csv": counterfactual,
        "heldout_output_loss_components.csv": loss_components,
    }
    for filename, frame in tables.items():
        frame.to_csv(table_dir / filename, index=False, lineterminator="\n")

    nt_tf = _value(effects, protocol="TF final count", mode="nonthinking")
    nt_ar = _value(effects, protocol="AR final count", mode="nonthinking")
    th_tf = _value(effects, protocol="TF final count", mode="thinking")
    th_ar = _value(effects, protocol="AR final count", mode="thinking")
    nt_l162 = _value(learning_summary, version="v16.2", mode="nonthinking")
    nt_l163 = _value(learning_summary, version="v16.3", mode="nonthinking")
    th_l162 = _value(learning_summary, version="v16.2", mode="thinking")
    th_l163 = _value(learning_summary, version="v16.3", mode="thinking")
    nt_cov_162 = _value(route_summary, version="v16.2", route="needle_coverage")
    nt_cov_163 = _value(route_summary, version="v16.3", route="needle_coverage")
    nt_enrich_162 = _value(route_summary, version="v16.2", route="needle_retrieval")
    nt_enrich_163 = _value(route_summary, version="v16.3", route="needle_retrieval")
    nt_query_162 = _value(route_summary, version="v16.2", route="answer_to_query")
    nt_query_163 = _value(route_summary, version="v16.3", route="answer_to_query")
    ordered_162 = _value(route_summary, version="v16.2", route="ordered_trace")
    ordered_163 = _value(route_summary, version="v16.3", route="ordered_trace")
    readout_162 = _value(route_summary, version="v16.2", route="trace_routing")
    readout_163 = _value(route_summary, version="v16.3", route="trace_routing")
    nt_l2_162 = _value(representation_summary, version="v16.2", mode="nonthinking", layer=2)
    nt_l2_163 = _value(representation_summary, version="v16.3", mode="nonthinking", layer=2)
    nt_l4_162 = _value(representation_summary, version="v16.2", mode="nonthinking", layer=4)
    nt_l4_163 = _value(representation_summary, version="v16.3", mode="nonthinking", layer=4)
    nt_geometry_l2_162 = _value(representation_geometry, version="v16.2", mode="nonthinking", site="final_answer", layer=2)
    nt_geometry_l2_163 = _value(representation_geometry, version="v16.3", mode="nonthinking", site="final_answer", layer=2)
    nt_geometry_l3_162 = _value(representation_geometry, version="v16.2", mode="nonthinking", site="final_answer", layer=3)
    nt_geometry_l3_163 = _value(representation_geometry, version="v16.3", mode="nonthinking", site="final_answer", layer=3)
    nt_geometry_l4_162 = _value(representation_geometry, version="v16.2", mode="nonthinking", site="final_answer", layer=4)
    nt_geometry_l4_163 = _value(representation_geometry, version="v16.3", mode="nonthinking", site="final_answer", layer=4)
    marker_geometry_l3_162 = _value(representation_geometry, version="v16.2", mode="thinking", site="trace_marker", layer=3)
    marker_geometry_l3_163 = _value(representation_geometry, version="v16.3", mode="thinking", site="trace_marker", layer=3)
    marker_geometry_l4_162 = _value(representation_geometry, version="v16.2", mode="thinking", site="trace_marker", layer=4)
    marker_geometry_l4_163 = _value(representation_geometry, version="v16.3", mode="thinking", site="trace_marker", layer=4)
    query_heads_above_80 = {
        version: int(
            (
                attention_head_maps[
                    (attention_head_maps["version"] == version)
                    & (attention_head_maps["role"] == "answer_query_mass")
                ]["value"]
                > 0.80
            ).sum()
        )
        for version in VERSION_ORDER
    }
    coverage_heads_above_50 = {
        version: int(
            (
                attention_head_maps[
                    (attention_head_maps["version"] == version)
                    & (attention_head_maps["role"] == "direct_target_recall")
                ]["value"]
                > 0.50
            ).sum()
        )
        for version in VERSION_ORDER
    }
    trace_exact_162 = _value(trace_metrics, version="v16.2", metric="trace_exact")
    trace_exact_163 = _value(trace_metrics, version="v16.3", metric="trace_exact")
    trace_recall_162 = _value(trace_metrics, version="v16.2", metric="trace_marker_recall")
    trace_recall_163 = _value(trace_metrics, version="v16.3", metric="trace_marker_recall")
    cf_l4_162 = _value(counterfactual, version="v16.2", layer=4)
    cf_l4_163 = _value(counterfactual, version="v16.3", layer=4)

    cfg = configs["v16.3"]
    model_specs = _read(v16_3, "model_specifications.csv")
    parameters = int(model_specs["parameters"].iloc[0])
    setting_rows = [
        ["Sequence layout", "query → data → output", "data → query → output", "唯一有意改变的训练接口"],
        ["Token positions before output", "query 1–5; data 6–261", "data 1–256; query 257–261", "output 均从 position 262 开始"],
        ["Causal visibility", "每个 data token 可看到 query", "query token 可看到全部 data", "计算方向发生反转"],
        ["Query-to-output RoPE distance", "257–261", "1–5", "query-last 大幅缩短 readout 距离"],
        ["Data-to-output RoPE distance", "1–256", "6–261", "data 整体远移 5 个位置"],
        ["Architecture", f"4L × 4H, d=256, MLP=1024", f"4L × 4H, d=256, MLP=1024", f"{parameters:,} parameters"],
        ["Training", "10,000 steps; batch 128; seed 1234", "相同", "AdamW, lr=3e-4, wd=0.01"],
        ["Loss schedule", "step≤1500 all-sequence; later task-output", "相同", "输出 token 数与权重相同"],
        ["Data/noise", "同一 Shakespeare split 与 needle pool", "语义内容相同", "split/pool 原始 hash 相同；sampling CSV 仅换行编码不同"],
        ["Initial state", "step-0 checkpoint", "51 state entries 全部相等", "两模式 max |Δ| = 0"],
    ]
    final_effect_rows = []
    for row in (nt_tf, nt_ar, th_tf, th_ar):
        p_text = f"{float(row['mcnemar_exact_p']):.3g}"
        final_effect_rows.append(
            [
                f"{MODE_LABEL[str(row['mode'])]} / {row['protocol']}",
                int(row["examples"]),
                _pct(row["v16_2_accuracy"]),
                _pct(row["v16_3_accuracy"]),
                _pp(row["paired_gain"]),
                f"[{_pp(row['bootstrap_ci_low'])}, {_pp(row['bootstrap_ci_high'])}]",
                f"{int(row['fixed_errors'])} / {int(row['regressed_examples'])}",
                p_text,
            ]
        )
    learning_rows = []
    for row in (nt_l162, nt_l163, th_l162, th_l163):
        crossing_50 = int(row["persistent_50_step"]) if pd.notna(row["persistent_50_step"]) else "—"
        crossing_80 = int(row["persistent_80_step"]) if pd.notna(row["persistent_80_step"]) else "—"
        crossing_90 = int(row["persistent_90_step"]) if pd.notna(row["persistent_90_step"]) else "—"
        learning_rows.append(
            [
                f"{VERSION_LABEL[str(row['version'])]} / {MODE_LABEL[str(row['mode'])]}",
                _pct(row["final_accuracy"]),
                _fmt(row["accuracy_aulc"]),
                crossing_50,
                crossing_80,
                crossing_90,
            ]
        )
    attention_rows = [
        ["NT direct data coverage", f"L{int(nt_cov_162['layer'])}H{int(nt_cov_162['head'])}", _fmt(nt_cov_162.heldout_value), f"L{int(nt_cov_163['layer'])}H{int(nt_cov_163['head'])}", _fmt(nt_cov_163.heldout_value), "top-N recall"],
        ["NT target enrichment", f"L{int(nt_enrich_162['layer'])}H{int(nt_enrich_162['head'])}", _fmt(nt_enrich_162.heldout_value), f"L{int(nt_enrich_163['layer'])}H{int(nt_enrich_163['head'])}", _fmt(nt_enrich_163.heldout_value), "needle enrichment"],
        ["NT answer → query", f"L{int(nt_query_162['layer'])}H{int(nt_query_162['head'])}", _fmt(nt_query_162.heldout_value), f"L{int(nt_query_163['layer'])}H{int(nt_query_163['head'])}", _fmt(nt_query_163.heldout_value), "query attention mass"],
        ["Thinking ordered retrieval", f"L{int(ordered_162['layer'])}H{int(ordered_162['head'])}", _fmt(ordered_162.heldout_value), f"L{int(ordered_163['layer'])}H{int(ordered_163['head'])}", _fmt(ordered_163.heldout_value), "top1 − chance"],
        ["Thinking trace readout", f"L{int(readout_162['layer'])}H{int(readout_162['head'])}", _fmt(readout_162.heldout_value), f"L{int(readout_163['layer'])}H{int(readout_163['head'])}", _fmt(readout_163.heldout_value), "trace attention mass"],
    ]
    representation_rows = [
        ["NT answer / L2", _fmt(nt_l2_162.ridge_r2), _pct(nt_l2_162.nearest_centroid_accuracy), _fmt(nt_l2_163.ridge_r2), _pct(nt_l2_163.nearest_centroid_accuracy)],
        ["NT answer / L4", _fmt(nt_l4_162.ridge_r2), _pct(nt_l4_162.nearest_centroid_accuracy), _fmt(nt_l4_163.ridge_r2), _pct(nt_l4_163.nearest_centroid_accuracy)],
    ]
    geometry_rows = [
        ["NT answer / L2", _fmt(nt_geometry_l2_162.pc1_label_r2), _fmt(nt_geometry_l2_162.effective_dimension, 2), int(nt_geometry_l2_162.monotonic_order_violations), _fmt(nt_geometry_l2_163.pc1_label_r2), _fmt(nt_geometry_l2_163.effective_dimension, 2), int(nt_geometry_l2_163.monotonic_order_violations)],
        ["NT answer / L3", _fmt(nt_geometry_l3_162.pc1_label_r2), _fmt(nt_geometry_l3_162.effective_dimension, 2), int(nt_geometry_l3_162.monotonic_order_violations), _fmt(nt_geometry_l3_163.pc1_label_r2), _fmt(nt_geometry_l3_163.effective_dimension, 2), int(nt_geometry_l3_163.monotonic_order_violations)],
        ["NT answer / L4", _fmt(nt_geometry_l4_162.pc1_label_r2), _fmt(nt_geometry_l4_162.effective_dimension, 2), int(nt_geometry_l4_162.monotonic_order_violations), _fmt(nt_geometry_l4_163.pc1_label_r2), _fmt(nt_geometry_l4_163.effective_dimension, 2), int(nt_geometry_l4_163.monotonic_order_violations)],
        ["Thinking marker / L3", _fmt(marker_geometry_l3_162.pc1_label_r2), _fmt(marker_geometry_l3_162.effective_dimension, 2), int(marker_geometry_l3_162.monotonic_order_violations), _fmt(marker_geometry_l3_163.pc1_label_r2), _fmt(marker_geometry_l3_163.effective_dimension, 2), int(marker_geometry_l3_163.monotonic_order_violations)],
        ["Thinking marker / L4", _fmt(marker_geometry_l4_162.pc1_label_r2), _fmt(marker_geometry_l4_162.effective_dimension, 2), int(marker_geometry_l4_162.monotonic_order_violations), _fmt(marker_geometry_l4_163.pc1_label_r2), _fmt(marker_geometry_l4_163.effective_dimension, 2), int(marker_geometry_l4_163.monotonic_order_violations)],
    ]

    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>v16.2 vs v16.3：Query 顺序如何改变计数学习与内部机制</title>
  <style>
    :root{{--ink:#172235;--muted:#5e6b7d;--line:#d8e1ea;--paper:#fff;--page:#eef3f7;--navy:#153d64;--blue:#3b6fb6;--orange:#df7b3f;--green:#238636;--amber:#b7791f}}
    *{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--page);color:var(--ink);font-family:Inter,"Noto Sans SC","Microsoft YaHei",system-ui,sans-serif;line-height:1.72}}
    header{{background:linear-gradient(135deg,#102a46,#1c527f);color:#fff;padding:54px max(5vw,24px) 46px}}header .wrap,main,footer{{max-width:1180px;margin:auto}}h1{{font-size:clamp(2rem,4vw,3.35rem);line-height:1.16;margin:0 0 16px}}header p{{max-width:920px;margin:8px 0;color:#e5eef7;font-size:1.08rem}}.eyebrow{{letter-spacing:.12em;text-transform:uppercase;font-weight:750;font-size:.78rem;color:#b9d9f1}}
    nav{{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);backdrop-filter:blur(8px);overflow-x:auto;white-space:nowrap;padding:10px max(3vw,16px)}}nav a{{color:#254b70;text-decoration:none;margin-right:20px;font-size:.9rem;font-weight:650}}
    main{{padding:30px 20px 70px}}section{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:clamp(22px,4vw,42px);margin:0 0 24px;box-shadow:0 8px 26px rgba(23,34,53,.045)}}h2{{color:var(--navy);font-size:clamp(1.55rem,2.5vw,2.15rem);line-height:1.3;margin:0 0 18px}}h3{{color:#244d73;line-height:1.38}}h4{{margin:.2rem 0 .5rem}}code{{background:#eef3f8;padding:.12em .32em;border-radius:4px}}a{{color:#1769aa}}.lead{{font-size:1.06rem;color:#344860}}
    .cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:22px 0}}.card{{border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:10px;padding:16px;background:#fbfdff}}.card.orange{{border-top-color:var(--orange)}}.card .value{{display:block;font-size:1.65rem;font-weight:790;color:#123e65}}.card .label{{color:var(--muted);font-size:.9rem}}
    .callout,.definition{{border-left:4px solid #3977aa;background:#f3f8fc;padding:14px 17px;margin:18px 0;border-radius:0 8px 8px 0}}.callout.warning{{border-left-color:var(--amber);background:#fff9ed}}.callout.success{{border-left-color:var(--green);background:#f2fbf4}}
    .table-wrap{{overflow-x:auto;margin:18px 0}}table{{width:100%;border-collapse:collapse;font-size:.92rem}}th,td{{border-bottom:1px solid var(--line);padding:10px 11px;text-align:left;vertical-align:top}}th{{background:#edf4fa;color:#244d73}}td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
    .layout-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:20px 0}}.layout-card{{border:1px solid var(--line);border-radius:10px;padding:16px;background:#fbfdff}}.sequence-strip{{display:grid;grid-template-columns:55px 1.2fr 4fr 1.1fr;gap:5px;margin:12px 0}}.sequence-strip.last{{grid-template-columns:55px 4fr 1.2fr 1.1fr}}.token-block{{padding:12px 7px;border-radius:7px;text-align:center;font-size:.82rem;font-weight:700}}.bos{{background:#e9eef5}}.query{{background:#f8dfcf;color:#7a3f1f}}.data{{background:#dceafa;color:#254f79}}.output{{background:#dff3e4;color:#236139}}.arrow-note{{font-size:.88rem;color:var(--muted)}}
    .report-figure{{margin:27px 0 32px;border:1px solid var(--line);border-radius:11px;padding:18px;background:#fff}}.report-figure h3{{margin:0 0 13px}}.report-figure img{{display:block;width:100%;height:auto}}figcaption{{margin-top:14px;color:#44566e;font-size:.93rem;line-height:1.65}}.figure-tag{{font-weight:800;color:#153d64;margin-right:.4em}}
    .section-evidence-summary{{margin:30px 0 0;padding:18px;border:1px solid #cbd5e1;border-radius:10px;background:#f8fafc}}.section-evidence-summary>h3{{margin:0 0 14px;font-size:1.08rem}}.section-evidence-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.section-evidence-column{{padding:14px 16px;border:1px solid #dbe3ec;border-radius:8px;background:#fff}}.section-evidence-column.supported{{border-top:4px solid var(--green);background:#f4fbf6}}.section-evidence-column.missing{{border-top:4px solid var(--amber);background:#fffaf0}}.section-evidence-column ul{{margin:0;padding-left:1.2rem}}.section-evidence-column li{{margin:.42rem 0}}
    .mechanism{{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:10px;margin:20px 0}}.mechanism .node{{border:1px solid var(--line);border-radius:9px;padding:15px;text-align:center;background:#f7fafc}}.mechanism .edge{{font-size:1.6rem;color:#5f7892}}.hypothesis{{border-style:dashed!important;border-color:var(--amber)!important;background:#fff9ed!important}}
    footer{{padding:0 22px 45px;color:#5c6b7c;font-size:.9rem}}@media(max-width:820px){{.cards{{grid-template-columns:1fr 1fr}}.layout-grid,.section-evidence-grid{{grid-template-columns:1fr}}.mechanism{{grid-template-columns:1fr}}.mechanism .edge{{transform:rotate(90deg);text-align:center}}}}@media(max-width:520px){{.cards{{grid-template-columns:1fr}}section{{padding:20px 16px}}}}
  </style>
</head>
<body>
<header><div class="wrap"><div class="eyebrow">Controlled query-order ablation · seed 1234 · RoPE</div><h1>Query 放在 data 前还是后，如何改变计数学习？</h1><p>v16.2（query-first）与 v16.3（data-first / query-last）的行为、learning dynamics、attention route 与 hidden-state representation 对比。</p><p>本报告只把证据支持到的层级写成结论；对尚未直接观测的 data→query 聚合明确标记为机制假说。</p></div></header>
<nav><a href="#summary">摘要</a><a href="#setting">设定</a><a href="#definitions">定义</a><a href="#behavior">结果</a><a href="#learning">学习动态</a><a href="#attention">Attention</a><a href="#representation">Representation</a><a href="#trace">Trace</a><a href="#synthesis">机制解释</a><a href="#limits">局限</a><a href="#repro">复现</a></nav>
<main>
<section id="summary"><h2>1. 执行摘要</h2><p class="lead">在同一 seed、相同初始权重、相同数据和相同输出绝对位置下，把 query 从 data 前移动到 data 后，主要改善了 <strong>Nonthinking</strong>：它更早学会、最终错误更少，并从“答案位置直接广泛扫描 data”转向“答案位置读取紧邻 query”的路由。Thinking 的最终答案已接近天花板，但 trace 生成明显更可靠。</p>
  <div class="cards"><div class="card"><span class="value">{_pp(nt_ar.paired_gain)}</span><span class="label">Nonthinking final AR gain</span></div><div class="card"><span class="value">{_pp(nt_tf.paired_gain)}</span><span class="label">Nonthinking final TF gain</span></div><div class="card orange"><span class="value">{_pp(trace_exact_163.value-trace_exact_162.value)}</span><span class="label">Thinking trace-exact gain</span></div><div class="card orange"><span class="value">{_fmt(nt_query_163.heldout_value)}</span><span class="label">v16.3 NT answer→query mass</span></div></div>
  <div class="callout success"><strong>最重要的机制变化。</strong>v16.2 Nonthinking 的 fixed L{int(nt_cov_162['layer'])}H{int(nt_cov_162['head'])} 对目标 occurrence 的 top-N recall 为 {_fmt(nt_cov_162.heldout_value)}；v16.3 对应最佳 fixed head 只有 {_fmt(nt_cov_163.heldout_value)}。反过来，v16.3 的 L{int(nt_query_163['layer'])}H{int(nt_query_163['head'])} 在答案位置把 {_pct(nt_query_163.heldout_value)} attention 放到五个 query tokens，而 v16.2 最强 answer→query head 为 {_pct(nt_query_162.heldout_value)}。行为提升与 direct-data head 消失同时发生，说明 query-last 不是简单强化旧路线，而是重组了路线。</div>
  {_evidence(["Query-last 对 Nonthinking 有大而一致的配对提升；Thinking 的答案提升很小，但 trace exact 大幅改善。","输出起点始终是 position 262，因此差异不能归因于答案绝对位置变化。"],["只有一个训练 seed，尚无跨初始化效应分布。","尚未直接 probe query-token state 或从 query token 出发的 attention；data→query→answer 目前是受多项间接证据支持的假说。"])}
</section>

<section id="setting"><h2>2. 实验设定与真正改变的变量</h2><div class="layout-grid"><div class="layout-card"><h3>v16.2 · query-first</h3><div class="sequence-strip"><div class="token-block bos">BOS<br>0</div><div class="token-block query">Query<br>1–5</div><div class="token-block data">Data<br>6–261</div><div class="token-block output">Output<br>262→</div></div><p class="arrow-note">因果方向：query 在 data 之前，所以每个 data token 都可以读取目标字符集合；但最终输出离 query 257–261 个位置。</p></div><div class="layout-card"><h3>v16.3 · query-last</h3><div class="sequence-strip last"><div class="token-block bos">BOS<br>0</div><div class="token-block data">Data<br>1–256</div><div class="token-block query">Query<br>257–261</div><div class="token-block output">Output<br>262→</div></div><p class="arrow-note">因果方向：data token 看不到未来 query；五个 query tokens 可以回看完整 data，并紧邻输出。</p></div></div>
  {_table(["项目","v16.2","v16.3","可比性解释"],setting_rows)}
  <div class="callout warning"><strong>一次移动同时改变三件事。</strong>它反转了 query/data 的因果可见性，把 query-to-output 距离从 257–261 缩短到 1–5，并把全部 data-to-output 距离增加 5。现有两组实验能识别“整个 layout 变更”的效应，却不能把这三部分彼此分离。</div>
  {_evidence(["corpus split 与 needle pool 的原始 SHA256 相同；training sampling 与 model specification 在统一换行后内容相同，原始 hash 差异只来自 CRLF/LF。","两种模式的 step-0 checkpoint 均有 51 个 state entries，跨版本最大绝对差为 0，torch RNG state 也一致。"],["需要 padding/copy-query 等额外布局，才能拆分因果可见性、query recency 与 data 距离。","GPU 训练的单次轨迹仍不能给出跨运行方差。"])}
</section>

<section id="definitions"><h2>3. 指标、样本配对与新定义</h2>
  <div class="definition"><strong>TF final-count accuracy。</strong>给定 gold prefix，在 <code>&lt;Ans&gt;</code> 后预测 count token 是否正确。最终测试每个 mode 有 500 个样本，即每个真实 count 50 个。</div>
  <div class="definition"><strong>AR final-count accuracy。</strong>从任务输出起点自行生成完整输出，再检查解析出的最终 count。最终测试每个 mode 有 100 个样本，即每个 count 10 个。TF 与 AR 的 16.2/16.3 行按 <code>mode + row/example id + set_id + count + corpus_start</code> 一一配对。</div>
  <div class="definition"><strong>Paired gain 与区间。</strong>对样本 <i>i</i> 定义 d<sub>i</sub>=correct<sub>16.3,i</sub>−correct<sub>16.2,i</sub>，报告 Δ=mean(d<sub>i</sub>)。95% 区间通过固定 seed 的 20,000 次样本对 bootstrap 得到；“修复/退化”分别统计 0→1 与 1→0。McNemar exact p 只检验两个方向的 discordant 数是否对称，不替代多 seed 复现。</div>
  <div class="definition"><strong>Accuracy AULC。</strong><code>AULC = ∫ accuracy(step) dstep / 10000</code>，使用 500-step checkpoint 梯形积分；越大表示整个训练过程中更早、更持续地正确。<strong>Persistent threshold</strong> 是最早一个 checkpoint，使该点及之后所有被测 checkpoint 都不再低于阈值。</div>
  <div class="definition"><strong>Attention route 指标。</strong><em>query mass</em> 是某 head 在指定 query position 的 softmax attention 对五个 query tokens 求和；<em>top-N target recall</em> 是从 256 个 data positions 取 attention 最高的 N 个位置（N 等于真实目标 occurrence 数），其中真实目标位置所占比例；<em>ordered retrieval margin</em>=correct-top1−1/N；<em>trace readout mass</em> 是最终答案 query 对全部 trace index/marker positions 的 attention 总和。</div>
  <div class="definition"><strong>Representation 指标。</strong>Ridge probe 在独立 train states 上拟合 count，并在 held-out states 上计算 <code>R²=1−Σ(y−ŷ)²/Σ(y−ȳ)²</code>；nearest-centroid accuracy 把 state 分到最近的十个训练 count centroids。图中的 best-layer 指同一 checkpoint 五个 residual depths 的最大值，只用于描述“信息最早在哪一层可读”，不等于该层被模型因果使用。</div>
  <div class="definition"><strong>Head map 与 centroid geometry。</strong>Head map 的每个格子是在 step 10,000、独立 heldout-reporting split 上，固定一个 layer/head 后对样本取均值；不再进行“每个样本挑最强 head”。对 hidden state，先按真实 count <i>n</i>=1…10 求十个 class centroids μ<sub>n</sub>，再对这十个均值做 PCA（mean-first PCA）。若全部 centroid PCs 的方差占比为 r<sub>j</sub>，定义有效维度 <code>d_eff=1/Σ r_j²</code>；越接近 1 表示 centroid 变化集中在一条主轴，越大表示分散在更多方向。<em>PC1 label R²</em> 是用第一主成分坐标线性拟合 count label 的 R²；<em>PC1–PC6 variance</em> 是前六主成分解释的 centroid 方差总和。三个量只描述几何，不直接衡量模型是否因果使用该 state。</div>
  <div class="definition"><strong>3D centroid-PCA 图。</strong>每个 panel 都直接读取最终 checkpoint 保存的 held-out hidden states，按上式计算十个 count centroids，并在该 panel 自己拟合的 PC1/PC2/PC3 上展示；点旁数字是 count，连线按 1→10。因为不同 panel 的 PCA basis 与坐标尺度独立，图只能比较顺序、分离和弯曲结构，不能比较绝对朝向或坐标值。</div>
  {_evidence(["最终 TF/AR 是严格的同样本配对比较；训练动态使用两边相同规模的固定 diagnostic suite。","指标将行为、路由和可解码表征分开，避免用 attention mass 或 probe 单独代替机制。"],["bootstrap 只反映有限测试样本的不确定性，不包含训练 seed 方差。","best-layer 与 final-head selection 存在选择步骤；attention head 使用独立 head-selection / heldout-reporting split 降低双重使用。"])}
</section>

<section id="behavior"><h2>4. 最终行为结果：主要收益集中在 Nonthinking</h2>
  {_table(["模式 / 协议","N","v16.2","v16.3","配对增益","95% paired bootstrap CI","修复 / 退化","McNemar p"],final_effect_rows,numeric={1,2,3,4,5,6,7})}
  {_figure(behavior_figure,"图 1。","行为准确率与训练进程","上排横轴是 training step（0–10,000），纵轴是 checkpoint diagnostic suite 的 AR final-count accuracy；黑色虚线为 step 1,500 loss-scope 切换。下排横轴是真实 count n=1…10，纵轴是最终独立 AR suite 中每个 count 的准确率，每个点有 10 个样本。左列 Nonthinking，右列 Thinking；蓝色 query-first，橙色 query-last。注意上排与最终表使用不同固定样本集，因此 v16.2 Nonthinking 在图末为 71%，而最终独立 AR 为 84%，二者不是同一估计量。","v16.2 and v16.3 autoregressive accuracy learning curves and final count-stratified accuracy")}
  <p>Nonthinking 的 TF gain 为 {_pp(nt_tf.paired_gain)}：83 个原错误被修复、4 个样本退化；AR gain 为 {_pp(nt_ar.paired_gain)}：12 个错误全部被修复且无退化。Thinking TF 已经同为 100%；AR 只有 1 个样本由错变对，因此 +1 pp 不足以单凭单 seed 声称稳定改善。</p>
  {_evidence(["Query-last 对 Nonthinking 的最终提升同时出现在 TF 和 AR，并且 AR 的 12 个 discordant 样本方向完全一致。","Thinking final answer 已处于 98–100% 天花板，layout 对答案正确率的可见增益很小。"],["每个 count 的 AR 只有 10 个样本，不宜把单个 count 的 10 pp 波动解释为结构特异效应。","需要更大独立 AR test set 与多 seed，尤其用于 Thinking 的 1 pp 差异。"])}
</section>

<section id="learning"><h2>5. Learning dynamics 与输出 loss</h2>
  {_table(["版本 / 模式","diagnostic final","accuracy AULC","persistent 50%","persistent 80%","persistent 90%"],learning_rows,numeric={1,2,3,4,5})}
  {_figure(effect_figure,"图 2。","输出损失、配对效应与 trace 质量","上排横轴是 training step，纵轴是 held-out task suite 中 final-count token 的 example-mean cross-entropy，并使用对数刻度；竖线为 step 1,500。左下横轴区分 Nonthinking/Thinking 与 TF/AR，纵轴是 v16.3−v16.2 的配对 accuracy，误差棒为 20,000 次 paired bootstrap 95% 区间。右下横轴是 Thinking trace 指标，纵轴是最终 100 条 AR 轨迹的均值。这里没有绘制 all-token task loss，因为 step 1,500 后 data/query token 不再属于训练目标，其 CE 上升不能解释为任务退化。","Held-out output loss curves, paired accuracy effects, and final thinking trace quality")}
  <p>在 checkpoint diagnostic suite 中，Nonthinking 的 query-last AULC 从 {_fmt(nt_l162.accuracy_aulc)} 提高到 {_fmt(nt_l163.accuracy_aulc)}；它的 80% persistent crossing 从 {nt_l162.persistent_80_step if pd.notna(nt_l162.persistent_80_step) else '未达到'} 提前到 {nt_l163.persistent_80_step if pd.notna(nt_l163.persistent_80_step) else '未达到'}。Thinking 也更早到达高准确率，但最终答案受天花板限制。</p>
  {_evidence(["Query-last 不只是最终更准：Nonthinking 的整个训练轨迹左移，AULC 更高、持久阈值更早。","final-count CE 的下降与准确率提升一致；step 1,500 后 all-token CE 的上升主要是 loss scope 改变造成。"],["500-step checkpoint 间隔只能把机制 onset 定位到一个区间。","AULC 仍受这一个 diagnostic suite 与单次训练轨迹影响。"])}
</section>

<section id="attention"><h2>6. Attention route：Nonthinking 从 direct-data 转向 query bottleneck</h2>
  {_table(["路由角色","v16.2 head","v16.2 heldout","v16.3 head","v16.3 heldout","指标"],attention_rows,numeric={2,4})}
  {_figure(attention_figure,"图 3。","固定 heads 的 attention route learning dynamics","四个 panel 的横轴均为 training step，竖线为 loss-scope 切换。左上纵轴是答案 query 的 top-N target-occurrence recall；右上是答案 query 对五个 query tokens 的 attention mass；左下是 Thinking trace index 对正确第 k 个 occurrence 的 top1-minus-chance；右下是 Thinking 最终答案对完整 trace span 的 attention mass。每条曲线先在 step-10,000 head-selection split 选定一个 head，再在独立 heldout-reporting split 的全部 checkpoints 跟踪同一 head，避免逐 checkpoint 挑最大值。","Fixed-head attention-route dynamics for direct data coverage, query routing, ordered retrieval, and trace readout")}
  {_figure(attention_heads_figure,"图 4。","最终 checkpoint 的逐层逐 head attention role map","每一行对应一个版本；每一列对应一种 role。每个 panel 的横轴是 head H0–H3，纵轴是 layer L1–L4，格内数字和颜色都是该 layer/head 在 step 10,000 heldout-reporting split 上的样本均值。第 1 列是答案位置落在五个 query tokens 上的 attention mass（v16.2 表中原字段为 task-prefix mass，v16.3 为 task-query mass，语义相同）；第 2 列是 Nonthinking top-N target-occurrence recall；第 3 列是 Thinking ordered retrieval 的 correct-top1−chance；第 4 列是 Thinking 最终答案对 trace span 的 attention mass。前三个 mass/recall 色标为 0–1；ordered retrieval 色标为 −0.10–0.70。","Layer-by-head heatmaps comparing four final attention roles in v16.2 and v16.3")}
  <div class="callout success"><strong>Nonthinking 路线重组。</strong>v16.2 的 L{int(nt_cov_162['layer'])}H{int(nt_cov_162['head'])} 最终直接覆盖 {_pct(nt_cov_162.heldout_value)} 目标 occurrence，target enrichment={_fmt(nt_enrich_162.heldout_value)}；v16.3 的最佳 fixed direct-data head 只有 {_pct(nt_cov_163.heldout_value)} coverage、enrichment={_fmt(nt_enrich_163.heldout_value)}。与此同时，v16.3 L{int(nt_query_163['layer'])}H{int(nt_query_163['head'])} 的答案→query mass 达 {_pct(nt_query_163.heldout_value)}。因此更高准确率不是来自更强的 v16.2-style broad head。</div>
  <p>逐 head 热图表明这不是单一“最佳 head”造成的错觉：v16.2 有 {coverage_heads_above_50['v16.2']} 个 heads 的 direct target recall 超过 0.50，而 v16.3 为 {coverage_heads_above_50['v16.3']}；相反，答案→query mass 超过 0.80 的 heads 从 v16.2 的 {query_heads_above_80['v16.2']} 个变为 v16.3 的 {query_heads_above_80['v16.3']} 个。head identity 本身不应跨 run 硬对齐，但 role 在 4×4 head 集合中的分布发生了清楚变化。</p>
  <p>Thinking 的 ordered retrieval margin 从 {_fmt(ordered_162.heldout_value)} 升到 {_fmt(ordered_163.heldout_value)}，而 trace readout mass 从 {_fmt(readout_162.heldout_value)} 降到 {_fmt(readout_163.heldout_value)}。这再次说明 attention mass 不是“贡献大小”的同义词：较小 readout mass 可以与更准确的 trace 和答案共存。</p>
  {_evidence(["direct-data 到 answer→query 的变化同时出现在 final-selected 曲线和全 16-head 热图中，不依赖只挑一个极值 head。","Thinking query-last 的 k-to-k selectivity 更高，尽管最终 trace-readout mass 略低。"],["现有表没有测量五个 query tokens 自己如何 attend data，也没有对 query state 做 probe/patch；两跳聚合尚未闭环。","attention weight 没有分解 V 与 W_O contribution，需要 Q/K/V 与 output patching；head role 也需要多 seed 的 permutation-invariant 对齐。"])}
</section>

<section id="representation"><h2>7. Hidden-state representation：Nonthinking count 更早在 L2 成形</h2>
  {_table(["位置 / 层","v16.2 R²","v16.2 centroid acc.","v16.3 R²","v16.3 centroid acc."],representation_rows,numeric={1,2,3,4})}
  {_figure(representation_figure,"图 5。","Count representation 的出现时间与层级位置","上排只看 Nonthinking final-answer state：横轴是 training step，左纵轴是五个 residual depths 中最大的 held-out ridge R²，右纵轴是最大的 nearest-centroid accuracy。下排横轴是 residual depth（Emb 表示 embedding output，L1–L4 表示各层输出），纵轴是 step-10,000 final-answer ridge R²；左为 Nonthinking，右为 Thinking。best-layer 曲线是描述性上界，不应当解释为该层具有因果必要性。","Learning dynamics and final layer profiles of hidden-state count representations")}
  <p>v16.2 Nonthinking 在 L2 只有 R²={_fmt(nt_l2_162.ridge_r2)}、centroid accuracy={_pct(nt_l2_162.nearest_centroid_accuracy)}，主要到 L4 才达到 R²={_fmt(nt_l4_162.ridge_r2)}。v16.3 在 L2 已达到 R²={_fmt(nt_l2_163.ridge_r2)}、centroid accuracy={_pct(nt_l2_163.nearest_centroid_accuracy)}，并在 L4 保持 R²={_fmt(nt_l4_163.ridge_r2)}。这与 L2 答案→query head 的出现相吻合。</p>
  {_table(["state / layer","v16.2 PC1 label R²","v16.2 d_eff","v16.2 order violations","v16.3 PC1 label R²","v16.3 d_eff","v16.3 order violations"],geometry_rows,numeric={1,2,3,4,5,6})}
  {_figure(representation_geometry_figure,"图 6。","逐层 count-centroid geometry 对比","三行分别是 Nonthinking final-answer、Thinking final-answer 与 Thinking trace-marker states；横轴统一为 residual depth L1–L4。左列纵轴是有效维度 d_eff；中列纵轴是 PC1 label R²；右列纵轴是前六个 centroid PCs 的累计解释方差。蓝线为 v16.2，橙线为 v16.3。d_eff 高低没有单调的好坏含义：它只表示十个 count centroids 的变化集中在少数方向还是分散在更多方向；PC1 label R² 衡量第一几何主轴与 count 顺序的一致程度，也不同于使用全部 256 维 state 的 ridge R²。","Layerwise effective dimension and mean-first PCA geometry of final-answer and trace-marker hidden states")}
  <p>几何量进一步区分了“可读”与“如何组织”。v16.2 的 Nonthinking count 在 L3 最接近一条有序主轴：PC1 label R²={_fmt(nt_geometry_l3_162.pc1_label_r2)}、d_eff={_fmt(nt_geometry_l3_162.effective_dimension,2)}；v16.3 则在 L2 已达到 PC1 label R²={_fmt(nt_geometry_l2_163.pc1_label_r2)}、d_eff={_fmt(nt_geometry_l2_163.effective_dimension,2)}，且 count-order violations 从 v16.2 L2 的 {int(nt_geometry_l2_162.monotonic_order_violations)} 个降为 {int(nt_geometry_l2_163.monotonic_order_violations)} 个。到 L4，v16.3 的全维 ridge R² 仍高，但 d_eff 升至 {_fmt(nt_geometry_l4_163.effective_dimension,2)}、PC1 label R² 降至 {_fmt(nt_geometry_l4_163.pc1_label_r2)}；这表示 count 信息被重新分布到多维几何，而不是“信息消失”。</p>
  {_figure(hidden_state_manifold_figure,"图 7。","真实 checkpoint hidden states 的 3D mean-first PCA","两行分别是 v16.2 与 v16.3；三列依次为 Nonthinking final-answer L2、Nonthinking final-answer L4、Thinking trace-marker L3。每个点是 held-out states 中一个真实 count（1–10）的 256 维 class centroid，点旁数字即 count，灰线按 1→10 连接。三个坐标轴是对该 panel 十个 centroids 独立拟合的 PC1、PC2、PC3，括号给出各自解释的 centroid 方差比例。因每个 panel 独立定向和缩放，只比较 count 顺序、相邻分离、弯曲与是否形成近一维轨迹；不要比较跨 panel 的绝对坐标或旋转方向。Thinking 选择 trace-marker 而不是 final-answer state，是为了减少 count-dependent answer position 的直接混淆。","Three-dimensional mean-first PCA of real held-out count centroids from final checkpoints")}
  <div class="callout warning"><strong>Thinking 的初始化基线。</strong>Thinking final-answer position 随 trace 长度、进而随 count 改变；随机初始化模型的 best-layer R² 已接近 1。因此本报告不把 Thinking answer-state 的高 R² 当作训练后算法证据，而主要使用 AR trace、ordered retrieval 和跨版本差异。要消除位置泄漏，需要固定答案位置或做 position-matched probe。</div>
  {_evidence(["Query-last 把 Nonthinking 的高可解码性与近一维有序 centroid geometry 一起提前到 L2；这一点在 probe、几何指标和真实 state 的 3D centroids 中相互一致。","v16.2 与 v16.3 都不是所有层都保持一条直线：后层会重组几何，因此 full-state ridge R² 与 PC1 geometry 必须分开解读。"],["线性可解码和 centroid geometry 都不等于模型因果使用；v16.3 尚未完成 centroid steering、state transplant 等与 v16.2 同规格的验证。","各 3D panel 的 PCA basis 独立，不能做逐坐标对齐；Thinking answer-state 仍受 count-dependent position 混淆，需要固定位置重测。"])}
</section>

<section id="trace"><h2>8. Thinking trace：答案近饱和，但轨迹更稳定</h2>
  <p>最终独立 AR suite 中，trace exact 从 {_pct(trace_exact_162.value)} 升到 {_pct(trace_exact_163.value)}，marker recall 从 {_pct(trace_recall_162.value)} 升到 {_pct(trace_recall_163.value)}；最终答案则只从 {_pct(th_ar.v16_2_accuracy)} 升到 {_pct(th_ar.v16_3_accuracy)}。这说明 query-last 对 Thinking 的主要可见收益是减少中间 trace 错误，而不是突破已经饱和的 final count。</p>
  <p>删除最后一个 index/marker pair 的 counterfactual 在两边都把 L4 ridge count prediction 约降低 1：v16.2 为 {_fmt(cf_l4_162.delta_ridge_count_prediction)}，v16.3 为 {_fmt(cf_l4_163.delta_ridge_count_prediction)}；gold-vs-(n−1) logit margin 分别变化 {_fmt(cf_l4_162.delta_gold_logit_margin,2)} 与 {_fmt(cf_l4_163.delta_gold_logit_margin,2)}。因此两种 layout 都继续使用 trace span/边界来形成最终 count readout。</p>
  <div class="callout warning"><strong>counterfactual 的解释边界。</strong>remove-final-pair 同时缩短 trace、移动后续相对位置并删除 token identity；它能证明输出依赖这段结构，但不能单独证明“最后 marker state 携带 scalar count”。</div>
  {_evidence(["Query-last 显著提高 Thinking trace exact 与 marker recall，并提高 ordered k-to-k selectivity。","两种 layout 对 trace 缩短都表现出约 −1 的内部 count shift，说明核心 span-sensitive readout 仍保留。"],["Thinking final answer 有天花板效应，需要更难 count range 或更长序列才能比较最终能力。","v16.3 还没有完成与 v16.2 相同的 head ablation、clean-to-corrupt patch 和 residual steering 全套 causal battery。"])}
</section>

<section id="synthesis"><h2>9. 机制综合：为什么 query-last 尤其帮助 Nonthinking？</h2>
  <div class="mechanism"><div class="node"><strong>v16.2 data tokens</strong><br>已知 query，可在线写入 target-specific features</div><div class="edge">→</div><div class="node"><strong>L4 broad head</strong><br>答案位置直接扫描目标 occurrence</div><div class="edge">→</div><div class="node"><strong>L4 count state</strong><br>较晚出现，AR 较慢</div></div>
  <div class="mechanism"><div class="node"><strong>v16.3 data tokens</strong><br>不知道未来 query</div><div class="edge">→</div><div class="node hypothesis"><strong>Query-token aggregator（假说）</strong><br>query 回看 data 并压缩所需统计量</div><div class="edge">→</div><div class="node"><strong>L{int(nt_query_163['layer'])} answer→query route</strong><br>{_pct(nt_query_163.heldout_value)} query mass，早期 count state</div></div>
  <p>数据最支持的解释是“把一个困难的长距离直接 readout，改造成靠近输出的 bottleneck”。v16.3 的 query tokens 位于 data 之后，理论上可以针对自身字符回看完整窗口；答案随后只需从最近的五个 query tokens 读取汇总。这个解释同时预测：direct broad head 变弱、query mass 变强、count representation 提前、Nonthinking 学得更快——四项都被观察到。</p>
  <p>但中间的 <em>Query-token aggregator</em> 仍是虚线：目前没有直接从 query positions 提取 hidden states、attention maps 或做 patch。其他可能解释包括单纯的 RoPE recency 优势、五个 query token 作为短期 key/value cache，或 data 整体平移五位引起的优化变化。</p>
  {_evidence(["四个相互独立的观测——行为、学习速度、fixed-head route、L2 state——共同支持 query bottleneck 解释。","v16.3 不是把 v16.2 direct broad aggregation 做得更强，而是以不同路线达到更好结果。"],["没有 query-site probe/patch，因此不能把 data→query 这一步写成已证实机制。","当前二点对比无法区分 causal visibility reversal、query recency 和 RoPE distance 三个子因素。"])}
</section>

<section id="limits"><h2>10. 证据边界与下一组最有信息量的实验</h2>
  <ol><li><strong>至少 5 个 seeds。</strong>重复最终 TF/AR、AULC、final-selected head roles 与 L2/L4 state onset；head 比较应使用 permutation-invariant role/subspace 指标。</li><li><strong>直接测 query bottleneck。</strong>在 v16.3 的 <code>&lt;CountChar&gt;</code> 与三个字符 token 位置收集 attention/state，测试是否能分别解码三字符 occurrence count 与总 count；再做 query-state transplant 和 data→query head patch。</li><li><strong>拆分三个 layout 因素。</strong>增加 beginning+end 双 query、query-last 但用 padding 固定 RoPE 距离、query-first 加末端 query copy、data 平移但 query 不变等控制。</li><li><strong>补齐 v16.3 causal battery。</strong>复现 v16.2 的 query-local head ablation、Q/K/V patch、residual centroid steering、early-stop 与 head↔state 双向干预。</li><li><strong>提高 Thinking 难度。</strong>扩大 count range、窗口长度或 noise，使 final answer 不再 98–100% 饱和，才能比较最终能力而不只比较 trace 稳定性。</li></ol>
  {_evidence(["现有结果足以把“整体 query-order layout”识别为这个 matched seed 中的有效干预。","报告已把直接观测、统计配对、机制推断和未验证中介分开。"],["尚不能给出跨训练随机性的平均处理效应。","尚不能把 layout 总效应归因到某一个相对距离、位置或可见性因素。"])}
</section>

<section id="repro"><h2>11. 产物、分析协议与复现</h2><p>报告由 <code>scripts/build_v16_2_v16_3_comparison_report.py</code> 从两个 run 的原始 CSV 与 step-0 checkpoints 重建。所有图嵌入 HTML，同时保存独立 PNG；全部派生表位于 <code>analysis/v16_2_vs_v16_3/tables/</code>。</p>
  <ul><li><a href="analysis/v16_2_vs_v16_3/tables/setting_and_identity_audit.csv">setting_and_identity_audit.csv</a>：数据、模型规格与初始化审计。</li><li><a href="analysis/v16_2_vs_v16_3/tables/paired_final_behavior.csv">paired_final_behavior.csv</a>：配对效应、bootstrap CI 与 McNemar 统计。</li><li><a href="analysis/v16_2_vs_v16_3/tables/checkpoint_learning_summary.csv">checkpoint_learning_summary.csv</a>：AULC 与 persistent crossings。</li><li><a href="analysis/v16_2_vs_v16_3/tables/attention_route_summary.csv">attention_route_summary.csv</a>：final-selected fixed-head 路由。</li><li><a href="analysis/v16_2_vs_v16_3/tables/attention_head_maps.csv">attention_head_maps.csv</a>：最终 4×4 heads 的四种 attention role。</li><li><a href="analysis/v16_2_vs_v16_3/tables/representation_summary.csv">representation_summary.csv</a>：逐层 final answer state probe。</li><li><a href="analysis/v16_2_vs_v16_3/tables/representation_geometry_final.csv">representation_geometry_final.csv</a>：最终逐层 centroid geometry。</li><li><a href="analysis/v16_2_vs_v16_3/tables/hidden_state_centroid_pca.csv">hidden_state_centroid_pca.csv</a>：3D 图的真实 state centroid 坐标与方差。</li><li><a href="analysis/v16_2_vs_v16_3/manifest.json">manifest.json</a>：输入与输出 SHA256。</li></ul>
  <div class="callout"><strong>分析 provenance 差异。</strong>v16.2 checkpoint-dynamics manifest 设置 <code>reuse_unaffected_legacy_artifacts=true</code>，v16.3 为 false。两边的 checkpoint inventory、样本预算与共同表字段一致；本报告只比较共同定义，并对最终行为直接使用严格配对样本。若要做发表级复现，建议用当前统一代码对两边都 <code>force</code> 重算一次。</div>
  {_evidence(["输入身份、step-0 tensor equality、样本配对和派生表均有机器可读审计。","HTML 自包含图像，不依赖本机绝对图片路径。"],["尚未在第二台机器或冻结容器中执行完整重建。","v16.2/v16.3 checkpoint dynamics 的缓存开关不同，发表前应统一强制重算。"])}
</section>
</main>
<footer>v16.2 vs v16.3 controlled query-order report · RoPE · seed {cfg['seed']} · {cfg['train_steps']:,} training steps · generated from local audited artifacts.</footer>
</body></html>"""

    output = v16_3 / REPORT_NAME
    _atomic_text(output, report)

    manifest = {
        "report": REPORT_NAME,
        "comparison": "v16.2 query-first vs v16.3 data-then-query",
        "v16_2_root": v16_2.name,
        "v16_3_root": v16_3.name,
        "config_sha256": {
            "v16_2": _sha256(v16_2 / "config.json"),
            "v16_3": _sha256(v16_3 / "config.json"),
        },
        **audit_payload,
        "figures": {path.name: _sha256(path) for path in sorted(figure_dir.glob("*.png"))},
        "tables": {path.name: _sha256(path) for path in sorted(table_dir.glob("*.csv"))},
        "report_sha256": _sha256(output),
    }
    _atomic_text(output_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    _validate_report(output, output_dir, manifest)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16-2", type=Path, default=DEFAULT_V16_2)
    parser.add_argument("--v16-3", type=Path, default=DEFAULT_V16_3)
    args = parser.parse_args()
    print(build(args.v16_2, args.v16_3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
