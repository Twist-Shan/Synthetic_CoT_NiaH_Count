#!/usr/bin/env python
"""Run large-model-aligned final geometry and milestone dynamics for v20."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from synthetic_counting_v20.aligned_geometry import (  # noqa: E402
    capture_mode_geometry,
    combine_splits,
    confirmation_pca_coordinates,
    evaluate_geometry_dataset,
    evaluate_geometry_layer,
    save_geometry_dataset,
    write_protocol_manifest,
)
from synthetic_counting_v20.training import load_v20_checkpoint_model  # noqa: E402
from synthetic_counting_v20.v10_port_analysis import load_context  # noqa: E402

# ``v10_port_analysis`` marks unavailable optional pandas accelerators with a
# ``None`` entry in ``sys.modules``.  Newer scikit-learn interprets the mere
# presence of that key as an imported pyarrow module, so remove only the
# sentinel entries before fitting estimators.
for _optional in ("pyarrow", "numexpr", "bottleneck"):
    if sys.modules.get(_optional) is None:
        sys.modules.pop(_optional, None)


DEFAULT_RUN = ROOT / "colab_results" / "v20_main_RoPE_count1-30_seed1234"
DEFAULT_MILESTONES = (0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)


def _parse_milestones(value: str) -> tuple[int, ...]:
    steps = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not steps:
        raise argparse.ArgumentTypeError("at least one milestone is required")
    return steps


def _capture_pair(model, ctx, mode: str, discovery_per_label: int, confirmation_per_label: int):
    discovery = capture_mode_geometry(
        model,
        ctx.vocab,
        ctx.train_examples,
        mode=mode,
        split="discovery",
        per_label=discovery_per_label,
        device=ctx.device,
        batch_size=min(32, ctx.cfg.analysis_batch_size),
    )
    confirmation = capture_mode_geometry(
        model,
        ctx.vocab,
        ctx.heldout_examples,
        mode=mode,
        split="confirmation",
        per_label=confirmation_per_label,
        device=ctx.device,
        batch_size=min(32, ctx.cfg.analysis_batch_size),
    )
    return {
        endpoint: combine_splits(discovery[endpoint], confirmation[endpoint])
        for endpoint in discovery
    }


def run_final_geometry(ctx, output: Path) -> tuple[dict[str, int], pd.DataFrame]:
    capture_dir = output / "captures"
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    all_selections = []
    all_coordinates = []
    selected_layers: dict[str, int] = {}
    for mode in ("nonthinking", "thinking"):
        print(f"[aligned geometry] final capture {mode}", flush=True)
        datasets = _capture_pair(
            ctx.models[mode],
            ctx,
            mode,
            discovery_per_label=10,
            confirmation_per_label=8,
        )
        for endpoint, dataset in datasets.items():
            save_geometry_dataset(dataset, capture_dir, endpoint)
            per_layer, selections, selected_layer = evaluate_geometry_dataset(
                dataset,
                endpoint=endpoint,
            )
            selected_layers[endpoint] = selected_layer
            all_metrics.append(per_layer)
            all_selections.append(selections)
            coordinates = confirmation_pca_coordinates(dataset, selected_layer)
            coordinates["endpoint"] = endpoint
            all_coordinates.append(coordinates)
            print(
                f"[aligned geometry] {endpoint}: discovery-selected layer {selected_layer}",
                flush=True,
            )
        del datasets
        gc.collect()

    metrics = pd.concat(all_metrics, ignore_index=True)
    selections = pd.concat(all_selections, ignore_index=True)
    coordinates = pd.concat(all_coordinates, ignore_index=True)
    metrics.to_csv(table_dir / "geometry_site_layer_metrics.csv", index=False)
    selections.to_csv(table_dir / "geometry_discovery_selected_metrics.csv", index=False)
    coordinates.to_csv(table_dir / "geometry_confirmation_pca_coordinates.csv", index=False)
    pd.DataFrame(
        [
            {"endpoint": endpoint, "selected_layer": layer}
            for endpoint, layer in selected_layers.items()
        ]
    ).to_csv(table_dir / "geometry_common_layer_selection.csv", index=False)
    return selected_layers, metrics


def run_training_dynamics(
    ctx,
    output: Path,
    selected_layers: dict[str, int],
    milestones: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for step in milestones:
        for mode in ("nonthinking", "thinking"):
            print(f"[aligned geometry dynamics] step={step} mode={mode}", flush=True)
            _, _, _, _, model = load_v20_checkpoint_model(
                ctx.run_dir,
                "rope",
                mode,
                step=step,
                device=ctx.device,
            )
            datasets = _capture_pair(
                model,
                ctx,
                mode,
                discovery_per_label=5,
                confirmation_per_label=4,
            )
            for endpoint, dataset in datasets.items():
                layer = selected_layers[endpoint]
                metrics = evaluate_geometry_layer(
                    dataset.states_by_layer[layer],
                    dataset.metadata,
                    tuple(range(1, 31)),
                    folds=3,
                )
                rows.append(
                    {
                        "step": int(step),
                        "mode": mode,
                        "endpoint": endpoint,
                        "fixed_final_discovery_selected_layer": int(layer),
                        **metrics,
                    }
                )
            del datasets, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "tables" / "geometry_training_dynamics.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--milestones",
        type=_parse_milestones,
        default=DEFAULT_MILESTONES,
        help="comma-separated checkpoint steps",
    )
    parser.add_argument("--skip-dynamics", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = run_dir / "analysis" / "synthetic_report"
    output.mkdir(parents=True, exist_ok=True)
    ctx = load_context(run_dir, device=args.device)
    selected_layers, metrics = run_final_geometry(ctx, output)
    if not args.skip_dynamics:
        run_training_dynamics(ctx, output, selected_layers, tuple(args.milestones))
    recorded_milestones = tuple(args.milestones)
    dynamics_path = output / "tables" / "geometry_training_dynamics.csv"
    if args.skip_dynamics and dynamics_path.exists():
        recorded_milestones = tuple(
            sorted(pd.read_csv(dynamics_path)["step"].astype(int).unique().tolist())
        )
    write_protocol_manifest(
        output / "geometry_protocol.json",
        {
            "run_dir": str(run_dir),
            "device": args.device,
            "final_discovery_examples_per_label": 10,
            "final_confirmation_examples_per_label": 8,
            "dynamics_discovery_examples_per_label": 5,
            "dynamics_confirmation_examples_per_label": 4,
            "dynamics_milestones": list(recorded_milestones),
            "fixed_layer_dynamics": (
                "Each endpoint tracks backward the physical layer selected at the final "
                "checkpoint on discovery only; this is a descriptive emergence curve, "
                "not a checkpoint-wise model-selection estimate."
            ),
            "final_metric_rows": int(len(metrics)),
        },
    )
    print(f"[aligned geometry] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
