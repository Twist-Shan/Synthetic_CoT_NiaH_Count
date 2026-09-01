#!/usr/bin/env python
"""Compare aligned nearest-class-centroid geometry for v22 and v23.

The protocol matches ``analyze_v20_aligned_geometry.py`` but loads only the
Thinking model, because the canonical v22 run intentionally reuses the v20
Non-thinking baseline and therefore has no local v22 Non-thinking checkpoint.
Layer selection is performed on discovery examples only; confirmation NCC is
reported after freezing the discovery-fitted PCA transform and centroids.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = Path(
    "\\\\?\\H:\\\u6211\u7684\u4e91\u7aef\u786c\u76d8\\Colab_Notebooks\\CoT_Counting\\"
    "Synthetic_CoT_NiaH_Count\\colab_results"
)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from synthetic_counting_v20.aligned_geometry import (  # noqa: E402
    capture_mode_geometry,
    combine_splits,
    evaluate_geometry_dataset,
)
from synthetic_counting_v20.config import config_from_dict  # noqa: E402
from synthetic_counting_v20.data import (  # noqa: E402
    V20Vocab,
    load_corpus_split,
    load_corpus_text,
    load_suite_manifests,
)
from synthetic_counting_v20.needle_pool import load_needle_pool  # noqa: E402
from synthetic_counting_v20.training import load_v20_checkpoint_model  # noqa: E402


for _optional in ("pyarrow", "numexpr", "bottleneck"):
    if sys.modules.get(_optional) is None:
        sys.modules.pop(_optional, None)


def _unique_run(results_root: Path, version: str) -> Path:
    matches = sorted(path for path in results_root.glob(f"{version}_main_*") if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {version} main run under {results_root}; "
            f"found {[path.name for path in matches]}"
        )
    return matches[0]


def _load_examples(run_dir: Path, device: str):
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    cfg = replace(cfg, device=device)
    if cfg.position_encodings != ("rope",):
        raise ValueError(f"{run_dir.name}: expected RoPE-only configuration")
    if cfg.count_tokenization != "atomic" or cfg.trace_format != "separator":
        raise ValueError(f"{run_dir.name}: expected atomic separator-trace configuration")

    vocab = V20Vocab.load(run_dir / "vocab.json")
    corpus = load_corpus_text()
    split = load_corpus_split(run_dir / "data" / "corpus_split.json", cfg, corpus)
    pool = load_needle_pool(
        run_dir / "data" / "needle_pool.json",
        cfg,
        split_fingerprint=split.split_fingerprint,
        vocab_fingerprint=vocab.fingerprint,
    )
    curves, _ = load_suite_manifests(
        run_dir / "data" / "loss_suite_manifests.json",
        split_fingerprint=split.split_fingerprint,
        pool_fingerprint=pool.pool_fingerprint,
    )
    reporting = []
    all_heldout = list(curves["heldout"]["task"])
    for count in range(cfg.count_min, cfg.count_max_threshold + 1):
        values = [example for example in all_heldout if int(example.count or 0) == count]
        reporting.extend(values[cfg.phase_head_selection_examples_per_count :])
    return cfg, vocab, list(curves["train"]["task"]), reporting


def analyze_run(
    run_dir: Path,
    *,
    version: str,
    device: str,
    discovery_per_label: int,
    confirmation_per_label: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    cfg, vocab, train_examples, reporting_examples = _load_examples(run_dir, device)
    _, loaded_vocab, _, _, model = load_v20_checkpoint_model(
        run_dir,
        "rope",
        "thinking",
        step=cfg.train_steps,
        device=device,
    )
    if loaded_vocab.fingerprint != vocab.fingerprint:
        raise ValueError(f"{version}: checkpoint vocabulary mismatch")
    model.eval()

    print(f"[{version}] capturing discovery states", flush=True)
    discovery = capture_mode_geometry(
        model,
        vocab,
        train_examples,
        mode="thinking",
        split="discovery",
        per_label=discovery_per_label,
        device=device,
        batch_size=batch_size,
    )
    print(f"[{version}] capturing confirmation states", flush=True)
    confirmation = capture_mode_geometry(
        model,
        vocab,
        reporting_examples,
        mode="thinking",
        split="confirmation",
        per_label=confirmation_per_label,
        device=device,
        batch_size=batch_size,
    )

    metric_frames = []
    selection_frames = []
    selected_layers: dict[str, int] = {}
    for endpoint in discovery:
        combined = combine_splits(discovery[endpoint], confirmation[endpoint])
        metrics, selections, selected_layer = evaluate_geometry_dataset(
            combined,
            endpoint=endpoint,
        )
        metrics.insert(0, "version", version)
        metrics.insert(1, "run_name", run_dir.name)
        selections.insert(0, "version", version)
        selections.insert(1, "run_name", run_dir.name)
        metric_frames.append(metrics)
        selection_frames.append(selections)
        selected_layers[endpoint] = int(selected_layer)
        chosen = metrics.loc[metrics["layer"].astype(int).eq(int(selected_layer))].iloc[0]
        print(
            f"[{version}] {endpoint}: layer={selected_layer}, "
            f"discovery NCC={chosen['discovery_oof_ncc_balanced_accuracy']:.4f}, "
            f"confirmation NCC={chosen['confirmation_ncc_balanced_accuracy']:.4f}",
            flush=True,
        )

    del model, discovery, confirmation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(selection_frames, ignore_index=True),
        selected_layers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "ncc_v22_v23")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--discovery-per-label", type=int, default=10)
    parser.add_argument("--confirmation-per-label", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    all_selections = []
    selected = {}
    for version in ("v22", "v23"):
        run_dir = _unique_run(results_root, version)
        metrics, selections, selected_layers = analyze_run(
            run_dir,
            version=version,
            device=args.device,
            discovery_per_label=args.discovery_per_label,
            confirmation_per_label=args.confirmation_per_label,
            batch_size=args.batch_size,
        )
        all_metrics.append(metrics)
        all_selections.append(selections)
        selected[version] = selected_layers

    metrics = pd.concat(all_metrics, ignore_index=True)
    selections = pd.concat(all_selections, ignore_index=True)
    metrics.to_csv(args.output / "geometry_site_layer_metrics.csv", index=False)
    selections.to_csv(args.output / "geometry_discovery_selected_metrics.csv", index=False)
    (args.output / "selected_layers.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
