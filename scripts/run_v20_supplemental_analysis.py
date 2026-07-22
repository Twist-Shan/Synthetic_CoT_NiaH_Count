#!/usr/bin/env python3
"""Add high-power final AR, interactive head dynamics, and mediation patches to v20."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import V20Vocab, load_corpus_split, load_corpus_text, load_suite_manifests
from synthetic_counting_v20.extended_analysis import collect_dense_attention_roles
from synthetic_counting_v20.needle_pool import load_needle_pool
from synthetic_counting_v20.training import (
    _write_final_autoregressive_artifacts,
    autoregressive_task_evaluation,
    load_v20_checkpoint_model,
    summarize_learning_tables,
)
from synthetic_counting_v20.v10_port_analysis import (
    _head_rankings,
    _atomic_csv,
    load_context,
    run_localization_transport_patching,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "colab_results/v20_main_RoPE_count1-30_seed1234"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_final_ar(run_dir: Path, device: str, examples_per_count: int) -> None:
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    cfg = replace(cfg, device=device)
    vocab = V20Vocab.load(run_dir / "vocab.json")
    corpus = load_corpus_text()
    split = load_corpus_split(run_dir / "data/corpus_split.json", cfg, corpus)
    pool = load_needle_pool(
        run_dir / "data/needle_pool.json",
        cfg,
        split_fingerprint=split.split_fingerprint,
        vocab_fingerprint=vocab.fingerprint,
    )
    _, test_suites = load_suite_manifests(
        run_dir / "data/loss_suite_manifests.json",
        split_fingerprint=split.split_fingerprint,
        pool_fingerprint=pool.pool_fingerprint,
    )
    selected = []
    for count in range(cfg.count_min, cfg.count_max_threshold + 1):
        bucket = [
            example for example in test_suites["task"] if int(example.count or 0) == count
        ][:examples_per_count]
        if len(bucket) != examples_per_count:
            raise ValueError(
                f"final AR needs {examples_per_count} test examples for count={count}; "
                f"found {len(bucket)}"
            )
        selected.extend(bucket)
    for mode in ("nonthinking", "thinking"):
        detail_path = run_dir / "tables/final_autoregressive_detail.csv"
        if detail_path.exists():
            existing = pd.read_csv(detail_path)
            covered = existing[existing["mode"] == mode].groupby("count").size()
            if (
                len(covered) == cfg.count_max_threshold
                and covered.ge(examples_per_count).all()
            ):
                continue
        _, loaded_vocab, _, _, model = load_v20_checkpoint_model(
            run_dir, "rope", mode, step=cfg.train_steps, device=device
        )
        frame = autoregressive_task_evaluation(
            model,
            cfg,
            loaded_vocab,
            selected,
            position_encoding="rope",
            mode=mode,
            step=cfg.train_steps,
        )
        _write_final_autoregressive_artifacts(
            frame,
            run_dir,
            examples_per_count=examples_per_count,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summarize_learning_tables(run_dir)


def run_mediation(run_dir: Path, device: str) -> None:
    ctx = load_context(run_dir, device)
    rankings, _ = _head_rankings(ctx)
    frame = run_localization_transport_patching(
        ctx,
        options=__import__(
            "synthetic_counting_v20.v10_port_analysis", fromlist=["PortOptions"]
        ).PortOptions(),
        ranking=rankings["thinking_targeted"],
    )
    table_dir = run_dir / "analysis/v10_port/tables"
    path = table_dir / "retrieval_localization_transport_patching.csv"
    _atomic_csv(frame, path)
    manifest_path = run_dir / "analysis/v10_port/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tables"] = [
        item for item in manifest.get("tables", []) if item["name"] != path.name
    ]
    manifest["tables"].append(
        {"name": path.name, "rows": int(len(frame)), "sha256": sha256(path)}
    )
    notes = manifest.setdefault("causal_identification_notes", [])
    note = (
        "pattern-only, target-source value-only, and post-layer residual patches use "
        "one length-matched retrieval corruption to separate localization from identity transport"
    )
    if note not in notes:
        notes.append(note)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--final-ar-examples-per-count", type=int, default=50)
    parser.add_argument(
        "--part",
        action="append",
        choices=("final-ar", "attention", "mediation"),
        default=[],
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    parts = args.part or ["final-ar", "attention", "mediation"]
    if "final-ar" in parts:
        run_final_ar(run_dir, args.device, args.final_ar_examples_per_count)
    if "attention" in parts:
        collect_dense_attention_roles(run_dir, device=args.device)
    if "mediation" in parts:
        run_mediation(run_dir, args.device)
    summary = run_dir / "tables/final_autoregressive_summary.csv"
    if summary.exists():
        print(pd.read_csv(summary).to_string(index=False))
    print(f"SUPPLEMENTAL_RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
