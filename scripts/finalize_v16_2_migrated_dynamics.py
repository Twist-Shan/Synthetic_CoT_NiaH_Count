#!/usr/bin/env python3
"""Finalize provenance after the audited revision-5 -> revision-6 migration.

This script is intentionally separate from normal analysis.  It labels reused AR
rows by diagnostic suite, re-hashes every part, stamps the migration-aware option
fingerprint, and rebuilds aggregate tables/plots from only verified parts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from synthetic_counting_v16_2.checkpoint_dynamics import (
    DynamicsOptions,
    _aggregate_parts,
    _artifact_manifest,
    _atomic_json,
    _fixed_head_behavior_link,
    _head_stability,
    _json_fingerprint,
    _state_similarity,
)
from synthetic_counting_v16_2.config import config_from_dict
from synthetic_counting_v16_2.data import V16_2Vocab, load_corpus_split, load_corpus_text
from synthetic_counting_v16_2.needle_pool import load_needle_pool
from synthetic_counting_v16_2.training import atomic_csv


def finalize(run_dir: Path, device: str) -> str:
    run_dir = run_dir.resolve()
    analysis_dir = run_dir / "analysis" / "checkpoint_dynamics"
    table_dir = run_dir / "tables"
    manifest_path = analysis_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    options = DynamicsOptions(**manifest["options"])
    options_payload = asdict(options)
    cfg = replace(
        config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8"))),
        device=device,
    )
    text = load_corpus_text()
    split = load_corpus_split(run_dir / "data" / "corpus_split.json", cfg, text)
    vocab = V16_2Vocab.load(run_dir / "vocab.json")
    pool = load_needle_pool(
        run_dir / "data" / "needle_pool.json",
        cfg,
        split_fingerprint=split.split_fingerprint,
        vocab_fingerprint=vocab.fingerprint,
    )
    fingerprint = _json_fingerprint(
        {
            "version": "v16_2_revision_6_deconfounded_states_ar_reporting",
            "config": cfg.to_dict(),
            "options": {
                key: value for key, value in options_payload.items() if key != "force"
            },
            "split": split.split_fingerprint,
            "pool": pool.pool_fingerprint,
        }
    )

    for part_dir in sorted((analysis_dir / "parts").glob("*")):
        marker_path = part_dir / "complete.json"
        if not marker_path.exists():
            raise FileNotFoundError(f"incomplete migrated part: {part_dir}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        ar_path = part_dir / "autoregressive.csv"
        if ar_path.exists():
            ar = pd.read_csv(ar_path)
            if "diagnostic_split" not in ar:
                ar["diagnostic_split"] = (
                    "heldout_reporting"
                    if (part_dir / "attention_behavior_link.csv").exists()
                    else "legacy_balanced_head_selection"
                )
                atomic_csv(ar, ar_path)
        for name in (
            "state_probe_summary_generated.csv",
            "state_by_count_generated.csv",
            "generated_state_status.csv",
        ):
            path = part_dir / name
            if path.exists():
                frame = pd.read_csv(path)
                if "diagnostic_split" not in frame:
                    frame["diagnostic_split"] = "legacy_balanced_generated_state_suite"
                    atomic_csv(frame, path)
        marker["options_fingerprint"] = fingerprint
        marker["artifacts"] = _artifact_manifest(part_dir)
        _atomic_json(marker, marker_path)

    tables = _aggregate_parts(analysis_dir, table_dir, fingerprint)
    stability = _head_stability(
        tables.get("checkpoint_attention_detail.csv", pd.DataFrame()), table_dir
    )
    _fixed_head_behavior_link(
        tables.get("checkpoint_attention_behavior_link.csv", pd.DataFrame()),
        stability,
        table_dir,
    )
    if options.run_similarity:
        _state_similarity(analysis_dir, table_dir, fingerprint)
    from synthetic_counting_v16_2.plots import plot_v16_2_checkpoint_dynamics

    plot_v16_2_checkpoint_dynamics(run_dir)
    manifest["options_fingerprint"] = fingerprint
    manifest["migration_provenance"] = (
        "revision-5 attention/counterfactual and explicitly labeled legacy AR suites reused; "
        "trace-progress states recomputed with total_count=10"
    )
    _atomic_json(manifest, manifest_path)
    return fingerprint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(finalize(args.run_dir, args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
