#!/usr/bin/env python3
"""Validate the generated v16.2 v10-port tables, figures, and HTML report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_FIGURES = {
    "hypothesis_causal_map.png",
    "representation_manifolds_2d.png",
    "representation_manifolds_3d.png",
    "representation_geometry_summary.png",
    "representation_learning_dynamics.png",
    "head_ablation_global_local.png",
    "retrieval_head_patching.png",
    "successor_head_patching.png",
    "successor_logit_lens_components.png",
    "successor_mlp_features.png",
    "final_query_head_transport.png",
    "length_preserving_trace_conflicts.png",
    "final_bridge_component_recovery.png",
    "residual_count_transport.png",
    "trace_early_stop_patching.png",
    "head_state_bidirectional.png",
}

EXPECTED_SECTION_EVIDENCE_IDS = (
    "questions",
    "setup",
    "definitions",
    "learning",
    "attention-representation",
    "residual-representation",
    "causal-heads",
    "causal-retrieval-conversion",
    "causal-final-readout",
    "causal-state",
    "causal-bidirectional",
    "data-noise",
    "limits",
    "runtime-repro",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(run_dir: Path) -> dict[str, object]:
    analysis_dir = run_dir / "analysis" / "v10_port"
    manifest_path = analysis_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("position_encoding") != "rope":
        raise AssertionError("manifest is not the intended RoPE run")
    run_id = str(manifest.get("run_id", ""))
    if "rope-nt-rope-t" not in run_id:
        raise AssertionError(f"unexpected run_id: {run_id}")

    table_dir = analysis_dir / "tables"
    total_rows = 0
    for item in manifest["tables"]:
        table_path = table_dir / item["name"]
        if not table_path.is_file():
            raise AssertionError(f"missing table: {table_path}")
        if _sha256(table_path) != item["sha256"]:
            raise AssertionError(f"hash mismatch: {table_path.name}")
        frame = pd.read_csv(table_path)
        if len(frame) != int(item["rows"]):
            raise AssertionError(f"row mismatch: {table_path.name}")
        numeric = frame.select_dtypes(include=[np.number])
        if np.isinf(numeric.to_numpy(dtype=float, copy=False)).any():
            raise AssertionError(f"infinite numeric value: {table_path.name}")
        total_rows += len(frame)

    interactive_path = table_dir / "interactive_hidden_state_pca.csv"
    if not interactive_path.is_file():
        raise AssertionError(f"missing interactive table: {interactive_path}")
    interactive = pd.read_csv(interactive_path)
    expected_sites = {
        ("nonthinking", "final_answer"),
        ("thinking", "final_answer"),
        ("thinking", "trace_index"),
        ("thinking", "trace_marker"),
    }
    actual_sites = set(
        interactive[["mode", "site"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if len(interactive) != 4200 or actual_sites != expected_sites:
        raise AssertionError(
            "interactive table coverage is incomplete: "
            f"rows={len(interactive)}, sites={sorted(actual_sites)}"
        )
    if sorted(interactive["step"].unique()) != list(range(0, 10001, 500)):
        raise AssertionError("interactive table does not contain all 21 checkpoints")
    if sorted(interactive["layer"].unique()) != list(range(5)):
        raise AssertionError("interactive table does not contain embedding plus layers 1-4")
    interactive_numeric = interactive.select_dtypes(include=[np.number])
    if not np.isfinite(interactive_numeric.to_numpy(dtype=float, copy=False)).all():
        raise AssertionError("interactive table contains non-finite numeric values")

    figure_dir = analysis_dir / "figures"
    actual_figures = {path.name for path in figure_dir.glob("*.png")}
    missing_figures = sorted(EXPECTED_FIGURES - actual_figures)
    if missing_figures:
        raise AssertionError(f"missing figures: {missing_figures}")

    report = run_dir / "v16_2_full_causal_report.html"
    report_text = report.read_text(encoding="utf-8")
    if "\ufffd" in report_text:
        raise AssertionError("report contains Unicode replacement characters")
    required_phrases = (
        run_id,
        "2D/3D representation",
        "因果机制",
        "length-preserving",
        "Normalized recovery",
        "v162-hs3d-canvas",
        "Checkpoint step",
        "reading-primer",
        "读图前先定义",
        "证据矩阵",
    )
    missing_phrases = [value for value in required_phrases if value not in report_text]
    if missing_phrases:
        raise AssertionError(f"report is missing required content: {missing_phrases}")

    report_hash = _sha256(report)
    html_files = sorted(path.name for path in run_dir.glob("*.html"))
    if html_files != [report.name]:
        raise AssertionError(f"expected one report HTML, found: {html_files}")

    html_sections = len(re.findall(r"<section\b", report_text))
    html_figures = len(re.findall(r"<figure\b", report_text))
    embedded_images = len(re.findall(r"<img\b", report_text))
    interactive_canvases = len(re.findall(r"<canvas\b", report_text))
    reading_guides = len(re.findall(r'class="figure-reading-guide"', report_text))
    evidence_summary_ids = re.findall(
        r'<aside class="section-evidence-summary" data-section-id="([^"]+)"',
        report_text,
    )
    evidence_conclusion_labels = report_text.count("目前可以得到的结论")
    evidence_gap_labels = report_text.count("欠缺的证据")
    if (
        html_sections < 14
        or html_figures < 37
        or embedded_images < 31
        or interactive_canvases < 1
        or reading_guides != html_figures
    ):
        raise AssertionError(
            "report structure is incomplete: "
            f"sections={html_sections}, figures={html_figures}, images={embedded_images}, "
            f"canvases={interactive_canvases}"
            f", reading_guides={reading_guides}"
        )
    if (
        evidence_summary_ids != list(EXPECTED_SECTION_EVIDENCE_IDS)
        or evidence_conclusion_labels != len(EXPECTED_SECTION_EVIDENCE_IDS)
        or evidence_gap_labels != len(EXPECTED_SECTION_EVIDENCE_IDS)
    ):
        raise AssertionError(
            "section evidence summaries are incomplete or out of order: "
            f"ids={evidence_summary_ids}, conclusions={evidence_conclusion_labels}, "
            f"gaps={evidence_gap_labels}"
        )

    return {
        "run_id": run_id,
        "tables": len(manifest["tables"]),
        "table_rows": total_rows,
        "figures": len(actual_figures),
        "report_bytes": report.stat().st_size,
        "report_sha256": report_hash,
        "html_sections": html_sections,
        "html_figures": html_figures,
        "embedded_images": embedded_images,
        "interactive_rows": len(interactive),
        "interactive_canvases": interactive_canvases,
        "reading_guides": reading_guides,
        "section_evidence_summaries": len(evidence_summary_ids),
        "html_files": html_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.run_dir.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
