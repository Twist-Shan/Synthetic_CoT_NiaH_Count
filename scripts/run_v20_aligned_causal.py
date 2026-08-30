#!/usr/bin/env python
"""Run the two missing large-model-aligned causal controls for synthetic v20."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from synthetic_counting_v20.aligned_causal import (  # noqa: E402
    run_nonthinking_prompt_evidence_restoration,
    run_thinking_trace_scope_restoration,
)
from synthetic_counting_v20.v10_port_analysis import load_context  # noqa: E402


DEFAULT_RUN = ROOT / "colab_results" / "v20_main_RoPE_count1-30_seed1234"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = run_dir / "analysis" / "synthetic_report"
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    ctx = load_context(run_dir, device=args.device)

    print("[aligned causal] non-thinking prompt evidence restoration", flush=True)
    prompt = run_nonthinking_prompt_evidence_restoration(ctx)
    prompt.to_csv(tables / "nonthinking_prompt_evidence_restoration.csv", index=False)

    print("[aligned causal] thinking trace scope restoration", flush=True)
    scope = run_thinking_trace_scope_restoration(ctx)
    scope.to_csv(tables / "thinking_trace_scope_restoration.csv", index=False)

    manifest = {
        "schema_version": "v20_aligned_causal_v1",
        "run_dir": str(run_dir),
        "device": args.device,
        "nonthinking_prompt_evidence": {
            "examples_per_count": 3,
            "counts": "2..30",
            "corruption": "replace the final true target character by a non-target character",
            "outcome": "logit margin gold N minus corrupted-count alternative N-1",
            "controls": [
                "ordinary state patched at the deleted target location",
            ],
            "location_transfer_test": (
                "target embedding patched at a matched ordinary location; this is "
                "a location-invariance test, not a negative control"
            ),
            "rows": int(len(prompt)),
        },
        "thinking_trace_scope": {
            "examples_per_k": 3,
            "k": "2..28",
            "donor": "terminal trace with total count exactly k",
            "receiver": "continuing trace with total count k+2",
            "outcome": "</Think> logit minus next-index (k+1) logit at marker k",
            "scopes": ["index_only", "marker_only", "index_plus_marker"],
            "control": "same-scope donor from another continuing trace of total k+2",
            "rows": int(len(scope)),
        },
        "claim_boundary": (
            "These are teacher-forced local interventions.  They establish local "
            "necessity/sufficiency for the next decision, not free-running global sufficiency."
        ),
    }
    (output / "causal_protocol.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[aligned causal] wrote {tables}", flush=True)


if __name__ == "__main__":
    main()
