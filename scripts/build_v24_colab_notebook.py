from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "Trace_Count_v22_NoIndex_Colab.ipynb"
TARGET = ROOT / "notebooks" / "Trace_Count_v24_NoIndex_Count10_Colab.ipynb"


def _set_cell_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def _replace(notebook: dict, old: str, new: str, *, expected: int = 1) -> None:
    matches = 0
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        count = source.count(old)
        if count:
            matches += count
            _set_cell_source(cell, source.replace(old, new))
    if matches != expected:
        raise RuntimeError(
            f"expected {expected} occurrence(s) of {old!r}, found {matches}"
        )


def build() -> Path:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    _set_cell_source(
        notebook["cells"][0],
        """# Trace Count v24: paired no-index trace with count range 1–10

This is a capacity/class-resolution control for v22. It keeps the 256-character
Shakespeare context, three-character query, RoPE model, atomic final answer,
separator trace, unit loss weights, seed, optimizer, and 10,000-step schedule.
The only task change is that the accepted total target-character count is 1–10
instead of 1–30. Because that changes the training distribution, both
Non-thinking and Thinking are retrained from the same seed.

After training, the notebook computes running-count and final-count NCC
separately for both modes. Layer selection and centroids use discovery data;
reported NCC uses a disjoint confirmation set. Dense snapshots are saved every
100 steps and recovery state every 500 steps.
""",
    )
    _replace(notebook, "synthetic_counting_v22", "synthetic_counting_v24", expected=7)
    _replace(notebook, "run_v22", "run_v24", expected=1)
    _replace(notebook, 'VERSION = "v22"', 'VERSION = "v24"')
    _replace(notebook, "COUNT_MAX_THRESHOLD = 30", "COUNT_MAX_THRESHOLD = 10")
    _replace(notebook, "    COUNT_MAX_THRESHOLD = 4\n", "")
    _replace(
        notebook,
        'assert PLANNED_CONFIG.enabled_model_variants == ("rope/thinking",)',
        "assert PLANNED_CONFIG.enabled_model_variants == (\n"
        '    "rope/nonthinking", "rope/thinking"\n'
        ")\n"
        "assert PLANNED_CONFIG.count_max_threshold == 10\n"
        "assert PLANNED_CONFIG.final_count_loss_weight == 1.0\n"
        "assert PLANNED_CONFIG.cot_trace_loss_weight == 1.0",
    )
    _replace(
        notebook,
        '    "--model-variant", "rope/thinking",\n',
        "",
    )
    _set_cell_source(
        notebook["cells"][5],
        """## 3. Auditable controlled settings and hypotheses

The primary comparison is within v24, not v24 Thinking against v22/v20.

- If Thinking running-count NCC rises substantially at 1–10 while final-count
  NCC remains high, v22 was plausibly limited by 30-way class resolution or
  per-class exposure.
- If running-count NCC stays weak, the separator trace more likely uses a
  contextual/distributed progress state that nearest-centroid geometry does not
  capture.
- Non-thinking is retrained because reducing the count range changes its data
  distribution too.
""",
    )
    _set_cell_source(
        notebook["cells"][9],
        """## 5. Train both count-1–10 models (live progress is streamed)

The variants are trained sequentially on the identical corpus split, needle
pool, manifests, seed, and accepted count distribution.
""",
    )
    _replace(
        notebook,
        '    RUN_DIR / "figures" / "milestone_local_head_causality.png",\n',
        '    RUN_DIR / "figures" / "milestone_local_head_causality.png",\n'
        '    RUN_DIR / "analysis" / "aligned_ncc" / "selected_confirmation_summary.csv",\n',
    )
    analysis_cell = "".join(notebook["cells"][12]["source"])
    marker = 'analysis_stages = "phase,plots"\nrun_streaming([*base_cmd, "--stage", analysis_stages])\n'
    replacement = marker + """
NCC_OUTPUT = RUN_DIR / "analysis" / "aligned_ncc"
run_streaming([
    sys.executable, "-u", "scripts/compare_v24_modes_ncc.py",
    "--results-root", str(RUN_DIR.parent),
    "--output", str(NCC_OUTPUT),
    "--device", DEVICE,
    "--discovery-per-label", "10",
    "--confirmation-per-label", "8",
    "--batch-size", "32",
])
"""
    if analysis_cell.count(marker) != 1:
        raise RuntimeError("could not locate the phase-analysis command")
    _set_cell_source(notebook["cells"][12], analysis_cell.replace(marker, replacement))
    diagnostic_cell = "".join(notebook["cells"][14]["source"])
    diagnostic_cell += """
display(pd.read_csv(NCC_OUTPUT / "selected_confirmation_summary.csv")[[
    "comparison_mode", "endpoint", "layer",
    "chance_balanced_accuracy",
    "confirmation_logistic_balanced_accuracy",
    "confirmation_ncc_balanced_accuracy",
    "confirmation_ncc_above_chance",
]])
"""
    _set_cell_source(notebook["cells"][14], diagnostic_cell)
    _replace(
        notebook,
        '    DRIVE_RUN_DIR / "checkpoints" / "rope" / "thinking" / "snapshot_index.csv",\n',
        '    DRIVE_RUN_DIR / "checkpoints" / "rope" / "nonthinking" / "snapshot_index.csv",\n'
        '    DRIVE_RUN_DIR / "checkpoints" / "rope" / "thinking" / "snapshot_index.csv",\n',
    )
    _replace(
        notebook,
        '    DRIVE_RUN_DIR / "checkpoints" / "rope" / "thinking" / "final" / "checkpoint.pt",\n',
        '    DRIVE_RUN_DIR / "checkpoints" / "rope" / "nonthinking" / "final" / "checkpoint.pt",\n'
        '    DRIVE_RUN_DIR / "checkpoints" / "rope" / "thinking" / "final" / "checkpoint.pt",\n'
        '    DRIVE_RUN_DIR / "analysis" / "aligned_ncc" / "selected_confirmation_summary.csv",\n',
    )

    notebook["metadata"]["colab"]["name"] = TARGET.name
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
