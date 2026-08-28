from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "Trace_Count_v22_NoIndex_Colab.ipynb"
TARGET = ROOT / "notebooks" / "Trace_Count_v23_NoIndex_FCW8_Colab.ipynb"


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
        """# Trace Count v23: final-count-upweighted paired no-index run

Controlled setting: 256 Shakespeare characters, a three-character query before
the data, RoPE, atomic final answers, and count range 1–30. This notebook
re-trains both Non-thinking and Thinking from the same seed. The Thinking trace
is identical to v22 and replaces every explicit ordinal index one-for-one by
the same `<Sep>` token:
`<Think> <Sep> marker_1 <Sep> marker_2 ... </Think> <Ans> <n>`.

The only objective change from v22 is `final_count_loss_weight=8`; the trace
weight remains 1. Dense scientific snapshots are stored every 100 steps, full
optimizer/RNG recovery state every 500 steps, and five snapshots are packed
into each shard. This yields a controlled v22→v23 loss-balance comparison and
a within-v23 Thinking/Non-thinking comparison.
""",
    )
    _replace(notebook, "synthetic_counting_v22", "synthetic_counting_v23", expected=7)
    _replace(notebook, "run_v22", "run_v23", expected=1)
    _replace(notebook, 'VERSION = "v22"', 'VERSION = "v23"')
    _set_cell_source(
        notebook["cells"][5],
        """## 3. Auditable experiment settings

`final_count_loss_weight=8` is the same per-token coefficient in both modes;
it does not equalize the count token's normalized fraction of total loss. After
the step-1,500 task-output switch, that fraction is `8/10 = 80%` for
Non-thinking (`<Ans>`, count, `<EOS>`) and `8/(2n+12)` for a Thinking example
with count `n` (about 19% at `n=15`, versus about 2.9% under v22's weight 1).
""",
    )
    _replace(
        notebook,
        "MAX_STEPS_FOR_LANGUAGE_PRED = 1_500\n",
        "MAX_STEPS_FOR_LANGUAGE_PRED = 1_500\nFINAL_COUNT_LOSS_WEIGHT = 8.0\n",
    )
    _replace(
        notebook,
        "    max_steps_for_language_pred=MAX_STEPS_FOR_LANGUAGE_PRED,\n",
        "    max_steps_for_language_pred=MAX_STEPS_FOR_LANGUAGE_PRED,\n"
        "    final_count_loss_weight=FINAL_COUNT_LOSS_WEIGHT,\n",
    )
    _replace(
        notebook,
        'assert PLANNED_CONFIG.enabled_model_variants == ("rope/thinking",)',
        "assert PLANNED_CONFIG.enabled_model_variants == (\n"
        '    "rope/nonthinking", "rope/thinking"\n'
        ")\n"
        "assert PLANNED_CONFIG.final_count_loss_weight == FINAL_COUNT_LOSS_WEIGHT",
    )
    _replace(
        notebook,
        '    "trace_format": PLANNED_CONFIG.trace_format,\n',
        '    "trace_format": PLANNED_CONFIG.trace_format,\n'
        '    "final_count_loss_weight": PLANNED_CONFIG.final_count_loss_weight,\n'
        '    "cot_trace_loss_weight": PLANNED_CONFIG.cot_trace_loss_weight,\n',
    )
    _replace(
        notebook,
        '    "--max-steps-for-language-pred", str(MAX_STEPS_FOR_LANGUAGE_PRED),\n',
        '    "--max-steps-for-language-pred", str(MAX_STEPS_FOR_LANGUAGE_PRED),\n'
        '    "--final-count-loss-weight", str(FINAL_COUNT_LOSS_WEIGHT),\n',
    )
    _replace(
        notebook,
        '    "--model-variant", "rope/thinking",\n',
        "",
    )
    _set_cell_source(
        notebook["cells"][9],
        """## 5. Train both FCW8 models (live progress is streamed)

The two variants are trained sequentially with identical data/configuration.
`rope/nonthinking` is the paired direct-answer control; `rope/thinking` is the
v22-matched separator-trace treatment.
""",
    )
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
        '    DRIVE_RUN_DIR / "checkpoints" / "rope" / "thinking" / "final" / "checkpoint.pt",\n',
    )

    notebook["metadata"]["colab"]["name"] = TARGET.name
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
