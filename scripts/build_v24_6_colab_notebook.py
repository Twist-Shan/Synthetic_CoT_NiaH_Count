from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "Trace_Count_v24_5_Pool20_MaxEnt_Colab.ipynb"
TARGET = ROOT / "notebooks" / "Trace_Count_v24_6_UntiedHead_Colab.ipynb"


def _set_cell_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def _replace(notebook: dict, old: str, new: str, *, expected: int | None = 1) -> None:
    matches = 0
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        count = source.count(old)
        if count:
            matches += count
            _set_cell_source(cell, source.replace(old, new))
    if expected is not None and matches != expected:
        raise RuntimeError(f"expected {expected} occurrence(s) of {old!r}, found {matches}")
    if expected is None and matches == 0:
        raise RuntimeError(f"could not locate {old!r}")


def build() -> Path:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    _replace(notebook, "synthetic_counting_v24_5", "synthetic_counting_v24_6", expected=None)
    _replace(notebook, "run_v24_5", "run_v24_6")
    _replace(notebook, 'VERSION = "v24.5"', 'VERSION = "v24.6"')
    _replace(
        notebook,
        'RUN_NAME = "v24.5_pool20_maxent_count1-10_seed1234"',
        'RUN_NAME = "v24.6_untied_head_pool20_count1-10_seed1234"',
    )

    old_audit = """from dataclasses import asdict
from synthetic_counting_v24_4.config import preset_config as v24_4_preset_config
V24_4_BASELINE_CONFIG = v24_4_preset_config(PRESET, seed=SEED, device=DEVICE)
changed_fields = {
    key for key, value in asdict(PLANNED_CONFIG).items()
    if asdict(V24_4_BASELINE_CONFIG).get(key) != value
}
assert changed_fields == {"version", "needle_pool_size"}, changed_fields
print("Controlled difference from v24.4:", sorted(changed_fields))
"""
    new_audit = """from dataclasses import asdict
from synthetic_counting_v24_5.config import preset_config as v24_5_preset_config
V24_5_BASELINE_CONFIG = v24_5_preset_config(PRESET, seed=SEED, device=DEVICE)
changed_fields = {
    key for key, value in asdict(PLANNED_CONFIG).items()
    if asdict(V24_5_BASELINE_CONFIG).get(key) != value
}
assert changed_fields == {"version", "tie_word_embeddings"}, changed_fields
print("Controlled difference from v24.5:", sorted(changed_fields))
"""
    _replace(notebook, old_audit, new_audit)
    _replace(
        notebook,
        "assert PLANNED_CONFIG.needle_pool_size == NEEDLE_POOL_SIZE\n",
        "assert PLANNED_CONFIG.needle_pool_size == NEEDLE_POOL_SIZE\n"
        "assert not PLANNED_CONFIG.tie_word_embeddings\n",
    )
    _replace(
        notebook,
        "trace_readout_success_criteria_met",
        "trace_readout_diagnostic_criteria_met",
        expected=None,
    )

    _set_cell_source(
        notebook["cells"][0],
        """# Trace Count v24.6: untied raw-answer head

This is a strict readout-bridge control for v24.5. The paired RoPE models,
20-set maximum-entropy sampler, component-normalized loss, 256-character
Shakespeare context, separator/no-index trace, atomic answer vocabulary, count
support 1–10, seed, optimizer, and 10,000-step schedule are unchanged.

The only substantive change is `tie_word_embeddings: True -> False`. The
untied LM output matrix is copied from the input embedding at initialization,
so v24.5 and v24.6 have exactly identical step-zero logits. Independent
gradients thereafter test whether the tied readout constraint caused v24.5's
count-specific raw-answer failures.

Success is judged only by the model's raw autoregressive count token after
`<Ans>`: overall accuracy at least 0.90, every count at least 0.85, maximum
minus minimum count accuracy at most 0.10, and Thinking trace exact at least
0.90. The trace-count decoder remains a diagnostic and is not a success metric.
""",
    )
    _set_cell_source(
        notebook["cells"][5],
        """## 3. Auditable one-variable readout contrast

V24.6 differs from v24.5 only in `tie_word_embeddings: True -> False` (plus
the version label). At step zero, the untied LM head is an exact copy of the
input embedding, so both versions produce identical logits. Once training
starts, the output classifier can learn count directions without simultaneously
changing how those same count tokens act as inputs when the model predicts
`<EOS>`.

For example, the hidden state before the answer may already distinguish count
5 from count 8. A tied head must use the `<5>` input embedding as its count-5
classification direction as well; an untied head gives those two roles separate
parameters while preserving standard autoregressive token generation.
""",
    )
    _set_cell_source(
        notebook["cells"][9],
        """## 5. Train the paired untied-head control

Non-thinking and Thinking are trained sequentially from the same seed. They
receive the same set/count draws, corpus windows, set orders, transformer
initialization, optimizer, and schedule. Both models use an untied output head,
so their within-version comparison remains controlled; only their output
grammar differs.
""",
    )

    notebook["metadata"]["colab"]["name"] = TARGET.name
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
