from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "Trace_Count_v24_6_UntiedHead_Colab.ipynb"
TARGET = ROOT / "notebooks" / "Trace_Count_v24_7_AnswerCompression_Colab.ipynb"


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

    _replace(notebook, "synthetic_counting_v24_6", "synthetic_counting_v24_7", expected=None)
    _replace(notebook, "run_v24_6", "run_v24_7")
    _replace(notebook, 'VERSION = "v24.6"', 'VERSION = "v24.7"')
    _replace(
        notebook,
        'RUN_NAME = "v24.6_untied_head_pool20_count1-10_seed1234"',
        'RUN_NAME = "v24.7_answer_compression_pool20_count1-10_seed1234"',
    )

    old_audit = """from dataclasses import asdict
from synthetic_counting_v24_5.config import preset_config as v24_5_preset_config
V24_5_BASELINE_CONFIG = v24_5_preset_config(PRESET, seed=SEED, device=DEVICE)
changed_fields = {
    key for key, value in asdict(PLANNED_CONFIG).items()
    if asdict(V24_5_BASELINE_CONFIG).get(key) != value
}
assert changed_fields == {"version", "tie_word_embeddings"}, changed_fields
print("Controlled difference from v24.5:", sorted(changed_fields))
"""
    new_audit = """from dataclasses import asdict
from synthetic_counting_v24_6.config import preset_config as v24_6_preset_config
V24_6_BASELINE_CONFIG = v24_6_preset_config(PRESET, seed=SEED, device=DEVICE)
changed_fields = {
    key for key, value in asdict(PLANNED_CONFIG).items()
    if asdict(V24_6_BASELINE_CONFIG).get(key) != value
}
assert changed_fields == {
    "version",
    "answer_query_contrastive_weight",
}, changed_fields
print("Controlled difference from v24.6:", sorted(changed_fields))
"""
    _replace(notebook, old_audit, new_audit)
    _replace(
        notebook,
        "assert not PLANNED_CONFIG.tie_word_embeddings\n",
        "assert not PLANNED_CONFIG.tie_word_embeddings\n"
        "assert PLANNED_CONFIG.answer_query_contrastive_weight == 0.1\n"
        "assert PLANNED_CONFIG.answer_query_contrastive_temperature == 0.1\n",
    )

    _set_cell_source(
        notebook["cells"][0],
        """# Trace Count v24.7: native answer-query representation compression

This is a strict training-objective control for v24.6. The paired RoPE models,
untied standard LM head, 20-set maximum-entropy sampler, component-normalized
token loss, 256-character Shakespeare context, separator/no-index trace,
atomic answer vocabulary, count support 1–10, seed, optimizer, and 10,000-step
schedule are unchanged.

The only substantive change is a weight-0.1 supervised contrastive loss on the
native `<Ans>` query residual during steps 1,501–10,000. Same-count examples in
the current batch are pulled together; different-count examples are separated.
It adds no tokens, auxiliary classifier, external decoder, or inference-time
rule. The ordinary model LM head must still emit the raw count token.

Success is judged only by raw autoregressive output: Thinking overall accuracy
at least 0.90, every count at least 0.85, maximum-minus-minimum count accuracy
at most 0.10, and trace exact at least 0.90. The trace-count decoder remains a
diagnostic and cannot satisfy the gate.
""",
    )
    _set_cell_source(
        notebook["cells"][5],
        """## 3. Auditable one-variable representation-loss contrast

V24.7 differs from v24.6 only by enabling answer-query supervised contrastive
compression with weight 0.1 (the shared temperature already defaults to 0.1).
The loss is inactive during the first 1,500 language-model steps and active
only in the task-output phase. It operates on the final normalized residual at
the `<Ans>` position whose ordinary LM logits predict the answer token.

For example, two prompts whose correct count is 7 should have nearby `<Ans>`
residuals even if they use different marker sets and Shakespeare windows; a
count-7 residual should be separated from count-6 and count-8 residuals. This
directly tests whether stronger representation compression raises both NCC and
the model's own raw answer accuracy.
""",
    )
    _set_cell_source(
        notebook["cells"][9],
        """## 5. Train the paired answer-compression control

Non-thinking and Thinking are trained sequentially from the same seed. They
receive the same set/count draws, corpus windows, set orders, transformer
initialization, optimizer, schedule, and answer-query contrastive objective.
Their within-version comparison remains controlled; only their output grammar
differs. No checkpoint from v24.6 is warm-started.
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
