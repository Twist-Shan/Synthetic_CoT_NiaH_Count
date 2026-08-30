from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "Trace_Count_v24_4_MaxEnt_SetCount_Colab.ipynb"
TARGET = ROOT / "notebooks" / "Trace_Count_v24_5_Pool20_MaxEnt_Colab.ipynb"


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

    _replace(notebook, "synthetic_counting_v24_4", "synthetic_counting_v24_5", expected=None)
    _replace(notebook, "run_v24_4", "run_v24_5")
    _replace(notebook, 'VERSION = "v24.4"', 'VERSION = "v24.5"')
    _replace(notebook, "NEEDLE_POOL_SIZE = 100", "NEEDLE_POOL_SIZE = 20")
    _replace(
        notebook,
        'RUN_NAME = "v24.4_maxent_setcount_count1-10_seed1234"',
        'RUN_NAME = "v24.5_pool20_maxent_count1-10_seed1234"',
    )

    old_audit = """from dataclasses import asdict
from synthetic_counting_v24_3.config import preset_config as v24_3_preset_config
V24_3_BASELINE_CONFIG = v24_3_preset_config(PRESET, seed=SEED, device=DEVICE)
changed_fields = {
    key for key, value in asdict(PLANNED_CONFIG).items()
    if asdict(V24_3_BASELINE_CONFIG).get(key) != value
}
assert changed_fields == {"version", "training_count_distribution"}, changed_fields
print("Controlled difference from v24.3:", sorted(changed_fields))
"""
    new_audit = """from dataclasses import asdict
from synthetic_counting_v24_4.config import preset_config as v24_4_preset_config
V24_4_BASELINE_CONFIG = v24_4_preset_config(PRESET, seed=SEED, device=DEVICE)
changed_fields = {
    key for key, value in asdict(PLANNED_CONFIG).items()
    if asdict(V24_4_BASELINE_CONFIG).get(key) != value
}
assert changed_fields == {"version", "needle_pool_size"}, changed_fields
print("Controlled difference from v24.4:", sorted(changed_fields))
"""
    _replace(notebook, old_audit, new_audit)
    _replace(
        notebook,
        "assert PLANNED_CONFIG.needle_pool_frequency_threshold == 10.0 / 256.0\n",
        "assert PLANNED_CONFIG.needle_pool_frequency_threshold == 10.0 / 256.0\n"
        "assert PLANNED_CONFIG.needle_pool_size == NEEDLE_POOL_SIZE\n",
    )
    _replace(
        notebook,
        'assert (target_set["target_probability"] - 0.01).abs().max() < 1e-9',
        'assert (target_set["target_probability"] - 0.05).abs().max() < 1e-9',
    )
    _replace(
        notebook,
        'assert set_exposure.groupby("mode")["value"].nunique().eq(100).all()',
        'assert set_exposure.groupby("mode")["value"].nunique().eq(20).all()',
    )
    _replace(
        notebook,
        'assert (set_exposure["training_share"] - 0.01).abs().max() < 0.001',
        'assert (set_exposure["training_share"] - 0.05).abs().max() < 0.001',
    )

    _set_cell_source(
        notebook["cells"][0],
        """# Trace Count v24.5: 20-set maximum-entropy control

This is a strict supervision-density rerun of v24.4. The paired RoPE models,
component-normalized loss, 256-character Shakespeare context, separator/no-index
trace, atomic answers, count support 1–10, seed, optimizer, maximum-entropy
set/count sampler, and 10,000-step schedule are unchanged.

The only substantive change is reducing the needle pool from 100 sets to 20.
Each semantic marker therefore receives roughly five times as many training
examples, while the sampler still enforces a uniform 5% set marginal and 10%
count marginal. This tests whether v24.4 failed because marker-specific
supervision was too sparse rather than because the no-index counter is absent.

Both modes are retrained and evaluated with the same held-out TF, exact-trace
conditional readout, count-confusion, phase, causal, and NCC diagnostics.
""",
    )
    _set_cell_source(
        notebook["cells"][5],
        """## 3. Auditable pool-size-only contrast

V24.5 differs from v24.4 only in `needle_pool_size: 100 -> 20` (plus the
version label). Counts 1–10, maximum-entropy sampling, loss coefficients, model,
seed, and schedule remain fixed.

For example, with 100 sets and 10 counts, 320,000 task examples provide about
320 examples per set/count cell before feasibility corrections. With 20 sets,
the same budget provides about 1,600 examples per cell. The counter must still
retrieve the queried marker, but it sees enough repeats to learn a stable
marker-invariant counting and answer readout rule.
""",
    )
    _set_cell_source(
        notebook["cells"][9],
        """## 5. Train the paired 20-set control

Non-thinking and Thinking are trained sequentially from the same seed. They
receive the same 20-set/count cell draws, corpus windows, set orders, model
initialization, optimizer, and schedule. Only their output grammar differs.
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
