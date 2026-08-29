from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "Trace_Count_v24_NoIndex_Count10_Colab.ipynb"
TARGET = ROOT / "notebooks" / "Trace_Count_v24_2_Balanced_Count10_Colab.ipynb"


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
        raise RuntimeError(
            f"expected {expected} occurrence(s) of {old!r}, found {matches}"
        )
    if expected is None and matches == 0:
        raise RuntimeError(f"could not locate {old!r}")


def build() -> Path:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    _replace(
        notebook,
        "synthetic_counting_v24",
        "synthetic_counting_v24_2",
        expected=None,
    )
    _replace(notebook, "run_v24", "run_v24_2")
    _replace(notebook, 'VERSION = "v24"', 'VERSION = "v24.2"')
    _replace(
        notebook,
        'OUT_ROOT = "runs/synthetic_counting_v24_2"',
        'OUT_ROOT = "runs/synthetic_counting_v24_2"',
    )
    _replace(
        notebook,
        "TASK_OCCURRENCE_RATIO = 1.0\n",
        'TASK_OCCURRENCE_RATIO = 1.0\nTRAINING_COUNT_DISTRIBUTION = "uniform"\n',
    )
    _replace(
        notebook,
        "    task_occurrence_ratio=TASK_OCCURRENCE_RATIO,\n",
        "    task_occurrence_ratio=TASK_OCCURRENCE_RATIO,\n"
        "    training_count_distribution=TRAINING_COUNT_DISTRIBUTION,\n",
    )
    _replace(
        notebook,
        '    "--task-occurrence-ratio", str(TASK_OCCURRENCE_RATIO),\n',
        '    "--task-occurrence-ratio", str(TASK_OCCURRENCE_RATIO),\n'
        '    "--training-count-distribution", TRAINING_COUNT_DISTRIBUTION,\n',
    )
    _replace(
        notebook,
        "assert PLANNED_CONFIG.trace_format == TRACE_FORMAT\n",
        "assert PLANNED_CONFIG.trace_format == TRACE_FORMAT\n"
        "assert PLANNED_CONFIG.version == VERSION\n"
        "assert PLANNED_CONFIG.training_count_distribution == TRAINING_COUNT_DISTRIBUTION\n",
    )
    _replace(
        notebook,
        '    "--output", str(NCC_OUTPUT),\n',
        '    "--output", str(NCC_OUTPUT),\n'
        '    "--run-prefix", RUN_DIR.name,\n'
        '    "--expected-version", VERSION,\n',
    )
    _replace(
        notebook,
        '    RUN_DIR / "tables" / "training_token_exposure_by_k.csv",\n',
        '    RUN_DIR / "tables" / "training_token_exposure_by_k.csv",\n'
        '    RUN_DIR / "tables" / "training_sampling_distribution.csv",\n',
    )
    _replace(
        notebook,
        '    DRIVE_RUN_DIR / "analysis" / "aligned_ncc" / "selected_confirmation_summary.csv",\n',
        '    DRIVE_RUN_DIR / "analysis" / "aligned_ncc" / "selected_confirmation_summary.csv",\n'
        '    DRIVE_RUN_DIR / "tables" / "training_sampling_distribution.csv",\n',
    )

    _set_cell_source(
        notebook["cells"][0],
        """# Trace Count v24.2: balanced-count control for paired no-index models

This is a strict single-variable rerun of v24. The model, 256-character
Shakespeare context, three-character query, RoPE, atomic answers, no-index
separator trace, 1–10 count support, 100-set pool with the 10/256 frequency
cap, unit loss weights, seed, optimizer, and 10,000-step schedule are unchanged.
Only the training count distribution changes from the naturally accepted
distribution to uniform sampling over counts 1–10. Non-thinking and Thinking
are both retrained on the same paired stream.

The primary behavioral endpoint is the 500-example balanced free-running final
test, especially counts 5 and 7. The notebook also reruns phase plots, local
head causality, and discovery/confirmation NCC so representational compression
can be compared with behavioral recovery. Dense snapshots are saved every 100
steps and full recovery state every 500 steps.
""",
    )
    _set_cell_source(
        notebook["cells"][5],
        """## 3. Auditable single-variable contrast

V24.2 differs from v24 only in `training_count_distribution`:

- v24: naturally accepted counts, ranging from 16.2% for count 1 to 5.4% for
  count 10 in the completed seed-1234 run;
- v24.2: target counts drawn uniformly from 1–10 before candidate filling.

If Thinking count-5/count-7 accuracy recovers, unequal exposure was causal or
contributory. If it remains selectively poor despite verified 10% exposure,
the stronger explanation is a count-dependent terminal-position/readout
failure. Fixed-length trace padding is intentionally not introduced here.
""",
    )
    _set_cell_source(
        notebook["cells"][9],
        """## 5. Train both balanced count-1–10 models (live progress is streamed)

The variants are trained sequentially with identical target-count draws,
candidate examples, corpus split, pool, manifests, seed, and optimizer settings.
""",
    )

    settings = "".join(notebook["cells"][6]["source"])
    audit_anchor = "print(PLANNED_CONFIG.to_dict())\n"
    audit_code = """from synthetic_counting_v24.config import preset_config as v24_preset_config
V24_BASELINE_CONFIG = v24_preset_config(PRESET, seed=SEED, device=DEVICE)
changed_fields = {
    key for key, value in PLANNED_CONFIG.to_dict().items()
    if V24_BASELINE_CONFIG.to_dict().get(key) != value
}
assert changed_fields == {"version", "training_count_distribution"}, changed_fields
print("Controlled difference from v24:", sorted(changed_fields))
"""
    if settings.count(audit_anchor) != 1:
        raise RuntimeError("could not locate the planned-config audit point")
    _set_cell_source(
        notebook["cells"][6],
        settings.replace(audit_anchor, audit_anchor + audit_code),
    )

    diagnostics = "".join(notebook["cells"][14]["source"])
    diagnostics += """
sampling = pd.read_csv(RUN_DIR / "tables" / "training_sampling_distribution.csv")
accepted = sampling[sampling["dimension"].eq("accepted_counts")].copy()
accepted["count"] = accepted["value"].astype(int)
accepted["training_share"] = accepted["examples"] / accepted["task_examples"]
accepted = accepted.sort_values(["mode", "count"])
assert accepted.groupby("mode")["count"].nunique().eq(10).all()
assert (accepted["training_share"] - 0.1).abs().max() < 0.005
display(accepted[["mode", "count", "examples", "training_share"]])
display(pd.read_csv(RUN_DIR / "tables" / "final_autoregressive_summary.csv"))
display(pd.read_csv(RUN_DIR / "tables" / "final_autoregressive_by_count.csv"))
"""
    _set_cell_source(notebook["cells"][14], diagnostics)

    notebook["metadata"]["colab"]["name"] = TARGET.name
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
