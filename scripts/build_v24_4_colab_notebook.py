from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "Trace_Count_v24_3_ComponentLoss_Count10_Colab.ipynb"
TARGET = ROOT / "notebooks" / "Trace_Count_v24_4_MaxEnt_SetCount_Colab.ipynb"


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

    _replace(notebook, "synthetic_counting_v24_3", "synthetic_counting_v24_4", expected=None)
    _replace(notebook, "run_v24_3", "run_v24_4")
    _replace(notebook, 'VERSION = "v24.3"', 'VERSION = "v24.4"')
    _replace(
        notebook,
        'TRAINING_COUNT_DISTRIBUTION = "uniform"',
        'TRAINING_COUNT_DISTRIBUTION = "maxent_set_count"',
    )
    _replace(
        notebook,
        'RUN_NAME = "v24.3_componentloss_count1-10_seed1234"',
        'RUN_NAME = "v24.4_maxent_setcount_count1-10_seed1234"',
    )

    old_audit = """from dataclasses import asdict
from synthetic_counting_v24_2.config import preset_config as v24_2_preset_config
V24_2_BASELINE_CONFIG = v24_2_preset_config(PRESET, seed=SEED, device=DEVICE)
changed_fields = {
    key for key, value in asdict(PLANNED_CONFIG).items()
    if asdict(V24_2_BASELINE_CONFIG).get(key) != value
}
assert changed_fields == {"version", "task_output_loss_reduction"}, changed_fields
print("Controlled difference from v24.2:", sorted(changed_fields))
"""
    new_audit = """from dataclasses import asdict
from synthetic_counting_v24_3.config import preset_config as v24_3_preset_config
V24_3_BASELINE_CONFIG = v24_3_preset_config(PRESET, seed=SEED, device=DEVICE)
changed_fields = {
    key for key, value in asdict(PLANNED_CONFIG).items()
    if asdict(V24_3_BASELINE_CONFIG).get(key) != value
}
assert changed_fields == {"version", "training_count_distribution"}, changed_fields
print("Controlled difference from v24.3:", sorted(changed_fields))
"""
    _replace(notebook, old_audit, new_audit)

    _replace(
        notebook,
        '    RUN_DIR / "tables" / "training_sampling_distribution.csv",\n',
        '    RUN_DIR / "tables" / "training_sampling_distribution.csv",\n'
        '    RUN_DIR / "tables" / "training_set_count_sampler_plan.csv",\n',
        expected=None,
    )
    _replace(
        notebook,
        '    DRIVE_RUN_DIR / "tables" / "training_sampling_distribution.csv",\n',
        '    DRIVE_RUN_DIR / "tables" / "training_sampling_distribution.csv",\n'
        '    DRIVE_RUN_DIR / "tables" / "training_set_count_sampler_plan.csv",\n',
    )

    _set_cell_source(
        notebook["cells"][0],
        """# Trace Count v24.4: maximum-entropy set × count sampling

This is a strict sampler-only rerun of v24.3. The paired RoPE models,
component-normalized loss, 256-character Shakespeare context, separator/no-index
trace, atomic answers, count support 1–10, 100-set pool, seed, optimizer, and
10,000-step schedule are unchanged.

The only change is the training distribution. V24.4 indexes every feasible
`(needle set, count)` cell in the training split and uses iterative
proportional fitting to obtain the maximum-entropy distribution with a uniform
1% set marginal and uniform 10% count marginal. Impossible cells retain zero
probability; no set is silently dropped.

The notebook trains both modes, audits realized joint exposure, reruns phase
plots and aligned NCC, and explicitly reports held-out teacher-forced accuracy,
free-running accuracy conditional on an exact trace, and count-confusion tables.
""",
    )
    _set_cell_source(
        notebook["cells"][5],
        """## 3. Auditable sampler-only contrast

V24.4 differs from v24.3 in one stored training field only:
`training_count_distribution = maxent_set_count`.

For example, suppose set A cannot naturally occur exactly nine times in a
256-character window. A naive uniform-cell sampler must either fail or drop A.
The maximum-entropy plan instead assigns zero mass to `(A, 9)` while adjusting
the other feasible cells so A still receives 1% total mass and count 9 still
receives 10% total mass. Loss coefficients and all model settings remain fixed.
""",
    )
    _set_cell_source(
        notebook["cells"][9],
        """## 5. Train the paired maximum-entropy sampler control

Non-thinking and Thinking are trained sequentially from the same seed. They
receive the same set/count cell draws, corpus windows, rendered set orders,
model initialization, optimizer, and schedule. Only their output grammar differs.
""",
    )

    diagnostics = "".join(notebook["cells"][14]["source"])
    diagnostics += """

sampler_plan = pd.read_csv(RUN_DIR / "tables" / "training_set_count_sampler_plan.csv")
target_set = sampler_plan.groupby("set_id", as_index=False)["target_probability"].sum()
target_count = sampler_plan.groupby("count", as_index=False)["target_probability"].sum()
assert (target_set["target_probability"] - 0.01).abs().max() < 1e-9
assert (target_count["target_probability"] - 0.10).abs().max() < 1e-9

set_exposure = sampling[sampling["dimension"].eq("set_ids")].copy()
set_exposure["training_share"] = set_exposure["examples"] / set_exposure["task_examples"]
assert set_exposure.groupby("mode")["value"].nunique().eq(100).all()
assert (set_exposure["training_share"] - 0.01).abs().max() < 0.001

final_summary = pd.read_csv(RUN_DIR / "tables" / "final_autoregressive_summary.csv")
final_by_count = pd.read_csv(RUN_DIR / "tables" / "final_autoregressive_by_count.csv")
final_detail = pd.read_csv(RUN_DIR / "tables" / "final_autoregressive_detail.csv")
eval_detail = pd.read_csv(RUN_DIR / "tables" / "eval_detail.csv")

thinking_summary = final_summary[final_summary["mode"].eq("thinking")].iloc[-1]
thinking_by_count = final_by_count[final_by_count["mode"].eq("thinking")].sort_values("count")
exact_trace = final_detail[
    final_detail["mode"].eq("thinking") & final_detail["trace_exact"].eq(1.0)
]
conditional_answer_accuracy_given_exact_trace = (
    exact_trace.groupby("count", as_index=False)["ar_accuracy"].mean()
)
readout_confusion = pd.crosstab(
    final_detail.loc[final_detail["mode"].eq("thinking"), "count"],
    final_detail.loc[final_detail["mode"].eq("thinking"), "ar_pred_count"],
    normalize="index",
)
heldout_tf = (
    eval_detail[
        eval_detail["mode"].eq("thinking")
        & eval_detail["step"].eq(MAX_TRAIN_STEPS)
    ]
    .groupby("count", as_index=False)["tf_final_accuracy"]
    .mean()
)

overall_accuracy = float(thinking_summary["ar_final_accuracy"])
minimum_count_accuracy = float(thinking_by_count["ar_final_accuracy"].min())
count_accuracy_spread = float(
    thinking_by_count["ar_final_accuracy"].max()
    - thinking_by_count["ar_final_accuracy"].min()
)
trace_exact_accuracy = float(thinking_summary["trace_exact"])
success_criteria_met = bool(
    overall_accuracy >= 0.90
    and minimum_count_accuracy >= 0.85
    and count_accuracy_spread <= 0.10
    and trace_exact_accuracy >= 0.90
)
print({
    "success_criteria_met": success_criteria_met,
    "overall_accuracy": overall_accuracy,
    "minimum_count_accuracy": minimum_count_accuracy,
    "count_accuracy_spread": count_accuracy_spread,
    "trace_exact_accuracy": trace_exact_accuracy,
})
display(target_set.describe())
display(target_count)
display(set_exposure.groupby("mode")["training_share"].describe())
display(thinking_by_count[["count", "ar_final_accuracy", "trace_exact"]])
display(heldout_tf)
display(conditional_answer_accuracy_given_exact_trace)
display(readout_confusion)
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
