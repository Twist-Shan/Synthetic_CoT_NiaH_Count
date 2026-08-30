from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "Trace_Count_v24_2_Balanced_Count10_Colab.ipynb"
TARGET = ROOT / "notebooks" / "Trace_Count_v24_3_ComponentLoss_Count10_Colab.ipynb"


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
        "synthetic_counting_v24_2",
        "synthetic_counting_v24_3",
        expected=None,
    )
    _replace(notebook, "run_v24_2", "run_v24_3")
    _replace(notebook, 'VERSION = "v24.2"', 'VERSION = "v24.3"')
    _replace(
        notebook,
        "RUN_NAME = None",
        'RUN_NAME = "v24.3_componentloss_count1-10_seed1234"',
    )
    _replace(
        notebook,
        'PRESET = "main"                 # change to debug for a short end-to-end check',
        'PRESET = "main"                 # fixed: count 1-10 requires the matched 256-char setting',
    )
    _replace(
        notebook,
        'TRAINING_COUNT_DISTRIBUTION = "uniform"\n',
        'TRAINING_COUNT_DISTRIBUTION = "uniform"\n'
        'TASK_OUTPUT_LOSS_REDUCTION = "component_normalized"\n',
    )
    _replace(
        notebook,
        "    training_count_distribution=TRAINING_COUNT_DISTRIBUTION,\n",
        "    training_count_distribution=TRAINING_COUNT_DISTRIBUTION,\n"
        "    task_output_loss_reduction=TASK_OUTPUT_LOSS_REDUCTION,\n",
    )
    _replace(
        notebook,
        '    "--training-count-distribution", TRAINING_COUNT_DISTRIBUTION,\n',
        '    "--training-count-distribution", TRAINING_COUNT_DISTRIBUTION,\n'
        '    "--task-output-loss-reduction", TASK_OUTPUT_LOSS_REDUCTION,\n',
    )

    old_audit = """from synthetic_counting_v24.config import preset_config as v24_preset_config
V24_BASELINE_CONFIG = v24_preset_config(PRESET, seed=SEED, device=DEVICE)
changed_fields = {
    key for key, value in PLANNED_CONFIG.to_dict().items()
    if V24_BASELINE_CONFIG.to_dict().get(key) != value
}
assert changed_fields == {"version", "training_count_distribution"}, changed_fields
print("Controlled difference from v24:", sorted(changed_fields))
"""
    new_audit = """from dataclasses import asdict
from synthetic_counting_v24_2.config import preset_config as v24_2_preset_config
V24_2_BASELINE_CONFIG = v24_2_preset_config(PRESET, seed=SEED, device=DEVICE)
changed_fields = {
    key for key, value in asdict(PLANNED_CONFIG).items()
    if asdict(V24_2_BASELINE_CONFIG).get(key) != value
}
assert changed_fields == {"version", "task_output_loss_reduction"}, changed_fields
print("Controlled difference from v24.2:", sorted(changed_fields))
"""
    _replace(notebook, old_audit, new_audit)
    _replace(
        notebook,
        "assert PLANNED_CONFIG.training_count_distribution == TRAINING_COUNT_DISTRIBUTION\n",
        "assert PLANNED_CONFIG.training_count_distribution == TRAINING_COUNT_DISTRIBUTION\n"
        "assert PLANNED_CONFIG.task_output_loss_reduction == TASK_OUTPUT_LOSS_REDUCTION\n",
    )
    _replace(
        notebook,
        "assert PLANNED_CONFIG.cot_trace_loss_weight == 1.0\n",
        "assert PLANNED_CONFIG.cot_trace_loss_weight == 1.0\n"
        "assert PLANNED_CONFIG.task_output_count_weight == 1.0\n"
        "assert PLANNED_CONFIG.task_output_trace_weight == 1.0\n"
        "assert PLANNED_CONFIG.task_output_structure_weight == 0.1\n",
    )

    _set_cell_source(
        notebook["cells"][0],
        """# Trace Count v24.3: component-normalized loss control

This is a strict loss-only rerun of v24.2. The paired RoPE models, 256-character
Shakespeare context, three-character query, separator/no-index trace, atomic
answers, count support 1–10, 100-set pool, uniform-count sampler, seed,
optimizer, and 10,000-step schedule are unchanged.

Steps 1–1,500 keep the original all-sequence token-weighted mean. From step
1,501 onward, the task output is partitioned into final-count, trace, and
structure regions. Each region is averaged within each example and then across
the batch. Thinking optimizes `L_count + L_trace + 0.1 L_structure`;
Non-thinking optimizes `L_count + 0.1 L_structure`. The count coefficient is
therefore exactly 1 in both modes and no longer shrinks with trace length.

The notebook trains both modes, verifies the loss switch and effective region
coefficients, reruns phase plots and clean NCC, and persists all artifacts to
Google Drive.
""",
    )
    _set_cell_source(
        notebook["cells"][5],
        """## 3. Auditable loss-only contrast

V24.3 differs from v24.2 in one stored training field only:
`task_output_loss_reduction = component_normalized`.

The first 1,500 steps are identical. During steps 1,501–10,000, final-count,
trace, and structural tokens are normalized as semantic regions rather than
pooled into one token mean. A simple example: a Thinking trace with five
matches has ten trace tokens but one count token. V24.2 gives that count token
only 1/15 of the task-output token mean; v24.3 gives the count region coefficient
1, the same coefficient used for Non-thinking. The sampler remains the v24.2
uniform-count sampler so this run does not test set balancing.
""",
    )
    _set_cell_source(
        notebook["cells"][9],
        """## 5. Train both component-loss count-1–10 models

The variants are trained sequentially with identical target-count draws,
candidate examples, corpus split, pool, manifests, seed, and optimizer settings.
Only the post-step-1,500 loss reduction differs from v24.2.
""",
    )

    diagnostics = "".join(notebook["cells"][14]["source"])
    diagnostics += """
metrics = pd.read_csv(RUN_DIR / "tables" / "train_metrics.csv")
metrics["component_reduction_active"] = metrics["component_reduction_active"].astype(str).str.lower().eq("true")
pre = metrics[metrics["step"].le(MAX_STEPS_FOR_LANGUAGE_PRED)]
post = metrics[metrics["step"].gt(MAX_STEPS_FOR_LANGUAGE_PRED)]
assert not pre["component_reduction_active"].any()
assert post["component_reduction_active"].all()
last_post = post.sort_values("step").groupby("mode", as_index=False).tail(1)
expected_count_share = {"nonthinking": 1.0 / 1.1, "thinking": 1.0 / 2.1}
expected_trace_share = {"nonthinking": 0.0, "thinking": 1.0 / 2.1}
for row in last_post.itertuples(index=False):
    assert abs(row.batch_final_count_region_coefficient_share - expected_count_share[row.mode]) < 1e-9
    assert abs(row.batch_trace_region_coefficient_share - expected_trace_share[row.mode]) < 1e-9
display(last_post[[
    "mode", "step", "train_total_loss", "gradient_norm",
    "batch_final_count_token_weight_share",
    "batch_final_count_region_coefficient_share",
    "batch_trace_region_coefficient_share",
    "batch_structure_region_coefficient_share",
    "train_objective_final_count_region_mean_loss",
    "train_objective_trace_region_mean_loss",
    "train_objective_structure_region_mean_loss",
]])
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
