from __future__ import annotations

import json
from pathlib import Path

from build_v32_colab_notebook import build as build_v32_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v33_Scheduled_Rollin_Screen_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v32_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v32", "synthetic_counting_v33")
        source = source.replace("run_v32", "run_v33")
        source = source.replace('VERSION = "v32"', 'VERSION = "v33"')
        source = source.replace(
            'manifest["version"] == "v32"', 'manifest["version"] == "v33"'
        )
        source = source.replace(
            "v32_maxent_countweight8_independent_L256_pool100_seed",
            "v33_rollin05_steps6000_independent_L256_pool100_seed",
        )
        source = source.replace('"--train-steps", "10000"', '"--train-steps", "6000"')
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v33 screen: independent models with scheduled roll-in

v33 directly targets the failure diagnosed in v32. Non-thinking and Thinking
remain two separately trained models with numerically identical step-0
parameters. The 256-character prompt, counts 1--10, 100 marker sets,
maximum-entropy set/count sampler, separator/no-index trace, four-layer /
four-head / width-256 architecture, partial count-only output untying, and
component-normalized count coefficient 8 are unchanged.

The only new optimization mechanism is scheduled roll-in during the
task-output phase. For generated continuation inputs, the model's own
previous-token prediction replaces the gold input with a probability that
rises linearly from 0 at step 1,500 to 0.5 at step 6,000. The gold trace and
all labels remain byte-for-byte unchanged; there is no auxiliary decoder,
shared model, calibration, test-time update, or inference change. The same
algorithm is used for both modes; Non-thinking has no generated continuation
input before its one-token answer, so its roll-in set is empty by construction.

The 6,000-step budget was fixed from the v32 control curve before this run:
at that point Non-thinking had not yet saturated, making this a test of whether
the trace improves sample-efficient learning once exposure bias is controlled.
The behavioral gate is unchanged: Thinking accuracy >= 0.90, minimum per-count
accuracy >= 0.80, trace exact >= 0.90, count spread <= 0.20, and
Thinking-minus-Non-thinking accuracy >= 0.10. A failed screen is retained and
rejected without checkpoint selection.
""",
    )

    runtime = "".join(_cell(notebook, "runtime-settings")["source"])
    runtime = runtime.replace(
        "from synthetic_counting_v31.config import preset_config as v31_preset_config",
        "from synthetic_counting_v32.config import preset_config as v32_preset_config",
    )
    runtime = runtime.replace(
        "baseline = v31_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)",
        "baseline = v32_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)",
    )
    runtime = runtime.replace(
        'assert changed_fields == {"version", "training_count_distribution"}, changed_fields',
        "assert changed_fields == {\n"
        '    "version", "train_steps", "phase_cloud_steps",\n'
        '    "task_output_scheduled_sampling_max_probability",\n'
        "}, changed_fields",
    )
    runtime = runtime.replace(
        'assert baseline.training_count_distribution == "uniform"\n'
        'assert planned.training_count_distribution == "maxent_set_count"',
        'assert baseline.training_count_distribution == "maxent_set_count"\n'
        'assert planned.training_count_distribution == "maxent_set_count"',
    )
    runtime = runtime.replace(
        'assert planned.task_output_count_weight == 8.0',
        'assert planned.task_output_count_weight == 8.0\n'
        'assert baseline.task_output_scheduled_sampling_max_probability == 0.0\n'
        'assert planned.task_output_scheduled_sampling_max_probability == 0.5\n'
        'assert baseline.train_steps == 10000\n'
        'assert planned.train_steps == 6000\n'
        'assert planned.phase_cloud_steps[-1] == 6000',
    )
    runtime = runtime.replace(
        '"controlled_changes_from_v31": sorted(changed_fields),',
        '"controlled_changes_from_v32": sorted(changed_fields),',
    )
    runtime = runtime.replace(
        '"loss": "unchanged v31 component-normalized CE; count coefficient 8",\n'
        '    "sampler": "maximum-entropy feasible set x count cells",',
        '"loss": "unchanged component-normalized CE; count coefficient 8",\n'
        '    "sampler": "unchanged maximum-entropy feasible set x count cells",\n'
        '    "rollin": "linear 0 -> 0.5 over task-output phase; gold targets unchanged",\n'
        '    "budget": "fixed 6000 steps from the v32 control curve",',
    )
    runtime = runtime.replace(
        '"parameter_delta_from_v31": parameter_delta,',
        '"parameter_delta_from_v32": parameter_delta,',
    )
    _set_source(notebook, "runtime-settings", runtime)

    prepare = "".join(_cell(notebook, "prepare-data")["source"])
    prepare = prepare.replace(
        "# measured v31 shortcut.",
        "# retain the v32 low-shortcut target distribution.",
    )
    _set_source(notebook, "prepare-data", prepare)

    mechanism = "".join(_cell(notebook, "mechanism")["source"])
    mechanism = mechanism.replace(
        'role_table["step"].eq(10000)', 'role_table["step"].eq(6000)'
    )
    _set_source(notebook, "mechanism", mechanism)

    notebook["metadata"]["colab"]["name"] = TARGET.name
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
