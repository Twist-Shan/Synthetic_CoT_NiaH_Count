from __future__ import annotations

import json
from pathlib import Path

from build_v32_colab_notebook import build as build_v32_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v34_TraceWeight8_Screen_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v32_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v32", "synthetic_counting_v34")
        source = source.replace("run_v32", "run_v34")
        source = source.replace('VERSION = "v32"', 'VERSION = "v34"')
        source = source.replace(
            'manifest["version"] == "v32"', 'manifest["version"] == "v34"'
        )
        source = source.replace(
            "v32_maxent_countweight8_independent_L256_pool100_seed",
            "v34_traceweight8_steps6000_independent_L256_pool100_seed",
        )
        source = source.replace('"--train-steps", "10000"', '"--train-steps", "6000"')
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v34 screen: independent models, gold prefixes, stronger trace loss

v34 is the loss-only response to the v33 scheduled-roll-in failure.
Non-thinking and Thinking remain two separately initialized-and-trained models
with numerically identical step-0 parameters. The 256-character prompt,
counts 1--10, 100 marker sets, maximum-entropy set/count sampler,
separator/no-index trace, four-layer / four-head / width-256 architecture,
partial count-only output untying, and component-normalized count coefficient
8 are unchanged from v32.

The only semantic optimization change is that the already
component-normalized Thinking trace region receives coefficient 8 instead of
1, matching the final-count coefficient. Training uses pure gold-prefix
teacher forcing: no input token is corrupted and the target trace is unchanged.
There is no shared model, auxiliary decoder, calibration, test-time update,
trace rewrite, constrained decoding, or inference-rule change. The fixed
6,000-step budget from v33 is retained to measure sample-efficient separation
before direct Non-thinking counting saturates.

The fixed seed-1234 behavioral gate is unchanged: Thinking accuracy >= 0.90,
minimum per-count accuracy >= 0.80, trace exact >= 0.90, count spread <= 0.20,
and Thinking-minus-Non-thinking accuracy >= 0.10. A failed screen is retained
and rejected without checkpoint selection; NCC and mechanism experiments run
only after the gate passes.
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
        '    "task_output_trace_weight",\n'
        "}, changed_fields",
    )
    runtime = runtime.replace(
        'assert baseline.training_count_distribution == "uniform"\n'
        'assert planned.training_count_distribution == "maxent_set_count"',
        'assert baseline.training_count_distribution == "maxent_set_count"\n'
        'assert planned.training_count_distribution == "maxent_set_count"',
    )
    runtime = runtime.replace(
        "assert planned.task_output_trace_weight == 1.0",
        "assert baseline.task_output_trace_weight == 1.0\n"
        "assert planned.task_output_trace_weight == 8.0\n"
        "assert baseline.task_output_scheduled_sampling_max_probability == 0.0\n"
        "assert planned.task_output_scheduled_sampling_max_probability == 0.0\n"
        "assert baseline.train_steps == 10000\n"
        "assert planned.train_steps == 6000\n"
        "assert planned.phase_cloud_steps[-1] == 6000",
    )
    runtime = runtime.replace(
        '"controlled_changes_from_v31": sorted(changed_fields),',
        '"controlled_changes_from_v32": sorted(changed_fields),',
    )
    runtime = runtime.replace(
        '"loss": "unchanged v31 component-normalized CE; count coefficient 8",\n'
        '    "sampler": "maximum-entropy feasible set x count cells",',
        '"loss": "component-normalized CE; count 8, trace 1 -> 8",\n'
        '    "sampler": "unchanged maximum-entropy feasible set x count cells",\n'
        '    "teacher_forcing": "pure gold-prefix inputs; scheduled sampling 0",\n'
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
