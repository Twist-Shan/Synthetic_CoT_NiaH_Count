from __future__ import annotations

import json
from pathlib import Path

from build_v34_colab_notebook import build as build_v34_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v35_EqualComponents_Screen_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v34_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v34", "synthetic_counting_v35")
        source = source.replace("run_v34", "run_v35")
        source = source.replace('VERSION = "v34"', 'VERSION = "v35"')
        source = source.replace(
            'manifest["version"] == "v34"', 'manifest["version"] == "v35"'
        )
        source = source.replace(
            "v34_traceweight8_steps6000_independent_L256_pool100_seed",
            "v35_equalcomponents8_steps6000_independent_L256_pool100_seed",
        )
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v35 screen: independent models with equal semantic components

v35 is the one-scalar correction to the failure diagnosed in v34.
Non-thinking and Thinking remain separately initialized-and-trained models
with identical step-0 parameters. The 256-character prompt, counts 1--10,
100 marker sets, maximum-entropy set/count sampler, separator/no-index trace,
four-layer / four-head / width-256 architecture, partial count-only output
untying, pure gold-prefix teacher forcing, and fixed 6,000-step schedule are
unchanged.

v34 assigned 49.69% of aggregate coefficient mass to final count, 49.69% to
trace markers, and only 0.62% to structure. It therefore learned marker
identity under gold prefixes but retained high `</Think>` loss and generated
almost always 9--10 markers. v35 changes only the component-normalized
structure coefficient from 0.1 to 8. Count, trace, and structure now each
receive one third of aggregate objective mass. No target token, input prefix,
parameter, sampler, decoder, calibration, test-time update, or inference rule
changes.

The fixed seed-1234 behavioral gate remains Thinking accuracy >= 0.90,
minimum per-count accuracy >= 0.80, trace exact >= 0.90, count spread <= 0.20,
and Thinking-minus-Non-thinking accuracy >= 0.10. A failed screen is retained
and rejected without checkpoint selection; NCC and mechanism experiments run
only after the gate passes.
""",
    )

    runtime = "".join(_cell(notebook, "runtime-settings")["source"])
    runtime = runtime.replace(
        "from synthetic_counting_v32.config import preset_config as v32_preset_config",
        "from synthetic_counting_v34.config import preset_config as v34_preset_config",
    )
    runtime = runtime.replace(
        "baseline = v32_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)",
        "baseline = v34_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)",
    )
    runtime = runtime.replace(
        "assert changed_fields == {\n"
        '    "version", "train_steps", "phase_cloud_steps",\n'
        '    "task_output_trace_weight",\n'
        "}, changed_fields",
        'assert changed_fields == {"version", "task_output_structure_weight"}, changed_fields',
    )
    runtime = runtime.replace(
        "assert baseline.task_output_trace_weight == 1.0\n"
        "assert planned.task_output_trace_weight == 8.0\n"
        "assert baseline.task_output_scheduled_sampling_max_probability == 0.0\n"
        "assert planned.task_output_scheduled_sampling_max_probability == 0.0\n"
        "assert baseline.train_steps == 10000\n"
        "assert planned.train_steps == 6000\n"
        "assert planned.phase_cloud_steps[-1] == 6000",
        "assert baseline.task_output_trace_weight == 8.0\n"
        "assert planned.task_output_trace_weight == 8.0\n"
        "assert baseline.task_output_structure_weight == 0.1\n"
        "assert planned.task_output_scheduled_sampling_max_probability == 0.0\n"
        "assert baseline.train_steps == 6000\n"
        "assert planned.train_steps == 6000\n"
        "assert planned.phase_cloud_steps[-1] == 6000",
    )
    runtime = runtime.replace(
        "assert planned.task_output_structure_weight == 0.1",
        "assert planned.task_output_structure_weight == 8.0\n"
        "assert (\n"
        "    planned.task_output_count_weight\n"
        "    == planned.task_output_trace_weight\n"
        "    == planned.task_output_structure_weight\n"
        ")",
    )
    runtime = runtime.replace(
        '"controlled_changes_from_v32": sorted(changed_fields),',
        '"controlled_changes_from_v34": sorted(changed_fields),',
    )
    runtime = runtime.replace(
        '"loss": "component-normalized CE; count 8, trace 1 -> 8",',
        '"loss": "component-normalized CE; count 8, trace 8, structure 0.1 -> 8",',
    )
    runtime = runtime.replace(
        '"parameter_delta_from_v32": parameter_delta,',
        '"parameter_delta_from_v34": parameter_delta,',
    )
    _set_source(notebook, "runtime-settings", runtime)

    notebook["metadata"]["colab"]["name"] = TARGET.name
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
