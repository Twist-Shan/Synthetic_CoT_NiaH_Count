from __future__ import annotations

import json
from pathlib import Path

from build_v28_colab_notebook import build as build_v28_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v29_CountWeight4_Multiseed_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v28_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))

    # v29 deliberately reuses the complete v28 notebook topology.  This
    # mechanical substitution is followed by explicit scientific-audit cells
    # below so the controlled difference cannot be hidden in prose or code.
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v28", "synthetic_counting_v29")
        source = source.replace("run_v28", "run_v29")
        source = source.replace('VERSION = "v28"', 'VERSION = "v29"')
        source = source.replace('manifest["version"] == "v28"', 'manifest["version"] == "v29"')
        source = source.replace(
            "v28_partial_count_readout_L256_pool100_seed",
            "v29_countweight4_fixed_partial_readout_L256_pool100_seed",
        )
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v29: single-scalar count-readout correction

This is a pre-registered three-seed paired experiment. It keeps v28's
256-character prompt, count support 1--10, 100 marker sets, uniform semantic
count sampler, separator/no-index trace, 4-layer/4-head/256-wide transformer,
partial count-only output untying, optimizer, 1,500-step language phase, and
10,000-step schedule.

The sole change from v28 is in the already component-normalized task-output
loss: the final-count region coefficient increases from 1 to 4. The trace
region remains 1 and structure remains 0.1. The trace tokens and targets are
unchanged. There is no conditional ten-way objective, auxiliary trace loss,
contrastive loss, post-hoc decoder, frozen phase, or test-time update.

Primary gate, fixed before opening the three final test sets: in every seed,
Thinking accuracy >= 0.90, minimum per-count accuracy >= 0.80, trace exact >=
0.90, count spread <= 0.20, and Thinking-minus-Non-thinking accuracy >= 0.10.
All completed seeds are retained regardless of outcome.
""",
    )

    runtime = "".join(_cell(notebook, "runtime-settings")["source"])
    runtime = runtime.replace(
        "from synthetic_counting_v24_3.config import preset_config as v24_3_preset_config",
        "from synthetic_counting_v28.config import preset_config as v28_preset_config",
    )
    runtime = runtime.replace(
        "baseline = v24_3_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)",
        "baseline = v28_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)",
    )
    runtime = runtime.replace(
        'assert changed_fields == {"version", "untie_atomic_count_readout"}, changed_fields',
        'assert changed_fields == {"version", "task_output_count_weight"}, changed_fields',
    )
    runtime = runtime.replace(
        "assert planned.untie_atomic_count_readout\n",
        "assert planned.untie_atomic_count_readout\n"
        "assert planned.task_output_count_weight == 4.0\n"
        "assert planned.task_output_trace_weight == 1.0\n"
        "assert planned.task_output_structure_weight == 0.1\n",
    )
    runtime = runtime.replace(
        "# Step-zero functional audit: partial untying changes parameter identity, not logits.",
        "# Step-zero audit: the scalar loss change cannot alter parameters or logits.",
    )
    runtime = runtime.replace(
        "extra_parameters = planned_model.parameter_count() - baseline_model.parameter_count()\n"
        "assert extra_parameters == 10 * planned.n_embd == 2560",
        "parameter_delta = planned_model.parameter_count() - baseline_model.parameter_count()\n"
        "assert parameter_delta == 0",
    )
    runtime = runtime.replace(
        '"controlled_changes_from_v24.3": sorted(changed_fields),',
        '"controlled_changes_from_v28": sorted(changed_fields),',
    )
    runtime = runtime.replace(
        '"loss": "unchanged v24.3 full-vocabulary component-normalized CE",',
        '"loss": "v28 component-normalized CE with count-region coefficient 4",',
    )
    runtime = runtime.replace(
        '"extra_parameters": extra_parameters,',
        '"parameter_delta_from_v28": parameter_delta,',
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
