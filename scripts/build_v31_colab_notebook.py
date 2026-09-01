from __future__ import annotations

import json
from pathlib import Path

from build_v29_colab_notebook import build as build_v29_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v31_CountWeight8_Screen_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v29_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v29", "synthetic_counting_v31")
        source = source.replace("run_v29", "run_v31")
        source = source.replace('VERSION = "v29"', 'VERSION = "v31"')
        source = source.replace(
            'manifest["version"] == "v29"', 'manifest["version"] == "v31"'
        )
        source = source.replace(
            "v29_countweight4_fixed_partial_readout_L256_pool100_seed",
            "v31_countweight8_independent_L256_pool100_seed",
        )
        source = source.replace("SEEDS = (1234, 2234, 3234)", "SEEDS = (1234,)")
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v31 screen: independent models, count coefficient 8

This is the first point of a pre-registered two-value scalar screen. It keeps
v29's two independent models, identical initialization, 256-character prompt,
counts 1--10, 100 marker sets, uniform semantic-count sampler, four-layer /
four-head / width-256 architecture, separator/no-index trace, partial
count-only output untying, optimizer, 1,500-step language phase, and
10,000-step schedule.

The sole change from v29 is that the already component-normalized final-count
coefficient increases from 4 to 8. Trace and structure stay at 1 and 0.1. There
is no shared model, auxiliary objective, decoder, calibration/frozen stage,
test-time update, extra layer, trace rewrite, or inference rule. Therefore the
two head banks and their training dynamics remain independently interpretable.

Screening seed 1234 is evaluated first. Its fixed behavioral gate is Thinking
accuracy >= 0.90, minimum per-count accuracy >= 0.80, trace exact >= 0.90,
count spread <= 0.20, and Thinking-minus-Non-thinking accuracy >= 0.10. A failed
screen is retained and rejected; it is not rescued by checkpoint selection.
""",
    )

    runtime = "".join(_cell(notebook, "runtime-settings")["source"])
    runtime = runtime.replace(
        "from synthetic_counting_v28.config import preset_config as v28_preset_config",
        "from synthetic_counting_v29.config import preset_config as v29_preset_config",
    )
    runtime = runtime.replace(
        "baseline = v28_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)",
        "baseline = v29_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)",
    )
    runtime = runtime.replace(
        "assert planned.task_output_count_weight == 4.0",
        "assert baseline.task_output_count_weight == 4.0\n"
        "assert planned.task_output_count_weight == 8.0",
    )
    runtime = runtime.replace(
        '"controlled_changes_from_v28": sorted(changed_fields),',
        '"controlled_changes_from_v29": sorted(changed_fields),',
    )
    runtime = runtime.replace(
        '"loss": "v28 component-normalized CE with count-region coefficient 4",',
        '"loss": "v29 component-normalized CE with count-region coefficient 8",',
    )
    runtime = runtime.replace(
        '"parameter_delta_from_v28": parameter_delta,',
        '"parameter_delta_from_v29": parameter_delta,',
    )
    _set_source(notebook, "runtime-settings", runtime)

    replacements = {
        "settings-heading": "## 3. Audit the one-scalar screening setting\n",
        "prepare-heading": "## 4. Prepare and audit the screening dataset\n",
        "train-heading": "## 5. Train the two independent models end-to-end\n",
        "ncc-heading": "## 6. Apply the fixed behavioral gate before NCC/mechanism analysis\n",
        "results-heading": "## 7. NCC only if the behavioral screen is retained\n",
        "mechanism-heading": "## 8. Training dynamics and retrieval roles only for a retained screen\n",
        "finish-heading": "## 9. Verify Drive persistence for the complete screening run\n",
    }
    for cell_id, source in replacements.items():
        _set_source(notebook, cell_id, source)

    notebook["metadata"]["colab"]["name"] = TARGET.name
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
