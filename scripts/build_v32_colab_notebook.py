from __future__ import annotations

import json
from pathlib import Path

from build_v31_colab_notebook import build as build_v31_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v32_MaxEnt_CountWeight8_Screen_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v31_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v31", "synthetic_counting_v32")
        source = source.replace("run_v31", "run_v32")
        source = source.replace('VERSION = "v31"', 'VERSION = "v32"')
        source = source.replace(
            'manifest["version"] == "v31"', 'manifest["version"] == "v32"'
        )
        source = source.replace(
            "v31_countweight8_independent_L256_pool100_seed",
            "v32_maxent_countweight8_independent_L256_pool100_seed",
        )
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v32 screen: independent models, maximum-entropy set/count sampling

v32 is the sampler-only control for v31. Non-thinking and Thinking remain two
independently trained models with identical initialization. The 256-character
prompt, counts 1--10, 100 marker sets, separator/no-index trace, four-layer /
four-head / width-256 architecture, partial count-only output untying,
component-normalized count coefficient 8, optimizer, 1,500-step language
phase, and 10,000-step schedule are unchanged.

The sole change is the accepted training distribution: v31's count-uniform
rejection sampler is replaced by the existing maximum-entropy distribution
over feasible `(marker set, count)` cells. The v31 audit found 0.547 bits of
set--count mutual information and 23.9% set-only Bayes count accuracy (10%
chance); the maximum-entropy target is 0.060 bits and 11.2%, while retaining
all 100 sets. There is no shared model, auxiliary loss, decoder, calibration,
test-time update, trace rewrite, or inference rule.

The fixed seed-1234 behavioral gate is unchanged: Thinking accuracy >= 0.90,
minimum per-count accuracy >= 0.80, trace exact >= 0.90, count spread <= 0.20,
and Thinking-minus-Non-thinking accuracy >= 0.10. A failed screen is retained
and rejected without checkpoint selection.
""",
    )

    runtime = "".join(_cell(notebook, "runtime-settings")["source"])
    runtime = runtime.replace(
        "from synthetic_counting_v29.config import preset_config as v29_preset_config",
        "from synthetic_counting_v31.config import preset_config as v31_preset_config",
    )
    runtime = runtime.replace(
        "baseline = v29_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)",
        "baseline = v31_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)",
    )
    runtime = runtime.replace(
        'assert changed_fields == {"version", "task_output_count_weight"}, changed_fields',
        'assert changed_fields == {"version", "training_count_distribution"}, changed_fields',
    )
    runtime = runtime.replace(
        'assert planned.training_count_distribution == "uniform"',
        'assert baseline.training_count_distribution == "uniform"\n'
        'assert planned.training_count_distribution == "maxent_set_count"',
    )
    runtime = runtime.replace(
        "assert baseline.task_output_count_weight == 4.0\n",
        "",
    )
    runtime = runtime.replace(
        '"controlled_changes_from_v29": sorted(changed_fields),',
        '"controlled_changes_from_v31": sorted(changed_fields),',
    )
    runtime = runtime.replace(
        '"loss": "v29 component-normalized CE with count-region coefficient 8",',
        '"loss": "unchanged v31 component-normalized CE; count coefficient 8",\n'
        '    "sampler": "maximum-entropy feasible set x count cells",',
    )
    runtime = runtime.replace(
        '"parameter_delta_from_v29": parameter_delta,',
        '"parameter_delta_from_v31": parameter_delta,',
    )
    _set_source(notebook, "runtime-settings", runtime)

    prepare = "".join(_cell(notebook, "prepare-data")["source"])
    prepare += """

# Pre-training sampler audit: compare the exact target distribution with the
# measured v31 shortcut.  This is metadata only and does not inspect test labels.
import numpy as np

from synthetic_counting_v20.pipeline import load_prepared_v20_data
from synthetic_counting_v20.training import _joint_set_count_sampler
audit_vocab = V20Vocab.build(planned, text)
audit_split, audit_pool, _, _ = load_prepared_v20_data(
    planned, audit_vocab, text, RUN_DIRS[SEEDS[0]]
)
joint = _joint_set_count_sampler(planned, text, audit_split, audit_pool)
pivot = joint.plan.pivot(
    index="set_id", columns="count", values="target_probability"
).fillna(0.0)
p = pivot.to_numpy(dtype=float)
ps = p.sum(axis=1, keepdims=True)
pc = p.sum(axis=0, keepdims=True)
mask = p > 0
target_mi_bits = float((p[mask] * np.log2((p / (ps @ pc))[mask])).sum())
target_set_only_bayes = float(p.max(axis=1).sum())
assert target_mi_bits < 0.07
assert target_set_only_bayes < 0.12
print({
    "maxent_target_set_count_MI_bits": target_mi_bits,
    "maxent_target_set_only_Bayes_accuracy": target_set_only_bayes,
    "chance": 0.10,
})
"""
    _set_source(notebook, "prepare-data", prepare)

    notebook["metadata"]["colab"]["name"] = TARGET.name
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
