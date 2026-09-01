from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from synthetic_counting_v34.config import preset_config as preset_v34
from synthetic_counting_v35.config import preset_config as preset_v35


ROOT = Path(__file__).resolve().parents[1]


def test_v35_changes_only_structure_weight_from_v34() -> None:
    baseline = preset_v34("main", device="cpu")
    candidate = preset_v35("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "task_output_structure_weight"}
    assert baseline.task_output_structure_weight == 0.1
    assert candidate.task_output_structure_weight == 8.0
    assert (
        candidate.task_output_count_weight
        == candidate.task_output_trace_weight
        == candidate.task_output_structure_weight
        == 8.0
    )
    assert candidate.task_output_scheduled_sampling_max_probability == 0.0
    assert candidate.train_steps == 6_000
    assert candidate.training_count_distribution == "maxent_set_count"
    assert candidate.trace_format == "separator"
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )


def test_v35_rejects_noncanonical_structure_weight() -> None:
    candidate = preset_v35("main", device="cpu")
    with pytest.raises(ValueError, match="task_output_structure_weight=8"):
        replace(candidate, task_output_structure_weight=0.1).validate()


def test_v35_screen_notebook_is_clean_and_predeclares_endpoint() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v35_EqualComponents_Screen_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v35"' in source
    assert "SEEDS = (1234,)" in source
    assert 'changed_fields == {"version", "task_output_structure_weight"}' in source
    assert "baseline.task_output_structure_weight == 0.1" in source
    assert "planned.task_output_structure_weight == 8.0" in source
    assert "planned.task_output_scheduled_sampling_max_probability == 0.0" in source
    assert '"--train-steps", "6000"' in source
    assert 'role_table["step"].eq(6000)' in source
    assert "v35_equalcomponents8_steps6000_independent_L256_pool100_seed" in source
    assert '"--stage", "phase,causal,extended,plots"' in source
    assert "shared_checkpoint" not in source
    assert "CALIBRATION_DIR" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
