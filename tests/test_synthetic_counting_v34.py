from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from synthetic_counting_v32.config import preset_config as preset_v32
from synthetic_counting_v34.config import preset_config as preset_v34


ROOT = Path(__file__).resolve().parents[1]


def test_v34_changes_only_declared_trace_weight_and_budget_fields_from_v32() -> None:
    baseline = preset_v32("main", device="cpu")
    candidate = preset_v34("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {
        "version",
        "train_steps",
        "phase_cloud_steps",
        "task_output_trace_weight",
    }
    assert baseline.task_output_trace_weight == 1.0
    assert candidate.task_output_trace_weight == 8.0
    assert candidate.task_output_count_weight == 8.0
    assert candidate.task_output_scheduled_sampling_max_probability == 0.0
    assert candidate.train_steps == 6_000
    assert candidate.training_count_distribution == "maxent_set_count"
    assert candidate.trace_format == "separator"
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )


def test_v34_rejects_noncanonical_trace_weight_or_budget() -> None:
    candidate = preset_v34("main", device="cpu")
    with pytest.raises(ValueError, match="task_output_trace_weight=8"):
        replace(candidate, task_output_trace_weight=1.0).validate()
    with pytest.raises(ValueError, match="requires train_steps=6000"):
        replace(candidate, train_steps=10_000).validate()


def test_v34_screen_notebook_is_clean_and_predeclares_endpoint() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v34_TraceWeight8_Screen_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v34"' in source
    assert "SEEDS = (1234,)" in source
    assert '"task_output_trace_weight"' in source
    assert "baseline.task_output_trace_weight == 1.0" in source
    assert "planned.task_output_trace_weight == 8.0" in source
    assert "planned.task_output_scheduled_sampling_max_probability == 0.0" in source
    assert '"--train-steps", "6000"' in source
    assert 'role_table["step"].eq(6000)' in source
    assert "v34_traceweight8_steps6000_independent_L256_pool100_seed" in source
    assert '"--stage", "phase,causal,extended,plots"' in source
    assert "shared_checkpoint" not in source
    assert "CALIBRATION_DIR" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
