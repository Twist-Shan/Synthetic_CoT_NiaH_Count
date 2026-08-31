from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from synthetic_counting_v20.training import training_loss_phase
from synthetic_counting_v35.config import preset_config as preset_v35
from synthetic_counting_v39.config import preset_config as preset_v39


ROOT = Path(__file__).resolve().parents[1]


def test_v39_changes_only_loss_schedule_boundary_from_v35() -> None:
    baseline = preset_v35("main", device="cpu")
    candidate = preset_v39("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "max_steps_for_language_pred"}
    assert candidate.max_steps_for_language_pred == 0
    assert baseline.max_steps_for_language_pred == 1_500
    assert candidate.train_steps == baseline.train_steps == 6_000
    assert candidate.lr == baseline.lr == 3e-4
    assert candidate.task_output_count_weight == baseline.task_output_count_weight == 8.0
    assert candidate.task_output_trace_weight == baseline.task_output_trace_weight == 8.0
    assert candidate.task_output_structure_weight == baseline.task_output_structure_weight == 8.0
    assert candidate.training_count_distribution == "maxent_set_count"
    assert candidate.trace_format == "separator"
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )


def test_v39_is_task_output_only_from_first_update() -> None:
    candidate = preset_v39("main", device="cpu")
    assert training_loss_phase(candidate, 1) == "task_output"
    assert training_loss_phase(candidate, candidate.train_steps) == "task_output"


def test_v39_preset_forces_task_only_boundary() -> None:
    candidate = preset_v39("main", device="cpu", max_steps_for_language_pred=500)
    assert candidate.max_steps_for_language_pred == 0
    with pytest.raises(ValueError, match="max_steps_for_language_pred"):
        replace(candidate, max_steps_for_language_pred=-1).validate()


def test_v39_screen_notebook_is_clean_and_guarded() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v39_TaskOnly_Screen_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v39"' in source
    assert "SEEDS = (1234,)" in source
    assert 'changed_fields == {"version", "max_steps_for_language_pred"}' in source
    assert "planned.max_steps_for_language_pred == 0" in source
    assert '"--max-steps-for-language-pred", "0"' in source
    assert "v39_taskonly_equalcomponents_steps6000_independent_L256_pool100_seed" in source
    assert 'role_table["step"].eq(6000)' in source
    assert '"--stage", "phase,causal,extended,plots"' in source
    assert "if not behavior_gate:" in source
    assert "NCC skipped: final behavioral gate failed" in source
    assert "Mechanism analyses skipped: final behavioral gate failed" in source
    assert "shared_checkpoint" not in source
    assert "CALIBRATION_DIR" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
