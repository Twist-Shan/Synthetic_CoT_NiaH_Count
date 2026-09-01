from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from synthetic_counting_v20.training import scheduled_sampling_probability
from synthetic_counting_v35.config import preset_config as preset_v35
from synthetic_counting_v38.config import preset_config as preset_v38


ROOT = Path(__file__).resolve().parents[1]


def test_v38_changes_only_mild_rollin_scalar_from_v35() -> None:
    baseline = preset_v35("main", device="cpu")
    candidate = preset_v38("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {
        "version",
        "task_output_scheduled_sampling_max_probability",
    }
    assert candidate.train_steps == baseline.train_steps == 6_000
    assert candidate.lr == baseline.lr == 3e-4
    assert candidate.min_lr == baseline.min_lr == 0.0
    assert candidate.task_output_count_weight == baseline.task_output_count_weight == 8.0
    assert candidate.task_output_trace_weight == baseline.task_output_trace_weight == 8.0
    assert candidate.task_output_structure_weight == baseline.task_output_structure_weight == 8.0
    assert candidate.training_count_distribution == "maxent_set_count"
    assert candidate.trace_format == "separator"
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )


def test_v38_rollin_is_mild_linear_and_thinking_only() -> None:
    candidate = preset_v38("main", device="cpu")
    assert scheduled_sampling_probability(candidate, 1_500, "thinking") == 0.0
    assert scheduled_sampling_probability(candidate, 3_750, "thinking") == pytest.approx(0.05)
    assert scheduled_sampling_probability(candidate, 6_000, "thinking") == pytest.approx(0.1)
    assert scheduled_sampling_probability(candidate, 6_000, "nonthinking") == 0.0


def test_v38_rejects_noncanonical_rollin() -> None:
    candidate = preset_v38("main", device="cpu")
    with pytest.raises(
        ValueError,
        match="task_output_scheduled_sampling_max_probability=0.1",
    ):
        replace(
            candidate,
            task_output_scheduled_sampling_max_probability=0.2,
        ).validate()


def test_v38_screen_notebook_is_clean_and_guarded() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v38_Mild_Rollin_Screen_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v38"' in source
    assert "SEEDS = (1234,)" in source
    assert '"task_output_scheduled_sampling_max_probability"' in source
    assert "planned.task_output_scheduled_sampling_max_probability == 0.1" in source
    assert '"--task-output-scheduled-sampling-max-probability", "0.1"' in source
    assert "v38_rollin0p1_equalcomponents_steps6000_independent_L256_pool100_seed" in source
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
