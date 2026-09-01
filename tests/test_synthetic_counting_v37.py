from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from synthetic_counting_v20.training import learning_rate
from synthetic_counting_v35.config import preset_config as preset_v35
from synthetic_counting_v37.config import preset_config as preset_v37


ROOT = Path(__file__).resolve().parents[1]


def test_v37_changes_only_predeclared_schedule_from_v35() -> None:
    baseline = preset_v35("main", device="cpu")
    candidate = preset_v37("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {
        "version",
        "train_steps",
        "lr_decay_steps",
        "min_lr",
        "phase_cloud_steps",
    }
    assert baseline.train_steps == 6_000
    assert baseline.lr_decay_steps is None
    assert baseline.min_lr == 0.0
    assert candidate.train_steps == 8_000
    assert candidate.lr_decay_steps == 6_000
    assert candidate.min_lr == 1e-5
    assert candidate.phase_cloud_steps[-1] == 8_000
    assert candidate.task_output_count_weight == 8.0
    assert candidate.task_output_trace_weight == 8.0
    assert candidate.task_output_structure_weight == 8.0
    assert candidate.task_output_scheduled_sampling_max_probability == 0.0
    assert candidate.training_count_distribution == "maxent_set_count"
    assert candidate.trace_format == "separator"
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )


def test_v37_uses_conservative_cosine_minimum_and_constant_tail() -> None:
    baseline = preset_v35("main", device="cpu")
    candidate = preset_v37("main", device="cpu")
    for step in (1, 500, 1_500, 3_000, 4_000):
        assert abs(learning_rate(candidate, step) - learning_rate(baseline, step)) < 1e-5
    assert learning_rate(baseline, 6_000) == 0.0
    assert learning_rate(candidate, 6_000) == pytest.approx(1e-5)
    assert learning_rate(candidate, 7_000) == pytest.approx(1e-5)
    assert learning_rate(candidate, 8_000) == pytest.approx(1e-5)


def test_v37_rejects_noncanonical_schedule() -> None:
    candidate = preset_v37("main", device="cpu")
    with pytest.raises(ValueError, match="min_lr=1e-05"):
        replace(candidate, min_lr=2e-5).validate()
    with pytest.raises(ValueError, match="lr_decay_steps=6000"):
        replace(candidate, lr_decay_steps=7_000).validate()


def test_v37_screen_notebook_is_clean_and_predeclares_endpoint() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v37_LowLRTail_Screen_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v37"' in source
    assert "SEEDS = (1234,)" in source
    assert '"train_steps", "lr_decay_steps", "min_lr", "phase_cloud_steps"' in source
    assert "planned.train_steps == 8000" in source
    assert "planned.lr_decay_steps == 6000" in source
    assert "planned.min_lr == 1e-5" in source
    assert '"--train-steps", "8000"' in source
    assert '"--lr-decay-steps", "6000"' in source
    assert '"--min-lr", "1e-5"' in source
    assert 'role_table["step"].eq(8000)' in source
    assert "v37_lowtail1em5_steps8000_independent_L256_pool100_seed" in source
    assert '"--stage", "phase,causal,extended,plots"' in source
    assert "if not behavior_gate:" in source
    assert "NCC skipped: final behavioral gate failed" in source
    assert "Mechanism analyses skipped: final behavioral gate failed" in source
    assert "shared_checkpoint" not in source
    assert "CALIBRATION_DIR" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
