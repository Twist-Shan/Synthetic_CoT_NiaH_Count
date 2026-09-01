from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from synthetic_counting_v20.training import learning_rate
from synthetic_counting_v32.config import preset_config as preset_v32
from synthetic_counting_v35.config import preset_config as preset_v35
from synthetic_counting_v36.config import preset_config as preset_v36


ROOT = Path(__file__).resolve().parents[1]


def test_v36_changes_only_decay_horizon_from_v35() -> None:
    baseline = preset_v35("main", device="cpu")
    candidate = preset_v36("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "lr_decay_steps"}
    assert baseline.lr_decay_steps is None
    assert candidate.lr_decay_steps == 10_000
    assert candidate.train_steps == baseline.train_steps == 6_000
    assert candidate.phase_cloud_steps == baseline.phase_cloud_steps
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


def test_v36_matches_v32_lr_trajectory_through_screen_endpoint() -> None:
    v32 = preset_v32("main", device="cpu")
    v35 = preset_v35("main", device="cpu")
    candidate = preset_v36("main", device="cpu")
    for step in (1, 500, 1_500, 4_000, 6_000):
        assert learning_rate(candidate, step) == pytest.approx(
            learning_rate(v32, step), abs=1e-15
        )
    assert learning_rate(v35, 6_000) == 0.0
    assert learning_rate(candidate, 6_000) > 0.0


def test_v36_rejects_noncanonical_or_short_decay_horizon() -> None:
    candidate = preset_v36("main", device="cpu")
    with pytest.raises(ValueError, match="lr_decay_steps=10000"):
        replace(candidate, lr_decay_steps=9_000).validate()
    baseline = preset_v35("main", device="cpu")
    with pytest.raises(ValueError, match="requires positive min_lr"):
        replace(baseline, lr_decay_steps=5_999).validate()


def test_v36_screen_notebook_is_clean_and_predeclares_endpoint() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v36_LR10k_Screen_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v36"' in source
    assert "SEEDS = (1234,)" in source
    assert 'changed_fields == {"version", "lr_decay_steps"}' in source
    assert "baseline.lr_decay_steps is None" in source
    assert "planned.lr_decay_steps == 10000" in source
    assert '"--train-steps", "6000"' in source
    assert '"--lr-decay-steps", "10000"' in source
    assert 'role_table["step"].eq(6000)' in source
    assert "v36_lr10k_steps6000_independent_L256_pool100_seed" in source
    assert '"--stage", "phase,causal,extended,plots"' in source
    assert "shared_checkpoint" not in source
    assert "CALIBRATION_DIR" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
