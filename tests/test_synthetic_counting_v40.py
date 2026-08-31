from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from synthetic_counting_v35.config import preset_config as preset_v35
from synthetic_counting_v40.config import preset_config as preset_v40


ROOT = Path(__file__).resolve().parents[1]


def test_v40_changes_only_count_support_from_v35() -> None:
    baseline = preset_v35("main", device="cpu")
    candidate = preset_v40("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "count_max_threshold"}
    assert candidate.count_max_threshold == 5
    assert baseline.count_max_threshold == 10
    assert candidate.seq_len == baseline.seq_len == 256
    assert candidate.needle_pool_size == baseline.needle_pool_size == 100
    assert candidate.needle_pool_frequency_threshold == baseline.needle_pool_frequency_threshold
    assert candidate.effective_needle_pool_seed == baseline.effective_needle_pool_seed
    assert candidate.train_steps == baseline.train_steps == 6_000
    assert candidate.max_steps_for_language_pred == baseline.max_steps_for_language_pred == 1_500
    assert candidate.lr == baseline.lr == 3e-4
    assert candidate.training_count_distribution == "maxent_set_count"
    assert candidate.trace_format == "separator"
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )


def test_v40_rejects_noncanonical_count_support() -> None:
    candidate = preset_v40("main", device="cpu")
    with pytest.raises(ValueError, match="count_max_threshold=5"):
        replace(candidate, count_max_threshold=6).validate()


def test_v40_screen_notebook_is_clean_and_guarded() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v40_Count5_Screen_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v40"' in source
    assert "SEEDS = (1234,)" in source
    assert 'changed_fields == {"version", "count_max_threshold"}' in source
    assert "planned.count_max_threshold == 5" in source
    assert "list(range(1, 6))" in source
    assert '"chance": 0.20' in source
    assert "marker_sets_identical_to_v35" in source
    assert "v40_count1to5_equalcomponents_steps6000_independent_L256_pool100_seed" in source
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
