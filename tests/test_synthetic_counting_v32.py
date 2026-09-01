from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from synthetic_counting_v31.config import preset_config as preset_v31
from synthetic_counting_v32.config import preset_config as preset_v32


ROOT = Path(__file__).resolve().parents[1]


def test_v32_changes_only_version_and_sampler_from_v31() -> None:
    baseline = preset_v31("main", device="cpu")
    candidate = preset_v32("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "training_count_distribution"}
    assert baseline.training_count_distribution == "uniform"
    assert candidate.training_count_distribution == "maxent_set_count"
    assert candidate.task_output_count_weight == 8.0
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )
    assert candidate.trace_format == "separator"
    assert candidate.n_layer == 4
    assert candidate.n_head == 4


def test_v32_rejects_a_noncanonical_sampler() -> None:
    candidate = preset_v32("main", device="cpu")
    with pytest.raises(ValueError, match="training_count_distribution='maxent_set_count'"):
        replace(candidate, training_count_distribution="uniform").validate()


def test_v32_screen_notebook_is_clean_and_audits_sampler_only_change() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v32_MaxEnt_CountWeight8_Screen_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v32"' in source
    assert "SEEDS = (1234,)" in source
    assert 'changed_fields == {"version", "training_count_distribution"}' in source
    assert 'planned.training_count_distribution == "maxent_set_count"' in source
    assert "planned.task_output_count_weight == 8.0" in source
    assert "v32_maxent_countweight8_independent_L256_pool100_seed" in source
    assert '"--stage", "phase,causal,extended,plots"' in source
    assert "shared_checkpoint" not in source
    assert "CALIBRATION_DIR" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
