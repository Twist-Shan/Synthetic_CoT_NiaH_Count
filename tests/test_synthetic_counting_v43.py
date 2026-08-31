from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.training import _retain_joint_starts
from synthetic_counting_v42.config import preset_config as preset_v42
from synthetic_counting_v43.config import preset_config as preset_v43


ROOT = Path(__file__).resolve().parents[1]


def test_v43_changes_only_within_cell_sampler_support_from_v42() -> None:
    baseline = preset_v42("main", device="cpu")
    candidate = preset_v43("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "joint_sampler_max_starts_per_cell"}
    assert baseline.joint_sampler_max_starts_per_cell == 8_192
    assert candidate.joint_sampler_max_starts_per_cell is None
    assert candidate.train_steps == baseline.train_steps == 8_000
    assert candidate.phase_cloud_steps == baseline.phase_cloud_steps
    assert candidate.count_max_threshold == baseline.count_max_threshold == 5
    assert candidate.seq_len == baseline.seq_len == 256
    assert (candidate.n_layer, candidate.n_head, candidate.n_embd, candidate.n_inner) == (
        baseline.n_layer,
        baseline.n_head,
        baseline.n_embd,
        baseline.n_inner,
    ) == (4, 6, 384, 1536)
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )
    assert config_from_dict(candidate.to_dict()) == candidate

    legacy_v42 = baseline.to_dict()
    legacy_v42.pop("joint_sampler_max_starts_per_cell")
    legacy_v42.pop("joint_sampler_within_cell_policy")
    assert config_from_dict(legacy_v42) == baseline


def test_v43_rejects_historical_within_cell_cap() -> None:
    candidate = preset_v43("main", device="cpu")
    with pytest.raises(ValueError, match="joint_sampler_max_starts_per_cell=None"):
        replace(candidate, joint_sampler_max_starts_per_cell=8_192).validate()


def test_joint_start_retention_support_policies() -> None:
    starts = np.arange(10, dtype=np.int32)
    full = _retain_joint_starts(starts, None)
    capped = _retain_joint_starts(starts, 3)
    assert full is starts
    np.testing.assert_array_equal(full, starts)
    np.testing.assert_array_equal(capped, np.array([0, 4, 9], dtype=np.int32))


def test_v43_screen_notebook_is_clean_and_guarded() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v43_FullStarts_Screen_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v43"' in source
    assert "SEEDS = (1234,)" in source
    assert (
        'changed_fields == {"version", "joint_sampler_max_starts_per_cell"}'
        in source
    )
    assert "baseline.joint_sampler_max_starts_per_cell == 8192" in source
    assert "planned.joint_sampler_max_starts_per_cell is None" in source
    assert "full_window_count" in source
    assert "retained_window_count" in source
    assert "all_legal_starts" in source
    assert "v43_count1to5_width384_heads6_steps8000_fullstarts_independent_L256_pool100_seed" in source
    assert 'role_table["step"].eq(8000)' in source
    assert '"--stage", "phase,causal,extended,plots"' in source
    assert "if not behavior_gate:" in source
    assert "NCC skipped: final behavioral gate failed" in source
    assert "Mechanism analyses skipped: final behavioral gate failed" in source
    assert "shared_checkpoint" not in source
    assert "CALIBRATION_DIR" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
