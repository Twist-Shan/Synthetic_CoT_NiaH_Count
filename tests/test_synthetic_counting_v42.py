from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from synthetic_counting_v20.training import learning_rate
from synthetic_counting_v41.config import preset_config as preset_v41
from synthetic_counting_v42.config import preset_config as preset_v42


ROOT = Path(__file__).resolve().parents[1]


def test_v42_changes_only_fresh_training_horizon_from_v41() -> None:
    baseline = preset_v41("main", device="cpu")
    candidate = preset_v42("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "train_steps", "phase_cloud_steps"}
    assert baseline.train_steps == 6_000
    assert candidate.train_steps == 8_000
    assert learning_rate(baseline, 6_000) == 0.0
    assert learning_rate(candidate, 6_000) > 0.0
    assert learning_rate(candidate, 8_000) == 0.0
    assert (candidate.n_layer, candidate.n_head, candidate.n_embd, candidate.n_inner) == (
        baseline.n_layer,
        baseline.n_head,
        baseline.n_embd,
        baseline.n_inner,
    ) == (4, 6, 384, 1536)
    assert candidate.count_max_threshold == baseline.count_max_threshold == 5
    assert candidate.seq_len == baseline.seq_len == 256
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )


def test_v42_rejects_noncanonical_train_steps() -> None:
    candidate = preset_v42("main", device="cpu")
    with pytest.raises(ValueError, match="train_steps=8000"):
        replace(candidate, train_steps=6_000).validate()


def test_v42_screen_notebook_is_clean_and_guarded() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v42_Width384_Steps8000_Screen_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v42"' in source
    assert "SEEDS = (1234,)" in source
    assert 'changed_fields == {"version", "train_steps", "phase_cloud_steps"}' in source
    assert "baseline.train_steps == 6000" in source
    assert "planned.train_steps == 8000" in source
    assert '"--train-steps", "8000"' in source
    assert "learning_rate(planned, 6000) > 0.0" in source
    assert "v42_count1to5_width384_heads6_steps8000_independent_L256_pool100_seed" in source
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
