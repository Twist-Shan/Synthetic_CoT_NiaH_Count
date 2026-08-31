from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from synthetic_counting_v40.config import preset_config as preset_v40
from synthetic_counting_v41.config import preset_config as preset_v41


ROOT = Path(__file__).resolve().parents[1]


def test_v41_changes_only_parallel_capacity_from_v40() -> None:
    baseline = preset_v40("main", device="cpu")
    candidate = preset_v41("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "n_head", "n_embd", "n_inner"}
    assert (baseline.n_layer, baseline.n_head, baseline.n_embd, baseline.n_inner) == (
        4,
        4,
        256,
        1024,
    )
    assert (candidate.n_layer, candidate.n_head, candidate.n_embd, candidate.n_inner) == (
        4,
        6,
        384,
        1536,
    )
    assert baseline.n_embd // baseline.n_head == candidate.n_embd // candidate.n_head == 64
    assert candidate.count_max_threshold == baseline.count_max_threshold == 5
    assert candidate.seq_len == baseline.seq_len == 256
    assert candidate.needle_pool_size == baseline.needle_pool_size == 100
    assert candidate.train_steps == baseline.train_steps == 6_000
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )


def test_v41_rejects_noncanonical_parallel_capacity() -> None:
    candidate = preset_v41("main", device="cpu")
    with pytest.raises(ValueError, match="6 heads"):
        replace(candidate, n_head=4).validate()


def test_v41_screen_notebook_is_clean_and_guarded() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v41_Width384_Screen_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v41"' in source
    assert "SEEDS = (1234,)" in source
    assert 'changed_fields == {"version", "n_head", "n_embd", "n_inner"}' in source
    assert "(planned.n_layer, planned.n_head, planned.n_embd, planned.n_inner) == (4, 6, 384, 1536)" in source
    assert '"--n-head", "6"' in source
    assert '"--n-embd", "384"' in source
    assert '"--n-inner", "1536"' in source
    assert "marker_sets_identical_to_v40" in source
    assert "v41_count1to5_width384_heads6_steps6000_independent_L256_pool100_seed" in source
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
