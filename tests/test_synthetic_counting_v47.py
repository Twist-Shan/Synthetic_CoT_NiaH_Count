from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v46.config import preset_config as preset_v46
from synthetic_counting_v47.behavior_gate import evaluate_behavior_gate
from synthetic_counting_v47.config import preset_config as preset_v47
from synthetic_counting_v47.preflight import run_preflight


def test_v47_changes_only_v46_horizon_and_phase_milestones() -> None:
    baseline = preset_v46("main", device="cpu")
    candidate = preset_v47("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "train_steps", "phase_cloud_steps"}
    assert candidate.train_steps == 10_000
    assert candidate.lr_decay_steps is None
    assert candidate.permute_task_context_tokens is True
    assert candidate.count_max_threshold == 10
    assert candidate.trace_format == "separator"
    assert candidate.joint_sampler_max_starts_per_cell is None
    assert (candidate.n_layer, candidate.n_head, candidate.n_embd, candidate.n_inner) == (
        4,
        4,
        256,
        1024,
    )
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )
    assert config_from_dict(candidate.to_dict()) == candidate


def test_v47_rejects_wrong_endpoint() -> None:
    with pytest.raises(ValueError, match="train_steps=10000"):
        replace(preset_v47("main", device="cpu"), train_steps=6_000).validate()


def test_v47_preflight_is_exposed() -> None:
    assert callable(run_preflight)


def _write_behavior_tables(root: Path) -> None:
    tables = root / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        [
            {"mode": "nonthinking", "ar_final_accuracy": 0.70, "trace_exact": float("nan")},
            {"mode": "thinking", "ar_final_accuracy": 0.92, "trace_exact": 0.91},
        ]
    ).to_csv(tables / "final_autoregressive_summary.csv", index=False)
    pd.DataFrame(
        [
            {"mode": mode, "count": count, "ar_final_accuracy": accuracy}
            for mode, accuracy in (("nonthinking", 0.70), ("thinking", 0.92))
            for count in range(1, 11)
        ]
    ).to_csv(tables / "final_autoregressive_by_count.csv", index=False)


def test_v47_behavior_gate_uses_fixed_10000_step_endpoint(tmp_path: Path) -> None:
    _write_behavior_tables(tmp_path)
    result = evaluate_behavior_gate(tmp_path)
    assert result["passed"] is True
    assert result["version"] == "v47"
    assert "10000-step" in result["endpoint_policy"]
