from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v47.config import preset_config as preset_v47
from synthetic_counting_v48.behavior_gate import evaluate_behavior_gate
from synthetic_counting_v48.config import preset_config as preset_v48
from synthetic_counting_v48.preflight import run_preflight


def test_v48_changes_only_v47_parallel_capacity() -> None:
    baseline = preset_v47("main", device="cpu")
    candidate = preset_v48("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "n_head", "n_embd", "n_inner"}
    assert candidate.train_steps == 10_000
    assert candidate.permute_task_context_tokens is True
    assert candidate.count_max_threshold == 10
    assert candidate.trace_format == "separator"
    assert candidate.joint_sampler_max_starts_per_cell is None
    assert (candidate.n_layer, candidate.n_head, candidate.n_embd, candidate.n_inner) == (
        4,
        6,
        384,
        1536,
    )
    assert candidate.n_embd // candidate.n_head == 64
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )
    assert config_from_dict(candidate.to_dict()) == candidate


def test_v48_rejects_compact_width() -> None:
    with pytest.raises(ValueError, match="6 heads"):
        replace(
            preset_v48("main", device="cpu"),
            n_head=4,
            n_embd=256,
            n_inner=1024,
        ).validate()


def test_v48_preflight_is_exposed() -> None:
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


def test_v48_behavior_gate_uses_fixed_10000_step_endpoint(tmp_path: Path) -> None:
    _write_behavior_tables(tmp_path)
    result = evaluate_behavior_gate(tmp_path)
    assert result["passed"] is True
    assert result["version"] == "v48"
    assert "10000-step" in result["endpoint_policy"]
