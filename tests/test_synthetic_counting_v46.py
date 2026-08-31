from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
import random

import pandas as pd
import pytest

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import permute_task_window
from synthetic_counting_v35.config import preset_config as preset_v35
from synthetic_counting_v46.behavior_gate import evaluate_behavior_gate
from synthetic_counting_v46.config import preset_config as preset_v46
from synthetic_counting_v46.preflight import run_preflight


def test_v46_is_v35_capacity_with_full_support_and_context_permutation() -> None:
    baseline = preset_v35("main", device="cpu")
    candidate = preset_v46("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {
        "version",
        "joint_sampler_max_starts_per_cell",
        "permute_task_context_tokens",
    }
    assert candidate.joint_sampler_max_starts_per_cell is None
    assert candidate.permute_task_context_tokens is True
    assert candidate.count_max_threshold == 10
    assert candidate.trace_format == "separator"
    assert candidate.train_steps == 6_000
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


def test_v46_rejects_natural_task_context_order() -> None:
    with pytest.raises(ValueError, match="permute_task_context_tokens=True"):
        replace(
            preset_v46("main", device="cpu"),
            permute_task_context_tokens=False,
        ).validate()


def test_task_window_permutation_is_reproducible_and_count_preserving() -> None:
    window = "aaaabbbccddefghijklmnopqrstuvwxyz"
    first = permute_task_window(window, random.Random(123))
    second = permute_task_window(window, random.Random(123))
    assert first == second
    assert first != window
    assert Counter(first) == Counter(window)
    assert first.count("a") == window.count("a")
    assert first.count("b") == window.count("b")


def test_v46_preflight_is_exposed() -> None:
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


def test_v46_behavior_gate_uses_fixed_6000_step_endpoint(tmp_path: Path) -> None:
    _write_behavior_tables(tmp_path)
    result = evaluate_behavior_gate(tmp_path)
    assert result["passed"] is True
    assert result["version"] == "v46"
    assert "6000-step" in result["endpoint_policy"]
