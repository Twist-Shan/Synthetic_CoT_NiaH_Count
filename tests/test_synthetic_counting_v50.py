from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import V20Example, V20Vocab, character_token, render_v20
from synthetic_counting_v49.config import preset_config as preset_v49
from synthetic_counting_v50.behavior_gate import evaluate_behavior_gate
from synthetic_counting_v50.config import preset_config as preset_v50
from synthetic_counting_v50.preflight import run_preflight


def _example() -> V20Example:
    marker = character_token("a")
    return V20Example(
        example_kind="counting_task",
        seq_tokens=[marker, marker, marker],
        corpus_region="train",
        corpus_start=0,
        corpus_end=3,
        prompt_sha256="count-3",
        set_id="set_000",
        needle_characters=("a", "b", "c"),
        rendered_set_order=("a", "b", "c"),
        needle_positions=(0, 1, 2),
        needle_markers=(marker, marker, marker),
        count=3,
        per_character_counts=(3, 0, 0),
    )


def test_v50_changes_only_v49_horizon_and_snapshot_schedule() -> None:
    baseline = preset_v49("main", device="cpu")
    candidate = preset_v50("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "train_steps", "phase_cloud_steps"}
    assert candidate.train_steps == 20_000
    assert candidate.lr_decay_steps is None
    assert candidate.min_lr == 0.0
    assert candidate.max_steps_for_language_pred == 1_500
    assert config_from_dict(candidate.to_dict()) == candidate


def test_v50_trace_serialization_is_identical_to_v49() -> None:
    old = preset_v49("debug", device="cpu")
    new = preset_v50("debug", device="cpu")
    old_vocab = V20Vocab.build(old, "abc xyz\n")
    new_vocab = V20Vocab.build(new, "abc xyz\n")
    assert render_v20(_example(), old_vocab, "thinking").tokens == render_v20(
        _example(), new_vocab, "thinking"
    ).tokens


def test_v50_preflight_is_exposed() -> None:
    assert callable(run_preflight)


def test_v50_behavior_gate_uses_fixed_20000_step_endpoint(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
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
    result = evaluate_behavior_gate(tmp_path)
    assert result["passed"] is True
    assert result["version"] == "v50"
    assert "20000-step" in result["endpoint_policy"]
