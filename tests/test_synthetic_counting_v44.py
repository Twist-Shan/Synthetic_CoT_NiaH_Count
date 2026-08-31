from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import V20Vocab, load_corpus_text
from synthetic_counting_v20.model import build_model
from synthetic_counting_v43.config import preset_config as preset_v43
from synthetic_counting_v44.behavior_gate import evaluate_behavior_gate
from synthetic_counting_v44.config import preset_config as preset_v44
from synthetic_counting_v44.preflight import run_preflight


def test_v44_changes_only_count_support_from_v43() -> None:
    baseline = preset_v43("main", device="cpu")
    candidate = preset_v44("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "count_max_threshold"}
    assert baseline.count_max_threshold == 5
    assert candidate.count_max_threshold == 10
    assert candidate.joint_sampler_max_starts_per_cell is None
    assert candidate.train_steps == baseline.train_steps == 8_000
    assert candidate.phase_cloud_steps == baseline.phase_cloud_steps
    assert candidate.seq_len == baseline.seq_len == 256
    assert candidate.max_render_len == 287
    assert candidate.n_positions == baseline.n_positions == 384
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

    text = load_corpus_text()
    baseline_vocab = V20Vocab.build(baseline, text)
    candidate_vocab = V20Vocab.build(candidate, text)
    assert len(candidate_vocab.id_to_token) - len(baseline_vocab.id_to_token) == 5
    assert build_model(baseline, baseline_vocab, device="cpu").parameter_count() == 7_130_496
    assert build_model(candidate, candidate_vocab, device="cpu").parameter_count() == 7_134_336


def test_v44_rejects_short_count_support() -> None:
    with pytest.raises(ValueError, match="count_max_threshold=10"):
        replace(preset_v44("main", device="cpu"), count_max_threshold=5).validate()


def test_v44_preflight_is_exposed() -> None:
    assert callable(run_preflight)


def _write_behavior_tables(
    root: Path,
    *,
    thinking: list[float],
    nonthinking: list[float],
    trace_exact: float,
) -> None:
    tables = root / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "mode": "nonthinking",
                "ar_final_accuracy": sum(nonthinking) / 10,
                "trace_exact": float("nan"),
            },
            {
                "mode": "thinking",
                "ar_final_accuracy": sum(thinking) / 10,
                "trace_exact": trace_exact,
            },
        ]
    ).to_csv(tables / "final_autoregressive_summary.csv", index=False)
    pd.DataFrame(
        [
            {"mode": mode, "count": count, "ar_final_accuracy": value}
            for mode, values in (("nonthinking", nonthinking), ("thinking", thinking))
            for count, value in enumerate(values, start=1)
        ]
    ).to_csv(tables / "final_autoregressive_by_count.csv", index=False)


def test_v44_behavior_gate_passes_only_a_balanced_thinking_advantage(tmp_path: Path) -> None:
    _write_behavior_tables(
        tmp_path,
        thinking=[0.90] * 10,
        nonthinking=[0.70] * 10,
        trace_exact=0.90,
    )
    result = evaluate_behavior_gate(tmp_path)
    assert result["passed"] is True
    assert result["metrics"]["thinking_minus_nonthinking_gap"] == pytest.approx(0.20)
    assert result["metrics"]["thinking_count_spread"] == 0.0


def test_v44_behavior_gate_rejects_one_collapsed_count(tmp_path: Path) -> None:
    _write_behavior_tables(
        tmp_path,
        thinking=[1.0] * 9 + [0.20],
        nonthinking=[0.60] * 10,
        trace_exact=0.95,
    )
    result = evaluate_behavior_gate(tmp_path)
    assert result["passed"] is False
    assert result["checks"]["thinking_min_count_accuracy"] is False
    assert result["checks"]["thinking_count_spread"] is False
