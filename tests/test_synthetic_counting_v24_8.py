from __future__ import annotations

import pandas as pd

from synthetic_counting_v24_8.readout_tail import (
    CandidateSpec,
    default_candidate_specs,
    summarize_gate,
)


def _frame(accuracies: dict[int, float], *, trace_exact: float = 1.0) -> pd.DataFrame:
    rows = []
    for count, accuracy in accuracies.items():
        for index in range(20):
            correct = float(index < round(20 * accuracy))
            rows.append(
                {
                    "count": count,
                    "ar_accuracy": correct,
                    "ar_answered": 1.0,
                    "trace_exact": trace_exact,
                }
            )
    return pd.DataFrame(rows)


def test_gate_requires_uniform_per_count_success() -> None:
    summary, by_count = summarize_gate(
        _frame({count: 0.95 for count in range(1, 11)}), mode="thinking"
    )
    assert summary.success_criteria_met
    assert summary.overall_accuracy == 0.95
    assert summary.minimum_count_accuracy == 0.95
    assert summary.count_accuracy_spread == 0.0
    assert len(by_count) == 10


def test_gate_rejects_high_overall_with_dead_count() -> None:
    accuracies = {count: 1.0 for count in range(1, 11)}
    accuracies[7] = 0.0
    summary, _ = summarize_gate(_frame(accuracies), mode="thinking")
    assert summary.overall_accuracy == 0.9
    assert summary.minimum_count_accuracy == 0.0
    assert summary.count_accuracy_spread == 1.0
    assert not summary.success_criteria_met


def test_candidate_specs_are_valid_and_ordered_conservatively() -> None:
    specs = default_candidate_specs()
    assert [spec.learning_rate for spec in specs] == [3e-4, 1e-3, 3e-3]
    for spec in specs:
        spec.validate()
    CandidateSpec("custom", 1e-4, 10, 1).validate()
