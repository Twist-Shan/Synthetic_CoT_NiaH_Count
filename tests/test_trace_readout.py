from __future__ import annotations

import pandas as pd

from synthetic_counting_v20.trace_readout import (
    trace_consistent_predictions,
    write_trace_readout_analysis,
)


def test_trace_readout_uses_only_closed_well_formed_generated_trace() -> None:
    frame = pd.DataFrame(
        [
            {
                "position_encoding": "rope",
                "mode": "thinking",
                "step": 10,
                "count": 5,
                "ar_accuracy": 0.0,
                "trace_exact": 1.0,
                "trace_marker_count_accuracy": 1.0,
                "trace_generated_marker_count": 5,
                "trace_format_valid": 1.0,
                "trace_closed": 1.0,
            },
            {
                "position_encoding": "rope",
                "mode": "thinking",
                "step": 10,
                "count": 7,
                "ar_accuracy": 1.0,
                "trace_exact": 0.0,
                "trace_marker_count_accuracy": 1.0,
                "trace_generated_marker_count": 7,
                "trace_format_valid": 0.0,
                "trace_closed": 1.0,
            },
        ]
    )
    output = trace_consistent_predictions(frame, count_max_threshold=10)
    assert output.loc[0, "trace_readout_pred_count"] == 5
    assert output.loc[0, "trace_readout_accuracy"] == 1.0
    assert pd.isna(output.loc[1, "trace_readout_pred_count"])
    assert output.loc[1, "trace_readout_accuracy"] == 0.0
    assert output.loc[1, "trace_readout_answered"] == 0.0


def test_trace_readout_writes_auditable_outputs(tmp_path) -> None:
    rows = []
    for count in range(1, 11):
        for example in range(20):
            correct = not (count == 10 and example == 0)
            rows.append(
                {
                    "position_encoding": "rope",
                    "mode": "thinking",
                    "step": 10_000,
                    "count": count,
                    "ar_accuracy": float(count % 2 == 0),
                    "trace_exact": float(correct),
                    "trace_marker_count_accuracy": float(correct),
                    "trace_generated_marker_count": count if correct else count - 1,
                    "trace_format_valid": 1.0,
                    "trace_closed": 1.0,
                }
            )
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame(rows).to_csv(tables / "final_autoregressive_detail.csv", index=False)

    manifest = write_trace_readout_analysis(tmp_path, count_max_threshold=10)

    assert manifest is not None
    summary = pd.read_csv(tables / "trace_readout_summary.csv").iloc[0]
    by_count = pd.read_csv(tables / "trace_readout_by_count.csv")
    assert summary["trace_readout_accuracy"] == 199 / 200
    assert summary["minimum_count_accuracy"] == 0.95
    assert summary["count_accuracy_spread"] == 0.05
    assert bool(summary["success_criteria_met"])
    assert by_count.loc[by_count["count"].eq(10), "trace_readout_accuracy"].item() == 0.95
    assert (tmp_path / "figures" / "trace_readout_by_count.png").exists()
    assert (tmp_path / "analysis" / "trace_readout" / "manifest.json").exists()
