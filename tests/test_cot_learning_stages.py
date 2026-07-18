from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_cot_learning_stages",
    ROOT / "scripts" / "analyze_cot_learning_stages.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_first_stable_step_ignores_temporary_threshold_crossing() -> None:
    frame = pd.DataFrame(
        {
            "step": [0, 500, 1000, 1500, 2000],
            "value": [0.1, 0.95, 0.8, 0.96, 0.97],
        }
    )
    assert MODULE.first_stable_step(frame, "value", 0.95, 0.9) == 1500.0


def test_two_segment_breakpoint_finds_delayed_learning_transition() -> None:
    steps = np.arange(0, 10_500, 500)
    values = np.where(steps < 4_000, 0.1, 0.1 + (steps - 4_000) / 7_000)
    frame = pd.DataFrame({"step": steps, "value": values})
    breakpoint = MODULE.two_segment_breakpoint(frame, "value")
    assert 3_500 <= breakpoint <= 5_000


def test_expected_exact_from_local_is_length_sensitive() -> None:
    short = MODULE.expected_exact_from_local(0.9, "1-10")
    long = MODULE.expected_exact_from_local(0.9, "21-30")
    assert 0.0 < long < short < 1.0


def test_geometry_metrics_detect_one_dimensional_count_axis() -> None:
    vectors = []
    labels = []
    for count in range(1, 11):
        for offset in (-0.01, 0.01):
            vectors.append(np.array([count + offset, 0.002 * offset, 0.0]))
            labels.append(count)
    metrics = MODULE._geometry_metrics(vectors, labels)
    assert metrics["pc1_label_r2"] > 0.999
    assert metrics["pc1_adjacent_consistency"] > 0.999
    assert metrics["effective_dimension"] < 1.01
