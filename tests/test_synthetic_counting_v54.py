from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import V20Example, V20Vocab, character_token, render_v20
from synthetic_counting_v20.training import scheduled_sampling_probability
from synthetic_counting_v51.config import preset_config as preset_v51
from synthetic_counting_v54.config import preset_config as preset_v54
from synthetic_counting_v54.preflight import run_preflight


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


def test_v54_changes_only_v51_mild_rollin_scalar() -> None:
    baseline = preset_v51("main", device="cpu")
    candidate = preset_v54("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {
        "version",
        "task_output_scheduled_sampling_max_probability",
    }
    assert candidate.task_output_scheduled_sampling_max_probability == 0.1
    assert candidate.n_layer == baseline.n_layer == 4
    assert candidate.n_head == baseline.n_head == 4
    assert candidate.n_embd == baseline.n_embd == 256
    assert candidate.task_output_count_weight == baseline.task_output_count_weight == 8.0
    assert candidate.task_output_trace_weight == baseline.task_output_trace_weight == 8.0
    assert candidate.task_output_structure_weight == baseline.task_output_structure_weight == 16.0
    assert candidate.train_steps == baseline.train_steps == 10_000
    assert config_from_dict(candidate.to_dict()) == candidate


def test_v54_rollin_is_mild_linear_and_thinking_only() -> None:
    candidate = preset_v54("main", device="cpu")
    assert scheduled_sampling_probability(candidate, 1_500, "thinking") == 0.0
    assert scheduled_sampling_probability(candidate, 5_750, "thinking") == pytest.approx(0.05)
    assert scheduled_sampling_probability(candidate, 10_000, "thinking") == pytest.approx(0.1)
    assert scheduled_sampling_probability(candidate, 10_000, "nonthinking") == 0.0


def test_v54_rejects_noncanonical_rollin() -> None:
    candidate = preset_v54("main", device="cpu")
    with pytest.raises(
        ValueError,
        match="task_output_scheduled_sampling_max_probability=0.1",
    ):
        replace(
            candidate,
            task_output_scheduled_sampling_max_probability=0.2,
        ).validate()


def test_v54_trace_serialization_is_identical_to_v51() -> None:
    old = preset_v51("debug", device="cpu")
    new = preset_v54("debug", device="cpu")
    old_vocab = V20Vocab.build(old, "abc xyz\n")
    new_vocab = V20Vocab.build(new, "abc xyz\n")
    assert render_v20(_example(), old_vocab, "thinking").tokens == render_v20(
        _example(), new_vocab, "thinking"
    ).tokens


def test_v54_preflight_is_exposed() -> None:
    assert callable(run_preflight)
