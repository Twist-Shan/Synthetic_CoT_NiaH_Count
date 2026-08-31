from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import V20Example, V20Vocab, character_token, render_v20
from synthetic_counting_v51.config import preset_config as preset_v51
from synthetic_counting_v55.config import preset_config as preset_v55
from synthetic_counting_v55.preflight import run_preflight


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


def test_v55_changes_only_v51_structure_weight() -> None:
    baseline = preset_v51("main", device="cpu")
    candidate = preset_v55("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "task_output_structure_weight"}
    assert candidate.task_output_count_weight == 8.0
    assert candidate.task_output_trace_weight == 8.0
    assert candidate.task_output_structure_weight == 32.0
    assert candidate.task_output_scheduled_sampling_max_probability == 0.0
    assert candidate.train_steps == 10_000
    assert (candidate.n_layer, candidate.n_head, candidate.n_embd) == (4, 4, 256)
    assert config_from_dict(candidate.to_dict()) == candidate


def test_v55_rejects_noncanonical_structure_weight() -> None:
    candidate = preset_v55("main", device="cpu")
    with pytest.raises(ValueError, match="task_output_structure_weight=32"):
        replace(candidate, task_output_structure_weight=24.0).validate()


def test_v55_trace_serialization_is_identical_to_v51() -> None:
    old = preset_v51("debug", device="cpu")
    new = preset_v55("debug", device="cpu")
    old_vocab = V20Vocab.build(old, "abc xyz\n")
    new_vocab = V20Vocab.build(new, "abc xyz\n")
    assert render_v20(_example(), old_vocab, "thinking").tokens == render_v20(
        _example(), new_vocab, "thinking"
    ).tokens


def test_v55_preflight_is_exposed() -> None:
    assert callable(run_preflight)
