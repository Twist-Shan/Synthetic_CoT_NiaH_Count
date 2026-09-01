from __future__ import annotations

from dataclasses import asdict

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import V20Example, V20Vocab, character_token, render_v20
from synthetic_counting_v20.model import build_model
from synthetic_counting_v51.config import preset_config as preset_v51
from synthetic_counting_v57.config import preset_config as preset_v57
from synthetic_counting_v57.preflight import run_preflight


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


def test_v57_changes_only_v51_parallel_width_factor() -> None:
    baseline = preset_v51("main", device="cpu")
    candidate = preset_v57("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "n_head", "n_embd", "n_inner"}
    assert candidate.n_layer == baseline.n_layer == 4
    assert (baseline.n_head, baseline.n_embd, baseline.n_inner) == (4, 256, 1024)
    assert (candidate.n_head, candidate.n_embd, candidate.n_inner) == (6, 384, 1536)
    assert baseline.n_embd // baseline.n_head == candidate.n_embd // candidate.n_head == 64
    assert candidate.batch_size == baseline.batch_size == 128
    assert candidate.task_output_structure_weight == 16.0
    assert config_from_dict(candidate.to_dict()) == candidate


def test_v57_adds_parallel_capacity_without_serial_depth() -> None:
    old = preset_v51("debug", device="cpu")
    new = preset_v57("debug", device="cpu")
    vocab = V20Vocab.build(new, "abc xyz\n")
    old_model = build_model(old, vocab, device="cpu")
    new_model = build_model(new, vocab, device="cpu")
    assert len(old_model.layers) == len(new_model.layers) == 4
    assert new_model.parameter_count() > old_model.parameter_count()


def test_v57_trace_serialization_is_identical_to_v51() -> None:
    old = preset_v51("debug", device="cpu")
    new = preset_v57("debug", device="cpu")
    old_vocab = V20Vocab.build(old, "abc xyz\n")
    new_vocab = V20Vocab.build(new, "abc xyz\n")
    assert render_v20(_example(), old_vocab, "thinking").tokens == render_v20(
        _example(), new_vocab, "thinking"
    ).tokens


def test_v57_preflight_is_exposed() -> None:
    assert callable(run_preflight)
