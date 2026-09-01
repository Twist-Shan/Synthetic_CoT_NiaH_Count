from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import V20Vocab, load_corpus_text
from synthetic_counting_v20.model import build_model
from synthetic_counting_v44.config import preset_config as preset_v44
from synthetic_counting_v45.config import preset_config as preset_v45
from synthetic_counting_v45.behavior_gate import main as behavior_gate_main
from synthetic_counting_v45.preflight import main as preflight_main


def test_v45_only_reallocates_model_geometry_from_v44() -> None:
    baseline = preset_v44("main", device="cpu")
    candidate = preset_v45("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "n_layer", "n_head", "n_embd", "n_inner"}
    assert (baseline.n_layer, baseline.n_head, baseline.n_embd, baseline.n_inner) == (
        4,
        6,
        384,
        1536,
    )
    assert (candidate.n_layer, candidate.n_head, candidate.n_embd, candidate.n_inner) == (
        6,
        5,
        320,
        1280,
    )
    assert baseline.n_embd // baseline.n_head == 64
    assert candidate.n_embd // candidate.n_head == 64
    assert candidate.count_max_threshold == baseline.count_max_threshold == 10
    assert candidate.joint_sampler_max_starts_per_cell is None
    assert candidate.trace_format == baseline.trace_format == "separator"
    assert candidate.train_steps == baseline.train_steps == 8_000
    assert candidate.enabled_model_variants == baseline.enabled_model_variants
    assert config_from_dict(candidate.to_dict()) == candidate

    text = load_corpus_text()
    vocab = V20Vocab.build(candidate, text)
    baseline_parameters = build_model(baseline, vocab, device="cpu").parameter_count()
    candidate_parameters = build_model(candidate, vocab, device="cpu").parameter_count()
    assert abs(candidate_parameters - baseline_parameters) / baseline_parameters < 0.05


def test_v45_rejects_v44_geometry() -> None:
    candidate = preset_v45("main", device="cpu")
    with pytest.raises(ValueError, match="requires 6 layers"):
        replace(candidate, n_layer=4).validate()


def test_v45_remote_audits_are_exposed() -> None:
    assert callable(preflight_main)
    assert callable(behavior_gate_main)
