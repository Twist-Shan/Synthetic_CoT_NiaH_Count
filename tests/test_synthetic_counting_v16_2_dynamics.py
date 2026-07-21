from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_counting_v16_2.analysis import (
    _attention_categories,
    fit_ridge,
    ridge_raw_direction,
)
from synthetic_counting_v16_2.checkpoint_dynamics import (
    DynamicsOptions,
    _artifact_manifest,
    _atomic_json,
    _geometry_metrics,
    _part_complete,
    _state_sampling_audit_rows,
    _state_sampling_suites,
    linear_cka,
)
from synthetic_counting_v16_2.config import preset_config
from synthetic_counting_v16_2.data import (
    V16_2Vocab,
    balanced_v16_2_examples,
    build_corpus_split,
    load_corpus_text,
    render_v16_2,
    render_v16_2_shortened_trace,
)
from synthetic_counting_v16_2.needle_pool import build_needle_pool
from synthetic_counting_v16_2.timing import record_cached_event, timed_event


@pytest.fixture(scope="module")
def prepared():
    cfg = preset_config("debug", device="cpu")
    text = load_corpus_text()
    split = build_corpus_split(cfg, text)
    vocab = V16_2Vocab.build(cfg, text)
    pool = build_needle_pool(cfg, text, split, vocab.fingerprint)
    return cfg, text, split, vocab, pool


def test_shortened_trace_removes_only_final_pair(prepared):
    cfg, text, split, vocab, pool = prepared
    example = next(
        item
        for item in balanced_v16_2_examples(
            cfg, vocab, text, split, pool, 1, 710, region_name="validation"
        )
        if item.count >= 2
    )
    gold = render_v16_2(example, vocab, "thinking")
    shortened = render_v16_2_shortened_trace(example, vocab)
    assert gold.spans is not None and shortened.spans is not None
    assert shortened.tokens[: gold.spans.think_pos + 1] == gold.tokens[: gold.spans.think_pos + 1]
    assert len(shortened.spans.trace_index_positions) == example.count - 1
    assert len(shortened.spans.trace_marker_positions) == example.count - 1
    assert shortened.tokens[shortened.spans.ans_pos :] == [
        "<Ans>", vocab.number_token(example.count), "<EOS>"
    ]
    removed = [
        gold.tokens[gold.spans.trace_index_positions[-1]],
        gold.tokens[gold.spans.trace_marker_positions[-1]],
    ]
    assert removed == [vocab.number_token(example.count), example.needle_markers[-1]]
    assert len(gold.tokens) - len(shortened.tokens) == 2


def test_attention_category_metrics_match_hand_calculation(prepared):
    cfg, text, split, vocab, pool = prepared
    example = balanced_v16_2_examples(
        cfg, vocab, text, split, pool, 1, 711, region_name="validation"
    )[0]
    rendered = render_v16_2(example, vocab, "nonthinking")
    weights = np.zeros(len(rendered.tokens), dtype=float)
    needles = list(rendered.prompt_needle_positions)
    prompt = list(range(rendered.spans.prompt_start, rendered.spans.prompt_end_exclusive))
    weights[prompt] = 1.0
    weights[needles] = 3.0
    weights /= weights.sum()
    metrics = _attention_categories(rendered, weights)
    expected_needle_mass = float(weights[needles].sum())
    expected_prompt_mass = float(weights[prompt].sum())
    assert metrics["prompt_needles_mass"] == pytest.approx(expected_needle_mass)
    assert metrics["prompt_mass"] == pytest.approx(expected_prompt_mass)
    observed_fraction = expected_needle_mass / expected_prompt_mass
    uniform_fraction = len(needles) / len(prompt)
    assert metrics["needle_attention_enrichment"] == pytest.approx(
        observed_fraction / uniform_fraction
    )
    assert metrics["top_n_needle_recall"] == 1
    assert metrics["top_n_needle_precision"] == 1


def test_geometry_and_linear_cka_are_deterministic():
    labels = np.repeat(np.arange(1, 5), 3)
    vectors = np.column_stack((labels.astype(float), np.zeros(len(labels))))
    geometry = _geometry_metrics(vectors, labels)
    assert geometry["pc1_label_r2"] == pytest.approx(1.0)
    assert geometry["pc1_adjacent_consistency"] == pytest.approx(1.0)
    assert geometry["monotonic_order_violations"] == 0
    assert linear_cka(vectors, vectors) == pytest.approx(1.0)
    assert linear_cka(vectors, 2 * vectors + 7) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="aligned"):
        linear_cka(vectors, vectors[:-1])


def test_runtime_events_deduplicate_and_record_cached(tmp_path: Path):
    with timed_event(
        tmp_path, scope="unit", block="metric", position_encoding="rope",
        mode="thinking", step=500, num_examples=10,
    ):
        random.Random(1).random()
    record_cached_event(
        tmp_path, scope="unit", block="metric", position_encoding="rope",
        mode="thinking", step=500,
    )
    frame = pd.read_csv(tmp_path / "tables" / "runtime_events.csv")
    assert len(frame) == 1
    assert frame.iloc[0].status == "cached"
    assert bool(frame.iloc[0].resumed_or_cached)


def test_trace_progress_sampling_holds_total_count_fixed(prepared):
    cfg, text, split, vocab, pool = prepared
    examples = balanced_v16_2_examples(
        cfg, vocab, text, split, pool, 2, 712, region_name="validation"
    )
    final_examples, trace_examples = _state_sampling_suites(
        examples, 2, cfg.count_max_threshold, "thinking"
    )
    assert {item.count for item in final_examples} == set(
        range(1, cfg.count_max_threshold + 1)
    )
    assert {item.count for item in trace_examples} == {cfg.count_max_threshold}
    audit = pd.DataFrame(
        _state_sampling_audit_rows(
            examples, 2, cfg.count_max_threshold, "thinking", "heldout"
        )
    )
    trace = audit[audit.sampling_suite == "fixed_total_count_trace_progress"]
    assert set(trace.total_count) == {cfg.count_max_threshold}
    assert set(trace.progress_label) == set(range(1, cfg.count_max_threshold + 1))
    assert trace.groupby(["site", "progress_label"]).size().nunique() == 1


def test_ridge_direction_is_returned_in_raw_hidden_coordinates():
    train = np.asarray([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    labels = np.asarray([0.0, 1.0, 2.0, 3.0])
    mean, scale, beta = fit_ridge(train, labels)
    del mean
    assert np.allclose(ridge_raw_direction((np.zeros((1, 2)), scale, beta)), beta[1:] / scale.ravel())


def test_part_cache_verifies_every_artifact_hash(tmp_path: Path):
    artifact = tmp_path / "metric.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    fingerprint = "unit-test"
    _atomic_json(
        {
            "status": "complete",
            "options_fingerprint": fingerprint,
            "artifacts": _artifact_manifest(tmp_path),
        },
        tmp_path / "complete.json",
    )
    assert _part_complete(tmp_path, fingerprint)
    artifact.write_text("value\n2\n", encoding="utf-8")
    assert not _part_complete(tmp_path, fingerprint)


def test_dependent_dynamics_switches_fail_loudly():
    with pytest.raises(ValueError, match="run_similarity requires run_states"):
        DynamicsOptions(run_states=False, run_counterfactual=False).validate()
    with pytest.raises(ValueError, match="run_counterfactual requires run_states"):
        DynamicsOptions(
            run_states=False, run_similarity=False, run_counterfactual=True
        ).validate()
