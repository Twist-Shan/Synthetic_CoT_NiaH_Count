from __future__ import annotations

import importlib.util
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import synthetic_counting_v16_3.training as training_module
from synthetic_counting_v16_3.cli import build_parser
from synthetic_counting_v16_3.config import config_from_dict, default_run_name, preset_config
from synthetic_counting_v16_3.data import (
    V16_3Vocab,
    balanced_v16_3_examples,
    build_corpus_split,
    build_loss_suite_manifests,
    build_test_suite_manifests,
    character_token,
    collate_v16_3,
    collate_v16_3_loss_weights,
    load_corpus_text,
    make_training_example,
    make_v16_3_example,
    render_v16_3,
    shifted_v16_3_token_losses,
)
from synthetic_counting_v16_3.needle_pool import build_needle_pool
from synthetic_counting_v16_3.plots import plot_v16_3_loss_suites
from synthetic_counting_v16_3.training import (
    autoregressive_task_evaluation,
    evaluate_loss_suite,
    paired_v16_3_model,
    planned_checkpoint_steps,
    prefix_permutation_consistency_evaluation,
    training_loss_phase,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared():
    cfg = preset_config("debug", device="cpu")
    text = load_corpus_text()
    split = build_corpus_split(cfg, text)
    vocab = V16_3Vocab.build(cfg, text)
    pool = build_needle_pool(cfg, text, split, vocab.fingerprint)
    return cfg, text, split, vocab, pool


def test_config_alias_ratio_validation_and_run_identity():
    cfg = preset_config(
        "main",
        task_occurrence_ratio=0.25,
        count_max_threshold=10,
        weight_decay=0.05,
        final_count_loss_weight=2.0,
        cot_trace_loss_weight=3.0,
        max_steps_for_language_pred=15,
        train_steps=123,
        eval_examples_per_count=7,
        enabled_model_variants=("rope/thinking", "rpe/nonthinking"),
    )
    assert cfg.count_max == cfg.count_max_threshold == 10
    assert preset_config("main").checkpoint_every == 500
    reference = preset_config("main", device="cuda")
    assert reference.position_encodings == ("rope",)
    assert reference.enabled_model_variants == ("rope/nonthinking", "rope/thinking")
    assert reference.task_occurrence_ratio == 1.0
    assert reference.eval_examples_per_count == 50
    assert reference.sequence_layout == "data_then_query"
    assert default_run_name(reference) == (
        "v16_3_main_data-query_L256_pool100x3_pf0p04_count1-10_taskr1_wd0p01_"
        "fcw1_cotw1_langsteps1500_steps10000_ckpt500_evaln500_"
        "rope-nt-rope-t_allseq-taskout_seed1234"
    )
    assert cfg.model_variants == (("rope", "thinking"), ("rpe", "nonthinking"))
    assert "taskr0p25" in default_run_name(cfg)
    assert "pool100x3" in default_run_name(cfg)
    assert "wd0p05" in default_run_name(cfg)
    assert "fcw2_cotw3_langsteps15_steps123" in default_run_name(cfg)
    assert "steps123_ckpt500_evaln70" in default_run_name(cfg)
    assert "allseq-taskout" in default_run_name(cfg)
    assert "rope-t-rpe-nt" in default_run_name(cfg)
    serialized = cfg.to_dict()
    assert config_from_dict(serialized) == cfg
    with pytest.raises(ValueError, match="sequence_layout"):
        preset_config("debug", sequence_layout="query_then_data")
    serialized["count_max"] = 9
    with pytest.raises(ValueError, match="disagrees"):
        config_from_dict(serialized)
    for ratio in (-0.01, 1.01):
        with pytest.raises(ValueError, match="task_occurrence_ratio"):
            preset_config("debug", task_occurrence_ratio=ratio)
    for name in ("final_count_loss_weight", "cot_trace_loss_weight"):
        for value in (0.0, -1.0, math.inf, math.nan):
            with pytest.raises(ValueError, match=name):
                preset_config("debug", **{name: value})
    for value in (-1.0, -math.inf, math.inf, math.nan):
        with pytest.raises(ValueError, match="weight_decay"):
            preset_config("debug", weight_decay=value)
    assert preset_config("debug", weight_decay=0.0).weight_decay == 0.0
    assert default_run_name(replace(cfg, weight_decay=0.0)) != default_run_name(cfg)
    for value in (-1, 1.5, True):
        with pytest.raises(ValueError, match="max_steps_for_language_pred"):
            preset_config("debug", max_steps_for_language_pred=value)
    with pytest.raises(ValueError, match="task_occurrence_ratio"):
        preset_config("debug", task_occurrence_ratio=0.0, max_steps_for_language_pred=2)
    all_language = preset_config(
        "debug", task_occurrence_ratio=0.0, max_steps_for_language_pred=6
    )
    assert training_loss_phase(all_language, 6) == "all_sequence"
    assert training_loss_phase(cfg, 15) == "all_sequence"
    assert training_loss_phase(cfg, 16) == "task_output"
    assert default_run_name(replace(cfg, max_steps_for_language_pred=16)) != default_run_name(cfg)
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="checkpoint_every"):
            preset_config("main", checkpoint_every=value)


def test_autoregressive_empty_suite_and_metric_names(prepared):
    cfg, _, _, vocab, _ = prepared
    model = paired_v16_3_model(cfg, vocab, "rope")
    empty = autoregressive_task_evaluation(
        model,
        cfg,
        vocab,
        [],
        position_encoding="rope",
        mode="thinking",
        step=0,
    )
    assert empty.empty


def test_resume_rejects_config_and_variant_identity_mismatch(prepared):
    cfg, _, split, vocab, pool = prepared
    common = {
        "position_encoding": "rope",
        "mode": "thinking",
        "step": 3,
        "pool_fingerprint": pool.pool_fingerprint,
        "split_fingerprint": split.split_fingerprint,
        "vocab_fingerprint": vocab.fingerprint,
    }
    with pytest.raises(ValueError, match="training config"):
        training_module._restore_checkpoint(
            {**common, "config": replace(cfg, train_steps=cfg.train_steps + 1).to_dict()},
            None,
            None,
            random.Random(0),
            pool,
            split,
            vocab,
            cfg,
            "rope",
            "thinking",
            3,
        )
    with pytest.raises(ValueError, match="position encoding"):
        training_module._restore_checkpoint(
            {**common, "config": cfg.to_dict(), "position_encoding": "rpe"},
            None,
            None,
            random.Random(0),
            pool,
            split,
            vocab,
            cfg,
            "rope",
            "thinking",
            3,
        )


def test_permutation_evaluation_separates_tf_and_ar(prepared):
    cfg, text, split, vocab, pool = prepared
    model = paired_v16_3_model(cfg, vocab, "rope")
    example = balanced_v16_3_examples(
        cfg, vocab, text, split, pool, 1, 713, region_name="validation"
    )[0]
    result = prefix_permutation_consistency_evaluation(
        model,
        cfg,
        vocab,
        [example],
        position_encoding="rope",
        mode="thinking",
    )
    row = result.iloc[0]
    assert row.tf_evaluation_context == "teacher_forced_gold_trace"
    assert row.ar_evaluation_context == "free_generation_from_prompt"
    assert row.permutation_accuracy == row.ar_permutation_accuracy
    assert row.predictions_json == row.ar_predictions_json
    assert 0.0 <= row.ar_answer_rate <= 1.0


def test_rpe_max_update_derives_distance_and_changes_run_identity():
    legacy = preset_config(
        "main", count_max_threshold=30, rpe_max_update=False,
        enabled_model_variants=("rpe/nonthinking",),
    )
    updated = preset_config(
        "main", count_max_threshold=30, rpe_max_update=True,
        enabled_model_variants=("rpe/nonthinking",),
    )
    assert legacy.max_relative_distance == 256
    assert updated.max_render_len == 327
    assert updated.max_relative_distance == updated.max_render_len - 1 == 326
    assert "rped326" not in default_run_name(legacy)
    assert "rped326" in default_run_name(updated)
    assert config_from_dict(updated.to_dict()) == updated
    with pytest.raises(ValueError, match="rpe_max_update"):
        preset_config("main", rpe_max_update=1)


def test_legacy_config_defaults_and_variant_validation():
    cfg = preset_config("debug", device="cpu")
    legacy = cfg.to_dict()
    legacy.pop("enabled_model_variants")
    legacy.pop("final_count_loss_weight")
    legacy.pop("cot_trace_loss_weight")
    legacy.pop("weight_decay")
    legacy.pop("max_steps_for_language_pred")
    legacy.pop("checkpoint_every")
    legacy.pop("rpe_max_update")
    loaded = config_from_dict(legacy)
    assert loaded.enabled_model_variants == ("rope/nonthinking", "rope/thinking")
    assert loaded.final_count_loss_weight == loaded.cot_trace_loss_weight == 1.0
    assert loaded.weight_decay == 0.01
    assert loaded.checkpoint_every == 1_000
    assert loaded.max_steps_for_language_pred == loaded.train_steps
    assert loaded.rpe_max_update is False
    assert loaded.loss_scope == "all_sequence"
    with pytest.raises(ValueError, match="at least one"):
        preset_config("debug", enabled_model_variants=())
    with pytest.raises(ValueError, match="duplicates"):
        preset_config(
            "debug", enabled_model_variants=("rope/thinking", "rope/thinking")
        )


def test_cli_exposes_weights_variants_steps_and_evaluation_size():
    args = build_parser().parse_args(
        [
            "--final-count-loss-weight", "4",
            "--cot-trace-loss-weight", "2",
            "--weight-decay", "0.05",
            "--rpe-max-update",
            "--max-steps-for-language-pred", "12",
            "--model-variant", "rope/thinking",
            "--model-variant", "rpe/nonthinking",
            "--train-steps", "17",
            "--checkpoint-every", "5",
            "--eval-examples-per-count", "3",
        ]
    )
    assert args.final_count_loss_weight == 4
    assert args.cot_trace_loss_weight == 2
    assert args.weight_decay == 0.05
    assert args.rpe_max_update is True
    assert args.max_steps_for_language_pred == 12
    assert args.model_variant == ["rope/thinking", "rpe/nonthinking"]
    assert args.train_steps == 17
    assert args.checkpoint_every == 5
    assert args.eval_examples_per_count == 3


def test_checkpoint_plan_includes_nonaligned_boundary_and_final():
    cfg = preset_config(
        "debug", train_steps=8, checkpoint_every=3,
        max_steps_for_language_pred=2,
    )
    assert planned_checkpoint_steps(cfg) == [0, 2, 3, 6, 8]


def test_guarded_corpus_regions_are_disjoint(prepared):
    cfg, text, split, _, _ = prepared
    assert split.train.end + cfg.seq_len - 1 == split.validation.start
    assert split.validation.end + cfg.seq_len - 1 == split.test.start
    assert split.test.end == len(text)
    for region in (split.train, split.validation, split.test):
        assert region.length >= cfg.seq_len


def test_pool_contains_unique_distinct_triples_under_threshold(prepared):
    cfg, text, split, _, pool = prepared
    assert len(pool) == cfg.needle_pool_size
    sets = [item.characters for item in pool.sets]
    assert len(set(sets)) == len(sets)
    for item in pool.sets:
        assert len(item.characters) == len(set(item.characters)) == 3
        assert item.characters == tuple(sorted(item.characters, key=ord))
        assert item.frequency_sum <= cfg.needle_pool_frequency_threshold + 1e-12
        expected = sum(text[split.train.start : split.train.end].count(char) for char in item.characters)
        assert math.isclose(item.frequency_sum, expected / split.train.length, abs_tol=1e-12)


def test_task_example_preserves_window_and_renders_data_before_three_character_query(prepared):
    cfg, text, split, vocab, pool = prepared
    example = make_v16_3_example(cfg, vocab, text, split, pool, random.Random(4), region_name="train")
    source = text[example.corpus_start : example.corpus_end]
    assert example.seq_tokens == [character_token(char) for char in source]
    assert len(set(example.needle_characters)) == 3
    expected_positions = tuple(
        index for index, char in enumerate(source) if char in set(example.needle_characters)
    )
    assert example.needle_positions == expected_positions
    assert example.count == len(expected_positions)
    nonthinking = render_v16_3(example, vocab, "nonthinking")
    thinking = render_v16_3(example, vocab, "thinking")
    for rendered in (nonthinking, thinking):
        assert rendered.spans is not None
        spans = rendered.spans
        assert spans.prompt_start == 1
        assert spans.prompt_end_exclusive == 1 + cfg.seq_len
        assert spans.query_start == spans.prompt_end_exclusive
        assert spans.query_end_exclusive == spans.query_start + 5
        assert rendered.tokens[spans.prompt_start : spans.prompt_end_exclusive] == example.seq_tokens
        assert rendered.tokens[spans.query_start] == "<CountChar>"
        assert rendered.tokens[spans.query_end_exclusive - 1] == "<Sep>"
        assert rendered.tokens[spans.query_start + 1 : spans.query_end_exclusive - 1] == [
            character_token(character) for character in example.rendered_set_order
        ]
        assert spans.query_positions == tuple(range(spans.query_start, spans.query_end_exclusive))
        assert all(position < spans.query_start for position in rendered.prompt_needle_positions)
    assert nonthinking.tokens[nonthinking.spans.query_end_exclusive] == "<Ans>"
    assert thinking.tokens[thinking.spans.query_end_exclusive] == "<Think>"
    assert len(nonthinking.tokens) == cfg.seq_len + 9
    assert len(thinking.tokens) == cfg.seq_len + 11 + 2 * example.count
    trace_markers = [thinking.tokens[position] for position in thinking.spans.trace_marker_positions]
    assert trace_markers == list(example.needle_markers)
    assert nonthinking.labels == nonthinking.input_ids
    assert thinking.labels == thinking.input_ids


def test_v16_2_query_first_and_v16_3_query_last_are_distinct(prepared):
    from synthetic_counting_v16_2.config import preset_config as preset_v16_2
    from synthetic_counting_v16_2.data import (
        V16_2Vocab,
        build_corpus_split as build_split_v16_2,
        make_v16_2_example,
        render_v16_2,
    )
    from synthetic_counting_v16_2.needle_pool import build_needle_pool as build_pool_v16_2

    _, text, _, vocab_v16_3, pool_v16_3 = prepared
    cfg_v16_2 = preset_v16_2("debug", device="cpu")
    split_v16_2 = build_split_v16_2(cfg_v16_2, text)
    vocab_v16_2 = V16_2Vocab.build(cfg_v16_2, text)
    pool_v16_2 = build_pool_v16_2(
        cfg_v16_2, text, split_v16_2, vocab_v16_2.fingerprint
    )
    example_v16_2 = make_v16_2_example(
        cfg_v16_2,
        vocab_v16_2,
        text,
        split_v16_2,
        pool_v16_2,
        random.Random(404),
        region_name="train",
    )
    rendered_v16_2 = render_v16_2(example_v16_2, vocab_v16_2, "nonthinking")
    assert rendered_v16_2.tokens[1] == "<CountChar>"
    assert rendered_v16_2.spans is not None
    assert rendered_v16_2.spans.prompt_start == 6

    cfg_v16_3, text, split_v16_3, _, _ = prepared
    example_v16_3 = make_v16_3_example(
        cfg_v16_3,
        vocab_v16_3,
        text,
        split_v16_3,
        pool_v16_3,
        random.Random(404),
        region_name="train",
    )
    rendered_v16_3 = render_v16_3(example_v16_3, vocab_v16_3, "nonthinking")
    assert rendered_v16_3.spans is not None
    assert rendered_v16_3.spans.prompt_start == 1
    assert rendered_v16_3.spans.query_start == 1 + cfg_v16_3.seq_len
    assert rendered_v16_3.tokens[rendered_v16_3.spans.query_start] == "<CountChar>"

    report_builder = (ROOT / "scripts" / "build_v16_2_rope_chinese_report.py").read_text(
        encoding="utf-8"
    )
    assert "Nonthinking（query-first）" in report_builder
    assert "prompt[256] · query[3]" not in report_builder


def test_task_relevant_loss_weights_are_shift_aligned(prepared):
    base, text, split, vocab, pool = prepared
    cfg = replace(base, final_count_loss_weight=7.0, cot_trace_loss_weight=3.0)
    example = make_v16_3_example(
        cfg, vocab, text, split, pool, random.Random(41), region_name="train"
    )
    nonthinking = render_v16_3(example, vocab, "nonthinking")
    thinking = render_v16_3(example, vocab, "thinking")
    weights = collate_v16_3_loss_weights([nonthinking, thinking], cfg, "cpu")
    assert weights[0, nonthinking.spans.count_pos] == 7
    assert weights[0].eq(3).sum() == 0
    assert weights[1, thinking.spans.count_pos] == 7
    for position in (
        *thinking.spans.trace_index_positions,
        *thinking.spans.trace_marker_positions,
    ):
        assert weights[1, position] == 3
    for position in (
        thinking.spans.think_pos,
        thinking.spans.think_close_pos,
        thinking.spans.ans_pos,
        thinking.spans.eos_pos,
    ):
        assert weights[1, position] == 1

    ids, labels, _ = collate_v16_3([nonthinking, thinking], vocab, "cpu")
    torch.manual_seed(9)
    logits = torch.randn(
        len(ids), ids.shape[1], len(vocab.id_to_token), requires_grad=True
    )
    weighted, losses, active = shifted_v16_3_token_losses(logits, labels, weights)
    shifted_weights = weights[:, 1:] * active
    expected = (losses * shifted_weights).sum() / shifted_weights.sum()
    assert float(weighted.detach()) == pytest.approx(float(expected.detach()))

    unit = torch.where(weights > 0, torch.ones_like(weights), weights)
    old_loss, _, _ = shifted_v16_3_token_losses(logits, labels)
    unit_loss, _, _ = shifted_v16_3_token_losses(logits, labels, unit)
    assert float(unit_loss.detach()) == pytest.approx(float(old_loss.detach()))
    weighted.backward()
    assert torch.isfinite(logits.grad).all()

    with pytest.raises(ValueError, match="same shape"):
        shifted_v16_3_token_losses(logits.detach(), labels, weights[:, :-1])


def test_scheduled_task_output_masks_are_mode_specific_and_shift_aligned(prepared):
    base, text, split, vocab, pool = prepared
    cfg = replace(
        base,
        max_steps_for_language_pred=2,
        final_count_loss_weight=7.0,
        cot_trace_loss_weight=3.0,
    )
    task = make_v16_3_example(
        cfg, vocab, text, split, pool, random.Random(142), region_name="train"
    )
    raw = make_training_example(
        replace(cfg, task_occurrence_ratio=0.0),
        vocab,
        text,
        split,
        pool,
        random.Random(143),
    )
    raw_rendered = render_v16_3(raw, vocab, "nonthinking")
    nonthinking = render_v16_3(task, vocab, "nonthinking")
    thinking = render_v16_3(task, vocab, "thinking")
    rendered = [raw_rendered, nonthinking, thinking]

    original = collate_v16_3_loss_weights(rendered, cfg, "cpu")
    at_boundary = collate_v16_3_loss_weights(rendered, cfg, "cpu", step=2)
    assert torch.equal(at_boundary, original)

    post = collate_v16_3_loss_weights(rendered, cfg, "cpu", step=3)
    assert post[0].eq(0).all()
    for row, item, start in (
        (1, nonthinking, nonthinking.spans.ans_pos),
        (2, thinking, thinking.spans.think_pos),
    ):
        assert start is not None
        assert post[row, :start].eq(0).all()
        assert post[row, start : item.spans.eos_pos + 1].gt(0).all()
        assert post[row, item.spans.eos_pos + 1 :].eq(0).all()
        assert post[row, item.spans.count_pos] == 7
    for position in (
        *thinking.spans.trace_index_positions,
        *thinking.spans.trace_marker_positions,
    ):
        assert post[2, position] == 3
    for position in (
        thinking.spans.think_pos,
        thinking.spans.think_close_pos,
        thinking.spans.ans_pos,
        thinking.spans.eos_pos,
    ):
        assert post[2, position] == 1

    ids, labels, _ = collate_v16_3(rendered, vocab, "cpu")
    torch.manual_seed(19)
    logits = torch.randn(len(ids), ids.shape[1], len(vocab.id_to_token), requires_grad=True)
    loss, losses, active = shifted_v16_3_token_losses(logits, labels, post)
    shifted_weights = post[:, 1:] * active
    expected = (losses * shifted_weights).sum() / shifted_weights.sum()
    assert float(loss.detach()) == pytest.approx(float(expected.detach()))
    loss.backward()
    assert logits.grad[0].eq(0).all()
    for row, item, start in (
        (1, nonthinking, nonthinking.spans.ans_pos),
        (2, thinking, thinking.spans.think_pos),
    ):
        assert start is not None
        assert logits.grad[row, : start - 1].eq(0).all()
        assert logits.grad[row, start - 1 : item.spans.eos_pos].abs().sum() > 0
        assert logits.grad[row, item.spans.eos_pos :].eq(0).all()

    with pytest.raises(ValueError, match="step"):
        collate_v16_3_loss_weights(rendered, cfg, "cpu", step=-1)


def test_ratio_zero_and_one_boundaries(prepared):
    base, text, split, vocab, pool = prepared
    zero = replace(base, task_occurrence_ratio=0.0)
    rng_zero = random.Random(22)
    raw = [make_training_example(zero, vocab, text, split, pool, rng_zero) for _ in range(12)]
    assert all(item.example_kind == "raw_lm" for item in raw)
    for example in raw:
        rendered_n = render_v16_3(example, vocab, "nonthinking")
        rendered_t = render_v16_3(example, vocab, "thinking")
        assert rendered_n.tokens == rendered_t.tokens == example.seq_tokens
        assert len(rendered_n.tokens) == zero.seq_len
        assert all(not token.startswith("<Count") for token in rendered_n.tokens)

    one = replace(base, task_occurrence_ratio=1.0)
    rng_one = random.Random(22)
    task = [make_training_example(one, vocab, text, split, pool, rng_one) for _ in range(12)]
    assert all(item.example_kind == "counting_task" for item in task)
    assert all(1 <= int(item.count) <= one.count_max_threshold for item in task)


def test_task_output_batch_resamples_an_all_raw_draw(prepared, monkeypatch):
    base, text, split, vocab, pool = prepared
    cfg = replace(base, batch_size=4, max_steps_for_language_pred=0)
    raw = make_training_example(
        replace(cfg, task_occurrence_ratio=0.0),
        vocab,
        text,
        split,
        pool,
        random.Random(201),
    )
    task = make_v16_3_example(
        cfg, vocab, text, split, pool, random.Random(202), region_name="train"
    )
    draws = iter([raw, raw, raw, raw, task, raw, raw, raw])
    monkeypatch.setattr(
        training_module,
        "make_training_example",
        lambda *_args, **_kwargs: next(draws),
    )
    examples, rendered = training_module._training_batch(
        cfg,
        vocab,
        text,
        split,
        pool,
        "nonthinking",
        random.Random(203),
        require_task=True,
    )
    assert sum(item.example_kind == "counting_task" for item in examples) == 1
    weights = collate_v16_3_loss_weights(rendered, cfg, "cpu", step=1)
    assert weights.sum() > 0


def test_balanced_examples_and_fixed_suite_composition(prepared):
    cfg, text, split, vocab, pool = prepared
    balanced = balanced_v16_3_examples(
        cfg, vocab, text, split, pool, 2, 91, region_name="validation"
    )
    assert {count: sum(item.count == count for item in balanced) for count in range(1, 5)} == {
        1: 2,
        2: 2,
        3: 2,
        4: 2,
    }
    curves = build_loss_suite_manifests(cfg, vocab, text, split, pool)
    test = build_test_suite_manifests(cfg, vocab, text, split, pool)
    expected = cfg.eval_examples_per_count * cfg.count_max_threshold
    for source, expected_region in (("train", "train"), ("heldout", "validation")):
        assert set(curves[source]) == {"raw", "task", "mixture"}
        assert all(len(values) == expected for values in curves[source].values())
        assert all(item.example_kind == "raw_lm" for item in curves[source]["raw"])
        assert all(item.example_kind == "counting_task" for item in curves[source]["task"])
        assert all(item.corpus_region == expected_region for values in curves[source].values() for item in values)
        task_in_mixture = sum(item.example_kind == "counting_task" for item in curves[source]["mixture"])
        assert task_in_mixture == round(cfg.task_occurrence_ratio * expected)
    assert all(item.corpus_region == "test" for values in test.values() for item in values)


class _CheatingLossModel(torch.nn.Module):
    """Makes deterministic position-dependent losses for aggregation tests."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size, device=input_ids.device)
        for row in range(batch):
            for position in range(length - 1):
                confidence = 0.1 + 0.01 * position
                logits[row, position, input_ids[row, position + 1]] = confidence
        return SimpleNamespace(logits=logits)


def test_suite_loss_is_equal_mean_over_input_sequences_and_batch_invariant(prepared):
    cfg, text, split, vocab, pool = prepared
    raw_cfg = replace(cfg, task_occurrence_ratio=0.5)
    raw = make_training_example(replace(raw_cfg, task_occurrence_ratio=0), vocab, text, split, pool, random.Random(5))
    task = make_v16_3_example(raw_cfg, vocab, text, split, pool, random.Random(6), region_name="train")
    examples = [raw, task]
    model = _CheatingLossModel(len(vocab.id_to_token))
    row, _ = evaluate_loss_suite(
        model,
        replace(raw_cfg, analysis_batch_size=2),
        vocab,
        examples,
        position_encoding="rope",
        mode="thinking",
        step=0,
        curve_source="train",
        suite="mixture",
    )
    rendered = [render_v16_3(example, vocab, "thinking") for example in examples]
    ids, labels, mask = collate_v16_3(rendered, vocab, "cpu")
    output = model(ids, mask)
    _, losses, active = shifted_v16_3_token_losses(output.logits, labels)
    per_sequence = [
        float((losses[index] * active[index]).sum() / active[index].sum())
        for index in range(len(examples))
    ]
    assert row["example_mean_cross_entropy"] == pytest.approx(np.mean(per_sequence))
    token_weighted = float((losses * active).sum() / active.sum())
    assert row["token_weighted_cross_entropy"] == pytest.approx(token_weighted)

    one_at_a_time, _ = evaluate_loss_suite(
        model,
        replace(raw_cfg, analysis_batch_size=1),
        vocab,
        examples,
        position_encoding="rope",
        mode="thinking",
        step=0,
        curve_source="train",
        suite="mixture",
    )
    assert one_at_a_time["example_mean_cross_entropy"] == pytest.approx(
        row["example_mean_cross_entropy"]
    )
    assert one_at_a_time["token_weighted_cross_entropy"] == pytest.approx(
        row["token_weighted_cross_entropy"]
    )


def test_plot_supports_each_mode_individually_and_jointly(tmp_path):
    cfg = preset_config(
        "debug", device="cpu", enabled_model_variants=("rope/nonthinking", "rope/thinking")
    )
    rows = []
    for mode in ("nonthinking", "thinking"):
        for source in ("train", "heldout"):
            for suite in ("raw", "task", "mixture"):
                for step in (0, 3, 6):
                    rows.append(
                        {
                            "step": step,
                            "position_encoding": "rope",
                            "mode": mode,
                            "curve_source": source,
                            "suite": suite,
                            "example_mean_cross_entropy": 4 - 0.1 * step + (source == "heldout") * 0.2,
                            "token_weighted_cross_entropy": 4,
                        }
                    )
    tables = tmp_path / "tables"
    tables.mkdir()
    figures = tmp_path / "figures"
    figures.mkdir()
    frame = pd.DataFrame(rows)
    for modes in (("nonthinking",), ("thinking",), ("nonthinking", "thinking")):
        frame[frame["mode"].isin(modes)].to_csv(tables / "eval_loss_curves.csv", index=False)
        plot_v16_3_loss_suites(cfg, tmp_path)
        assert (figures / "learning_loss_suites_train_vs_heldout.png").stat().st_size > 0


def test_v16_3_notebook_compiles_and_legacy_v16_runner_is_isolated(tmp_path):
    notebook_path = ROOT / "notebooks" / "Trace_Count_v16_3_Colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "drive.mount" in "".join(code_cells[0]["source"])
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "Colab Notebooks/NIAH_synthetic" in source
    assert 'DRIVE_RESULTS_ROOT = DRIVE_REPO_ROOT / "colab_results"' in source
    assert "Colab_Notebooks/CoT_Counting" not in source
    assert '"--no-deps"' in source
    assert 'sys.path.insert(0, src_root)' in source
    assert "Notebook kernel imported stale package" in source
    assert "Subprocess imported stale package" in source
    assert "test_process.check_returncode()" in source
    assert 'SEQUENCE_LAYOUT = "data_then_query"' in source
    assert '"sequence_layout": PLANNED_CONFIG.sequence_layout' in source
    assert "AUTO_DISCONNECT = True" in source
    assert "drive.flush_and_unmount()" in source
    assert "runtime.unassign()" in source
    assert "TASK_OCCURRENCE_RATIO =" in source
    assert '"--task-occurrence-ratio", str(TASK_OCCURRENCE_RATIO)' in source
    for editable_setting in (
        "FINAL_COUNT_LOSS_WEIGHT",
        "COT_TRACE_LOSS_WEIGHT",
        "WEIGHT_DECAY",
        "RPE_max_update",
        "RUN_ROPE_NONTHINKING",
        "RUN_ROPE_THINKING",
        "RUN_RPE_NONTHINKING",
        "RUN_RPE_THINKING",
        "MAX_TRAIN_STEPS",
        "MAX_STEPS_FOR_LANGUAGE_PRED",
        "CHECKPOINT_EVERY_STEPS",
        "EVAL_EXAMPLES_PER_COUNT",
    ):
        assert f"{editable_setting} =" in source
    assert "examples for each count; suite size = this value x COUNT_MAX_THRESHOLD" in source
    assert '"--final-count-loss-weight", str(FINAL_COUNT_LOSS_WEIGHT)' in source
    assert '"--cot-trace-loss-weight", str(COT_TRACE_LOSS_WEIGHT)' in source
    assert '"--weight-decay", str(WEIGHT_DECAY)' in source
    assert 'rpe_max_update=RPE_max_update' in source
    assert '"--rpe-max-update" if RPE_max_update else "--no-rpe-max-update"' in source
    assert '"max_relative_distance": PLANNED_CONFIG.max_relative_distance' in source
    assert "weight_decay=WEIGHT_DECAY" in source
    assert '"weight_decay": WEIGHT_DECAY' in source
    assert '"--train-steps", str(MAX_TRAIN_STEPS)' in source
    assert '"--max-steps-for-language-pred", str(MAX_STEPS_FOR_LANGUAGE_PRED)' in source
    assert '"--checkpoint-every", str(CHECKPOINT_EVERY_STEPS)' in source
    assert "max_steps_for_language_pred=MAX_STEPS_FOR_LANGUAGE_PRED" in source
    assert '"task_output_only_steps": TASK_OUTPUT_ONLY_STEPS' in source
    assert '"--eval-examples-per-count", str(EVAL_EXAMPLES_PER_COUNT)' in source
    assert 'base_cmd += ["--model-variant", variant]' in source
    assert "--stage\", \"prepare" in source
    for cell in code_cells:
        compile("".join(cell["source"]), str(notebook_path), "exec")

    builder_path = ROOT / "scripts" / "build_v16_3_notebook.py"
    spec = importlib.util.spec_from_file_location("build_v16_3_notebook", builder_path)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.OUTPUT = tmp_path / notebook_path.name
    generated_path = builder.build()
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    generated_source = "\n".join(
        "".join(cell["source"])
        for cell in generated["cells"]
        if cell["cell_type"] == "code"
    )
    # The builder retains stable repository defaults, while users may edit the
    # generated notebook's runtime-settings cell for individual experiments.
    assert "FINAL_COUNT_LOSS_WEIGHT = 1.0" in generated_source
    assert "COT_TRACE_LOSS_WEIGHT = 1.0" in generated_source
    assert "WEIGHT_DECAY = 0.01" in generated_source
    assert "TASK_OCCURRENCE_RATIO = 1.0" in generated_source
    assert 'SEQUENCE_LAYOUT = "data_then_query"' in generated_source
    assert "RPE_max_update = False" in generated_source
    assert "RUN_ROPE_NONTHINKING = True" in generated_source
    assert "RUN_ROPE_THINKING = True" in generated_source
    assert "RUN_RPE_NONTHINKING = False" in generated_source
    assert "RUN_RPE_THINKING = False" in generated_source
    assert "MAX_TRAIN_STEPS = 10_000" in generated_source
    assert "MAX_STEPS_FOR_LANGUAGE_PRED = 1_500" in generated_source
    assert "EVAL_EXAMPLES_PER_COUNT = 50" in generated_source
    assert "CHECKPOINT_EVERY_STEPS = 500" in generated_source
    assert "RUN_CHECKPOINT_DYNAMICS = True" in generated_source
    assert "DYNAMICS_ATTENTION_EXAMPLES_PER_COUNT = 20" in generated_source
    assert "DYNAMICS_AR_EXAMPLES_PER_COUNT = 10" in generated_source
    assert "DYNAMICS_STATE_TRAIN_EXAMPLES_PER_COUNT = 40" in generated_source
    assert "DYNAMICS_STATE_EVAL_EXAMPLES_PER_COUNT = 15" in generated_source
    assert generated["metadata"] == notebook["metadata"]
    assert [cell["id"] for cell in generated["cells"]] == [
        cell["id"] for cell in notebook["cells"]
    ]
    for generated_cell, notebook_cell in zip(generated["cells"], notebook["cells"]):
        if notebook_cell["id"] == "runtime-settings":
            assert generated_cell["cell_type"] == notebook_cell["cell_type"] == "code"
            continue
        assert generated_cell == notebook_cell

    legacy_runner = (ROOT / "src" / "synthetic_counting_v16" / "run_v16.py").read_text(encoding="utf-8")
    assert "synthetic_counting_v16_3" not in legacy_runner
