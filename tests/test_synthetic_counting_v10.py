from __future__ import annotations

import random

import numpy as np
import torch

from synthetic_counting_v10.attention_causal import (
    _head_mask,
    attention_categories,
    collect_attention,
)
from synthetic_counting_v10.config import preset_config
from synthetic_counting_v10.core import (
    Vocab,
    build_model,
    component_target_positions,
    make_example,
    render,
)
from synthetic_counting_v10.state_causal import (
    SITE_NONTHINKING_FINAL,
    _fit_pca,
    fit_count_directions,
)
from synthetic_counting_v10.report_followups import (
    _forward_with_position_local_head_ablation,
    _layer_matched_order,
    _local_query_positions,
)


def tiny_setup():
    cfg = preset_config("debug", count_max=6)
    vocab = Vocab.build(cfg)
    model = build_model(cfg, vocab, "cpu").eval()
    return cfg, vocab, model


def test_v10_uses_separate_v2_formats_and_shared_numeric_tokens():
    cfg, vocab, _ = tiny_setup()
    example = make_example(cfg, vocab, random.Random(1), count=6)
    direct = render(example, vocab, "nonthinking")
    thinking = render(example, vocab, "thinking")

    assert direct.tokens[-3:] == ["<Ans>", "<6>", "<EOS>"]
    assert thinking.tokens[-4:] == ["</Think>", "<Ans>", "<6>", "<EOS>"]
    assert [thinking.tokens[pos] for pos in thinking.spans.trace_index_positions] == [
        f"<{value}>" for value in range(1, 7)
    ]
    assert thinking.tokens[thinking.spans.count_pos] == thinking.tokens[thinking.spans.trace_index_positions[-1]]
    assert len(thinking.tokens) <= cfg.n_positions
    assert component_target_positions(direct)["final_count"] == [direct.spans.count_pos]
    assert len(component_target_positions(thinking)["trace_marker"]) == 6


def test_attention_categories_partition_the_causal_row():
    cfg, vocab, _ = tiny_setup()
    example = make_example(cfg, vocab, random.Random(2), count=4)
    item = render(example, vocab, "thinking")
    row = np.zeros(len(item.tokens), dtype=float)
    row[: item.spans.ans_pos + 1] = 1.0 / (item.spans.ans_pos + 1)
    categories = attention_categories(item, row)
    assert abs(categories["classified_mass"] + categories["other_or_query_self_mass"] - 1.0) < 1e-6
    assert categories["prompt_needles_mass"] > 0
    assert 0 <= categories["broad_attention_score"] <= categories["prompt_needles_mass"]


def test_tiny_attention_collection_and_head_mask():
    cfg, vocab, model = tiny_setup()
    examples = [make_example(cfg, vocab, random.Random(3), count=3)]
    detail, summary = collect_attention(model, vocab, examples, "thinking", "cpu")
    assert not detail.empty
    assert {"final_count_query", "targeted_retrieval_query", "successor_query"}.issubset(
        set(summary.query_kind)
    )
    mask = _head_mask(model, [(0, 1), (1, 0)], "cpu")
    assert mask.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_pca_and_direction_geometry_recover_a_known_count_axis():
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(1, 7), 8)
    values = rng.normal(scale=0.01, size=(len(labels), 12))
    values[:, 0] += labels * 2.0
    _, coordinates, variance = _fit_pca(values, 6)
    assert coordinates.shape == (len(labels), 6)
    assert variance[:6].sum() <= 1.0 + 1e-8

    states = {SITE_NONTHINKING_FINAL: [values, values + 0.1]}
    eval_values = values + rng.normal(scale=0.005, size=values.shape)
    directions, geometry, centroids = fit_count_directions(
        states,
        {SITE_NONTHINKING_FINAL: labels},
        {SITE_NONTHINKING_FINAL: [eval_values, eval_values + 0.1]},
        {SITE_NONTHINKING_FINAL: labels},
    )
    assert (SITE_NONTHINKING_FINAL, 0, "adjacent_mean") in directions
    assert geometry.projection_r2_heldout.max() > 0.99
    assert (SITE_NONTHINKING_FINAL, 0, 6) in centroids


def test_v10_main_matches_requested_v2_architecture_and_count_range():
    cfg = preset_config("main")
    assert (cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.n_inner) == (4, 4, 256, 1024)
    assert (cfg.seq_len, cfg.count_min, cfg.count_max) == (256, 1, 30)
    assert cfg.early_stop_patience == 0
    assert cfg.max_render_len == 322
    assert cfg.n_positions >= cfg.max_render_len


def test_layer_matched_controls_preserve_the_ranked_layer_sequence():
    ranking = [(2, 1), (3, 0), (2, 3), (0, 2), (3, 2), (0, 1)]
    low = _layer_matched_order(ranking, reverse_within_layer=True)
    random_order = _layer_matched_order(ranking, seed=19)
    assert [layer for layer, _ in low] == [layer for layer, _ in ranking]
    assert [layer for layer, _ in random_order] == [layer for layer, _ in ranking]
    assert sorted(low) == sorted(ranking)
    assert sorted(random_order) == sorted(ranking)


def test_position_local_ablation_runs_at_trace_and_answer_queries_without_leaking_hooks():
    cfg, vocab, model = tiny_setup()
    example = make_example(cfg, vocab, random.Random(7), count=4)
    item = render(example, vocab, "thinking")
    ids = torch.tensor([item.input_ids], dtype=torch.long)
    mask = torch.ones_like(ids)
    baseline = model(input_ids=ids, attention_mask=mask).logits
    local = _forward_with_position_local_head_ablation(
        model,
        ids,
        mask,
        [(0, 0)],
        [_local_query_positions(item, "trace_index_tokens")],
    ).logits
    restored = model(input_ids=ids, attention_mask=mask).logits
    assert local.shape == baseline.shape
    assert not torch.allclose(local, baseline)
    assert torch.allclose(restored, baseline)
    assert _local_query_positions(item, "ans_token") == [item.spans.ans_pos]
