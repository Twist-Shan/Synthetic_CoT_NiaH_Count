from __future__ import annotations

import random

import numpy as np
import torch

from synthetic_counting_extensions.v5_4_count_state_causal import (
    Direction,
    SITE_NONTHINKING,
    SITE_THINKING_FIXED,
    _fit_direction,
    capture_block_residuals,
    estimate_directions,
    forward_with_residual_intervention,
    make_site_batch,
    render_fixed_trace_prefix,
)
from synthetic_niah_v5.data import make_example
from synthetic_niah_v5.model import make_model
from synthetic_niah_v5.vocab import Vocab


def tiny_model(vocab: Vocab):
    return make_model(
        {
            "vocab_size": len(vocab.id_to_token),
            "bos_token_id": vocab.bos_id,
            "eos_token_id": vocab.eos_id,
            "pad_token_id": vocab.pad_id,
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 16,
            "n_inner": 32,
            "n_positions": 64,
            "n_ctx": 64,
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
        },
        "cpu",
    ).eval()


def test_fixed_trace_control_keeps_close_position_constant_across_prompt_counts():
    vocab = Vocab.build(include_trace_indices=True)
    ex2 = make_example(12, 2, random.Random(1))
    ex8 = make_example(12, 8, random.Random(2))
    ids2, pos2 = render_fixed_trace_prefix(ex2, vocab, 5)
    ids8, pos8 = render_fixed_trace_prefix(ex8, vocab, 5)
    assert len(ids2) == len(ids8)
    assert pos2 == pos8
    tokens2 = vocab.decode(ids2)
    tokens8 = vocab.decode(ids8)
    assert tokens2[-11:] == tokens8[-11:]
    assert tokens2[-1] == "</Think>"


def test_adjacent_direction_recovers_known_linear_count_axis():
    rng = np.random.default_rng(7)
    y = np.repeat(np.arange(1, 11), 8)
    x = rng.normal(scale=0.01, size=(len(y), 6))
    x[:, 0] += y * 2.0
    direction, summary = _fit_direction(
        x,
        y,
        method="adjacent_mean",
        rng=rng,
        source_site=SITE_NONTHINKING,
        layer=0,
    )
    assert isinstance(direction, Direction)
    assert direction.vector[0] > 0.99
    assert direction.step_size > 1.9
    assert summary["adjacent_delta_cosine_mean"] > 0.99
    assert summary["projection_r2_train"] > 0.99


def test_direction_estimation_returns_count_specific_centroids():
    y = np.repeat(np.arange(1, 11), 3)
    x = np.stack([np.array([float(count), float(count * count), 1.0]) for count in y])
    states = {
        SITE_NONTHINKING: [x],
        SITE_THINKING_FIXED: [x + np.array([0.0, 0.0, 2.0])],
    }
    directions, centroids, geometry, cross = estimate_directions(states, y, seed=11)
    assert (SITE_NONTHINKING, 0, "ridge") in directions
    assert np.allclose(centroids[(SITE_NONTHINKING, 0, 4)], np.array([4.0, 16.0, 1.0]))
    assert len(geometry) == 6
    assert len(cross) == 2


def test_block_residual_capture_and_additive_intervention_run_end_to_end():
    vocab = Vocab.build(include_trace_indices=True)
    model = tiny_model(vocab)
    examples = [make_example(8, 2, random.Random(3)), make_example(8, 4, random.Random(4))]
    batch = make_site_batch(examples, vocab, SITE_NONTHINKING, "cpu", fixed_trace_count=3)
    baseline_logits, states = capture_block_residuals(model, batch)
    assert len(states) == 2
    assert states[0].shape == (2, 16)
    zero = torch.zeros(16)
    steered_logits = forward_with_residual_intervention(model, batch, layer=0, additive=zero)
    assert torch.allclose(baseline_logits, steered_logits, atol=1e-6)


def test_position_matched_thinking_batch_has_one_query_position():
    vocab = Vocab.build(include_trace_indices=True)
    examples = [make_example(12, count, random.Random(10 + count)) for count in (1, 5, 10)]
    batch = make_site_batch(examples, vocab, SITE_THINKING_FIXED, "cpu", fixed_trace_count=4)
    assert len(set(batch.query_positions.tolist())) == 1
    assert batch.input_ids.shape[0] == 3
