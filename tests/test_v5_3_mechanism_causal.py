from __future__ import annotations

import random

import torch

from synthetic_counting_extensions.v5_3_mechanism_causal import (
    _head_mask,
    _normalized_recovery,
    collect_attention_signatures,
    delete_last_needle,
    marker_identity_corruption,
    render_trace_override,
    run_patching,
    run_progress_state_transplant,
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
    )


def test_head_mask_uses_zero_based_layer_and_head_indices():
    vocab = Vocab.build(include_trace_indices=True)
    mask = _head_mask(tiny_model(vocab), "cpu", [(0, 1), (1, 0)])
    assert mask.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_marker_identity_corruption_preserves_count_and_positions():
    ex = make_example(24, 4, random.Random(7))
    corrupt = marker_identity_corruption(ex, 3)
    assert corrupt.count == ex.count
    assert corrupt.needle_positions == ex.needle_positions
    assert corrupt.needle_markers[:2] == ex.needle_markers[:2]
    assert corrupt.needle_markers[2] != ex.needle_markers[2]
    assert corrupt.seq_tokens[ex.needle_positions[2]] == corrupt.needle_markers[2]


def test_delete_last_needle_reduces_count_without_changing_length():
    ex = make_example(24, 4, random.Random(8))
    corrupt = delete_last_needle(ex)
    assert corrupt.count == 3
    assert len(corrupt.seq_tokens) == len(ex.seq_tokens)
    assert corrupt.needle_positions == ex.needle_positions[:-1]
    assert corrupt.seq_tokens[ex.needle_positions[-1]] != ex.needle_markers[-1]


def test_trace_override_uses_index_marker_pairs_and_close_query():
    vocab = Vocab.build(include_trace_indices=True)
    ex = make_example(16, 3, random.Random(9))
    prefix, query_pos = render_trace_override(ex, vocab, 2)
    tokens = vocab.decode(prefix)
    assert tokens[-6:] == ["<Think/>", "<I1>", ex.needle_markers[0], "<I2>", ex.needle_markers[1], "</Think>"]
    assert query_pos == len(tokens) - 1


def test_normalized_recovery_has_expected_endpoints():
    assert _normalized_recovery(4.0, -2.0, -2.0) == 0.0
    assert _normalized_recovery(4.0, -2.0, 4.0) == 1.0
    assert _normalized_recovery(4.0, -2.0, 1.0) == 0.5


def test_tiny_model_attention_and_patch_hooks_run_end_to_end():
    vocab = Vocab.build(include_trace_indices=True)
    model = tiny_model(vocab)
    ex = make_example(8, 4, random.Random(10))
    detail, summary = collect_attention_signatures(model, vocab, [ex], "cpu")
    assert not detail.empty
    assert {"final_count_query", "trace_marker_query", "successor_query"}.issubset(set(summary.query_kind))

    groups = {
        "targeted_top1": [(0, 0)],
        "targeted_top2": [(0, 0), (0, 1)],
        "direct_broad_top1": [(1, 0)],
        "direct_broad_top2": [(1, 0), (1, 1)],
        "trace_readout_top2": [(0, 0), (1, 0)],
        "random_2": [(0, 1), (1, 1)],
        "all_heads": [(0, 0), (0, 1), (1, 0), (1, 1)],
    }
    patch_rows, patch_summary = run_patching(model, vocab, [ex], groups, "cpu")
    assert not patch_rows.empty
    assert {"retrieval_identity", "nonthinking_count_readout", "thinking_count_readout"}.issubset(set(patch_summary.experiment))
    assert patch_rows.normalized_recovery.notna().any()

    progress_rows, progress_summary = run_progress_state_transplant(model, vocab, [ex], "cpu")
    assert not progress_rows.empty
    assert set(progress_summary.donor_offset) == {-1, 0, 1}
