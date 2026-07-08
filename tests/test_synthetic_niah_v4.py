from __future__ import annotations

import random

from synthetic_niah_v4.config import build_config
from synthetic_niah_v4.data import make_example
from synthetic_niah_v4.render import (
    IGNORE_INDEX,
    non_thinking_eval_prefix,
    render_non_thinking,
    render_thinking,
    thinking_generation_prefix,
    thinking_oracle_trace_prefix,
    trace_tokens_for_example,
)
from synthetic_niah_v4.vocab import Vocab, count_token


def test_v4_vocab_is_v2_style():
    vocab = Vocab.build()
    assert len(vocab.id_to_token) == 90
    assert vocab.id_to_token[:6] == ["<PAD>", "<BOS>", "<EOS>", "<Ans>", "<Think/>", "</Think>"]
    assert vocab.decode(vocab.encode(["<BOS>", "<N0>", "<A>", "<Ans>", "<3>", "<EOS>"])) == [
        "<BOS>",
        "<N0>",
        "<A>",
        "<Ans>",
        "<3>",
        "<EOS>",
    ]


def test_v4_generator_count_correctness():
    ex = make_example(seq_len=32, count=4, rng=random.Random(0))
    assert ex.count == 4
    assert len(ex.seq_tokens) == 32
    assert ex.needle_positions == sorted(ex.needle_positions)
    assert len(ex.needle_positions) == len(ex.needle_markers) == 4
    for pos, marker in zip(ex.needle_positions, ex.needle_markers):
        assert ex.seq_tokens[pos] == marker


def test_v4_non_thinking_render_masks_prompt_and_ans():
    vocab = Vocab.build()
    ex = make_example(seq_len=16, count=3, rng=random.Random(1))
    rendered = render_non_thinking(ex, vocab)
    assert rendered.token_strs[rendered.spans.ans_pos] == "<Ans>"
    assert rendered.token_strs[rendered.spans.final_count_pos] == count_token(ex.count)
    assert all(label == IGNORE_INDEX for label in rendered.labels[: rendered.spans.final_count_pos])
    assert rendered.labels[rendered.spans.final_count_pos] == vocab.count_id(ex.count)
    assert rendered.labels[rendered.spans.eos_pos] == vocab.eos_id


def test_v4_thinking_render_supervises_trace_then_answer():
    vocab = Vocab.build()
    ex = make_example(seq_len=16, count=3, rng=random.Random(2))
    rendered = render_thinking(ex, vocab)
    trace = trace_tokens_for_example(ex)
    assert rendered.token_strs[rendered.spans.think_open_pos] == "<Think/>"
    assert rendered.token_strs[rendered.spans.trace_token_positions[0] : rendered.spans.trace_token_positions[-1] + 1] == trace
    assert rendered.token_strs[rendered.spans.think_close_pos] == "</Think>"
    assert rendered.token_strs[rendered.spans.ans_pos] == "<Ans>"
    assert rendered.token_strs[rendered.spans.final_count_pos] == count_token(ex.count)
    assert rendered.labels[rendered.spans.trace_token_positions[0]] == vocab.token_to_id[trace[0]]
    assert rendered.labels[rendered.spans.think_close_pos] == vocab.think_close_id
    assert rendered.labels[rendered.spans.ans_pos] == vocab.ans_id
    assert rendered.labels[rendered.spans.final_count_pos] == vocab.count_id(ex.count)


def test_v4_eval_prefixes():
    vocab = Vocab.build()
    ex = make_example(seq_len=8, count=2, rng=random.Random(3))
    assert vocab.decode(non_thinking_eval_prefix(ex, vocab)) == ["<BOS>"] + ex.seq_tokens + ["<Ans>"]
    assert vocab.decode(thinking_generation_prefix(ex, vocab)) == ["<BOS>"] + ex.seq_tokens + ["<Think/>"]
    assert vocab.decode(thinking_oracle_trace_prefix(ex, vocab)) == (
        ["<BOS>"] + ex.seq_tokens + ["<Think/>"] + trace_tokens_for_example(ex) + ["</Think>", "<Ans>"]
    )


def test_v4_debug_config_matches_v2_gpt2_shape():
    class Args:
        preset = "debug"
        seq_len = None
        count_min = None
        count_max = None
        out_root = None
        run_name = None
        device = "cpu"
        eval_examples_per_count = None
        probe_examples_per_count = None
        steering_examples_per_count = None
        train_steps = None
        seeds = "1234"

    cfg = build_config(Args())
    assert cfg.model.n_layer == 2
    assert cfg.model.n_head == 2
    assert cfg.model.n_embd == 128
    assert cfg.model.n_positions >= 128
    assert cfg.seeds == [1234]
