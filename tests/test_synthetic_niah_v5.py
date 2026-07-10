from __future__ import annotations

import random

from synthetic_niah_v5.data import (
    IGNORE_INDEX,
    make_example,
    nonthinking_query,
    render_nonthinking,
    render_thinking,
    thinking_query,
    trace_tokens_for_example,
)
from synthetic_niah_v5.evaluation import parse_thinking_generation, trace_metric_dict
from synthetic_niah_v5.vocab import Vocab, count_token


def test_v5_tokenizer_round_trip():
    vocab = Vocab.build()
    tokens = ["<BOS>", "<THINK_ON>", "<N0>", "<A>", "<Think/>", "</Think>", "<C3>", "<EOS>"]
    assert vocab.decode(vocab.encode(tokens)) == tokens
    assert len(vocab.id_to_token) == 90
    assert vocab.think_on_id != vocab.think_off_id


def test_v5_base_generator_count_correctness():
    ex = make_example(32, 5, random.Random(0))
    assert ex.count == 5
    assert len(ex.seq_tokens) == 32
    assert len(ex.needle_positions) == len(ex.needle_markers) == ex.count
    assert ex.needle_positions == sorted(ex.needle_positions)
    assert sum(tok in {f"<{chr(ord('A') + i)}>" for i in range(10)} for tok in ex.seq_tokens) == ex.count
    for pos, marker in zip(ex.needle_positions, ex.needle_markers):
        assert ex.seq_tokens[pos] == marker


def test_v5_thinking_render_and_mask_marker_only():
    vocab = Vocab.build()
    ex = make_example(16, 3, random.Random(1))
    rendered = render_thinking(ex, vocab)
    assert rendered.token_strs[0] == "<BOS>"
    assert rendered.token_strs[rendered.spans.mode_pos] == "<THINK_ON>"
    assert rendered.spans.seq_start == 2
    assert rendered.token_strs[rendered.spans.think_open_pos] == "<Think/>"
    assert rendered.token_strs[rendered.spans.trace_token_positions[0] : rendered.spans.trace_token_positions[-1] + 1] == trace_tokens_for_example(ex)
    assert rendered.token_strs[rendered.spans.think_close_pos] == "</Think>"
    assert rendered.token_strs[rendered.spans.count_pos] == count_token(ex.count)
    assert all(label == IGNORE_INDEX for label in rendered.labels[: rendered.spans.think_open_pos + 1])
    assert rendered.labels[rendered.spans.trace_token_positions[0]] == vocab.token_to_id[ex.needle_markers[0]]
    assert rendered.labels[rendered.spans.think_close_pos] == vocab.think_close_id
    assert rendered.labels[rendered.spans.count_pos] == vocab.count_id(ex.count)


def test_v5_nonthinking_has_explicit_mode_and_supervises_close():
    vocab = Vocab.build()
    ex = make_example(16, 4, random.Random(2))
    rendered = render_nonthinking(ex, vocab)
    assert rendered.token_strs[rendered.spans.mode_pos] == "<THINK_OFF>"
    assert rendered.token_strs[rendered.spans.think_open_pos] == "<Think/>"
    assert rendered.token_strs[rendered.spans.think_close_pos] == "</Think>"
    assert rendered.labels[rendered.spans.think_open_pos] == IGNORE_INDEX
    assert rendered.labels[rendered.spans.think_close_pos] == vocab.think_close_id
    assert rendered.labels[rendered.spans.count_pos] == vocab.count_id(ex.count)


def test_v5_query_construction():
    vocab = Vocab.build()
    ex = make_example(8, 2, random.Random(3))
    assert vocab.decode(thinking_query(ex, vocab)) == ["<BOS>", "<THINK_ON>"] + ex.seq_tokens + ["<Think/>"]
    assert vocab.decode(nonthinking_query(ex, vocab)) == ["<BOS>", "<THINK_OFF>"] + ex.seq_tokens + ["<Think/>"]
    assert thinking_query(ex, vocab) != nonthinking_query(ex, vocab)


def test_v5_eval_parser_edge_cases():
    ok = parse_thinking_generation(["<A>", "<B>", "</Think>", "<C2>", "<EOS>"], ["<A>", "<B>"])
    assert ok.final_count == 2
    assert not ok.invalid_count
    assert trace_metric_dict(ok, ["<A>", "<B>"])["trace_exact"] == 1.0

    missing = parse_thinking_generation(["<A>", "<B>", "<C2>"], ["<A>", "<B>"])
    assert missing.missing_close
    assert missing.invalid_count

    premature = parse_thinking_generation(["</Think>", "<C3>"], ["<A>", "<B>", "<C>"])
    assert premature.premature_close
    assert premature.final_count == 3

    invalid = parse_thinking_generation(["<A>", "</Think>", "<N0>"], ["<A>"])
    assert invalid.invalid_count

    dup = parse_thinking_generation(["<A>", "<A>", "</Think>", "<C1>"], ["<A>"])
    assert dup.duplicate_rate > 0.0
