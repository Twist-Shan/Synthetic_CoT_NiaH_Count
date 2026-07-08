from __future__ import annotations

import random

from synthetic_counting_v6.data import make_example, validate_example
from synthetic_counting_v6.generation import parse_generated_suffix, trace_metrics
from synthetic_counting_v6.render import IGNORE_INDEX, render_non_thinking, render_thinking_sep_trace, trace_tokens_for_example
from synthetic_counting_v6.vocab import MARKER_TOKENS, NUMBER_TOKENS, Vocab


def _example():
    rng = random.Random(123)
    return make_example(seq_len=32, count=3, rng=rng, seed=123, example_id="unit")


def test_v6_vocab_has_sep_and_expected_size():
    vocab = Vocab.build()
    assert len(vocab.id_to_token) == 91
    assert vocab.id_to_token[vocab.sep_id] == "<Sep>"
    assert all(token in vocab.token_to_id for token in NUMBER_TOKENS)


def test_v6_base_generation_validates_marker_count():
    ex = _example()
    validate_example(ex)
    assert sum(token in MARKER_TOKENS for token in ex.seq_tokens) == ex.count
    assert len(ex.needle_positions) == len(ex.needle_markers) == ex.count


def test_v6_thinking_trace_uses_sep_not_numeric_indices():
    ex = _example()
    trace = trace_tokens_for_example(ex)
    assert trace == ["<Sep>", ex.needle_markers[0], "<Sep>", ex.needle_markers[1], "<Sep>", ex.needle_markers[2]]
    assert not any(token in NUMBER_TOKENS for token in trace)


def test_v6_non_thinking_labels_only_answer_and_eos():
    vocab = Vocab.build()
    ex = _example()
    rendered = render_non_thinking(ex, vocab)
    supervised = [idx for idx, label in enumerate(rendered.labels) if label != IGNORE_INDEX]
    assert supervised == [rendered.spans.final_count_pos, rendered.spans.eos_pos]
    assert rendered.token_strs[rendered.spans.ans_pos] == "<Ans>"
    assert rendered.token_strs[rendered.spans.final_count_pos] == f"<{ex.count}>"


def test_v6_thinking_labels_start_at_first_sep():
    vocab = Vocab.build()
    ex = _example()
    rendered = render_thinking_sep_trace(ex, vocab)
    assert rendered.token_strs[rendered.spans.think_open_pos] == "<Think/>"
    assert rendered.labels[rendered.spans.think_open_pos] == IGNORE_INDEX
    assert rendered.token_strs[rendered.spans.sep_positions[0]] == "<Sep>"
    assert rendered.labels[rendered.spans.sep_positions[0]] == vocab.sep_id
    assert len(rendered.spans.sep_positions) == ex.count
    assert rendered.token_strs[rendered.spans.trace_token_positions[0] : rendered.spans.trace_token_positions[-1] + 1] == trace_tokens_for_example(ex)
    assert not any(token in NUMBER_TOKENS for token in rendered.token_strs[rendered.spans.trace_token_positions[0] : rendered.spans.trace_token_positions[-1] + 1])


def test_v6_generation_parser_trace_metrics():
    ex = _example()
    good = trace_tokens_for_example(ex) + ["</Think>", "<Ans>", f"<{ex.count}>", "<EOS>"]
    parsed = parse_generated_suffix(good, ex)
    assert parsed.pred_count == ex.count
    metrics = trace_metrics(parsed, ex)
    assert metrics["trace_exact_match_rate"] == 1.0
    assert metrics["trace_delimiter_count_accuracy"] == 1.0

    bad = ["<Sep>", ex.needle_markers[0], "</Think>", "<Ans>", "<10>", "<EOS>"]
    parsed_bad = parse_generated_suffix(bad, ex)
    bad_metrics = trace_metrics(parsed_bad, ex)
    assert parsed_bad.pred_count == 10
    assert bad_metrics["premature_close_rate"] == 1.0
    assert bad_metrics["trace_exact_match_rate"] == 0.0

