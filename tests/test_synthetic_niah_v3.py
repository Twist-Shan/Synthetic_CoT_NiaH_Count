from __future__ import annotations

import random

import numpy as np
import torch

from synthetic_niah_v3.attention import _entropy
from synthetic_niah_v3.data import balanced_examples, make_example
from synthetic_niah_v3.objectives import build_training_weights
from synthetic_niah_v3.render import render_non_thinking, render_thinking, trace_tokens_for_example
from synthetic_niah_v3.model import make_model
from synthetic_niah_v3.train import load_checkpoint, save_checkpoint
from synthetic_niah_v3.trace_parse import parse_thinking_generation
from synthetic_niah_v3.vocab import Vocab


def test_vocab_and_generation_contracts():
    vocab = Vocab.build()
    assert len(vocab.id_to_token) == 90
    assert len(vocab.encode(["<10>"])) == 1
    rng = random.Random(0)
    ex = make_example(256, 10, rng)
    assert len(ex.seq_tokens) == 256
    assert ex.count == len(ex.needle_positions) == len(ex.needle_markers)
    assert ex.needle_positions == sorted(ex.needle_positions)


def test_rendering_and_objective_masks():
    vocab = Vocab.build()
    ex = make_example(32, 3, random.Random(1))
    non = render_non_thinking(ex, vocab)
    think = render_thinking(ex, vocab)
    assert non.token_strs.count("<Ans>") == 1
    assert non.token_strs[non.spans.final_count_pos] == "<3>"
    assert think.token_strs[think.spans.think_open_pos] == "<Think/>"
    assert think.token_strs[think.spans.think_close_pos] == "</Think>"
    assert think.token_strs[think.spans.trace_token_positions[0] : think.spans.trace_token_positions[-1] + 1] == trace_tokens_for_example(ex)
    non_weights = build_training_weights(non.tokens, non.spans, "non_thinking")
    assert torch.nonzero(non_weights).flatten().tolist() == [non.spans.ans_pos, non.spans.final_count_pos]
    think_weights = build_training_weights(think.tokens, think.spans, "thinking")
    assert think_weights[think.spans.seq_start : think.spans.seq_end_exclusive].sum().item() == 0.0
    assert think_weights[think.spans.think_open_pos].item() == 1.0
    assert think_weights[think.spans.ans_pos].item() == 1.0


def test_eval_parser_and_attention_metric_sanity():
    parsed = parse_thinking_generation(["<1>", "<A>", "<2>", "<B>", "</Think>", "<Ans>", "<2>", "<EOS>"])
    assert not parsed.invalid
    assert parsed.final_count == 2
    assert _entropy(np.eye(3)[0]) >= 0.0


def test_checkpoint_save_load_roundtrip(tmp_path):
    cfg = {
        "vocab_size": 90,
        "n_layers": 1,
        "n_heads": 1,
        "d_model": 16,
        "d_mlp": 32,
        "dropout": 0.0,
        "context_len": 64,
    }
    model = make_model(cfg, "cpu")
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path, {"config": cfg, "note": "metadata is intentionally present"})
    other = make_model(cfg, "cpu")
    load_checkpoint(other, path, "cpu")
