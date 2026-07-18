from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from synthetic_counting_v11.config import preset_config
from synthetic_counting_v11.data import (
    IGNORE_INDEX,
    Vocab,
    balanced_examples,
    count_sampling_probabilities,
    make_example,
    render,
)
from synthetic_counting_v11.model import build_model


ROOT = Path(__file__).resolve().parents[1]


def test_v15_v17_restore_v10_width_and_completion_only_loss():
    for version in ("v15", "v16", "v17"):
        cfg = preset_config(version, "main", device="cpu")
        assert (cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.n_inner) == (4, 4, 256, 1024)
        assert cfg.loss_scope == "completion"
        assert "completion-only" in cfg.to_dict()["training_objective"]


def test_v15_and_v16_have_exactly_four_rope_rpe_mode_variants():
    for version in ("v15", "v16"):
        cfg = preset_config(version, "main")
        assert cfg.position_encodings == ("rope", "rpe")
        assert cfg.model_variants == (
            ("rope", "nonthinking"),
            ("rope", "thinking"),
            ("rpe", "nonthinking"),
            ("rpe", "thinking"),
        )


def test_v16_config_round_trip_is_stable_for_checkpoint_resume():
    cfg = preset_config("v16", "debug", device="cpu")
    serialized = json.loads(json.dumps(cfg.to_dict()))

    assert serialized == cfg.to_dict()
    assert serialized["target_characters"] == list(cfg.target_characters)


def test_v15_v17_label_boundaries_match_v10_completion_only_training():
    old_cfg = preset_config("v14", "debug")
    old_vocab = Vocab.build(old_cfg)
    old_item = render(make_example(old_cfg, old_vocab, random.Random(1), count=2), old_vocab, "nonthinking")
    assert old_item.labels[old_item.spans.prompt_start] == IGNORE_INDEX
    assert sum(label != IGNORE_INDEX for label in old_item.labels) == 2

    new_cfg = preset_config("v15", "debug")
    new_vocab = Vocab.build(new_cfg)
    example = make_example(new_cfg, new_vocab, random.Random(1), count=2)
    nonthinking = render(example, new_vocab, "nonthinking")
    thinking = render(example, new_vocab, "thinking")

    assert nonthinking.labels[nonthinking.spans.prompt_start] == IGNORE_INDEX
    assert sum(label != IGNORE_INDEX for label in nonthinking.labels) == 2
    first_trace_index = thinking.spans.trace_index_positions[0]
    assert all(label == IGNORE_INDEX for label in thinking.labels[:first_trace_index])
    assert all(label != IGNORE_INDEX for label in thinking.labels[first_trace_index:])


def test_v15_uses_shakespeare_haystack_with_inserted_marker_needles():
    cfg = preset_config("v15", "debug")
    vocab = Vocab.build(cfg)
    example = make_example(cfg, vocab, random.Random(2), count=4)

    assert cfg.noise_source == "shakespeare_char"
    assert cfg.task_type == "inserted_marker"
    assert len(example.needle_positions) == 4
    assert all(example.seq_tokens[pos] in vocab.markers for pos in example.needle_positions)
    assert any(token.startswith("<CH_") for token in example.seq_tokens)


def test_v16_counts_native_target_characters_and_names_the_target_in_the_prompt():
    cfg = preset_config("v16", "main")
    vocab = Vocab.build(cfg)
    examples = balanced_examples(cfg, vocab, 1, seed=3)

    assert sorted(example.count for example in examples) == list(range(1, 31))
    for example in examples:
        assert example.target_token is not None
        assert example.target_character in cfg.target_characters
        positions = [
            index for index, token in enumerate(example.seq_tokens) if token == example.target_token
        ]
        assert positions == example.needle_positions
        assert len(positions) == example.count
        assert all(marker == example.target_token for marker in example.needle_markers)

    item = render(examples[0], vocab, "thinking")
    assert item.tokens[1:4] == ["<CountChar>", examples[0].target_token, "<Sep>"]
    assert item.spans.prompt_start == 4
    first_trace_index = item.spans.trace_index_positions[0]
    assert all(label == IGNORE_INDEX for label in item.labels[:first_trace_index])
    assert all(label != IGNORE_INDEX for label in item.labels[first_trace_index:])


def test_v17_power_and_exponential_samplers_are_decreasing_long_tails():
    for sampler in ("power", "exponential"):
        cfg = preset_config("v17", "main", count_sampling=sampler)
        probabilities = count_sampling_probabilities(cfg)
        counts = np.arange(cfg.count_min, cfg.count_max + 1)
        assert np.isclose(probabilities.sum(), 1.0)
        assert np.all(np.diff(probabilities) < 0)
        assert float(np.sum(counts * probabilities)) < 10.0
        assert probabilities[0] > probabilities[-1] * 20


def test_v15_v17_models_forward_at_d256_for_every_requested_position_encoding():
    for version in ("v15", "v16", "v17"):
        cfg = preset_config(version, "debug", device="cpu")
        vocab = Vocab.build(cfg)
        item = render(make_example(cfg, vocab, random.Random(4), count=2), vocab, "thinking")
        ids = torch.tensor([item.input_ids], dtype=torch.long)
        for position_encoding in cfg.position_encodings:
            model = build_model(cfg, vocab, position_encoding, "cpu").eval()
            with torch.no_grad():
                output = model(ids)
            assert output.logits.shape == (1, len(item.input_ids), len(vocab.id_to_token))
            assert model.config.n_embd == 256
            assert model.layers[0].attention.head_dim == 64


def test_v15_v17_notebooks_are_colab_ready_and_mount_drive_first():
    for version in range(15, 18):
        path = ROOT / "notebooks" / f"Trace_Count_v{version}_Colab.ipynb"
        assert path.exists()
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        first = "".join(code_cells[0].get("source", []))
        assert "drive.mount" in first
        assert f"synthetic_counting_v{version}.run_v{version}" in path.read_text(encoding="utf-8")
