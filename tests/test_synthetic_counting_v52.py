from __future__ import annotations

from dataclasses import asdict

import torch

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import (
    V20Example,
    V20Vocab,
    character_token,
    collate_v20_loss_weights,
    render_v20,
)
from synthetic_counting_v20.training import component_normalized_task_output_loss
from synthetic_counting_v51.config import preset_config as preset_v51
from synthetic_counting_v52.config import preset_config as preset_v52
from synthetic_counting_v52.preflight import run_preflight


def _example() -> V20Example:
    marker = character_token("a")
    return V20Example(
        example_kind="counting_task",
        seq_tokens=[marker, marker, marker],
        corpus_region="train",
        corpus_start=0,
        corpus_end=3,
        prompt_sha256="count-3",
        set_id="set_000",
        needle_characters=("a", "b", "c"),
        rendered_set_order=("a", "b", "c"),
        needle_positions=(0, 1, 2),
        needle_markers=(marker, marker, marker),
        count=3,
        per_character_counts=(3, 0, 0),
    )


def test_v52_changes_only_v51_trace_weight() -> None:
    baseline = preset_v51("main", device="cpu")
    candidate = preset_v52("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "task_output_trace_weight"}
    assert candidate.task_output_count_weight == 8.0
    assert candidate.task_output_trace_weight == 16.0
    assert candidate.task_output_structure_weight == 16.0
    assert candidate.train_steps == 10_000
    assert config_from_dict(candidate.to_dict()) == candidate


def test_v52_trace_serialization_is_identical_to_v51() -> None:
    old = preset_v51("debug", device="cpu")
    new = preset_v52("debug", device="cpu")
    old_vocab = V20Vocab.build(old, "abc xyz\n")
    new_vocab = V20Vocab.build(new, "abc xyz\n")
    assert render_v20(_example(), old_vocab, "thinking").tokens == render_v20(
        _example(), new_vocab, "thinking"
    ).tokens


def test_v52_adds_exactly_one_extra_trace_coefficient() -> None:
    old = preset_v51("debug", device="cpu", max_steps_for_language_pred=1)
    new = preset_v52("debug", device="cpu", max_steps_for_language_pred=1)
    vocab = V20Vocab.build(new, "abc xyz\n")
    item = render_v20(_example(), vocab, "thinking")
    weights = collate_v20_loss_weights([item], new, "cpu", step=2)
    token_losses = torch.arange(1, len(item.tokens), dtype=torch.float32).unsqueeze(0)
    active = torch.ones_like(token_losses, dtype=torch.bool)
    old_loss, old_regions = component_normalized_task_output_loss(
        token_losses, active, weights, [item], old
    )
    new_loss, new_regions = component_normalized_task_output_loss(
        token_losses, active, weights, [item], new
    )
    assert old_regions.keys() == new_regions.keys()
    for name in old_regions:
        torch.testing.assert_close(old_regions[name], new_regions[name])
    torch.testing.assert_close(
        new_loss - old_loss,
        8.0 * new_regions["trace"],
    )


def test_v52_preflight_is_exposed() -> None:
    assert callable(run_preflight)
