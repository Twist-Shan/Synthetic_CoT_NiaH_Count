from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest
import torch

from synthetic_counting_v20.config import config_from_dict, default_run_name
from synthetic_counting_v20.data import (
    V20Example,
    V20Vocab,
    character_token,
    collate_v20_loss_weights,
    render_v20,
)
from synthetic_counting_v20.training import component_normalized_task_output_loss
from synthetic_counting_v47.config import preset_config as preset_v47
from synthetic_counting_v49.behavior_gate import evaluate_behavior_gate
from synthetic_counting_v49.config import preset_config as preset_v49
from synthetic_counting_v49.preflight import run_preflight


def _example(count: int) -> V20Example:
    sequence = [character_token("a")] * count
    return V20Example(
        example_kind="counting_task",
        seq_tokens=sequence,
        corpus_region="train",
        corpus_start=0,
        corpus_end=len(sequence),
        prompt_sha256=f"count-{count}",
        set_id="set_000",
        needle_characters=("a", "b", "c"),
        rendered_set_order=("a", "b", "c"),
        needle_positions=tuple(range(count)),
        needle_markers=tuple(sequence),
        count=count,
        per_character_counts=(count, 0, 0),
    )


def test_v49_changes_only_v47_loss_partition() -> None:
    baseline = preset_v47("main", device="cpu")
    candidate = preset_v49("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "task_output_trace_delimiters_as_structure"}
    assert candidate.task_output_trace_delimiters_as_structure is True
    assert (candidate.n_layer, candidate.n_head, candidate.n_embd, candidate.n_inner) == (
        4,
        4,
        256,
        1024,
    )
    assert candidate.train_steps == 10_000
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )
    assert "trace-markers_grammar-delimiters" in default_run_name(candidate)
    assert config_from_dict(candidate.to_dict()) == candidate


def test_v49_rejects_historical_separator_partition() -> None:
    with pytest.raises(
        ValueError,
        match="requires task_output_trace_delimiters_as_structure=True",
    ):
        replace(
            preset_v49("main", device="cpu"),
            task_output_trace_delimiters_as_structure=False,
        ).validate()


def test_v49_trace_serialization_is_identical_to_v47() -> None:
    old = preset_v47("debug", device="cpu")
    new = preset_v49("debug", device="cpu")
    old_vocab = V20Vocab.build(old, "abc xyz\n")
    new_vocab = V20Vocab.build(new, "abc xyz\n")
    old_item = render_v20(_example(3), old_vocab, "thinking")
    new_item = render_v20(_example(3), new_vocab, "thinking")
    assert old_item.tokens == new_item.tokens
    assert old_item.tokens[-11:] == [
        "<Think>",
        "<Sep>",
        character_token("a"),
        "<Sep>",
        character_token("a"),
        "<Sep>",
        character_token("a"),
        "</Think>",
        "<Ans>",
        "<3>",
        "<EOS>",
    ]


def test_v49_balances_separator_continue_and_stop_in_structure() -> None:
    old = preset_v47("debug", device="cpu", max_steps_for_language_pred=1)
    new = preset_v49("debug", device="cpu", max_steps_for_language_pred=1)
    vocab = V20Vocab.build(new, "abc xyz\n")
    item = render_v20(_example(3), vocab, "thinking")
    assert item.spans is not None and item.spans.think_pos is not None
    weights = collate_v20_loss_weights([item], new, "cpu", step=2)
    token_losses = torch.zeros((1, len(item.tokens) - 1), dtype=torch.float32)
    active = torch.ones_like(token_losses, dtype=torch.bool)

    count_positions = set(item.spans.count_positions)
    delimiter_positions = {
        position
        for group in item.spans.trace_index_token_groups
        for position in group
    }
    marker_positions = set(item.spans.trace_marker_positions)
    output_positions = set(range(item.spans.think_pos, len(item.tokens)))
    other_structure_positions = (
        output_positions - count_positions - delimiter_positions - marker_positions
    )
    for position in count_positions:
        token_losses[0, position - 1] = 13.0
    for position in delimiter_positions:
        token_losses[0, position - 1] = 5.0
    for position in marker_positions:
        token_losses[0, position - 1] = 7.0
    for position in other_structure_positions:
        token_losses[0, position - 1] = 11.0

    new_loss, new_regions = component_normalized_task_output_loss(
        token_losses, active, weights, [item], new
    )
    old_loss, old_regions = component_normalized_task_output_loss(
        token_losses, active, weights, [item], old
    )

    expected_new_structure = (
        len(delimiter_positions) * 5.0 + len(other_structure_positions) * 11.0
    ) / (len(delimiter_positions) + len(other_structure_positions))
    assert float(new_regions["final_count"]) == pytest.approx(13.0)
    assert float(new_regions["trace"]) == pytest.approx(7.0)
    assert float(new_regions["structure"]) == pytest.approx(expected_new_structure)
    assert float(new_loss) == pytest.approx(
        8.0 * (13.0 + 7.0 + expected_new_structure)
    )
    assert float(old_regions["trace"]) == pytest.approx(6.0)
    assert float(old_regions["structure"]) == pytest.approx(11.0)
    assert float(old_loss) == pytest.approx(8.0 * (13.0 + 6.0 + 11.0))


def test_v49_nonthinking_loss_is_identical_to_v47() -> None:
    old = preset_v47("debug", device="cpu", max_steps_for_language_pred=1)
    new = preset_v49("debug", device="cpu", max_steps_for_language_pred=1)
    vocab = V20Vocab.build(new, "abc xyz\n")
    item = render_v20(_example(3), vocab, "nonthinking")
    weights = collate_v20_loss_weights([item], new, "cpu", step=2)
    token_losses = torch.arange(
        1, len(item.tokens), dtype=torch.float32
    ).unsqueeze(0)
    active = torch.ones_like(token_losses, dtype=torch.bool)
    old_loss, old_regions = component_normalized_task_output_loss(
        token_losses, active, weights, [item], old
    )
    new_loss, new_regions = component_normalized_task_output_loss(
        token_losses, active, weights, [item], new
    )
    torch.testing.assert_close(old_loss, new_loss)
    assert old_regions.keys() == new_regions.keys()
    for name in old_regions:
        torch.testing.assert_close(old_regions[name], new_regions[name])


def test_v49_preflight_is_exposed() -> None:
    assert callable(run_preflight)


def _write_behavior_tables(root: Path) -> None:
    tables = root / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        [
            {"mode": "nonthinking", "ar_final_accuracy": 0.70, "trace_exact": float("nan")},
            {"mode": "thinking", "ar_final_accuracy": 0.92, "trace_exact": 0.91},
        ]
    ).to_csv(tables / "final_autoregressive_summary.csv", index=False)
    pd.DataFrame(
        [
            {"mode": mode, "count": count, "ar_final_accuracy": accuracy}
            for mode, accuracy in (("nonthinking", 0.70), ("thinking", 0.92))
            for count in range(1, 11)
        ]
    ).to_csv(tables / "final_autoregressive_by_count.csv", index=False)


def test_v49_behavior_gate_uses_fixed_10000_step_endpoint(tmp_path: Path) -> None:
    _write_behavior_tables(tmp_path)
    result = evaluate_behavior_gate(tmp_path)
    assert result["passed"] is True
    assert result["version"] == "v49"
    assert "10000-step" in result["endpoint_policy"]
