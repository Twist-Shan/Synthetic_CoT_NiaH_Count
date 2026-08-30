from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

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
from synthetic_counting_v24_2.config import preset_config as preset_v24_2
from synthetic_counting_v24_3.config import preset_config as preset_v24_3


ROOT = Path(__file__).resolve().parents[1]


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


def _region_positions(item):
    assert item.spans is not None
    count = set(item.spans.count_positions)
    trace = {
        *(position for group in item.spans.trace_index_token_groups for position in group),
        *item.spans.trace_marker_positions,
    }
    output_start = item.spans.ans_pos if item.mode == "nonthinking" else item.spans.think_pos
    assert output_start is not None
    active = set(range(output_start, len(item.tokens)))
    return count, trace, active - count - trace


def test_v24_3_changes_only_version_and_task_output_loss_reduction():
    baseline = preset_v24_2("main", device="cpu")
    loss_only = preset_v24_3("main", device="cpu")
    changed = {
        key for key, value in asdict(loss_only).items() if asdict(baseline).get(key) != value
    }
    assert changed == {"version", "task_output_loss_reduction"}
    assert loss_only.training_count_distribution == "uniform"
    assert loss_only.task_output_loss_reduction == "component_normalized"
    assert (
        loss_only.task_output_count_weight,
        loss_only.task_output_trace_weight,
        loss_only.task_output_structure_weight,
    ) == (1.0, 1.0, 0.1)
    assert "v24.3_main_" in default_run_name(loss_only)
    assert "taskloss-component_normalized-c1-t1-s0p1" in default_run_name(loss_only)
    assert config_from_dict(loss_only.to_dict()) == loss_only


def test_v24_3_rejects_legacy_task_output_reduction():
    payload = preset_v24_3("main", device="cpu").to_dict()
    payload["task_output_loss_reduction"] = "token_weighted_mean"
    with pytest.raises(
        ValueError,
        match="requires task_output_loss_reduction='component_normalized'",
    ):
        config_from_dict(payload)


@pytest.mark.parametrize(
    ("mode", "expected", "expected_shares"),
    [
        ("nonthinking", 2.3, (1.0 / 1.1, 0.0, 0.1 / 1.1)),
        ("thinking", 6.3, (1.0 / 2.1, 1.0 / 2.1, 0.1 / 2.1)),
    ],
)
def test_component_normalized_loss_has_mode_invariant_count_coefficient(
    mode: str,
    expected: float,
    expected_shares: tuple[float, float, float],
):
    cfg = preset_v24_3("debug", device="cpu", max_steps_for_language_pred=1)
    vocab = V20Vocab.build(cfg, "abc xyz\n")
    item = render_v20(_example(3), vocab, mode)
    weights = collate_v20_loss_weights([item], cfg, "cpu", step=2)
    token_losses = torch.zeros((1, len(item.tokens) - 1), dtype=torch.float32)
    active = torch.ones_like(token_losses, dtype=torch.bool)
    count_positions, trace_positions, structure_positions = _region_positions(item)
    for position in count_positions:
        token_losses[0, position - 1] = 2.0
    for position in trace_positions:
        token_losses[0, position - 1] = 4.0
    for position in structure_positions:
        token_losses[0, position - 1] = 3.0

    loss, regions = component_normalized_task_output_loss(
        token_losses, active, weights, [item], cfg
    )
    torch.testing.assert_close(loss, torch.tensor(expected), rtol=0, atol=1e-6)
    assert float(regions["final_count"]) == 2.0
    assert float(regions["structure"]) == 3.0
    if mode == "thinking":
        assert float(regions["trace"]) == 4.0
    else:
        assert "trace" not in regions

    denominator = 1.1 if mode == "nonthinking" else 2.1
    actual_shares = (
        cfg.task_output_count_weight / denominator,
        (cfg.task_output_trace_weight / denominator) if mode == "thinking" else 0.0,
        cfg.task_output_structure_weight / denominator,
    )
    assert actual_shares == pytest.approx(expected_shares)


def test_trace_region_mean_is_not_diluted_by_trace_length():
    cfg = preset_v24_3("debug", device="cpu", max_steps_for_language_pred=1)
    vocab = V20Vocab.build(cfg, "abc xyz\n")
    items = [render_v20(_example(count), vocab, "thinking") for count in (1, 4)]
    weights = collate_v20_loss_weights(items, cfg, "cpu", step=2)
    width = max(len(item.tokens) for item in items) - 1
    token_losses = torch.zeros((2, width), dtype=torch.float32)
    active = torch.zeros_like(token_losses, dtype=torch.bool)
    for row, item in enumerate(items):
        active[row, : len(item.tokens) - 1] = True
        count_positions, trace_positions, structure_positions = _region_positions(item)
        for position in count_positions:
            token_losses[row, position - 1] = 2.0
        for position in trace_positions:
            token_losses[row, position - 1] = 4.0
        for position in structure_positions:
            token_losses[row, position - 1] = 3.0

    loss, regions = component_normalized_task_output_loss(
        token_losses, active, weights, items, cfg
    )
    torch.testing.assert_close(loss, torch.tensor(6.3), rtol=0, atol=1e-6)
    torch.testing.assert_close(regions["trace"], torch.tensor(4.0), rtol=0, atol=1e-6)


def test_v24_3_colab_notebook_is_clean_and_audits_loss_only_change():
    path = ROOT / "notebooks" / "Trace_Count_v24_3_ComponentLoss_Count10_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v24.3"' in source
    assert 'RUN_NAME = "v24.3_componentloss_count1-10_seed1234"' in source
    assert 'TASK_OUTPUT_LOSS_REDUCTION = "component_normalized"' in source
    assert '"--task-output-loss-reduction", TASK_OUTPUT_LOSS_REDUCTION' in source
    assert 'changed_fields == {"version", "task_output_loss_reduction"}' in source
    assert "synthetic_counting_v24_3.run_v24_3" in source
    assert "component_reduction_active" in source
    assert "batch_final_count_region_coefficient_share" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
