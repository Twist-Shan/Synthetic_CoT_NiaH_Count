from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from synthetic_counting_v20.data import V20Rendered, V20Spans
from synthetic_counting_v20.training import (
    scheduled_sampling_inputs,
    scheduled_sampling_probability,
)
from synthetic_counting_v32.config import preset_config as preset_v32
from synthetic_counting_v33.config import preset_config as preset_v33


ROOT = Path(__file__).resolve().parents[1]


class _AlwaysToken(torch.nn.Module):
    def __init__(self, token: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.token = int(token)
        self.vocab_size = int(vocab_size)

    def forward(self, *, input_ids, attention_mask=None):
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -10.0,
            device=input_ids.device,
        )
        logits[..., self.token] = 10.0
        return SimpleNamespace(logits=logits)


def _rendered(mode: str) -> V20Rendered:
    thinking = mode == "thinking"
    spans = V20Spans(
        bos_pos=0,
        prompt_start=1,
        prompt_end_exclusive=2,
        think_pos=2 if thinking else None,
        trace_index_positions=(3, 5) if thinking else (),
        trace_index_token_groups=((3,), (5,)) if thinking else (),
        trace_marker_positions=(4,) if thinking else (),
        think_close_pos=5 if thinking else None,
        ans_pos=6,
        count_positions=(7,),
        eos_pos=7,
        task_prefix_positions=(1,),
    )
    ids = list(range(8))
    return V20Rendered(
        example_kind="counting_task",
        mode=mode,
        tokens=[str(value) for value in ids],
        input_ids=ids,
        labels=list(ids),
        spans=spans,
        prompt_needle_positions=(),
        count=2,
        trace_format="separator",
    )


def test_v33_changes_only_declared_rollin_and_budget_fields_from_v32() -> None:
    baseline = preset_v32("main", device="cpu")
    candidate = preset_v33("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {
        "version",
        "train_steps",
        "phase_cloud_steps",
        "task_output_scheduled_sampling_max_probability",
    }
    assert candidate.train_steps == 6_000
    assert candidate.task_output_scheduled_sampling_max_probability == 0.5
    assert candidate.training_count_distribution == "maxent_set_count"
    assert candidate.trace_format == "separator"
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )


def test_v33_rollin_schedule_is_mode_specific_and_linear() -> None:
    cfg = preset_v33("main", device="cpu")
    assert scheduled_sampling_probability(cfg, 1_500, "thinking") == 0.0
    assert scheduled_sampling_probability(cfg, 6_000, "thinking") == 0.5
    assert scheduled_sampling_probability(cfg, 6_000, "nonthinking") == 0.0
    midpoint = scheduled_sampling_probability(cfg, 3_750, "thinking")
    assert midpoint == pytest.approx(0.25)


def test_rollin_changes_only_thinking_generated_prefix_inputs() -> None:
    items = [_rendered("thinking"), _rendered("nonthinking")]
    ids = torch.tensor([item.input_ids for item in items], dtype=torch.long)
    attention = torch.ones_like(ids)
    corrupted, stats = scheduled_sampling_inputs(
        _AlwaysToken(token=31),
        ids,
        attention,
        items,
        probability=1.0,
    )
    assert torch.equal(corrupted[0, :3], ids[0, :3])
    assert torch.equal(corrupted[0, 3:7], torch.full((4,), 31))
    assert torch.equal(corrupted[0, 7:], ids[0, 7:])
    assert torch.equal(corrupted[1], ids[1])
    assert stats == {
        "eligible_tokens": 4,
        "selected_tokens": 4,
        "changed_tokens": 4,
        "changed_fraction": 1.0,
    }


def test_v33_rejects_noncanonical_rollin_or_budget() -> None:
    candidate = preset_v33("main", device="cpu")
    with pytest.raises(
        ValueError,
        match="task_output_scheduled_sampling_max_probability=0.5",
    ):
        replace(
            candidate,
            task_output_scheduled_sampling_max_probability=0.25,
        ).validate()
    with pytest.raises(ValueError, match="requires train_steps=6000"):
        replace(candidate, train_steps=10_000).validate()


def test_v33_screen_notebook_is_clean_and_predeclares_endpoint() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v33_Scheduled_Rollin_Screen_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v33"' in source
    assert "SEEDS = (1234,)" in source
    assert '"task_output_scheduled_sampling_max_probability"' in source
    assert "planned.task_output_scheduled_sampling_max_probability == 0.5" in source
    assert '"--train-steps", "6000"' in source
    assert 'role_table["step"].eq(6000)' in source
    assert "v33_rollin05_steps6000_independent_L256_pool100_seed" in source
    assert '"--stage", "phase,causal,extended,plots"' in source
    assert "shared_checkpoint" not in source
    assert "CALIBRATION_DIR" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
