from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import V20Rendered, V20Spans
from synthetic_counting_v20.training import answer_query_supervised_contrastive_loss
from synthetic_counting_v24_6.config import preset_config as preset_v24_6
from synthetic_counting_v24_7.config import preset_config as preset_v24_7


ROOT = Path(__file__).resolve().parents[1]


def _rendered(count: int, ans_pos: int = 2) -> V20Rendered:
    spans = V20Spans(
        bos_pos=0,
        prompt_start=1,
        prompt_end_exclusive=2,
        think_pos=None,
        trace_index_positions=(),
        trace_index_token_groups=(),
        trace_marker_positions=(),
        think_close_pos=None,
        ans_pos=ans_pos,
        count_positions=(ans_pos + 1,),
        eos_pos=ans_pos + 2,
        task_prefix_positions=(1,),
    )
    return V20Rendered(
        example_kind="counting_task",
        mode="nonthinking",
        tokens=["x"] * (ans_pos + 3),
        input_ids=[0] * (ans_pos + 3),
        labels=[0] * (ans_pos + 3),
        spans=spans,
        prompt_needle_positions=(),
        count=count,
        trace_format="separator",
    )


def test_v24_7_changes_only_version_and_contrastive_weight() -> None:
    baseline = preset_v24_6("main", device="cpu")
    compressed = preset_v24_7("main", device="cpu")
    changed = {
        key
        for key, value in asdict(compressed).items()
        if asdict(baseline).get(key) != value
    }
    assert changed == {"version", "answer_query_contrastive_weight"}
    assert compressed.answer_query_contrastive_weight == 0.1
    assert compressed.answer_query_contrastive_temperature == 0.1
    assert not compressed.tie_word_embeddings
    assert config_from_dict(compressed.to_dict()) == compressed


def test_answer_query_contrastive_loss_rewards_same_count_clustering() -> None:
    rendered = [_rendered(1), _rendered(1), _rendered(2), _rendered(2)]
    clustered = torch.zeros(4, 5, 2, requires_grad=True)
    clustered.data[:, 2] = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    mixed = torch.zeros(4, 5, 2, requires_grad=True)
    mixed.data[:, 2] = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )
    clustered_loss = answer_query_supervised_contrastive_loss(
        clustered, rendered, temperature=0.1
    )
    mixed_loss = answer_query_supervised_contrastive_loss(
        mixed, rendered, temperature=0.1
    )
    assert clustered_loss < mixed_loss
    clustered_loss.backward()
    assert clustered.grad is not None
    assert torch.isfinite(clustered.grad).all()


def test_answer_query_contrastive_loss_handles_no_positive_pairs() -> None:
    hidden = torch.randn(3, 5, 4, requires_grad=True)
    rendered = [_rendered(1), _rendered(2), _rendered(3)]
    loss = answer_query_supervised_contrastive_loss(
        hidden, rendered, temperature=0.1
    )
    assert loss.item() == 0.0
    loss.backward()
    assert hidden.grad is not None


def test_v24_7_colab_notebook_is_clean_and_raw_accuracy_is_the_gate() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v24_7_AnswerCompression_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v24.7"' in source
    assert "PLANNED_CONFIG.answer_query_contrastive_weight == 0.1" in source
    assert '"answer_query_contrastive_weight",' in source
    assert 'RUN_NAME = "v24.7_answer_compression_pool20_count1-10_seed1234"' in source
    assert '"success_criteria_met": success_criteria_met' in source
    assert "trace_readout_diagnostic_criteria_met" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
