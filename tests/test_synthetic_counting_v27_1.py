from __future__ import annotations

from pathlib import Path

import torch

from synthetic_counting_v20.data import V20Rendered, V20Spans
from synthetic_counting_v24_8.readout_tail import _trace_safety_loss
from synthetic_counting_v27_1 import calibration


def _thinking_rendered() -> V20Rendered:
    spans = V20Spans(
        bos_pos=0,
        prompt_start=2,
        prompt_end_exclusive=4,
        think_pos=4,
        trace_index_positions=(5,),
        trace_index_token_groups=((5,),),
        trace_marker_positions=(6,),
        think_close_pos=7,
        ans_pos=8,
        count_positions=(9,),
        eos_pos=10,
        task_prefix_positions=(1,),
    )
    ids = list(range(11))
    return V20Rendered(
        "counting_task", "thinking", [str(i) for i in ids], ids, ids, spans, (), 1, "separator"
    )


def test_trace_safety_uses_only_pre_answer_targets() -> None:
    rendered = _thinking_rendered()
    ids = torch.tensor([rendered.input_ids])
    logits = torch.zeros((1, len(rendered.input_ids), 16), requires_grad=True)
    loss = _trace_safety_loss(logits, ids, [rendered])
    loss.backward()
    active_query_positions = set(range(4, 8))
    gradient_positions = set(
        torch.nonzero(logits.grad.abs().sum(dim=-1)[0], as_tuple=False).flatten().tolist()
    )
    assert gradient_positions == active_query_positions


def test_v27_1_wrapper_uses_trace_safe_candidates(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_tail(source_run, output_dir, **kwargs):
        captured.update(source_run=source_run, output_dir=output_dir, **kwargs)
        return Path(output_dir)

    monkeypatch.setattr(calibration, "run_readout_tail", fake_tail)
    output = calibration.run_v27_1_calibration(
        tmp_path / "source", tmp_path / "out", device="cpu"
    )
    assert output == tmp_path / "out"
    assert captured["experiment"] == "v27.1"
    assert captured["expected_source_version"] == "v24.3"
    assert captured["readout_mode"] == "tied_unembedding"
    assert [item.trace_safety_weight for item in captured["candidates"]] == [0.1, 0.3, 1.0]
