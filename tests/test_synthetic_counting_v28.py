from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

from synthetic_counting_v20.data import V20Vocab
from synthetic_counting_v20.model import build_model
from synthetic_counting_v24_3.config import preset_config as preset_v24_3
from synthetic_counting_v28.config import preset_config as preset_v28


CORPUS = "abc xyz\n"
ROOT = Path(__file__).resolve().parents[1]


def test_v28_changes_only_version_and_partial_count_readout_from_v24_3() -> None:
    baseline = preset_v24_3("main", device="cpu")
    candidate = preset_v28("main", device="cpu")
    changed = {
        key
        for key, value in asdict(candidate).items()
        if asdict(baseline).get(key) != value
    }
    assert changed == {"version", "untie_atomic_count_readout"}
    assert candidate.tie_word_embeddings
    assert candidate.untie_atomic_count_readout
    assert candidate.trace_format == "separator"
    assert candidate.training_count_distribution == "uniform"
    assert candidate.task_output_loss_reduction == "component_normalized"
    assert candidate.answer_query_contrastive_weight == 0.0


def test_v28_step_zero_function_matches_v24_3_exactly() -> None:
    baseline_cfg = preset_v24_3("debug", device="cpu")
    candidate_cfg = preset_v28("debug", device="cpu")
    baseline_vocab = V20Vocab.build(baseline_cfg, CORPUS)
    candidate_vocab = V20Vocab.build(candidate_cfg, CORPUS)
    assert baseline_vocab == candidate_vocab
    baseline = build_model(baseline_cfg, baseline_vocab, device="cpu").eval()
    candidate = build_model(candidate_cfg, candidate_vocab, device="cpu").eval()
    ids = torch.tensor(
        [[baseline_vocab.token_to_id["<BOS>"], baseline_vocab.token_to_id["<Ans>"]]],
        dtype=torch.long,
    )
    with torch.no_grad():
        baseline_logits = baseline(ids).logits
        candidate_logits = candidate(ids).logits
    torch.testing.assert_close(candidate_logits, baseline_logits, rtol=0, atol=0)
    assert candidate.count_lm_head is not None
    torch.testing.assert_close(
        candidate.count_lm_head.weight,
        candidate.token_embedding.weight[candidate_vocab.number_ids],
        rtol=0,
        atol=0,
    )
    assert candidate.parameter_count() - baseline.parameter_count() == (
        len(candidate_vocab.number_ids) * candidate_cfg.n_embd
    )


def test_v28_count_output_gradient_does_not_enter_count_input_rows() -> None:
    cfg = preset_v28("debug", device="cpu")
    vocab = V20Vocab.build(cfg, CORPUS)
    model = build_model(cfg, vocab, device="cpu")
    ids = torch.tensor(
        [[vocab.token_to_id["<BOS>"], vocab.token_to_id["<Ans>"]]],
        dtype=torch.long,
    )
    logits = model(ids).logits
    logits[0, -1, vocab.number_ids].sum().backward()
    assert model.count_lm_head is not None
    assert model.count_lm_head.weight.grad is not None
    assert float(model.count_lm_head.weight.grad.abs().sum()) > 0
    embedding_grad = model.token_embedding.weight.grad
    assert embedding_grad is not None
    torch.testing.assert_close(
        embedding_grad[vocab.number_ids],
        torch.zeros_like(embedding_grad[vocab.number_ids]),
        rtol=0,
        atol=0,
    )


def test_partial_and_full_untying_cannot_be_enabled_together() -> None:
    cfg = preset_v28("debug", device="cpu")
    with pytest.raises(ValueError, match="requires tie_word_embeddings=True"):
        replace(cfg, tie_word_embeddings=False).validate()


def test_v28_notebook_is_clean_and_preregisters_the_three_seed_gate() -> None:
    path = (
        ROOT
        / "notebooks"
        / "Trace_Count_v28_PartialCountReadout_Multiseed_Colab.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v28"' in source
    assert "SEEDS = (1234, 2234, 3234)" in source
    assert 'changed_fields == {"version", "untie_atomic_count_readout"}' in source
    assert "10 * planned.n_embd == 2560" in source
    assert '"--stage", "phase,causal,extended,plots"' in source
    assert "synthetic_counting_v27" not in source
    assert "CALIBRATION_DIR" not in source
    assert "trace_safety" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
