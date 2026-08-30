from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import V20Vocab
from synthetic_counting_v20.model import build_model
from synthetic_counting_v24_5.config import preset_config as preset_v24_5
from synthetic_counting_v24_6.config import preset_config as preset_v24_6


ROOT = Path(__file__).resolve().parents[1]


def test_v24_6_changes_only_version_and_embedding_tying() -> None:
    baseline = preset_v24_5("main", device="cpu")
    untied = preset_v24_6("main", device="cpu")
    changed = {
        key for key, value in asdict(untied).items()
        if asdict(baseline).get(key) != value
    }
    assert changed == {"version", "tie_word_embeddings"}
    assert baseline.tie_word_embeddings
    assert not untied.tie_word_embeddings
    assert config_from_dict(untied.to_dict()) == untied


def test_untied_head_matches_tied_logits_at_step_zero_but_has_separate_storage() -> None:
    corpus = "abc xyz\n"
    tied_cfg = preset_v24_5("debug", device="cpu")
    untied_cfg = preset_v24_6("debug", device="cpu")
    tied_vocab = V20Vocab.build(tied_cfg, corpus)
    untied_vocab = V20Vocab.build(untied_cfg, corpus)
    assert tied_vocab.id_to_token == untied_vocab.id_to_token
    tied = build_model(tied_cfg, tied_vocab, device="cpu").eval()
    untied = build_model(untied_cfg, untied_vocab, device="cpu").eval()
    assert tied.lm_head is None
    assert untied.lm_head is not None
    assert untied.lm_head.weight.data_ptr() != untied.token_embedding.weight.data_ptr()
    torch.testing.assert_close(untied.lm_head.weight, untied.token_embedding.weight)
    tokens = torch.arange(8).remainder(len(tied_vocab.id_to_token)).unsqueeze(0)
    with torch.no_grad():
        torch.testing.assert_close(tied(tokens).logits, untied(tokens).logits)


def test_v24_6_colab_notebook_is_clean_and_raw_accuracy_is_the_gate() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v24_6_UntiedHead_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v24.6"' in source
    assert "assert not PLANNED_CONFIG.tie_word_embeddings" in source
    assert 'changed_fields == {"version", "tie_word_embeddings"}' in source
    assert 'RUN_NAME = "v24.6_untied_head_pool20_count1-10_seed1234"' in source
    assert '"success_criteria_met": success_criteria_met' in source
    assert "trace_readout_diagnostic_criteria_met" in source
    assert "trace_readout_success_criteria_met" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
