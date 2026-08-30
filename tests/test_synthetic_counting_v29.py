from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import torch
import pytest

from synthetic_counting_v20.data import V20Vocab
from synthetic_counting_v20.model import build_model
from synthetic_counting_v28.config import preset_config as preset_v28
from synthetic_counting_v29.config import preset_config as preset_v29


CORPUS = "abc xyz\n"
ROOT = Path(__file__).resolve().parents[1]


def test_v29_changes_only_version_and_count_region_weight_from_v28() -> None:
    baseline = preset_v28("main", device="cpu")
    candidate = preset_v29("main", device="cpu")
    changed = {
        key
        for key, value in asdict(candidate).items()
        if asdict(baseline).get(key) != value
    }
    assert changed == {"version", "task_output_count_weight"}
    assert candidate.task_output_count_weight == 4.0
    assert candidate.task_output_trace_weight == 1.0
    assert candidate.task_output_structure_weight == 0.1
    assert candidate.trace_format == "separator"
    assert candidate.tie_word_embeddings
    assert candidate.untie_atomic_count_readout
    assert candidate.answer_query_contrastive_weight == 0.0


def test_v29_rejects_a_noncanonical_count_region_weight() -> None:
    candidate = preset_v29("debug", device="cpu")
    with pytest.raises(ValueError, match="requires task_output_count_weight=4"):
        replace(candidate, task_output_count_weight=1.0).validate()


def test_shared_v29_cli_injects_the_canonical_weight_and_both_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from synthetic_counting_v20.cli import main as shared_main
    import synthetic_counting_v20.pipeline as pipeline

    captured: dict[str, object] = {}

    def capture(cfg: object, **kwargs: object) -> None:
        captured["cfg"] = cfg
        captured["kwargs"] = kwargs

    monkeypatch.setattr(pipeline, "run_v20_pipeline", capture)
    shared_main(["--preset", "debug"], version="v29")
    cfg = captured["cfg"]
    assert cfg.version == "v29"
    assert cfg.task_output_count_weight == 4.0
    assert cfg.enabled_model_variants == ("rope/nonthinking", "rope/thinking")


def test_v29_step_zero_model_matches_v28_exactly() -> None:
    baseline_cfg = preset_v28("debug", device="cpu")
    candidate_cfg = preset_v29("debug", device="cpu")
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
    assert candidate.parameter_count() == baseline.parameter_count()


def test_v29_notebook_is_clean_and_audits_the_single_scalar_change() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v29_CountWeight4_Multiseed_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v29"' in source
    assert "SEEDS = (1234, 2234, 3234)" in source
    assert 'changed_fields == {"version", "task_output_count_weight"}' in source
    assert "planned.task_output_count_weight == 4.0" in source
    assert "v29_countweight4_fixed_partial_readout" in source
    assert '"--stage", "phase,causal,extended,plots"' in source
    assert "CALIBRATION_DIR" not in source
    assert "trace_safety" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
