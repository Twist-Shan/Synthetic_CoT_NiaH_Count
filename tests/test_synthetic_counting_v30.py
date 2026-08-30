from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from synthetic_counting_v20.data import V20Vocab
from synthetic_counting_v20.model import build_model
from synthetic_counting_v29.config import preset_config as preset_v29
from synthetic_counting_v30.config import preset_config as preset_v30


CORPUS = "abc xyz\n"
ROOT = Path(__file__).resolve().parents[1]


def test_v30_changes_only_version_and_depth_from_v29() -> None:
    baseline = preset_v29("main", device="cpu")
    candidate = preset_v30("main", device="cpu")
    changed = {
        key
        for key, value in asdict(candidate).items()
        if asdict(baseline).get(key) != value
    }
    assert changed == {"version", "n_layer"}
    assert baseline.n_layer == 4
    assert candidate.n_layer == 6
    assert candidate.n_head == baseline.n_head == 4
    assert candidate.n_embd == baseline.n_embd == 256
    assert candidate.trace_format == "separator"
    assert candidate.task_output_count_weight == 4.0
    assert candidate.answer_query_contrastive_weight == 0.0


def test_v30_rejects_a_noncanonical_depth() -> None:
    candidate = preset_v30("debug", device="cpu")
    with pytest.raises(ValueError, match="requires 6 layers"):
        replace(candidate, n_layer=4).validate()


def test_shared_v30_cli_injects_depth_weight_and_both_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from synthetic_counting_v20.cli import main as shared_main
    import synthetic_counting_v20.pipeline as pipeline

    captured: dict[str, object] = {}

    def capture(cfg: object, **kwargs: object) -> None:
        captured["cfg"] = cfg
        captured["kwargs"] = kwargs

    monkeypatch.setattr(pipeline, "run_v20_pipeline", capture)
    shared_main(["--preset", "debug"], version="v30")
    cfg = captured["cfg"]
    assert cfg.version == "v30"
    assert cfg.n_layer == 6
    assert cfg.task_output_count_weight == 4.0
    assert cfg.enabled_model_variants == ("rope/nonthinking", "rope/thinking")


def test_v30_adds_exactly_two_transformer_blocks() -> None:
    baseline_cfg = preset_v29("debug", device="cpu")
    candidate_cfg = preset_v30("debug", device="cpu")
    baseline_vocab = V20Vocab.build(baseline_cfg, CORPUS)
    candidate_vocab = V20Vocab.build(candidate_cfg, CORPUS)
    assert baseline_vocab == candidate_vocab
    baseline = build_model(baseline_cfg, baseline_vocab, device="cpu")
    candidate = build_model(candidate_cfg, candidate_vocab, device="cpu")
    assert len(baseline.layers) == 4
    assert len(candidate.layers) == 6
    assert candidate.parameter_count() > baseline.parameter_count()


def test_v30_notebook_is_clean_and_audits_the_depth_only_change() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v30_Depth6_Multiseed_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v30"' in source
    assert "SEEDS = (1234, 2234, 3234)" in source
    assert 'changed_fields == {"version", "n_layer"}' in source
    assert "baseline.n_layer == 4" in source
    assert "planned.n_layer == 6" in source
    assert "v30_depth6_countweight4_partial_readout" in source
    assert '"--stage", "phase,causal,extended,plots"' in source
    assert "CALIBRATION_DIR" not in source
    assert "trace_safety" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
