from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from synthetic_counting_v20.config import (
    config_from_dict,
    default_run_name,
    preset_config as preset_v20,
)
from synthetic_counting_v20.data import (
    V20Example,
    V20Vocab,
    character_token,
    collate_v20_loss_weights,
    render_v20,
)
from synthetic_counting_v22.config import preset_config as preset_v22
from synthetic_counting_v23.cli import main as v23_main
from synthetic_counting_v23.config import preset_config as preset_v23


ROOT = Path(__file__).resolve().parents[1]


def _example(count: int = 3) -> V20Example:
    markers = tuple(character_token("a") for _ in range(count))
    return V20Example(
        example_kind="counting_task",
        seq_tokens=list(markers),
        corpus_region="validation",
        corpus_start=0,
        corpus_end=count,
        prompt_sha256="v23-test",
        set_id="set",
        needle_characters=("a", "b", "c"),
        rendered_set_order=("a", "b", "c"),
        needle_positions=tuple(range(count)),
        needle_markers=markers,
        count=count,
        set_frequency_sum=0.1,
        set_frequency_bin=1,
        per_character_counts=(count, 0, 0),
    )


def test_v23_is_paired_separator_config_with_fixed_final_weight():
    v22 = preset_v22("main", device="cpu")
    v23 = preset_v23("main", device="cpu")
    assert (v23.version, v23.count_tokenization, v23.trace_format) == (
        "v23",
        "atomic",
        "separator",
    )
    assert v23.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )
    assert v23.final_count_loss_weight == 8.0
    assert v23.cot_trace_loss_weight == 1.0
    assert v23.seq_len == v22.seq_len == 256
    assert v23.count_max_threshold == v22.count_max_threshold == 30
    assert v23.max_render_len == v22.max_render_len == 327
    assert config_from_dict(v23.to_dict()) == v23
    assert "fcw8_cotw1" in default_run_name(v23)


def test_v23_version_rejects_noncanonical_final_weight():
    with pytest.raises(ValueError, match="requires final_count_loss_weight=8"):
        preset_v20(
            "main",
            version="v23",
            count_tokenization="atomic",
            trace_format="separator",
            final_count_loss_weight=7.0,
        )
    # The version-specific constructor makes the requested canonical setting
    # authoritative even if a stale caller supplies a conflicting override.
    assert preset_v23("main", final_count_loss_weight=7.0).final_count_loss_weight == 8.0


def test_v23_is_grammar_and_vocabulary_matched_to_v22():
    corpus = "abc xyz\n"
    v22 = preset_v22("main", device="cpu")
    v23 = preset_v23("main", device="cpu")
    vocab22 = V20Vocab.build(v22, corpus)
    vocab23 = V20Vocab.build(v23, corpus)
    rendered22 = render_v20(_example(), vocab22, "thinking")
    rendered23 = render_v20(_example(), vocab23, "thinking")
    assert vocab22.id_to_token == vocab23.id_to_token
    assert vocab22.fingerprint == vocab23.fingerprint
    assert rendered22.tokens == rendered23.tokens
    assert rendered22.input_ids == rendered23.input_ids


def test_v23_applies_weight_eight_to_both_final_count_targets():
    cfg = preset_v23("main", device="cpu")
    vocab = V20Vocab.build(cfg, "abc xyz\n")
    rendered = [
        render_v20(_example(), vocab, "nonthinking"),
        render_v20(_example(), vocab, "thinking"),
    ]
    weights = collate_v20_loss_weights(
        rendered,
        cfg,
        "cpu",
        step=cfg.max_steps_for_language_pred + 1,
    )
    for row, item in enumerate(rendered):
        assert item.spans is not None
        assert torch.all(weights[row, list(item.spans.count_positions)] == 8.0)
    thinking = rendered[1]
    assert thinking.spans is not None
    assert torch.all(weights[1, list(thinking.spans.trace_marker_positions)] == 1.0)


def test_v23_cli_defaults_to_the_paired_weighted_run(monkeypatch, tmp_path):
    captured = {}

    def fake_pipeline(cfg, **kwargs):
        captured["cfg"] = cfg
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(
        "synthetic_counting_v20.pipeline.run_v20_pipeline",
        fake_pipeline,
    )
    v23_main(["--preset", "debug", "--stage", "prepare", "--out-root", str(tmp_path)])
    cfg = captured["cfg"]
    assert cfg.version == "v23"
    assert cfg.final_count_loss_weight == 8.0
    assert cfg.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )


def test_v23_cli_rejects_conflicting_weight():
    with pytest.raises(SystemExit):
        v23_main(["--final-count-loss-weight", "7"])


def test_v23_colab_notebook_is_clean_paired_and_auditable():
    path = ROOT / "notebooks" / "Trace_Count_v23_NoIndex_FCW8_Colab.ipynb"
    assert path.exists()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v23"' in source
    assert 'TRACE_FORMAT = "separator"' in source
    assert "FINAL_COUNT_LOSS_WEIGHT = 8.0" in source
    assert '"rope/nonthinking"' in source
    assert '"rope/thinking"' in source
    assert "COUNT_MAX_THRESHOLD = 30" in source
    assert "MAX_TRAIN_STEPS = 10_000" in source
    assert "CHECKPOINT_EVERY_STEPS = 100" in source
    assert "RECOVERY_EVERY_STEPS = 500" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
