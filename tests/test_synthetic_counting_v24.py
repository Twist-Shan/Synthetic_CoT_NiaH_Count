from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import (
    V20Vocab,
    balanced_v20_examples,
    build_corpus_split,
    load_corpus_text,
)
from synthetic_counting_v20.needle_pool import build_needle_pool
from synthetic_counting_v24.config import preset_config


ROOT = Path(__file__).resolve().parents[1]


def test_v24_is_paired_v22_grammar_with_count_range_1_to_10():
    cfg = preset_config("main", device="cpu")
    assert cfg.version == "v24"
    assert cfg.trace_format == "separator"
    assert cfg.count_tokenization == "atomic"
    assert cfg.count_max_threshold == 10
    assert cfg.needle_pool_frequency_threshold == 10.0 / 256.0
    assert cfg.enabled_model_variants == ("rope/nonthinking", "rope/thinking")
    assert cfg.final_count_loss_weight == cfg.cot_trace_loss_weight == 1.0
    assert cfg.train_steps == 10_000
    assert config_from_dict(cfg.to_dict()) == cfg


def test_v24_rejects_a_mismatched_serialized_count_range():
    cfg = preset_config("main", device="cpu").to_dict()
    cfg["count_max_threshold"] = 30
    cfg["count_max"] = 30
    with pytest.raises(ValueError, match="requires count_max_threshold=10"):
        config_from_dict(cfg)


def test_v24_pool_supports_balanced_count_1_to_10_suites_in_every_split():
    cfg = preset_config("main", device="cpu")
    text = load_corpus_text()
    split = build_corpus_split(cfg, text)
    vocab = V20Vocab.build(cfg, text)
    pool = build_needle_pool(cfg, text, split, vocab.fingerprint)
    assert max(item.frequency_sum for item in pool.sets) <= 10.0 / 256.0
    for offset, region in enumerate(("train", "validation", "test")):
        examples = balanced_v20_examples(
            cfg,
            vocab,
            text,
            split,
            pool,
            examples_per_count=1,
            seed=cfg.seed + offset,
            region_name=region,
        )
        assert {int(item.count) for item in examples} == set(range(1, 11))


def test_v24_colab_notebook_is_clean_paired_and_runs_ncc():
    path = ROOT / "notebooks" / "Trace_Count_v24_NoIndex_Count10_Colab.ipynb"
    assert path.exists()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v24"' in source
    assert "COUNT_MAX_THRESHOLD = 10" in source
    assert "NEEDLE_POOL_FREQUENCY_THRESHOLD = 10.0 / 256.0" in source
    assert "COUNT_MAX_THRESHOLD = 30" not in source
    assert '"rope/nonthinking", "rope/thinking"' in source
    assert "compare_v24_modes_ncc.py" in source
    assert "selected_confirmation_summary.csv" in source
    assert "confirmation_ncc_above_chance" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
