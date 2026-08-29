from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthetic_counting_v20.config import config_from_dict, default_run_name
from synthetic_counting_v24.config import preset_config as preset_v24
from synthetic_counting_v24_2.config import preset_config as preset_v24_2


ROOT = Path(__file__).resolve().parents[1]


def test_v24_2_changes_only_version_and_training_count_distribution():
    baseline = preset_v24("main", device="cpu")
    balanced = preset_v24_2("main", device="cpu")
    changed = {
        key
        for key, value in balanced.to_dict().items()
        if baseline.to_dict().get(key) != value
    }
    assert changed == {"version", "training_count_distribution"}
    assert balanced.version == "v24.2"
    assert balanced.training_count_distribution == "uniform"
    assert balanced.enabled_model_variants == ("rope/nonthinking", "rope/thinking")
    assert balanced.count_max_threshold == 10
    assert balanced.final_count_loss_weight == balanced.cot_trace_loss_weight == 1.0
    assert "v24.2_main_" in default_run_name(balanced)
    assert "countdist-uniform" in default_run_name(balanced)


def test_v24_2_rejects_a_serialized_natural_distribution():
    payload = preset_v24_2("main", device="cpu").to_dict()
    payload["training_count_distribution"] = "natural"
    with pytest.raises(
        ValueError,
        match="requires training_count_distribution='uniform'",
    ):
        config_from_dict(payload)


def test_v24_2_colab_notebook_is_clean_and_audits_balance():
    path = ROOT / "notebooks" / "Trace_Count_v24_2_Balanced_Count10_Colab.ipynb"
    assert path.exists()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v24.2"' in source
    assert 'TRAINING_COUNT_DISTRIBUTION = "uniform"' in source
    assert '"--training-count-distribution", TRAINING_COUNT_DISTRIBUTION' in source
    assert "changed_fields == {\"version\", \"training_count_distribution\"}" in source
    assert '"--run-prefix", RUN_DIR.name' in source
    assert '"--expected-version", VERSION' in source
    assert "training_sampling_distribution.csv" in source
    assert "final_autoregressive_by_count.csv" in source
    assert "synthetic_counting_v24_2.run_v24_2" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")


def test_v24_ncc_script_accepts_v24_2_run_identity():
    source = (ROOT / "scripts" / "compare_v24_modes_ncc.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--run-prefix"' in source
    assert 'parser.add_argument("--expected-version"' in source
    assert "expected_version=args.expected_version" in source
