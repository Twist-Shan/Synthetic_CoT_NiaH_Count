from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from synthetic_counting_v20.cli import build_parser
from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.training import _maximum_entropy_cell_probabilities
from synthetic_counting_v24_3.config import preset_config as preset_v24_3
from synthetic_counting_v24_4.config import preset_config as preset_v24_4


ROOT = Path(__file__).resolve().parents[1]


def test_v24_4_changes_only_version_and_training_sampler() -> None:
    baseline = preset_v24_3("main", device="cpu")
    joint = preset_v24_4("main", device="cpu")
    changed = {
        key for key, value in asdict(joint).items()
        if asdict(baseline).get(key) != value
    }
    assert changed == {"version", "training_count_distribution"}
    assert joint.training_count_distribution == "maxent_set_count"
    assert joint.task_output_loss_reduction == "component_normalized"
    assert config_from_dict(joint.to_dict()) == joint


def test_maxent_sampler_matches_both_marginals_with_structural_zeros() -> None:
    support = np.asarray(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=bool,
    )
    probabilities = _maximum_entropy_cell_probabilities(support)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.full(3, 1 / 3), atol=1e-12)
    np.testing.assert_allclose(probabilities.sum(axis=0), np.full(3, 1 / 3), atol=1e-12)
    assert np.all(probabilities[~support] == 0)


def test_shared_cli_accepts_v24_4_maxent_sampler() -> None:
    args = build_parser("v24.4").parse_args(
        ["--training-count-distribution", "maxent_set_count"]
    )
    assert args.training_count_distribution == "maxent_set_count"


def test_v24_4_colab_notebook_is_clean_and_audits_sampler_only_change() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v24_4_MaxEnt_SetCount_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v24.4"' in source
    assert 'TRAINING_COUNT_DISTRIBUTION = "maxent_set_count"' in source
    assert 'changed_fields == {"version", "training_count_distribution"}' in source
    assert 'RUN_NAME = "v24.4_maxent_setcount_count1-10_seed1234"' in source
    assert "conditional_answer_accuracy_given_exact_trace" in source
    assert "success_criteria_met" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
