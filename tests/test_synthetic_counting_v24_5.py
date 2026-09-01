from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from synthetic_counting_v20.cli import build_parser
from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v24_4.config import preset_config as preset_v24_4
from synthetic_counting_v24_5.config import preset_config as preset_v24_5


ROOT = Path(__file__).resolve().parents[1]


def test_v24_5_changes_only_version_and_pool_size() -> None:
    baseline = preset_v24_4("main", device="cpu")
    reduced = preset_v24_5("main", device="cpu")
    changed = {
        key for key, value in asdict(reduced).items()
        if asdict(baseline).get(key) != value
    }
    assert changed == {"version", "needle_pool_size"}
    assert reduced.needle_pool_size == 20
    assert reduced.training_count_distribution == "maxent_set_count"
    assert config_from_dict(reduced.to_dict()) == reduced


def test_shared_cli_accepts_v24_5() -> None:
    args = build_parser("v24.5").parse_args(
        ["--needle-pool-size", "20", "--training-count-distribution", "maxent_set_count"]
    )
    assert args.needle_pool_size == 20


def test_v24_5_colab_notebook_is_clean_and_auditable() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v24_5_Pool20_MaxEnt_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v24.5"' in source
    assert "NEEDLE_POOL_SIZE = 20" in source
    assert 'changed_fields == {"version", "needle_pool_size"}' in source
    assert 'RUN_NAME = "v24.5_pool20_maxent_count1-10_seed1234"' in source
    assert "success_criteria_met" in source
    assert "trace_readout_success_criteria_met" in source
    assert 'DRIVE_RUN_DIR / "tables" / "trace_readout_summary.csv"' in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
