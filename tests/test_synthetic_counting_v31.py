from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from synthetic_counting_v29.config import preset_config as preset_v29
from synthetic_counting_v31.config import preset_config as preset_v31


ROOT = Path(__file__).resolve().parents[1]


def test_v31_changes_only_version_and_count_coefficient_from_v29() -> None:
    baseline = preset_v29("main", device="cpu")
    candidate = preset_v31("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "task_output_count_weight"}
    assert baseline.task_output_count_weight == 4.0
    assert candidate.task_output_count_weight == 8.0
    assert candidate.enabled_model_variants == (
        "rope/nonthinking",
        "rope/thinking",
    )
    assert candidate.trace_format == "separator"
    assert candidate.n_layer == 4
    assert candidate.n_head == 4


def test_v31_rejects_a_noncanonical_count_coefficient() -> None:
    candidate = preset_v31("main", device="cpu")
    with pytest.raises(ValueError, match="task_output_count_weight=8"):
        replace(candidate, task_output_count_weight=4.0).validate()


def test_shared_v31_cli_injects_weight_and_independent_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from synthetic_counting_v20.cli import main as shared_main
    import synthetic_counting_v20.pipeline as pipeline

    captured: dict[str, object] = {}

    def capture(cfg: object, **kwargs: object) -> None:
        captured["cfg"] = cfg

    monkeypatch.setattr(pipeline, "run_v20_pipeline", capture)
    shared_main(["--preset", "main"], version="v31")
    cfg = captured["cfg"]
    assert cfg.version == "v31"
    assert cfg.task_output_count_weight == 8.0
    assert cfg.enabled_model_variants == ("rope/nonthinking", "rope/thinking")


def test_v31_screen_notebook_is_clean_and_audits_the_scalar_only_change() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v31_CountWeight8_Screen_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v31"' in source
    assert "SEEDS = (1234,)" in source
    assert 'changed_fields == {"version", "task_output_count_weight"}' in source
    assert "planned.task_output_count_weight == 8.0" in source
    assert "v31_countweight8_independent_L256_pool100_seed" in source
    assert '"--stage", "phase,causal,extended,plots"' in source
    assert "shared_checkpoint" not in source
    assert "CALIBRATION_DIR" not in source
    assert "trace_safety" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
