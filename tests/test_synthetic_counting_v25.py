from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v24_7.config import preset_config as preset_v24_7
from synthetic_counting_v25.config import preset_config as preset_v25
from synthetic_counting_v25_1 import calibration


ROOT = Path(__file__).resolve().parents[1]


def test_v25_changes_only_retrieval_pressure_and_memory_consequences() -> None:
    baseline = preset_v24_7("main", device="cpu")
    stressed = preset_v25("main", device="cpu")
    changed = {
        key
        for key, value in asdict(stressed).items()
        if asdict(baseline).get(key) != value
    }
    assert changed == {
        "version",
        "seq_len",
        "n_positions",
        "needle_pool_frequency_threshold",
        "batch_size",
    }
    assert stressed.seq_len == 1024
    assert stressed.count_max_threshold == 10
    assert stressed.max_render_len == 1055
    assert stressed.n_positions == 1056
    assert stressed.needle_pool_frequency_threshold == 10.0 / 1024.0
    assert stressed.trace_format == "separator"
    assert not stressed.tie_word_embeddings
    assert config_from_dict(stressed.to_dict()) == stressed


def test_v25_debug_is_small_and_valid() -> None:
    cfg = preset_v25("debug", device="cpu")
    assert cfg.seq_len == 64
    assert cfg.count_max_threshold == 4
    assert cfg.max_render_len < cfg.n_positions
    assert cfg.batch_size == 4


def test_v25_notebook_is_clean_and_has_behavioral_gap_gate() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v25_LongContext_RetrievalPressure_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v25"' in source
    assert '"--seq-len", "1024"' not in source  # supplied canonically by the wrapper
    assert "thinking_minus_nonthinking" in source
    assert "accuracy_gap >= 0.05" in source
    assert 'manifest["experiment"] == "v25.1"' in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")


def test_v25_1_wrapper_preserves_experiment_identity(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_tail(source_run, output_dir, **kwargs):
        captured.update(source_run=source_run, output_dir=output_dir, **kwargs)
        return Path(output_dir)

    monkeypatch.setattr(calibration, "run_readout_tail", fake_tail)
    output = calibration.run_v25_1_calibration(
        tmp_path / "source", tmp_path / "out", device="cpu"
    )
    assert output == tmp_path / "out"
    assert captured["experiment"] == "v25.1"
    assert captured["expected_source_version"] == "v25"
