from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v24_3.config import preset_config as preset_v24_3
from synthetic_counting_v26.config import preset_config as preset_v26
from synthetic_counting_v26_1 import calibration


ROOT = Path(__file__).resolve().parents[1]


def test_v26_changes_only_native_head_tying_from_v24_3() -> None:
    baseline = preset_v24_3("main", device="cpu")
    candidate = preset_v26("main", device="cpu")
    changed = {
        key
        for key, value in asdict(candidate).items()
        if asdict(baseline).get(key) != value
    }
    assert changed == {"version", "tie_word_embeddings"}
    assert candidate.seq_len == 256
    assert candidate.count_max_threshold == 10
    assert candidate.needle_pool_size == 100
    assert candidate.training_count_distribution == "uniform"
    assert candidate.task_output_loss_reduction == "component_normalized"
    assert not candidate.tie_word_embeddings
    assert candidate.answer_query_contrastive_weight == 0.0
    assert config_from_dict(candidate.to_dict()) == candidate


def test_v26_debug_keeps_viable_data_geometry_and_a_small_schedule() -> None:
    cfg = preset_v26("debug", device="cpu")
    assert cfg.seq_len == 256
    assert cfg.count_max_threshold == 10
    assert cfg.max_render_len < cfg.n_positions
    assert cfg.batch_size == 4
    assert cfg.train_steps == 6


def test_v26_notebook_is_clean_and_audits_pairing() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v26_L256_DiverseSet_Untied_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v26"' in source
    assert 'assert changed_fields == {"version", "tie_word_embeddings"}' in source
    assert "paired_sampling_exactly_matched" in source
    assert "maximum_count_relative_error" in source
    assert "thinking_minus_nonthinking" in source
    assert "accuracy_gap >= 0.05" in source
    assert 'manifest["experiment"] == "v26.1"' in source
    assert 'manifest["source_version"] == "v26"' in source
    assert "1024" not in "".join(_cell["source"][0] for _cell in notebook["cells"] if _cell["cell_type"] == "markdown")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")


def test_v26_1_wrapper_preserves_experiment_identity(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_tail(source_run, output_dir, **kwargs):
        captured.update(source_run=source_run, output_dir=output_dir, **kwargs)
        return Path(output_dir)

    monkeypatch.setattr(calibration, "run_readout_tail", fake_tail)
    output = calibration.run_v26_1_calibration(
        tmp_path / "source", tmp_path / "out", device="cpu"
    )
    assert output == tmp_path / "out"
    assert captured["experiment"] == "v26.1"
    assert captured["expected_source_version"] == "v26"
