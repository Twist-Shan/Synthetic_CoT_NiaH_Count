from pathlib import Path

import pytest

from synthetic_counting_extensions.v5_2_switch_diagnostics import resolve_v5_run_dir


def make_run(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "vocab.json").write_text("{}", encoding="utf-8")
    checkpoint = path / "checkpoints" / "final.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    return path


def test_resolves_direct_v5_run(tmp_path: Path) -> None:
    run = make_run(tmp_path / "outputs" / "v5")
    assert resolve_v5_run_dir(run) == run.resolve()


def test_resolves_nested_downloaded_result_bundle(tmp_path: Path) -> None:
    run = make_run(tmp_path / "v5_synthetic_niah_v5" / "v5")
    assert resolve_v5_run_dir(tmp_path) == run.resolve()


def test_invalid_parent_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config.json, vocab.json, checkpoints/final.pt"):
        resolve_v5_run_dir(tmp_path)
