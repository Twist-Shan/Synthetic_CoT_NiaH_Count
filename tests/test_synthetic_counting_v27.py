from __future__ import annotations

from pathlib import Path

from synthetic_counting_v27 import calibration


def test_v27_wrapper_uses_v24_3_tied_unembedding(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_tail(source_run, output_dir, **kwargs):
        captured.update(source_run=source_run, output_dir=output_dir, **kwargs)
        return Path(output_dir)

    monkeypatch.setattr(calibration, "run_readout_tail", fake_tail)
    output = calibration.run_v27_calibration(
        tmp_path / "source", tmp_path / "out", device="cpu"
    )
    assert output == tmp_path / "out"
    assert captured["experiment"] == "v27"
    assert captured["expected_source_version"] == "v24.3"
    assert captured["readout_mode"] == "tied_unembedding"
