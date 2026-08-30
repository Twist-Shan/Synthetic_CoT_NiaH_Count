from __future__ import annotations

from dataclasses import asdict, replace
import json
import random
from pathlib import Path

import pytest
import torch

import synthetic_counting_v20.training as training_module
from synthetic_counting_v20.data import (
    V20Example,
    V20Vocab,
    character_token,
    collate_v20,
    collate_v20_loss_weights,
    render_v20,
)
from synthetic_counting_v20.model import build_model
from synthetic_counting_v20.training import (
    DenseSnapshotWriter,
    checkpoint_steps,
    equal_mode_training_loss,
    load_dense_snapshot_state,
    paired_joint_training_batch,
)
from synthetic_counting_v29.config import preset_config as preset_v29
from synthetic_counting_v31.config import preset_config as preset_v31


CORPUS = "abc xyz\n"
ROOT = Path(__file__).resolve().parents[1]


def _example(count: int) -> V20Example:
    sequence = [character_token("a")] * count
    return V20Example(
        example_kind="counting_task",
        seq_tokens=sequence,
        corpus_region="train",
        corpus_start=0,
        corpus_end=len(sequence),
        prompt_sha256=f"synthetic-{count}",
        set_id="set",
        needle_characters=("a", "b", "c"),
        rendered_set_order=("a", "b", "c"),
        needle_positions=tuple(range(count)),
        needle_markers=tuple(sequence),
        count=count,
        per_character_counts=(count, 0, 0),
    )


def test_v31_changes_only_version_batch_and_mode_coupling_from_v29() -> None:
    baseline = preset_v29("main", device="cpu")
    candidate = preset_v31("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "batch_size", "training_mode_coupling"}
    assert candidate.batch_size == 256
    assert candidate.training_mode_coupling == "paired_joint"
    assert candidate.n_layer == baseline.n_layer == 4
    assert candidate.trace_format == baseline.trace_format == "separator"
    assert candidate.task_output_count_weight == baseline.task_output_count_weight == 4.0


def test_v31_rejects_independent_training_or_unpaired_batch() -> None:
    candidate = preset_v31("main", device="cpu")
    with pytest.raises(ValueError, match="training_mode_coupling"):
        replace(candidate, training_mode_coupling="independent").validate()
    with pytest.raises(ValueError, match="batch_size=256"):
        replace(candidate, batch_size=254).validate()


def test_v31_batch_pairs_identical_semantics_with_unchanged_renderings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = preset_v31("main", device="cpu")
    vocab = V20Vocab.build(cfg, CORPUS)

    def fake_batch(local_cfg, *_args, **_kwargs):
        examples = [_example(index % 10 + 1) for index in range(local_cfg.batch_size)]
        return examples, []

    monkeypatch.setattr(training_module, "_training_batch", fake_batch)
    examples, rendered = paired_joint_training_batch(
        cfg,
        vocab,
        CORPUS,
        None,
        None,
        random.Random(7),
    )
    assert len(examples) == len(rendered) == 256
    for index in range(0, len(examples), 2):
        assert examples[index] is examples[index + 1]
        assert rendered[index].mode == "nonthinking"
        assert rendered[index + 1].mode == "thinking"
        assert rendered[index].tokens == render_v20(examples[index], vocab, "nonthinking").tokens
        assert rendered[index + 1].tokens == render_v20(examples[index], vocab, "thinking").tokens


def test_v31_equal_mode_reduction_does_not_token_weight_the_long_trace() -> None:
    cfg = preset_v31("main", device="cpu")
    vocab = V20Vocab.build(cfg, CORPUS)
    example = _example(4)
    rendered = [
        render_v20(example, vocab, "nonthinking"),
        render_v20(example, vocab, "thinking"),
    ]
    _, labels, _ = collate_v20(rendered, vocab, "cpu")
    active = labels[:, 1:] != -100
    token_losses = torch.zeros_like(active, dtype=torch.float32)
    token_losses[0] = 2.0
    token_losses[1] = 4.0

    all_sequence_weights = collate_v20_loss_weights(rendered, cfg, "cpu", step=1)
    loss, _, mode_losses = equal_mode_training_loss(
        token_losses,
        active,
        all_sequence_weights,
        rendered,
        cfg,
        component_reduction_active=False,
    )
    torch.testing.assert_close(mode_losses["nonthinking"], torch.tensor(2.0))
    torch.testing.assert_close(mode_losses["thinking"], torch.tensor(4.0))
    torch.testing.assert_close(loss, torch.tensor(3.0))

    task_weights = collate_v20_loss_weights(rendered, cfg, "cpu", step=1_501)
    task_loss, _, task_mode_losses = equal_mode_training_loss(
        token_losses,
        active,
        task_weights,
        rendered,
        cfg,
        component_reduction_active=True,
    )
    torch.testing.assert_close(task_mode_losses["nonthinking"], torch.tensor(8.2))
    torch.testing.assert_close(task_mode_losses["thinking"], torch.tensor(20.4))
    torch.testing.assert_close(task_loss, torch.tensor(14.3))


def test_v31_nonthinking_analysis_routes_to_the_shared_checkpoint(tmp_path) -> None:
    cfg = preset_v31("main", device="cpu")
    vocab = V20Vocab.build(cfg, CORPUS)
    model = build_model(cfg, vocab, device="cpu")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(cfg.to_dict()), encoding="utf-8"
    )
    root = run_dir / "checkpoints" / "rope" / "thinking"
    writer = DenseSnapshotWriter(root, cfg, "rope", "thinking", None, run_dir)
    writer.add(model, 0, force=True)

    thinking = checkpoint_steps(run_dir, "rope", "thinking")
    nonthinking = checkpoint_steps(run_dir, "rope", "nonthinking")
    assert thinking == nonthinking
    state = load_dense_snapshot_state(run_dir, "rope", "nonthinking", 0)
    torch.testing.assert_close(
        state["layers.0.attention.qkv.bias"].float(),
        model.layers[0].attention.qkv.bias,
    )


def test_shared_v31_cli_injects_the_canonical_joint_design(
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
    assert cfg.batch_size == 256
    assert cfg.training_mode_coupling == "paired_joint"
    assert cfg.enabled_model_variants == ("rope/nonthinking", "rope/thinking")


def test_v31_notebook_is_clean_and_audits_the_shared_mode_change() -> None:
    path = ROOT / "notebooks" / "Trace_Count_v31_PairedJoint_Multiseed_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v31"' in source
    assert "SEEDS = (1234, 2234, 3234)" in source
    assert (
        'changed_fields == {"version", "batch_size", "training_mode_coupling"}'
        in source
    )
    assert 'planned.training_mode_coupling == "paired_joint"' in source
    assert 'set(metrics["mode"]) == {"joint"}' in source
    assert 'metrics["batch_nonthinking_rows"].eq(128).all()' in source
    assert 'metrics["batch_thinking_rows"].eq(128).all()' in source
    assert "v31_paired_joint_shared_L256_pool100_seed" in source
    assert '"--stage", "phase,causal,extended,plots"' in source
    assert "CALIBRATION_DIR" not in source
    assert "trace_safety" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
