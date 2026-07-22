from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pandas as pd
import pytest
import torch
import synthetic_counting_v20.training as training_module

from synthetic_counting_v20.config import config_from_dict, preset_config as preset_v20
from synthetic_counting_v20.data import V20Example, V20Vocab, character_token, render_v20
from synthetic_counting_v20.model import build_model
from synthetic_counting_v20.phase_transition import build_training_token_exposure
from synthetic_counting_v20.training import (
    DenseSnapshotWriter,
    _copy_if_changed,
    _write_final_autoregressive_artifacts,
    checkpoint_steps,
    load_dense_snapshot_state,
)
from synthetic_counting_v21.config import preset_config as preset_v21


ROOT = Path(__file__).resolve().parents[1]


def _example(count: int) -> V20Example:
    sequence = [character_token("a")] * count
    return V20Example(
        example_kind="counting_task",
        seq_tokens=sequence,
        corpus_region="validation",
        corpus_start=0,
        corpus_end=len(sequence),
        prompt_sha256="synthetic",
        set_id="set",
        needle_characters=("a", "b", "c"),
        rendered_set_order=("a", "b", "c"),
        needle_positions=tuple(range(count)),
        needle_markers=tuple(sequence),
        count=count,
        per_character_counts=(count, 0, 0),
    )


def test_main_settings_are_rope_query_first_counts_1_to_30():
    v20 = preset_v20("main", device="cpu")
    v21 = preset_v21("main", device="cpu")
    assert (v20.version, v20.count_tokenization) == ("v20", "atomic")
    assert (v21.version, v21.count_tokenization) == ("v21", "digitwise")
    for cfg in (v20, v21):
        assert cfg.position_encodings == ("rope",)
        assert cfg.enabled_model_variants == ("rope/nonthinking", "rope/thinking")
        assert cfg.query_layout == "query_first"
        assert cfg.count_max_threshold == 30
        assert cfg.checkpoint_every == 100
        assert cfg.recovery_every == cfg.snapshot_shard_every == 500
        assert cfg.max_render_len <= cfg.n_positions
        assert config_from_dict(cfg.to_dict()) == cfg


def test_atomic_and_digitwise_rendering_change_only_number_spelling():
    corpus = "abc xyz\n"
    example = _example(12)
    cfg20 = preset_v20("main", device="cpu")
    cfg21 = preset_v21("main", device="cpu")
    vocab20 = V20Vocab.build(cfg20, corpus)
    vocab21 = V20Vocab.build(cfg21, corpus)
    item20 = render_v20(example, vocab20, "thinking")
    item21 = render_v20(example, vocab21, "thinking")
    assert item20.tokens[:6] == item21.tokens[:6]
    assert item20.spans is not None and item21.spans is not None
    assert item20.spans.prompt_start == item21.spans.prompt_start == 6
    assert item20.spans.think_pos == item21.spans.think_pos
    assert vocab20.number_tokens(12) == ("<12>",)
    assert vocab21.number_tokens(12) == ("<D1>", "<D2>")
    assert len(item20.spans.trace_index_token_groups[9]) == 1
    assert len(item21.spans.trace_index_token_groups[9]) == 2
    assert len(item20.spans.count_positions) == 1
    assert len(item21.spans.count_positions) == 2
    assert vocab21.decode_number_tokens(vocab21.number_tokens(12)) == 12


def test_shared_initialization_and_sdpa_manual_equivalence():
    corpus = "abc xyz\n"
    cfg20 = preset_v20("debug", device="cpu")
    cfg21 = preset_v21("debug", device="cpu")
    vocab20 = V20Vocab.build(cfg20, corpus)
    vocab21 = V20Vocab.build(cfg21, corpus)
    model20 = build_model(cfg20, vocab20, device="cpu").eval()
    model21 = build_model(cfg21, vocab21, device="cpu").eval()
    assert torch.equal(model20.layers[0].attention.qkv.weight, model21.layers[0].attention.qkv.weight)
    ids = torch.tensor([[vocab20.token_to_id["<BOS>"], vocab20.token_to_id["<CountChar>"]]])
    mask = torch.ones_like(ids)
    fast = model20(ids, mask).logits
    manual = model20(ids, mask, output_attentions=True).logits
    torch.testing.assert_close(fast, manual, rtol=1e-5, atol=1e-6)


def test_analysis_attention_intervention_disables_fast_path_and_is_local():
    cfg = preset_v20("debug", device="cpu")
    vocab = V20Vocab.build(cfg, "abc xyz\n")
    model = build_model(cfg, vocab, device="cpu").eval()
    ids = torch.tensor(
        [[vocab.token_to_id["<BOS>"], vocab.token_to_id["<CountChar>"]]]
    )
    attention = model.layers[0].attention
    calls = []

    def zero_first_head(_query, _key, value, weights):
        calls.append(True)
        patched = weights.clone()
        patched[:, 0] = 0
        return value, patched

    baseline = model(ids).logits
    attention.intervention = zero_first_head
    try:
        changed = model(ids).logits
    finally:
        attention.intervention = None
    assert calls == [True]
    assert not torch.equal(baseline, changed)
    assert all(layer.attention.intervention is None for layer in model.layers)


def test_controlled_count_distribution_is_explicit_and_unambiguous():
    uniform = preset_v20(
        "debug",
        device="cpu",
        task_occurrence_ratio=1.0,
        training_count_distribution="uniform",
    )
    assert uniform.training_count_distribution == "uniform"
    assert config_from_dict(uniform.to_dict()) == uniform
    with pytest.raises(ValueError, match="task_occurrence_ratio=1"):
        preset_v20(
            "debug",
            device="cpu",
            task_occurrence_ratio=0.5,
            training_count_distribution="uniform",
        )


def test_uniform_training_batch_draws_targets_before_candidates(monkeypatch):
    cfg = preset_v20(
        "debug",
        device="cpu",
        batch_size=8,
        task_occurrence_ratio=1.0,
        training_count_distribution="uniform",
    )
    vocab = V20Vocab.build(cfg, "abc xyz\n")
    stream = iter([_example(count) for count in (1, 2, 3, 4)] * 20)
    monkeypatch.setattr(
        training_module,
        "make_training_example",
        lambda *_args, **_kwargs: next(stream),
    )
    seed = 9
    expected_rng = random.Random(seed)
    expected = [
        expected_rng.randint(cfg.count_min, cfg.count_max_threshold)
        for _ in range(cfg.batch_size)
    ]
    examples, rendered = training_module._training_batch(
        cfg,
        vocab,
        "abc xyz\n",
        None,
        None,
        "thinking",
        random.Random(seed),
    )
    assert [example.count for example in examples] == expected
    assert len(rendered) == cfg.batch_size


def test_final_ar_storage_keeps_numeric_rows_and_bounds_failure_text(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "position_encoding": "rope",
                "mode": "thinking",
                "row_id": row,
                "count": 1 + row // 6,
                "ar_accuracy": 0.0,
                "generated_tokens": "long trace " * 20,
            }
            for row in range(12)
        ]
    )
    _write_final_autoregressive_artifacts(
        frame, tmp_path, examples_per_count=6
    )
    detail = pd.read_csv(tmp_path / "tables/final_autoregressive_detail.csv")
    failures = pd.read_csv(tmp_path / "tables/final_autoregressive_failures.csv")
    assert len(detail) == 12
    assert "generated_tokens" not in detail
    assert failures.groupby("count").size().eq(5).all()
    assert failures["generated_tokens"].str.startswith("long trace").all()


def test_dense_snapshot_shard_roundtrip(tmp_path):
    cfg = preset_v20("debug", device="cpu")
    vocab = V20Vocab.build(cfg, "abc xyz\n")
    model = build_model(cfg, vocab, device="cpu")
    run_dir = tmp_path / "run"
    root = run_dir / "checkpoints" / "rope" / "thinking"
    writer = DenseSnapshotWriter(root, cfg, "rope", "thinking", None, run_dir)
    writer.add(model, 0, force=True)
    with torch.no_grad():
        model.layers[0].attention.qkv.bias.add_(1)
    writer.add(model, 3, force=True)
    assert [step for step, _ in checkpoint_steps(run_dir, "rope", "thinking")] == [0, 3]
    state = load_dense_snapshot_state(run_dir, "rope", "thinking", 3)
    torch.testing.assert_close(state["layers.0.attention.qkv.bias"].float(), torch.ones_like(model.layers[0].attention.qkv.bias))


def test_incremental_sync_does_not_skip_same_size_new_latest(tmp_path):
    source = tmp_path / "source.pt"
    target = tmp_path / "drive" / "latest.pt"
    source.write_bytes(b"old!")
    _copy_if_changed(source, target)
    old_time = source.stat().st_mtime_ns
    source.write_bytes(b"new!")
    os.utime(source, ns=(old_time + 10_000_000, old_time + 10_000_000))
    _copy_if_changed(source, target)
    assert target.read_bytes() == b"new!"


def test_training_exposure_distinguishes_atomic_digits_and_nonthinking(tmp_path):
    state = {
        "accepted_counts": {"1": 2, "2": 3, "3": 5, "4": 7, "12": 7},
    }
    rows = [
        {
            "position_encoding": "rope",
            "mode": mode,
            "step": 10,
            "cumulative_sampling_json": json.dumps(state),
        }
        for mode in ("thinking", "nonthinking")
    ]
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    pd.DataFrame(rows).to_csv(table_dir / "train_metrics.csv", index=False)
    atomic = build_training_token_exposure(preset_v20("main", device="cpu"), tmp_path)
    digitwise_cfg = preset_v21("main", device="cpu")
    digitwise = build_training_token_exposure(digitwise_cfg, tmp_path)
    think_k2 = atomic[(atomic["mode"] == "thinking") & (atomic["k"] == 2)].iloc[0]
    nonthink_k2 = atomic[(atomic["mode"] == "nonthinking") & (atomic["k"] == 2)].iloc[0]
    assert think_k2["trace_index_token_exposure"] == 22
    assert think_k2["continue_target_exposure_after_marker_k"] == 19
    assert think_k2["close_target_exposure_after_marker_k"] == 3
    assert nonthink_k2["trace_index_token_exposure"] == 0
    atomic_k12 = atomic[(atomic["mode"] == "thinking") & (atomic["k"] == 12)].iloc[0]
    digitwise_k12 = digitwise[(digitwise["mode"] == "thinking") & (digitwise["k"] == 12)].iloc[0]
    assert atomic_k12["trace_index_token_exposure"] == 7
    assert digitwise_k12["trace_index_token_exposure"] == 14


def test_colab_notebooks_are_clean_and_encode_the_storage_policy():
    for version in ("v20", "v21"):
        path = ROOT / "notebooks" / f"Trace_Count_{version}_Colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        assert "CHECKPOINT_EVERY_STEPS = 100" in source
        assert "RECOVERY_EVERY_STEPS = 500" in source
        assert "COUNT_MAX_THRESHOLD = 30" in source
        assert '"rope/nonthinking"' in source and '"rope/thinking"' in source
        assert "AUTO_DISCONNECT = True" in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
