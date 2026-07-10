from __future__ import annotations

import shutil
from dataclasses import replace

from synthetic_counting_extensions.v7_v8_sweeps import (
    SweepConfig,
    Vocab,
    preset_configs,
    render,
    train_one,
    validation_split,
)
from synthetic_counting_extensions.v7_v8_sweeps import Example


def test_numeric_tokens_are_shared_between_trace_and_answer() -> None:
    cfg = SweepConfig(experiment="v8", train_count_max=30, eval_count_max=30)
    vocab = Vocab(cfg)

    assert vocab.count_token(7) == "<7>"
    assert vocab.index_token(7) == "<7>"
    assert vocab.token_to_id[vocab.count_token(7)] == vocab.token_to_id[vocab.index_token(7)]
    assert "<C7>" not in vocab.token_to_id
    assert "<I7>" not in vocab.token_to_id
    assert len(vocab.id_to_token) == 6 + 64 + 10 + 30


def test_thinking_render_reuses_the_final_number_token() -> None:
    cfg = SweepConfig(
        experiment="v8",
        seq_len=4,
        train_count_max=3,
        eval_count_max=3,
    )
    vocab = Vocab(cfg)
    example = Example(
        seq_tokens=["<M0>", "<N0>", "<M1>", "<M2>"],
        count=3,
        needle_positions=[0, 2, 3],
        needle_markers=["<M0>", "<M1>", "<M2>"],
    )

    rendered = render(example, vocab, "thinking")
    assert rendered["tokens"] == [
        "<BOS>",
        "<M0>",
        "<N0>",
        "<M1>",
        "<M2>",
        "<Think>",
        "<1>",
        "<M0>",
        "<2>",
        "<M1>",
        "<3>",
        "<M2>",
        "</Think>",
        "<Ans>",
        "<3>",
        "<EOS>",
    ]


def test_v7_main_is_2048_only() -> None:
    configs = preset_configs("v7", "main")

    assert [cfg.seq_len for cfg in configs] == [2048]
    assert all(cfg.train_count_min == 1 and cfg.train_count_max == 10 for cfg in configs)
    assert all(cfg.eval_count_min == 1 and cfg.eval_count_max == 10 for cfg in configs)
    assert all(cfg.train_steps == 10000 for cfg in configs)
    assert all(cfg.effective_batch_size == 128 for cfg in configs)
    assert all(cfg.checkpoint_every == 2000 for cfg in configs)


def test_periodic_checkpoints_sync_and_resume_from_drive(tmp_path) -> None:
    initial_cfg = SweepConfig(
        experiment="v7",
        preset="debug",
        seq_len=8,
        train_count_max=2,
        eval_count_max=2,
        train_steps=2,
        batch_size=2,
        grad_accum_steps=1,
        warmup_steps=1,
        log_every=1,
        checkpoint_every=2,
        n_layer=1,
        n_head=1,
        n_embd=16,
        device="cpu",
    )
    local_run = tmp_path / "local_run"
    drive_run = tmp_path / "drive_run"
    train_one(
        initial_cfg,
        Vocab(initial_cfg),
        "nonthinking",
        local_run,
        sync_run_dir=drive_run,
    )

    remote_step_2 = drive_run / "checkpoints" / "nonthinking" / "step_000002" / "checkpoint.pt"
    assert remote_step_2.exists()
    assert (drive_run / "checkpoints" / "nonthinking" / "model.pt").exists()

    shutil.rmtree(local_run)
    (drive_run / "checkpoints" / "nonthinking" / "model.pt").unlink()
    resumed_cfg = replace(initial_cfg, train_steps=4)
    resumed = train_one(
        resumed_cfg,
        Vocab(resumed_cfg),
        "nonthinking",
        local_run,
        sync_run_dir=drive_run,
    )

    assert int(resumed["step"].max()) == 4
    assert (drive_run / "checkpoints" / "nonthinking" / "step_000004" / "checkpoint.pt").exists()


def test_v8_main_is_one_many_needle_setting_with_three_validation_ranges() -> None:
    configs = preset_configs("v8", "main")

    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.seq_len == 256
    assert (cfg.train_count_min, cfg.train_count_max) == (1, 30)
    assert (cfg.eval_count_min, cfg.eval_count_max) == (1, 30)
    assert cfg.train_steps == 10000
    assert cfg.effective_batch_size == 128
    assert validation_split(1) == "val_1_10"
    assert validation_split(10) == "val_1_10"
    assert validation_split(11) == "val_11_20"
    assert validation_split(20) == "val_11_20"
    assert validation_split(21) == "val_21_30"
    assert validation_split(30) == "val_21_30"


def test_v8_debug_exercises_all_three_validation_ranges() -> None:
    cfg = preset_configs("v8", "debug")[0]
    assert cfg.seq_len == 48
    assert cfg.train_count_max == 30
    assert cfg.eval_count_max == 30


def test_v9_main_keeps_v2_data_and_reduces_model_capacity() -> None:
    configs = preset_configs("v9", "main")

    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.seq_len == 256
    assert (cfg.train_count_min, cfg.train_count_max) == (1, 10)
    assert (cfg.eval_count_min, cfg.eval_count_max) == (1, 10)
    assert (cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.n_inner) == (3, 4, 128, 384)
    assert cfg.train_steps == 10000
    assert cfg.effective_batch_size == 128
    assert cfg.checkpoint_every == 2000
