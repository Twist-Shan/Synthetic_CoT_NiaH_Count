from __future__ import annotations

import json
from pathlib import Path

from build_v29_colab_notebook import build as build_v29_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v31_PairedJoint_Multiseed_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v29_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v29", "synthetic_counting_v31")
        source = source.replace("run_v29", "run_v31")
        source = source.replace('VERSION = "v29"', 'VERSION = "v31"')
        source = source.replace(
            'manifest["version"] == "v29"', 'manifest["version"] == "v31"'
        )
        source = source.replace(
            "v29_countweight4_fixed_partial_readout_L256_pool100_seed",
            "v31_paired_joint_shared_L256_pool100_seed",
        )
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v31: paired joint-mode shared model

This is a pre-registered three-seed experiment built directly from v29. It
keeps the 256-character prompt, counts 1--10, 100 marker sets, uniform count
sampler, four-layer/four-head/256-wide model, separator/no-index trace, partial
count-only output untying, component-normalized count coefficient 4, optimizer,
1,500-step language phase, and 10,000-step schedule.

The only conceptual change is that Non-thinking and Thinking are now two modes
of one shared model. Every semantic example is rendered in both unchanged
formats (128 rows per mode per step). Each complete per-mode v29 objective is
computed first, then the two scalar losses are averaged 1:1, so the longer
Thinking trace cannot silently receive more token weight.

There is no auxiliary loss, post-hoc decoder, calibration/frozen stage,
test-time training, extra layer, trace rewrite, or inference rule. Both modes
are evaluated from the exact same checkpoint.

Primary gate, fixed before opening confirmation seeds: in every retained seed,
Thinking accuracy >= 0.90, minimum per-count accuracy >= 0.80, trace exact >=
0.90, count spread <= 0.20, and Thinking-minus-Non-thinking accuracy >= 0.10.
All completed seeds are retained regardless of outcome.
""",
    )

    _set_source(
        notebook,
        "runtime-settings",
        '''from dataclasses import asdict
from synthetic_counting_v31.config import preset_config
from synthetic_counting_v29.config import preset_config as v29_preset_config
from synthetic_counting_v20.data import V20Vocab, load_corpus_text
from synthetic_counting_v20.model import build_model

VERSION = "v31"
PRESET = "main"
SEEDS = (1234, 2234, 3234)
DEVICE = "cuda"
OUT_ROOT = "runs/synthetic_counting_v31"
CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT
SKIP_COMPLETED = True
RUN_NAMES = {
    seed: f"v31_paired_joint_shared_L256_pool100_seed{seed}" for seed in SEEDS
}
RUN_DIRS = {seed: Path(OUT_ROOT) / RUN_NAMES[seed] for seed in SEEDS}
DRIVE_RUN_DIRS = {seed: CHECKPOINT_SYNC_ROOT / RUN_NAMES[seed] for seed in SEEDS}

planned = preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
baseline = v29_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
changed_fields = {
    key for key, value in asdict(planned).items()
    if asdict(baseline).get(key) != value
}
assert changed_fields == {"version", "batch_size", "training_mode_coupling"}, changed_fields
assert planned.seq_len == 256
assert planned.max_render_len == 287
assert planned.n_positions == 384
assert planned.count_max_threshold == 10
assert planned.needle_pool_size == 100
assert planned.training_count_distribution == "uniform"
assert planned.batch_size == 256
assert planned.training_mode_coupling == "paired_joint"
assert planned.enabled_model_variants == ("rope/nonthinking", "rope/thinking")
assert planned.trace_format == "separator"
assert planned.task_output_loss_reduction == "component_normalized"
assert planned.tie_word_embeddings
assert planned.untie_atomic_count_readout
assert planned.task_output_count_weight == 4.0
assert planned.task_output_trace_weight == 1.0
assert planned.task_output_structure_weight == 0.1
assert planned.answer_query_contrastive_weight == 0.0

# Step-zero audit: shared training changes neither parameters nor logits.
text = load_corpus_text()
baseline_vocab = V20Vocab.build(baseline, text)
planned_vocab = V20Vocab.build(planned, text)
assert baseline_vocab == planned_vocab
baseline_model = build_model(baseline, baseline_vocab, device="cpu").eval()
planned_model = build_model(planned, planned_vocab, device="cpu").eval()
probe_ids = torch.tensor([[baseline_vocab.token_to_id["<BOS>"], baseline_vocab.token_to_id["<Ans>"]]])
with torch.no_grad():
    max_step0_logit_diff = float(
        (baseline_model(probe_ids).logits - planned_model(probe_ids).logits).abs().max()
    )
assert max_step0_logit_diff == 0.0
parameter_delta = planned_model.parameter_count() - baseline_model.parameter_count()
assert parameter_delta == 0
del baseline_model, planned_model

print({
    "controlled_changes_from_v29": sorted(changed_fields),
    "seeds": SEEDS,
    "trace": "unchanged (<Sep> marker) repeated n times",
    "semantic_examples_per_step": planned.batch_size // 2,
    "rows_per_mode_per_step": planned.batch_size // 2,
    "mode_loss_reduction": "equal mean after independent per-mode reduction",
    "shared_checkpoint": True,
    "parameter_delta_from_v29": parameter_delta,
    "step0_max_logit_difference": max_step0_logit_diff,
    "posthoc_calibration": False,
})

def command_for(seed):
    command = [
        sys.executable, "-u", "-m", "synthetic_counting_v31.run_v31",
        "--preset", PRESET,
        "--device", DEVICE,
        "--seed", str(seed),
        "--train-steps", "10000",
        "--max-steps-for-language-pred", "1500",
        "--checkpoint-every", "100",
        "--recovery-every", "500",
        "--snapshot-shard-every", "500",
        "--eval-every", "500",
        "--ar-eval-every", "1000",
        "--ar-examples-per-count", "2",
        "--permutation-examples-per-count", "1",
        "--eval-examples-per-count", "10",
        "--final-examples-per-count", "50",
        "--phase-head-selection-examples-per-count", "2",
        "--phase-examples-per-count", "1",
        "--out-root", OUT_ROOT,
        "--run-name", RUN_NAMES[seed],
        "--checkpoint-sync-root", str(CHECKPOINT_SYNC_ROOT),
    ]
    if SKIP_COMPLETED:
        command.append("--skip-completed")
    return command
''',
    )

    _set_source(
        notebook,
        "train",
        '''training_started = time.perf_counter()
for seed in SEEDS:
    print(f"\\n===== shared paired-mode seed {seed} =====", flush=True)
    run_streaming([*command_for(seed), "--stage", "train"])

    metrics = pd.read_csv(RUN_DIRS[seed] / "tables" / "train_metrics.csv")
    assert set(metrics["mode"]) == {"joint"}
    assert metrics["training_mode_coupling"].eq("paired_joint").all()
    assert metrics["batch_nonthinking_rows"].eq(128).all()
    assert metrics["batch_thinking_rows"].eq(128).all()

    sampling = pd.read_csv(RUN_DIRS[seed] / "tables" / "training_sampling_distribution.csv")
    accepted = sampling[
        sampling["dimension"].eq("accepted_counts") & sampling["mode"].eq("joint")
    ].copy()
    accepted["value"] = accepted["value"].astype(int)
    accepted = accepted.sort_values("value")
    assert accepted["value"].tolist() == list(range(1, 11))
    relative_error = (
        accepted["examples"].sub(accepted["examples"].mean()).abs()
        / accepted["examples"].mean()
    )
    assert float(relative_error.max()) < 0.01

    specifications = pd.read_csv(RUN_DIRS[seed] / "tables" / "model_specifications.csv")
    assert specifications["shared_checkpoint"].all()
    assert set(specifications["checkpoint_storage_mode"]) == {"thinking"}
    display(pd.read_csv(RUN_DIRS[seed] / "tables" / "final_autoregressive_summary.csv"))
print(f"All shared paired-mode training: {time.perf_counter() - training_started:.1f} seconds")
''',
    )

    _set_source(
        notebook,
        "finish",
        '''import json

required = []
for seed in SEEDS:
    drive_run = DRIVE_RUN_DIRS[seed]
    required.extend([
        drive_run / "config.json",
        drive_run / "manifest.json",
        drive_run / "checkpoints" / "rope" / "thinking" / "final" / "checkpoint.pt",
        drive_run / "tables" / "model_specifications.csv",
        drive_run / "tables" / "final_autoregressive_summary.csv",
        drive_run / "tables" / "final_autoregressive_by_count.csv",
    ])
missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
assert not missing, missing
for seed in SEEDS:
    manifest = json.loads((DRIVE_RUN_DIRS[seed] / "manifest.json").read_text())
    assert manifest["version"] == "v31"
    assert manifest["stages"]["train"]["status"] == "complete"
print("Drive persistence verified for all seeds:", DRIVE_RUN_DIRS)

if Path("/content").exists():
    print("Experiment and persistence checks complete; disconnecting in 10 seconds.")
    time.sleep(10)
    from google.colab import runtime
    runtime.unassign()
''',
    )

    notebook["metadata"]["colab"]["name"] = TARGET.name
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
