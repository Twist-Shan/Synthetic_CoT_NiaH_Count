from __future__ import annotations

import json
from pathlib import Path

from build_v35_colab_notebook import build as build_v35_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v36_LR10k_Screen_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v35_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v35", "synthetic_counting_v36")
        source = source.replace("run_v35", "run_v36")
        source = source.replace('VERSION = "v35"', 'VERSION = "v36"')
        source = source.replace(
            'manifest["version"] == "v35"', 'manifest["version"] == "v36"'
        )
        source = source.replace(
            "v35_equalcomponents8_steps6000_independent_L256_pool100_seed",
            "v36_lr10k_steps6000_independent_L256_pool100_seed",
        )
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v36 screen: 6k independent runs on the 10k cosine horizon

v36 is a schedule-only control for v35. Non-thinking and Thinking remain two
separately initialized-and-trained models with identical step-0 parameters.
The 256-character prompt, counts 1--10, 100 marker sets, maximum-entropy
set/count sampler, separator/no-index trace, four-layer / four-head / width-256
architecture, partial count-only output untying, pure gold-prefix teacher
forcing, equal component-normalized count/trace/structure coefficients, seed,
and exactly 6,000 optimizer updates are unchanged.

The sole substantive change is the cosine decay horizon: v35 implicitly used
`lr_decay_steps = train_steps = 6000`, so its learning rate was exactly zero at
the screen endpoint. v36 keeps training at 6,000 updates but uses the original
v32 10,000-step cosine horizon. Its learning-rate trajectory through step 6,000
is therefore identical to v32, without adding updates, changing targets, or
selecting a checkpoint after looking at accuracy.

The fixed seed-1234 behavioral gate remains Thinking accuracy >= 0.90,
minimum per-count accuracy >= 0.80, trace exact >= 0.90, count spread <= 0.20,
and Thinking-minus-Non-thinking accuracy >= 0.10. A failed screen is retained
and rejected; NCC and mechanism experiments run only after the gate passes.
""",
    )

    _set_source(
        notebook,
        "runtime-settings",
        """from dataclasses import asdict
from synthetic_counting_v36.config import preset_config
from synthetic_counting_v35.config import preset_config as v35_preset_config
from synthetic_counting_v32.config import preset_config as v32_preset_config
from synthetic_counting_v20.data import V20Vocab, load_corpus_text
from synthetic_counting_v20.model import build_model
from synthetic_counting_v20.training import learning_rate

VERSION = "v36"
PRESET = "main"
SEEDS = (1234,)
DEVICE = "cuda"
OUT_ROOT = "runs/synthetic_counting_v36"
CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT
SKIP_COMPLETED = True
RUN_NAMES = {
    seed: f"v36_lr10k_steps6000_independent_L256_pool100_seed{seed}" for seed in SEEDS
}
RUN_DIRS = {seed: Path(OUT_ROOT) / RUN_NAMES[seed] for seed in SEEDS}
DRIVE_RUN_DIRS = {seed: CHECKPOINT_SYNC_ROOT / RUN_NAMES[seed] for seed in SEEDS}

planned = preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
baseline = v35_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
v32_schedule = v32_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
changed_fields = {
    key for key, value in asdict(planned).items()
    if asdict(baseline).get(key) != value
}
assert changed_fields == {"version", "lr_decay_steps"}, changed_fields
assert planned.seq_len == baseline.seq_len == 256
assert planned.max_render_len == baseline.max_render_len == 287
assert planned.n_positions == baseline.n_positions == 384
assert planned.count_max_threshold == baseline.count_max_threshold == 10
assert planned.needle_pool_size == baseline.needle_pool_size == 100
assert planned.training_count_distribution == baseline.training_count_distribution == "maxent_set_count"
assert planned.batch_size == baseline.batch_size == 128
assert planned.trace_format == baseline.trace_format == "separator"
assert planned.task_output_loss_reduction == baseline.task_output_loss_reduction == "component_normalized"
assert planned.tie_word_embeddings and baseline.tie_word_embeddings
assert planned.untie_atomic_count_readout and baseline.untie_atomic_count_readout
assert planned.task_output_count_weight == baseline.task_output_count_weight == 8.0
assert planned.task_output_trace_weight == baseline.task_output_trace_weight == 8.0
assert planned.task_output_structure_weight == baseline.task_output_structure_weight == 8.0
assert planned.task_output_scheduled_sampling_max_probability == 0.0
assert baseline.task_output_scheduled_sampling_max_probability == 0.0
assert planned.answer_query_contrastive_weight == baseline.answer_query_contrastive_weight == 0.0
assert planned.train_steps == baseline.train_steps == 6000
assert baseline.lr_decay_steps is None
assert planned.lr_decay_steps == 10000
assert planned.phase_cloud_steps == baseline.phase_cloud_steps
for step in (1, 500, 1500, 4000, 6000):
    assert abs(learning_rate(planned, step) - learning_rate(v32_schedule, step)) < 1e-15
assert learning_rate(baseline, 6000) == 0.0
assert learning_rate(planned, 6000) > 0.0

# Step-zero audit: an optimizer schedule cannot alter parameters or logits.
text = load_corpus_text()
baseline_vocab = V20Vocab.build(baseline, text)
planned_vocab = V20Vocab.build(planned, text)
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
    "controlled_changes_from_v35": sorted(changed_fields),
    "seeds": SEEDS,
    "trace": "unchanged (<Sep> marker) repeated n times",
    "loss": "unchanged component-normalized CE; count 8, trace 8, structure 8",
    "sampler": "unchanged maximum-entropy feasible set x count cells",
    "teacher_forcing": "pure gold-prefix inputs; scheduled sampling 0",
    "optimizer_updates": 6000,
    "cosine_decay_horizon": planned.lr_decay_steps,
    "v35_lr_at_6000": learning_rate(baseline, 6000),
    "v36_lr_at_6000": learning_rate(planned, 6000),
    "parameter_delta_from_v35": parameter_delta,
    "step0_max_logit_difference": max_step0_logit_diff,
    "posthoc_calibration": False,
})

def command_for(seed):
    command = [
        sys.executable, "-u", "-m", "synthetic_counting_v36.run_v36",
        "--preset", PRESET,
        "--device", DEVICE,
        "--seed", str(seed),
        "--train-steps", "6000",
        "--lr-decay-steps", "10000",
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
""",
    )

    notebook["metadata"]["colab"]["name"] = TARGET.name
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
