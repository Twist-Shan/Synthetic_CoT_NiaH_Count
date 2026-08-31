from __future__ import annotations

import json
from pathlib import Path
import textwrap

from build_v35_colab_notebook import build as build_v35_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v37_LowLRTail_Screen_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v35_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v35", "synthetic_counting_v37")
        source = source.replace("run_v35", "run_v37")
        source = source.replace('VERSION = "v35"', 'VERSION = "v37"')
        source = source.replace(
            'manifest["version"] == "v35"', 'manifest["version"] == "v37"'
        )
        source = source.replace(
            "v35_equalcomponents8_steps6000_independent_L256_pool100_seed",
            "v37_lowtail1em5_steps8000_independent_L256_pool100_seed",
        )
        source = source.replace('role_table["step"].eq(6000)', 'role_table["step"].eq(8000)')
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v37 screen: independent models with a low-LR consolidation tail

v37 is a schedule-only follow-up to v35. Non-thinking and Thinking remain two
separately initialized-and-trained models with identical step-0 parameters.
The 256-character prompt, counts 1--10, 100 marker sets, maximum-entropy
set/count sampler, separator/no-index trace, architecture, partial count-only
output untying, pure gold-prefix teacher forcing, equal component-normalized
count/trace/structure coefficients, and seed are unchanged.

v35 learned accurate teacher-forced readout around step 4,000 but its cosine
learning rate reached zero at step 6,000 while free-running trace length was
still improving. v36's 10,000-step horizon kept the mid-training rate too high
and destabilized final-count CE. v37 instead uses a standard cosine-to-minimum
schedule: it decays toward `1e-5` through step 6,000, then holds `1e-5` through
the predeclared 8,000-step endpoint. It adds no trace token, auxiliary loss,
roll-in, shared model, checkpoint selection, calibration, or inference rule.

The behavioral gate remains Thinking accuracy >= 0.90, minimum per-count
accuracy >= 0.80, trace exact >= 0.90, count spread <= 0.20, and
Thinking-minus-Non-thinking accuracy >= 0.10. NCC and mechanism experiments
run only if the final 8,000-step screen passes.
""",
    )

    _set_source(
        notebook,
        "runtime-settings",
        """from dataclasses import asdict
from synthetic_counting_v37.config import preset_config
from synthetic_counting_v35.config import preset_config as v35_preset_config
from synthetic_counting_v20.data import V20Vocab, load_corpus_text
from synthetic_counting_v20.model import build_model
from synthetic_counting_v20.training import learning_rate

VERSION = "v37"
PRESET = "main"
SEEDS = (1234,)
DEVICE = "cuda"
OUT_ROOT = "runs/synthetic_counting_v37"
CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT
SKIP_COMPLETED = True
RUN_NAMES = {
    seed: f"v37_lowtail1em5_steps8000_independent_L256_pool100_seed{seed}" for seed in SEEDS
}
RUN_DIRS = {seed: Path(OUT_ROOT) / RUN_NAMES[seed] for seed in SEEDS}
DRIVE_RUN_DIRS = {seed: CHECKPOINT_SYNC_ROOT / RUN_NAMES[seed] for seed in SEEDS}

planned = preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
baseline = v35_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
changed_fields = {
    key for key, value in asdict(planned).items()
    if asdict(baseline).get(key) != value
}
assert changed_fields == {
    "version", "train_steps", "lr_decay_steps", "min_lr", "phase_cloud_steps",
}, changed_fields
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
assert planned.task_output_scheduled_sampling_max_probability == baseline.task_output_scheduled_sampling_max_probability == 0.0
assert planned.answer_query_contrastive_weight == baseline.answer_query_contrastive_weight == 0.0
assert baseline.train_steps == 6000
assert baseline.lr_decay_steps is None
assert baseline.min_lr == 0.0
assert planned.train_steps == 8000
assert planned.lr_decay_steps == 6000
assert planned.min_lr == 1e-5
assert planned.phase_cloud_steps[-1] == 8000
assert learning_rate(baseline, 6000) == 0.0
assert learning_rate(planned, 6000) == planned.min_lr
assert learning_rate(planned, 7000) == planned.min_lr
assert learning_rate(planned, 8000) == planned.min_lr
for step in (1, 500, 1500, 3000, 4000):
    assert abs(learning_rate(planned, step) - learning_rate(baseline, step)) < 1e-5

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
    "models": "two independent mode-specific models",
    "trace": "unchanged (<Sep> marker) repeated n times",
    "loss": "unchanged component-normalized CE; count 8, trace 8, structure 8",
    "sampler": "unchanged maximum-entropy feasible set x count cells",
    "teacher_forcing": "pure gold-prefix inputs; scheduled sampling 0",
    "optimizer_updates": planned.train_steps,
    "cosine_decay_horizon": planned.lr_decay_steps,
    "minimum_lr": planned.min_lr,
    "parameter_delta_from_v35": parameter_delta,
    "step0_max_logit_difference": max_step0_logit_diff,
    "checkpoint_selection": False,
    "posthoc_calibration": False,
})

def command_for(seed):
    command = [
        sys.executable, "-u", "-m", "synthetic_counting_v37.run_v37",
        "--preset", PRESET,
        "--device", DEVICE,
        "--seed", str(seed),
        "--train-steps", "8000",
        "--lr-decay-steps", "6000",
        "--min-lr", "1e-5",
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

    # Make the stated behavioral gate executable, not merely descriptive.
    # Colab's Run all must never start expensive post-hoc analyses for a
    # rejected screen.
    ncc_source = "".join(_cell(notebook, "results")["source"])
    _set_source(
        notebook,
        "results",
        "if not behavior_gate:\n"
        '    print("NCC skipped: final behavioral gate failed")\n'
        "else:\n"
        + textwrap.indent(ncc_source, "    "),
    )
    mechanism_source = "".join(_cell(notebook, "mechanism")["source"])
    _set_source(
        notebook,
        "mechanism",
        "if not behavior_gate:\n"
        '    print("Mechanism analyses skipped: final behavioral gate failed")\n'
        "else:\n"
        + textwrap.indent(mechanism_source, "    "),
    )

    notebook["metadata"]["colab"]["name"] = TARGET.name
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
