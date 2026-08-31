from __future__ import annotations

import json
from pathlib import Path
import textwrap

from build_v35_colab_notebook import build as build_v35_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v38_Mild_Rollin_Screen_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v35_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v35", "synthetic_counting_v38")
        source = source.replace("run_v35", "run_v38")
        source = source.replace('VERSION = "v35"', 'VERSION = "v38"')
        source = source.replace(
            'manifest["version"] == "v35"', 'manifest["version"] == "v38"'
        )
        source = source.replace(
            "v35_equalcomponents8_steps6000_independent_L256_pool100_seed",
            "v38_rollin0p1_equalcomponents_steps6000_independent_L256_pool100_seed",
        )
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v38 screen: independent models with mild trace roll-in

v38 is a one-scalar training follow-up to v35. Non-thinking and Thinking remain
two separately initialized-and-trained models with identical step-0
parameters. The 256-character prompt, counts 1--10, 100 marker sets,
maximum-entropy set/count sampler, separator/no-index trace, architecture,
partial count-only output untying, gold targets, equal component-normalized
count/trace/structure coefficients, optimizer, 6,000-step schedule, and seed
are unchanged.

v37 showed that extending the v35 trajectory with 2,000 low-learning-rate
updates did not close the teacher-forced/free-running gap: Thinking final
accuracy was 0.584 and trace exact was 0.480. v38 therefore tests the direct
exposure-bias hypothesis. During the task-output phase only, Thinking linearly
replaces eligible gold continuation inputs with its own preceding predictions,
from probability 0 at step 1,500 to a mild maximum of 0.1 at step 6,000. The
serialized trace and every target remain unchanged. Non-thinking has no
eligible intermediate trace positions, so its optimization path is the v35
baseline. There is no shared model, auxiliary loss, checkpoint selection,
post-hoc calibration, or inference-time change.

The behavioral gate remains Thinking accuracy >= 0.90, minimum per-count
accuracy >= 0.80, trace exact >= 0.90, count spread <= 0.20, and
Thinking-minus-Non-thinking accuracy >= 0.10. NCC and mechanism experiments
run only if the final 6,000-step screen passes.
""",
    )

    _set_source(
        notebook,
        "runtime-settings",
        """from dataclasses import asdict
from synthetic_counting_v38.config import preset_config
from synthetic_counting_v35.config import preset_config as v35_preset_config
from synthetic_counting_v20.data import V20Vocab, load_corpus_text
from synthetic_counting_v20.model import build_model
from synthetic_counting_v20.training import scheduled_sampling_probability

VERSION = "v38"
PRESET = "main"
SEEDS = (1234,)
DEVICE = "cuda"
OUT_ROOT = "runs/synthetic_counting_v38"
CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT
SKIP_COMPLETED = True
RUN_NAMES = {
    seed: f"v38_rollin0p1_equalcomponents_steps6000_independent_L256_pool100_seed{seed}" for seed in SEEDS
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
    "version", "task_output_scheduled_sampling_max_probability",
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
assert baseline.task_output_scheduled_sampling_max_probability == 0.0
assert planned.task_output_scheduled_sampling_max_probability == 0.1
assert planned.answer_query_contrastive_weight == baseline.answer_query_contrastive_weight == 0.0
assert planned.train_steps == baseline.train_steps == 6000
assert planned.lr == baseline.lr == 3e-4
assert planned.min_lr == baseline.min_lr == 0.0
assert scheduled_sampling_probability(planned, 1500, "thinking") == 0.0
assert scheduled_sampling_probability(planned, 3750, "thinking") == 0.05
assert scheduled_sampling_probability(planned, 6000, "thinking") == 0.1
assert scheduled_sampling_probability(planned, 6000, "nonthinking") == 0.0

# The training-only roll-in scalar cannot change step-zero parameters/logits.
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
    "targets": "unchanged gold trace/count targets",
    "loss": "unchanged component-normalized CE; count 8, trace 8, structure 8",
    "sampler": "unchanged maximum-entropy feasible set x count cells",
    "thinking_rollin": "linear 0 -> 0.1 after step 1500",
    "nonthinking_rollin": 0.0,
    "optimizer_updates": planned.train_steps,
    "parameter_delta_from_v35": parameter_delta,
    "step0_max_logit_difference": max_step0_logit_diff,
    "checkpoint_selection": False,
    "posthoc_calibration": False,
})

def command_for(seed):
    command = [
        sys.executable, "-u", "-m", "synthetic_counting_v38.run_v38",
        "--preset", PRESET,
        "--device", DEVICE,
        "--seed", str(seed),
        "--train-steps", "6000",
        "--max-steps-for-language-pred", "1500",
        "--task-output-scheduled-sampling-max-probability", "0.1",
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

    # Colab's Run all must never launch expensive post-hoc work for a rejected
    # behavioral screen.
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
