from __future__ import annotations

import json
from pathlib import Path

from build_v41_colab_notebook import build as build_v41_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v42_Width384_Steps8000_Screen_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v41_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v41", "synthetic_counting_v42")
        source = source.replace("run_v41", "run_v42")
        source = source.replace('VERSION = "v41"', 'VERSION = "v42"')
        source = source.replace(
            'manifest["version"] == "v41"', 'manifest["version"] == "v42"'
        )
        source = source.replace(
            "v41_count1to5_width384_heads6_steps6000_independent_L256_pool100_seed",
            "v42_count1to5_width384_heads6_steps8000_independent_L256_pool100_seed",
        )
        source = source.replace('role_table["step"].eq(6000)', 'role_table["step"].eq(8000)')
        source = source.replace(
            "marker_sets_identical_to_v40", "marker_sets_identical_to_v41"
        )
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v42 screen: 8,000-step optimization-horizon control

v42 is a one-variable optimization control on top of v41. Non-thinking and
Thinking are again separately initialized and trained from scratch. The
256-character prompt, count support 1--5, exact 100 three-character marker
sets, maximum-entropy set/count sampler, separator/no-index trace, partial
count-only output untying, gold-prefix teacher forcing, equal
component-normalized count/trace/structure coefficients, four-layer/six-head/
width-384 architecture, batch size, warmup, peak learning rate, clipping,
seed, and inference are unchanged.

The only substantive change is `train_steps: 6000 -> 8000`. With no explicit
`lr_decay_steps`, the existing cosine rule therefore has an 8,000-step horizon.
Both models restart at step zero; v42 does not continue v41. This directly
tests the predeclared explanation that v41's late Thinking transition had not
fully converged. There is no curriculum, scheduled sampling, auxiliary
objective, shared model, checkpoint selection, calibration, or test-time
update, and the trace text and targets are byte-for-byte unchanged.

The preregistered behavioral gate remains Thinking accuracy >= 0.90, minimum
per-count accuracy >= 0.80, trace exact >= 0.90, count spread <= 0.20, and
Thinking-minus-Non-thinking accuracy >= 0.10. NCC and mechanism experiments
run only if the final 8,000-step screen passes.
""",
    )

    _set_source(
        notebook,
        "settings-heading",
        "## 3. Audit the one-variable optimization-horizon control\n",
    )

    _set_source(
        notebook,
        "runtime-settings",
        """from dataclasses import asdict
from synthetic_counting_v42.config import preset_config
from synthetic_counting_v41.config import preset_config as v41_preset_config
from synthetic_counting_v20.data import V20Vocab, load_corpus_text
from synthetic_counting_v20.model import build_model
from synthetic_counting_v20.training import learning_rate

VERSION = "v42"
PRESET = "main"
SEEDS = (1234,)
DEVICE = "cuda"
OUT_ROOT = "runs/synthetic_counting_v42"
CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT
SKIP_COMPLETED = True
RUN_NAMES = {
    seed: f"v42_count1to5_width384_heads6_steps8000_independent_L256_pool100_seed{seed}" for seed in SEEDS
}
RUN_DIRS = {seed: Path(OUT_ROOT) / RUN_NAMES[seed] for seed in SEEDS}
DRIVE_RUN_DIRS = {seed: CHECKPOINT_SYNC_ROOT / RUN_NAMES[seed] for seed in SEEDS}

planned = preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
baseline = v41_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
changed_fields = {
    key for key, value in asdict(planned).items()
    if asdict(baseline).get(key) != value
}
assert changed_fields == {"version", "train_steps", "phase_cloud_steps"}, changed_fields
assert planned.count_max_threshold == baseline.count_max_threshold == 5
assert planned.seq_len == baseline.seq_len == 256
assert planned.max_render_len == baseline.max_render_len == 277
assert planned.n_positions == baseline.n_positions == 384
assert planned.needle_pool_size == baseline.needle_pool_size == 100
assert planned.needle_pool_frequency_threshold == baseline.needle_pool_frequency_threshold == 10.0 / 256.0
assert planned.effective_needle_pool_seed == baseline.effective_needle_pool_seed
assert planned.training_count_distribution == baseline.training_count_distribution == "maxent_set_count"
assert planned.batch_size == baseline.batch_size == 128
assert planned.trace_format == baseline.trace_format == "separator"
assert planned.task_output_loss_reduction == baseline.task_output_loss_reduction == "component_normalized"
assert planned.max_steps_for_language_pred == baseline.max_steps_for_language_pred == 1500
assert planned.tie_word_embeddings and baseline.tie_word_embeddings
assert planned.untie_atomic_count_readout and baseline.untie_atomic_count_readout
assert planned.task_output_count_weight == baseline.task_output_count_weight == 8.0
assert planned.task_output_trace_weight == baseline.task_output_trace_weight == 8.0
assert planned.task_output_structure_weight == baseline.task_output_structure_weight == 8.0
assert planned.task_output_scheduled_sampling_max_probability == baseline.task_output_scheduled_sampling_max_probability == 0.0
assert planned.answer_query_contrastive_weight == baseline.answer_query_contrastive_weight == 0.0
assert planned.lr == baseline.lr == 3e-4
assert planned.min_lr == baseline.min_lr == 0.0
assert planned.lr_decay_steps == baseline.lr_decay_steps is None
assert baseline.train_steps == 6000
assert planned.train_steps == 8000
assert learning_rate(baseline, 6000) == 0.0
assert learning_rate(planned, 6000) > 0.0
assert learning_rate(planned, 8000) == 0.0
assert (planned.n_layer, planned.n_head, planned.n_embd, planned.n_inner) == (
    baseline.n_layer, baseline.n_head, baseline.n_embd, baseline.n_inner
) == (4, 6, 384, 1536)

text = load_corpus_text()
baseline_vocab = V20Vocab.build(baseline, text)
planned_vocab = V20Vocab.build(planned, text)
baseline_model = build_model(baseline, baseline_vocab, device="cpu").eval()
planned_model = build_model(planned, planned_vocab, device="cpu").eval()
assert planned_model.parameter_count() == baseline_model.parameter_count() == 7_130_496
del baseline_model, planned_model

print({
    "controlled_changes_from_v41": sorted(changed_fields),
    "seeds": SEEDS,
    "models": "two independently reinitialized mode-specific models",
    "trace": "unchanged (<Sep> marker) repeated n times",
    "count_support": "1..5 (unchanged from v41)",
    "architecture": "4 layers, 6 heads, width 384, head dimension 64 (unchanged)",
    "sampler": "same maximum-entropy feasible set x count family",
    "marker_pool_spec": "exact v41 threshold, size, and seed; set equality checked after prepare",
    "loss": "unchanged v41 schedule and component-normalized CE; count 8, trace 8, structure 8",
    "optimizer_updates": "6000 -> 8000 from fresh initialization",
    "cosine_horizon": "6000 -> 8000; same warmup, peak LR, and zero endpoint",
    "parameters": 7_130_496,
    "checkpoint_selection": False,
    "posthoc_calibration": False,
})

def command_for(seed):
    command = [
        sys.executable, "-u", "-m", "synthetic_counting_v42.run_v42",
        "--preset", PRESET,
        "--device", DEVICE,
        "--seed", str(seed),
        "--train-steps", "8000",
        "--max-steps-for-language-pred", "1500",
        "--n-layer", "4",
        "--n-head", "6",
        "--n-embd", "384",
        "--n-inner", "1536",
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
