from __future__ import annotations

import json
from pathlib import Path

from build_v40_colab_notebook import build as build_v40_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v41_Width384_Screen_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v40_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v40", "synthetic_counting_v41")
        source = source.replace("run_v40", "run_v41")
        source = source.replace('VERSION = "v40"', 'VERSION = "v41"')
        source = source.replace(
            'manifest["version"] == "v40"', 'manifest["version"] == "v41"'
        )
        source = source.replace(
            "v40_count1to5_equalcomponents_steps6000_independent_L256_pool100_seed",
            "v41_count1to5_width384_heads6_steps6000_independent_L256_pool100_seed",
        )
        source = source.replace(
            "marker_sets_identical_to_v35", "marker_sets_identical_to_v40"
        )
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v41 screen: parallel retrieval-capacity control

v41 is a single compound architecture control on top of v40. Non-thinking and
Thinking remain two separately initialized-and-trained models. The
256-character prompt, count support 1--5, exact 100 three-character marker
sets, maximum-entropy set/count sampler, separator/no-index trace, partial
count-only output untying, gold-prefix teacher forcing, equal
component-normalized count/trace/structure coefficients, optimizer, 6,000-step
schedule, seed, and inference are unchanged.

Serial depth remains four layers. Residual width grows 256 -> 384 while the
attention head dimension stays exactly 64 (4 -> 6 heads) and the MLP stays 4x
the residual width (1024 -> 1536). This tests whether v40's low free-running
trace/readout stability is a parallel retrieval-capacity bottleneck without
giving Non-thinking additional serial computation. The trace text and targets
are byte-for-byte unchanged. There is no curriculum, scheduled sampling,
auxiliary objective, shared model, checkpoint selection, calibration, or
test-time update.

The preregistered behavioral gate remains Thinking accuracy >= 0.90, minimum
per-count accuracy >= 0.80, trace exact >= 0.90, count spread <= 0.20, and
Thinking-minus-Non-thinking accuracy >= 0.10. NCC and mechanism experiments
run only if the final 6,000-step screen passes.
""",
    )

    _set_source(
        notebook,
        "runtime-settings",
        """from dataclasses import asdict
from synthetic_counting_v41.config import preset_config
from synthetic_counting_v40.config import preset_config as v40_preset_config
from synthetic_counting_v20.data import V20Vocab, load_corpus_text
from synthetic_counting_v20.model import build_model

VERSION = "v41"
PRESET = "main"
SEEDS = (1234,)
DEVICE = "cuda"
OUT_ROOT = "runs/synthetic_counting_v41"
CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT
SKIP_COMPLETED = True
RUN_NAMES = {
    seed: f"v41_count1to5_width384_heads6_steps6000_independent_L256_pool100_seed{seed}" for seed in SEEDS
}
RUN_DIRS = {seed: Path(OUT_ROOT) / RUN_NAMES[seed] for seed in SEEDS}
DRIVE_RUN_DIRS = {seed: CHECKPOINT_SYNC_ROOT / RUN_NAMES[seed] for seed in SEEDS}

planned = preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
baseline = v40_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
changed_fields = {
    key for key, value in asdict(planned).items()
    if asdict(baseline).get(key) != value
}
assert changed_fields == {"version", "n_head", "n_embd", "n_inner"}, changed_fields
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
assert planned.train_steps == baseline.train_steps == 6000
assert planned.lr == baseline.lr == 3e-4
assert planned.min_lr == baseline.min_lr == 0.0
assert (baseline.n_layer, baseline.n_head, baseline.n_embd, baseline.n_inner) == (4, 4, 256, 1024)
assert (planned.n_layer, planned.n_head, planned.n_embd, planned.n_inner) == (4, 6, 384, 1536)
assert baseline.n_embd // baseline.n_head == planned.n_embd // planned.n_head == 64

text = load_corpus_text()
baseline_vocab = V20Vocab.build(baseline, text)
planned_vocab = V20Vocab.build(planned, text)
baseline_model = build_model(baseline, baseline_vocab, device="cpu").eval()
planned_model = build_model(planned, planned_vocab, device="cpu").eval()
baseline_parameters = baseline_model.parameter_count()
planned_parameters = planned_model.parameter_count()
assert planned_parameters > baseline_parameters
del baseline_model, planned_model

print({
    "controlled_changes_from_v40": sorted(changed_fields),
    "seeds": SEEDS,
    "models": "two independent mode-specific models",
    "trace": "unchanged (<Sep> marker) repeated n times",
    "count_support": "1..5 (unchanged from v40)",
    "serial_depth": "4 layers (unchanged)",
    "parallel_capacity": "4x256 -> 6x384; head dimension remains 64",
    "sampler": "same maximum-entropy feasible set x count family",
    "marker_pool_spec": "exact v40 threshold, size, and seed; set equality checked after prepare",
    "loss": "unchanged v40 schedule and component-normalized CE; count 8, trace 8, structure 8",
    "optimizer_updates": planned.train_steps,
    "baseline_parameters": baseline_parameters,
    "planned_parameters": planned_parameters,
    "checkpoint_selection": False,
    "posthoc_calibration": False,
})

def command_for(seed):
    command = [
        sys.executable, "-u", "-m", "synthetic_counting_v41.run_v41",
        "--preset", PRESET,
        "--device", DEVICE,
        "--seed", str(seed),
        "--train-steps", "6000",
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
