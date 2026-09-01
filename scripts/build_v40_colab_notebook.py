from __future__ import annotations

import json
from pathlib import Path
import textwrap

from build_v35_colab_notebook import build as build_v35_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v40_Count5_Screen_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v35_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v35", "synthetic_counting_v40")
        source = source.replace("run_v35", "run_v40")
        source = source.replace('VERSION = "v35"', 'VERSION = "v40"')
        source = source.replace(
            'manifest["version"] == "v35"', 'manifest["version"] == "v40"'
        )
        source = source.replace(
            "v35_equalcomponents8_steps6000_independent_L256_pool100_seed",
            "v40_count1to5_equalcomponents_steps6000_independent_L256_pool100_seed",
        )
        source = source.replace("list(range(1, 11))", "list(range(1, 6))")
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v40 screen: count-range 1--5 difficulty control

v40 is a single-difficulty-variable control for v35. Non-thinking and Thinking
remain two separately initialized-and-trained models. The 256-character
prompt, exact 100 three-character marker sets, maximum-entropy set/count
sampler, separator/no-index trace, four-layer/four-head/width-256 architecture,
partial count-only output untying, gold-prefix teacher forcing, equal
component-normalized count/trace/structure coefficients, optimizer, 6,000-step
schedule, seed, and inference are unchanged.

The only substantive task change is count support `1..10 -> 1..5`. The marker
pool threshold and seed remain fixed, and the notebook explicitly verifies
that the 100 target-character sets are identical to v35. Thinking still emits
one unchanged `<Sep>` pair per counted marker before the atomic answer. Thus
the retrieval prompt remains 256 characters; only the longest required
no-index trace is reduced. There is no curriculum, scheduled sampling,
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
from synthetic_counting_v40.config import preset_config
from synthetic_counting_v35.config import preset_config as v35_preset_config
from synthetic_counting_v20.data import V20Vocab, load_corpus_text
from synthetic_counting_v20.model import build_model

VERSION = "v40"
PRESET = "main"
SEEDS = (1234,)
DEVICE = "cuda"
OUT_ROOT = "runs/synthetic_counting_v40"
CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT
SKIP_COMPLETED = True
RUN_NAMES = {
    seed: f"v40_count1to5_equalcomponents_steps6000_independent_L256_pool100_seed{seed}" for seed in SEEDS
}
RUN_DIRS = {seed: Path(OUT_ROOT) / RUN_NAMES[seed] for seed in SEEDS}
DRIVE_RUN_DIRS = {seed: CHECKPOINT_SYNC_ROOT / RUN_NAMES[seed] for seed in SEEDS}

planned = preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
baseline = v35_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
changed_fields = {
    key for key, value in asdict(planned).items()
    if asdict(baseline).get(key) != value
}
assert changed_fields == {"version", "count_max_threshold"}, changed_fields
assert planned.count_max_threshold == 5
assert baseline.count_max_threshold == 10
assert planned.seq_len == baseline.seq_len == 256
assert planned.max_render_len == 277
assert baseline.max_render_len == 287
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

text = load_corpus_text()
baseline_vocab = V20Vocab.build(baseline, text)
planned_vocab = V20Vocab.build(planned, text)
baseline_model = build_model(baseline, baseline_vocab, device="cpu").eval()
planned_model = build_model(planned, planned_vocab, device="cpu").eval()
parameter_delta = planned_model.parameter_count() - baseline_model.parameter_count()
assert parameter_delta < 0
del baseline_model, planned_model

print({
    "controlled_changes_from_v35": sorted(changed_fields),
    "seeds": SEEDS,
    "models": "two independent mode-specific models",
    "trace": "unchanged (<Sep> marker) repeated n times",
    "count_support": "1..5 (v35: 1..10)",
    "sampler": "same maximum-entropy feasible set x count family",
    "marker_pool_spec": "exact v35 threshold, size, and seed; set equality checked after prepare",
    "loss": "unchanged v35 schedule and component-normalized CE; count 8, trace 8, structure 8",
    "optimizer_updates": planned.train_steps,
    "parameter_delta_from_v35": parameter_delta,
    "checkpoint_selection": False,
    "posthoc_calibration": False,
})

def command_for(seed):
    command = [
        sys.executable, "-u", "-m", "synthetic_counting_v40.run_v40",
        "--preset", PRESET,
        "--device", DEVICE,
        "--seed", str(seed),
        "--train-steps", "6000",
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

    prepare_source = "".join(_cell(notebook, "prepare-data")["source"])
    prepare_source = prepare_source.replace(
        "assert target_set_only_bayes < 0.12",
        "assert target_set_only_bayes < 0.24",
    ).replace('"chance": 0.10', '"chance": 0.20')
    prepare_source += """

from synthetic_counting_v20.data import build_corpus_split
from synthetic_counting_v20.needle_pool import build_needle_pool
baseline_split = build_corpus_split(baseline, text)
baseline_pool = build_needle_pool(
    baseline, text, baseline_split, baseline_vocab.fingerprint
)
assert [item.characters for item in audit_pool.sets] == [
    item.characters for item in baseline_pool.sets
]
print({"marker_sets_identical_to_v35": True, "marker_set_count": len(audit_pool.sets)})
"""
    _set_source(notebook, "prepare-data", prepare_source)

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
