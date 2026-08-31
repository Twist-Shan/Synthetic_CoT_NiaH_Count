from __future__ import annotations

import json
from pathlib import Path

from build_v42_colab_notebook import build as build_v42_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v43_FullStarts_Screen_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v42_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v42", "synthetic_counting_v43")
        source = source.replace("run_v42", "run_v43")
        source = source.replace('VERSION = "v42"', 'VERSION = "v43"')
        source = source.replace(
            'manifest["version"] == "v42"', 'manifest["version"] == "v43"'
        )
        source = source.replace(
            "v42_count1to5_width384_heads6_steps8000_independent_L256_pool100_seed",
            "v43_count1to5_width384_heads6_steps8000_fullstarts_independent_L256_pool100_seed",
        )
        source = source.replace(
            "marker_sets_identical_to_v41", "marker_sets_identical_to_v42"
        )
        source = source.replace('"v28_package"', '"v43_package"')
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v43 screen: exact full-support within-cell sampling

v43 is a one-variable data-support control on top of v42. Non-thinking and
Thinking are again separately initialized and trained from scratch. The
256-character prompt, count support 1--5, exact 100 three-character marker
sets, maximum-entropy set/count cell probabilities, separator/no-index trace,
partial count-only output untying, gold-prefix teacher forcing, equal
component-normalized count/trace/structure coefficients, four-layer/six-head/
width-384 architecture, batch size, optimizer, 8,000-step schedule, seed, and
inference are unchanged.

The only substantive change is the support used after a set x count cell has
been selected. v42 deterministically retained at most 8,192 evenly spaced
legal starts per cell, while the fixed evaluation suites sampled from the full
corpus region. v43 uniformly samples from every legal start in the selected
cell. The notebook audits that every feasible cell has
`retained_window_count == full_window_count` before training and again from
the training artifact. Count/set marginals and all trace text remain unchanged.

This directly tests whether v42's high dynamic-batch fit but poor fixed-suite
final-count cross-entropy was caused by a finite within-cell shortcut. There
is no curriculum, loss reweighting, scheduled sampling, auxiliary objective,
shared model, checkpoint selection, calibration, or test-time update.

The preregistered behavioral gate remains Thinking accuracy >= 0.90, minimum
per-count accuracy >= 0.80, trace exact >= 0.90, count spread <= 0.20, and
Thinking-minus-Non-thinking accuracy >= 0.10. NCC and mechanism experiments
run only if the final 8,000-step screen passes.
""",
    )

    _set_source(
        notebook,
        "settings-heading",
        "## 3. Audit the one-variable full-support sampler control\n",
    )

    _set_source(
        notebook,
        "runtime-settings",
        """from dataclasses import asdict
from synthetic_counting_v43.config import preset_config
from synthetic_counting_v42.config import preset_config as v42_preset_config
from synthetic_counting_v20.data import V20Vocab, load_corpus_text
from synthetic_counting_v20.model import build_model
from synthetic_counting_v20.training import learning_rate

VERSION = "v43"
PRESET = "main"
SEEDS = (1234,)
DEVICE = "cuda"
OUT_ROOT = "runs/synthetic_counting_v43"
CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT
SKIP_COMPLETED = True
RUN_NAMES = {
    seed: f"v43_count1to5_width384_heads6_steps8000_fullstarts_independent_L256_pool100_seed{seed}" for seed in SEEDS
}
RUN_DIRS = {seed: Path(OUT_ROOT) / RUN_NAMES[seed] for seed in SEEDS}
DRIVE_RUN_DIRS = {seed: CHECKPOINT_SYNC_ROOT / RUN_NAMES[seed] for seed in SEEDS}

planned = preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
baseline = v42_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
planned_values = asdict(planned)
baseline_values = asdict(baseline)
changed_fields = {
    key for key, value in planned_values.items()
    if baseline_values.get(key) != value
}
assert changed_fields == {"version", "joint_sampler_max_starts_per_cell"}, changed_fields
assert baseline.joint_sampler_max_starts_per_cell == 8192
assert planned.joint_sampler_max_starts_per_cell is None
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
assert planned.train_steps == baseline.train_steps == 8000
assert planned.phase_cloud_steps == baseline.phase_cloud_steps
for step in (0, 500, 1500, 4000, 6000, 8000):
    assert learning_rate(planned, step) == learning_rate(baseline, step)
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
    "controlled_changes_from_v42": sorted(changed_fields),
    "seeds": SEEDS,
    "models": "two independently reinitialized mode-specific models",
    "trace": "unchanged (<Sep> marker) repeated n times",
    "count_support": "1..5 (unchanged from v42)",
    "architecture": "4 layers, 6 heads, width 384, head dimension 64 (unchanged)",
    "cell_distribution": "same maximum-entropy feasible set x count probabilities",
    "within_cell_sampler": "uniform over all legal starts; v42 cap 8192 removed",
    "marker_pool_spec": "exact v42 threshold, size, and seed; set equality checked after prepare",
    "loss": "unchanged v42 schedule and component-normalized CE; count 8, trace 8, structure 8",
    "optimizer_updates": "8000 from fresh initialization for each mode",
    "parameters": 7_130_496,
    "checkpoint_selection": False,
    "posthoc_calibration": False,
})

def command_for(seed):
    command = [
        sys.executable, "-u", "-m", "synthetic_counting_v43.run_v43",
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

    _set_source(
        notebook,
        "prepare-data",
        """for seed in SEEDS:
    run_streaming([*command_for(seed), "--stage", "prepare"])
    print("Prepared:", RUN_DIRS[seed].resolve(), "->", DRIVE_RUN_DIRS[seed])


# Pre-training sampler audit. This inspects legal training-region support only;
# it does not inspect held-out examples or labels.
import gc
import numpy as np

from synthetic_counting_v20.pipeline import load_prepared_v20_data
from synthetic_counting_v20.training import (
    _JOINT_SET_COUNT_SAMPLER_CACHE,
    _joint_set_count_sampler,
)

audit_vocab = V20Vocab.build(planned, text)
audit_split, audit_pool, _, _ = load_prepared_v20_data(
    planned, audit_vocab, text, RUN_DIRS[SEEDS[0]]
)
joint = _joint_set_count_sampler(planned, text, audit_split, audit_pool)
sampler_plan = joint.plan.copy()
feasible_plan = sampler_plan[sampler_plan["feasible"]].copy()
assert len(feasible_plan)
assert feasible_plan["full_window_count"].eq(
    feasible_plan["retained_window_count"]
).all()
assert sampler_plan["within_cell_sampling_policy"].eq("all_legal_starts").all()

support_by_count = feasible_plan.groupby("count", as_index=False)[
    ["full_window_count", "retained_window_count"]
].sum()
support_by_count["retention_fraction"] = (
    support_by_count["retained_window_count"]
    / support_by_count["full_window_count"]
)
assert support_by_count["retention_fraction"].eq(1.0).all()
display(support_by_count)
print({
    "feasible_cells": int(len(feasible_plan)),
    "full_legal_starts": int(feasible_plan["full_window_count"].sum()),
    "retained_legal_starts": int(feasible_plan["retained_window_count"].sum()),
    "retention_fraction": 1.0,
    "within_cell_sampling_policy": "all_legal_starts",
})

# The max-entropy target distribution remains low-shortcut and unchanged;
# only the within-cell corpus support differs from v42.
pivot = sampler_plan.pivot(
    index="set_id", columns="count", values="target_probability"
).fillna(0.0)
p = pivot.to_numpy(dtype=float)
ps = p.sum(axis=1, keepdims=True)
pc = p.sum(axis=0, keepdims=True)
mask = p > 0
target_mi_bits = float((p[mask] * np.log2((p / (ps @ pc))[mask])).sum())
target_set_only_bayes = float(p.max(axis=1).sum())
assert target_mi_bits < 0.07
assert target_set_only_bayes < 0.24
print({
    "maxent_target_set_count_MI_bits": target_mi_bits,
    "maxent_target_set_only_Bayes_accuracy": target_set_only_bayes,
    "chance": 0.20,
})

from synthetic_counting_v20.data import build_corpus_split
from synthetic_counting_v20.needle_pool import build_needle_pool
baseline_split = build_corpus_split(baseline, text)
baseline_pool = build_needle_pool(
    baseline, text, baseline_split, baseline_vocab.fingerprint
)
assert [item.characters for item in audit_pool.sets] == [
    item.characters for item in baseline_pool.sets
]
print({"marker_sets_identical_to_v42": True, "marker_set_count": len(audit_pool.sets)})

# Persist the preregistered support audit before optimization begins.
for seed in SEEDS:
    for root in (RUN_DIRS[seed], DRIVE_RUN_DIRS[seed]):
        audit_path = root / "tables" / "pretraining_full_support_audit.csv"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        sampler_plan.to_csv(audit_path, index=False)
_JOINT_SET_COUNT_SAMPLER_CACHE.clear()
del joint
gc.collect()
""",
    )

    _set_source(
        notebook,
        "train",
        """training_started = time.perf_counter()
for seed in SEEDS:
    print(f"\\n===== paired seed {seed} =====", flush=True)
    run_streaming([*command_for(seed), "--stage", "train"])

    plan = pd.read_csv(
        RUN_DIRS[seed] / "tables" / "training_set_count_sampler_plan.csv"
    )
    feasible = plan[plan["feasible"]].copy()
    assert feasible["full_window_count"].eq(
        feasible["retained_window_count"]
    ).all()
    assert plan["within_cell_sampling_policy"].eq("all_legal_starts").all()
    print({
        "seed": seed,
        "post_training_full_legal_starts": int(feasible["full_window_count"].sum()),
        "post_training_retained_legal_starts": int(feasible["retained_window_count"].sum()),
        "full_support_verified": True,
    })

    sampling = pd.read_csv(
        RUN_DIRS[seed] / "tables" / "training_sampling_distribution.csv"
    )
    accepted = sampling[sampling["dimension"].eq("accepted_counts")].copy()
    accepted["value"] = accepted["value"].astype(int)
    count_table = accepted.pivot(
        index="mode", columns="value", values="examples"
    ).sort_index(axis=1)
    assert list(count_table.columns) == list(range(1, 6))
    assert count_table.loc["nonthinking"].equals(count_table.loc["thinking"])
    relative_error = (
        count_table.sub(count_table.mean(axis=1), axis=0).abs()
        .div(count_table.mean(axis=1), axis=0)
    )
    assert float(relative_error.to_numpy().max()) < 0.01
    display(pd.read_csv(RUN_DIRS[seed] / "tables" / "final_autoregressive_summary.csv"))
print(f"All paired training: {time.perf_counter() - training_started:.1f} seconds")
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
