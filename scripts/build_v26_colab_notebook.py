from __future__ import annotations

import json
from pathlib import Path

from build_v25_colab_notebook import build as build_v25_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v26_L256_DiverseSet_Untied_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    # Reuse the already audited environment/streaming/evaluation skeleton, then
    # replace every scientific setting explicitly below.
    source_path = build_v25_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v25_1", "synthetic_counting_v26_1")
        source = source.replace("synthetic_counting_v25", "synthetic_counting_v26")
        source = source.replace("v25.1", "v26.1").replace("v25", "v26")
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v26/v26.1: 256-token diverse-set untied-head control

This paired experiment returns to the 256-character v24.3 regime, where the
historical held-out result was Thinking 0.632 versus Non-thinking 0.372.  Both
modes use the same 100 three-character marker sets, exactly count-balanced
training sampler, 4-layer/4-head transformer, separator/no-index trace,
component-normalized loss, optimizer, seed, examples, and 10,000-step
schedule.

The sole training change from v24.3 is an untied native LM head.  Its weights
are copied from the input embedding at step zero, so the initial function is
identical; later atomic-number output gradients cannot distort the input
number embeddings.  No answer-query contrastive or probe loss is used.  NCC
therefore remains a post-hoc measurement of an emergent count representation.

V26.1 then applies one validation-selected number-row calibration schedule to
both frozen backbones.  It can align an existing representation with native
number tokens, but cannot create missing retrieval or count information.  The
primary gate requires high, count-uniform Thinking accuracy, trace fidelity,
and a positive held-out Thinking-minus-Non-thinking accuracy gap.
""",
    )
    drive = "".join(_cell(notebook, "drive-login")["source"])
    drive = drive.replace(
        'drive.mount("/content/drive")',
        'drive.mount("/content/drive", timeout_ms=300000)',
    )
    _set_source(notebook, "drive-login", drive)
    _cell(notebook, "settings-heading")["source"] = [
        "## 3. Audit the 256-token paired setting\n"
    ]
    _set_source(
        notebook,
        "runtime-settings",
        """from dataclasses import asdict
from synthetic_counting_v26.config import preset_config
from synthetic_counting_v24_3.config import preset_config as v24_3_preset_config

VERSION = "v26"
PRESET = "main"
SEED = 1234
DEVICE = "cuda"
RUN_NAME = "v26_L256_pool100_uniform_count_untied_seed1234"
CALIBRATION_NAME = "v26.1_native_head_calibration_L256_pool100_seed2478"
OUT_ROOT = "runs/synthetic_counting_v26"
CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT
SKIP_COMPLETED = True

PLANNED_CONFIG = preset_config(PRESET, seed=SEED, device=DEVICE)
BASELINE_CONFIG = v24_3_preset_config(PRESET, seed=SEED, device=DEVICE)
changed_fields = {
    key for key, value in asdict(PLANNED_CONFIG).items()
    if asdict(BASELINE_CONFIG).get(key) != value
}
assert changed_fields == {"version", "tie_word_embeddings"}, changed_fields
assert PLANNED_CONFIG.seq_len == 256
assert PLANNED_CONFIG.max_render_len == 287
assert PLANNED_CONFIG.n_positions == 384
assert PLANNED_CONFIG.count_max_threshold == 10
assert PLANNED_CONFIG.needle_pool_size == 100
assert PLANNED_CONFIG.needle_pool_frequency_threshold == 10.0 / 256.0
assert PLANNED_CONFIG.training_count_distribution == "uniform"
assert PLANNED_CONFIG.batch_size == 128
assert PLANNED_CONFIG.enabled_model_variants == ("rope/nonthinking", "rope/thinking")
assert PLANNED_CONFIG.trace_format == "separator"
assert PLANNED_CONFIG.task_output_loss_reduction == "component_normalized"
assert not PLANNED_CONFIG.tie_word_embeddings
assert PLANNED_CONFIG.answer_query_contrastive_weight == 0.0
print({
    "controlled_changes_from_v24.3": sorted(changed_fields),
    "sequence_layout": "<BOS> query[5] data[256] output",
    "longest_thinking_sequence": PLANNED_CONFIG.max_render_len,
    "position_budget": PLANNED_CONFIG.n_positions,
    "answer_support": "1..10 atomic tokens",
    "trace": "(<Sep> marker) repeated n times",
    "sampler": "uniform semantic count; natural feasible-set exposure",
    "comparison_scope": "paired within-v26; identical examples and optimization settings",
})

RUN_DIR = Path(OUT_ROOT) / RUN_NAME
DRIVE_RUN_DIR = CHECKPOINT_SYNC_ROOT / RUN_NAME
CALIBRATION_DIR = DRIVE_RESULTS_ROOT / CALIBRATION_NAME
""",
    )
    _cell(notebook, "prepare-heading")["source"] = [
        "## 4. Prepare and audit the fixed 256-token data\n"
    ]
    _cell(notebook, "train-heading")["source"] = [
        "## 5. Train the paired 256-token models\n"
    ]
    _set_source(
        notebook,
        "train",
        """training_started = time.perf_counter()
run_streaming([*base_cmd, "--stage", "train"])
print(f"Paired training block: {time.perf_counter() - training_started:.1f} seconds")

sampling = pd.read_csv(RUN_DIR / "tables" / "training_sampling_distribution.csv")
accepted = sampling[sampling["dimension"].eq("accepted_counts")].copy()
count_table = accepted.pivot(index="mode", columns="value", values="examples").sort_index(axis=1)
assert list(count_table.columns.astype(int)) == list(range(1, 11))
assert count_table.loc["nonthinking"].equals(count_table.loc["thinking"])
count_relative_error = (
    count_table.sub(count_table.mean(axis=1), axis=0)
    .abs()
    .div(count_table.mean(axis=1), axis=0)
)
assert float(count_relative_error.to_numpy().max()) < 0.01

set_rows = sampling[sampling["dimension"].eq("set_ids")].copy()
set_table = set_rows.pivot(index="mode", columns="value", values="examples").sort_index(axis=1)
assert set_table.shape == (2, 100)
assert set_table.loc["nonthinking"].equals(set_table.loc["thinking"])
set_cv = set_table.std(axis=1) / set_table.mean(axis=1)
assert float(set_cv.max()) < 0.25
assert int(set_table.min(axis=1).min()) > 0
print({
    "maximum_count_relative_error": float(count_relative_error.to_numpy().max()),
    "set_exposure_cv": set_cv.to_dict(),
    "minimum_set_examples": int(set_table.min(axis=1).min()),
    "maximum_set_examples": int(set_table.max(axis=1).max()),
    "paired_sampling_exactly_matched": True,
})
display(pd.read_csv(RUN_DIR / "tables" / "final_autoregressive_summary.csv"))
display(pd.read_csv(RUN_DIR / "tables" / "final_autoregressive_by_count.csv"))
""",
    )
    _set_source(
        notebook,
        "calibration-heading",
        """## 6. Symmetric native-head calibration

The schedule is selected using validation prompts in Thinking mode and then
applied unchanged to the paired Non-thinking checkpoint.  Only the ten native
atomic-number rows can move; both transformers, input embeddings, attention
heads, and all hidden-state geometry remain frozen.  Test prompts are opened
once after both choices are fixed.
""",
    )
    calibration = "".join(_cell(notebook, "calibration")["source"])
    calibration = calibration.replace('"--batch-size", "32"', '"--batch-size", "128"')
    _set_source(notebook, "calibration", calibration)
    ncc = "".join(_cell(notebook, "ncc")["source"])
    ncc = ncc.replace('"--batch-size", "8"', '"--batch-size", "32"')
    _set_source(notebook, "ncc", ncc)

    notebook["metadata"]["colab"]["name"] = TARGET.name
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
