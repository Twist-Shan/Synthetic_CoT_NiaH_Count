from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v25_LongContext_RetrievalPressure_Colab.ipynb"


def _markdown(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build() -> Path:
    cells = [
        _markdown(
            "title",
            """# Trace Count v25/v25.1: long-context retrieval pressure

This paired experiment is designed to test a behavioral Thinking advantage
without deliberately undertraining Non-thinking.  Relative to v24.7, both
modes retain the same 4-layer/4-head transformer, count support 1–10, no-index
separator trace, 20-set maximum-entropy sampler, component-normalized loss,
untied native LM head, answer-query contrastive term, optimizer, seed, and
10,000-step schedule.  The substantive task change is a 1,024-character
Shakespeare prompt instead of 256 characters.

The marker-frequency cap is changed mechanically from `10/256` to `10/1024`,
so the answer range and trace length do not grow.  The test therefore asks
whether serial targeted retrieval preserves adjacent-count resolution when
the same 1–10 target occurrences must be found among four times as many prompt
tokens.  Batch size is reduced equally for both modes to fit the longer
sequence.

After paired training, v25.1 applies the same validation-selected native
number-row calibration to both modes while freezing their complete
transformers.  This removes avoidable atomic-token readout misalignment but
cannot create a count representation absent from the frozen backbone.  The
primary behavioral gate is high and count-uniform Thinking accuracy together
with a positive held-out Thinking-minus-Non-thinking accuracy gap.
""",
        ),
        _markdown("drive-heading", "## 1. Mount Google Drive"),
        _code(
            "drive-login",
            """from pathlib import Path

DRIVE_RESULTS_ROOT = Path(
    "/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/"
    "Synthetic_CoT_NiaH_Count/colab_results"
)
DRIVE_READY = False
if Path("/content").exists():
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    DRIVE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    DRIVE_READY = True
    print("Drive ready:", DRIVE_RESULTS_ROOT)
else:
    print("Local runtime: Drive mount skipped")
""",
        ),
        _markdown("setup-heading", "## 2. Clone the audited implementation and verify GPU"),
        _code(
            "environment-setup",
            """import os
import signal
import subprocess
import sys
import time
from pathlib import Path

assert DRIVE_READY, "Run the Drive cell first"
REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
REPO_REF = "agent/remove-misplaced-realistic-artifacts"
preferred = Path("/content/Synthetic_CoT_NiaH_Count")
candidates = [Path.cwd(), *Path.cwd().parents, preferred]
repo = next((path.resolve() for path in candidates if (path / "pyproject.toml").exists()), None)
if repo is None:
    subprocess.run(
        ["git", "clone", "--branch", REPO_REF, "--single-branch", REPO_URL, str(preferred)],
        check=True,
    )
    repo = preferred
elif (repo / ".git").exists():
    subprocess.run(["git", "-C", str(repo), "fetch", "origin", REPO_REF], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", REPO_REF], check=True)
    subprocess.run(["git", "-C", str(repo), "pull", "--ff-only", "origin", REPO_REF], check=True)
os.chdir(repo)

probe = subprocess.run(
    [sys.executable, "-c", "import numpy,pandas,scipy,matplotlib,seaborn"],
    capture_output=True,
    text=True,
)
if probe.returncode:
    print(probe.stderr[-2000:])
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
            "--force-reinstall", "numpy==1.26.4", "pandas==2.2.3",
            "scipy==1.13.1", "matplotlib==3.8.4", "seaborn==0.13.2",
        ],
        check=True,
    )
    os.kill(os.getpid(), signal.SIGKILL)
    raise RuntimeError("Scientific ABI repaired. Reconnect and rerun all cells.")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."], check=True)
src_root = str(repo / "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)
os.environ["PYTHONPATH"] = src_root + os.pathsep + os.environ.get("PYTHONPATH", "")

import codecs
import pandas as pd
import torch
import synthetic_counting_v25
import synthetic_counting_v25_1
from IPython.display import display

def run_streaming(command):
    command = [str(part) for part in command]
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    assert process.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        chunk = os.read(process.stdout.fileno(), 4096)
        if not chunk:
            break
        print(decoder.decode(chunk), end="", flush=True)
    print(decoder.decode(b"", final=True), end="", flush=True)
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print({
    "repo": str(repo),
    "repo_ref": REPO_REF,
    "repo_commit": commit,
    "v25_package": synthetic_counting_v25.__file__,
    "v25_1_package": synthetic_counting_v25_1.__file__,
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
assert torch.cuda.is_available()
""",
        ),
        _markdown("settings-heading", "## 3. Audit the retrieval-pressure setting"),
        _code(
            "runtime-settings",
            """from dataclasses import asdict
from synthetic_counting_v25.config import preset_config
from synthetic_counting_v24_7.config import preset_config as v24_7_preset_config

VERSION = "v25"
PRESET = "main"
SEED = 1234
DEVICE = "cuda"
RUN_NAME = "v25_long_context_L1024_pool20_count1-10_seed1234"
CALIBRATION_NAME = "v25.1_native_head_calibration_L1024_pool20_count1-10_seed2478"
OUT_ROOT = "runs/synthetic_counting_v25"
CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT
SKIP_COMPLETED = True

PLANNED_CONFIG = preset_config(PRESET, seed=SEED, device=DEVICE)
BASELINE_CONFIG = v24_7_preset_config(PRESET, seed=SEED, device=DEVICE)
changed_fields = {
    key for key, value in asdict(PLANNED_CONFIG).items()
    if asdict(BASELINE_CONFIG).get(key) != value
}
assert changed_fields == {
    "version", "seq_len", "n_positions", "needle_pool_frequency_threshold", "batch_size"
}, changed_fields
assert PLANNED_CONFIG.seq_len == 1024
assert PLANNED_CONFIG.max_render_len == 1055
assert PLANNED_CONFIG.n_positions == 1056
assert PLANNED_CONFIG.count_max_threshold == 10
assert PLANNED_CONFIG.needle_pool_frequency_threshold == 10.0 / 1024.0
assert PLANNED_CONFIG.enabled_model_variants == ("rope/nonthinking", "rope/thinking")
assert PLANNED_CONFIG.trace_format == "separator"
assert PLANNED_CONFIG.task_output_loss_reduction == "component_normalized"
assert not PLANNED_CONFIG.tie_word_embeddings
assert PLANNED_CONFIG.answer_query_contrastive_weight == 0.1
print({
    "controlled_changes_from_v24.7": sorted(changed_fields),
    "sequence_layout": "<BOS> query[5] data[1024] output",
    "longest_thinking_sequence": PLANNED_CONFIG.max_render_len,
    "position_budget": PLANNED_CONFIG.n_positions,
    "answer_support": "1..10 atomic tokens",
    "trace": "(<Sep> marker) repeated n times",
    "comparison_scope": "paired within-v25; equal examples and optimization settings",
})

RUN_DIR = Path(OUT_ROOT) / RUN_NAME
DRIVE_RUN_DIR = CHECKPOINT_SYNC_ROOT / RUN_NAME
CALIBRATION_DIR = DRIVE_RESULTS_ROOT / CALIBRATION_NAME
""",
        ),
        _markdown("prepare-heading", "## 4. Prepare and audit the fixed long-context data"),
        _code(
            "prepare-data",
            """base_cmd = [
    sys.executable, "-u", "-m", "synthetic_counting_v25.run_v25",
    "--preset", PRESET,
    "--device", DEVICE,
    "--seed", str(SEED),
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
    "--run-name", RUN_NAME,
    "--checkpoint-sync-root", str(CHECKPOINT_SYNC_ROOT),
]
if SKIP_COMPLETED:
    base_cmd.append("--skip-completed")
run_streaming([*base_cmd, "--stage", "prepare"])

print("Prepared local run:", RUN_DIR.resolve())
print("Drive target:", DRIVE_RUN_DIR)
""",
        ),
        _markdown("train-heading", "## 5. Train the paired long-context models"),
        _code(
            "train",
            """training_started = time.perf_counter()
run_streaming([*base_cmd, "--stage", "train"])
print(f"Paired training block: {time.perf_counter() - training_started:.1f} seconds")

sampler_plan = pd.read_csv(RUN_DIR / "tables" / "training_set_count_sampler_plan.csv")
assert int(sampler_plan["feasible"].sum()) == 200
set_marginal = sampler_plan.groupby("set_id")["target_probability"].sum()
count_marginal = sampler_plan.groupby("count")["target_probability"].sum()
assert (set_marginal - 0.05).abs().max() < 1e-9
assert (count_marginal - 0.10).abs().max() < 1e-9
print({
    "feasible_set_count_cells": f"{int(sampler_plan['feasible'].sum())}/{len(sampler_plan)}",
    "maximum_set_marginal_error": float((set_marginal - 0.05).abs().max()),
    "maximum_count_marginal_error": float((count_marginal - 0.10).abs().max()),
})
display(pd.read_csv(RUN_DIR / "tables" / "final_autoregressive_summary.csv"))
display(pd.read_csv(RUN_DIR / "tables" / "final_autoregressive_by_count.csv"))
""",
        ),
        _markdown(
            "calibration-heading",
            """## 6. Symmetric native-head calibration

The calibration schedule is selected on validation prompts and then applied to
both modes.  Only the ten existing atomic-number rows of the ordinary LM head
can move; the backbone and all attention/hidden-state geometry remain exactly
the trained v25 values.  Test prompts are opened once after selection.
""",
        ),
        _code(
            "calibration",
            """run_streaming([
    sys.executable, "-u", "-m", "synthetic_counting_v25_1.run_v25_1",
    "--source-run", str(DRIVE_RUN_DIR),
    "--output-dir", str(CALIBRATION_DIR),
    "--device", DEVICE,
    "--batch-size", "32",
    "--eval-every", "100",
    "--validation-per-count", "10",
    "--seed", "2478",
])
""",
        ),
        _markdown("ncc-heading", "## 7. Measure count compression in the frozen backbones"),
        _code(
            "ncc",
            """NCC_OUTPUT = RUN_DIR / "analysis" / "aligned_ncc"
run_streaming([
    sys.executable, "-u", "scripts/compare_v24_modes_ncc.py",
    "--results-root", str(RUN_DIR.parent),
    "--output", str(NCC_OUTPUT),
    "--run-prefix", RUN_DIR.name,
    "--expected-version", VERSION,
    "--device", DEVICE,
    "--discovery-per-label", "10",
    "--confirmation-per-label", "8",
    "--batch-size", "8",
])
display(pd.read_csv(NCC_OUTPUT / "selected_confirmation_summary.csv")[[
    "comparison_mode", "endpoint", "layer",
    "chance_balanced_accuracy",
    "confirmation_logistic_balanced_accuracy",
    "confirmation_ncc_balanced_accuracy",
    "confirmation_ncc_above_chance",
]])
""",
        ),
        _markdown("results-heading", "## 8. Evaluate the behavioral advantage gate"),
        _code(
            "results",
            """summary = pd.read_csv(CALIBRATION_DIR / "final_summary.csv")
by_count = {
    mode: pd.read_csv(CALIBRATION_DIR / "final" / mode / "final_autoregressive_by_count.csv")
    for mode in ("thinking", "nonthinking")
}
display(summary)
for mode, frame in by_count.items():
    print(mode)
    display(frame)

thinking = summary[summary["mode"].eq("thinking")].iloc[0]
nonthinking = summary[summary["mode"].eq("nonthinking")].iloc[0]
accuracy_gap = float(thinking.test_overall_accuracy - nonthinking.test_overall_accuracy)
advantage_gate = bool(
    float(thinking.test_overall_accuracy) >= 0.90
    and float(thinking.test_minimum_count_accuracy) >= 0.85
    and float(thinking.test_trace_exact_accuracy) >= 0.90
    and accuracy_gap >= 0.05
)
result = {
    "thinking_accuracy": float(thinking.test_overall_accuracy),
    "nonthinking_accuracy": float(nonthinking.test_overall_accuracy),
    "thinking_minus_nonthinking": accuracy_gap,
    "thinking_minimum_count_accuracy": float(thinking.test_minimum_count_accuracy),
    "thinking_count_spread": float(thinking.test_count_accuracy_spread),
    "thinking_trace_exact": float(thinking.test_trace_exact_accuracy),
    "behavioral_advantage_gate": advantage_gate,
}
print(result)
""",
        ),
        _markdown("finish-heading", "## 9. Verify persistence; preserve the runtime if another setting is needed"),
        _code(
            "finish",
            """import json

required = [
    DRIVE_RUN_DIR / "config.json",
    DRIVE_RUN_DIR / "manifest.json",
    DRIVE_RUN_DIR / "checkpoints" / "rope" / "nonthinking" / "final" / "checkpoint.pt",
    DRIVE_RUN_DIR / "checkpoints" / "rope" / "thinking" / "final" / "checkpoint.pt",
    CALIBRATION_DIR / "manifest.json",
    CALIBRATION_DIR / "final_summary.csv",
    CALIBRATION_DIR / "final" / "nonthinking" / "checkpoint.pt",
    CALIBRATION_DIR / "final" / "thinking" / "checkpoint.pt",
]
missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
assert not missing, missing
manifest = json.loads((CALIBRATION_DIR / "manifest.json").read_text())
assert manifest["status"] == "complete"
assert manifest["experiment"] == "v25.1"
assert manifest["source_version"] == "v25"
print("Drive persistence verified:", DRIVE_RUN_DIR, CALIBRATION_DIR)

if advantage_gate and Path("/content").exists():
    print("Behavioral advantage gate passed; disconnecting in 10 seconds.")
    time.sleep(10)
    from google.colab import runtime
    runtime.unassign()
else:
    print("Advantage gate did not pass; keeping the runtime available for the next controlled setting.")
""",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": TARGET.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
