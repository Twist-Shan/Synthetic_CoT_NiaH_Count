from __future__ import annotations

import json
from pathlib import Path

from build_v26_colab_notebook import build as build_v26_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v28_PartialCountReadout_Multiseed_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def _remove_cells(notebook: dict, *cell_ids: str) -> None:
    remove = set(cell_ids)
    notebook["cells"] = [
        cell for cell in notebook["cells"] if cell.get("id") not in remove
    ]


def _insert_before(notebook: dict, before_id: str, *cells: dict) -> None:
    index = next(
        i for i, cell in enumerate(notebook["cells"]) if cell.get("id") == before_id
    )
    notebook["cells"][index:index] = list(cells)


def build() -> Path:
    source_path = build_v26_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    _remove_cells(notebook, "calibration-heading", "calibration")

    _set_source(
        notebook,
        "title",
        """# Trace Count v28: minimal partial count-readout control

This is a pre-registered three-seed paired experiment.  It keeps v24.3's
256-character prompt, count support 1--10, 100 marker sets, uniform semantic
count sampler, separator/no-index trace, 4-layer/4-head/256-wide transformer,
optimizer, 1,500-step language phase, component-normalized task loss, and
10,000-step schedule.

The sole scientific change is present from initialization in both modes: the
ten atomic count-token output vectors are independent of their input embedding
rows, while every other vocabulary row remains tied.  The complete transformer
is trained end-to-end.  There is no conditional ten-way objective, auxiliary
trace-safety loss, contrastive loss, post-hoc calibration, or test-time update.

Primary gate, fixed before opening the three final test sets: in every seed,
Thinking accuracy >= 0.90, minimum per-count accuracy >= 0.80, trace exact >=
0.90, count spread <= 0.20, and Thinking-minus-Non-thinking accuracy >= 0.10.
All seeds are retained regardless of outcome.
""",
    )
    _set_source(
        notebook,
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
import synthetic_counting_v28
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
    "v28_package": synthetic_counting_v28.__file__,
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
assert torch.cuda.is_available()
""",
    )
    _set_source(
        notebook,
        "settings-heading",
        "## 3. Audit the single-change three-seed setting\n",
    )
    _set_source(
        notebook,
        "runtime-settings",
        """from dataclasses import asdict
from synthetic_counting_v28.config import preset_config
from synthetic_counting_v24_3.config import preset_config as v24_3_preset_config
from synthetic_counting_v20.data import V20Vocab, load_corpus_text
from synthetic_counting_v20.model import build_model

VERSION = "v28"
PRESET = "main"
SEEDS = (1234, 2234, 3234)
DEVICE = "cuda"
OUT_ROOT = "runs/synthetic_counting_v28"
CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT
SKIP_COMPLETED = True
RUN_NAMES = {
    seed: f"v28_partial_count_readout_L256_pool100_seed{seed}" for seed in SEEDS
}
RUN_DIRS = {seed: Path(OUT_ROOT) / RUN_NAMES[seed] for seed in SEEDS}
DRIVE_RUN_DIRS = {seed: CHECKPOINT_SYNC_ROOT / RUN_NAMES[seed] for seed in SEEDS}

planned = preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
baseline = v24_3_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)
changed_fields = {
    key for key, value in asdict(planned).items()
    if asdict(baseline).get(key) != value
}
assert changed_fields == {"version", "untie_atomic_count_readout"}, changed_fields
assert planned.seq_len == 256
assert planned.max_render_len == 287
assert planned.n_positions == 384
assert planned.count_max_threshold == 10
assert planned.needle_pool_size == 100
assert planned.training_count_distribution == "uniform"
assert planned.batch_size == 128
assert planned.trace_format == "separator"
assert planned.task_output_loss_reduction == "component_normalized"
assert planned.tie_word_embeddings
assert planned.untie_atomic_count_readout
assert planned.answer_query_contrastive_weight == 0.0

# Step-zero functional audit: partial untying changes parameter identity, not logits.
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
extra_parameters = planned_model.parameter_count() - baseline_model.parameter_count()
assert extra_parameters == 10 * planned.n_embd == 2560
del baseline_model, planned_model

print({
    "controlled_changes_from_v24.3": sorted(changed_fields),
    "seeds": SEEDS,
    "trace": "unchanged (<Sep> marker) repeated n times",
    "loss": "unchanged v24.3 full-vocabulary component-normalized CE",
    "extra_parameters": extra_parameters,
    "step0_max_logit_difference": max_step0_logit_diff,
    "posthoc_calibration": False,
})

def command_for(seed):
    command = [
        sys.executable, "-u", "-m", "synthetic_counting_v28.run_v28",
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
""",
    )
    _set_source(
        notebook,
        "prepare-heading",
        "## 4. Prepare and audit the three paired datasets\n",
    )
    _set_source(
        notebook,
        "prepare-data",
        """for seed in SEEDS:
    run_streaming([*command_for(seed), "--stage", "prepare"])
    print("Prepared:", RUN_DIRS[seed].resolve(), "->", DRIVE_RUN_DIRS[seed])
""",
    )
    _set_source(
        notebook,
        "train-heading",
        "## 5. Train all paired seeds end-to-end\n",
    )
    _set_source(
        notebook,
        "train",
        """training_started = time.perf_counter()
for seed in SEEDS:
    print(f"\\n===== paired seed {seed} =====", flush=True)
    run_streaming([*command_for(seed), "--stage", "train"])
    sampling = pd.read_csv(RUN_DIRS[seed] / "tables" / "training_sampling_distribution.csv")
    accepted = sampling[sampling["dimension"].eq("accepted_counts")].copy()
    accepted["value"] = accepted["value"].astype(int)
    count_table = accepted.pivot(index="mode", columns="value", values="examples").sort_index(axis=1)
    assert list(count_table.columns) == list(range(1, 11))
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
    _set_source(
        notebook,
        "ncc-heading",
        "## 6. Behavioral gate fixed before NCC/mechanism analysis\n",
    )
    _set_source(
        notebook,
        "ncc",
        """seed_rows = []
for seed in SEEDS:
    summary = pd.read_csv(RUN_DIRS[seed] / "tables" / "final_autoregressive_summary.csv")
    by_count = pd.read_csv(RUN_DIRS[seed] / "tables" / "final_autoregressive_by_count.csv")
    thinking = summary[summary["mode"].eq("thinking")].iloc[0]
    nonthinking = summary[summary["mode"].eq("nonthinking")].iloc[0]
    thinking_counts = by_count[by_count["mode"].eq("thinking")]
    row = {
        "seed": seed,
        "thinking_accuracy": float(thinking.ar_final_accuracy),
        "nonthinking_accuracy": float(nonthinking.ar_final_accuracy),
        "gap": float(thinking.ar_final_accuracy - nonthinking.ar_final_accuracy),
        "thinking_min_count": float(thinking_counts.ar_final_accuracy.min()),
        "thinking_count_spread": float(
            thinking_counts.ar_final_accuracy.max() - thinking_counts.ar_final_accuracy.min()
        ),
        "thinking_trace_exact": float(thinking.trace_exact),
    }
    row["seed_gate"] = bool(
        row["thinking_accuracy"] >= 0.90
        and row["thinking_min_count"] >= 0.80
        and row["thinking_count_spread"] <= 0.20
        and row["thinking_trace_exact"] >= 0.90
        and row["gap"] >= 0.10
    )
    seed_rows.append(row)
behavior = pd.DataFrame(seed_rows)
behavior_gate = bool(behavior.seed_gate.all())
display(behavior)
print({
    "all_seed_behavior_gate": behavior_gate,
    "mean_thinking_accuracy": float(behavior.thinking_accuracy.mean()),
    "mean_nonthinking_accuracy": float(behavior.nonthinking_accuracy.mean()),
    "mean_gap": float(behavior.gap.mean()),
})
""",
    )
    _set_source(
        notebook,
        "results-heading",
        "## 7. NCC across all retained seeds\n",
    )
    _set_source(
        notebook,
        "results",
        """ncc_rows = []
for seed in SEEDS:
    output = RUN_DIRS[seed] / "analysis" / "aligned_ncc"
    run_streaming([
        sys.executable, "-u", "scripts/compare_v24_modes_ncc.py",
        "--results-root", str(RUN_DIRS[seed].parent),
        "--output", str(output),
        "--run-prefix", RUN_DIRS[seed].name,
        "--expected-version", VERSION,
        "--device", DEVICE,
        "--discovery-per-label", "10",
        "--confirmation-per-label", "8",
        "--batch-size", "32",
    ])
    frame = pd.read_csv(output / "selected_confirmation_summary.csv")
    frame.insert(0, "seed", seed)
    ncc_rows.append(frame)
ncc_summary = pd.concat(ncc_rows, ignore_index=True)
display(ncc_summary[[
    "seed", "comparison_mode", "endpoint", "layer",
    "confirmation_logistic_balanced_accuracy",
    "confirmation_ncc_balanced_accuracy",
]])
""",
    )

    _insert_before(
        notebook,
        "finish-heading",
        {
            "cell_type": "markdown",
            "id": "mechanism-heading",
            "metadata": {},
            "source": [
                "## 8. Full training-dynamics and retrieval-role audit on the reference seed\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "mechanism",
            "metadata": {},
            "outputs": [],
            "source": """REFERENCE_SEED = SEEDS[0]
run_streaming([
    *command_for(REFERENCE_SEED),
    "--stage", "phase,causal,extended,plots",
])
role_table = pd.read_csv(
    RUN_DIRS[REFERENCE_SEED]
    / "analysis" / "extended" / "tables" / "attention_role_dynamics.csv"
)
fixed_final = role_table[
    role_table["step"].eq(10000) & role_table["is_fixed_role_head"].eq(1)
]
display(fixed_final[[
    "role", "mode", "layer", "head", "score", "selection_split", "reporting_split"
]])
""".splitlines(keepends=True),
        },
    )
    _set_source(
        notebook,
        "finish-heading",
        "## 9. Verify Drive persistence and retain every seed\n",
    )
    _set_source(
        notebook,
        "finish",
        """import json

required = []
for seed in SEEDS:
    drive_run = DRIVE_RUN_DIRS[seed]
    required.extend([
        drive_run / "config.json",
        drive_run / "manifest.json",
        drive_run / "checkpoints" / "rope" / "nonthinking" / "final" / "checkpoint.pt",
        drive_run / "checkpoints" / "rope" / "thinking" / "final" / "checkpoint.pt",
        drive_run / "tables" / "final_autoregressive_summary.csv",
        drive_run / "tables" / "final_autoregressive_by_count.csv",
    ])
missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
assert not missing, missing
for seed in SEEDS:
    manifest = json.loads((DRIVE_RUN_DIRS[seed] / "manifest.json").read_text())
    assert manifest["version"] == "v28"
    assert manifest["stages"]["train"]["status"] == "complete"
print("Drive persistence verified for all seeds:", DRIVE_RUN_DIRS)

if Path("/content").exists():
    print("Experiment and persistence checks complete; disconnecting in 10 seconds.")
    time.sleep(10)
    from google.colab import runtime
    runtime.unassign()
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
