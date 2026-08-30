from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v24_8_NativeHeadCalibration_Colab.ipynb"


def _markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build() -> Path:
    cells = [
        _markdown(
            """# Trace Count v24.8: native LM-head number-row calibration

V24.7 already generates an almost perfectly correct separator/no-index trace
and its native `<Ans>` residual has held-out count NCC/logistic accuracy of
1.0, yet the ordinary LM head maps counts 3, 7, and 9 to adjacent even-number
tokens. V24.8 tests the corresponding training diagnosis without changing the
trace, sampler, transformer, vocabulary, or inference.

Starting from each paired v24.7 final checkpoint, the complete backbone,
input embeddings, final layer norm, and non-number unembedding rows are
frozen. A balanced training-only tail applies cross-entropy to the ten existing
atomic-number rows at the ordinary answer query. Candidate learning rates are
selected only on the validation split. The test split is evaluated once after
selection, and success still means the model's own raw autoregressive token:
overall >= 0.90, every count >= 0.85, count spread <= 0.10, and Thinking trace
exact >= 0.90. No probe or trace decoder can satisfy the gate.
"""
        ),
        _markdown("## 1. Mount Google Drive"),
        _code(
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
"""
        ),
        _markdown("## 2. Clone the audited implementation and verify the GPU"),
        _code(
            """import os
import signal
import subprocess
import sys
from pathlib import Path

assert DRIVE_READY, "Run the Drive cell first"
REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
REPO_REF = "agent/remove-misplaced-realistic-artifacts"
repo = Path("/content/Synthetic_CoT_NiaH_Count")
if not repo.exists():
    subprocess.run(
        ["git", "clone", "--branch", REPO_REF, "--single-branch", REPO_URL, str(repo)],
        check=True,
    )
else:
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

import pandas as pd
import torch
import synthetic_counting_v24_8

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print({
    "repo": str(repo),
    "repo_ref": REPO_REF,
    "repo_commit": commit,
    "package": synthetic_counting_v24_8.__file__,
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
assert torch.cuda.is_available()
"""
        ),
        _markdown(
            """## 3. Audit the source failure and the unchanged interface

The source run must show the exact pattern that motivates a readout-only tail:
high trace fidelity and a linearly decodable answer-query representation, but
failed native raw output. The v24.8 code exposes no trace-generation setting;
its only trainable tensor is the pre-existing untied LM-head weight, and its
loss slices the logits to the ten pre-existing number rows.
"""
        ),
        _code(
            """SOURCE_RUN = DRIVE_RESULTS_ROOT / "v24.7_answer_compression_pool20_count1-10_seed1234"
OUTPUT_DIR = DRIVE_RESULTS_ROOT / "v24.8_native_head_calibration_pool20_count1-10_seed2478"
assert SOURCE_RUN.exists(), SOURCE_RUN

source_summary = pd.read_csv(SOURCE_RUN / "tables" / "final_autoregressive_summary.csv")
source_thinking = source_summary[source_summary["mode"].eq("thinking")].iloc[0]
source_ncc = pd.read_csv(
    SOURCE_RUN / "analysis" / "aligned_ncc" / "selected_confirmation_summary.csv"
)
answer_ncc = source_ncc[
    source_ncc["endpoint"].eq("thinking_answer_query")
]["confirmation_ncc_balanced_accuracy"].iloc[0]
print({
    "source_thinking_raw_ar": float(source_thinking.ar_final_accuracy),
    "source_thinking_trace_exact": float(source_thinking.trace_exact),
    "source_thinking_answer_query_ncc": float(answer_ncc),
})
assert float(source_thinking.ar_final_accuracy) < 0.90
assert float(source_thinking.trace_exact) >= 0.95
assert float(answer_ncc) >= 0.95

from pathlib import Path
implementation = Path(synthetic_counting_v24_8.__file__).with_name("readout_tail.py").read_text()
assert "model.lm_head.weight.requires_grad_(True)" in implementation
assert "for parameter in model.parameters():" in implementation
assert "logits[row, item.spans.count_pos - 1, number_ids]" in implementation
assert "render_v20_shortened_trace" not in implementation
print("Audit passed: training-only native number-row calibration; trace unchanged.")
"""
        ),
        _markdown(
            """## 4. Run validation-selected paired calibration

Thinking is calibrated first. The schedule tries increasingly aggressive
learning rates only if the preceding candidate fails the validation gate. Once
a Thinking setting passes (or is the best available), exactly that setting is
applied to Non-thinking. The balanced max-entropy train sampler is reused and
the test split remains sealed until model selection is complete.
"""
        ),
        _code(
            """import codecs

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

run_streaming([
    sys.executable, "-u", "-m", "synthetic_counting_v24_8.run_v24_8",
    "--source-run", SOURCE_RUN,
    "--output-dir", OUTPUT_DIR,
    "--device", "cuda",
    "--batch-size", "128",
    "--eval-every", "100",
    "--validation-per-count", "10",
    "--seed", "2478",
])
"""
        ),
        _markdown("## 5. Inspect raw-token results and verify Drive persistence"),
        _code(
            """import json
import time

summary = pd.read_csv(OUTPUT_DIR / "final_summary.csv")
display(summary)
for mode in ("thinking", "nonthinking"):
    display(pd.read_csv(OUTPUT_DIR / "final" / mode / "final_autoregressive_by_count.csv"))

thinking = summary[summary["mode"].eq("thinking")].iloc[0]
assert bool(thinking.test_success_criteria_met), summary
assert float(thinking.test_overall_accuracy) >= 0.90
assert float(thinking.test_minimum_count_accuracy) >= 0.85
assert float(thinking.test_count_accuracy_spread) <= 0.10
assert float(thinking.test_trace_exact_accuracy) >= 0.90

required = [
    OUTPUT_DIR / "manifest.json",
    OUTPUT_DIR / "final_summary.csv",
    OUTPUT_DIR / "final" / "thinking" / "checkpoint.pt",
    OUTPUT_DIR / "final" / "thinking" / "final_autoregressive_summary.csv",
    OUTPUT_DIR / "final" / "thinking" / "final_autoregressive_by_count.csv",
    OUTPUT_DIR / "final" / "nonthinking" / "checkpoint.pt",
]
missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
assert not missing, missing
manifest = json.loads((OUTPUT_DIR / "manifest.json").read_text())
assert manifest["status"] == "complete"
assert manifest["trace_change"] is False
assert manifest["inference_change"] is False
print("Drive persistence verified:", OUTPUT_DIR)

if Path("/content").exists():
    print("Disconnecting in 10s after successful raw-output gate and persistence check...")
    time.sleep(10)
    from google.colab import runtime
    runtime.unassign()
"""
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
        "nbformat_minor": 0,
    }
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
