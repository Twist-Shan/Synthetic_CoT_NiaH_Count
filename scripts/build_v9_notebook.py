from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "Trace_Count_v9_Colab.ipynb"


def lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def markdown(cell_id: str, text: str, *, tags: list[str] | None = None) -> dict:
    metadata = {"tags": tags} if tags else {}
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": metadata,
        "source": lines(text),
    }


def code(cell_id: str, text: str, *, tags: list[str] | None = None) -> dict:
    metadata = {"tags": tags} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": metadata,
        "outputs": [],
        "source": lines(text),
    }


cells = [
    markdown(
        "v9-title",
        """
# Trace Count v9: Query-Conditioned Pair Counting

The earlier v9 saturated because the answer was exactly the total number of marker
tokens. This version keeps length 256 and count 1-10, but removes that shortcut.

- **Query:** every prompt starts with `<Query> <M_q>`.
- **Positive record:** only `<M_q> <Needle> payload` is counted.
- **Hard negatives:** query decoys, other-marker needles, and other-marker decoys.
- **Distractors:** 32-64 negative pairs, mixed with noise in a length-256 prompt.
- **Thinking trace:** `<1> payload_1 ... <n> payload_n`, forcing target retrieval.
- **Targets:** trace indices and final answers share the same `<1>...<10>` tokens.
- **Models:** one non-thinking model and one thinking model.
- **Ultra-small architecture:** 3 layers, 3 heads, `d_model=48`, `MLP=96`.
- **Training:** 10,000 steps, effective batch size 128, no loss weighting.
- **Reliability:** recoverable checkpoints are synced to Google Drive every 2,000 steps.

The query-marker frequency and `<Needle>` frequency are sampled independently of the
gold count. Solving the task therefore requires conjunction, selective retrieval, and
aggregation rather than global marker-frequency estimation.
""",
    ),
    markdown(
        "google-drive-login-heading",
        """
## Google Drive Login

在实验开始时挂载一次 Google Drive。训练 checkpoint 和最终结果都会复用该挂载，避免实验结束时再次登录。
""",
        tags=["google-drive-login"],
    ),
    code(
        "google-drive-login",
        """
from pathlib import Path
import sys

DRIVE_RESULTS_ROOT = Path(
    "/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/"
    "Synthetic_CoT_NiaH_Count/colab_results"
)
DRIVE_MOUNTED = False

def ensure_google_drive_mounted() -> bool:
    global DRIVE_MOUNTED
    if not ("google.colab" in sys.modules or Path("/content").exists()):
        print("Not in Colab; Google Drive mount skipped.")
        return False
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    DRIVE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    DRIVE_MOUNTED = True
    print("Google Drive ready:", DRIVE_RESULTS_ROOT)
    return True

ensure_google_drive_mounted()
""",
        tags=["google-drive-login"],
    ),
    markdown("v9-setup-heading", "## 1. Environment and Repository Setup"),
    code(
        "v9-setup",
        """
from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
IN_COLAB = "google.colab" in sys.modules or Path("/content").exists()
PULL_REPO = True
INSTALL_DEPS = False

if IN_COLAB:
    repo_dir = Path("/content/Synthetic_CoT_NiaH_Count")
    cwd = Path.cwd()
    if (cwd / ".git").exists():
        repo_dir = cwd
    elif not repo_dir.exists():
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
    os.chdir(repo_dir)
    if PULL_REPO and (repo_dir / ".git").exists():
        subprocess.run(["git", "pull"], check=False)

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if INSTALL_DEPS:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40", "pandas", "matplotlib", "tqdm"],
        check=True,
    )

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], check=True)

import pandas as pd
import torch
from IPython.display import Image, Markdown, display

display(Markdown(f"**Repo root:** `{ROOT}`  \\n**Device:** `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}`"))
""",
    ),
    markdown("v9-runtime-heading", "## 2. Runtime Settings"),
    code(
        "v9-runtime",
        """
PRESET = "main"  # use "debug" for a four-step pipeline check
OUT_ROOT = "runs/trace_count_v9_conditional_ultrasmall"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SKIP_COMPLETED = True

LIVE_CHECKPOINT_ROOT = DRIVE_RESULTS_ROOT / "v9_conditional_ultrasmall_live_checkpoints" if DRIVE_MOUNTED else None
if LIVE_CHECKPOINT_ROOT is not None:
    LIVE_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

print({
    "PRESET": PRESET,
    "OUT_ROOT": OUT_ROOT,
    "DEVICE": DEVICE,
    "LIVE_CHECKPOINT_ROOT": str(LIVE_CHECKPOINT_ROOT),
})
""",
    ),
    markdown(
        "v9-capacity-heading",
        """
## 3. Capacity Check

The table below constructs both architectures on CPU and reports their exact trainable
parameter counts. The capacity reference uses 4 layers, 4 heads, `d_model=256`, and
`MLP=1024`; the v9 row uses the same conditional-pair task and vocabulary but reduces
capacity much more aggressively.
""",
    ),
    code(
        "v9-capacity",
        """
from dataclasses import replace

from synthetic_counting_extensions.v7_v8_sweeps import (
    Vocab,
    build_model_cfg,
    make_example,
    preset_configs,
    render,
)
from synthetic_niah_v5.model import make_model

settings = preset_configs("v9", PRESET)
v9_cfg = settings[0]

def exact_parameter_count(cfg) -> int:
    cpu_cfg = replace(cfg, device="cpu")
    vocab = Vocab(cpu_cfg)
    model = make_model(build_model_cfg(cpu_cfg, vocab), "cpu")
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    del model
    return int(count)

v2_reference = replace(
    v9_cfg,
    n_layer=4,
    n_head=4,
    n_embd=256,
    n_inner=1024,
    checkpoint_every=0,
)
v2_params = exact_parameter_count(v2_reference)
v9_params = exact_parameter_count(v9_cfg)

capacity_table = pd.DataFrame([
    {
        "model": "4L capacity reference",
        "layers": 4,
        "heads": 4,
        "d_model": 256,
        "head_dim": 64,
        "mlp_width": 1024,
        "trainable_parameters": v2_params,
        "fraction_of_v2": 1.0,
    },
    {
        "model": "v9 reduced",
        "layers": v9_cfg.n_layer,
        "heads": v9_cfg.n_head,
        "d_model": v9_cfg.n_embd,
        "head_dim": v9_cfg.n_embd // v9_cfg.n_head,
        "mlp_width": v9_cfg.n_inner,
        "trainable_parameters": v9_params,
        "fraction_of_v2": v9_params / v2_params,
    },
])
display(capacity_table)
display(pd.DataFrame([vars(cfg) | {"effective_batch_size": cfg.effective_batch_size} for cfg in settings]))
""",
    ),
    markdown(
        "v9-data-check-heading",
        """
## 4. Conditional-Count Shortcut Audit

The answer counts only `<M_q> <Needle> payload` records. The two tempting marginal shortcuts are
invalid by construction: total `<M_q>` occurrences and total `<Needle>` occurrences
are sampled independently of the gold count. All three hard-negative pair types must
also be present.
""",
    ),
    code(
        "v9-data-check",
        """
import random

preview_vocab = Vocab(v9_cfg)
preview_count = min(5, v9_cfg.max_count)
preview = make_example(v9_cfg, preview_vocab, random.Random(123), count=preview_count)
preview_render = render(preview, preview_vocab, "thinking")
query_occurrences_in_body = preview.seq_tokens.count(preview.query_marker) - 1
preview_stats = pd.DataFrame([{
    "gold_count": preview.count,
    "query_marker": preview.query_marker,
    "query_marker_occurrences_in_body": query_occurrences_in_body,
    "needle_qualifier_occurrences": preview.seq_tokens.count("<Needle>"),
    "query_decoys": preview.pair_type_counts["query_decoy"],
    "other_marker_needles": preview.pair_type_counts["other_needle"],
    "other_marker_decoys": preview.pair_type_counts["other_decoy"],
    "total_distractor_pairs": preview.distractor_count,
    "prompt_length": len(preview.seq_tokens),
}])
display(preview_stats)
think_pos = preview_render["anchors"]["think_pos"]
display(Markdown("**Prompt prefix:** `" + " ".join(preview.seq_tokens[:2]) + "`"))
display(Markdown("**First trace event:** `" + " ".join(preview_render["tokens"][think_pos + 1:think_pos + 3]) + "`"))
assert query_occurrences_in_body > preview.count
assert preview.seq_tokens.count("<Needle>") > preview.count
assert all(preview.pair_type_counts[name] > 0 for name in ["query_decoy", "other_needle", "other_decoy"])
assert preview_stats.loc[0, "prompt_length"] == 256
""",
    ),
    markdown(
        "v9-run-heading",
        """
## 5. Train and Evaluate

This runs exactly two independently initialized models. At steps 2k, 4k, 6k, 8k,
and 10k, each model writes a recoverable checkpoint containing model weights,
optimizer state, step, and RNG state, then immediately syncs it to Drive.
""",
    ),
    code(
        "v9-run",
        """
from synthetic_counting_extensions.v7_v8_sweeps import run_sweep

display(Markdown("**Training runs:** `1 setting x 2 models = 2`"))
combined = run_sweep(
    "v9",
    PRESET,
    OUT_ROOT,
    skip_completed=SKIP_COMPLETED,
    device=DEVICE,
    checkpoint_sync_root=LIVE_CHECKPOINT_ROOT,
)
display(combined)
""",
    ),
    markdown("v9-results-heading", "## 6. Results"),
    code(
        "v9-results",
        """
result_root = Path(OUT_ROOT)
display(combined.groupby(["mode"], as_index=False)[["accuracy", "mae"]].mean())

for run in sorted(result_root.glob("v9_*")):
    display(Markdown(f"### `{run.name}`"))
    for figure_name in ["accuracy_by_count.png", "accuracy_heatmap.png", "accuracy_by_validation_split.png"]:
        figure = run / "figures" / figure_name
        if figure.exists():
            display(Image(filename=str(figure)))
    report = run / "report" / "report.html"
    if report.exists():
        display(Markdown(f"Generated report: `{report}`"))
""",
    ),
    markdown("v9-save-heading", "## 7. Save Complete Results to Google Drive"),
    code(
        "v9-save",
        """
from datetime import datetime

SAVE_TO_DRIVE = True
DRIVE_SAVE_COMPLETED = False

if SAVE_TO_DRIVE and ensure_google_drive_mounted():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source = Path(OUT_ROOT)
    destination = DRIVE_RESULTS_ROOT / f"v9_conditional_ultrasmall_{PRESET}_{timestamp}"
    shutil.copytree(source, destination, dirs_exist_ok=True)
    manifest = {
        "experiment": "v9",
        "preset": PRESET,
        "source": str(source),
        "live_checkpoint_root": str(LIVE_CHECKPOINT_ROOT),
        "saved_at": timestamp,
    }
    (destination / "drive_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    DRIVE_SAVE_COMPLETED = True
    display(Markdown(f"**Saved result bundle:** `{destination}`"))
else:
    print("Drive result save skipped.")
""",
    ),
    markdown("v9-github-heading", "## 8. Optional GitHub Result Upload"),
    code(
        "v9-github",
        """
PUSH_RESULTS_TO_GITHUB = False
GIT_BRANCH = "v9-results"

if PUSH_RESULTS_TO_GITHUB:
    subprocess.run(["git", "checkout", "-B", GIT_BRANCH], check=True)
    subprocess.run(["git", "add", OUT_ROOT], check=True)
    subprocess.run(["git", "commit", "-m", "Add v9 conditional-count results"], check=False)
    subprocess.run(["git", "push", "-u", "origin", GIT_BRANCH], check=True)
else:
    print("PUSH_RESULTS_TO_GITHUB is False")
""",
    ),
    markdown("v9-disconnect-heading", "## 9. Optional Runtime Disconnect"),
    code(
        "v9-disconnect",
        """
AUTO_DISCONNECT_AFTER_DRIVE_SAVE = False

if AUTO_DISCONNECT_AFTER_DRIVE_SAVE and DRIVE_SAVE_COMPLETED and IN_COLAB:
    from google.colab import runtime
    print("Drive save confirmed; disconnecting Colab runtime.")
    runtime.unassign()
else:
    print("Runtime remains connected.")
""",
    ),
]

# Normalize this legacy text cell explicitly because an older Windows save left
# invalid bytes in its source. The generated notebook remains clean and portable.
for cell in cells:
    if cell.get("id") == "google-drive-login-heading":
        cell["source"] = lines(
            """
## Google Drive Login

Mount Google Drive before training. Periodic checkpoints and final results reuse this mount.
"""
        )

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"wrote {OUTPUT}")
