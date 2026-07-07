from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Trace_Count_v3_Colab.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(True),
    }


cells = [
    md(
        r"""
# Trace Count v3: Synthetic NIAH Counting, No Loss Ablation

This notebook wraps the repo implementation in `synthetic_niah_v3`. It follows
the new `pipeline_v3_codex_prompt.md`:

- exactly two training conditions per seed: `non_thinking` and `thinking`;
- no old loss-mask ablation, no `full_lm`, no `final_heavy`, no `trace_only`;
- Round 1: hard length/noise evaluation at train length 256 and eval lengths 256/512/1024;
- Round 2: corrupted-trace diagnostics for the thinking model;
- Round 3: probes, attention retrieval, and single-head ablation.

中文速记：v3 现在不是 sweep 一堆 objective，而是集中比较两个模型：
1. `non_thinking`: prompt 后直接给 `<Ans>`，只训练最后数字 readout；
2. `thinking`: prompt 后给 `<Think/>`，训练 indexed trace + 最后数字 readout。
        """
    ),
    code(
        r"""
from __future__ import annotations

from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
INSTALL_PACKAGE = False  # keep False in Colab to avoid numpy/pandas ABI churn

IN_COLAB = "google.colab" in sys.modules or Path("/content").exists()
if IN_COLAB:
    repo_dir = Path("/content/Synthetic_CoT_NiaH_Count")
    cwd = Path.cwd()
    if (cwd / ".git").exists() or (cwd / "synthetic_niah_v3").exists():
        repo_dir = cwd
    elif (repo_dir / ".git").exists() or (repo_dir / "synthetic_niah_v3").exists():
        pass
    elif repo_dir.exists() and any(repo_dir.iterdir()):
        print(f"Using existing non-git directory: {repo_dir}")
    else:
        if repo_dir.exists():
            repo_dir.rmdir()
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
    os.chdir(repo_dir)

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if not (ROOT / "synthetic_niah_v3").exists() and (ROOT / ".git").exists():
    print("synthetic_niah_v3/ not found; trying git pull --ff-only ...")
    subprocess.run(["git", "pull", "--ff-only"], check=False)
if not (ROOT / "synthetic_niah_v3").exists():
    raise FileNotFoundError(
        "Could not find synthetic_niah_v3/. This Colab runtime has the v3 notebook "
        "but not the new v3 package files. Push/pull the latest repo, or upload the "
        "synthetic_niah_v3 directory plus configs/syn_v3_no_loss_*.yaml."
    )
if INSTALL_PACKAGE and (ROOT / "pyproject.toml").exists():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

import pandas as pd
from IPython.display import HTML, Markdown, display
import torch

print("cwd:", ROOT)
print("python:", sys.executable)
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
        """
    ),
    md(
        r"""
## Runtime Settings

`debug` is for a quick end-to-end sanity run. `main` follows the full pipeline.

Important count of training runs for the notebook defaults:

- `debug`: 2 models x 1 seed = 2 training runs.
- `main` with `SEEDS = "1234"`: 2 models x 1 seed = 2 training runs.

The underlying CLI still supports the full paper-style seed list. For that,
set `SEEDS = ""` to use preset defaults, or set `SEEDS = "1234,1235,1236,1237,1238"`.

There is no loss-mask ablation in this notebook. If you see objective names such
as `full_lm`, `final_heavy`, or `trace_only`, you are running an older v3 notebook.
        """
    ),
    code(
        r"""
PRESET = "debug"  # "debug" or "main"
ROUND = "all"     # "all", "1_hard_eval", "2_corrupted_trace", "3_mechanistic"
SEEDS = "1234"    # keep one seed for development; empty uses preset multi-seed defaults
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = "runs/syn_v3_no_loss"
RUN_NAME = ""     # optional fixed run name
SKIP_COMPLETED = True
SKIP_TRAINING = False

print({
    "PRESET": PRESET,
    "ROUND": ROUND,
    "SEEDS": SEEDS or "preset default",
    "DEVICE": DEVICE,
    "OUT_ROOT": OUT_ROOT,
    "SKIP_COMPLETED": SKIP_COMPLETED,
    "SKIP_TRAINING": SKIP_TRAINING,
})
        """
    ),
    md(
        r"""
## Run v3 Pipeline

This cell trains/evaluates according to the selected round.

Outputs are written under:

```text
runs/syn_v3_no_loss/{timestamp}_{preset}/
```

At the end it prints `FINAL_RUN_DIR` and `FINAL_REPORT`.
        """
    ),
    code(
        r"""
cmd = [
    sys.executable,
    "-u",
    "-m",
    "synthetic_niah_v3.run_v3",
    "--preset",
    PRESET,
    "--round",
    ROUND,
    "--device",
    DEVICE,
    "--out_root",
    OUT_ROOT,
]
if SEEDS:
    cmd += ["--seeds", SEEDS]
if RUN_NAME:
    cmd += ["--run_name", RUN_NAME]
if SKIP_COMPLETED:
    cmd.append("--skip_completed")
if SKIP_TRAINING:
    cmd.append("--skip_training")

if not Path("synthetic_niah_v3/run_v3.py").exists():
    raise FileNotFoundError(
        "Missing synthetic_niah_v3/run_v3.py in this runtime. Rerun the setup cell after "
        "git pulling the latest repo, or upload the synthetic_niah_v3/ package directory."
    )

print(" ".join(cmd), flush=True)
log_file = Path(OUT_ROOT) / "last_pipeline.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
captured_lines = []
assert proc.stdout is not None
with log_file.open("w", encoding="utf-8") as log:
    log.write(" ".join(cmd) + "\n")
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
        captured_lines.append(line.rstrip("\n"))
returncode = proc.wait()
if returncode != 0:
    print(f"\nPipeline failed with return code {returncode}. Log file: {log_file}")
    print("---- Last 200 log lines ----")
    print("\n".join(captured_lines[-200:]))
    raise subprocess.CalledProcessError(returncode, cmd)

FINAL_RUN_DIR = None
FINAL_REPORT = None
for line in captured_lines:
    if line.startswith("FINAL_RUN_DIR "):
        FINAL_RUN_DIR = Path(line.split(" ", 1)[1].strip())
    if line.startswith("FINAL_REPORT "):
        FINAL_REPORT = Path(line.split(" ", 1)[1].strip())

if FINAL_RUN_DIR is None or FINAL_REPORT is None:
    raise RuntimeError("Could not parse FINAL_RUN_DIR / FINAL_REPORT from runner output.")

display(Markdown(f"**Run directory:** `{FINAL_RUN_DIR}`"))
display(Markdown(f"**HTML report:** `{FINAL_REPORT}`"))
        """
    ),
    md(
        r"""
## Inspect Key Tables

These tables are the main CSV artifacts:

- `round1_final_checkpoint_by_count.csv`: final accuracy by count and length;
- `round2_follow_rule_summary.csv`: corrupted-trace follow-rule diagnostics;
- `round3_attention_leaderboard.csv`: best retrieval-like heads;
- `round3_head_ablation_results.csv`: causal single-head ablation effects.
        """
    ),
    code(
        r"""
TABLES_DIR = FINAL_RUN_DIR / "tables"
for name in [
    "round1_final_checkpoint_by_count.csv",
    "round2_follow_rule_summary.csv",
    "round3_attention_leaderboard.csv",
    "round3_head_ablation_results.csv",
]:
    path = TABLES_DIR / name
    display(Markdown(f"### `{name}`"))
    if path.exists() and path.stat().st_size > 0:
        display(pd.read_csv(path).head(20))
    else:
        display(Markdown("_Not generated for this selected round._"))
        """
    ),
    md(
        r"""
## Open Self-contained HTML Report

The report explains each figure, axis definition, group definition, and main
interpretation. It is saved as a self-contained HTML file that can be opened
locally or uploaded with the run folder.
        """
    ),
    code(
        r"""
html_text = FINAL_REPORT.read_text(encoding="utf-8")
display(HTML(html_text))
        """
    ),
    md(
        r"""
## Save Results to Google Drive

Rerun this after the pipeline completes. It copies the whole run directory and
this notebook to your Drive results folder.
        """
    ),
    code(
        r"""
SAVE_TO_DRIVE = True
DRIVE_RESULTS_ROOT = Path("/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/Synthetic_CoT_NiaH_Count/colab_results")

if IN_COLAB and SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    target = DRIVE_RESULTS_ROOT / FINAL_RUN_DIR.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(FINAL_RUN_DIR, target)
    nb_src = Path("notebooks/Trace_Count_v3_Colab.ipynb")
    if nb_src.exists():
        (target / "notebooks").mkdir(exist_ok=True)
        shutil.copy2(nb_src, target / "notebooks" / nb_src.name)
    print("Saved to Drive:", target)
else:
    print("Skipped Drive copy.")
        """
    ),
    md(
        r"""
## Optional: Commit and Push Notebook / v3 Code to GitHub

Set `PUSH_TO_GITHUB = True` only after reviewing the generated files.
        """
    ),
    code(
        r"""
PUSH_TO_GITHUB = False
GIT_COMMIT_MESSAGE = "Add synthetic NIAH counting v3 no-loss pipeline"

if PUSH_TO_GITHUB:
    subprocess.run(["git", "status", "--short"], check=False)
    subprocess.run([
        "git",
        "add",
        "synthetic_niah_v3",
        "configs/syn_v3_no_loss_debug.yaml",
        "configs/syn_v3_no_loss_main.yaml",
        "notebooks/Trace_Count_v3_Colab.ipynb",
        "notebooks/pipeline_v3_codex_prompt.md",
        "scripts/build_v3_notebook.py",
        "tests/test_synthetic_niah_v3.py",
        "pyproject.toml",
    ], check=True)
    subprocess.run(["git", "commit", "-m", GIT_COMMIT_MESSAGE], check=True)
    subprocess.run(["git", "push"], check=True)
else:
    print("PUSH_TO_GITHUB=False; skipped git commit/push.")
        """
    ),
]


def main() -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "colab": {"name": "Trace_Count_v3_Colab.ipynb", "provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
