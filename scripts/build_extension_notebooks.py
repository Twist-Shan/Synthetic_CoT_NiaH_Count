from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n").splitlines(True)}


SETUP = r"""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
IN_COLAB = "google.colab" in sys.modules or Path("/content").exists()
INSTALL_DEPS = False
PULL_REPO = True

if IN_COLAB:
    repo_dir = Path("/content/Synthetic_CoT_NiaH_Count")
    cwd = Path.cwd()
    if (cwd / ".git").exists() or (cwd / "README.md").exists():
        repo_dir = cwd
    elif not repo_dir.exists():
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
    os.chdir(repo_dir)
    if PULL_REPO and (repo_dir / ".git").exists():
        subprocess.run(["git", "pull", "--ff-only"], check=False)

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if INSTALL_DEPS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40", "pandas", "matplotlib", "tqdm"], check=True)

import pandas as pd
from IPython.display import Markdown, display, Image

display(Markdown(f"**Repo root:** `{ROOT}`"))
"""


SAVE_CELL = r"""
# Save result folder to Google Drive.
SAVE_TO_DRIVE = True
DRIVE_DEST_ROOT = "/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/Synthetic_CoT_NiaH_Count/colab_results"

if SAVE_TO_DRIVE and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    dest_root = Path(DRIVE_DEST_ROOT)
    dest_root.mkdir(parents=True, exist_ok=True)
    if "RUN_DIR" in globals() and RUN_DIR is not None:
        src = Path(RUN_DIR)
    elif "OUT_ROOT" in globals():
        src = Path(OUT_ROOT)
    else:
        src = None
    if src is not None and src.exists():
        dest = dest_root / src.name
        src_resolved = src.resolve()
        dest_root_resolved = dest_root.resolve()
        if src_resolved == dest_root_resolved or dest_root_resolved in src_resolved.parents:
            display(Markdown(f"**Already stored in Drive:** `{src}`"))
        else:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            display(Markdown(f"**Saved to Drive:** `{dest}`"))
    else:
        display(Markdown("No RUN_DIR/OUT_ROOT found to save."))
else:
    display(Markdown("Drive save skipped."))
"""


GITHUB_CELL = r"""
# Optional: commit and push notebook/code changes to GitHub.
PUSH_TO_GITHUB = False
GIT_COMMIT_MESSAGE = "Add synthetic counting experiment notebook"

if PUSH_TO_GITHUB:
    subprocess.run(["git", "status", "--short"], check=False)
    subprocess.run(["git", "add", "notebooks", "synthetic_counting_extensions", "scripts"], check=True)
    subprocess.run(["git", "commit", "-m", GIT_COMMIT_MESSAGE], check=True)
    subprocess.run(["git", "push"], check=True)
else:
    display(Markdown("GitHub push skipped. Set `PUSH_TO_GITHUB = True` after checking the diff."))
"""


DISCONNECT_CELL = r"""
# Optional: disconnect Colab runtime after saving.
AUTO_DISCONNECT_AFTER_SAVE = False

if AUTO_DISCONNECT_AFTER_SAVE and IN_COLAB:
    from google.colab import runtime
    runtime.unassign()
else:
    display(Markdown("Auto-disconnect skipped."))
"""


def write_notebook(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def build_v5_2() -> None:
    cells = [
        md(
            """
# Trace Count v5.2: switch-token diagnostics

目标：解释为什么 mixed thinking-toggle Transformer 里，开关看起来学会了，但 targeted retrieval 不显著。

核心检查：

1. `<Think/>` / `</Think>` 的 switch logits 是否把 thinking 与 non-thinking 分开；
2. retrieval 是否测在正确的 **prediction query** 上，而不是 marker token 已经可见之后；
3. marker-only trace 是否因为缺少显式 `index_token_k`，天然不容易形成 v2 那种 k-to-k retrieval head。
            """
        ),
        md("## 1. Setup"),
        code(SETUP),
        md("## 2. Runtime settings"),
        code(
            r"""
# A completed v5 checkpoint is required; v5.2 does not retrain the model.
# Set this only when you want to force a particular run or result bundle.
RUN_DIR_OVERRIDE = ""
AUTO_MOUNT_DRIVE_FOR_V5 = True
PREFER_DRIVE_V5 = True
DRIVE_RESULTS_ROOT = Path(
    "/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/"
    "Synthetic_CoT_NiaH_Count/colab_results"
)

from synthetic_counting_extensions.v5_2_switch_diagnostics import resolve_v5_run_dir


def find_v5_run() -> Path:
    if RUN_DIR_OVERRIDE:
        return resolve_v5_run_dir(RUN_DIR_OVERRIDE)

    local_search_roots = [
        Path("outputs/v5"),
        Path("outputs"),
        Path("runs"),
        Path("colab_results"),
    ]
    drive_search_roots = [DRIVE_RESULTS_ROOT / "v5", DRIVE_RESULTS_ROOT]

    if IN_COLAB and PREFER_DRIVE_V5 and AUTO_MOUNT_DRIVE_FOR_V5:
        if not Path("/content/drive/MyDrive").exists():
            from google.colab import drive

            drive.mount("/content/drive")
        for root in drive_search_roots:
            try:
                return resolve_v5_run_dir(root)
            except FileNotFoundError:
                pass

    for root in local_search_roots:
        try:
            return resolve_v5_run_dir(root)
        except FileNotFoundError:
            pass

    if IN_COLAB and not PREFER_DRIVE_V5 and AUTO_MOUNT_DRIVE_FOR_V5:
        if not Path("/content/drive/MyDrive").exists():
            from google.colab import drive

            drive.mount("/content/drive")
        for root in drive_search_roots:
            try:
                return resolve_v5_run_dir(root)
            except FileNotFoundError:
                pass

    searched = [str(path) for path in local_search_roots]
    if IN_COLAB:
        searched.append(str(DRIVE_RESULTS_ROOT))
    raise FileNotFoundError(
        "No completed v5 run was found. Run Trace_Count_v5_Colab first, or set "
        "RUN_DIR_OVERRIDE to a saved v5 result folder. Searched: " + ", ".join(searched)
    )


RUN_DIR = find_v5_run()

EXAMPLES_PER_COUNT = 100
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

print({
    "RUN_DIR": str(RUN_DIR),
    "CHECKPOINT": str(RUN_DIR / "checkpoints" / "final.pt"),
    "EXAMPLES_PER_COUNT": EXAMPLES_PER_COUNT,
    "DEVICE": DEVICE,
})
            """
        ),
        md("## 3. Run v5.2 diagnostics"),
        code(
            r"""
from synthetic_counting_extensions.v5_2_switch_diagnostics import run_switch_and_retrieval_diagnostics

outputs = run_switch_and_retrieval_diagnostics(RUN_DIR, examples_per_count=EXAMPLES_PER_COUNT, device=DEVICE)
display(Markdown(f"**Report:** `{RUN_DIR / 'v5_2_switch_diagnostics' / 'report' / 'report.html'}`"))
display(outputs["switch_summary"])
display(outputs["prediction_query_head_summary"].sort_values(["correct_top1", "correct_prompt_needle_mass"], ascending=False).head(12))
            """
        ),
        md("## 4. Display figures"),
        code(
            r"""
FIG_DIR = RUN_DIR / "v5_2_switch_diagnostics" / "figures"
for name in [
    "switch_probability_summary.png",
    "prediction_query_correct_top1.png",
    "prediction_query_correct_mass.png",
    "prediction_query_marker_margin.png",
    "post_marker_correct_top1.png",
]:
    path = FIG_DIR / name
    if path.exists():
        display(Markdown(f"### {name}"))
        display(Image(filename=str(path)))
            """
        ),
        md("## 5. Save / GitHub / disconnect"),
        code(SAVE_CELL),
        code(GITHUB_CELL),
        code(DISCONNECT_CELL),
    ]
    write_notebook(NOTEBOOK_DIR / "Trace_Count_v5_2_Colab.ipynb", cells)


def build_v2_2_followup() -> None:
    cells = [
        md(
            """
# Trace Count v2.2 Follow-up: successor localization and aggregation

目标：继续深挖 v2.2 targeted retrieval 之后发生了什么。

问题：

1. L3 targeted retrieval 定位到第 k 个 needle 之后，模型如何生成下一个 index / close？
2. 下一次 targeted retrieval 是如何接到第 k+1 个 needle 的？
3. 最终 `<Ans>` 是看全部 trace、最后一个 trace marker，还是 prompt aggregation？
            """
        ),
        md("## 1. Setup"),
        code(SETUP),
        md("## 2. Runtime settings"),
        code(
            r"""
V2_RUN_DIR = Path("colab_results/v2_marker_trace_main_seed1234_20260706_215757/run")
if not (V2_RUN_DIR / "checkpoints" / "final" / "thinking" / "config.json").exists():
    candidates = list(Path("colab_results").glob("v2_marker_trace*/run")) + list(Path("runs").glob("v2_marker_trace*"))
    valid = [p for p in candidates if (p / "checkpoints" / "final" / "thinking" / "config.json").exists()]
    if valid:
        V2_RUN_DIR = valid[-1]

EXAMPLES_PER_COUNT = 50
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print({"V2_RUN_DIR": str(V2_RUN_DIR), "EXAMPLES_PER_COUNT": EXAMPLES_PER_COUNT, "DEVICE": DEVICE})
            """
        ),
        md("## 3. Run follow-up diagnostics"),
        code(
            r"""
from synthetic_counting_extensions.v2_2_followup import run_v2_2_followup

outputs = run_v2_2_followup(V2_RUN_DIR, examples_per_count=EXAMPLES_PER_COUNT, device=DEVICE)
RUN_DIR = V2_RUN_DIR / "v2_2_followup_mechanism"
display(Markdown(f"**Report:** `{RUN_DIR / 'report' / 'report.html'}`"))
display(outputs["successor_transition_head_summary"].sort_values("next_token_margin", ascending=False).head(12))
display(outputs["answer_trace_attention_head_summary"].sort_values("all_trace_marker_mass", ascending=False).head(12))
            """
        ),
        md("## 4. Display figures"),
        code(
            r"""
FIG_DIR = RUN_DIR / "figures"
for name in [
    "successor_next_token_margin.png",
    "successor_current_marker_self_mass.png",
    "successor_next_prompt_needle_mass.png",
    "answer_all_trace_marker_mass.png",
    "answer_last_trace_marker_mass.png",
    "trace_length_override_follows_trace.png",
]:
    path = FIG_DIR / name
    if path.exists():
        display(Markdown(f"### {name}"))
        display(Image(filename=str(path)))
            """
        ),
        md("## 5. Save / GitHub / disconnect"),
        code(SAVE_CELL),
        code(GITHUB_CELL),
        code(DISCONNECT_CELL),
    ]
    write_notebook(NOTEBOOK_DIR / "Trace_Count_v2_2_Followup_Colab.ipynb", cells)


def build_v7() -> None:
    cells = [
        md(
            """
# Trace Count v7: pure context-length extension of v2

这个 notebook 只改变 v2 的 prompt 长度，count 始终为 1–10：

- length 1024：non-thinking + thinking；
- length 2048：non-thinking + thinking。

Main 一共训练四个模型。trace index 与最终答案共享 `<1>...<10>` 数字 token。

评估同时记录 `tf_accuracy` 和 `ar_accuracy`。其中 `tf_accuracy` 是 teacher-forced answer position 上的 final-count readout；`ar_accuracy` 是从 prompt 后 autoregressively 生成到 final answer 后的准确率。报告里的 `accuracy` 默认使用 `ar_accuracy`。
            """
        ),
        md("## 1. Setup"),
        code(SETUP),
        md("## 2. Runtime settings"),
        code(
            r"""
PRESET = "main"  # "debug" or "main"
OUT_ROOT = "runs/trace_count_v7_cot_advantage"
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
SKIP_COMPLETED = True
print({"PRESET": PRESET, "OUT_ROOT": OUT_ROOT, "DEVICE": DEVICE})
            """
        ),
        md("## 3. Run v7 sweep"),
        code(
            r"""
from synthetic_counting_extensions.v7_v8_sweeps import preset_configs, run_sweep

settings = preset_configs("v7", PRESET)
display(pd.DataFrame([vars(cfg) | {"effective_batch_size": cfg.effective_batch_size} for cfg in settings]))
display(Markdown(f"**Training runs:** `{len(settings)} settings × 2 models = {2 * len(settings)}`"))

combined = run_sweep("v7", PRESET, OUT_ROOT, skip_completed=SKIP_COMPLETED, device=DEVICE)
display(combined.head())
display(combined.groupby(["setting", "mode"], as_index=False)["accuracy"].mean())
            """
        ),
        md("## 4. Display reports"),
        code(
            r"""
for run in Path(OUT_ROOT).glob("v7_*"):
    report = run / "report" / "report.html"
    figures = [
        run / "figures" / "accuracy_by_count.png",
        run / "figures" / "accuracy_by_validation_split.png",
    ]
    if figures[0].exists():
        display(Markdown(f"## {run.name}\nReport: `{report}`"))
        for fig in figures:
            if fig.exists():
                display(Image(filename=str(fig)))
            """
        ),
        md("## 5. Save / GitHub / disconnect"),
        code(SAVE_CELL),
        code(GITHUB_CELL),
        code(DISCONNECT_CELL),
    ]
    write_notebook(NOTEBOOK_DIR / "Trace_Count_v7_Colab.ipynb", cells)


def build_v8() -> None:
    cells = [
        md(
            """
# Trace Count v8: pure needle-count extension of v2

目标：保持 v2 的 prompt 长度 256，只把 needle count 扩展到 1–30。

Main 只训练两个模型：一个 non-thinking，一个 thinking。训练和验证都覆盖 1–30，
因此 21–30 不再包含“未训练数字 token”的伪 OOD。trace index 与最终答案共享 `<1>...<30>`。

输出重点：

1. autoregressive final-count accuracy by gold count；
2. `tf_accuracy` / `ar_accuracy` 同时保存，图和阈值默认使用 `ar_accuracy`；
3. `val_1_10`、`val_11_20`、`val_21_30` 三个 balanced validation split；
4. first count below 0.9 accuracy；
5. CoT 和 non-thinking 的崩塌阈值是否不同。
            """
        ),
        md("## 1. Setup"),
        code(SETUP),
        md("## 2. Runtime settings"),
        code(
            r"""
PRESET = "main"  # "debug" or "main"
OUT_ROOT = "runs/trace_count_v8_many_needles"
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
SKIP_COMPLETED = True
print({"PRESET": PRESET, "OUT_ROOT": OUT_ROOT, "DEVICE": DEVICE})
            """
        ),
        md("## 3. Run v8 sweep"),
        code(
            r"""
from synthetic_counting_extensions.v7_v8_sweeps import preset_configs, run_sweep

settings = preset_configs("v8", PRESET)
display(pd.DataFrame([vars(cfg) | {"effective_batch_size": cfg.effective_batch_size} for cfg in settings]))
display(Markdown(f"**Training runs:** `{len(settings)} setting × 2 models = {2 * len(settings)}`"))

combined = run_sweep("v8", PRESET, OUT_ROOT, skip_completed=SKIP_COMPLETED, device=DEVICE)
display(combined.head())
split_summary = (
    combined.groupby(["setting", "mode", "validation_split"], as_index=False)
    .agg(
        count_min=("count", "min"),
        count_max=("count", "max"),
        tf_accuracy=("tf_accuracy", "mean"),
        ar_accuracy=("ar_accuracy", "mean"),
        accuracy=("accuracy", "mean"),
        mae=("mae", "mean"),
    )
)
display(Markdown("## Validation performance by needle-count range"))
display(split_summary)
threshold = []
for (setting, mode), g in combined.groupby(["setting", "mode"]):
    bad = g[g["accuracy"] < 0.9]
    threshold.append({"setting": setting, "mode": mode, "first_count_below_0.9": int(bad["count"].iloc[0]) if len(bad) else "none"})
display(pd.DataFrame(threshold))
            """
        ),
        md("## 4. Display reports"),
        code(
            r"""
for run in Path(OUT_ROOT).glob("v8_*"):
    report = run / "report" / "report.html"
    figures = [
        run / "figures" / "accuracy_by_count.png",
        run / "figures" / "accuracy_by_validation_split.png",
    ]
    if figures[0].exists():
        display(Markdown(f"## {run.name}\nReport: `{report}`"))
        for fig in figures:
            if fig.exists():
                display(Image(filename=str(fig)))
            """
        ),
        md("## 5. Save / GitHub / disconnect"),
        code(SAVE_CELL),
        code(GITHUB_CELL),
        code(DISCONNECT_CELL),
    ]
    write_notebook(NOTEBOOK_DIR / "Trace_Count_v8_Colab.ipynb", cells)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    build_v5_2()
    build_v7()
    build_v8()
    for name in [
        "Trace_Count_v5_2_Colab.ipynb",
        "Trace_Count_v7_Colab.ipynb",
        "Trace_Count_v8_Colab.ipynb",
    ]:
        print(NOTEBOOK_DIR / name)


if __name__ == "__main__":
    main()
