from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


SPECS = {
    "v15": {
        "title": "Shakespeare Inserted Needles with RoPE/RPE",
        "summary": (
            "在长度 256 的连续 Tiny Shakespeare 字符窗口中随机覆盖 1-30 个位置，插入独立 marker。"
            "比较 RoPE 与 learned relative-position bias (RPE)，并分别训练 non-thinking 与 thinking。"
        ),
        "models": "RoPE/RPE x non-thinking/thinking = 4",
        "task": "count inserted marker needles",
    },
    "v16": {
        "title": "Native Shakespeare Target-Letter Counting with RoPE/RPE",
        "summary": (
            "不再插入 marker。每个样本先给出目标字符，再在长度 256 的连续 Tiny Shakespeare 窗口中"
            "计数该字符的原生出现次数；目标来自 S,H,A,K,E,R 及对应小写字母。"
        ),
        "models": "RoPE/RPE x non-thinking/thinking = 4",
        "task": "count native occurrences of an explicitly named target character",
    },
    "v17": {
        "title": "Decreasing Long-Tail Synthetic Needles",
        "summary": (
            "保持 v10 的长度 256、APE、uniform synthetic haystack 与 inserted-marker 任务；"
            "训练 count 使用常规递减长尾分布，即 needle 越多，样本越少。balanced validation 仍逐 count 等量。"
        ),
        "models": "APE x non-thinking/thinking = 2",
        "task": "count inserted marker needles under a decreasing long-tail training distribution",
    },
}


def markdown(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str, cell_id: str, tags: list[str] | None = None) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {"tags": tags} if tags else {},
        "execution_count": None,
        "outputs": [],
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def cells(version: str, spec: dict[str, str]) -> list[dict]:
    module = f"synthetic_counting_{version}.run_{version}"
    v17_settings = """
    COUNT_SAMPLING = "power"       # "power" or "exponential"
    POWER_ALPHA = 1.0              # p(n) proportional to n ** (-alpha)
    EXPONENTIAL_BETA = 0.15        # p(n) proportional to exp(-beta * (n - 1))
    """ if version == "v17" else ""
    v17_cmd = """
    if VERSION == "v17":
        cmd += [
            "--count-sampling", COUNT_SAMPLING,
            "--power-alpha", str(POWER_ALPHA),
            "--exponential-beta", str(EXPONENTIAL_BETA),
        ]
    """ if version == "v17" else ""
    v17_overrides = (
        "count_sampling=COUNT_SAMPLING, power_alpha=POWER_ALPHA, "
        "exponential_beta=EXPONENTIAL_BETA,"
        if version == "v17"
        else ""
    )

    return [
        markdown(
            f"""
            # Trace Count {version}: {spec['title']}

            {spec['summary']}

            | 项目 | 正式设置 |
            | --- | --- |
            | 模型 | 每个 position-encoding x mode 组合独立 random init；4 layers；4 heads；d_model=256；MLP=1024 |
            | 任务 | {spec['task']} |
            | prompt/count | prompt length 256；count 1-30 |
            | 模型数 | {spec['models']} |
            | 训练目标 | **v10-style completion-only causal next-token loss** |
            | 数字 token | trace index 与最终答案共享 `<1>...<30>` |

            训练时完整 gold prefix 仍在 causal context 中，但 prompt/haystack 不计 loss：

            - **non-thinking**：只监督最终 count token 与 `<EOS>`。
            - **thinking**：从第一个 trace 数字开始，监督 trace、`</Think>`、`<Ans>`、最终 count 与 `<EOS>`。

            这里的 autoregressive loss 是 teacher-forced causal next-token CE；训练时不是自由 rollout。
            自由生成性能由独立的 autoregressive evaluation 衡量。
            """,
            "title",
        ),
        markdown("## 1. 先挂载 Google Drive", "drive-heading"),
        code(
            """
            from pathlib import Path

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
            "drive-login",
            ["google-drive-login"],
        ),
        markdown("## 2. Repo 与 Python 环境", "setup-heading"),
        code(
            """
            import json
            import os
            import signal
            import subprocess
            import sys
            from pathlib import Path

            REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
            preferred = Path("/content/Synthetic_CoT_NiaH_Count")
            candidates = [Path.cwd(), *Path.cwd().parents, preferred]
            repo = next((p.resolve() for p in candidates if (p / "pyproject.toml").exists()), None)
            if repo is None:
                subprocess.run(["git", "clone", REPO_URL, str(preferred)], check=True)
                repo = preferred
            elif Path("/content").exists() and (repo / ".git").exists():
                subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=False)
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
                if Path("/content").exists():
                    os.kill(os.getpid(), signal.SIGKILL)
                raise RuntimeError("Scientific ABI repaired. Restart and rerun all cells.")

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"],
                check=True,
            )
            src_root = (repo / "src").resolve()
            if str(src_root) not in sys.path:
                sys.path.insert(0, str(src_root))

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch
            from IPython.display import Image, Markdown, display

            print({
                "repo": str(repo),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            })
            """,
            "environment-setup",
        ),
        code(
            """
            RUN_TESTS = True
            if RUN_TESTS:
                subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "tests/test_synthetic_counting_v15_v17.py"],
                    check=True,
                )
            """,
            "tests",
        ),
        markdown("## 3. Runtime settings", "settings-heading"),
        code(
            f"""
            VERSION = "{version}"
            PRESET = "main"                  # use "debug" for an end-to-end check
            STAGE = "all"                    # train, attention, state, plots, or all
            SEED = 1234
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            OUT_ROOT = f"runs/synthetic_counting_{{VERSION}}"
            RUN_NAME = f"{{VERSION}}_{{PRESET}}_completion_seed{{SEED}}"
            SKIP_COMPLETED = True
            {dedent(v17_settings).strip()}
            CHECKPOINT_SYNC_ROOT = (
                DRIVE_RESULTS_ROOT / f"{{VERSION}}_live_checkpoints" if DRIVE_READY else None
            )

            from synthetic_counting_v11.config import preset_config
            PLANNED_CONFIG = preset_config(
                VERSION,
                PRESET,
                seed=SEED,
                device=DEVICE,
                {v17_overrides}
            )
            assert (PLANNED_CONFIG.n_layer, PLANNED_CONFIG.n_head) == (4, 4)
            assert (PLANNED_CONFIG.n_embd, PLANNED_CONFIG.n_inner) == (256, 1024)
            assert PLANNED_CONFIG.loss_scope == "completion"
            print({{
                "version": VERSION,
                "preset": PRESET,
                "device": DEVICE,
                "model_variants": PLANNED_CONFIG.model_variants,
                "number_of_models": len(PLANNED_CONFIG.model_variants),
                "task_type": PLANNED_CONFIG.task_type,
                "noise_source": PLANNED_CONFIG.noise_source,
                "count_sampling": PLANNED_CONFIG.count_sampling,
                "training_objective": PLANNED_CONFIG.to_dict()["training_objective"],
            }})
            """,
            "runtime-settings",
        ),
        markdown(f"## 4. Run {version}", "run-heading"),
        code(
            f"""
            cmd = [
                sys.executable, "-u", "-m", "{module}",
                "--preset", PRESET,
                "--stage", STAGE,
                "--device", DEVICE,
                "--seed", str(SEED),
                "--out-root", OUT_ROOT,
                "--run-name", RUN_NAME,
            ]
            if SKIP_COMPLETED:
                cmd.append("--skip-completed")
            if CHECKPOINT_SYNC_ROOT is not None:
                cmd += ["--checkpoint-sync-root", str(CHECKPOINT_SYNC_ROOT)]
            {dedent(v17_cmd).strip()}

            print(" ".join(cmd), flush=True)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            captured = []
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                captured.append(line.rstrip())
            returncode = process.wait()
            if returncode:
                print("---- Last 200 log lines ----")
                print("\\n".join(captured[-200:]))
                raise subprocess.CalledProcessError(returncode, cmd)

            RUN_DIR = Path(OUT_ROOT) / RUN_NAME
            assert (RUN_DIR / "config.json").exists(), RUN_DIR
            print("RUN_DIR =", RUN_DIR.resolve())
            """,
            "run-pipeline",
        ),
        markdown(
            """
            ## 5. Inspect settings and learning dynamics

            - `train_total_loss`：只在上述 completion token 上平均的 next-token CE。
            - `tf_final_accuracy`：给定 gold prefix，在最终 count token 上取 count vocabulary 内 argmax。
            - `tf_trace_marker_accuracy`：仅用于 thinking；给定 gold trace prefix，预测每个 marker identity。
            - `ar_final_accuracy`：从 prompt 后自由生成完整 completion，再检查最终 count；它会暴露 trace 错误传播。
            - v17 训练分布递减，但 `eval_by_count.csv` 始终对 1-30 每个 count 等量评估。
            """,
            "learning-definitions",
        ),
        code(
            """
            config = json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))
            display(pd.DataFrame([{
                "version": config["version"],
                "architecture": config["architecture"],
                "task_type": config["task_type"],
                "noise_source": config["noise_source"],
                "position_encodings": ", ".join(config["position_encodings"]),
                "training_objective": config["training_objective"],
                "count_sampling": config["count_sampling_definition"],
            }]))

            for name in (
                "model_specifications.csv",
                "training_count_distribution.csv",
                "time_to_99.csv",
                "autoregressive_by_bin.csv",
            ):
                path = RUN_DIR / "tables" / name
                if path.exists() and path.stat().st_size:
                    display(Markdown(f"**{name}**"))
                    display(pd.read_csv(path))
            """,
            "inspect-tables",
        ),
        code(
            """
            for name in (
                "training_count_distribution.png",
                "learning_loss.png",
                "learning_accuracy_by_bin.png",
                "final_accuracy_by_count.png",
            ):
                path = RUN_DIR / "figures" / name
                if path.exists():
                    display(Markdown(f"**{name}**"))
                    display(Image(filename=str(path)))
            """,
            "learning-figures",
        ),
        markdown(
            """
            ## 6. Attention and hidden-state diagnostics

            Attention heatmaps are descriptive routing signatures rather than causal proof. Broad score rewards
            attention mass spread across prompt needles; raw k-to-k mass is attention from trace index `k` to the
            kth target occurrence; trace-marker mass measures final-query readout from the gold trace. State tables
            report held-out count probes and PCA of per-count centroids.
            """,
            "analysis-heading",
        ),
        code(
            """
            for pattern in (
                "attention_signatures_*.png",
                "targeted_retrieval_by_bin_*.png",
                "state_probe_*.png",
                "state_pca_variance_*.png",
                "state_centroids_*.png",
            ):
                for path in sorted((RUN_DIR / "figures").glob(pattern)):
                    display(Markdown(f"**{path.name}**"))
                    display(Image(filename=str(path)))
            """,
            "analysis-figures",
        ),
        markdown("## 7. Save the complete result bundle to Google Drive", "save-heading"),
        code(
            """
            import shutil
            from datetime import datetime

            DRIVE_SAVE_COMPLETED = False
            SAVED_RESULT_DIR = None
            if DRIVE_READY:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                SAVED_RESULT_DIR = DRIVE_RESULTS_ROOT / f"{RUN_NAME}_{stamp}"
                shutil.copytree(RUN_DIR, SAVED_RESULT_DIR, dirs_exist_ok=True)
                DRIVE_SAVE_COMPLETED = (SAVED_RESULT_DIR / "config.json").exists()
                print("Saved:", SAVED_RESULT_DIR)
                print("Verified:", DRIVE_SAVE_COMPLETED)
            else:
                print("Drive unavailable; local result remains at", RUN_DIR.resolve())
            """,
            "save-drive",
            ["google-drive-save"],
        ),
        markdown("## 8. Optional: push notebook/code changes to GitHub", "github-heading"),
        code(
            """
            PUSH_TO_GITHUB = False
            COMMIT_MESSAGE = f"Add/update {VERSION} experiment"
            if PUSH_TO_GITHUB:
                subprocess.run(["git", "status", "--short"], check=True)
                subprocess.run(
                    ["git", "add", "src", "tests", "notebooks", "scripts", "docs", "README.md", "pyproject.toml"],
                    check=True,
                )
                subprocess.run(["git", "commit", "-m", COMMIT_MESSAGE], check=False)
                subprocess.run(["git", "push", "origin", "HEAD"], check=True)
            else:
                print("PUSH_TO_GITHUB=False; no GitHub write performed")
            """,
            "github-push",
        ),
        code(
            """
            AUTO_DISCONNECT_AFTER_DRIVE_SAVE = False
            if AUTO_DISCONNECT_AFTER_DRIVE_SAVE and DRIVE_SAVE_COMPLETED and Path("/content").exists():
                from google.colab import runtime
                runtime.unassign()
            else:
                print("Runtime kept connected")
            """,
            "disconnect-runtime",
        ),
    ]


def build(version: str) -> Path:
    notebook = {
        "cells": cells(version, SPECS[version]),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": f"Trace_Count_{version}_Colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOKS / f"Trace_Count_{version}_Colab.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for version in SPECS:
        print(build(version))
