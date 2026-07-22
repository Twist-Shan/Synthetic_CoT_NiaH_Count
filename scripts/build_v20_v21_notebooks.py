from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def markdown(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str, cell_id: str, tags: list[str] | None = None) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {"tags": tags} if tags else {},
        "execution_count": None,
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def build(version: str) -> Path:
    if version not in {"v20", "v21"}:
        raise ValueError(version)
    package = f"synthetic_counting_{version}"
    module = f"{package}.run_{version}"
    tokenization = "atomic: one dedicated token for every integer 1..30" if version == "v20" else "digit-wise: shared <D0>..<D9> tokens"
    causal_note = (
        "The full v10-compatible causal suite runs after the phase analysis."
        if version == "v20"
        else "The shared phase-local causal interventions run; the atomic-token-only v10 port is marked not applicable."
    )
    cells = [
        markdown(
            f"""
            # Trace Count {version}: query-first RoPE, counts 1–30

            Controlled setting: 256 Shakespeare characters, a three-character query before
            the data, RoPE, paired nonthinking/thinking models, and count range 1–30.
            Number representation is **{tokenization}**. {causal_note}

            Dense scientific snapshots are stored every 100 steps, full optimizer/RNG recovery
            state every 500 steps, and five snapshots are packed into each shard. This avoids
            thousands of Drive files and avoids raw per-token attention exports.
            """,
            "title",
        ),
        markdown("## 1. Mount Google Drive", "drive-heading"),
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
        markdown("## 2. Clone/update the repo on local Colab storage", "setup-heading"),
        code(
            f"""
            import os
            import signal
            import subprocess
            import sys
            import time
            from pathlib import Path

            assert DRIVE_READY, "Run the Drive cell first"
            REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
            preferred = Path("/content/Synthetic_CoT_NiaH_Count")
            candidates = [Path.cwd(), *Path.cwd().parents, preferred]
            repo = next((path.resolve() for path in candidates if (path / "pyproject.toml").exists()), None)
            if repo is None:
                subprocess.run(["git", "clone", REPO_URL, str(preferred)], check=True)
                repo = preferred
            elif (repo / ".git").exists():
                subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
            assert (repo / "src" / "{package}").is_dir(), f"{package} is absent from {{repo}}"
            os.chdir(repo)

            scientific_probe = subprocess.run(
                [sys.executable, "-c", "import numpy,pandas,scipy,matplotlib,seaborn"],
                capture_output=True,
                text=True,
            )
            if scientific_probe.returncode:
                print(scientific_probe.stderr[-2000:])
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
                raise RuntimeError("Scientific ABI repaired. Reconnect and rerun all cells.")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."], check=True)
            src_root = str(repo / "src")
            if src_root not in sys.path:
                sys.path.insert(0, src_root)
            os.environ["PYTHONPATH"] = src_root + os.pathsep + os.environ.get("PYTHONPATH", "")

            import numpy as np
            import pandas as pd
            import torch
            import {package}
            from IPython.display import Image, display

            def run_streaming(command):
                import codecs
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

            print({{
                "repo": str(repo),
                "package": str(Path({package}.__file__).resolve()),
                "torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }})
            """,
            "environment-setup",
        ),
        markdown("## 3. Auditable experiment settings", "settings-heading"),
        code(
            f"""
            VERSION = "{version}"
            PRESET = "main"                 # change to debug for a short end-to-end check
            SEED = 1234
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            COUNT_MAX_THRESHOLD = 30
            NEEDLE_POOL_SIZE = 100
            NEEDLE_POOL_FREQUENCY_THRESHOLD = 0.12
            TASK_OCCURRENCE_RATIO = 1.0
            MAX_TRAIN_STEPS = 10_000
            MAX_STEPS_FOR_LANGUAGE_PRED = 1_500
            CHECKPOINT_EVERY_STEPS = 100     # model-only scientific snapshot
            RECOVERY_EVERY_STEPS = 500       # full optimizer/RNG recovery state
            SNAPSHOT_SHARD_EVERY_STEPS = 500 # five 100-step snapshots per file
            EVAL_EVERY_STEPS = 500
            AR_EVAL_EVERY_STEPS = 1_000
            AR_EXAMPLES_PER_COUNT = 2
            PERMUTATION_EXAMPLES_PER_COUNT = 1  # each expands to all six query orders
            EVAL_EXAMPLES_PER_COUNT = 10
            FINAL_EXAMPLES_PER_COUNT = 50
            PHASE_SELECTION_EXAMPLES_PER_COUNT = 2
            PHASE_REPORT_EXAMPLES_PER_COUNT = 1
            if PRESET == "debug":
                COUNT_MAX_THRESHOLD = 4
                MAX_TRAIN_STEPS = 6
                MAX_STEPS_FOR_LANGUAGE_PRED = 6
                CHECKPOINT_EVERY_STEPS = 3
                RECOVERY_EVERY_STEPS = 3
                SNAPSHOT_SHARD_EVERY_STEPS = 3
                EVAL_EVERY_STEPS = 3
                AR_EVAL_EVERY_STEPS = 3
                AR_EXAMPLES_PER_COUNT = 1
                EVAL_EXAMPLES_PER_COUNT = 2
                FINAL_EXAMPLES_PER_COUNT = 2
                PHASE_SELECTION_EXAMPLES_PER_COUNT = 1
            OUT_ROOT = "runs/synthetic_counting_{version}"
            RUN_NAME = None
            SKIP_COMPLETED = True
            AUTO_DISCONNECT = True
            DISCONNECT_DELAY_SECONDS = 10
            # The run folder itself is the Drive result folder; no second checkpoint copy.
            CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT

            from {package}.config import preset_config
            PLANNED_CONFIG = preset_config(
                PRESET,
                seed=SEED,
                device=DEVICE,
                count_max_threshold=COUNT_MAX_THRESHOLD,
                needle_pool_size=NEEDLE_POOL_SIZE,
                needle_pool_frequency_threshold=NEEDLE_POOL_FREQUENCY_THRESHOLD,
                task_occurrence_ratio=TASK_OCCURRENCE_RATIO,
                train_steps=MAX_TRAIN_STEPS,
                max_steps_for_language_pred=MAX_STEPS_FOR_LANGUAGE_PRED,
                checkpoint_every=CHECKPOINT_EVERY_STEPS,
                recovery_every=RECOVERY_EVERY_STEPS,
                snapshot_shard_every=SNAPSHOT_SHARD_EVERY_STEPS,
                eval_every=EVAL_EVERY_STEPS,
                ar_eval_every=AR_EVAL_EVERY_STEPS,
                ar_examples_per_count=AR_EXAMPLES_PER_COUNT,
                permutation_examples_per_count=PERMUTATION_EXAMPLES_PER_COUNT,
                eval_examples_per_count=EVAL_EXAMPLES_PER_COUNT,
                final_examples_per_count=FINAL_EXAMPLES_PER_COUNT,
                phase_head_selection_examples_per_count=PHASE_SELECTION_EXAMPLES_PER_COUNT,
                phase_examples_per_count=PHASE_REPORT_EXAMPLES_PER_COUNT,
            )
            print(PLANNED_CONFIG.to_dict())
            print({{
                "sequence_layout": "<BOS> query[5] data[256] output",
                "number_representation": PLANNED_CONFIG.count_tokenization,
                "max_render_len": PLANNED_CONFIG.max_render_len,
                "dense_snapshots_per_model": len(range(0, MAX_TRAIN_STEPS + 1, CHECKPOINT_EVERY_STEPS)),
                "snapshot_files_per_model": MAX_TRAIN_STEPS // SNAPSHOT_SHARD_EVERY_STEPS + 1,
                "recovery_policy": "rolling latest plus pinned objective boundary and final",
            }})
            """,
            "runtime-settings",
        ),
        markdown("## 4. Prepare fixed data, pool, and evaluation manifests", "prepare-heading"),
        code(
            f"""
            base_cmd = [
                sys.executable, "-u", "-m", "{module}",
                "--preset", PRESET,
                "--device", DEVICE,
                "--seed", str(SEED),
                "--count-max-threshold", str(COUNT_MAX_THRESHOLD),
                "--needle-pool-size", str(NEEDLE_POOL_SIZE),
                "--needle-pool-frequency-threshold", str(NEEDLE_POOL_FREQUENCY_THRESHOLD),
                "--task-occurrence-ratio", str(TASK_OCCURRENCE_RATIO),
                "--train-steps", str(MAX_TRAIN_STEPS),
                "--max-steps-for-language-pred", str(MAX_STEPS_FOR_LANGUAGE_PRED),
                "--checkpoint-every", str(CHECKPOINT_EVERY_STEPS),
                "--recovery-every", str(RECOVERY_EVERY_STEPS),
                "--snapshot-shard-every", str(SNAPSHOT_SHARD_EVERY_STEPS),
                "--eval-every", str(EVAL_EVERY_STEPS),
                "--ar-eval-every", str(AR_EVAL_EVERY_STEPS),
                "--ar-examples-per-count", str(AR_EXAMPLES_PER_COUNT),
                "--permutation-examples-per-count", str(PERMUTATION_EXAMPLES_PER_COUNT),
                "--eval-examples-per-count", str(EVAL_EXAMPLES_PER_COUNT),
                "--final-examples-per-count", str(FINAL_EXAMPLES_PER_COUNT),
                "--phase-head-selection-examples-per-count", str(PHASE_SELECTION_EXAMPLES_PER_COUNT),
                "--phase-examples-per-count", str(PHASE_REPORT_EXAMPLES_PER_COUNT),
                "--out-root", OUT_ROOT,
                "--checkpoint-sync-root", str(CHECKPOINT_SYNC_ROOT),
                "--model-variant", "rope/nonthinking",
                "--model-variant", "rope/thinking",
            ]
            if RUN_NAME is not None:
                base_cmd += ["--run-name", RUN_NAME]
            if SKIP_COMPLETED:
                base_cmd.append("--skip-completed")
            run_streaming([*base_cmd, "--stage", "prepare"])

            from synthetic_counting_v20.config import default_run_name
            RUN_DIR = Path(OUT_ROOT) / (RUN_NAME or default_run_name(PLANNED_CONFIG))
            DRIVE_RUN_DIR = CHECKPOINT_SYNC_ROOT / RUN_DIR.name
            print("RUN_DIR:", RUN_DIR.resolve())
            print("DRIVE_RUN_DIR:", DRIVE_RUN_DIR)
            """,
            "prepare-data",
        ),
        markdown("## 5. Train the paired models (live progress is streamed)", "train-heading"),
        code(
            """
            training_started = time.perf_counter()
            run_streaming([*base_cmd, "--stage", "train"])
            print(f"Training block: {time.perf_counter() - training_started:.1f} seconds")
            """,
            "train",
        ),
        markdown("## 6. Dense phase, geometry, and causal analyses", "analysis-heading"),
        code(
            """
            analysis_started = time.perf_counter()
            analysis_stages = "phase,causal,plots" if PRESET == "main" else "phase,plots"
            run_streaming([*base_cmd, "--stage", analysis_stages])
            print(f"Analysis block: {time.perf_counter() - analysis_started:.1f} seconds")

            expected = [
                RUN_DIR / "analysis" / "phase_transition" / "manifest.json",
                RUN_DIR / "analysis" / "phase_transition" / "interactive_manifold_3d.html",
                RUN_DIR / "analysis" / "phase_transition" / "tables" / "phase_transition_candidates.csv",
                RUN_DIR / "tables" / "training_token_exposure_by_k.csv",
                RUN_DIR / "figures" / "dense_fixed_head_emergence.png",
                RUN_DIR / "figures" / "dense_marker_manifold_emergence.png",
                RUN_DIR / "figures" / "milestone_local_head_causality.png",
            ]
            if VERSION == "v20" and PRESET == "main":
                expected.append(RUN_DIR / "analysis" / "v10_port" / "manifest.json")
            missing = [str(path) for path in expected if not path.exists()]
            assert not missing, "Missing required outputs: " + str(missing)
            for path in expected:
                print(path.relative_to(RUN_DIR), path.stat().st_size)
            """,
            "analysis",
        ),
        markdown("## 7. Inspect the main diagnostics", "inspect-heading"),
        code(
            """
            for filename in (
                "training_token_exposure_by_k.png",
                "dense_phase_behavior_by_count.png",
                "dense_fixed_head_emergence.png",
                "dense_marker_manifold_emergence.png",
                "milestone_local_head_causality.png",
            ):
                path = RUN_DIR / "figures" / filename
                if path.exists():
                    display(Image(filename=str(path)))
            display(pd.read_csv(RUN_DIR / "analysis" / "phase_transition" / "tables" / "fixed_head_rankings.csv").head(12))
            display(pd.read_csv(RUN_DIR / "analysis" / "phase_transition" / "tables" / "milestone_local_head_causality.csv"))
            """,
            "inspect",
        ),
        markdown("## 8. Verify Drive persistence and optionally disconnect", "finish-heading"),
        code(
            """
            # Re-running plots also triggers the pipeline's incremental final Drive sync.
            run_streaming([*base_cmd, "--stage", "plots"])
            required_drive_files = [
                DRIVE_RUN_DIR / "config.json",
                DRIVE_RUN_DIR / "manifest.json",
                DRIVE_RUN_DIR / "analysis" / "phase_transition" / "manifest.json",
                DRIVE_RUN_DIR / "checkpoints" / "rope" / "thinking" / "snapshot_index.csv",
                DRIVE_RUN_DIR / "checkpoints" / "rope" / "thinking" / "final" / "checkpoint.pt",
            ]
            missing = [str(path) for path in required_drive_files if not path.exists()]
            assert not missing, "Drive persistence check failed: " + str(missing)
            print("Drive persistence verified:", DRIVE_RUN_DIR)

            if AUTO_DISCONNECT and Path("/content").exists():
                print(f"Disconnecting in {DISCONNECT_DELAY_SECONDS}s after successful persistence check...")
                time.sleep(DISCONNECT_DELAY_SECONDS)
                try:
                    from google.colab import runtime
                    runtime.unassign()
                except Exception:
                    os.kill(os.getpid(), 9)
            """,
            "finish",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": f"Trace_Count_{version}_Colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = ROOT / "notebooks" / f"Trace_Count_{version}_Colab.ipynb"
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    return output


if __name__ == "__main__":
    for selected in ("v20", "v21"):
        print(build(selected))
