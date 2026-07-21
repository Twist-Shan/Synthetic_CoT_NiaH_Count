from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "Trace_Count_v16_3_Colab.ipynb"


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


def build() -> Path:
    cells = [
        markdown(
            """
            # Trace Count v16_3: data first, three-character query second

            This revision copies the complete v16.2 RoPE training and analysis pipeline while
            changing one controlled factor: both nonthinking and thinking task sequences place
            the full Shakespeare data window before the `<CountChar> ... <Sep>` query. It retains
            fixed train/validation/test suites, live Drive checkpoints, resumable training,
            checkpoint dynamics, final result export, and optional automatic runtime disconnect.
            """,
            "title",
        ),
        markdown("## 1. Mount Google Drive first", "drive-heading"),
        code(
            """
            from pathlib import Path

            # Match v16/v16.1: Drive stores checkpoints and result bundles only.
            # Source code is cloned to the fast Colab-local filesystem in the next cell.
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
        markdown("## 2. Repository and environment", "setup-heading"),
        code(
            """
            import os
            import signal
            import subprocess
            import sys
            import time
            from pathlib import Path

            setup_started = time.perf_counter()
            assert DRIVE_READY, "Run the Google Drive mount cell before environment setup"
            DRIVE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

            # Match v16/v16.1: clone or fast-forward the public repo on Colab's local disk.
            REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
            preferred = Path("/content/Synthetic_CoT_NiaH_Count")
            candidates = [Path.cwd(), *Path.cwd().parents, preferred]
            repo = next(
                (path.resolve() for path in candidates if (path / "pyproject.toml").exists()),
                None,
            )
            if repo is None:
                subprocess.run(["git", "clone", REPO_URL, str(preferred)], check=True)
                repo = preferred
            elif Path("/content").exists() and (repo / ".git").exists():
                subprocess.run(
                    ["git", "-C", str(repo), "pull", "--ff-only"],
                    check=True,
                )
            assert (repo / "src" / "synthetic_counting_v16_3").is_dir(), (
                f"The checked-out repository at {repo} does not contain v16.3. "
                "Confirm that the clone/pull reached the current origin/main."
            )
            os.chdir(repo)

            # Use the same ABI repair used by v16 when a reused Colab runtime has an
            # inconsistent NumPy/pandas/scientific stack.
            scientific_probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import numpy,pandas,scipy,matplotlib,seaborn",
                ],
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
                raise RuntimeError(
                    "Scientific ABI repaired. Reconnect and rerun all cells."
                )

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."],
                check=True,
            )

            # Editable-install .pth files are processed only when Python starts. Make the
            # copied src-layout package importable in this already-running notebook kernel.
            src_root = str(repo / "src")
            if src_root not in sys.path:
                sys.path.insert(0, src_root)
            os.environ["PYTHONPATH"] = src_root + os.pathsep + os.environ.get("PYTHONPATH", "")

            import synthetic_counting_v16_3
            import numpy as np
            import pandas as pd
            import torch
            from IPython.display import Image, Markdown, display

            package_path = Path(synthetic_counting_v16_3.__file__).resolve()
            assert package_path.is_relative_to(Path(src_root)), (
                f"Notebook kernel imported stale package from {package_path}"
            )
            subprocess_package_path = Path(
                subprocess.check_output(
                    [
                        sys.executable,
                        "-c",
                        "import synthetic_counting_v16_3 as p; print(p.__file__)",
                    ],
                    text=True,
                ).strip()
            ).resolve()
            assert subprocess_package_path.is_relative_to(Path(src_root)), (
                f"Subprocess imported stale package from {subprocess_package_path}"
            )

            corpus_path = (
                repo
                / "src"
                / "synthetic_counting_v11"
                / "resources"
                / "tiny_shakespeare"
                / "input.txt"
            )
            assert corpus_path.exists(), f"Tiny Shakespeare is missing: {corpus_path}"

            print({
                "repo": str(repo),
                "repo_url": REPO_URL,
                "drive_results": str(DRIVE_RESULTS_ROOT),
                "src_root": src_root,
                "kernel_package": str(package_path),
                "subprocess_package": str(subprocess_package_path),
                "corpus": str(corpus_path),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            })
            print(f"Environment setup block: {time.perf_counter() - setup_started:.1f} seconds")
            """,
            "environment-setup",
        ),
        code(
            """
            RUN_TESTS = False  # optional developer preflight; not required for training
            if RUN_TESTS:
                test_process = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "tests/test_synthetic_counting_v16_3.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                print(test_process.stdout, end="")
                test_process.check_returncode()
            else:
                print("Optional repository tests skipped; pipeline validation remains enabled.")
            """,
            "tests",
        ),
        markdown("## 3. Easy-to-edit settings", "settings-heading"),
        code(
            """
            VERSION = "v16_3"
            SEQUENCE_LAYOUT = "data_then_query"  # fixed in v16.3: data window, then query, then output
            PRESET = "main"                 # use "debug" for an end-to-end check
            SEED = 1234
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            TASK_OCCURRENCE_RATIO = 1.0      # reference run: every example is a counting task
            COUNT_MAX_THRESHOLD = 10
            WEIGHT_DECAY = 0.01              # AdamW decay on all trainable parameters; 0.0 disables it
            FINAL_COUNT_LOSS_WEIGHT = 1.0    # >1 upweights the final numeric answer target
            COT_TRACE_LOSS_WEIGHT = 1.0      # >1 upweights CoT trace indices and marker characters
            RPE_max_update = False           # unused by the reference RoPE pair
            RUN_ROPE_NONTHINKING = True
            RUN_ROPE_THINKING = True
            RUN_RPE_NONTHINKING = False      # optional ablation; not in the reference report
            RUN_RPE_THINKING = False         # optional ablation; not in the reference report
            MAX_TRAIN_STEPS = 10_000         # optimizer steps for each enabled model
            MAX_STEPS_FOR_LANGUAGE_PRED = 1_500  # through this step use all-token LM loss; afterward train task output only
            CHECKPOINT_EVERY_STEPS = 500      # save step 0, every N steps, the objective boundary, and the final step
            EVAL_EXAMPLES_PER_COUNT = 50     # examples for each count; suite size = this value x COUNT_MAX_THRESHOLD (reference: 500)
            NEEDLE_POOL_SIZE = 100
            NEEDLE_POOL_FREQUENCY_THRESHOLD = 0.04
            OUT_ROOT = "runs/synthetic_counting_v16_3"
            RUN_NAME = None                  # default name records all important settings
            SKIP_COMPLETED = True
            AUTO_DISCONNECT = True           # after the final Drive copy succeeds
            DISCONNECT_DELAY_SECONDS = 10    # leave a short window for the final save message
            CHECKPOINT_SYNC_ROOT = (
                DRIVE_RESULTS_ROOT / "v16_3_live_checkpoints" if DRIVE_READY else None
            )

            ENABLED_MODEL_VARIANTS = tuple(
                variant
                for enabled, variant in (
                    (RUN_ROPE_NONTHINKING, "rope/nonthinking"),
                    (RUN_ROPE_THINKING, "rope/thinking"),
                    (RUN_RPE_NONTHINKING, "rpe/nonthinking"),
                    (RUN_RPE_THINKING, "rpe/thinking"),
                )
                if enabled
            )
            if not ENABLED_MODEL_VARIANTS:
                raise ValueError("Enable at least one model variant")

            from synthetic_counting_v16_3.config import preset_config
            PLANNED_CONFIG = preset_config(
                PRESET,
                seed=SEED,
                device=DEVICE,
                task_occurrence_ratio=TASK_OCCURRENCE_RATIO,
                count_max_threshold=COUNT_MAX_THRESHOLD,
                weight_decay=WEIGHT_DECAY,
                final_count_loss_weight=FINAL_COUNT_LOSS_WEIGHT,
                cot_trace_loss_weight=COT_TRACE_LOSS_WEIGHT,
                rpe_max_update=RPE_max_update,
                enabled_model_variants=ENABLED_MODEL_VARIANTS,
                train_steps=MAX_TRAIN_STEPS,
                max_steps_for_language_pred=MAX_STEPS_FOR_LANGUAGE_PRED,
                checkpoint_every=CHECKPOINT_EVERY_STEPS,
                eval_examples_per_count=EVAL_EXAMPLES_PER_COUNT,
                needle_pool_size=NEEDLE_POOL_SIZE,
                needle_pool_frequency_threshold=NEEDLE_POOL_FREQUENCY_THRESHOLD,
            )
            assert PLANNED_CONFIG.sequence_layout == SEQUENCE_LAYOUT
            EVAL_EXAMPLES_PER_SUITE = EVAL_EXAMPLES_PER_COUNT * COUNT_MAX_THRESHOLD
            PERIODIC_TF_EXAMPLES_PER_MODEL = 7 * EVAL_EXAMPLES_PER_SUITE
            LANGUAGE_PREDICTION_STEPS = min(MAX_STEPS_FOR_LANGUAGE_PRED, MAX_TRAIN_STEPS)
            TASK_OUTPUT_ONLY_STEPS = max(0, MAX_TRAIN_STEPS - MAX_STEPS_FOR_LANGUAGE_PRED)
            PLANNED_CHECKPOINT_STEPS = sorted({
                0,
                *range(CHECKPOINT_EVERY_STEPS, MAX_TRAIN_STEPS + 1, CHECKPOINT_EVERY_STEPS),
                min(MAX_STEPS_FOR_LANGUAGE_PRED, MAX_TRAIN_STEPS),
                MAX_TRAIN_STEPS,
            })
            print({
                "config": PLANNED_CONFIG.to_dict(),
                "sequence_layout": PLANNED_CONFIG.sequence_layout,
                "sequence_templates": {
                    "nonthinking": "<BOS> data[256] <CountChar> q1 q2 q3 <Sep> <Ans> count <EOS>",
                    "thinking": "<BOS> data[256] <CountChar> q1 q2 q3 <Sep> <Think> trace </Think> <Ans> count <EOS>",
                },
                "enabled_model_variants": ENABLED_MODEL_VARIANTS,
                "number_of_models": len(ENABLED_MODEL_VARIANTS),
                "weight_decay": WEIGHT_DECAY,
                "rpe_max_update": RPE_max_update,
                "max_relative_distance": PLANNED_CONFIG.max_relative_distance,
                "language_prediction_steps": LANGUAGE_PREDICTION_STEPS,
                "task_output_only_steps": TASK_OUTPUT_ONLY_STEPS,
                "task_output_starts": {"nonthinking": "<Ans>", "thinking": "<Think>"},
                "max_steps_per_model": MAX_TRAIN_STEPS,
                "total_planned_optimizer_steps": len(ENABLED_MODEL_VARIANTS) * MAX_TRAIN_STEPS,
                "checkpoint_every_steps": CHECKPOINT_EVERY_STEPS,
                "planned_checkpoint_steps_per_model": PLANNED_CHECKPOINT_STEPS,
                "numeric_checkpoints_per_model": len(PLANNED_CHECKPOINT_STEPS),
                "eval_examples_per_suite": EVAL_EXAMPLES_PER_SUITE,
                "periodic_teacher_forced_examples_per_model": PERIODIC_TF_EXAMPLES_PER_MODEL,
            })
            """,
            "runtime-settings",
        ),
        markdown("## 4. Prepare split, pool, and fixed evaluation suites", "prepare-heading"),
        code(
            """
            base_cmd = [
                sys.executable, "-u", "-m", "synthetic_counting_v16_3.run_v16_3",
                "--preset", PRESET,
                "--device", DEVICE,
                "--seed", str(SEED),
                "--task-occurrence-ratio", str(TASK_OCCURRENCE_RATIO),
                "--count-max-threshold", str(COUNT_MAX_THRESHOLD),
                "--weight-decay", str(WEIGHT_DECAY),
                "--final-count-loss-weight", str(FINAL_COUNT_LOSS_WEIGHT),
                "--cot-trace-loss-weight", str(COT_TRACE_LOSS_WEIGHT),
                "--train-steps", str(MAX_TRAIN_STEPS),
                "--max-steps-for-language-pred", str(MAX_STEPS_FOR_LANGUAGE_PRED),
                "--checkpoint-every", str(CHECKPOINT_EVERY_STEPS),
                "--eval-examples-per-count", str(EVAL_EXAMPLES_PER_COUNT),
                "--needle-pool-size", str(NEEDLE_POOL_SIZE),
                "--needle-pool-frequency-threshold", str(NEEDLE_POOL_FREQUENCY_THRESHOLD),
                "--out-root", OUT_ROOT,
            ]
            base_cmd.append("--rpe-max-update" if RPE_max_update else "--no-rpe-max-update")
            for variant in ENABLED_MODEL_VARIANTS:
                base_cmd += ["--model-variant", variant]
            if RUN_NAME is not None:
                base_cmd += ["--run-name", RUN_NAME]
            if CHECKPOINT_SYNC_ROOT is not None:
                base_cmd += ["--checkpoint-sync-root", str(CHECKPOINT_SYNC_ROOT)]
            if SKIP_COMPLETED:
                base_cmd.append("--skip-completed")
            prepare_started = time.perf_counter()
            subprocess.run([*base_cmd, "--stage", "prepare"], check=True)
            print(f"Prepare block: {time.perf_counter() - prepare_started:.1f} seconds")

            from synthetic_counting_v16_3.config import default_run_name
            RUN_DIR = Path(OUT_ROOT) / (RUN_NAME or default_run_name(PLANNED_CONFIG))
            display(pd.read_csv(RUN_DIR / "tables" / "needle_pool.csv"))
            display(Image(filename=str(RUN_DIR / "figures" / "needle_pool_frequency_distribution.png")))
            """,
            "prepare-data",
        ),
        markdown("## 5. Train, analyze, and plot", "run-heading"),
        code(
            """
            training_started = time.perf_counter()
            subprocess.run([*base_cmd, "--stage", "train,attention,state,plots"], check=True)
            print(f"Training/final-diagnostics block: {time.perf_counter() - training_started:.1f} seconds")
            print("RUN_DIR =", RUN_DIR.resolve())
            """,
            "run-pipeline",
        ),
        markdown("## 6. Analyze internal mechanisms across saved checkpoints", "dynamics-heading"),
        code(
            """
            # Metric families are switchable, with explicit dependencies: CKA and
            # counterfactual readouts require hidden-state collection. Defaults use fixed,
            # count-balanced examples at every checkpoint for comparable trajectories.
            RUN_CHECKPOINT_DYNAMICS = True
            RUN_ATTENTION_DYNAMICS = True
            RUN_HIDDEN_STATE_DYNAMICS = True
            RUN_GENERATED_TRACE_DYNAMICS = True
            RUN_COUNTERFACTUAL_DYNAMICS = True
            RUN_REPRESENTATION_STABILITY = True
            FORCE_CHECKPOINT_DYNAMICS = False  # False resumes completed checkpoint partitions

            DYNAMICS_ATTENTION_EXAMPLES_PER_COUNT = 20  # 10 select heads + 10 held-out reporting examples
            DYNAMICS_AR_EXAMPLES_PER_COUNT = 10         # autoregressive generations per true count
            DYNAMICS_STATE_TRAIN_EXAMPLES_PER_COUNT = 40  # ridge/centroid fitting examples per count
            DYNAMICS_STATE_EVAL_EXAMPLES_PER_COUNT = 15   # held-out state examples per count

            if RUN_CHECKPOINT_DYNAMICS:
                from synthetic_counting_v16_3.training import checkpoint_steps

                checkpoint_inventory = {
                    variant: [step for step, _ in checkpoint_steps(RUN_DIR, *variant.split("/"))]
                    for variant in ENABLED_MODEL_VARIANTS
                }
                print("Checkpoint inventory:", checkpoint_inventory)
                required_steps = {min(MAX_STEPS_FOR_LANGUAGE_PRED, MAX_TRAIN_STEPS), MAX_TRAIN_STEPS}
                for variant, available_steps in checkpoint_inventory.items():
                    missing = required_steps - set(available_steps)
                    if missing:
                        raise FileNotFoundError(
                            f"{variant} is missing required checkpoint step(s) {sorted(missing)}; "
                            "restore the run's checkpoints from Google Drive before analysis"
                        )
                dynamics_cmd = [
                    sys.executable, "-u", "scripts/analyze_v16_3_checkpoint_dynamics.py",
                    str(RUN_DIR),
                    "--device", DEVICE,
                    "--attention-examples-per-count", str(DYNAMICS_ATTENTION_EXAMPLES_PER_COUNT),
                    "--ar-examples-per-count", str(DYNAMICS_AR_EXAMPLES_PER_COUNT),
                    "--state-train-examples-per-count", str(DYNAMICS_STATE_TRAIN_EXAMPLES_PER_COUNT),
                    "--state-eval-examples-per-count", str(DYNAMICS_STATE_EVAL_EXAMPLES_PER_COUNT),
                ]
                for enabled, flag in (
                    (RUN_ATTENTION_DYNAMICS, "--skip-attention"),
                    (RUN_HIDDEN_STATE_DYNAMICS, "--skip-states"),
                    (RUN_GENERATED_TRACE_DYNAMICS, "--skip-generated"),
                    (RUN_COUNTERFACTUAL_DYNAMICS, "--skip-counterfactual"),
                    (RUN_REPRESENTATION_STABILITY, "--skip-similarity"),
                ):
                    if not enabled:
                        dynamics_cmd.append(flag)
                if FORCE_CHECKPOINT_DYNAMICS:
                    dynamics_cmd.append("--force")
                dynamics_started = time.perf_counter()
                subprocess.run(dynamics_cmd, check=True)
                print(f"Checkpoint-dynamics block: {time.perf_counter() - dynamics_started:.1f} seconds")
                print("Dynamics manifest:", RUN_DIR / "analysis" / "checkpoint_dynamics" / "manifest.json")
                print("Detailed tables:", RUN_DIR / "tables" / "checkpoint_*.csv")
                for figure_name in (
                    "checkpoint_mechanism_overview.png",
                    "checkpoint_attention_retrieval_emergence.png",
                    "checkpoint_final_count_probe_heatmap.png",
                    "checkpoint_counterfactual_trace_readout.png",
                    "runtime_breakdown.png",
                ):
                    figure_path = RUN_DIR / "figures" / figure_name
                    if figure_path.exists():
                        display(Image(filename=str(figure_path)))
            else:
                print("Checkpoint-dynamics analysis skipped; saved checkpoints remain available.")
            """,
            "checkpoint-dynamics",
        ),
        markdown("## 7. Inspect train-versus-held-out loss curves", "inspect-heading"),
        code(
            """
            inspect_started = time.perf_counter()
            losses = pd.read_csv(RUN_DIR / "tables" / "eval_loss_curves.csv")
            display(losses)
            display(Image(filename=str(RUN_DIR / "figures" / "learning_loss_suites_train_vs_heldout.png")))
            test_summary = RUN_DIR / "tables" / "test_loss_summary.csv"
            if test_summary.exists():
                display(Markdown("**Final-only untouched test results**"))
                display(pd.read_csv(test_summary))
            print(f"Result-display block: {time.perf_counter() - inspect_started:.1f} seconds")
            """,
            "inspect-losses",
        ),
        markdown("## 8. Save the complete result bundle and disconnect", "save-heading"),
        code(
            """
            import shutil
            from datetime import datetime

            save_started = time.perf_counter()
            destination = None
            if DRIVE_READY:
                destination = DRIVE_RESULTS_ROOT / f"{RUN_DIR.name}_{datetime.now():%Y%m%d_%H%M%S}"
                shutil.copytree(RUN_DIR, destination, dirs_exist_ok=True)
                print("Saved:", destination)
            else:
                print("Drive unavailable; results remain at", RUN_DIR.resolve())
            print(f"Result-copy block: {time.perf_counter() - save_started:.1f} seconds")

            # Disconnect only after the complete result copy has returned successfully.
            # Checkpoints were already mirrored throughout training under
            # colab_results/v16_3_live_checkpoints, so an interrupted run can resume too.
            if DRIVE_READY:
                from google.colab import drive, runtime

                drive.flush_and_unmount()
                print("Google Drive flushed and unmounted.")
                if AUTO_DISCONNECT:
                    print(f"Disconnecting this Colab runtime in {DISCONNECT_DELAY_SECONDS} seconds...")
                    time.sleep(DISCONNECT_DELAY_SECONDS)
                    runtime.unassign()
                else:
                    print("AUTO_DISCONNECT=False; runtime remains connected.")
            """,
            "save-drive",
            ["google-drive-save"],
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(build())
