from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def lines(source: str) -> list[str]:
    source = textwrap.dedent(source).strip("\n") + "\n"
    return source.splitlines(keepends=True)


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": lines(source),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": lines(source),
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


MOUNT = r'''
from pathlib import Path

IN_COLAB = False
DRIVE_READY = False
try:
    from google.colab import drive
    drive.mount("/content/drive")
    IN_COLAB = True
    DRIVE_READY = True
except ImportError:
    print("Local runtime: Google Drive mount skipped.")

DRIVE_RESULTS_ROOT = Path(
    "/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/"
    "Synthetic_CoT_NiaH_Count/colab_results"
)
if DRIVE_READY:
    DRIVE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
print({"in_colab": IN_COLAB, "drive_ready": DRIVE_READY})
'''


SETUP = r'''
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
if IN_COLAB:
    ROOT = Path("/content/Synthetic_CoT_NiaH_Count")
    if not (ROOT / ".git").exists():
        subprocess.run(["git", "clone", REPO_URL, str(ROOT)], check=True)
    else:
        subprocess.run(["git", "-C", str(ROOT), "pull", "--ff-only"], check=True)
else:
    candidates = [Path.cwd(), Path.cwd().parent]
    ROOT = next((path.resolve() for path in candidates if (path / "pyproject.toml").exists()), None)
    if ROOT is None:
        raise FileNotFoundError("Run this notebook from the repository or use Colab.")

os.chdir(ROOT)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
print("repo =", ROOT)
print("python =", sys.executable)
'''


STREAM_HELPER = r'''
def stream_command(command, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(map(str, command)), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        captured = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            captured.append(line.rstrip())
        returncode = process.wait()
    if returncode:
        print("---- Last 160 log lines ----")
        print("\n".join(captured[-160:]))
        raise subprocess.CalledProcessError(returncode, command)
'''


SAVE = r'''
import json
import shutil
from datetime import datetime

DRIVE_SAVE_COMPLETED = False
if DRIVE_READY:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = DRIVE_RESULTS_ROOT / f"{RUN_NAME}_{stamp}"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(RUN_DIR, destination)
    DRIVE_SAVE_COMPLETED = True
    print("Saved result bundle:", destination)
else:
    print("Drive is unavailable; results remain at", RUN_DIR)
'''


PUSH = r'''
# Optional. Keep disabled unless you intentionally want to commit notebook/code changes.
PUSH_TO_GITHUB = False
if PUSH_TO_GITHUB:
    subprocess.run(["git", "status", "--short"], check=True)
    print("Review the status above, then commit and push intentionally from a terminal.")
'''


DISCONNECT = r'''
AUTO_DISCONNECT_AFTER_DRIVE_SAVE = True
if IN_COLAB and AUTO_DISCONNECT_AFTER_DRIVE_SAVE and DRIVE_SAVE_COMPLETED:
    from google.colab import runtime
    print("Drive save verified; disconnecting the Colab runtime.")
    runtime.unassign()
else:
    print("Runtime kept alive (not Colab, save incomplete, or auto-disconnect disabled).")
'''


def build_v16_1() -> dict[str, object]:
    cells = [
        markdown(
            """
            # Trace Count v16.1: Split-Local Indexed Shakespeare Windows

            This version keeps the v16 task and four models (`RoPE/RPE x
            non-thinking/thinking`) but changes the data loader. Tiny Shakespeare
            is split before indexing; training windows are consumed without
            replacement and natural count imbalance is preserved.
            """
        ),
        code(MOUNT),
        markdown("## 1. Repository and environment"),
        code(SETUP),
        code(STREAM_HELPER),
        markdown(
            """
            ## 2. Runtime settings

            The main preset uses 10,000 steps and four independent models. A
            checkpoint is copied to Drive every configured checkpoint interval,
            so reconnecting with `SKIP_COMPLETED=True` resumes both model and
            sampler state exactly.
            """
        ),
        code(
            r'''
            import torch
            from synthetic_counting_v11.config import preset_config

            PRESET = "main"  # "debug" first for a short end-to-end check
            STAGE = "all"
            SEED = 1234
            MIN_CANDIDATE_WINDOWS = 128
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            OUT_ROOT = "runs/synthetic_counting_v16_1"
            RUN_NAME = f"v16_1_{PRESET}_split_windows_seed{SEED}"
            SKIP_COMPLETED = True
            CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT / "v16_1_live_checkpoints" if DRIVE_READY else None

            PLANNED_CONFIG = preset_config(
                "v16_1",
                PRESET,
                seed=SEED,
                device=DEVICE,
                min_candidate_windows=MIN_CANDIDATE_WINDOWS,
            )
            print({
                "run_name": RUN_NAME,
                "device": DEVICE,
                "models": PLANNED_CONFIG.model_variants,
                "split_fractions": (
                    PLANNED_CONFIG.corpus_train_fraction,
                    PLANNED_CONFIG.corpus_validation_fraction,
                    1 - PLANNED_CONFIG.corpus_train_fraction - PLANNED_CONFIG.corpus_validation_fraction,
                ),
                "window_sampling": PLANNED_CONFIG.window_sampling,
                "min_candidate_windows": PLANNED_CONFIG.min_candidate_windows,
                "natural_class_imbalance": True,
            })
            '''
        ),
        markdown("## 3. Run training and analysis"),
        code(
            r'''
            cmd = [
                sys.executable, "-u", "-m", "synthetic_counting_v16_1.run_v16_1",
                "--preset", PRESET,
                "--stage", STAGE,
                "--device", DEVICE,
                "--seed", str(SEED),
                "--min-candidate-windows", str(MIN_CANDIDATE_WINDOWS),
                "--out-root", OUT_ROOT,
                "--run-name", RUN_NAME,
            ]
            if SKIP_COMPLETED:
                cmd.append("--skip-completed")
            if CHECKPOINT_SYNC_ROOT is not None:
                cmd += ["--checkpoint-sync-root", str(CHECKPOINT_SYNC_ROOT)]
            stream_command(cmd, Path(OUT_ROOT) / "last_pipeline.log")
            RUN_DIR = Path(OUT_ROOT) / RUN_NAME
            assert (RUN_DIR / "config.json").exists(), RUN_DIR
            print("RUN_DIR =", RUN_DIR.resolve())
            '''
        ),
        markdown("## 4. Inspect the indexed data distribution and results"),
        code(
            r'''
            import pandas as pd
            from IPython.display import Image, display

            for filename in [
                "window_index_summary.csv",
                "training_count_distribution.csv",
                "final_accuracy_by_count.csv",
                "attention_head_summary.csv",
                "state_probe_summary.csv",
            ]:
                path = RUN_DIR / "tables" / filename
                if path.exists() and path.stat().st_size:
                    print("\n", filename)
                    display(pd.read_csv(path))

            figures = sorted((RUN_DIR / "figures").glob("*.png"))
            print(f"Generated {len(figures)} figures")
            for path in figures:
                print(path.name)
                display(Image(filename=str(path)))
            '''
        ),
        markdown("## 5. Save the complete result bundle to Google Drive"),
        code(SAVE),
        markdown("## 6. Optional GitHub step"),
        code(PUSH),
        markdown("## 7. Disconnect only after a verified Drive save"),
        code(DISCONNECT),
    ]
    return notebook(cells)


def build_v18() -> dict[str, object]:
    cells = [
        markdown(
            """
            # Trace Count v18: Skewed Count-128 with Explicit Retrieval Trace

            Four length-1024 models compare uniform versus power-law
            `p(c) proportional to c^-1.5` sampling, crossed with direct and CoT
            output. Prompts contain 256 noise token types and 10 marker types;
            counts range from 1 to 128.

            CoT emits `I_1 M_type(1) ... I_n M_type(n) END C_n`. The ordinal
            `I_k` and final scalar `C_n` use disjoint token families, so targeted
            retrieval and final counting can be analyzed separately.

            **Main/all is four separate 10,000-step training runs.** Drive
            checkpoint sync preserves every 2,000-step checkpoint. The final
            checkpoints then feed v10-style learning-dynamics, attention-head,
            and hidden-state analyses.
            """
        ),
        code(MOUNT),
        markdown("## 1. Repository and environment"),
        code(SETUP),
        code(STREAM_HELPER),
        markdown("## 2. Configuration and four-run suite manifest"),
        code(
            r'''
            import torch
            from synthetic_counting_v18.config import canonical_run_specs, preset_config, select_specs

            PRESET = "main"  # use "debug" for a quick end-to-end check
            SUITE = "all"    # "power", "uniform", or "all"
            # all = train + attention + state + plots. Existing checkpoints can
            # be analyzed with "attention", "state", or "plots" without retraining.
            STAGE = "all"
            SEED = 1234
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            OUT_ROOT = "runs/synthetic_counting_v18"
            RUN_NAME = f"v18_{PRESET}_{SUITE}_seed{SEED}"
            SKIP_COMPLETED = True
            CHECKPOINT_SYNC_ROOT = DRIVE_RESULTS_ROOT / "v18_live_checkpoints" if DRIVE_READY else None

            CONFIG = preset_config(PRESET, seed=SEED, device=DEVICE)
            SPECS = select_specs(SUITE, canonical_run_specs()) if PRESET == "main" else ()
            print(CONFIG.to_dict())
            print({"suite": SUITE, "number_of_training_runs": len(SPECS) if SPECS else 2})
            if PRESET == "main" and SUITE == "all":
                print("This launches 4 x 10,000-step runs: uniform/power x direct/CoT.")
            '''
        ),
        code(
            r'''
            import pandas as pd
            from IPython.display import display

            if SPECS:
                display(pd.DataFrame([spec.__dict__ for spec in SPECS]))
            '''
        ),
        markdown("## 3. Run the focused suite"),
        code(
            r'''
            cmd = [
                sys.executable, "-u", "-m", "synthetic_counting_v18.run_v18",
                "--preset", PRESET,
                "--suite", SUITE,
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
            stream_command(cmd, Path(OUT_ROOT) / "last_pipeline.log")
            RUN_DIR = Path(OUT_ROOT) / RUN_NAME
            assert (RUN_DIR / "config.json").exists(), RUN_DIR
            print("RUN_DIR =", RUN_DIR.resolve())
            '''
        ),
        markdown(
            """
            ## 4. Definitions and result tables

            All behavioral metrics are **free-running greedy-generation** metrics.
            `primary_accuracy` and `token_accuracy` measure the final scalar `C_n`,
            making direct and CoT directly comparable.
            `enumeration_accuracy` checks whether CoT produced a well-formed
            `I_1...I_n` trace of the correct length. `trace_marker_accuracy`
            measures per-position marker identity, and `trace_exact_accuracy`
            requires every index-marker pair to match before `END`.

            Counts are reported in four balanced diagnostic bands: `1-32`, `33-64`,
            `65-96`, and `97-128`. These evaluations are separate from the skewed
            power-law training sampler.

            **Attention scores.** For attention row `A[q,j]`,
            `prompt_needles_mass` sums over prompt-needle positions.
            `needle_entropy_normalized` is entropy within that subset divided by
            `log(n)`; their product is `broad_attention_score`. At CoT query `I_k`,
            `correct_prompt_needle_mass=A[I_k, needle_k]` is raw k-to-k mass,
            `diagonal_dominance` divides it by total prompt-needle mass, and
            `correct_top1` tests whether `needle_k` is the strongest needle. At
            `M_k`, `next_prompt_needle_mass` probes successor preparation. At `END`,
            `trace_markers_mass` measures final readout attention to trace markers.

            **State scores.** Hidden-state index 0 is the embedding; indices 1-4
            are residual states after Transformer Layers 1-4. Held-out nearest-
            centroid accuracy tests exact label separation. Ridge `R^2` tests an
            approximately linear count/progress direction. A position-only baseline
            detects absolute-position leakage. PCA is fit to exact-label centroids.
            """
        ),
        code(
            r'''
            from IPython.display import Image, display

            for filename in [
                "suite_manifest.csv",
                "final_summary.csv",
                "final_by_band.csv",
                "final_by_count.csv",
                "dynamics_summary.csv",
                "dynamics_by_band.csv",
                "attention_summary.csv",
                "state_probe_summary.csv",
                "state_pca_variance.csv",
            ]:
                path = RUN_DIR / "tables" / filename
                if path.exists() and path.stat().st_size:
                    print("\n", filename)
                    display(pd.read_csv(path))
            for path in sorted((RUN_DIR / "figures").glob("*.png")):
                print(path.name)
                display(Image(filename=str(path)))
            '''
        ),
        markdown("## 5. Interactive PC1-PC6 count-centroid geometry"),
        code(
            r'''
            import itertools
            import ipywidgets as widgets
            import plotly.express as px

            centroids = pd.read_csv(RUN_DIR / "tables" / "state_centroids_pca.csv")
            pc_columns = [column for column in centroids.columns if column.startswith("pc")]
            axis_options = list(itertools.combinations(pc_columns, 3))
            run_widget = widgets.Dropdown(options=sorted(centroids["run_name"].unique()), description="run")
            site_widget = widgets.Dropdown(options=sorted(centroids["site"].unique()), description="site")
            layer_widget = widgets.Dropdown(options=sorted(centroids["layer"].unique()), description="state")
            axes_widget = widgets.Dropdown(options=[(" / ".join(x), x) for x in axis_options], description="PC axes")

            def show_geometry(run_name, site, layer, pc_axes):
                subset = centroids[
                    (centroids["run_name"] == run_name)
                    & (centroids["site"] == site)
                    & (centroids["layer"] == layer)
                ].sort_values("state_label")
                if subset.empty:
                    print("No rows for this run/site/state-index combination.")
                    return
                x, y, z = pc_axes
                figure = px.scatter_3d(
                    subset, x=x, y=y, z=z, color="state_label",
                    hover_data=["state_label", "mode", "distribution"],
                    title=f"{run_name} | {site} | hidden-state index {layer}",
                    color_continuous_scale="Viridis",
                )
                figure.update_traces(marker={"size": 4})
                figure.show()

            controls = {"run_name": run_widget, "site": site_widget, "layer": layer_widget, "pc_axes": axes_widget}
            display(widgets.HBox([run_widget, site_widget, layer_widget, axes_widget]))
            display(widgets.interactive_output(show_geometry, controls))
            '''
        ),
        markdown("## 6. Save the complete result bundle to Google Drive"),
        code(SAVE),
        markdown("## 7. Optional GitHub step"),
        code(PUSH),
        markdown("## 8. Disconnect only after a verified Drive save"),
        code(DISCONNECT),
    ]
    return notebook(cells)


def _replace_strings(value: object, replacements: tuple[tuple[str, str], ...]) -> object:
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    return value


def build_v19() -> dict[str, object]:
    """Derive the v19 Colab workflow from v18, then replace its semantics."""

    result = _replace_strings(
        build_v18(),
        (
            ("synthetic_counting_v18", "synthetic_counting_v19"),
            ("run_v18", "run_v19"),
            ("Trace Count v18", "Trace Count v19"),
            ("v18_", "v19_"),
            ("v18 live", "v19 live"),
        ),
    )
    assert isinstance(result, dict)
    cells = result["cells"]
    assert isinstance(cells, list)
    cells[0]["source"] = lines(
        r'''
        # Trace Count v19: v18 with Shared Decimal Digit Tokenization

        This notebook isolates one change from v18: **every ordinal index and
        final count is rendered with the same ten decimal digit tokens**.
        Data distributions, length 1024, count range 1-128, ten marker types,
        model architecture, optimizer, and the four-run
        `uniform/power x direct/CoT` comparison stay fixed.

        Direct completion is `<Count> digits(n) <NumEnd>`. CoT completion is
        `<Index> digits(1) M1 ... <Index> digits(n) Mn </Think>
        <Count> digits(n) <NumEnd>`. For example, index 12 is three semantic
        pieces: `<Index>`, digit `1`, digit `2`; the final digit `2` is the
        attention query that predicts marker 12. The final scalar uses the
        same digits but is introduced by `<Count>`.

        **Main/all is four independent 10,000-step runs.** Checkpoints sync to
        Drive every 2,000 steps. All reported behavior is free-running greedy
        generation, including multi-digit parsing and the explicit number-end
        delimiter.
        '''
    )
    for cell in cells:
        source = "".join(cell.get("source", []))
        if source.startswith("## 4. Definitions and result tables"):
            cell["source"] = lines(
                r'''
                ## 4. Definitions and result tables

                **Digit grammar.** `D0..D9` are shared by trace indices and the
                final answer. `<Index>` and `<Count>` identify the number's role;
                `<NumEnd>` terminates the final scalar. Leading zeros, zero, and
                values outside the configured range are invalid. Thus exact
                final-count accuracy requires every generated digit and the
                delimiter to parse to the gold integer.

                **Free-running metrics.** `primary_accuracy` and
                `token_accuracy` compare the parsed final integer with gold.
                `enumeration_accuracy` requires a well-formed sequence of
                indices `1..n` followed by `</Think>`. `trace_marker_accuracy`
                measures position-wise marker identity, and
                `trace_exact_accuracy` requires the complete digit-indexed trace.

                **Attention anchors.** For trace step `k`, the query that predicts
                marker `M_k` is the final decimal digit of `k`, not `<Index>` and
                not necessarily its first digit. `correct_prompt_needle_mass` is
                raw attention from that query to prompt needle `k`.
                `diagonal_dominance` normalizes that mass within the prompt-needle
                subset, while `correct_top1` asks whether it is the strongest
                needle. At marker `M_k`, `next_prompt_needle_mass` probes successor
                preparation. The final-answer attention/state anchor is the
                `<Count>` role token, whose next-token logits start the scalar.

                **State scores.** Hidden-state index 0 is the embedding; indices
                1-4 are residual states after Layers 1-4. Held-out nearest-
                centroid accuracy and ridge `R^2` measure exact and approximately
                linear count/progress readability. PCA is fitted to label
                centroids and exported through PC1-PC6.
                '''
            )
    return result


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    targets = {
        NOTEBOOK_DIR / "Trace_Count_v16_1_Colab.ipynb": build_v16_1(),
        NOTEBOOK_DIR / "Trace_Count_v18_Colab.ipynb": build_v18(),
        NOTEBOOK_DIR / "Trace_Count_v19_Colab.ipynb": build_v19(),
    }
    for path, contents in targets.items():
        path.write_text(json.dumps(contents, ensure_ascii=False, indent=1), encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
