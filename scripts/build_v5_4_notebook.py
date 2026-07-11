from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "Trace_Count_v5_4_Count_State_Causal_Colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


cells = [
    md(
        """
# Trace Count v5.4: Causal Count-State Analysis

This notebook restores the trained explicit-switch v5 model and addresses a narrower question than attention retrieval:

1. **THINK_OFF:** how is prompt needle count written into the final-answer state?
2. **THINK_ON:** after a trace-shaped retrieval computation, is there a causal count state, and is it shared with THINK_OFF?

The notebook does **not retrain** the model. It distinguishes linear readability from causality using count-direction steering, position-matched residual swaps, and clean-to-corrupt mediation.

Important interpretation: the fixed-trace THINK_ON condition is a counterfactual position control, not an ID accuracy benchmark.
"""
    ),
    md("## 1. Mount Google Drive first"),
    code(
        """
from pathlib import Path

IN_COLAB = Path('/content').exists()
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
else:
    print('Local runtime: Google Drive mount skipped.')
"""
    ),
    md("## 2. Repository and environment"),
    code(
        """
import os, subprocess, sys
from pathlib import Path

REPO_URL = 'https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git'
if IN_COLAB:
    REPO_DIR = Path('/content/Synthetic_CoT_NiaH_Count')
    if not (REPO_DIR / '.git').exists():
        subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
    else:
        subprocess.run(['git', '-C', str(REPO_DIR), 'pull', '--ff-only'], check=True)
    os.chdir(REPO_DIR)
else:
    REPO_DIR = Path.cwd().resolve()

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'], check=True)
print('repo =', REPO_DIR)
print('python =', sys.executable)
"""
    ),
    md("## 3. Restore the completed indexed-trace v5 checkpoint"),
    code(
        """
import json, shutil

DRIVE_RESULTS_ROOT = Path('/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/Synthetic_CoT_NiaH_Count/colab_results') if IN_COLAB else REPO_DIR / 'colab_results'
SOURCE_RUN_OVERRIDE = None
LOCAL_RUN = Path('/content/v5_explicit_switch_source') if IN_COLAB else REPO_DIR / 'colab_results' / 'v5_explicit_switch'

def is_v5_run(path):
    return all((path / rel).exists() for rel in ['config.json', 'vocab.json', 'checkpoints/final.pt'])

if SOURCE_RUN_OVERRIDE is not None:
    source_run = Path(SOURCE_RUN_OVERRIDE)
else:
    direct = DRIVE_RESULTS_ROOT / 'v5_explicit_switch'
    candidates = [direct] + [p.parent for p in DRIVE_RESULTS_ROOT.rglob('config.json')]
    valid = [p for p in candidates if is_v5_run(p)]
    if not valid:
        raise FileNotFoundError(f'No completed v5 run under {DRIVE_RESULTS_ROOT}.')
    source_run = max(valid, key=lambda p: (p / 'checkpoints' / 'final.pt').stat().st_mtime)

cfg = json.loads((source_run / 'config.json').read_text(encoding='utf-8'))
if not cfg.get('trace_indices', False):
    raise ValueError('v5.4 requires corrected indexed traces: trace_indices=true.')
if IN_COLAB:
    if LOCAL_RUN.exists():
        shutil.rmtree(LOCAL_RUN)
    shutil.copytree(source_run, LOCAL_RUN, ignore=shutil.ignore_patterns('v5_4_count_state_causal'))
else:
    LOCAL_RUN = source_run
print('source_run =', source_run)
print('local_run =', LOCAL_RUN)
print({'model': cfg['model'], 'train': cfg['train'], 'trace_indices': cfg['trace_indices']})
"""
    ),
    md(
        """
## 4. Runtime settings

- `debug`: verifies every intervention and output table on a small sample.
- `main`: uses an independent 1,000-example direction-training set, 200-example causal test set, and 90 paired needle-deletion examples.
- `FIXED_TRACE_COUNT=5`: every position-control THINK_ON sequence contains exactly five constant index/marker pairs. Prompt count still varies from 1 to 10, so the close token stays at one absolute position.
"""
    ),
    code(
        """
PRESET = 'debug'  # change to 'main' after the debug run succeeds
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
FIXED_TRACE_COUNT = 5
BATCH_SIZE = 64
ALPHAS = (-2, -1, 0, 1, 2)

if PRESET == 'debug':
    TRAIN_EXAMPLES_PER_COUNT = 5
    EVAL_EXAMPLES_PER_COUNT = 2
    MEDIATION_EXAMPLES_PER_COUNT = 2
else:
    TRAIN_EXAMPLES_PER_COUNT = 100
    EVAL_EXAMPLES_PER_COUNT = 20
    MEDIATION_EXAMPLES_PER_COUNT = 10

print({
    'preset': PRESET,
    'device': DEVICE,
    'fixed_trace_count': FIXED_TRACE_COUNT,
    'direction_train_n': 10 * TRAIN_EXAMPLES_PER_COUNT,
    'causal_eval_n': 10 * EVAL_EXAMPLES_PER_COUNT,
    'mediation_pairs': 9 * MEDIATION_EXAMPLES_PER_COUNT,
})
"""
    ),
    md(
        """
## 5. Run all causal count-state experiments

The runner saves per-example rows, summaries, fitted directions, configuration, and figures under `v5_4_count_state_causal/` inside the restored run.
"""
    ),
    code(
        """
from synthetic_counting_extensions.v5_4_count_state_causal import run_v5_4_count_state_causal

outputs = run_v5_4_count_state_causal(
    LOCAL_RUN,
    train_examples_per_count=TRAIN_EXAMPLES_PER_COUNT,
    eval_examples_per_count=EVAL_EXAMPLES_PER_COUNT,
    mediation_examples_per_count=MEDIATION_EXAMPLES_PER_COUNT,
    fixed_trace_count=FIXED_TRACE_COUNT,
    batch_size=BATCH_SIZE,
    alphas=ALPHAS,
    device=DEVICE,
)
OUT_DIR = LOCAL_RUN / 'v5_4_count_state_causal'
print('OUT_DIR =', OUT_DIR)
"""
    ),
    md(
        """
## 6. Experiment A: readable geometry versus a genuine `+1` direction

At the same query token and block output, compute class means `mu_n`. `adjacent_delta_cosine_mean` asks whether all vectors `mu_(n+1)-mu_n` point in the same direction. `projection_r2_heldout` asks whether a direction fitted on an independent training set predicts numeric count on held-out prompts.

High values are stronger than a generic 10-class probe, but still descriptive until steering changes the model output.
"""
    ),
    code(
        """
from IPython.display import Image, Markdown, display
display(outputs['direction_geometry'].sort_values(['site', 'layer', 'method']))
display(outputs['cross_mode_direction_cosine'])
display(Image(filename=str(OUT_DIR / 'figures' / 'count_direction_geometry.png')))
"""
    ),
    md(
        """
## 7. Experiment B: causal count-direction steering

At one block output, either add `alpha * step_size * direction` or apply the class-specific centroid transport `mu_(n+alpha)-mu_n` to the query residual, then run all later computation normally.

- Y-axis: probability-weighted expected count after intervention minus the same example at `alpha=0`.
- A causal count direction should show a monotonic slope with alpha.
- `centroid_transport` asks whether moving from the count-`n` centroid toward the count-`n+alpha` centroid changes the answer when no single global axis exists.
- `random_orthogonal` and `shuffled_ridge` are equal-scale controls.
- `cross_adjacent_mean` and `cross_centroid_transport` apply THINK_OFF geometry to a THINK_ON state and test whether both modes share a causal count representation.

Evidence ladder: global direction steering is the strongest claim of one arithmetic axis; centroid transport supports a class-specific count manifold; the whole-state swap below is the broadest sufficiency test.
"""
    ),
    code(
        """
display(outputs['steering_summary'].sort_values(['target_site', 'layer', 'direction_method', 'alpha']))
display(Image(filename=str(OUT_DIR / 'figures' / 'count_direction_steering.png')))
"""
    ),
    md(
        """
## 8. Experiment C: position-matched residual state swap

Replace a count-n receiver residual with a count-(n+1) donor residual at the same layer and absolute token position. `follows_donor` is the fraction of outputs equal to n+1. The same-count, different-prompt swap checks whether the result is merely arbitrary prompt transfer.

This tests whether the whole residual state is sufficient to transfer count. It is less selective than direction steering because the complete 256-dimensional vector is replaced.
"""
    ),
    code(
        """
display(outputs['state_swap_summary'])
display(Image(filename=str(OUT_DIR / 'figures' / 'count_state_swap.png')))
"""
    ),
    md(
        """
## 9. Experiment D: where a `+1 needle` difference enters the causal path

Replace one prompt needle with noise, preserving prompt length. For THINK_ON position control, clean and corrupt prompts receive exactly the same constant trace, so trace length, trace content, and close position are fixed.

- `state_step_units`: projection of `h_clean-h_corrupt` onto the count direction, divided by one fitted count step.
- `normalized_recovery`: how much clean `<C_n>` versus corrupt `<C_(n-1)>` margin is restored by patching a clean residual/head group.
- Residual patches locate the layer where the answer state becomes sufficient.
- Head patches distinguish direct broad aggregation from targeted retrieval and trace-readout candidates.
"""
    ),
    code(
        """
display(outputs['mediation_summary'])
display(Image(filename=str(OUT_DIR / 'figures' / 'count_residual_mediation.png')))
head_figure = OUT_DIR / 'figures' / 'count_head_mediation.png'
if head_figure.exists():
    display(Image(filename=str(head_figure)))
else:
    display(Markdown('No head-group figure: run v5.3 first so `head_groups.json` is available.'))
"""
    ),
    md(
        """
## 10. Interpretation checklist

Do not conclude “arithmetic direction” from R² alone. A strong result requires:

1. held-out adjacent differences are approximately parallel;
2. positive/negative alpha cause monotonic positive/negative count shifts;
3. random and shuffled controls do not;
4. whole-state swaps transfer donor count at a position-matched site;
5. needle-deletion patches restore both the direction projection and the clean answer margin;
6. the THINK_OFF direction transfers into at least one THINK_ON readout condition.
"""
    ),
    code(
        """
import pandas as pd

steer = outputs['steering_summary']
key = steer[
    steer['direction_method'].isin(['adjacent_mean', 'cross_adjacent_mean', 'centroid_transport', 'cross_centroid_transport', 'random_orthogonal'])
    & steer['alpha'].isin([-1, 1])
][['target_site', 'source_site', 'direction_method', 'layer', 'alpha', 'causal_expected_shift', 'desired_accuracy']]
display(key.sort_values(['target_site', 'layer', 'direction_method', 'alpha']))

swap = outputs['state_swap_summary'][['site', 'control', 'layer', 'follows_donor', 'causal_expected_shift']]
display(swap.sort_values(['site', 'layer', 'control']))
"""
    ),
    md("## 11. Save the complete v5.4 bundle to Google Drive"),
    code(
        """
from datetime import datetime

DRIVE_SAVE_COMPLETED = False
if IN_COLAB:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    destination = DRIVE_RESULTS_ROOT / f'v5_4_count_state_causal_seed1234_{stamp}'
    shutil.copytree(OUT_DIR, destination)
    DRIVE_SAVE_COMPLETED = True
    print('saved to', destination)
else:
    print('Local output:', OUT_DIR)
"""
    ),
    md("## 12. Optional: disconnect Colab only after the Drive save"),
    code(
        """
AUTO_DISCONNECT = False
if IN_COLAB and AUTO_DISCONNECT:
    if not DRIVE_SAVE_COMPLETED:
        raise RuntimeError('Refusing to disconnect before a confirmed Drive save.')
    from google.colab import runtime
    runtime.unassign()
else:
    print('Runtime left connected. Set AUTO_DISCONNECT=True after inspecting the result cells.')
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(OUTPUT)
