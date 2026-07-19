# Synthetic Trace-Enumeration Counting

This repository contains a sequence of synthetic counting experiments, from the original loss-mask study through controlled CoT, attention, steering, mixed-mode, separator-trace, scaling, and conditional-count variants.

The repo is organized for GitHub + Colab:

```text
configs/                 Model and experiment YAMLs
docs/pipelines/          Experiment specifications and design prompts
notebooks/               Colab notebooks only
scripts/                 Pipeline, notebook, and report utilities
src/                     All importable Python packages
tests/                   Unit and integration tests
artifacts/               Ignored local scratch outputs, including tmp runs
```

Importable code follows a standard `src/` layout. Public module names are unchanged:

| Package | Role |
| --- | --- |
| `trace_counting` | Original v0 data, training, evaluation, probes, and plots |
| `synthetic_niah_v3` | v3 attention and causal analysis |
| `synthetic_niah_v4` | v4 directions, steering, and activation patching |
| `synthetic_niah_v5` | v5 mixed thinking-mode model |
| `synthetic_counting_v6` | v6 separator-trace experiment |
| `synthetic_counting_extensions` | v2.2, v5.2, and v7-v9 extensions |
| `synthetic_counting_v10` | v10 two-model count-30 dynamics, attention causality, and count-state interventions |
| `synthetic_counting_v11` | Shared small-model core for v11-v14 positional, scaling, fixed-data, and Shakespeare experiments |
| `synthetic_counting_v12` | v12 APE wrapper: length 512 and count range 1-50 |
| `synthetic_counting_v13` | v13 APE wrapper: finite fixed-dataset training |
| `synthetic_counting_v14` | v14 APE wrapper: Shakespeare character haystacks |
| `synthetic_counting_v15` | v15 RoPE/RPE Shakespeare haystacks with inserted needles and prompt + completion all-sequence loss |
| `synthetic_counting_v16` | v16 RoPE/RPE native Shakespeare target-letter counting |
| `synthetic_counting_v16_2` | v16.2 split-local Shakespeare window indexing and without-replacement training |
| `synthetic_counting_v17` | v17 RoPE decreasing long-tail synthetic training distribution |
| `synthetic_counting_v18` | Four-model length-1024 count-128 comparison with uniform/power sampling and explicit index-marker CoT traces |
| `synthetic_counting_v19` | v18 comparison with all trace indices and final counts tokenized as shared decimal digits |

Commands such as `python -m synthetic_niah_v4.run_v4` therefore remain valid after
`pip install -e .`.

## Install

Local or Colab:

```bash
pip install -r requirements.txt
pip install -e .
```

If you use Colab, open `notebooks/Trace_Count_v0_Colab.ipynb` and run all cells. The notebook clones/updates the repo when needed, installs dependencies, runs tests, generates the full v0 dataset, trains/evaluates every loss-mask regime, summarizes results, runs probes, displays plots, saves outputs to Google Drive, and prepares an optional GitHub result upload.

For the more NiaH-like v1 experiment, open `notebooks/Trace_Count_v1_Colab.ipynb`. It trains two all-token next-token-prediction models on shorter sparse-counting data: a `think_trace` model with explicit thinking/count trace tokens and an `answer_only` model without thinking tokens. The v1 notebook evaluates ID counts `0-5` and count-OOD `5-10`, then runs linear probes, ridge count-direction extraction, answer-state steering, and attention-to-needle analysis.

For the controlled marker-trace v2 experiment, open `notebooks/Trace_Count_v2_Colab.ipynb`. This notebook follows `docs/pipelines/pipeline_v2_codex_prompt.md`: fixed prompt length, 64 noise-token types, 10 countable marker-token types, count range `1..10`, and two separately trained random-init decoder-only Transformers (`non_thinking` and `thinking`). It intentionally has no ID/OOD split, no variable sequence length, and no steering. The notebook reports training/eval curves by low/mid/high count bin, exact-count accuracy, hidden-state probes, and attention/retrieval diagnostics.

For the v3 attention-head deep dive, open `notebooks/Trace_Count_v3_Colab.ipynb` after a v2 run has produced `checkpoints/final/thinking`. v3 is now analysis-only: it reloads the v2 GPT-2-style thinking model, decomposes attention for the final trace index token into prompt-needle vs previous-trace-token mass, and runs lightweight targeted-head ablations. It does not train a new model, does not generate HTML, and follows `docs/pipelines/pipeline_v3_codex_prompt.md`.

For the v4 steering experiment, open `notebooks/Trace_Count_v4_Colab.ipynb` or run:

```bash
python -m synthetic_niah_v4.run_v4 --preset debug --stage all
python -m synthetic_niah_v4.run_v4 --preset main --stage all --device cuda --seeds 1234
```

v4 deliberately returns to the v2 setting: two separately trained random-init GPT-2 LMs with learned absolute positional embeddings, fixed prompt length, count range `1..10`, and no loss-mask ablation. Its purpose is mechanistic: cache hidden states, fit count directions, compare direction extraction methods, and test causal steering/patching.

Manual v1 run:

```bash
python scripts/run_v1_niah_like.py \
  --data_root data/trace_count_v1_seed0 \
  --out_root runs/trace_count_v1_seed0 \
  --model_config configs/model/small_main.yaml \
  --max_steps 10000 \
  --batch_size 128 \
  --skip_completed
```

The v2 notebook is self-contained and produces the checkpoints that v3 analyzes. The v3 notebook is also self-contained, but it expects to find a v2 run under `runs/v2_marker_trace_seed1234_main` or `colab_results/v2_marker_trace_*_seed*/run`; set `V2_RUN_DIR_OVERRIDE` in the notebook if your v2 artifacts live elsewhere.

For the v5 mixed thinking-toggle experiment, use `notebooks/Trace_Count_v5_Colab.ipynb` or run:

```bash
python -m synthetic_niah_v5.run_v5 --preset debug --stage all
python -m synthetic_niah_v5.run_v5 --preset main --stage all --device cuda
```

v5 trains one shared v2-style GPT-2 LM with learned absolute positional embeddings and an explicit soft switch. Its two formats are `<BOS> <THINK_ON> prompt <Think/> trace </Think> <Cn> <EOS>` and `<BOS> <THINK_OFF> prompt <Think/> </Think> <Cn> <EOS>`. The mode token is visible before the prompt, and the non-thinking `</Think>` is supervised, so the model must learn to close the block itself. Legacy v5 checkpoints without the two mode tokens must be retrained.

For causal analysis of how this model represents and uses count, run `notebooks/Trace_Count_v5_4_Count_State_Causal_Colab.ipynb`. It restores a completed indexed-trace v5 checkpoint and does not retrain. The notebook separates four claims that ordinary probes conflate: held-out readability, a shared global `+1` direction, class-specific centroid transport, and sufficiency of the complete residual state. It also uses position-matched needle-deletion patching to identify which head groups write an answer-relevant count state in THINK_OFF and after a trace-shaped THINK_ON computation. The full protocol is in `docs/pipelines/pipeline_v5_4_count_state_causal.md`.

For the v6 separator-trace experiment, use `notebooks/Trace_Count_v6_Colab.ipynb` or run:

```bash
python -m synthetic_counting_v6.run_v6_experiment --preset debug --stage all
python -m synthetic_counting_v6.run_v6_experiment --preset main --stage all --device cuda --skip-completed
```

v6 is directly comparable to v2, except the thinking trace removes numeric index tokens. Instead of `<Think/> <1> <A> <2> <B> ...`, it uses `<Think/> <Sep> <A> <Sep> <B> ...`. The final answer still uses numeric tokens `<1>` through `<10>`. This tests whether the targeted-retrieval/counting behavior survives without explicit prefix-count leakage in the trace.

For v10, use `notebooks/Trace_Count_v10_Colab.ipynb` or run:

```bash
python -m synthetic_counting_v10.run_v10 \
  --preset main --stage all --device cuda \
  --out-root runs/synthetic_counting_v10 \
  --run-name v10_main_seed1234 \
  --skip-completed
```

v10 returns to two separately trained v2-style Transformers, extends the balanced
needle range to `1..30`, records full learning dynamics and resumable checkpoints,
then runs cumulative top-1 through top-16 attention-head ablations, multi-offset
head-output patching, per-Layer 2/3/6-PC count manifolds, geometry steering, final-state
`m -> n` transplants, and within-CoT progress/early-stop transplants. The full protocol
and mathematical definitions are in
`docs/pipelines/pipeline_v10_two_model_count30_causal.md`.

For the controlled small-architecture follow-ups, use:

```bash
python -m synthetic_counting_v11.run_v11 --preset main --stage all --device cuda
python -m synthetic_counting_v12.run_v12 --preset main --stage all --device cuda
python -m synthetic_counting_v13.run_v13 --preset main --stage all --device cuda
python -m synthetic_counting_v14.run_v14 --preset main --stage all --device cuda
```

The matching Colab entry points are `notebooks/Trace_Count_v11_Colab.ipynb` through
`Trace_Count_v14_Colab.ipynb`. All four versions are hard-locked to **4 layers, 4 heads,
`d_model=64`, and MLP size 256**. v12-v14 reuse the v10-style implementation path but
never the v10 `d_model=256` model. v11 compares learned APE, RoPE, and learned relative
position bias (RPE); v12 uses APE with length 512 and counts 1-50; v13 uses a persisted
finite training set; and v14 replaces i.i.d. noise with contiguous Shakespeare character
windows. Each version trains separate non-thinking and thinking models and reports only
learning dynamics, descriptive attention, and descriptive hidden-state geometry. See
`docs/pipelines/pipeline_v11_v14_small_architecture.md` for the controlled protocol.

Regenerate all four notebooks after editing their shared builder:

```bash
python scripts/build_v11_v14_notebooks.py
```

For the v10-capacity v15-v17 experiments, use:

```bash
python -m synthetic_counting_v15.run_v15 --preset main --stage all --device cuda
python -m synthetic_counting_v16.run_v16 --preset main --stage all --device cuda
python -m synthetic_counting_v17.run_v17 --preset main --stage all --device cuda
```

The matching Colab notebooks are `Trace_Count_v15_Colab.ipynb` through
`Trace_Count_v17_Colab.ipynb`. v15 and v16 each train four independent models
(`RoPE/RPE x non-thinking/thinking`); v17 trains two independent RoPE models.
All use 4 layers, 4 heads, `d_model=256`, MLP size 1024, prompt length 256,
counts 1-30, and shared trace/final number tokens. **v15 and v16 use prompt +
completion all-sequence next-token cross-entropy:** every non-padding target
after `<BOS>` contributes, including the task prefix, prompt/haystack, trace,
final answer, and `<EOS>`. **v17 remains v10-style completion-only**, so its
prompt/haystack tokens are causal context but are ignored by the loss. v15
inserts markers into Tiny Shakespeare windows;
v16 counts explicitly named native Shakespeare characters without inserted
markers; v17 retains v10 synthetic data, switches to RoPE with base 10000, and uses a decreasing power-law or
exponential count distribution, so examples with more needles are rarer, while
validation remains balanced. See
`docs/pipelines/pipeline_v15_v17_completion.md` for exact formulas and formats.

Regenerate these three notebooks with:

```bash
python scripts/build_v15_v17_notebooks.py
```

For v16.2, use `notebooks/Trace_Count_v16_2_Colab.ipynb` or run:

```bash
python -m synthetic_counting_v16_2.run_v16_2 \
  --preset main --stage all --device cuda \
  --min-candidate-windows 128 \
  --out-root runs/synthetic_counting_v16_2 \
  --run-name v16_2_main_split_windows_seed1234 \
  --skip-completed
```

v16.2 keeps the four v16 models (`RoPE/RPE x non-thinking/thinking`) and the
native Shakespeare target-character task, but changes the sampling protocol.
It splits the raw corpus into 80%/10%/10% train/validation/test regions before
constructing windows, rejects `(target character, exact count)` strata with
fewer than 128 candidates, and consumes eligible training windows without
replacement. Counts are not rebalanced: the corpus determines the realized
training distribution. Checkpoints include sampler cursors and RNG state, so a
resumed run consumes the exact next windows. See
`docs/pipelines/pipeline_v16_2_split_window_sampling.md`.

For v18, use `notebooks/Trace_Count_v18_Colab.ipynb` or run a staged suite:

```bash
python -m synthetic_counting_v18.run_v18 \
  --preset main --suite power --stage all --device cuda \
  --out-root runs/synthetic_counting_v18 \
  --run-name v18_main_power_seed1234 \
  --skip-completed
```

v18 uses an independent implementation while retaining the v10-style analysis
surface. It trains four length-1024 models: uniform versus
`p(c) proportional to c^-1.5`, crossed with direct versus CoT output. Prompts
use 256 noise token types, 10 marker types, and counts 1-128. CoT emits explicit
`I_k, M_type(k)` pairs followed by `END` and a separate final `C_n` token, which
keeps targeted retrieval and scalar count readout independently measurable.
The pipeline records learning dynamics by count band, broad/targeted/successor/
trace-readout attention signatures, count/progress probes, and PC1-PC6 centroid
geometry. `--suite all` launches four separate 10,000-step trainings. See
`docs/pipelines/pipeline_v18_reference_reproduction.md`.

Regenerate both notebooks with:

```bash
python scripts/build_v16_2_v18_notebooks.py
```

For v19, use `notebooks/Trace_Count_v19_Colab.ipynb` or run:

```bash
python -m synthetic_counting_v19.run_v19 \
  --preset main --suite all --stage all --device cuda \
  --out-root runs/synthetic_counting_v19 \
  --run-name v19_main_all_seed1234 \
  --skip-completed
```

v19 holds the v18 data distributions, model, optimizer, and four-run design
fixed, but replaces the 128 index tokens and 128 count tokens with ten shared
decimal digit tokens. Role tokens keep ordinal progress and final count
semantically distinct: CoT uses `<Index> 1 2 M_k`, while the answer uses
`<Count> 1 2 <NumEnd>`. This makes multi-digit composition part of the task and
requires digit-aware generation, parsing, attention anchors, and state probes.
See `docs/pipelines/pipeline_v19_digit_tokenization.md`.

Regenerate the v19 notebook with:

```bash
python scripts/build_v19_notebook.py
```

To rebuild the v3 notebook after editing its generator:

```bash
python scripts/build_v3_notebook.py
```

## Full v0 Sweep

```bash
python scripts/run_pipeline.py --config configs/experiment/v0_seed0.yaml --stage data
python scripts/run_loss_mask_sweep.py \
  --data_dir data/trace_count_v0_seed0 \
  --model_config configs/model/small_main.yaml \
  --model_name small_main \
  --out_root runs/trace_count_v0_seed0 \
  --seeds 0 \
  --max_steps 10000 \
  --batch_size 128 \
  --progress_every 100 \
  --eval_limit 1024 \
  --skip_completed
python -m trace_counting.summarize \
  --runs_dir runs/trace_count_v0_seed0 \
  --out_csv runs/trace_count_v0_seed0/summary.csv \
  --print_markdown
```

This is the notebook default. It trains 7 runs: 7 loss-mask regimes for seed 0. For an exhaustive paper-quality run, use `--max_steps 50000` and remove `--eval_limit 1024`.

To monitor a run from a second VSCode/Colab terminal:

```bash
python scripts/monitor_runs.py \
  --runs_dir runs/trace_count_v0_seed0 \
  --max_steps 10000 \
  --interval 60
```

The training command also prints a plain progress line every 100 optimizer steps, which is easier to see in Colab/VSCode notebooks than a live `tqdm` bar. The monitor reports GPU utilization, latest step/loss/validation metrics, checkpoint status, and how recently each `train_log.jsonl` changed.

## Manual Commands

Generate data:

```bash
python -m trace_counting.generate_data \
  --out_dir data/trace_count_v0_seed0 \
  --max_count 64 \
  --noise_vocab_size 64 \
  --train_lengths 32,64,128 \
  --train_counts 0:24 \
  --examples_per_pair_train 512 \
  --examples_per_pair_val 128 \
  --seeds 0
```

Train one run:

```bash
python -m trace_counting.train \
  --data_dir data/trace_count_v0_seed0 \
  --model_config configs/model/small_main.yaml \
  --loss_mask completion_final_weighted \
  --final_weight 10 \
  --seed 0 \
  --out_dir runs/trace_count_v0_seed0/small_main/completion_final_weighted_fw10_seed0
```

Evaluate:

```bash
python -m trace_counting.eval \
  --checkpoint runs/trace_count_v0_seed0/small_main/completion_final_weighted_fw10_seed0/checkpoints/final \
  --data_dir data/trace_count_v0_seed0 \
  --splits val_id,val_length_ood,val_density_shift_low,val_density_shift_high \
  --out_dir runs/trace_count_v0_seed0/small_main/completion_final_weighted_fw10_seed0/eval
```

Probe hidden states:

```bash
python -m trace_counting.probes \
  --checkpoint runs/trace_count_v0_seed0/small_main/completion_final_weighted_fw10_seed0/checkpoints/final \
  --data_dir data/trace_count_v0_seed0 \
  --split val_id \
  --out_dir runs/trace_count_v0_seed0/small_main/completion_final_weighted_fw10_seed0/probes \
  --anchors ans,think_open,think_close,source,trace_index,trace_marker \
  --layers all
```

Plot and summarize:

```bash
python -m trace_counting.plots --run_dir runs/trace_count_v0_seed0/small_main/completion_final_weighted_fw10_seed0
python -m trace_counting.summarize --runs_dir runs/trace_count_v0_seed0 --out_csv runs/trace_count_v0_seed0/summary.csv
```

For the original three-seed matrix, use `configs/experiment/v0.yaml` and pass `--seeds 0,1,2`.

## Loss Masks

Implemented masks:

- `full_sequence`
- `full_sequence_final_weighted`
- `completion_only`
- `completion_final_weighted`
- `final_count_only`

All regimes use the same `full_tokens`; only `labels` and `loss_weights` change.

## Tests

```bash
pytest
```

The tests cover the generator schema, trace order, `n=0` formatting, exact loss-mask indices, final-count weighting, and autoregressive parsing validity.
