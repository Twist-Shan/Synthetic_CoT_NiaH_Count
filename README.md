# Synthetic Trace-Enumeration Counting

This repository implements the `Synthetic Trace-Enumeration Counting Pipeline v0`: a small decoder-only transformer is trained from scratch on symbolic sparse-counting examples, while the experiment varies only the loss mask.

The repo is organized for GitHub + Colab:

```text
synthetic_experiments_pipeline.md  Experiment spec
configs/                           Model and experiment YAMLs
notebooks/                         Colab-friendly result notebook
scripts/                           Pipeline orchestration
src/trace_counting/                Data, training, eval, probes, plots
tests/                             Unit tests for generation, masks, parsing
docs/                              Workflow notes
```

## Install

Local or Colab:

```bash
pip install -r requirements.txt
pip install -e .
```

If you use Colab, open `notebooks/Trace_Count_v0_Colab.ipynb` and run all cells. The notebook clones/updates the repo when needed, installs dependencies, runs tests, generates the full v0 dataset, trains/evaluates every loss-mask regime, summarizes results, runs probes, displays plots, saves outputs to Google Drive, and prepares an optional GitHub result upload.

For the more NiaH-like v1 experiment, open `notebooks/Trace_Count_v1_Colab.ipynb`. It trains two all-token next-token-prediction models on shorter sparse-counting data: a `think_trace` model with explicit thinking/count trace tokens and an `answer_only` model without thinking tokens. The v1 notebook evaluates ID counts `0-5` and count-OOD `5-10`, then runs linear probes, ridge count-direction extraction, answer-state steering, and attention-to-needle analysis.

For the controlled marker-trace v2 experiment, open `notebooks/Trace_Count_v2_Colab.ipynb`. This notebook follows `notebooks/pipeline_v2_codex_prompt.md`: fixed prompt length, 64 noise-token types, 10 countable marker-token types, count range `1..10`, and two separately trained random-init decoder-only Transformers (`non_thinking` and `thinking`). It intentionally has no ID/OOD split, no variable sequence length, and no steering. The notebook reports training/eval curves by low/mid/high count bin, exact-count accuracy, hidden-state probes, and attention/retrieval diagnostics.

For the v3 attention-head deep dive, open `notebooks/Trace_Count_v3_Colab.ipynb` after a v2 run has produced `checkpoints/final/thinking`. v3 is now analysis-only: it reloads the v2 GPT-2-style thinking model, decomposes attention for the final trace index token into prompt-needle vs previous-trace-token mass, and runs lightweight targeted-head ablations. It does not train a new model, does not generate HTML, and follows `notebooks/pipeline_v3_codex_prompt.md`.

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

For the v6 separator-trace experiment, use `notebooks/Trace_Count_v6_Colab.ipynb` or run:

```bash
python -m synthetic_counting_v6.run_v6_experiment --preset debug --stage all
python -m synthetic_counting_v6.run_v6_experiment --config synthetic_counting_v6/configs/main.yaml --stage all --device cuda --skip-completed
```

v6 is directly comparable to v2, except the thinking trace removes numeric index tokens. Instead of `<Think/> <1> <A> <2> <B> ...`, it uses `<Think/> <Sep> <A> <Sep> <B> ...`. The final answer still uses numeric tokens `<1>` through `<10>`. This tests whether the targeted-retrieval/counting behavior survives without explicit prefix-count leakage in the trace.

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
