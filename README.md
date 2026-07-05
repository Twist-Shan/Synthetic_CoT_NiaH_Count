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

If you use Colab, clone the GitHub repo first, then run the install cells in `notebooks/Trace_Count_v0_Colab.ipynb`.

## One-Command Debug Run

```bash
python scripts/run_pipeline.py --config configs/experiment/debug.yaml --stage all
```

This generates a tiny balanced dataset, trains `tiny_debug` for 100 steps, evaluates all validation splits, runs a small hidden-state probe, and writes plots.

## Manual Commands

Generate data:

```bash
python -m trace_counting.generate_data \
  --out_dir data/trace_count_v0 \
  --max_count 64 \
  --noise_vocab_size 64 \
  --train_lengths 32,64,128 \
  --train_counts 0:24 \
  --examples_per_pair_train 512 \
  --examples_per_pair_val 128 \
  --seeds 0,1,2
```

Train one run:

```bash
python -m trace_counting.train \
  --data_dir data/trace_count_v0 \
  --model_config configs/model/small_main.yaml \
  --loss_mask completion_final_weighted \
  --final_weight 10 \
  --seed 0 \
  --out_dir runs/trace_count_v0/small_main/completion_final_weighted_fw10_seed0
```

Evaluate:

```bash
python -m trace_counting.eval \
  --checkpoint runs/trace_count_v0/small_main/completion_final_weighted_fw10_seed0/checkpoints/final \
  --data_dir data/trace_count_v0 \
  --splits val_id,val_length_ood,val_density_shift_low,val_density_shift_high \
  --out_dir runs/trace_count_v0/small_main/completion_final_weighted_fw10_seed0/eval
```

Probe hidden states:

```bash
python -m trace_counting.probes \
  --checkpoint runs/trace_count_v0/small_main/completion_final_weighted_fw10_seed0/checkpoints/final \
  --data_dir data/trace_count_v0 \
  --split val_id \
  --out_dir runs/trace_count_v0/small_main/completion_final_weighted_fw10_seed0/probes \
  --anchors ans,think_open,think_close,source,trace_index,trace_marker \
  --layers all
```

Plot and summarize:

```bash
python -m trace_counting.plots --run_dir runs/trace_count_v0/small_main/completion_final_weighted_fw10_seed0
python -m trace_counting.summarize --runs_dir runs/trace_count_v0 --out_csv runs/trace_count_v0/summary.csv
```

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
