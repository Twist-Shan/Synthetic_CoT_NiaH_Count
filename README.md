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
  --eval_limit 1024 \
  --skip_completed
python -m trace_counting.summarize \
  --runs_dir runs/trace_count_v0_seed0 \
  --out_csv runs/trace_count_v0_seed0/summary.csv \
  --print_markdown
```

This is the notebook default. It trains 7 runs: 7 loss-mask regimes for seed 0. For an exhaustive paper-quality run, use `--max_steps 50000` and remove `--eval_limit 1024`.

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
