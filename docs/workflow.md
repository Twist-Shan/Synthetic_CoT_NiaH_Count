# Trace-Counting Workflow

This repo follows a task-first research pipeline:

```text
generate data -> train loss-mask ablations -> evaluate -> extract/probe hidden states -> plot and summarize
```

The canonical task is defined in `synthetic_experiments_pipeline.md`. Every model sees the same tokenized examples; only the loss mask changes.

## Quick Smoke Run

```bash
pip install -e .
python scripts/run_pipeline.py --config configs/experiment/debug.yaml --stage all
```

Outputs are written to:

```text
data/trace_count_v0_debug/
runs/trace_count_v0_debug/tiny_debug/completion_final_weighted_fw10_seed0/
```

## Full v0 Run

```bash
python scripts/run_pipeline.py --config configs/experiment/v0.yaml --stage data
python scripts/run_pipeline.py --config configs/experiment/v0.yaml --stage train
python scripts/run_pipeline.py --config configs/experiment/v0.yaml --stage eval
python scripts/run_pipeline.py --config configs/experiment/v0.yaml --stage probe
python scripts/run_pipeline.py --config configs/experiment/v0.yaml --stage plots
```

Use `--override` style flags on the direct module CLIs when running sweeps over `loss_mask`, `final_weight`, model size, and seed.

## Main Entry Points

- `python -m trace_counting.generate_data`
- `python -m trace_counting.train`
- `python -m trace_counting.eval`
- `python -m trace_counting.probes`
- `python -m trace_counting.plots`
- `python -m trace_counting.summarize`

## Expected Run Directory

```text
run_dir/
  config.yaml
  vocab.json
  train_log.jsonl
  checkpoints/
    step_00001000/
    final/
  eval/
    val_id_metrics.json
    predictions_val_id.jsonl
  probes/
    probe_summary.csv
  plots/
    training_curves.png
    tf_accuracy_by_count.png
```
