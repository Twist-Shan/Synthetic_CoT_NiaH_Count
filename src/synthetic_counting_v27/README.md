# Synthetic counting v27

v27 is a post-training readout test on the paired v24.3 L256 checkpoints.  It
does not change the separator trace, data, backbone, inference graph, or tied
embedding/unembedding parameterization.  It freezes every parameter except the
ten existing atomic count-token rows and optimizes final-answer cross entropy.

The experiment uses the same validation-selected learning-rate schedule for
Thinking and Non-thinking.  The test split is evaluated only after the schedule
has been fixed.  This directly tests whether v24.3 Thinking's stronger
answer-query count representation can be converted into uniformly better raw
generation without giving it an auxiliary decoder.

Run with:

```bash
python -m synthetic_counting_v27.cli \
  --source-run /path/to/v24.3_componentloss_count1-10_seed1234 \
  --output-dir /path/to/v27_tied_number_row_calibration_L256_pool100_seed2478 \
  --device cuda
```
