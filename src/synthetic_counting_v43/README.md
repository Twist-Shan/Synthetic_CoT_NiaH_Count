# Synthetic counting v43

v43 is the full-support sampler control for v42. Non-thinking and Thinking
remain two independently initialized and trained models. The 256-character
prompt, count support 1--5, exact marker sets, maximum-entropy set/count cell
probabilities, separator/no-index trace, 4-layer/6-head/width-384 architecture,
loss, optimizer, 8,000-step schedule, seed, and inference are unchanged.

The only substantive change is within-cell sampling. Versions through v42
deterministically retained at most 8,192 evenly spaced legal corpus starts in
each feasible set x count cell, while evaluation sampled from the full corpus
region. v43 uniformly samples from every legal start in the selected cell and
records both full and retained support in the sampler audit. This removes a
train/evaluation support mismatch without changing targets, traces, marginal
count balance, model capacity, loss weighting, or test-time behavior.
