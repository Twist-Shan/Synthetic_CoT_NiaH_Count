# Synthetic counting v24.3

V24.3 is a loss-only control for v24.2. It keeps the paired RoPE models,
separator/no-index trace, count support 1--10, 100-set pool, uniform-count
sampler, seed, optimizer, and 10,000-step schedule unchanged.

Steps 1--1,500 retain the original all-sequence token-weighted mean. Starting
at step 1,501, task-output loss is

```
L = mean(final-count CE) + mean(trace CE) + 0.1 * mean(structure CE)
```

for Thinking, and omits the absent trace term for Non-thinking. Each region is
first averaged within an example and then across the batch. Thus the final
count has coefficient 1 in both modes and is no longer diluted by trace length.

The training sampler is deliberately still `uniform`: v24.3 does not test set
balancing. That intervention belongs in the subsequent sampler control.
