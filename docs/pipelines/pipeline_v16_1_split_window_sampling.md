# v16.1: split-local indexed-window training

## Question

v16 counted native target characters in Tiny Shakespeare, but online random
window sampling could revisit the same text repeatedly and could make it hard to
audit train/validation overlap. v16.1 changes only the data access protocol.

## Protocol

1. Read the standard Tiny Shakespeare corpus once.
2. Split raw character positions into 80% train, 10% validation, and 10% test.
3. Build sliding-window indices separately inside each split. No indexed window
   may cross a split boundary.
4. Index a training item by `(target character, exact count, window start)`.
5. Keep only strata with at least `min_candidate_windows` candidates. The main
   preset uses 128 and the debug preset uses 2.
6. During a sampler epoch, draw from all remaining indexed items with
   probability proportional to the remaining stratum size. This is equivalent
   to uniform sampling over eligible indexed items, without replacement.
7. When every eligible item has been consumed, reshuffle deterministically and
   begin the next sampler epoch.

The count classes are intentionally not rebalanced. Their empirical frequency
comes from the corpus, target-letter set, and window length. The run writes
`window_index_summary.csv` and `training_count_distribution.csv` so the realized
imbalance can be audited.

## Reproducibility

Checkpoint state includes the sampler epoch, per-stratum cursors, and selection
RNG state. Resuming therefore reproduces the exact next training windows rather
than merely restarting from the same model weights.

The model, objective, modes, and character-counting task remain aligned with
v16: RoPE/RPE x non-thinking/thinking, 4 layers, 4 heads, width 256, and
prompt-plus-completion causal next-token loss.

