# Synthetic counting v44

v44 restores the final task support to counts 1--10 on top of v43. It retains
two independently initialized and trained models, 256-character prompts, the
same 100 three-character marker sets, the maximum-entropy set/count sampler,
full support over every legal within-cell corpus window, the separator/no-index
trace, pure gold-prefix teacher forcing, equal component-normalized loss, and
the four-layer/six-head/width-384 model.

Relative to v43, the only configured change is `count_max_threshold: 5 -> 10`.
The screening endpoint remains fixed at 8,000 optimizer updates. The behavioral
gate is evaluated on 50 held-out examples per count and requires Thinking
overall accuracy at least 0.90, minimum per-count accuracy at least 0.80,
Thinking count spread at most 0.20, Thinking trace exact at least 0.90, and a
Thinking-minus-Non-thinking accuracy gap of at least 0.10. Mechanistic analysis
must not be used to rescue a failed behavioral screen.
