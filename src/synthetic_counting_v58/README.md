# Synthetic counting v58

v58 is the second point in the clean parallel-capacity sweep from v51. It
preserves two independently initialized modes, batch 128, 256-character
permuted contexts, counts 1--10, maximum-entropy full-support sampling, the
unchanged separator/no-index trace, pure teacher forcing, the 8/8/16 component
loss, four transformer layers, and the fixed 10,000-step endpoint.

The only linked architecture factor relative to v57 changes from 6 heads / 384
residual dimensions / 1536 MLP dimensions to 8 heads / 512 residual dimensions
/ 2048 MLP dimensions. Head dimension remains 64. This tests whether a larger
parallel retrieval bank improves trace completeness and count-uniform behavior
without adding serial depth or changing supervision.

Before optimization, the behavior gate was revised to match the comparative
claim: Thinking accuracy at least 0.75, a Thinking-minus-Non-thinking gap of at
least 0.30, minimum per-count Thinking accuracy at least 0.70, and per-count
max-minus-min spread at most 0.20. Exact trace identity remains diagnostic and
is evaluated mechanistically rather than used to discard an otherwise useful,
uniform paired model.
