# Synthetic counting v48

v48 is the parallel-capacity control for v47. Both modes remain independently
initialized and trained for the same fixed 10,000 updates. Counts, 256-character
task windows, fresh count-preserving task-context permutations, full-support
maximum-entropy set x count sampling, no-index separator traces, loss,
optimizer, seed, and inference are unchanged.

The only substantive change is the four-layer model's parallel capacity:
`4 heads / width 256 / MLP 1024` becomes
`6 heads / width 384 / MLP 1536`, preserving 64 dimensions per head. This tests
whether v47's residual errors on long marker traces are a targeted-retrieval
capacity bottleneck while retaining a sufficiently large head bank for later
role-specialization and differentiation analysis.
