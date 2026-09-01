# Synthetic counting v45

v45 is a parameter-budget geometry screen on top of v44. Both modes remain
independently initialized and trained. Counts 1--10, 256-character prompts,
the v35 marker sets, maximum-entropy set/count probabilities, full legal-window
support, separator/no-index traces, objective, optimizer, 8,000-step endpoint,
and inference are unchanged.

The sole conceptual change reallocates capacity from parallel width to serial
depth: `4 layers / 6 heads / width 384 / MLP 1536` becomes
`6 layers / 5 heads / width 320 / MLP 1280`. Both use 64 dimensions per head
and have nearly matched parameter counts. This tests whether targeted serial
retrieval becomes reliable without further widening the direct broad-retrieval
path available to Non-thinking.
