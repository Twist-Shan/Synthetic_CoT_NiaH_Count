# Synthetic counting v41

v41 is a width/head-capacity screen on top of v40.  It preserves two
independently trained mode-specific models, 256-character prompts, count
support 1--5, the exact 100 three-character marker sets, maximum-entropy
set/count sampling, the separator/no-index trace, partial atomic-count readout
untying, equal component-normalized losses, and the 6,000-update schedule.

The only compound architecture control keeps serial depth at four layers and
attention head dimension at 64 while scaling residual width 256 -> 384, heads
4 -> 6, and the 4x MLP 1024 -> 1536.  It tests whether v40's low free-running
stability is a parallel retrieval-capacity bottleneck.  It does not add serial
depth, alter the trace or targets, share checkpoints between modes, select an
early checkpoint, or perform test-time adaptation.
