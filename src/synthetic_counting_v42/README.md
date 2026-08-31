# Synthetic counting v42

v42 is the optimization-horizon control for v41.  Both mode-specific models
are independently reinitialized and trained from scratch.  The 256-character
prompt, count support 1--5, exact marker sets, maximum-entropy sampler,
separator/no-index trace, 4-layer/6-head/width-384 architecture, loss, warmup,
peak learning rate, clipping, and inference are unchanged.

The only substantive change is 6,000 -> 8,000 optimizer updates.  Because the
existing cosine schedule uses `train_steps` when no explicit decay horizon is
provided, the cosine horizon is correspondingly 8,000 steps.  This tests the
predeclared hypothesis that v41 stopped before its late Thinking transition
fully converged; it is not a continuation, early-checkpoint selection, shared
model, calibration, or test-time adaptation.
