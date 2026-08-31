# Synthetic counting v37

v37 is the low-learning-rate consolidation control for v35. It retains two
independent models, the unchanged separator/no-index trace, maximum-entropy
set/count sampler, 256-character prompts, counts 1--10, 100 marker sets, equal
component-normalized count/trace/structure coefficients, pure teacher forcing,
architecture, readout, and seed.

The schedule cosines toward `min_lr=1e-5` through step 6,000, then holds that
minimum through a predeclared 8,000-step endpoint. This keeps the stable v35
trajectory nearly unchanged while adding a conservative trace-length
consolidation tail. It does not select an early checkpoint, alter trace
content, add an auxiliary objective, or change inference.
