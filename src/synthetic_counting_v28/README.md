# Synthetic counting v28

V28 is the minimal from-scratch readout-decoupling control for v24.3.  It
retains the exact 256-character input, count-balanced 1--10 task, 100-set pool,
separator/no-index trace, paired examples, four-layer model, optimizer, seed,
10,000-step schedule, and component-normalized objective.

Only the parameterization of the ten atomic count-token output vectors changes.
They are initialized exactly equal to the corresponding input embeddings but
are independent parameters throughout end-to-end training.  Every other output
row remains tied to its input embedding.  The transformer is never frozen, and
there is no conditional ten-way loss, trace-safety auxiliary objective,
post-hoc calibration, or test-time adaptation.

The primary hypothesis is that this minimal decoupling preserves the tied trace
copying circuit while allowing the high-NCC Thinking answer state to learn a
native full-vocabulary count readout.  The paired Non-thinking model receives
the identical readout capacity, data, answer loss, and optimization budget.
