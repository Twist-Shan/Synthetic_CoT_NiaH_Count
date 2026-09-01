# Synthetic counting v24.8

v24.8 is a training-only readout-calibration tail initialized from the paired
v24.7 final checkpoints. It does not change the separator/no-index trace,
sampler, transformer, vocabulary, or inference procedure.

The v24.7 answer-query residual is already perfectly count-decodable by held-out
NCC/logistic probes, while the native LM head maps several odd counts to adjacent
even counts. v24.8 therefore freezes the complete transformer, final layer norm,
input embeddings, and every non-number unembedding row. On balanced training
examples it optimizes only the ten existing atomic-number rows of the native LM
head using cross-entropy at the ordinary `<Ans>` query. Validation raw
autoregressive accuracy selects the checkpoint; the test split is evaluated
exactly once after selection. No auxiliary classifier or inference-time decoder
is added.
