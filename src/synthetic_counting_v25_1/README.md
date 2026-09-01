# Synthetic counting v25.1

V25.1 applies the same validation-selected, training-only native LM-head
calibration used by v24.8 to the paired v25 long-context checkpoints.  The
transformer, final norm, input embeddings, trace, data, and inference remain
unchanged.  Both modes receive the same candidate schedule; validation can
retain step zero when a model is already optimal.

This stage removes atomic-token row alignment as a confound.  It cannot create
long-context count information missing from a frozen answer-query residual, so
a remaining Thinking/Non-thinking accuracy gap is attributable to the learned
retrieval/representation path rather than to an avoidable output-row mapping
failure.
