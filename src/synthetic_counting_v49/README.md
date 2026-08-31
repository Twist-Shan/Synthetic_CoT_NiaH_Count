# Synthetic counting v49

v49 is a one-variable training-loss control on top of v47.  Both modes retain
independent initialization and training, the compact 4-layer / 4-head / width
256 model, 10,000-step cosine horizon, counts 1--10, full-support maximum-
entropy set x count sampling, freshly permuted 256-character task contexts,
and the unchanged no-index trace

`<Think> (<Sep> marker)*n </Think> <Ans> count`.

Only the semantic partition inside the component-normalized Thinking loss is
changed.  Repeated `<Sep>` tokens are grammar decisions to continue, so v49
places them in the same structure component as the single `</Think>` stop
decision.  The trace component therefore measures and supervises retrieved
marker identities only.  This removes the count-dependent continue/stop
weight imbalance diagnosed from v47's predominantly one-step-short errors.
For Non-thinking there are no trace delimiters, so the objective is numerically
unchanged.
