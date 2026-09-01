# Synthetic counting v50

v50 is a one-variable convergence control on top of v49. Both modes retain
independent initialization and training, the compact 4-layer / 4-head / width
256 model, counts 1--10, full-support maximum-entropy set x count sampling,
freshly permuted 256-character task contexts, grammar-balanced component loss,
and the unchanged no-index trace

`<Think> (<Sep> marker)*n </Think> <Ans> count`.

The only optimization change is the pre-registered endpoint and cosine decay
horizon: 10,000 steps in v49 becomes 20,000 steps in v50. The extra phase-cloud
entries only record the extended trajectory and do not alter optimization.
This tests whether v49 was horizon-limited: its marker-identity objective was
still decreasing when the learning rate reached zero at step 10,000.
