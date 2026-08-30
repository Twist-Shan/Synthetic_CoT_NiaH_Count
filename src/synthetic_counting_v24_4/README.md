# Synthetic counting v24.4

V24.4 is a sampler-only control for v24.3. It keeps the model, data pool,
component-normalized loss, optimizer, seed, count support, trace grammar, and
10,000-step schedule fixed. The only experimental change is a
maximum-entropy distribution over feasible `(needle set, count)` cells with
uniform set and count marginals.
