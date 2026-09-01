# Synthetic counting v32

v32 is the sampler-only control for v31.  Non-thinking and Thinking are still
trained as independent models from identical initialization.  The 256-character
prompt, counts 1--10, 100 marker sets, separator/no-index trace, four-layer
architecture, count-only untied readout, count coefficient 8, optimizer, and
10,000-step schedule are unchanged.

Only the training sampler changes from count-uniform rejection sampling to the
existing maximum-entropy distribution over feasible `(marker set, count)`
cells.  This preserves uniform count and set marginals while suppressing the
empirically measured set-identity shortcut.  Evaluation remains on the same
held-out, count-balanced corpus split.
