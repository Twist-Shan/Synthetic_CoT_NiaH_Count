# Synthetic counting v35

v35 is the one-scalar correction to v34's diagnosed component imbalance.
It retains two independent models, pure gold-prefix teacher forcing, the
maximum-entropy set/count sampler, 256-character prompts, counts 1--10, 100
marker sets, the separator/no-index trace, architecture, readout, optimizer,
and fixed 6,000-step budget.

The only change is `task_output_structure_weight: 0.1 -> 8`.  Together with
v34's count and trace coefficients of 8, component normalization now assigns
one third of the aggregate task objective to each of count, trace, and
structure.  This directly supervises continue-vs-close and answer-boundary
decisions without changing trace content or inference.
