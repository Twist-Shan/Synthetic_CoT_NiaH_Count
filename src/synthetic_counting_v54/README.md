# Synthetic counting v54

v54 is a one-variable exposure-bias control on top of v51. Both modes retain
independent initialization and optimization, 256-character permuted contexts,
counts 1--10, maximum-entropy set/count sampling over all legal starts, the
unchanged separator/no-index trace, the 8/8/16 component-normalized objective,
the 4L/4H/256D architecture, and the fixed 10,000-step endpoint.

The only changed scalar is
`task_output_scheduled_sampling_max_probability: 0 -> 0.1`. After the initial
1,500-step all-sequence phase, Thinking linearly replaces a mild fraction of
eligible gold continuation inputs with the model's own preceding predictions,
while retaining the original gold targets. Non-thinking has no intermediate
trace positions, so the operation is a no-op for that mode. The serialized
trace, supervision targets, final evaluation, and inference are unchanged.
