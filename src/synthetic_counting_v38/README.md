# Synthetic counting v38

v38 is the mild exposure-bias control for v35.  It preserves two independent
mode-specific models, the separator/no-index trace, 256-character prompts,
counts 1--10, 100 marker sets, maximum-entropy set/count sampling, partial
atomic-count readout untying, equal component-normalized count/trace/structure
losses, and the fixed 6,000-step schedule.

The only changed scalar is
`task_output_scheduled_sampling_max_probability: 0 -> 0.1`.  During the
task-output phase, Thinking linearly replaces a small fraction of eligible
gold continuation inputs with its own preceding predictions while retaining
the original gold targets.  This changes neither the serialized trace nor
inference.  Non-thinking has no eligible intermediate trace positions, so its
optimization path remains the v35 baseline apart from the version label.
