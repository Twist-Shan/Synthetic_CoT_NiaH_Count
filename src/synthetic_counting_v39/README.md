# Synthetic counting v39

v39 is the loss-schedule control for v35. It preserves two independently
trained mode-specific models, the separator/no-index trace, 256-character
prompts, counts 1--10, 100 marker sets, maximum-entropy set/count sampling,
partial atomic-count readout untying, equal component-normalized
count/trace/structure losses, and the fixed 6,000-update cosine schedule.

The only substantive change is `max_steps_for_language_pred: 1500 -> 0`.
Thus every update optimizes only the task output, beginning inclusively at
`<Ans>` for Non-thinking and `<Think>` for Thinking. Gold targets and prefixes
are unchanged, scheduled sampling is disabled, and inference is unchanged.
