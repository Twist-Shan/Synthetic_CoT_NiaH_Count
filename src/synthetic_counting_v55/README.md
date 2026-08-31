# Synthetic counting v55

v55 is a one-variable grammar-loss control on top of v51. Both independently
initialized modes retain 256-character permuted contexts, counts 1--10, all
legal starts, maximum-entropy set/count sampling, the unchanged separator/
no-index trace, 4L/4H/256D architecture, pure teacher forcing, and the fixed
10,000-step endpoint.

The only substantive change is `task_output_structure_weight: 16 -> 32`.
Marker identities remain at weight 8 and the final count remains at weight 8.
Because separators are assigned to the structure region, this gives additional
supervision to continue-versus-close and answer-boundary decisions without
changing the serialized trace, gold targets, data, or inference.
