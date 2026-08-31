# Synthetic counting v51

v51 is a one-variable grammar-loss control on top of v49. Both modes retain
independent initialization and training, the compact 4-layer / 4-head / width
256 model, the fixed 10,000-step cosine schedule, counts 1--10, full-support
maximum-entropy set x count sampling, freshly permuted 256-character task
contexts, and the unchanged no-index trace

`<Think> (<Sep> marker)*n </Think> <Ans> count`.

Only `task_output_structure_weight` changes, from 8 to 16. The structure
component contains the Thinking continue/stop grammar (`<Sep>` and
`</Think>`) plus output tags. This directly targets v49's nearly symmetric
one-marker-short and one-marker-long errors. Count and marker-identity weights,
all serialized targets, the optimizer, and inference are unchanged.
