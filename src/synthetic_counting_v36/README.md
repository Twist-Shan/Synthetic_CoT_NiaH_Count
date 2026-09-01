# Synthetic counting v36

v36 is the schedule-only control for v35. It retains two independent models,
the unchanged separator/no-index trace, maximum-entropy set/count sampler,
256-character prompts, counts 1--10, 100 marker sets, equal component-normalized
count/trace/structure coefficients, pure teacher forcing, architecture, readout,
seed, and exactly 6,000 optimizer updates.

The only substantive change is `lr_decay_steps: None -> 10000`. The cosine
schedule therefore matches v32 through step 6,000 instead of reaching zero at
the screening endpoint. This tests whether v35's remaining long-count failures
come from premature annealing without changing the trace, task, inference, or
training budget.
