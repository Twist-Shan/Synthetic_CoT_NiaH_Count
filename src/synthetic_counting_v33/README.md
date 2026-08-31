# Synthetic counting v33

v33 retains v32's 256-character task, count range 1--10, 100 marker sets,
maximum-entropy set/count sampler, separator/no-index trace, four-layer model,
partial count-only output untying, and component-normalized count coefficient 8.
Non-thinking and Thinking are still two independently trained models with
identical initial parameters.

The new optimization mechanism is a linear scheduled roll-in during the
task-output phase.  For Thinking only (Non-thinking has no generated prefix
tokens before its answer target), selected continuation inputs are replaced by
the model's own previous-token predictions, increasing from probability zero
at step 1,500 to 0.5 at step 6,000.  Gold targets, trace text, parameters, and
inference are unchanged.  The fixed 6,000-step budget is declared before the
v33 run from the v32 learning curve, where Non-thinking has not yet saturated;
this tests whether the trace gives a sample-efficiency advantage once exposure
bias is controlled.
