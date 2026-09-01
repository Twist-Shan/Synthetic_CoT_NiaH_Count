# Synthetic counting v52

v52 is a one-variable marker-identity loss control on top of v51. Both modes
remain independently initialized 4-layer, 4-head, 256-dimensional RoPE models
trained for 10,000 optimizer steps on the same balanced count-1-to-10 task.
The Thinking serialization remains unchanged and contains no explicit index:

`<Think> (<Sep> marker)*n </Think> <Ans> count`

Only `task_output_trace_weight` changes, from 8 to 16. The count, marker, and
grammar component weights are therefore 8/16/16. This tests whether v51's gap
between largely correct generated trace lengths and less accurate marker
identity sequences is an objective-allocation problem.
