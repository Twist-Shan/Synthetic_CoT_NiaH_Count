# Synthetic counting v47

v47 is the fixed-horizon convergence control for v46. Both Non-thinking and
Thinking are independently reinitialized and receive the same 10,000 optimizer
updates with the same 10,000-step cosine schedule. There is no continuation or
checkpoint selection.

Every other substantive setting is unchanged: counts are exactly 1--10, the
context length is 256, task windows receive a fresh count-preserving random
permutation, the full-support maximum-entropy set x count sampler is used, and
the Thinking target remains
`<Think> (<Sep> marker)*n </Think> <Ans> count` without an explicit index. This
tests the v46 diagnosis that long-count serial traces were still improving when
the 6,000-step schedule reached zero, while preserving an equal training budget
across the two independently trained modes.
