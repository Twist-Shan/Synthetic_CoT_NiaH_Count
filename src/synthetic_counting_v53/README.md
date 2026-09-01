# Synthetic counting v53

v53 is a one-variable serial-depth control on top of v51. Non-thinking and
Thinking remain independently initialized models trained for 10,000 updates
on the same balanced count-1-to-10 task. The Thinking target is unchanged and
contains no explicit index:

`<Think> (<Sep> marker)*n </Think> <Ans> count`

Only `n_layer` changes, from 4 to 6. Residual width remains 256, the head bank
remains four 64-dimensional heads per layer, the MLP remains 1024, and the
count/marker/grammar component weights return to v51's 8/8/16. This tests
whether the remaining long-trace failures are limited by serial computation
rather than objective allocation or parallel retrieval width.
