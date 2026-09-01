# Synthetic counting v46

v46 returns to v35's compact four-layer, four-head, width-256 model and its
fixed 6,000-update budget.  Counts remain exactly 1--10; Non-thinking and
Thinking are independently initialized and trained; the Thinking target stays
`<Think> (<Sep> marker)*n </Think> <Ans> count` with no explicit index.

The v43 full-support maximum-entropy set x count sampler is retained as a data
correctness control.  The substantive task intervention is a fresh random
permutation of every selected counting window.  This preserves all 256 source
characters, all marker multiplicities, the answer, and the ordered marker
trace, while removing contiguous Shakespeare-window and absolute-start
shortcuts.  Raw language-model windows retain natural order.  The purpose is
to make the paired behavioral comparison depend on broad aggregation versus
serial targeted retrieval rather than memorization of local corpus details.
