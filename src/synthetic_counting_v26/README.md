# Synthetic counting v26

V26 returns to the 256-character, 100-set, count-balanced v24.3 regime.  It
keeps the paired 4-layer/4-head transformer, separator/no-index trace,
component-normalized task-output loss, optimizer, seed, and 10,000-step
schedule.  Its only training change from v24.3 is an untied native LM head.
The head is copied from the input embedding at step zero, so logits initially
match exactly; subsequent atomic-number output gradients cannot distort input
number embeddings.

No contrastive or probe loss is used.  Count NCC is therefore evaluated as an
emergent representation, while the paired Non-thinking model remains a fully
trained control on exactly the same examples.
