# Synthetic counting v30

v30 is the depth-only capacity control for v29. It keeps the 256-character
input, count support 1--10, uniform semantic-count sampler, 100 marker sets,
separator/no-index trace, four attention heads, width 256, partial count-only
output untying, component-normalized count coefficient 4, optimizer, and
10,000-step schedule unchanged.

The sole change is `n_layer: 4 -> 6`. Both Thinking and Non-thinking are
trained from scratch. There is no auxiliary loss, post-hoc decoder, frozen
phase, test-time update, or trace-format change.
