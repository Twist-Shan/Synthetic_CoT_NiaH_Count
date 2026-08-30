# Synthetic counting v31

v31 is a shared-model control built from v29. It keeps the 256-character
input, count support 1--10, uniform semantic-count sampler, 100 marker sets,
four-layer/four-head model, no-index separator trace, component-normalized
loss, count coefficient 4, and partial count-only output untying unchanged.

The sole conceptual change is mode coupling. One model is trained on paired
views of every semantic example: the unchanged Non-thinking answer-only view
and the unchanged Thinking separator-trace view. Each view has 128 rows per
step; its complete v29 objective is computed separately and the two losses are
averaged equally. Thus trace length cannot change the relative mode weight.

There is no auxiliary loss, post-hoc decoder, frozen/calibration stage,
test-time training, trace rewrite, extra layer, or inference-time rule. Both
modes are evaluated from the same checkpoint, aligning the synthetic
comparison with two generation modes of one large language model.
