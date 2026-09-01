# Synthetic counting v56

v56 is a one-variable optimization-variance control on top of v51. It keeps
the two independent mode-specific models, 256-character permuted contexts,
counts 1--10, maximum-entropy full-support sampling, the unchanged separator/
no-index trace, pure teacher forcing, 8/8/16 component loss, 4L/4H/256D model,
learning-rate schedule, and fixed 10,000 optimizer updates.

The only substantive change is `batch_size: 128 -> 256`. This doubles the
number of independently sampled training examples per update while preserving
the target distribution. It is intended to reduce gradient variance across
set/count cells and long traces. Because optimizer steps are held fixed, v56
uses twice as many training examples as v51 and is not presented as an
equal-compute comparison.
