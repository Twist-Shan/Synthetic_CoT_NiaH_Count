# Synthetic counting v34

v34 is the loss-only alternative to v33's failed token-level scheduled
roll-in.  It restores pure gold-prefix teacher forcing and retains v32's
maximum-entropy set/count data, 256-character prompt, counts 1--10, 100 marker
sets, separator/no-index trace, four-layer model, partial count-only output
untying, and two independent mode-specific models.

During the task-output phase, the already component-normalized trace region
receives coefficient 8 instead of 1, matching the final-count coefficient.
All trace tokens, labels, parameters, and inference rules are unchanged.  The
same fixed 6,000-step efficiency budget used in v33 is retained, so v34 tests
whether stronger exact-trace supervision—not corrupted prefixes—can produce a
stable Thinking advantage before direct Non-thinking counting saturates.
