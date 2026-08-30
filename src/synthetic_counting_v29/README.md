# Synthetic counting v29

v29 is a single-scalar correction to v28.  It keeps the 256-character input,
count support 1--10, uniform semantic-count sampler, 100 marker sets,
separator/no-index trace, paired model architecture, partial count-only output
untying, optimizer, and 10,000-step schedule unchanged.

During the component-normalized task-output phase, only
`task_output_count_weight` changes from 1 to 4.  The trace coefficient remains
1 and the structure coefficient remains 0.1.  There is no auxiliary loss,
post-hoc decoder, frozen phase, test-time update, or trace-format change.
