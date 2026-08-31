# Synthetic counting v40

v40 is the count-range difficulty control for v35. It preserves two
independently trained mode-specific models, 256-character prompts, the exact
100 three-character marker sets, maximum-entropy set/count sampling, the
separator/no-index trace, partial atomic-count readout untying, equal
component-normalized count/trace/structure losses, and the 6,000-update v35
schedule.

The only substantive task change is `count_max_threshold: 10 -> 5`. The trace
format and targets are otherwise unchanged: Thinking emits the same `<Sep>`
pair once per counted marker and then the atomic answer. This tests whether
long no-index traces are the dominant current bottleneck while retaining the
same 256-token broad-versus-targeted retrieval problem.
