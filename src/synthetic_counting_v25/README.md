# Synthetic counting v25

V25 is a paired retrieval-pressure setting, not a readout handicap.  Relative
to v24.7, it keeps the model width/depth, count support 1--10, 20-set
maximum-entropy sampler, component-normalized loss, untied LM head,
answer-query contrastive term, atomic answers, and separator/no-index trace.

The substantive change is a 1,024-character Shakespeare context instead of
256 characters.  The position budget grows only as required by the rendered
sequence, and the marker-pool frequency cap changes from `10/256` to `10/1024`
so the expected target count and answer support remain matched.  Batch size 32
keeps GPU memory bounded.  Both modes receive the same examples, optimizer
schedule, seed, and validation/test prompts.

The intended diagnostic is whether serial targeted retrieval preserves
Thinking accuracy under long-context search while Non-thinking's direct broad
aggregation loses adjacent-count resolution.  Any final native-head
calibration is applied symmetrically and freezes the transformer, so it cannot
create a retrieval representation that the backbone did not learn.
