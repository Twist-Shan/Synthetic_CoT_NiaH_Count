# v24.2: count-balanced control for v24

v24.2 changes exactly one experimental factor relative to v24: training counts
are sampled uniformly from 1 through 10 instead of following the accepted
natural Shakespeare-window distribution. Both `rope/nonthinking` and
`rope/thinking` are retrained on the same paired stream.

Everything else remains fixed: the 256-character Shakespeare context,
three-character query, 100-set pool with the `10/256` frequency cap, RoPE
architecture, separator/no-index trace, atomic answer tokens, unit final/trace
loss weights, seed 1234, 10,000-step schedule, optimizer, evaluation manifests,
and checkpoint cadence.

The primary question is whether v24 Thinking's count-5/count-7 final-answer
collapse disappears under balanced exposure. The same final 500-example
free-running test, per-count confusion table, NCC analysis, phase plots, and
local head-causality diagnostics are produced for direct comparison with v24.

```bash
python -m synthetic_counting_v24_2.run_v24_2 \
  --preset main \
  --stage prepare,train,phase,plots \
  --skip-completed
```
