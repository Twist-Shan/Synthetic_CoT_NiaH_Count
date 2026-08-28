# v23: final-count-upweighted paired no-index run

v23 preserves the v22 Thinking grammar exactly:

```text
<Think> <Sep> marker_1 <Sep> marker_2 ... <Sep> marker_n </Think> <Ans> <n>
```

Its single objective change is `final_count_loss_weight=8`. Unlike the
canonical v22 run, v23 retrains both `rope/nonthinking` and `rope/thinking` so
their comparison controls the answer-token weight, data, seed, initialization,
architecture, optimizer, 10,000-step schedule, and evaluation manifests.

This creates two identifiable comparisons:

- v22 Thinking vs v23 Thinking tests whether v22's weak answer readout was
  caused by final-token loss dilution.
- v23 Non-thinking vs v23 Thinking tests the effect of the separator trace when
  both models receive the same 8x supervision on the final count.

The 8x coefficient is the same **per-token weight**, not an attempt to make the
final count the same fraction of total loss in the two modes. After the
task-output switch, its normalized share is `8/10 = 80%` for Non-thinking and
`8/(2n+12)` for a Thinking example with count `n` (about 19% at `n=15`).

Run the short paired smoke test:

```bash
python -m synthetic_counting_v23.run_v23 \
  --preset debug \
  --stage prepare,train,phase,plots \
  --skip-completed
```

Run the canonical experiment:

```bash
python -m synthetic_counting_v23.run_v23 \
  --preset main \
  --stage prepare,train,phase,plots \
  --skip-completed
```

The supplied `notebooks/Trace_Count_v23_NoIndex_FCW8_Colab.ipynb` mounts Drive,
trains both modes, streams live progress, resumes optimizer/RNG state, checks
the final artifacts for both models, and disconnects only after persistence is
verified.
