# v22: v20-matched no-index separator trace

v22 changes exactly the Thinking trace grammar from v20:

```text
v20: <Think> <1> marker_1 <2> marker_2 ... <n> marker_n </Think> <Ans> <n>
v22: <Think> <Sep> marker_1 <Sep> marker_2 ... <Sep> marker_n </Think> <Ans> <n>
```

The corpus split, query-first 256-character prompt, three-character target set,
RoPE model, optimizer, loss schedule, count range 1..30, atomic final answer,
checkpoint cadence, and evaluation manifests remain matched to v20.  Because
the Non-thinking objective is unchanged, the canonical v22 run trains only
`rope/thinking` and uses the existing v20 Non-thinking run as its comparison.

This is a **de-indexed supervised trace**, not a claim that the small model has
the same natural-language internal counter as a large reasoning model.  It
tests the narrower and identifiable question: are explicit ordinal trace tokens
necessary for ordered targeted retrieval and head-bank differentiation?

Run the short local/Colab smoke test:

```bash
python -m synthetic_counting_v22.run_v22 --preset debug --stage prepare,train,phase,plots --skip-completed
```

Run the canonical experiment:

```bash
python -m synthetic_counting_v22.run_v22 \
  --preset main \
  --stage prepare,train,phase,plots \
  --model-variant rope/thinking \
  --skip-completed
```

The supplied `notebooks/Trace_Count_v22_NoIndex_Colab.ipynb` additionally mounts
Drive, streams live progress, resumes optimizer/RNG state, verifies the final
checkpoint and analysis artifacts, and disconnects only after persistence is
confirmed.
