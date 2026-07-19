# v19: shared decimal digit tokenization

v19 changes exactly one conceptual axis from v18: ordinal indices and final
counts are no longer atomic tokens. Both are written with the same decimal
digits `D0..D9`. The model, optimizer, prompt distribution, count range, and
four-run comparison remain aligned with v18.

## Controlled comparison

All main runs use:

- prompt length 1024;
- 256 noise token types and 10 disjoint marker types;
- count range 1-128;
- uniform or power-law training counts, with
  `p(c) proportional to c^-1.5` for the power condition;
- direct or CoT completion, giving four independent models;
- a 4-layer, 4-head, width-256 decoder with MLP width 1024 and RoPE;
- completion-only next-token cross-entropy, AdamW, batch 32, and 10,000 steps.

## Vocabulary and grammar

The vocabulary contains 283 tokens:

- `N0..N255`: prompt noise;
- `M0..M9`: marker identities;
- `ANSWER`, `START`, `END`, and `PAD`;
- `<Index>`, `<Count>`, and `<NumEnd>`;
- shared digits `D0..D9`.

Direct completion for count 12:

```text
prompt ANSWER <Count> D1 D2 <NumEnd>
```

CoT completion for a two-needle example:

```text
prompt START <Index> D1 M_a <Index> D2 M_b END <Count> D2 <NumEnd>
```

For a trace long enough to reach index 12, that local segment is:

```text
<Index> D1 D2 M_12
```

`<Index>` and `<Count>` separate semantic roles without giving each number an
atomic embedding. Leading zeros are rejected. A generated number must end at a
marker boundary or `<NumEnd>` and parse to `1..128`.

## Digit-aware analysis anchors

At trace step `k`, marker `M_k` is predicted by the final digit of `k`. This is
the k-to-k attention query. For one-digit indices it is the only digit; for
index 128 it is `D8`, after `<Index> D1 D2`.

The final-answer anchor is `<Count>`, because its next-token distribution
predicts the first answer digit. Subsequent answer digits are autoregressively
conditioned on earlier digits, and `<NumEnd>` verifies termination. State and
attention tables therefore describe the scalar-entry state at `<Count>`;
free-running final accuracy still evaluates the complete parsed number.

The exported analysis remains v18-compatible:

- behavior and learning dynamics by `1-32`, `33-64`, `65-96`, `97-128`;
- broad prompt-needle attention for direct counting;
- k-to-k raw mass, needle-conditional diagonal dominance, top-1 retrieval,
  successor preparation, and final trace readout for CoT;
- held-out nearest-centroid and ridge probes;
- PC1-PC6 count/progress centroid geometry for embedding and Layers 1-4.

## Running

```bash
python -m synthetic_counting_v19.run_v19 \
  --preset main --suite all --stage all --device cuda \
  --out-root runs/synthetic_counting_v19 \
  --run-name v19_main_all_seed1234 \
  --skip-completed
```

Use `notebooks/Trace_Count_v19_Colab.ipynb` for Drive mounting, streaming
progress, checkpoint synchronization, tables, figures, PC1-PC6 interaction,
result export, and optional runtime disconnection.
