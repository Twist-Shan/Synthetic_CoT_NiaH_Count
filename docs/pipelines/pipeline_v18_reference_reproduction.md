# v18: skewed count-128 with explicit index-marker traces

v18 is a focused four-model comparison designed to preserve inspectable
targeted retrieval while testing a strongly imbalanced count distribution.

The final checkpoints also support the v10-style descriptive analysis stack.
The stages are separable, so an interrupted analysis does not require training
again:

- `--stage attention`: broad aggregation, k-to-k retrieval, successor
  preparation, and final trace readout for every layer and attention head;
- `--stage state`: held-out count/progress probes and PC1-PC6 centroid geometry
  at the embedding and post-Layer-1 through post-Layer-4 residual states;
- `--stage plots`: recreate all learning, attention, and state figures from the
  saved CSV tables.

## Four training runs

All runs use prompt length 1024 and count range 1-128:

| distribution | direct | CoT |
|---|---:|---:|
| uniform, `p(c)=1/128` | yes | yes |
| power, `p(c) proportional to c^-1.5` | yes | yes |

The power distribution deliberately makes large counts rare. `suite_manifest.csv`
is the machine-readable source of truth.

## Disjoint vocabulary

- Noise: 256 tokens, `N0..N255`.
- Needles: 10 tokens, `M0..M9`.
- Trace indices: 128 tokens, `I1..I128`.
- Final counts: a separate 128-token family, `C1..C128`.
- Control: `ANSWER`, `START`, `END`, and `PAD`.

The separation between `I_k` and `C_n` is intentional: an attention analysis can
ask whether the query at `I_k` retrieves the identity of the kth prompt needle,
without conflating ordinal progress with the final scalar answer.

Direct completion:

```text
N...M...N  ANSWER  C_n
```

CoT completion, where prompt needles are ordered by position:

```text
N...M...N  START  I_1 M_type(1) I_2 M_type(2) ... I_n M_type(n) END C_n
```

Only completion tokens receive next-token cross-entropy. Prompt tokens remain
causal context but have label `IGNORE_INDEX`.

## Evaluation metrics

Evaluation is free-running greedy generation, not teacher forcing.

- `token_accuracy`: whether the generated final `C_n` equals the gold count.
- `enumeration_accuracy`: whether CoT generated a well-formed `I_1..I_n` sequence
  and then closed with `END` at the correct length.
- `trace_marker_accuracy`: fraction of the `n` trace marker identities matching
  the position-sorted prompt markers; missing markers count as errors.
- `trace_exact_accuracy`: one only when every index and marker identity is correct.
- `primary_accuracy`: final scalar `C_n` accuracy for both direct and CoT, so the
  two modes are directly comparable.

## Model and optimizer

- Randomly initialized pre-norm decoder-only Transformer.
- 4 layers, 4 heads, width 256, head width 64, MLP width 1024 with GELU.
- RoPE with base 10000; final LayerNorm; tied embedding and unembedding.
- AdamW beta `(0.9, 0.95)`, weight decay 0.01.
- Learning rate `3e-4`, 200-step linear warmup, cosine decay to zero.
- Batch size 32, 10,000 steps, clipping 1.0, bf16 on supported CUDA.

## Analysis outputs

Learning dynamics and final behavior are split into `1-32`, `33-64`, `65-96`,
and `97-128`. The output bundle includes:

- `tables/dynamics_by_band.csv`, `final_by_band.csv`, and `final_by_count.csv`;
- `tables/attention_detail.csv` and `attention_summary.csv`;
- `tables/state_probe_summary.csv`, `state_centroids_pca.csv`, and
  `state_pca_variance.csv`;
- learning curves, layer-head attention heatmaps, probe heatmaps, PCA variance
  plots, and count-centroid projections under `figures/`.

The notebook includes an interactive Plotly view that can select a model,
semantic state site, hidden-state index, and any three axes among PC1-PC6.
