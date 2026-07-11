# v5.4 Causal Count-State Analysis

## Why this experiment exists

The v5 probe shows that count is linearly readable from several hidden states. Readability alone is not a mechanism: a probe may exploit information that the language model never uses. v5.4 asks whether a count direction and count-state residual are *causal* for the answer.

The two target mechanisms are:

1. **THINK_OFF direct counting:** prompt needles are broadly aggregated into the fixed `</Think>` answer query.
2. **THINK_ON post-retrieval counting:** after the indexed trace has been supplied, a count state is available at the final `</Think>` query and controls `<C_n>`.

This pipeline restores the trained explicit-switch v5 checkpoint. It does not retrain the model.

## Sites

All layers are zero-based. Interventions are applied to the full 256-dimensional residual immediately after a Transformer block.

### `nonthinking_close`

```text
<BOS> <THINK_OFF> prompt[256] <Think/> </Think>
                                           ^ query predicts <C_n>
```

The query is always at absolute position 259. Count and position are therefore independent.

### `thinking_natural_close`

```text
<BOS> <THINK_ON> prompt[256] <Think/> <I1>M1 ... <In>Mn </Think>
                                                               ^ query
```

This is the in-distribution sequence, but the query position is `259+2n`. It is used only as a secondary ecological target for cross-mode steering; directions are not fitted here because position perfectly leaks count.

### `thinking_fixed_trace_close`

```text
<BOS> <THINK_ON> prompt[256] <Think/> <I1>A <I2>B ... <I5>E </Think>
                                                                  ^ query
```

The trace canvas has a fixed number of index/marker pairs and a constant marker template for every prompt count. The close token is at one absolute position. This is a counterfactual position-control: it is not an ID accuracy benchmark, but it lets us ask whether prompt count remains encoded after a trace-shaped computation without the `259+2n` shortcut.

## Experiment 1: count-direction geometry

For every site and block layer, collect the 256-dimensional query residual on an independent direction-training split. Fit:

- `adjacent_mean`: `d = mean_n(mu_(n+1) - mu_n)`;
- `ridge`: a ridge-regression direction predicting numeric count;
- `shuffled_ridge`: the same estimator after permuting count labels.

The direction is normalized and oriented so increasing projection means increasing count. Report:

```text
step_size = mean_n (mu_(n+1)-mu_n) dot d
projection_R2 = held-out R2 from the one-dimensional projection h dot d
adjacent cosine = mean cosine between all mu_(n+1)-mu_n vectors
```

High held-out R2 means readable geometry. High adjacent cosine is stronger evidence that one approximately shared `+1` direction exists. Neither is yet causal.

## Experiment 2: causal direction steering

At one block output and one query token, add:

```text
h' = h + alpha * step_size * d
alpha in {-2, -1, 0, 1, 2}
```

Then run all remaining blocks and the LM head normally. Restrict readout to `<C1>...<C10>` and report the change from the same example at `alpha=0` in predicted count and probability-weighted expected count.

Directions tested:

- the target site's own adjacent-mean and ridge directions;
- `centroid_transport`: for an example with count `n`, add the class-specific displacement `mu_(clip(n+alpha))-mu_n`;
- shuffled-label and random orthogonal controls;
- the THINK_OFF direction and THINK_OFF centroid transport transferred into both thinking sites.

A causal arithmetic direction should produce a monotonic dose response, move by approximately one count per unit alpha, and outperform equal-norm controls. Centroid transport is less restrictive: it asks whether the model uses a class-specific count manifold even when no single global `+1` axis exists. Cross-mode transfer tests whether the two modes share either representation.

These interventions form an evidence ladder:

1. **global direction**: one approximately translation-invariant arithmetic axis;
2. **centroid transport**: count-specific state geometry is causally usable;
3. **whole-state swap**: the complete residual state is sufficient, but may contain more than count.

## Experiment 3: position-matched residual state swap

Pair a receiver with count `n` and a donor with count `n+1`. At the same site, token position, and block layer, replace the receiver's complete 256-dimensional residual with the donor residual, then continue the forward pass.

Primary outcome:

```text
follows_donor = 1[predicted count == n+1]
```

Control: swap a residual from a different prompt with the *same* count. The THINK_OFF and fixed-trace thinking sites have matched absolute positions, so a donor-count effect cannot be explained by moving the query token.

This is a strong sufficiency test, although a whole-state swap may transfer more than count alone. Direction steering is the more selective intervention.

## Experiment 4: paired needle deletion and mediation

Create a clean/corrupt pair by replacing the final prompt needle with a noise token. Prompt length is unchanged and count changes `n -> n-1`.

At every block:

1. project `h_clean - h_corrupt` onto the fitted adjacent count direction and express it in one-count step units;
2. patch the clean full residual into the corrupt run and measure recovery of the clean `<C_n>` versus `<C_(n-1)>` logit margin;
3. when v5.3 head groups are available, patch clean pre-`c_proj` head-output slices for direct-broad, targeted-retrieval, and trace-readout groups.

For the thinking position-control, clean and corrupt prompts receive the *same constant fixed-length trace*. Thus the sequence length, close position, and supplied trace are held fixed; only prompt count changes.

```text
normalized_recovery = (patched_margin - corrupt_margin) / (clean_margin - corrupt_margin)
```

This locates where a `+1 needle` difference first becomes aligned with the count direction and which components causally write an answer-relevant state.

## Interpretation rules

- Probe/geometry only: information is readable.
- Monotonic direction steering with matched controls: the direction is causally effective.
- Successful centroid transport with weak global steering: count is causal but represented as class-specific states rather than one shared linear `+1` direction.
- Position-matched state swap: the layer state is sufficient to transfer count.
- Needle-delete patch recovery: the component mediates the prompt-count difference.
- A null fixed-trace result does not prove no post-retrieval state exists, because the constant trace is counterfactual. It must be reported next to the natural-sequence cross-mode steering result.

## Outputs

`v5_4_count_state_causal/tables/` contains per-example and summary CSVs:

- `direction_geometry.csv`
- `cross_mode_direction_cosine.csv`
- `steering_rows.csv`, `steering_summary.csv`
- `state_swap_rows.csv`, `state_swap_summary.csv`
- `mediation_rows.csv`, `mediation_summary.csv`

`figures/` contains geometry, dose-response, state-swap, residual-mediation, and optional head-mediation panels. `directions.npz` stores fitted global vectors, `count_centroids.npz` stores every site/layer/count centroid, and `run_config.json` records every sample count and control setting.
