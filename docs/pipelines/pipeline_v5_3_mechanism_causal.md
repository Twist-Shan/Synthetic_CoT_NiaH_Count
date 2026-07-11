# v5.3 Mechanism Diagnostics

## Goal

Use the trained v5 explicit-switch model to distinguish two candidate counting mechanisms:

- `THINK_OFF`: broad collection of prompt needles into one final-answer state.
- `THINK_ON`: indexed item-specific retrieval, followed by a progress/stop update and trace-mediated final readout.

This pipeline does not retrain the model. It requires a completed corrected v5 run with `trace_indices=true`.

## Sites and numbering

All layer/head identifiers are zero-based. `L1H0` means head 0 in the second Transformer block.

- Direct final-count query: `</Think>` in the non-thinking sequence; this state predicts `<C_n>`.
- CoT retrieval query k: `<I_k>`; this state predicts marker `M_k`.
- CoT successor query k: marker `M_k`; this state predicts `<I_{k+1}>` or `</Think>`.
- CoT final-count query: `</Think>` after the indexed trace; this state predicts `<C_n>`.

## Descriptive attention

For a query row `A(q, :)`:

```text
prompt_needles_mass = sum_j A(q, needle_j)
needle_entropy_normalized = H(A(q, needles) / prompt_needles_mass) / log(n)
broad_aggregation_score = prompt_needles_mass * needle_entropy_normalized
correct_top1 = 1[argmax_j A(<I_k>, needle_j) = k]
diagonal_dominance = A(<I_k>, needle_k) / prompt_needles_mass
```

The direct hypothesis predicts high broad aggregation at its final-count query. The CoT hypothesis predicts high `correct_top1`, diagonal dominance, and raw correct-needle mass at `<I_k>`. CoT final readout is summarized by attention to all and last trace markers.

## Head ablation

Heads are ranked from an independent attention sample into:

- targeted retrieval heads;
- direct broad-aggregation heads;
- trace-readout heads.

The pipeline masks top-1/top-2/top-4 groups, random matched-size controls, each whole layer, and all heads. It records both teacher-forced logit-margin drops and free-running final/trace accuracy.

## Clean-to-corrupt patching

Head-output patching replaces the selected concatenated head slice immediately before `attn.c_proj`. Residual patching replaces the full residual state after a block.

### Retrieval identity

Change one prompt needle's marker identity while preserving count and all absolute positions. Patch clean information at `<I_k>` into the corrupt run and measure recovery of the clean marker against the corrupt marker.

### Direct count readout

Delete the final needle by replacing it with a noise token. Prompt length and answer position remain matched. Patch the clean direct-answer state into the corrupt run and measure recovery of `<C_n>` against `<C_{n-1}>`.

### Thinking count readout

Apply the same deletion and patch at the post-trace count query. Since trace lengths differ, clean and corrupt query positions differ. The output explicitly records `position_matched=0`; this result is secondary because learned absolute position is a confound.

For all patches:

```text
normalized_recovery = (patched_margin - corrupt_margin) / (clean_margin - corrupt_margin)
```

The value is not clipped. Zero means no recovery and one means full clean recovery.

## Progress and final aggregation

The trace-conflict experiment holds the prompt fixed while forcing traces of length 1-10, then measures whether final count follows prompt or trace. The progress transplant replaces the residual after marker k with a state from k-1 or k+1 and tests the next-index logit. A different-prompt, same-k, same-position transplant is the primary control. The k±1 intervention remains position-confounded and should be interpreted only when it exceeds the matched control.

## Outputs

`v5_3_mechanism_causal/tables/` contains per-example rows and grouped summaries. `figures/` contains attention, ablation, patching, conflict, and transplant plots. The Colab notebook saves this complete directory to Drive.
