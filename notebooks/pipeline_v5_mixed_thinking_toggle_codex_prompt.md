# Synthetic NIAH Counting v5: Explicit Thinking Soft Switch

## 1. Research question

Train one decoder-only Transformer that can execute either a thinking trace or a direct-answer path. The mode is selected by an explicit token that appears before the prompt, so the two targets never share an ambiguous conditioning prefix.

This version replaces the legacy v5 interface in which both modes started with the same `<BOS> prompt <Think/>` prefix and the non-thinking close token was supplied by the evaluator.

## 2. Exact sequence formats

For a prompt with `n` needles whose left-to-right marker identities are `M1 ... Mn`:

```text
thinking:
<BOS> <THINK_ON> prompt <Think/> M1 ... Mn </Think> <Cn> <EOS>

non-thinking:
<BOS> <THINK_OFF> prompt <Think/> </Think> <Cn> <EOS>
```

`<THINK_ON>` and `<THINK_OFF>` are atomic vocabulary tokens. They are control inputs, not generated outputs.

The default thinking trace contains marker tokens only. With `--trace-indices`, it becomes:

```text
<Think/> <I1> M1 <I2> M2 ... <In> Mn </Think>
```

## 3. Vocabulary

```text
special: <BOS>, <EOS>, <THINK_ON>, <THINK_OFF>, <Think/>, </Think>
noise:   <N0> ... <N63>
markers: <A> ... <J>
counts:  <C1> ... <C10>
optional trace indices: <I1> ... <I10>
```

The default vocabulary size is 90, or 100 with trace indices. Legacy 88/98-token v5 checkpoints are intentionally rejected because their embedding and output matrices are incompatible with the explicit-switch vocabulary.

## 4. Data generation

- Prompt length: 256 in `main`, 64 in `debug`.
- Count: uniform over 1..10.
- Needle positions: unique positions sampled uniformly without replacement.
- Marker identity: independent uniform draw from ten marker types; marker identities may repeat.
- Noise: independent uniform draw from 64 noise-token types.
- Trace order: prompt position from left to right.
- Batch mixture: 50% thinking and 50% non-thinking by default.

The same model and optimizer see both formats.

## 5. Loss mask

Prompt and control inputs are masked from the causal LM loss:

```text
masked input prefix:
<BOS> <THINK_ON/OFF> prompt <Think/>
```

The complete mode-specific continuation is supervised:

```text
thinking supervised targets:
M1 ... Mn </Think> <Cn> <EOS>

non-thinking supervised targets:
</Think> <Cn> <EOS>
```

The non-thinking `</Think>` must not be masked. It is the learned behavioral consequence of `<THINK_OFF>` and is evaluated autoregressively.

Per-component training logs:

```text
loss_total
loss_thinking_trace
loss_thinking_final_count
loss_nonthinking_close
loss_nonthinking_final_count
```

## 6. Model

Use the v2-style random-initialized Hugging Face `GPT2LMHeadModel`:

```yaml
architecture: GPT2LMHeadModel
position_embedding: learned_absolute
n_layer: 4
n_head: 4
n_embd: 256
n_inner: 1024
n_positions: 384
dropout: 0.0
activation: gelu_new
```

Main training:

```yaml
steps: 10000
batch_size: 128
optimizer: AdamW
lr: 3e-4
weight_decay: 0.01
warmup_steps: 500
seed: 1234
```

## 7. Autoregressive evaluation

### Thinking mode

Provide:

```text
<BOS> <THINK_ON> prompt <Think/>
```

The model must generate the marker trace, `</Think>`, the final count, and optionally `<EOS>`.

### Non-thinking mode

Provide:

```text
<BOS> <THINK_OFF> prompt <Think/>
```

The model must generate `</Think>` as its first token and then the final count. The evaluator does not provide the close token.

### Metrics

- `final_accuracy`: generated final count equals the gold count.
- `final_mae`: absolute count error when a count token is generated.
- `first_token_switch_accuracy`: ON starts a non-empty trace; OFF emits `</Think>` first.
- `empty_trace_rate`: the generated thinking block contains no trace tokens.
- Thinking-only trace exactness, marker precision/recall, premature close, missing close, and invalid count.

## 8. Mode-switch diagnostic

At the logits immediately after `<Think/>`, compare the two mode-conditioned prefixes. Save `tables/mode_switch.csv` with:

```text
step, mode, count, count_bin, n_examples,
p_close_after_think,
p_any_marker_after_think,
p_gold_first_marker_after_think,
p_desired_next_token,
desired_next_token,
argmax_token_after_think,
argmax_is_close,
argmax_is_gold_first_marker,
argmax_is_desired
```

Expected behavior:

- `<THINK_ON>`: the desired next token is the first marker, or `<I1>` when trace indices are enabled.
- `<THINK_OFF>`: the desired next token is `</Think>`.

Unlike the legacy ambiguous-prefix table, this is a conditional routing test: both prefixes contain the same prompt and `<Think/>`, but they differ at the earlier mode token.

## 9. Representation and attention analyses

Cache hidden states at the mode token, prompt markers, `<Think/>`, `</Think>`, trace markers, and final count positions. Compare thinking and non-thinking routes for the same underlying prompt.

For thinking, measure trace-to-prompt retrieval. For non-thinking, measure direct readout attention from `</Think>` / pre-count positions. All prompt-position calculations must use `RenderSpans.seq_start`, because the explicit mode token shifts the prompt from position 1 to position 2.

## 10. Outputs

```text
outputs/v5_explicit_switch/
  config.json
  vocab.json
  checkpoints/final.pt
  tables/train_log.csv
  tables/eval_by_step.csv
  tables/eval_examples.csv
  tables/mode_switch.csv
  tables/attention_metrics.csv
  figures/train_loss_by_step_and_mode.png
  figures/final_accuracy_by_step_mode.png
  figures/final_accuracy_by_count_mode.png
  figures/trace_metrics_by_count.png
  figures/mode_switch_accuracy_by_step.png
  report.md
  report.html
```

## 11. Commands

```bash
python -m synthetic_niah_v5.run_v5 --preset debug --stage all
python -m synthetic_niah_v5.run_v5 --preset main --stage all --device cuda
```

Use `notebooks/Trace_Count_v5_Colab.ipynb` for Colab. It saves the completed result bundle under:

```text
/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/Synthetic_CoT_NiaH_Count/colab_results/
```

## 12. Acceptance criteria

1. Vocabulary round-trip includes both mode tokens.
2. Both rendered variants place the mode token at position 1 and prompt at position 2.
3. Thinking labels supervise trace, close, count, and EOS.
4. Non-thinking labels supervise close, count, and EOS.
5. Thinking and non-thinking query prefixes differ before the prompt.
6. Non-thinking evaluation starts before `</Think>` and requires the model to generate it.
7. `mode_switch.csv` reports next-token routing separately for ON and OFF.
8. Debug `--stage all` produces tables, plots, and reports without using a legacy checkpoint.
