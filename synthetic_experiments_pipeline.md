# Synthetic Trace-Enumeration Counting Pipeline v0

This file specifies the first synthetic experiment to implement. The task is intentionally minimal: a random token sequence contains sparse marker tokens `X`, `Y`, and `Z` among noise tokens `N0..N63`; the model is given an explicit trace that enumerates the markers in left-to-right order; the model must output the total count token.

The core experiment is not to vary the task. The core experiment is to vary the **loss mask** on the same tokenized examples.

Canonical example:

```text
<BOS> N4 X N9 Y N2 Z N7 <Think> <I1> X <I2> Y <I3> Z <Think> <ANS> <C3> <EOS>
```

Interpretation:

```text
source sequence:       N4 X N9 Y N2 Z N7
positive markers:      X, Y, Z
trace:                 <I1> X <I2> Y <I3> Z
final answer:          <C3>
```

---

## 1. Scientific purpose

The draft motivates synthetic sparse-counting experiments as a controlled bridge between NIAH counting and trained small transformers. The relevant planned experiment is symbolic sparse counting with think tokens and trace-supervised enumeration, using small transformers to study whether counter-like states emerge during training. The draft also stresses token-localization: the operational counter may live at list markers, item-final tokens, the final think token, or the answer prefix rather than at the item token itself.

This v0 isolates one question:

> Given an explicit trace of the counted markers, which loss mask best trains a small causal transformer to use the trace and output the correct final count?

This is a narrower version of the draft's S1/S3 synthetic plan. It deliberately uses only symbolic marker/noise sequences and trace enumeration.

---

## 2. Task definition

### 2.1 Vocabulary

Use a fixed hand-written tokenizer. Each whitespace-separated token is one vocabulary item. Do not use a pretrained tokenizer.

Required tokens:

```text
special:
  <PAD>, <BOS>, <EOS>, <Think>, <ANS>

positive markers:
  X, Y, Z

noise tokens:
  N0, N1, ..., N63

trace index tokens:
  <I1>, <I2>, ..., <I64>

count answer tokens:
  <C0>, <C1>, ..., <C64>
```

The maximum count for v0 is `64`. Keep this configurable as `max_count`.

Important practical constraint: because `<Ck>` is an atomic class token, evaluation on count values whose `<Ck>` token was never supervised in training is not a clean extrapolation test. For v0, keep all evaluation counts inside the supervised count range. Test length generalization and density robustness first. If true unseen-count extrapolation is needed later, replace atomic `<Ck>` answers with decimal digits or a compositional unary answer format.

### 2.2 Base example generation

Parameters:

```yaml
seq_len: L
count: n
positive_vocab: [X, Y, Z]
noise_vocab: [N0, ..., N63]
max_count: 64
```

Constraints:

```text
0 <= n <= min(L, max_count)
```

Generation algorithm:

```python
positions = sorted(sample_without_replacement(range(L), n))
seq = []
marker_tokens = []
for i in range(L):
    if i in positions:
        marker = random_choice(["X", "Y", "Z"])
        seq.append(marker)
        marker_tokens.append(marker)
    else:
        seq.append(random_choice(["N0", ..., "N63"]))

trace = []
for k, marker in enumerate(marker_tokens, start=1):
    trace.extend([f"<I{k}>", marker])

full_tokens = ["<BOS>"] + seq + ["<Think>"] + trace + ["<Think>", "<ANS>", f"<C{n}>", "<EOS>"]
```

For `n = 0`, the trace is empty:

```text
<BOS> N4 N9 N2 N7 <Think> <Think> <ANS> <C0> <EOS>
```

### 2.3 Trace convention

The trace must enumerate positive markers in the same order as they appear in the source sequence.

Example:

```text
source:  N4 Z N9 X N2 Z N7
trace:   <I1> Z <I2> X <I3> Z
answer:  <C3>
```

Do not sort by marker type. Do not group all `X`s before `Y`s or `Z`s.

---

## 3. Dataset splits

Use balanced generation over `(L, n)`, not natural binomial sampling. This avoids the model learning the marginal count distribution.

Recommended v0 splits:

```yaml
train:
  lengths: [32, 64, 128]
  counts: 0..24
  examples_per_pair: 512

val_id:
  lengths: [32, 64, 128]
  counts: 0..24
  examples_per_pair: 128

val_length_ood:
  lengths: [256, 512]
  counts: 0..24
  examples_per_pair: 128

val_density_shift_low:
  lengths: [512]
  counts: 0..8
  examples_per_pair: 128

val_density_shift_high:
  lengths: [64]
  counts: 16..24
  examples_per_pair: 128
```

Use three random seeds for the final runs:

```yaml
seeds: [0, 1, 2]
```

For debugging, use:

```yaml
debug:
  train_examples_per_pair: 8
  val_examples_per_pair: 4
  lengths: [16, 32]
  counts: 0..4
```

---

## 4. Saved JSONL schema

Write one JSON object per example.

Required fields:

```json
{
  "example_id": "train_L64_n3_seed0_000001",
  "split": "train",
  "seed": 0,
  "seq_len": 64,
  "count": 3,
  "source_tokens": ["N4", "X", "N9", "Y", "N2", "Z", "N7"],
  "positive_positions_source": [1, 3, 5],
  "positive_markers": ["X", "Y", "Z"],
  "trace_tokens": ["<I1>", "X", "<I2>", "Y", "<I3>", "Z"],
  "answer_token": "<C3>",
  "full_tokens": ["<BOS>", "N4", "X", "N9", "Y", "N2", "Z", "N7", "<Think>", "<I1>", "X", "<I2>", "Y", "<I3>", "Z", "<Think>", "<ANS>", "<C3>", "<EOS>"],
  "spans": {
    "source_start": 1,
    "source_end_exclusive": 8,
    "think_open_idx": 8,
    "trace_start": 9,
    "trace_end_exclusive": 15,
    "think_close_idx": 15,
    "ans_idx": 16,
    "count_idx": 17,
    "eos_idx": 18,
    "trace_pairs": [
      {"k": 1, "index_idx": 9, "marker_idx": 10, "marker": "X", "source_idx": 2},
      {"k": 2, "index_idx": 11, "marker_idx": 12, "marker": "Y", "source_idx": 4},
      {"k": 3, "index_idx": 13, "marker_idx": 14, "marker": "Z", "source_idx": 6}
    ]
  }
}
```

Index convention:

```text
All token indices are full-token indices, starting from 0.
source_idx is also a full-token index, not a zero-based index inside source_tokens.
```

For `n = 0`, `trace_pairs = []`, `trace_start == trace_end_exclusive`, and `think_close_idx == think_open_idx + 1`.

---

## 5. Tokenization and batching

Implement `VocabTokenizer`:

```python
class VocabTokenizer:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    def encode(self, tokens: list[str]) -> list[int]: ...
    def decode(self, ids: list[int]) -> list[str]: ...
```

Batch collation:

```python
input_ids:      LongTensor[B, T_max]
attention_mask: LongTensor[B, T_max]   # 1 for real tokens, 0 for pad
labels:         LongTensor[B, T_max]    # token ids or -100
loss_weights:   FloatTensor[B, T_max]   # 0 for ignored/pad, positive for supervised positions
metadata:       list[dict]
```

Padding:

```text
input_ids padded with <PAD>
attention_mask padded with 0
labels padded with -100
loss_weights padded with 0.0
```

---

## 6. Loss-mask ablations

All regimes use the exact same `full_tokens`. Only `labels` and `loss_weights` differ.

### 6.1 Causal LM alignment

Use standard causal next-token prediction. With Hugging Face causal LM models, pass `input_ids` to the model, but compute the weighted loss manually:

```python
logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

# logits[:, t, :] predicts input_ids[:, t+1]
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()
shift_weights = loss_weights[:, 1:].contiguous()

ce = cross_entropy(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1),
    ignore_index=-100,
    reduction="none",
).view(batch_size, -1)

valid = (shift_labels != -100).float()
loss = (ce * valid * shift_weights).sum() / (valid * shift_weights).sum().clamp_min(1.0)
```

Meaning of `labels[j]`:

```text
labels[j] = input_ids[j] means: supervise prediction of token j from the prefix ending at token j-1.
labels[j] = -100 means: ignore prediction of token j.
```

Always set `labels[0] = -100` because `<BOS>` is never predicted.

### 6.2 Example token indices

For the running example:

```text
idx token
0   <BOS>
1   N4
2   X
3   N9
4   Y
5   N2
6   Z
7   N7
8   <Think>
9   <I1>
10  X
11  <I2>
12  Y
13  <I3>
14  Z
15  <Think>
16  <ANS>
17  <C3>
18  <EOS>
```

### 6.3 Regime A: `full_sequence`

Supervise every non-BOS token.

```text
supervised token indices: 1..eos_idx
weights: all 1.0
```

This is the strict full-sequence language-modeling baseline. It asks the model to predict random source noise tokens too. This may waste capacity because the noise sequence is deliberately unpredictable, but that is the point of this ablation.

Example labels:

```text
idx:    0      1   2  3   4  5   6  7   8        9     10 11    12 13    14 15       16    17    18
label: -100   N4  X  N9  Y  N2  Z  N7  <Think> <I1>  X  <I2> Y  <I3> Z  <Think> <ANS> <C3> <EOS>
weight: 0     1   1  1   1  1   1  1   1        1     1  1     1  1     1  1       1     1     1
```

### 6.4 Regime B: `full_sequence_final_weighted`

Supervise every non-BOS token, but upweight the final answer token.

```text
supervised token indices: 1..eos_idx
weight at count_idx: final_weight, default 10.0
weight at eos_idx: eos_weight, default 1.0
all other supervised weights: 1.0
```

Recommended sweep:

```yaml
final_weight: [2.0, 5.0, 10.0, 20.0]
eos_weight: 1.0
```

This tests whether full LM training becomes more useful when the answer readout is emphasized.

### 6.5 Regime C: `completion_only`

Mask the source sequence. Supervise only the trace and answer continuation.

```text
supervised token indices: think_open_idx..eos_idx
ignored token indices: 0..source_end_exclusive-1
weights: all supervised weights 1.0
```

For the example:

```text
supervised: <Think> <I1> X <I2> Y <I3> Z <Think> <ANS> <C3> <EOS>
ignored:    <BOS> N4 X N9 Y N2 Z N7
```

This is the normal teacher-forced trace-generation setting. It asks the model to generate the trace and then the answer, but does not penalize it for failing to predict random source noise.

### 6.6 Regime D: `completion_final_weighted`

Mask the source sequence. Supervise the trace and answer continuation. Upweight the final count token.

```text
supervised token indices: think_open_idx..eos_idx
weight at count_idx: final_weight, default 10.0
weight at eos_idx: eos_weight, default 1.0
all other supervised continuation weights: 1.0
```

Recommended sweep:

```yaml
final_weight: [2.0, 5.0, 10.0, 20.0]
eos_weight: 1.0
```

This is the main candidate regime. It avoids the random-source-token loss while still learning to emit the explicit trace. The final answer receives more gradient so the model cannot solve trace generation while neglecting the final count readout.

### 6.7 Regime E: `final_count_only`

Provide the full gold trace as input, but supervise only the final count token.

```text
supervised token indices: [count_idx]
ignored token indices: all others
weight at count_idx: 1.0
```

For the example:

```text
input:      <BOS> N4 X N9 Y N2 Z N7 <Think> <I1> X <I2> Y <I3> Z <Think> <ANS> <C3> <EOS>
supervise: only <C3>
```

This regime does **not** teach the model to generate the trace. It only tests whether the model can read the final answer from a supplied trace. Therefore, evaluate it primarily with teacher-forced trace prefixes, not with free trace generation.

Optional variant:

```yaml
final_count_only_include_eos: false
```

Keep the default as `false`. If enabled, also supervise `<EOS>` with weight `1.0`, but report it as a separate ablation.

---

## 7. Model

Use a small decoder-only transformer trained from scratch. Do not initialize from a pretrained checkpoint.

Preferred implementation for speed: `transformers.GPT2LMHeadModel` with a custom config and custom integer tokenizer.

### 7.1 Tiny debug config

```yaml
model_name: tiny_debug
n_layer: 2
n_head: 2
n_embd: 64
n_inner: 256
activation_function: gelu_new
resid_pdrop: 0.0
embd_pdrop: 0.0
attn_pdrop: 0.0
n_positions: 2048
vocab_size: auto
```

### 7.2 Main v0 config

```yaml
model_name: small_main
n_layer: 4
n_head: 4
n_embd: 128
n_inner: 512
activation_function: gelu_new
resid_pdrop: 0.0
embd_pdrop: 0.0
attn_pdrop: 0.0
n_positions: 2048
vocab_size: auto
```

### 7.3 Optional larger config

Only run this after the pipeline is validated:

```yaml
model_name: medium_optional
n_layer: 6
n_head: 8
n_embd: 256
n_inner: 1024
activation_function: gelu_new
resid_pdrop: 0.0
embd_pdrop: 0.0
attn_pdrop: 0.0
n_positions: 2048
vocab_size: auto
```

---

## 8. Training

### 8.1 Optimizer

```yaml
optimizer: AdamW
learning_rate: 3.0e-4
betas: [0.9, 0.95]
weight_decay: 0.1
grad_clip_norm: 1.0
warmup_steps: 1000
lr_schedule: cosine
batch_size: 128
grad_accum_steps: 1
max_steps: 50000
eval_every: 1000
save_every: 5000
precision: bf16_if_available_else_fp32
```

Use the same training examples, model config, optimizer config, and seed across loss-mask regimes. The only changed variable should be `loss_mask` and, for weighted regimes, `final_weight`.

### 8.2 Training command interface

Implement commands like:

```bash
python -m src.generate_data \
  --out_dir data/trace_count_v0 \
  --max_count 64 \
  --noise_vocab_size 64 \
  --train_lengths 32,64,128 \
  --train_counts 0:24 \
  --examples_per_pair_train 512 \
  --examples_per_pair_val 128 \
  --seeds 0,1,2

python -m src.train \
  --data_dir data/trace_count_v0 \
  --model_config configs/model/small_main.yaml \
  --loss_mask completion_final_weighted \
  --final_weight 10 \
  --seed 0 \
  --out_dir runs/trace_count_v0/small_main/completion_final_weighted_fw10_seed0

python -m src.eval \
  --checkpoint runs/trace_count_v0/small_main/completion_final_weighted_fw10_seed0/checkpoint_final \
  --data_dir data/trace_count_v0 \
  --splits val_id,val_length_ood,val_density_shift_low,val_density_shift_high \
  --out_dir runs/trace_count_v0/small_main/completion_final_weighted_fw10_seed0/eval
```

---

## 9. Evaluation

Implement two evaluation modes. They answer different questions.

### 9.1 Teacher-forced final-count evaluation

Purpose: evaluate the final count readout when the model is given the correct source and correct trace.

For each example, feed the prefix through `<ANS>`:

```text
<BOS> source <Think> trace <Think> <ANS>
```

Then read the logits at the `<ANS>` position:

```python
logits_for_next = logits[ans_idx, :]
count_logits = logits_for_next[count_token_ids]
pred_count = argmax(count_logits)
```

Metrics:

```text
count_accuracy
mean_absolute_error
undercount_rate
overcount_rate
count_nll
accuracy_by_count
accuracy_by_seq_len
accuracy_by_density
```

This is the primary metric for `final_count_only`, because that regime does not learn to generate the trace.

### 9.2 Autoregressive trace-and-answer evaluation

Purpose: evaluate whether the model can generate the trace and final answer from the source sequence.

Prefix:

```text
<BOS> source
```

Generate greedily until `<EOS>` or max new tokens:

```yaml
max_new_tokens: 2 * max_count + 4
sampling: false
temperature: null
top_p: null
```

Expected generation:

```text
<Think> <I1> X <I2> Y ... <Think> <ANS> <Cn> <EOS>
```

Parse generation:

1. Find first `<Think>`.
2. Find second `<Think>` after the first.
3. Find `<ANS>` after second `<Think>`.
4. The next token after `<ANS>` should be a count token `<Ck>`.
5. The next token after `<Ck>` should preferably be `<EOS>`, but count accuracy can still be computed if `<EOS>` is missing.

Trace metrics:

```text
trace_exact_match
trace_index_accuracy
trace_marker_precision
trace_marker_recall
trace_duplicate_rate
trace_length_error
format_validity
```

Count metrics:

```text
count_accuracy
mean_absolute_error
undercount_rate
overcount_rate
invalid_answer_rate
```

Do not compare `final_count_only` against trace-supervised regimes using this autoregressive metric as the main score. It is expected to fail at trace generation because trace tokens receive no loss.

---

## 10. Experiment matrix

Run the following first:

```yaml
models:
  - tiny_debug
  - small_main

seeds:
  - 0
  - 1
  - 2

loss_masks:
  - full_sequence
  - full_sequence_final_weighted:
      final_weight: [5, 10]
  - completion_only
  - completion_final_weighted:
      final_weight: [5, 10]
  - final_count_only
```

Minimal first pass:

```yaml
model: tiny_debug
seed: 0
loss_masks:
  - full_sequence
  - completion_only
  - completion_final_weighted_final_weight_10
  - final_count_only
```

Expected output table:

```text
model | seed | loss_mask | final_weight | split | tf_count_acc | ar_count_acc | trace_exact | format_valid | mae | under | over
```

Where:

```text
tf_count_acc = teacher-forced final-count accuracy
ar_count_acc = autoregressive trace-and-answer count accuracy
```

---

## 11. Logging and artifacts

Each run directory should contain:

```text
run_dir/
  config.yaml
  vocab.json
  train_log.jsonl
  checkpoints/
    step_00001000/
    step_00002000/
    ...
    final/
  eval/
    val_id_metrics.json
    val_length_ood_metrics.json
    val_density_shift_low_metrics.json
    val_density_shift_high_metrics.json
    predictions_val_id.jsonl
    predictions_val_length_ood.jsonl
  plots/
    accuracy_by_step.png
    tf_accuracy_by_count.png
    ar_accuracy_by_count.png
    trace_exact_by_count.png
    loss_breakdown_by_segment.png
```

Log segment-wise loss during training:

```text
source_loss
think_boundary_loss
trace_index_loss
trace_marker_loss
answer_prefix_loss
count_loss
eos_loss
total_weighted_loss
```

Segment loss definitions:

```text
source_loss: labels on source tokens N*/X/Y/Z before first <Think>
think_boundary_loss: labels on the two <Think> tokens
trace_index_loss: labels on <I1>..<In>
trace_marker_loss: labels on trace marker tokens X/Y/Z
answer_prefix_loss: label on <ANS>
count_loss: label on <Cn>
eos_loss: label on <EOS>
```

For regimes where a segment is masked, report `null` or `NaN`, not zero.

---

## 12. Unit tests

Implement tests before long training.

### 12.1 Generator tests

For every generated example:

```python
assert full_tokens[0] == "<BOS>"
assert full_tokens[-1] == "<EOS>"
assert full_tokens[spans["think_open_idx"]] == "<Think>"
assert full_tokens[spans["think_close_idx"]] == "<Think>"
assert full_tokens[spans["ans_idx"]] == "<ANS>"
assert full_tokens[spans["count_idx"]] == f"<C{count}>"
assert len(positive_positions_source) == count
assert len(trace_tokens) == 2 * count
```

Trace order test:

```python
source_marker_tokens = [full_tokens[idx] for idx in positive_source_full_indices]
trace_marker_tokens = [pair["marker"] for pair in spans["trace_pairs"]]
assert trace_marker_tokens == source_marker_tokens
```

### 12.2 Loss-mask tests

Using the running example, verify exact supervised indices.

```python
example_indices = {
    "source": list(range(1, 8)),
    "think_open": 8,
    "trace": list(range(9, 15)),
    "think_close": 15,
    "ans": 16,
    "count": 17,
    "eos": 18,
}
```

Expected supervised sets:

```python
full_sequence = set(range(1, 19))
full_sequence_final_weighted = set(range(1, 19))
completion_only = set(range(8, 19))
completion_final_weighted = set(range(8, 19))
final_count_only = {17}
```

Weight checks:

```python
assert weight[17] == final_weight for weighted regimes
assert weight[j] == 1.0 for other supervised non-pad labels
assert weight[j] == 0.0 when labels[j] == -100
```

### 12.3 Evaluation tests

Teacher-forced final-count eval should use `logits[ans_idx]` to predict `full_tokens[count_idx]`.

Autoregressive parsing should mark invalid if any of these are missing:

```text
first <Think>
second <Think>
<ANS>
count token after <ANS>
```

---

## 13. Implementation order for Codex

Implement in this order:

1. `src/tokenizer.py`
   - fixed vocab construction
   - encode/decode
   - save/load `vocab.json`

2. `src/generate_data.py`
   - balanced `(L, n)` generation
   - trace rendering
   - span metadata
   - JSONL writing
   - generator unit tests

3. `src/loss_masks.py`
   - `build_labels_and_weights(example, loss_mask, final_weight, eos_weight)`
   - exact tests using the running example

4. `src/dataset.py`
   - JSONL dataset
   - batch collator with padding
   - labels and loss weights

5. `src/model.py`
   - GPT2 config from scratch
   - no pretrained tokenizer/model

6. `src/train.py`
   - weighted causal LM loss
   - logging
   - checkpointing
   - eval during training

7. `src/eval.py`
   - teacher-forced final-count eval
   - autoregressive trace-and-answer eval
   - metrics by count and length
   - predictions JSONL

8. `src/plots.py`
   - accuracy by step
   - accuracy by count
   - loss breakdown by segment

---

## 14. Acceptance criteria

The implementation is acceptable when all of the following pass:

```text
[ ] Generated examples exactly match the canonical format.
[ ] Trace enumerates source markers in left-to-right order.
[ ] n=0 examples render as <Think> <Think> <ANS> <C0> <EOS>.
[ ] Loss masks supervise exactly the intended token indices.
[ ] Weighted regimes upweight only count_idx unless configured otherwise.
[ ] Training runs with tiny_debug for at least 100 steps without NaNs.
[ ] Teacher-forced eval reports count accuracy and MAE.
[ ] Autoregressive eval reports format validity, trace metrics, and count metrics.
[ ] Results are grouped by loss_mask, final_weight, seed, split, count, and sequence length.
```