# Codex Prompt: Synthetic NIAH-Style Counting v6 — Separator Trace, No Indices

Implement v6 of the controlled synthetic NIAH-style counting experiment for small decoder-only Transformers trained from scratch.

v6 should copy the v2 experimental setup as closely as possible, with one intentional change:

```text
v2 thinking trace:
  <Think/> <1> <A> <2> <B> <3> <C> </Think> <Ans> <3>

v6 thinking trace:
  <Think/> <Sep> <A> <Sep> <B> <Sep> <C> </Think> <Ans> <3>
```

That is, **remove numeric trace indices from the thinking trace** and replace every item index with one fixed delimiter token `<Sep>`. The final answer still uses numeric tokens `<1>` ... `<10>`.

The motivation is to keep the v2 trace-supervised setup while removing the obvious prefix-count leakage from trace tokens `<1>`, `<2>`, ..., `<n>`. v6 should test whether the model still learns sequential retrieval and counting when the trace item boundary token is the same at every step.

Do **not** add OOD, distractors, realistic NIAH records, steering, patching, mixed thinking toggle, or loss-mask ablations in v6. Keep this experiment simple and directly comparable to v2.

---

## 0. Scientific question

We want to compare two separately trained small Transformers:

1. **non-thinking model**: directly predicts the final count after `<Ans>`;
2. **separator-trace thinking model**: generates a delimiter-marker trace, then predicts the final count.

The main v6 question is:

> Does a de-indexed trace with repeated `<Sep>` tokens still induce item-specific retrieval and useful count states, or was the v2 trace largely helped by explicit numeric index tokens?

### Non-thinking condition

The model sees a fixed-length noisy sequence and directly predicts the final count after `<Ans>`:

```text
<BOS> <N1> ... <A> ... <N2> ... <B> ... <C> ... <N7> <Ans> <3> <EOS>
```

At test time, give the prefix through `<Ans>`:

```text
<BOS> <N1> ... <A> ... <N2> ... <B> ... <C> ... <N7> <Ans>
```

The model predicts the next token. Accuracy is exact match between the predicted numeric token and the gold count token.

### Thinking condition with fixed separator

The model sees the same kind of noisy sequence, then a separator-delimited trace, then the final answer:

```text
<BOS> <N1> ... <A> ... <N2> ... <B> ... <C> ... <N7>
<Think/> <Sep> <A> <Sep> <B> <Sep> <C> </Think> <Ans> <3> <EOS>
```

At test time, give the prefix through the opening think token:

```text
<BOS> <N1> ... <A> ... <N2> ... <B> ... <C> ... <N7> <Think/>
```

The model should greedily generate:

```text
<Sep> marker_1 <Sep> marker_2 ... <Sep> marker_n </Think> <Ans> <n> <EOS>
```

Accuracy is still computed only from the numeric token generated after `<Ans>`.

The trace must enumerate countable markers in left-to-right prompt order. If the prompt contains markers `<A>`, `<B>`, `<C>` at the needle positions, the trace is:

```text
<Sep> <A> <Sep> <B> <Sep> <C>
```

If the prompt contains repeated markers, the trace repeats the exact marker types:

```text
<BOS> ... <C> ... <C> ... <C>
<Think/> <Sep> <C> <Sep> <C> <Sep> <C> </Think> <Ans> <3> <EOS>
```

---

## 1. Token vocabulary

Use a hand-built integer vocabulary. Do not use BPE or a pretrained tokenizer.

### Special tokens

```text
<PAD>
<BOS>
<EOS>
<Ans>
<Think/>
</Think>
<Sep>
```

`<Sep>` is the fixed delimiter used before every trace marker. It is the only trace boundary token. It must not encode the item index.

### Noise tokens

```text
<N0>, <N1>, ..., <N63>
```

There are exactly 64 noise-token types.

### Countable marker tokens

```text
<A>, <B>, <C>, <D>, <E>, <F>, <G>, <H>, <I>, <J>
```

There are exactly 10 marker-token types. Every occurrence of any of these tokens in the prompt body counts as one needle.

### Numeric answer tokens

```text
<1>, <2>, ..., <10>
```

In v6 these numeric tokens are used **only** for final count answers. They are not used inside the thinking trace.

`<10>` is a single token.

### Expected vocabulary size

```text
7 special + 64 noise + 10 markers + 10 numbers = 91 tokens
```

Save the vocabulary to:

```text
run_dir/vocab.json
```

---

## 2. Base data generation

A base example consists only of the fixed-length prompt body and metadata. Both the non-thinking and thinking renderers should use the same base example.

### Config defaults

```yaml
seq_len: 256              # number of prompt-body tokens, excluding <BOS> and completion tokens
noise_vocab_size: 64
marker_vocab_size: 10
min_count: 1
max_count: 10
train_steps: 20000
batch_size: 128
eval_every: 500
log_every: 50
test_examples_per_count: 1000
val_examples_per_count: 200
seed: 1234
```

Debug config:

```yaml
seq_len: 64
train_steps: 200
batch_size: 32
eval_every: 50
log_every: 10
test_examples_per_count: 20
val_examples_per_count: 20
seed: 1234
```

### Sampling rule

For each base example:

1. Sample needle count `n` uniformly from `{1, 2, ..., 10}`.
2. Sample `n` unique positions uniformly without replacement from `range(seq_len)`.
3. Sort these positions for trace construction.
4. For each selected position, sample a marker token uniformly from the 10 marker-token types, independently with replacement.
5. For every non-needle position, sample a noise token uniformly from the 64 noise-token types, independently with replacement.
6. Insert the sampled marker tokens at the sampled needle positions.
7. Gold count is `n`.
8. Store the exact metadata.

All of the following must be uniform:

- needle count;
- needle positions;
- marker-token type at each needle;
- left-to-right marker sequence induced by sampled positions;
- noise-token identity at every non-needle position.

### Metadata schema

```python
@dataclass
class BaseExample:
    seq_tokens: list[str]              # length == seq_len
    count: int                         # 1..10
    needle_positions: list[int]        # sorted ascending, length == count
    needle_markers: list[str]          # marker tokens in left-to-right order, length == count
    seed: int | None = None
```

Validation:

```python
assert len(seq_tokens) == seq_len
assert count == len(needle_positions) == len(needle_markers)
assert all(seq_tokens[pos] == marker for pos, marker in zip(needle_positions, needle_markers))
assert sum(tok in marker_vocab for tok in seq_tokens) == count
```

---

## 3. Rendering

### 3.1 Non-thinking rendering

Given a base example:

```text
<BOS> seq_tokens <Ans> <count> <EOS>
```

Example:

```text
<BOS> <N1> <A> <N9> <B> <N2> <C> <N7> <Ans> <3> <EOS>
```

The supervised completion begins at the count token after `<Ans>`.

### 3.2 Thinking rendering: separator trace

Given the same base example:

```text
<BOS> seq_tokens <Think/> <Sep> marker_1 <Sep> marker_2 ... <Sep> marker_n </Think> <Ans> <n> <EOS>
```

Example:

```text
<BOS> <N1> <A> <N9> <B> <N2> <C> <N7>
<Think/> <Sep> <A> <Sep> <B> <Sep> <C> </Think> <Ans> <3> <EOS>
```

The trace must enumerate markers in left-to-right order of prompt-body needle positions.

The supervised completion begins immediately after `<Think/>`, i.e. at the first `<Sep>` token for all examples.

### 3.3 Important v6 invariant

There must be no numeric index tokens in the trace. This is invalid in v6:

```text
<Think/> <1> <A> <2> <B> <3> <C> </Think>
```

This is valid:

```text
<Think/> <Sep> <A> <Sep> <B> <Sep> <C> </Think>
```

---

## 4. Model

Train two separate decoder-only Transformer language models from scratch:

```text
model_non_thinking
model_thinking_sep_trace
```

They must have the same architecture and vocabulary. They differ only in rendered training sequence and evaluation protocol.

Recommended implementation: Hugging Face `GPT2LMHeadModel` with `GPT2Config`, random initialization, and the custom integer vocabulary above. Do not load pretrained weights.

Use the v2 GPT-2-style learned absolute position architecture. Do not use RoPE, RMSNorm, SwiGLU, ALiBi, pretrained tokenizers, or pretrained weights.

Default model config:

```yaml
architecture: gpt2_lm_head
position_embedding: learned_absolute
n_layer: 4
n_head: 4
n_embd: 256
n_positions: 320          # must exceed max rendered sequence length
activation_function: gelu_new
resid_pdrop: 0.0
embd_pdrop: 0.0
attn_pdrop: 0.0
tie_word_embeddings: true
```

Maximum rendered length:

```text
non-thinking: 1 + seq_len + 1 + 1 + 1 = seq_len + 4
thinking:     1 + seq_len + 1 + 2 * max_count + 1 + 1 + 1 + 1 = seq_len + 2 * max_count + 6
```

For `seq_len = 256` and `max_count = 10`, the thinking length is `282`, so `n_positions = 320` is sufficient.

Debug model config:

```yaml
n_layer: 2
n_head: 2
n_embd: 128
n_positions: 128          # valid only with debug seq_len = 64
activation_function: gelu_new
resid_pdrop: 0.0
embd_pdrop: 0.0
attn_pdrop: 0.0
```

---

## 5. Training objective

Use standard causal next-token prediction, but mask out the random prompt-body prefix. The point is not to model the noise distribution; the point is to learn the counting or trace computation.

Use `labels` with `-100` for ignored positions.

### 5.1 Non-thinking labels

Rendered tokens:

```text
<BOS> seq_tokens <Ans> <n> <EOS>
```

Loss should be applied to:

```text
<n>, <EOS>
```

All earlier positions should have label `-100`.

In Hugging Face-style labels, the label value should be present at the token position being predicted. Therefore, the label at the `<n>` token position is `<n>`, which is predicted from the hidden state at `<Ans>`.

### 5.2 Thinking labels

Rendered tokens:

```text
<BOS> seq_tokens <Think/> <Sep> marker_1 <Sep> marker_2 ... <Sep> marker_n </Think> <Ans> <n> <EOS>
```

Loss should be applied to:

```text
<Sep>, marker_1, <Sep>, marker_2, ..., <Sep>, marker_n, </Think>, <Ans>, <n>, <EOS>
```

All earlier positions, including `<BOS>`, prompt-body tokens, and `<Think/>`, should have label `-100`.

This means the thinking model is trained to generate the full separator trace and then the final answer.

### 5.3 Comparable final-answer loss

Because the thinking model has many more supervised tokens than the non-thinking model, raw completion loss is not directly comparable.

During evaluation, always report:

1. `eval_completion_loss`: masked loss over all supervised completion tokens for that model.
2. `eval_final_answer_loss`: cross-entropy on only the final count token after `<Ans>`.

For the thinking model, `eval_final_answer_loss` should be computed under teacher forcing with the gold separator trace prefix.

---

## 6. Evaluation v6

There is no ID/OOD split in v6. Train and test are independent samples from the same generator distribution.

The test set must be exactly balanced by gold count:

```text
count = 1: test_examples_per_count examples
count = 2: test_examples_per_count examples
...
count = 10: test_examples_per_count examples
```

Define three count bins:

```python
low  = {1, 2, 3}
mid  = {4, 5, 6}
high = {7, 8, 9, 10}
```

Report metrics by exact count and by bin.

### 6.1 Non-thinking evaluation

For each base example, construct the prefix:

```text
<BOS> seq_tokens <Ans>
```

Run one forward pass. Take logits at the final prefix position, i.e. the hidden state at `<Ans>`.

Restrict logits to numeric token IDs:

```text
<1>, <2>, ..., <10>
```

Prediction:

```python
pred_count = argmax(logits_at_ans[numeric_token_ids])
```

Accuracy:

```python
pred_count == gold_count
```

Also compute final-answer cross-entropy:

```python
CE(logits_at_ans over numeric_token_ids, gold_count)
```

Do not use free-form generation for the non-thinking model.

### 6.2 Thinking generated-trace evaluation

For each base example, construct the prefix:

```text
<BOS> seq_tokens <Think/>
```

Then greedily decode up to:

```python
max_new_tokens = 2 * max_count + 4
```

The expected generated suffix is:

```text
<Sep> marker_1 <Sep> marker_2 ... <Sep> marker_n </Think> <Ans> <n> <EOS>
```

Stop early if `<EOS>` is generated.

Parse the first numeric token immediately after the first generated `<Ans>`.

Cases:

- If `<Ans>` is generated and the next generated token is one of `<1>` ... `<10>`, use it as `pred_count`.
- If `<Ans>` is missing, mark the example as invalid and incorrect.
- If `<Ans>` is present but the next token is not a valid numeric token, mark the example as invalid and incorrect.

Primary accuracy:

```python
pred_count == gold_count
```

Trace metrics:

```text
invalid_rate
ans_generated_rate
think_close_generated_rate
trace_exact_match_rate
trace_marker_precision
trace_marker_recall
trace_delimiter_count_accuracy
premature_close_rate
missing_close_rate
```

Definitions:

```text
trace_exact_match_rate:
  generated trace tokens before </Think> exactly equal:
  <Sep> marker_1 <Sep> marker_2 ... <Sep> marker_n

trace_marker_precision:
  fraction of generated marker tokens in the trace that match the gold marker sequence in order.

trace_marker_recall:
  fraction of gold markers recovered in the generated trace in correct left-to-right order.

trace_delimiter_count_accuracy:
  whether the number of generated <Sep> tokens before </Think> equals gold count n.

premature_close_rate:
  model emits </Think> before producing n marker items.

missing_close_rate:
  model fails to emit </Think> before max_new_tokens or <EOS>.
```

There is no `trace_index_accuracy` in v6, because there are no index tokens.

The final paper figure should still use numeric-answer accuracy as the main metric.

### 6.3 Oracle-trace final-readout evaluation

In addition to free-run accuracy, compute teacher-forced final-answer accuracy and loss under the gold separator trace.

Prefix:

```text
<BOS> seq_tokens <Think/> <Sep> marker_1 ... <Sep> marker_n </Think> <Ans>
```

Evaluate the next-token logits restricted to numeric tokens `<1>` ... `<10>`.

Report:

```text
oracle_trace_final_accuracy
oracle_trace_final_answer_loss
```

This isolates final count readout from trace generation quality.

### 6.4 Teacher-forced loss evaluation for thinking model

Compute teacher-forced losses on the gold rendered thinking sequence:

```text
eval_completion_loss
eval_trace_loss
eval_final_answer_loss
```

`eval_trace_loss` should include the generated separator and marker tokens, plus `</Think>` and `<Ans>` if convenient. `eval_final_answer_loss` must only include the final numeric count token.

---

## 7. Training loop and checkpoints

Train both models in the same script and evaluate them at the same steps.

Suggested file:

```text
run_v6_experiment.py
```

At every training step:

1. Generate or sample a batch of base examples.
2. Render one batch for non-thinking.
3. Render the same base examples for separator-trace thinking.
4. Update `model_non_thinking` on non-thinking labels.
5. Update `model_thinking_sep_trace` on separator-trace thinking labels.
6. Log train losses.

Use separate optimizers.

Default optimizer:

```yaml
optimizer: AdamW
learning_rate: 3.0e-4
betas: [0.9, 0.95]
weight_decay: 0.1
grad_clip_norm: 1.0
warmup_steps: 500
lr_schedule: cosine
```

At `eval_every` steps:

1. Evaluate both models on the same fixed test set.
2. Save metrics by exact count and by bin.
3. Save a checkpoint.
4. Optionally run lightweight probe evaluation at selected steps.

Use fixed eval/probe/attention pools generated once per run. Do not resample eval data at every checkpoint.

Suggested run directory:

```text
runs/v6_separator_trace_seed1234/
  config.yaml
  vocab.json
  data/
    eval_pool.jsonl
    probe_train_pool.jsonl
    probe_test_pool.jsonl
    attention_pool.jsonl
  metrics_train.csv
  metrics_eval_by_count.csv
  metrics_eval_by_bin.csv
  checkpoints/
    step_000500/
    step_001000/
    ...
  plots/
  probes/
  attention/
```

---

## 8. Metrics to save

Use tidy CSV format. Each row should include enough metadata to plot without re-running evaluation.

### `metrics_train.csv`

Columns:

```text
step, model_type,
train_loss, train_completion_loss, train_trace_loss, train_final_answer_loss,
learning_rate
```

`model_type` is one of:

```text
non_thinking
thinking_sep_trace
```

For non-thinking, `train_trace_loss` should be `NaN`.

For training, `train_final_answer_loss` can be computed on the current batch under teacher forcing.

### `metrics_eval_by_count.csv`

Columns:

```text
step, model_type, eval_mode, count, accuracy, n_examples, invalid_rate,
eval_completion_loss, eval_trace_loss, eval_final_answer_loss,
mae, under_rate, over_rate,
trace_exact_match_rate, trace_marker_precision, trace_marker_recall,
trace_delimiter_count_accuracy,
premature_close_rate, missing_close_rate, ans_generated_rate, think_close_generated_rate
```

For non-thinking, trace metrics should be `NaN`.

For thinking, use two `eval_mode` values:

```text
generated_trace
oracle_trace_final_readout
```

### `metrics_eval_by_bin.csv`

Columns:

```text
step, model_type, eval_mode, count_bin, accuracy, n_examples, invalid_rate,
eval_completion_loss, eval_trace_loss, eval_final_answer_loss,
mae, under_rate, over_rate,
trace_exact_match_rate, trace_marker_precision, trace_marker_recall,
trace_delimiter_count_accuracy,
premature_close_rate, missing_close_rate, ans_generated_rate, think_close_generated_rate
```

`count_bin` is `low`, `mid`, or `high`.

---

## 9. Required plots

Generate publication-readable plots into:

```text
run_dir/plots/
```

Use consistent titles, legends, axis labels, and step units.

### 9.1 Training loss over steps

File:

```text
plots/train_loss_vs_step.png
```

Content:

- x-axis: training step;
- y-axis: training loss;
- lines:
  - non-thinking train masked loss;
  - thinking separator-trace train masked completion loss;
  - optionally final-answer train loss for both models.

Important note in title or caption: raw completion loss is not directly comparable across model types because the thinking model has a longer supervised completion.

### 9.2 Test final-answer loss over steps

File:

```text
plots/eval_final_answer_loss_vs_step.png
```

Content:

- x-axis: training step;
- y-axis: final-answer cross-entropy;
- lines:
  - non-thinking;
  - thinking separator-trace under teacher-forced gold trace.

This is the most comparable loss curve.

### 9.3 Test accuracy over steps by count bin

File:

```text
plots/eval_accuracy_by_bin_vs_step.png
```

Content:

- x-axis: training step;
- y-axis: exact count accuracy;
- separate curves for:
  - non-thinking low;
  - non-thinking mid;
  - non-thinking high;
  - thinking generated-trace low;
  - thinking generated-trace mid;
  - thinking generated-trace high;
  - optionally thinking oracle-trace final-readout low/mid/high as dashed lines.

This is the main behavioral figure for v6.

### 9.4 Final checkpoint accuracy by exact count

File:

```text
plots/final_accuracy_by_count.png
```

Content:

- x-axis: gold count 1..10;
- y-axis: exact count accuracy;
- curves or grouped bars:
  - non-thinking;
  - thinking generated-trace;
  - thinking oracle-trace final-readout.

### 9.5 Heatmap: accuracy by count and training step

Files:

```text
plots/accuracy_heatmap_by_count_and_step_non_thinking.png
plots/accuracy_heatmap_by_count_and_step_thinking_generated_trace.png
plots/accuracy_heatmap_by_count_and_step_thinking_oracle_trace.png
```

- x-axis: training step;
- y-axis: gold count 1..10;
- cell value: accuracy.

### 9.6 Trace quality plots

Files:

```text
plots/trace_exact_by_count.png
plots/trace_marker_precision_recall_by_count.png
plots/trace_delimiter_count_accuracy_by_count.png
plots/premature_close_missing_close_by_count.png
```

These are v6-specific and replace v2's index-accuracy plots.

---

## 10. Hidden-state probe analysis

Implement this after the training/evaluation loop works.

Use teacher-forced sequences for probes so that token positions are known exactly. Free-run generation is too unstable for aligned probes, especially early in training.

### 10.1 Probe data

Create fixed probe datasets:

```yaml
probe_train_examples_per_count: 500
probe_test_examples_per_count: 500
```

Sample them from the same generator distribution, balanced by count.

### 10.2 Hidden states to collect

Run models with:

```python
output_hidden_states=True
```

Collect residual-stream hidden states from every layer, including embedding output if available.

For each example, store activations at the following anchors.

#### Non-thinking anchors

Rendered teacher-forced sequence:

```text
<BOS> seq_tokens <Ans> <n> <EOS>
```

Anchors:

```text
ans_token: hidden state at <Ans>, which predicts final count <n>
last_prompt_token: hidden state at final prompt-body token before <Ans>
prompt_marker_k: hidden state at kth prompt needle marker
```

Labels:

```text
final_count = n
```

#### Thinking anchors

Rendered teacher-forced sequence:

```text
<BOS> seq_tokens <Think/> <Sep> marker_1 ... <Sep> marker_n </Think> <Ans> <n> <EOS>
```

Anchors for final-count probes:

```text
think_start: hidden state at <Think/>
think_end: hidden state at </Think>
ans_token: hidden state at <Ans>
pre_ans_token: hidden state immediately before <Ans>, usually </Think>
```

Label:

```text
final_count = n
```

Anchors for prefix-count probes:

```text
pre_sep_k:
  hidden state at token immediately before the kth <Sep> is generated;
  for k = 1, this is <Think/>;
  for k > 1, this is marker_{k-1}.

sep_token_k:
  hidden state at the kth <Sep> token.
  This state predicts marker_k and is the main retrieval-query candidate in v6.

marker_token_k:
  hidden state at generated trace marker_k.

post_marker_k:
  hidden state immediately after generated trace marker_k, if it exists.
```

Labels:

```text
prefix_count = k
```

Important anti-leakage note:

- v6 removes numeric token identity leakage because there are no `<1>`, `<2>`, ..., `<k>` tokens in the trace.
- However, absolute position and trace length still correlate with `k` and `n`.
- Always report position-only and trace-length-only baselines.
- Do not claim a count-vector from probe accuracy alone.

### 10.3 Probe models

Implement both:

1. Multinomial logistic regression classifier for count labels 1..10.
2. Ridge regression probe predicting scalar count, then rounded to nearest integer in 1..10.

Use scikit-learn. Standardize features with train-set mean and variance.

Metrics:

```text
probe_accuracy
probe_mae
probe_r2
```

Report metrics by:

```text
model_type
checkpoint_step
layer
anchor_type
label_type   # final_count or prefix_count
```

### 10.4 Probe baselines

For every probe table, include:

```text
position_only_accuracy
position_only_r2
trace_length_only_accuracy
trace_length_only_r2
token_id_only_accuracy
token_id_only_r2
shuffled_label_accuracy
shuffled_label_r2
```

For v6, `sep_token_k` has constant token id, so token-id-only should not predict prefix count at that anchor. If it does, there is a bug in feature construction.

### 10.5 Probe plots

Save to:

```text
run_dir/probes/
```

Required plots:

```text
probe_final_count_accuracy_heatmap_non_thinking.png
probe_final_count_accuracy_heatmap_thinking_sep_trace.png
probe_prefix_count_accuracy_heatmap_thinking_sep_trace.png
probe_prefix_count_mae_heatmap_thinking_sep_trace.png
probe_accuracy_vs_training_step_ans_token.png
probe_sep_token_prefix_probe_minus_position_baseline.png
```

Heatmap axes:

- x-axis: layer;
- y-axis: anchor type;
- color: probe accuracy or MAE.

Training-step plot:

- x-axis: checkpoint step;
- y-axis: probe accuracy;
- lines for key anchors:
  - non-thinking `<Ans>`;
  - thinking `<Think/>`;
  - thinking `</Think>`;
  - thinking `<Ans>`;
  - thinking `sep_token_k` prefix-count probe.

---

## 11. Attention and retrieval analysis

Implement this after the main metrics and probes work.

Use teacher-forced gold sequences for attention analysis, because retrieval alignment requires exact trace positions.

Run models with:

```python
output_attentions=True
```

Use a manageable attention-analysis set:

```yaml
attention_examples_per_count: 100
```

Balanced by count.

### 11.1 Prompt needle positions

Each base example stores:

```python
needle_positions: list[int]
```

In the rendered sequence, prompt body starts after `<BOS>`, so the rendered token index of prompt needle `j` is:

```python
rendered_needle_pos_j = 1 + needle_positions[j]
```

where `j` is zero-based.

### 11.2 Thinking retrieval attention with separator trace

For each thinking example and each trace item `k`, define the correct prompt needle as the kth needle in left-to-right order.

Useful query positions:

```text
sep_token_k:       position of kth <Sep> token in the trace
marker_token_k:    position of generated marker_k in the trace
pre_sep_k:         token immediately before kth <Sep>
post_marker_k:     token immediately after marker_k, if it exists
```

The most important query anchor is `sep_token_k`, because its hidden state predicts `marker_k`. This replaces v2's `index_token_k` attention analysis.

For each layer and head, construct an attention matrix:

```text
A[k, j] = attention mass from query position for trace item k to rendered prompt needle j
```

Because each needle is a single token in this version, this is just the attention weight from query token to the prompt needle token.

Compute this matrix separately for query anchors:

```text
sep_token_k
marker_token_k
pre_sep_k
post_marker_k
```

Aggregate across examples by averaging.

Retrieval metrics:

```text
diagonal_mass = mean_k A[k, k]
off_diagonal_mass = mean_{k != j} A[k, j]
diagonal_dominance = diagonal_mass / (diagonal_mass + off_diagonal_mass + eps)
correct_top1_rate = fraction where argmax_j A[k, j] == k
needle_attention_mass = sum_j A[k, j]
noise_attention_mass = attention mass to prompt-body non-needle positions
needle_vs_noise_ratio = needle_attention_mass / (noise_attention_mass + eps)
entropy_over_prompt_body
```

Report by:

```text
checkpoint_step
layer
head
query_anchor
count
count_bin
```

### 11.3 Repeated-marker diagnostic

Marker-only traces can be ambiguous when a prompt contains repeated marker types:

```text
... <A> ... <B> ... <A> ...
<Think/> <Sep> <A> <Sep> <B> <Sep> <A> ...
```

For attention retrieval, report metrics on three subsets:

```text
all examples
unique_marker_examples: no marker type repeats inside the example
repeated_marker_examples: at least one marker type repeats
```

The `correct_top1_rate` metric is strongest on `unique_marker_examples`, because identical marker tokens create content-level ambiguity even if position-level retrieval is correct.

### 11.4 Non-thinking attention

For the non-thinking model, there is no item-specific trace. Use the `<Ans>` token as the query position.

For each layer and head, compute:

```text
ans_to_all_needles_mass = sum attention from <Ans> to all prompt needle positions
ans_to_noise_mass = sum attention from <Ans> to prompt-body non-needle positions
needle_vs_noise_ratio = ans_to_all_needles_mass / (ans_to_noise_mass + eps)
attention_entropy_over_prompt_body
```

Also compute whether the top-`n` prompt-body positions by attention contain the true needle positions:

```text
top_n_retrieval_recall = |top_n_positions ∩ needle_positions| / n
```

### 11.5 Attention plots

Save to:

```text
run_dir/attention/
```

Required plots:

```text
attention_thinking_sep_diagonal_dominance_by_layer_head.png
attention_thinking_sep_correct_top1_by_layer_head.png
attention_thinking_sep_needle_mass_by_layer_head.png
attention_nonthinking_ans_needle_mass_by_layer_head.png
attention_nonthinking_topn_recall_by_layer_head.png
attention_matrix_thinking_sep_best_head_low.png
attention_matrix_thinking_sep_best_head_mid.png
attention_matrix_thinking_sep_best_head_high.png
attention_unique_vs_repeated_marker_diagnostic.png
```

For attention matrix plots:

- x-axis: prompt needle index `j`;
- y-axis: trace item index `k`;
- color: average attention mass;
- a near-diagonal pattern indicates item-specific retrieval.

Do not claim attention is causal. Treat attention as a diagnostic to identify retrieval-like heads and possible patching sites.

---

## 12. Optional comparison against v2 checkpoints

If v2 checkpoints or v2 report data are available, add a light comparison section. This is optional and should not block v6.

Compare:

```text
v2 indexed trace:
  <Think/> <1> A <2> B <3> C </Think> <Ans> <3>

v6 separator trace:
  <Think/> <Sep> A <Sep> B <Sep> C </Think> <Ans> <3>
```

Useful comparison metrics:

```text
step_to_99_percent_final_accuracy
step_to_99_percent_trace_exact
top retrieval head correct_top1
top retrieval head diagonal_dominance
prefix-count probe accuracy minus position baseline
oracle-trace final-readout accuracy
```

Interpretation:

- If v6 matches v2 behavior and attention, explicit numeric indices were not necessary for trace-guided retrieval.
- If v6 learns more slowly or has weaker trace exactness, v2 indices were functioning as helpful supervision or scaffolding.
- If v6 still has high probe accuracy but only at position-predictable anchors, do not over-interpret it as counter geometry.

---

## 13. Code structure

Implement modular code. Suggested layout:

```text
src/synthetic_counting_v6/
  __init__.py
  config.py
  vocab.py
  data.py
  render.py
  model.py
  train.py
  eval.py
  generation.py
  probes.py
  attention.py
  plots.py
  report.py
  run_v6_experiment.py
  configs/
    debug.yaml
    main.yaml
  tests/
    test_vocab.py
    test_data_generation.py
    test_rendering.py
    test_eval_parsing.py
    test_probes.py
    test_attention.py
```

### Responsibilities

`vocab.py`

- build token-to-id and id-to-token dictionaries;
- save/load `vocab.json`;
- expose marker IDs, noise IDs, numeric IDs, separator ID, and special IDs.

`data.py`

- generate `BaseExample` objects;
- generate balanced validation/test/probe/attention sets;
- support deterministic seeds.

`render.py`

- render non-thinking and separator-trace thinking sequences;
- create `input_ids`, `attention_mask`, and `labels`;
- return metadata with anchor positions.

`model.py`

- construct random-init GPT2-like decoder-only model;
- expose config.

`train.py`

- one training step for each model;
- optimizer and scheduler;
- checkpoint save/load.

`generation.py`

- greedy generation for separator-trace thinking mode;
- generated-suffix parser;
- robust handling of missing `</Think>`, missing `<Ans>`, invalid count tokens, duplicate markers, and malformed delimiter-marker patterns.

`eval.py`

- non-thinking next-token evaluation at `<Ans>`;
- thinking greedy free-run evaluation from `<Think/>`;
- thinking oracle-trace final-readout evaluation;
- teacher-forced loss evaluation;
- metrics by count and bin.

`probes.py`

- collect hidden states;
- fit logistic/ridge probes;
- compute position/token/trace-length baselines;
- save metrics and heatmaps.

`attention.py`

- collect attentions;
- compute retrieval metrics;
- save attention heatmaps;
- support unique-marker and repeated-marker subsets.

`plots.py`

- generate all required training/evaluation/probe/attention plots.

`report.py`

- generate `report.html` and `report.md` with setup, metrics, plots, and cautious interpretation.

`run_v6_experiment.py`

- parse config;
- create run dir;
- train both models;
- evaluate at checkpoints;
- generate plots;
- run final probe and attention analyses;
- generate report.

---

## 14. CLI behavior

The main script should support:

```bash
python -m synthetic_counting_v6.run_v6_experiment \
  --config src/synthetic_counting_v6/configs/main.yaml \
  --run_dir runs/v6_separator_trace_seed1234
```

Debug run:

```bash
python -m synthetic_counting_v6.run_v6_experiment \
  --config src/synthetic_counting_v6/configs/debug.yaml \
  --run_dir runs/debug_v6
```

The debug run must finish quickly and produce all required metric files and plots, even if the model does not learn well.

---

## 15. Acceptance checks

The implementation is complete when all of the following pass.

### Data and rendering checks

- `seq_tokens` length is exactly `seq_len`.
- Every example has count in `1..10`.
- Count equals the number of marker tokens in the prompt body.
- Needle positions are sorted and match marker tokens in `seq_tokens`.
- Non-thinking rendering exactly matches:

```text
<BOS> seq_tokens <Ans> <n> <EOS>
```

- Thinking rendering exactly matches:

```text
<BOS> seq_tokens <Think/> <Sep> marker_1 ... <Sep> marker_n </Think> <Ans> <n> <EOS>
```

- Thinking rendering never includes numeric trace indices before `</Think>`.
- The number of `<Sep>` tokens in the gold trace equals `count`.
- The final numeric answer token is still `<n>`.

### Label-mask checks

- Non-thinking labels supervise only `<n>` and `<EOS>`.
- Thinking labels supervise:

```text
<Sep>, marker_1, ..., <Sep>, marker_n, </Think>, <Ans>, <n>, <EOS>
```

- Prompt-body tokens and `<Think/>` are masked to `-100`.
- `<Sep>` tokens are supervised in thinking examples.

### Evaluation checks

- Non-thinking test prefix ends at `<Ans>`.
- Non-thinking prediction uses only logits over `<1>` ... `<10>`.
- Thinking generated-trace test prefix ends at `<Think/>`.
- Thinking generation expects delimiter-marker pairs.
- Thinking evaluation parses the first numeric token after generated `<Ans>`.
- Missing `<Ans>` or invalid numeric token is counted as invalid and incorrect.
- Oracle-trace final-readout evaluation uses gold separator trace and measures count logits after `<Ans>`.
- Metrics are reported by exact count and by low/mid/high count bin.

### Plot checks

The following files exist after a run:

```text
plots/train_loss_vs_step.png
plots/eval_final_answer_loss_vs_step.png
plots/eval_accuracy_by_bin_vs_step.png
plots/final_accuracy_by_count.png
plots/accuracy_heatmap_by_count_and_step_non_thinking.png
plots/accuracy_heatmap_by_count_and_step_thinking_generated_trace.png
plots/trace_exact_by_count.png
plots/trace_delimiter_count_accuracy_by_count.png
```

### Probe checks

- Prefix-count probes do not use numeric index-token hidden states because v6 has no trace index tokens.
- Probe metrics are saved by layer and anchor.
- Position-only and trace-length-only baselines are reported.
- For `sep_token_k`, token-id-only baseline should be close to chance for prefix count.
- Probe plots are produced for final-count and prefix-count probes.

### Attention checks

- Thinking attention matrices align trace item index `k` to prompt needle index `j` using `sep_token_k` and other separator-trace anchors.
- Non-thinking attention uses `<Ans>` as the query position.
- Attention metrics are saved by layer/head and count bin.
- Attention metrics are reported for all examples, unique-marker examples, and repeated-marker examples.
- Attention plots are produced.

### Report checks

The report must answer:

1. Did the non-thinking model solve final count prediction?
2. Did the separator-trace thinking model solve generated trace and final count prediction?
3. Did oracle-trace final readout succeed?
4. Did separator-trace thinking produce targeted retrieval attention similar to v2 indexed thinking?
5. Did removal of index tokens reduce prefix-count probe leakage relative to v2?
6. Are any high probe scores explainable by position or trace-length baselines?

---

## 16. Explicit exclusions for v6

Do not implement the following in v6:

- variable sequence length;
- ID/OOD split;
- distractor marker classes;
- query token specifying which marker type to count;
- natural-language prompts;
- JSON output;
- city-score NIAH;
- activation patching;
- steering;
- v4 hidden-state steering grid;
- v5 mixed thinking/non-thinking single model;
- multiple loss-mask ablation regimes;
- pretrained models;
- RoPE or custom architecture changes.

Keep v6 small, deterministic, and directly comparable to v2.

---

## 17. Expected interpretation section

The report conclusion should use one of these categories.

### A. Separator trace matches indexed trace

```text
The separator-trace model reaches similar final accuracy, trace exactness, and targeted retrieval attention as v2 indexed-trace thinking. Numeric index tokens were not necessary for the main trace-guided retrieval behavior.
```

### B. Separator trace works behaviorally but retrieval is weaker

```text
The separator-trace model solves final count prediction but shows slower trace learning, lower trace exactness, or weaker diagonal retrieval attention. Numeric indices may provide useful scaffolded supervision for trace generation or item alignment.
```

### C. Separator trace fails to learn reliable traces

```text
The separator-trace model does not consistently generate delimiter-marker traces before the answer. The indexed v2 trace may have been doing substantial work by giving the model explicit prefix-count structure.
```

### D. Probe-only evidence

```text
Count is linearly decodable at some anchors, but the probe advantage disappears under position or trace-length baselines. Do not claim a count-vector from this result.
```

Use cautious language. v6 can support claims about de-indexed trace behavior and retrieval diagnostics, but it does not by itself establish causal count-vector steering.
