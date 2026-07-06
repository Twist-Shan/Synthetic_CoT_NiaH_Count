# Codex Prompt: Synthetic NIAH-Style Counting v2

Implement a single controlled synthetic counting experiment for small decoder-only Transformers trained from scratch.

This is not a natural-language NIAH benchmark. It is the minimal symbolic bridge experiment:

- fixed-length token sequence;
- 64 noise-token types;
- 10 countable marker-token types;
- count range 1 to 10;
- two separately trained models:
  - non-thinking model;
  - thinking/trace model;
- evaluation grouped by needle-count bin: low, mid, high;
- curves over training step for loss and accuracy;
- hidden-state probe analysis;
- attention/retrieval analysis.

Do **not** implement ID/OOD splits, distractors, natural language prompts, JSON, city-score records, query templates, or variable sequence length in this version.

---

## 0. Scientific question

We want to compare two ways a small Transformer can learn sparse counting.

### Non-thinking condition

The model sees a fixed-length noisy sequence and directly predicts the final count after `<Ans>`:

```text
<BOS> <N1> ... <A> ... <N2> ... <B> <C> ... <N7> <Ans> <3> <EOS>
```

At test time, give the prefix through `<Ans>`:

```text
<BOS> <N1> ... <A> ... <N2> ... <B> <C> ... <N7> <Ans>
```

The model predicts the next token. Accuracy is exact match between the predicted numeric token and the gold count token.

### Thinking condition

The model sees the same kind of noisy sequence, then an explicit indexed trace, then the final answer:

```text
<BOS> <N1> ... <A> ... <N2> ... <B> <C> ... <N7>
<Think/> <1> <A> <2> <B> <3> <C> </Think> <Ans> <3> <EOS>
```

At test time, give the prefix through the opening think token:

```text
<BOS> <N1> ... <A> ... <N2> ... <B> <C> ... <N7> <Think/>
```

The model should greedily generate the trace, `</Think>`, `<Ans>`, the final number, and optionally `<EOS>`. Accuracy is still computed only from the numeric token generated after `<Ans>`.

The trace should enumerate countable markers in left-to-right prompt order. If the prompt contains markers `<A>`, `<B>`, `<C>` at the needle positions, the trace is:

```text
<1> <A> <2> <B> <3> <C>
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
```

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

### Numeric tokens

```text
<1>, <2>, ..., <10>
```

Use the same numeric tokens both for trace indices and final count answers. `<10>` is a single token.

### Expected vocabulary size

```text
6 special + 64 noise + 10 markers + 10 numbers = 90 tokens
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

Also implement a debug config:

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
- left-to-right marker sequence induced by the sampled positions;
- noise-token identity at every non-needle position.

### Metadata schema

Each base example should have:

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

### 3.2 Thinking rendering

Given the same base example:

```text
<BOS> seq_tokens <Think/> <1> marker_1 <2> marker_2 ... <n> marker_n </Think> <Ans> <n> <EOS>
```

Example:

```text
<BOS> <N1> <A> <N9> <B> <N2> <C> <N7>
<Think/> <1> <A> <2> <B> <3> <C> </Think> <Ans> <3> <EOS>
```

The trace must enumerate markers in left-to-right order of prompt-body needle positions.

The supervised completion begins immediately after `<Think/>`, i.e. at `<1>` for all examples.

---

## 4. Model

Train two separate decoder-only Transformer language models from scratch:

```text
model_non_thinking
model_thinking
```

They must have the same architecture and vocabulary. They differ only in the rendered training sequence and evaluation protocol.

Recommended implementation: Hugging Face `GPT2LMHeadModel` with `GPT2Config`, but with random initialization and the custom integer vocabulary above. Do not load pretrained weights.

Default model config:

```yaml
n_layer: 4
n_head: 4
n_embd: 256
n_positions: 320          # must exceed max rendered sequence length
activation_function: gelu_new
resid_pdrop: 0.0
embd_pdrop: 0.0
attn_pdrop: 0.0
```

Maximum rendered length:

```text
non-thinking: 1 + seq_len + 1 + 1 + 1 = seq_len + 4
thinking:     1 + seq_len + 1 + 2 * max_count + 1 + 1 + 1 + 1 = seq_len + 2 * max_count + 6
```

For `seq_len = 256` and `max_count = 10`, the thinking length is `282`, so `n_positions = 320` is sufficient.

---

## 5. Training objective

Use standard causal next-token prediction, but mask out the random prompt-body prefix. The point is not to model the noise distribution; the point is to learn the counting/trace computation.

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
<BOS> seq_tokens <Think/> <1> marker_1 ... <n> marker_n </Think> <Ans> <n> <EOS>
```

Loss should be applied to:

```text
<1>, marker_1, <2>, marker_2, ..., <n>, marker_n, </Think>, <Ans>, <n>, <EOS>
```

All earlier positions, including `<BOS>`, prompt-body tokens, and `<Think/>`, should have label `-100`.

This means the thinking model is trained to generate the full trace and then the final answer.

### 5.3 Comparable final-answer loss

Because the thinking model has many more supervised tokens than the non-thinking model, raw completion loss is not directly comparable.

During evaluation, always report:

1. `eval_completion_loss`: masked loss over all supervised completion tokens for that model.
2. `eval_final_answer_loss`: cross-entropy on only the final count token after `<Ans>`.

For the thinking model, `eval_final_answer_loss` should be computed under teacher forcing with the gold trace prefix.

---

## 6. Evaluation v2

There is no ID/OOD split in this version. Train and test are independent samples from the same generator distribution.

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

### 6.2 Thinking evaluation

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
<1> marker_1 <2> marker_2 ... <n> marker_n </Think> <Ans> <n> <EOS>
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

Also report:

```text
invalid_rate
ans_generated_rate
trace_exact_match_rate
trace_marker_precision
trace_marker_recall
trace_index_accuracy
```

The final paper figure should still use numeric-answer accuracy as the main metric.

### 6.3 Teacher-forced loss evaluation for thinking model

In addition to free-run accuracy, compute teacher-forced losses on the gold rendered thinking sequence:

```text
eval_completion_loss
eval_final_answer_loss
```

This gives stable loss curves even when free-run trace generation is poor early in training.

---

## 7. Training loop and checkpoints

Train both models in the same script and evaluate them at the same steps.

Suggested file:

```text
run_v2_experiment.py
```

At every training step:

1. Generate or sample a batch of base examples.
2. Render one batch for non-thinking.
3. Render the same base examples for thinking.
4. Update `model_non_thinking` on non-thinking labels.
5. Update `model_thinking` on thinking labels.
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

Suggested run directory:

```text
runs/v2_marker_trace_seed1234/
  config.yaml
  vocab.json
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
step, model_type, train_loss, train_completion_loss, train_final_answer_loss, learning_rate
```

`model_type` is one of:

```text
non_thinking
thinking
```

For training, `train_final_answer_loss` can be computed on the current batch under teacher forcing.

### `metrics_eval_by_count.csv`

Columns:

```text
step, model_type, count, accuracy, n_examples, invalid_rate,
eval_completion_loss, eval_final_answer_loss,
mae, under_rate, over_rate,
trace_exact_match_rate, trace_marker_precision, trace_marker_recall, trace_index_accuracy
```

For non-thinking, trace metrics should be `NaN`.

### `metrics_eval_by_bin.csv`

Columns:

```text
step, model_type, count_bin, accuracy, n_examples, invalid_rate,
eval_completion_loss, eval_final_answer_loss,
mae, under_rate, over_rate,
trace_exact_match_rate, trace_marker_precision, trace_marker_recall, trace_index_accuracy
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
  - thinking train masked completion loss;
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
  - thinking teacher-forced gold trace.

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
  - thinking low;
  - thinking mid;
  - thinking high.

This is the main behavioral figure for v2.

### 9.4 Final checkpoint accuracy by exact count

File:

```text
plots/final_accuracy_by_count.png
```

Content:

- x-axis: gold count 1..10;
- y-axis: exact count accuracy;
- two model curves or grouped bars:
  - non-thinking;
  - thinking.

### 9.5 Heatmap: accuracy by count and training step

File:

```text
plots/accuracy_heatmap_by_count_and_step_{model_type}.png
```

Create one for each model.

- x-axis: training step;
- y-axis: gold count 1..10;
- cell value: accuracy.

This should make the low/mid/high count effect visible.

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
```

Labels:

```text
final_count = n
```

#### Thinking anchors

Rendered teacher-forced sequence:

```text
<BOS> seq_tokens <Think/> <1> marker_1 ... <n> marker_n </Think> <Ans> <n> <EOS>
```

Anchors for final-count probes:

```text
think_start: hidden state at <Think/>
think_end: hidden state at </Think>
ans_token: hidden state at <Ans>
```

Label:

```text
final_count = n
```

Anchors for prefix-count probes:

Use two sets:

1. `pre_index_k`: the hidden state at the token immediately before numeric index `<k>` is generated.
   - for `k = 1`, this is `<Think/>`;
   - for `k > 1`, this is the previous trace marker token `marker_{k-1}`.
2. `post_marker_k`: the hidden state at `marker_k`.

Labels:

```text
prefix_count = k
```

Important anti-leakage rule:

Do **not** use the hidden state at numeric token `<k>` itself as a prefix-count probe feature, because the token identity directly reveals the label.

The `post_marker_k` anchor is useful but partly influenced by the preceding numeric index. Report it separately from `pre_index_k`.

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

### 10.4 Probe plots

Save to:

```text
run_dir/probes/
```

Required plots:

```text
probe_final_count_accuracy_heatmap_non_thinking.png
probe_final_count_accuracy_heatmap_thinking.png
probe_prefix_count_accuracy_heatmap_thinking.png
probe_prefix_count_mae_heatmap_thinking.png
probe_accuracy_vs_training_step_ans_token.png
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
  - thinking `<Ans>`.

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

### 11.2 Thinking retrieval attention

For each thinking example and each trace item `k`, define the correct prompt needle as the kth needle in left-to-right order.

Useful query positions:

```text
index_token_k: position of numeric token <k> in the trace
marker_token_k: position of marker_k in the trace
pre_index_k: token immediately before <k>
```

For each layer and head, construct an attention matrix:

```text
A[k, j] = attention mass from query position for trace item k to rendered prompt needle j
```

Because each needle is a single token in this version, this is just the attention weight from query token to the prompt needle token.

Compute this matrix separately for query anchors:

```text
index_token_k
marker_token_k
pre_index_k
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

### 11.3 Non-thinking attention

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

### 11.4 Attention plots

Save to:

```text
run_dir/attention/
```

Required plots:

```text
attention_thinking_diagonal_dominance_by_layer_head.png
attention_thinking_correct_top1_by_layer_head.png
attention_nonthinking_ans_needle_mass_by_layer_head.png
attention_nonthinking_topn_recall_by_layer_head.png
attention_matrix_thinking_best_head_low.png
attention_matrix_thinking_best_head_mid.png
attention_matrix_thinking_best_head_high.png
```

For the attention matrix plots:

- x-axis: prompt needle index `j`;
- y-axis: trace item index `k`;
- color: average attention mass;
- a near-diagonal pattern indicates item-specific retrieval.

Do not claim attention is causal. Treat attention as a diagnostic to identify retrieval-like heads and possible patching sites.

---

## 12. Code structure

Implement modular code. Suggested layout:

```text
synthetic_counting_v2/
  __init__.py
  config.py
  vocab.py
  data.py
  render.py
  model.py
  train.py
  eval.py
  probes.py
  attention.py
  plots.py
  run_v2_experiment.py
  configs/
    debug.yaml
    main.yaml
  tests/
    test_vocab.py
    test_data_generation.py
    test_rendering.py
    test_eval_parsing.py
```

### Responsibilities

`vocab.py`

- build token-to-id and id-to-token dictionaries;
- save/load `vocab.json`;
- expose marker IDs, noise IDs, numeric IDs, and special IDs.

`data.py`

- generate `BaseExample` objects;
- generate balanced validation/test/probe sets;
- support deterministic seeds.

`render.py`

- render non-thinking and thinking sequences;
- create `input_ids`, `attention_mask`, and `labels`;
- return metadata with anchor positions.

`model.py`

- construct random-init GPT2-like decoder-only model;
- expose config.

`train.py`

- one training step for each model;
- optimizer and scheduler;
- checkpoint save/load.

`eval.py`

- non-thinking next-token evaluation at `<Ans>`;
- thinking greedy free-run evaluation from `<Think/>`;
- teacher-forced loss evaluation;
- metrics by count and bin.

`probes.py`

- collect hidden states;
- fit logistic/ridge probes;
- save metrics and heatmaps.

`attention.py`

- collect attentions;
- compute retrieval metrics;
- save attention heatmaps.

`plots.py`

- generate all required training/evaluation plots.

`run_v2_experiment.py`

- parse config;
- create run dir;
- train both models;
- evaluate at checkpoints;
- generate plots;
- run final probe and attention analyses.

---

## 13. CLI behavior

The main script should support:

```bash
python -m synthetic_counting_v2.run_v2_experiment \
  --config synthetic_counting_v2/configs/main.yaml \
  --run_dir runs/v2_marker_trace_seed1234
```

Debug run:

```bash
python -m synthetic_counting_v2.run_v2_experiment \
  --config synthetic_counting_v2/configs/debug.yaml \
  --run_dir runs/debug_v2
```

The debug run must finish quickly and produce all required metric files and plots, even if the model does not learn well.

---

## 14. Acceptance checks

The implementation is complete when all of the following pass.

### Data and rendering checks

- `seq_tokens` length is exactly `seq_len`.
- Every example has count in `1..10`.
- Count equals the number of marker tokens in the prompt body.
- Needle positions are sorted and match the marker tokens in `seq_tokens`.
- Non-thinking rendering exactly matches:

```text
<BOS> seq_tokens <Ans> <n> <EOS>
```

- Thinking rendering exactly matches:

```text
<BOS> seq_tokens <Think/> <1> marker_1 ... <n> marker_n </Think> <Ans> <n> <EOS>
```

### Evaluation checks

- Non-thinking test prefix ends at `<Ans>`.
- Non-thinking prediction uses only logits over `<1>` ... `<10>`.
- Thinking test prefix ends at `<Think/>`.
- Thinking evaluation uses greedy generation and parses the first numeric token after generated `<Ans>`.
- Missing `<Ans>` or invalid numeric token is counted as invalid and incorrect.
- Metrics are reported by exact count and by low/mid/high count bin.

### Plot checks

The following files exist after a run:

```text
plots/train_loss_vs_step.png
plots/eval_final_answer_loss_vs_step.png
plots/eval_accuracy_by_bin_vs_step.png
plots/final_accuracy_by_count.png
plots/accuracy_heatmap_by_count_and_step_non_thinking.png
plots/accuracy_heatmap_by_count_and_step_thinking.png
```

### Probe checks

- Probe features do not use hidden states at numeric index tokens `<k>` themselves for prefix-count labels.
- Probe metrics are saved by layer and anchor.
- Probe plots are produced for final-count and prefix-count probes.

### Attention checks

- Thinking attention matrices align trace item index `k` to prompt needle index `j`.
- Non-thinking attention uses `<Ans>` as the query position.
- Attention metrics are saved by layer/head and count bin.
- Attention plots are produced.

---

## 15. Explicit exclusions for v2

Do not implement the following in this version:

- variable sequence length;
- ID/OOD split;
- distractor marker classes;
- query token specifying which marker type to count;
- natural-language prompts;
- JSON output;
- city-score NIAH;
- activation patching;
- steering;
- multiple loss-mask ablation regimes;
- pretrained models.

Keep the experiment small, deterministic, and easy to inspect.
