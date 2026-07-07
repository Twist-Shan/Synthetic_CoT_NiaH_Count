# Codex Prompt: Synthetic NIAH Counting v3, no loss-mask ablation

Refactor the current synthetic counting pipeline into **v3-no-loss-ablation** and run the experiment suite.

This version intentionally removes the old Round 2 loss-mask ablation. Use only one fixed objective per model type:

- `non_thinking`: train final count readout from `<Ans>`.
- `thinking`: train indexed trace generation plus final count readout from `<Think/>`.

The goal is to focus compute on hard length/noise evaluation, corrupted-trace diagnostics, and mechanistic analysis.

The current v2 result was behaviorally saturated at `seq_len = 256`: both non-thinking and thinking reached final count accuracy near 1.0. The useful v2 signal was mechanistic: thinking developed a strong trace-indexed retrieval pattern, while non-thinking looked more like broad aggregation at `<Ans>`. v3 should make the task harder by testing longer noise sequences and should test whether the trace route is robust, interpretable, and causally relevant.

Do **not** implement realistic NIAH, natural language, JSON, city-score records, distractors, query templates, or loss-policy sweeps in this run.

---

## 0. Core scientific question

We want to test whether explicit indexed think traces create a different and more robust sparse-counting route than direct answering.

The v3-no-loss-ablation suite should answer three questions:

1. **Hard evaluation:** when training length is fixed at 256, does the thinking model generalize better to longer noise sequences such as 512 and 1024?
2. **Corrupted-trace readout:** when the trace conflicts with the prompt, does the final answer follow the prompt count, the trace pair count, the last index token, the max index token, or the marker count?
3. **Mechanistic evidence:** are hidden states and attention heads merely diagnostic, or do they causally mediate retrieval and counting?

Use only two main training conditions:

```text
non_thinking
thinking
```

No loss-mask ablation. No `full_lm`, `final_heavy`, `trace_only`, or other policy sweep.

---

## 1. Data and vocabulary

### 1.1 Vocabulary

Use a hand-built integer vocabulary. Do not use BPE and do not use a pretrained tokenizer.

Special tokens:

```text
<PAD>
<BOS>
<EOS>
<Ans>
<Think/>
</Think>
```

Noise tokens:

```text
<N0>, <N1>, ..., <N63>
```

Countable marker tokens:

```text
<A>, <B>, <C>, <D>, <E>, <F>, <G>, <H>, <I>, <J>
```

Numeric tokens:

```text
<1>, <2>, ..., <10>
```

`<10>` must be a single token.

Expected vocabulary size:

```text
6 special + 64 noise + 10 markers + 10 numbers = 90 tokens
```

Save the vocabulary to:

```text
run_dir/vocab.json
```

### 1.2 Base example generator

A base example is a prompt-body sequence plus metadata. Both the non-thinking and thinking renderers use the same base example.

Training generation:

```yaml
train_seq_len: 256
count_range: 1..10
noise_vocab_size: 64
marker_vocab_size: 10
```

For each base example:

1. Sample count `n` uniformly from `{1, 2, ..., 10}`.
2. Sample `n` unique needle positions uniformly without replacement from `range(seq_len)`.
3. Sort the positions ascending.
4. At each selected position, sample one marker uniformly from the 10 marker tokens, independently with replacement.
5. At every non-needle position, sample one noise token uniformly from the 64 noise tokens.
6. Gold count is `n`.
7. Store exact metadata.

Metadata schema:

```python
@dataclass
class BaseExample:
    seq_len: int
    seq_tokens: list[str]
    count: int
    needle_positions: list[int]      # sorted ascending, length == count
    needle_markers: list[str]        # marker tokens in left-to-right order
    seed: int | None = None
```

Validation:

```python
assert len(seq_tokens) == seq_len
assert count == len(needle_positions) == len(needle_markers)
assert all(seq_tokens[p] == m for p, m in zip(needle_positions, needle_markers))
assert all(tok.startswith("<N") or tok in marker_vocab for tok in seq_tokens)
```

### 1.3 Renderers

Non-thinking sequence:

```text
<BOS> seq_tokens <Ans> <n> <EOS>
```

Example:

```text
<BOS> <N1> <N7> <A> <N2> <B> <C> <N9> <Ans> <3> <EOS>
```

Non-thinking evaluation prefix:

```text
<BOS> seq_tokens <Ans>
```

Accuracy is computed from the next-token logits at this position, restricted to numeric tokens `<1>` through `<10>`.

Thinking sequence:

```text
<BOS> seq_tokens <Think/> <1> marker_1 <2> marker_2 ... <n> marker_n </Think> <Ans> <n> <EOS>
```

Example:

```text
<BOS> <N1> <N7> <A> <N2> <B> <C> <N9>
<Think/> <1> <A> <2> <B> <3> <C> </Think> <Ans> <3> <EOS>
```

Thinking evaluation prefix for generated-trace mode:

```text
<BOS> seq_tokens <Think/>
```

The model should greedily generate:

```text
<1> marker_1 <2> marker_2 ... <n> marker_n </Think> <Ans> <n> <EOS>
```

Final accuracy is still computed only from the numeric token after `<Ans>`.

Important: the trace must enumerate prompt markers from left to right. If the prompt contains repeated markers, the trace repeats the same marker identities:

```text
<BOS> ... <C> ... <C> ... <C>
<Think/> <1> <C> <2> <C> <3> <C> </Think> <Ans> <3> <EOS>
```

### 1.4 Render span metadata

Every renderer must return token IDs plus span metadata.

```python
@dataclass
class RenderSpans:
    bos_pos: int
    seq_start: int
    seq_end_exclusive: int
    think_open_pos: int | None
    trace_token_positions: list[int]       # positions of <1>, marker_1, <2>, marker_2, ...
    trace_index_positions: list[int]       # positions of <1>, <2>, ...
    trace_marker_positions: list[int]      # positions of marker_1, marker_2, ...
    think_close_pos: int | None
    ans_pos: int
    final_count_pos: int
    eos_pos: int
```

Also keep prompt needle positions in rendered-token coordinates:

```python
prompt_needle_token_positions = [seq_start + p for p in base.needle_positions]
```

---

## 2. Model

Train small decoder-only Transformers from scratch.

Train exactly two separately initialized model types per seed:

```text
non_thinking
thinking
```

### 2.1 Required architecture

Use a small causal Transformer with:

```yaml
n_layers: 4
n_heads: 4
d_model: 256
d_mlp: 1024
dropout: 0.0
vocab_size: 90
```

Use **RoPE or ALiBi**, not learned absolute position embeddings, because v3 evaluates length generalization to positions beyond the training length. If the current repo uses GPT-2 learned absolute position embeddings, replace or wrap the attention module so that v3 uses RoPE.

Maximum sequence capacity must cover:

```text
max_eval_seq_len + max_trace_len + answer_suffix
= 1024 + 2 * 10 + 4 = 1048
```

Set model context length to at least `1152`; `2048` is safer.

### 2.2 Optimizer defaults

Use AdamW:

```yaml
optimizer: adamw
learning_rate: 3e-4
betas: [0.9, 0.95]
weight_decay: 0.1
warmup_steps: 500
train_steps: 10000
batch_size: 128
grad_clip_norm: 1.0
eval_every: 500
log_every: 50
checkpoint_every: 1000
```

Support a debug preset:

```yaml
preset: debug
train_steps: 200
batch_size: 32
eval_every: 50
checkpoint_every: 100
seq_lens_eval: [256, 512]
test_examples_per_count: 20
probe_examples_per_count: 50
attention_examples_per_count: 20
seeds: [1234]
```

Support a main preset:

```yaml
preset: main
train_steps: 10000
batch_size: 128
eval_every: 500
checkpoint_every: 1000
seq_lens_eval: [256, 512, 1024]
test_examples_per_count: 1000
probe_examples_per_count: 500
attention_examples_per_count: 100
seeds: [1234, 1235, 1236, 1237, 1238]
```

If compute is limited, implement all code paths and run `debug` first. The `main` command should be ready even if not executed immediately.

---

## 3. Fixed training objectives

All training is standard causal next-token prediction with shifted labels. The pipeline still needs a loss mask, but only to implement the two fixed objectives. Do **not** expose or sweep loss policies.

Use:

```python
def build_training_weights(tokens: list[int], spans: RenderSpans, model_type: str) -> torch.Tensor:
    """Return float tensor of shape [seq_len] with one weight per label position.
    A weight of 0 masks the position.
    A positive weight includes the next-token CE at that position.
    CE at position t predicts tokens[t + 1].
    """
```

### 3.1 Non-thinking objective

Rendered tokens:

```text
<BOS> seq_tokens <Ans> <n> <EOS>
```

Train only answer readout:

```text
weight 1.0 at <Ans> position          # predicts final numeric token <n>
weight 1.0 at final count position    # predicts <EOS>
weight 0.0 everywhere else
```

Do not supervise random prompt-body tokens. Do not supervise generation of `<Ans>`, because evaluation provides `<Ans>`.

Diagnostic losses to log:

```text
train_total_loss
train_final_count_ce
train_eos_ce
```

### 3.2 Thinking objective

Rendered tokens:

```text
<BOS> seq_tokens <Think/> <1> marker_1 <2> marker_2 ... <n> marker_n </Think> <Ans> <n> <EOS>
```

Train trace generation plus answer readout:

```text
weight 1.0 at <Think/> position          # predicts <1>
weight 1.0 on every trace token position # predicts the next trace token or </Think>
weight 1.0 at </Think> position          # predicts <Ans>
weight 1.0 at <Ans> position             # predicts final numeric token <n>
weight 1.0 at final count position       # predicts <EOS>
weight 0.0 on <BOS> and prompt-body positions
```

Do not supervise random prompt-body tokens. Do not supervise generation of `<Think/>`, because evaluation provides `<Think/>`.

Diagnostic losses to log:

```text
train_total_loss
train_trace_index_ce
train_trace_marker_ce
train_think_close_ce
train_ans_token_ce
train_final_count_ce
train_eos_ce
```

### 3.3 Training logs

At every `log_every` step, log:

```text
step
model_type
seed
train_total_loss
train_final_count_ce
train_trace_ce              # thinking only; NaN for non-thinking
train_eos_ce
learning_rate
```

At every `eval_every` step, run the hard-eval subset and log:

```text
step
model_type
seed
seq_len_eval
count
count_bin
final_accuracy
final_mae
final_answer_ce
trace_exact_rate            # thinking only
trace_marker_recall          # thinking only
invalid_generation_rate      # thinking only
```

---

## 4. Round 1: Hard length/noise evaluation

### 4.1 Purpose

The v2 task was too easy because train and test both used `seq_len = 256` and count range `1..10`. In v3, keep the count range fixed but evaluate with longer noise sequences.

This isolates the question:

> Does the model still count correctly when the number of needles is unchanged but the amount of noise grows?

### 4.2 Training

Train two baseline conditions:

```text
non_thinking
thinking
```

Use training examples with:

```text
seq_len = 256
count = 1..10
```

### 4.3 Evaluation sets

Create deterministic, count-balanced evaluation sets for each seed and each eval length:

```text
seq_len_eval in {256, 512, 1024}
count in {1, 2, ..., 10}
test_examples_per_count = 1000 for main, 20 for debug
```

For each `(seq_len_eval, count)`, generate exactly `test_examples_per_count` base examples with that exact count. Needle positions, marker identities, and noise tokens remain uniform.

Group counts into bins:

```text
low  = {1, 2, 3}
mid  = {4, 5, 6}
high = {7, 8, 9, 10}
```

### 4.4 Non-thinking evaluation

Input prefix:

```text
<BOS> seq_tokens <Ans>
```

Take logits at the last position and restrict to numeric tokens `<1>` through `<10>`.

Metrics:

```text
final_accuracy
final_mae
undercount_rate
overcount_rate
final_answer_ce
accuracy_by_exact_count
accuracy_by_bin
accuracy_by_seq_len
```

### 4.5 Thinking evaluation: generated-trace mode

Input prefix:

```text
<BOS> seq_tokens <Think/>
```

Greedy generate up to:

```python
max_new_tokens = 2 * max_count + 4 + 4  # trace pairs + </Think> + <Ans> + count + <EOS> + slack
```

Parse generated tokens:

1. Trace tokens before first `</Think>`.
2. First `<Ans>` after `</Think>`.
3. First numeric token after `<Ans>`.
4. Optional `<EOS>`.

A sample is invalid if:

```text
</Think> is missing
<Ans> is missing
no numeric token appears after <Ans>
the numeric token is not in <1> through <10>
```

Metrics:

```text
final_accuracy
final_mae
undercount_rate
overcount_rate
invalid_generation_rate
trace_exact_rate
trace_marker_recall
trace_marker_precision
trace_index_accuracy
duplicate_marker_position_rate
missing_trace_item_rate
extra_trace_item_rate
```

Trace exact means the generated trace equals exactly:

```text
<1> marker_1 <2> marker_2 ... <n> marker_n
```

Do not require the final answer to be correct for trace exact; report both separately.

### 4.6 Oracle-trace final-readout evaluation

For the thinking model, also run an oracle-trace readout check:

```text
<BOS> seq_tokens <Think/> gold_trace </Think> <Ans>
```

Take logits at `<Ans>` and restrict to numeric tokens. This measures whether the model can read the final count from a correct trace, separated from its ability to generate the trace.

Metrics:

```text
oracle_trace_final_accuracy
oracle_trace_final_mae
oracle_trace_final_answer_ce
```

### 4.7 Round 1 plots and tables

Create:

```text
figures/round1_train_loss_by_step.png
figures/round1_final_accuracy_by_step_and_seq_len.png
figures/round1_accuracy_by_count_final.png
figures/round1_accuracy_heatmap_count_x_seq_len.png
figures/round1_trace_metrics_by_seq_len.png
figures/round1_oracle_vs_generated_trace_accuracy.png
```

The key plot is accuracy vs step with separate lines for:

```text
model_type x seq_len_eval x count_bin
```

Also compute:

```text
step_to_90_accuracy
step_to_95_accuracy
step_to_99_accuracy
AUC_accuracy_over_training
```

for each model type, length, and bin.

Write:

```text
tables/round1_eval_by_step.csv
tables/round1_final_checkpoint_by_count.csv
tables/round1_step_to_thresholds.csv
```

---

## 5. Round 2: Corrupted-trace evaluation

### 5.1 Purpose

Separate three computations in the thinking model:

1. generating a correct retrieval trace;
2. reading the final count from the trace;
3. relying on shortcut cues such as trace length or the last index token.

This round applies only to the `thinking` model.

Use the final checkpoint and optionally selected checkpoints:

```text
steps: [1000, 2000, 4000, 6000, 8000, 10000]
```

### 5.2 Evaluation modes

#### Mode A: generated-trace eval

Prefix:

```text
<BOS> seq_tokens <Think/>
```

Model generates trace, `</Think>`, `<Ans>`, final number.

Metrics are the same as Round 1 thinking evaluation.

#### Mode B: oracle-trace final-readout eval

Prefix:

```text
<BOS> seq_tokens <Think/> gold_trace </Think> <Ans>
```

The model predicts the next token. Compute numeric accuracy from restricted numeric logits.

This measures whether the model can read the final count from a correct trace.

#### Mode C: corrupted-trace final-readout eval

Use the same prompt sequence but replace the gold trace with corrupted traces. Then provide prefix through `<Ans>` and measure the next numeric token.

Base prefix form:

```text
<BOS> seq_tokens <Think/> corrupted_trace </Think> <Ans>
```

Corruption types:

##### `wrong_indices_correct_markers`

Keep marker sequence correct but replace index tokens with wrong numeric tokens.

Example for true count 3:

```text
<1> <A> <1> <B> <1> <C>
```

or cyclic-shift indices:

```text
<2> <A> <3> <B> <1> <C>
```

##### `correct_indices_wrong_markers`

Keep indices correct but replace every marker with a random marker token, independent of the prompt.

```text
<1> <J> <2> <D> <3> <A>
```

##### `shuffled_trace_order`

Shuffle the `(index, marker)` pairs while keeping the original pair contents.

For true count 3:

```text
<2> <B> <1> <A> <3> <C>
```

##### `deleted_one_item`

Delete one trace pair. For true count `n`, trace length becomes `n - 1`.

##### `duplicated_one_item`

Duplicate one trace pair. For true count `n`, trace length becomes `n + 1`, unless `n = 10`; for `n = 10`, skip this corruption or keep length capped and report skipped count.

##### `extra_random_item`

Append one random valid-looking pair with index `<n+1>` if `n < 10`.

##### `last_index_replaced`

Keep trace pairs and markers correct, but replace the final index token with another numeric token.

Example:

```text
<1> <A> <2> <B> <9> <C>
```

##### `indices_removed`

Remove all index tokens but keep markers:

```text
<A> <B> <C>
```

##### `markers_removed`

Keep indices but remove markers:

```text
<1> <2> <3>
```

##### `empty_trace`

Use no trace tokens at all:

```text
<BOS> seq_tokens <Think/> </Think> <Ans>
```

This tests whether the model can fall back to the prompt when trace information is absent.

### 5.3 Diagnostic labels for corrupted traces

For each corrupted trace, compute these possible answer rules:

```python
prompt_count = true number of prompt needles
trace_pair_count = number of generated/corrupted marker-index pairs
last_index_value = numeric value of last index token if present else None
max_index_value = max numeric index token if present else None
marker_count_in_trace = number of marker tokens in trace
```

For model prediction `pred`, classify:

```text
follows_prompt_count
follows_trace_pair_count
follows_last_index
follows_max_index
follows_marker_count
other
```

A prediction can match multiple rules; record all boolean flags.

### 5.4 Round 2 plots and tables

Create:

```text
figures/round2_corruption_accuracy_by_type.png
figures/round2_follow_rule_breakdown.png
figures/round2_confusion_pred_vs_prompt_count.png
figures/round2_confusion_pred_vs_trace_pair_count.png
figures/round2_confusion_pred_vs_last_index.png
figures/round2_corruption_by_seq_len.png
```

Write:

```text
tables/round2_corrupted_trace_results.csv
tables/round2_follow_rule_summary.csv
```

Minimum columns:

```text
model_type
seed
checkpoint_step
seq_len_eval
count
corruption_type
prompt_count
trace_pair_count
last_index_value
max_index_value
marker_count_in_trace
pred_count
correct_prompt_count
follows_prompt_count
follows_trace_pair_count
follows_last_index
follows_max_index
follows_marker_count
invalid
```

---

## 6. Round 3: Hidden-state probes, attention retrieval, and causal tests

### 6.1 Purpose

Round 3 should determine whether mechanistic differences observed in v2 are robust and whether they are causal.

The report must avoid claiming that a head or probe is causal unless an intervention changes behavior.

Run Round 3 on final checkpoints for:

```text
non_thinking
thinking
```

Optionally also run on selected checkpoints:

```text
steps: [1000, 2000, 4000, 6000, 8000, 10000]
```

---

### 6.2 Hidden-state cache export

Implement a cache function that returns residual-stream hidden states and attention probabilities.

Required cache names:

```text
resid_pre[layer]
resid_post[layer]
attn_probs[layer, head]
attn_out[layer, head]        # if easy
mlp_out[layer]               # if easy
```

At minimum, implement `resid_post` and `attn_probs`.

Use no dropout and deterministic evaluation.

Save caches only for probe/attention subsets, not the whole test set.

---

### 6.3 Probe analysis

#### 6.3.1 Anchors

Non-thinking anchors:

```text
ans_pos                  # <Ans> token position
pre_ans_pos              # token immediately before <Ans>
needle_prompt_positions  # prompt positions of actual markers
noise_prompt_positions   # sampled noise positions as negative/control anchors
```

Thinking anchors:

```text
think_open_pos           # <Think/>
pre_index_k              # position immediately before index token <k>; avoids direct label leakage
index_k_pos              # index token <k>; report but mark as leakage-prone
marker_k_pos             # marker token after <k>
post_marker_k            # position immediately after marker_k
think_close_pos          # </Think>
ans_pos                  # <Ans>
pre_ans_pos              # token immediately before <Ans>
```

Do not use `index_k_pos` as the main prefix-count evidence because the token identity itself leaks `k`. It is allowed as a sanity check only.

#### 6.3.2 Probe targets

Fit probes for:

```text
final_count: n in 1..10
prefix_count: k in 1..n, only for per-item anchors
is_needle: binary marker-vs-noise for prompt positions
```

#### 6.3.3 Probe models

Implement:

```text
multinomial logistic regression for count classification
ridge regression for numeric count prediction
```

Use scikit-learn if available. Otherwise implement a simple PyTorch linear probe.

#### 6.3.4 Probe controls

Implement these controls:

1. **position-only baseline:** probe from absolute position ID or relative position ID only.
2. **trace-length-only baseline:** for thinking final-count probes, predict count from trace length only.
3. **embedding-only baseline:** probe raw token embedding states at layer 0; report as sanity only.
4. **held-out marker-type split:** optionally train on examples excluding one marker type and test on that marker type.

The report should not treat probe accuracy as causal evidence.

#### 6.3.5 Probe outputs

Write:

```text
tables/round3_probe_results.csv
figures/round3_probe_accuracy_layer_by_anchor.png
figures/round3_probe_r2_layer_by_anchor.png
figures/round3_probe_vs_position_baseline.png
```

Minimum columns:

```text
model_type
seed
checkpoint_step
seq_len_eval
layer
resid_site
anchor_type
target_type
probe_type
train_accuracy
test_accuracy
r2
mae
position_only_accuracy
trace_length_only_accuracy
embedding_only_accuracy
leakage_prone
```

---

### 6.4 Attention retrieval analysis

#### 6.4.1 Thinking trace-to-prompt retrieval

For each thinking example with gold or generated trace, construct a matrix:

```text
A[layer, head, k, j]
```

where:

- `k` indexes trace item query positions;
- `j` indexes prompt needle positions;
- query position should be one of:
  - `index_k_pos`
  - `marker_k_pos`
  - `post_marker_k`
- key positions are prompt needle token positions.

For each layer/head/query-anchor, compute:

```text
correct_top1_rate: argmax_j A[k, j] == k
diagonal_dominance: mean diag(A) / mean row_sum(A)
needle_mass: attention mass to all prompt needles
noise_mass: attention mass to sampled noise positions
needle_to_noise_ratio: needle_mass / max(noise_mass, eps)
entropy_over_prompt_positions
off_diagonal_mass
```

Use left-to-right needle index `k` as the gold retrieval target.

#### 6.4.2 Non-thinking final-answer retrieval

For non-thinking, use `<Ans>` as the query position.

Compute attention mass from `<Ans>` to:

```text
all prompt needle positions
all prompt noise positions
sampled noise positions
```

Metrics:

```text
top_n_recall: whether the top n attended prompt positions include all n needles
needle_mass
noise_mass
needle_to_noise_ratio
entropy_over_prompt_positions
```

#### 6.4.3 Attention plots and tables

Create:

```text
figures/round3_attention_head_leaderboard.png
figures/round3_thinking_trace_to_prompt_heatmap_best_head.png
figures/round3_nonthinking_ans_to_prompt_attention.png
figures/round3_attention_metrics_by_count_bin.png
figures/round3_attention_metrics_by_seq_len.png
```

Write:

```text
tables/round3_attention_head_metrics.csv
```

Minimum columns:

```text
model_type
seed
checkpoint_step
seq_len_eval
layer
head
query_anchor
count_bin
correct_top1_rate
diagonal_dominance
needle_mass
noise_mass
needle_to_noise_ratio
entropy
top_n_recall
```

---

### 6.5 Causal tests: head ablation and attention masking

Attention alone is not causal. Implement at least simple causal tests.

#### 6.5.1 Single-head ablation

For each top attention head from the leaderboard, run ablation on evaluation examples:

```python
attn_out[layer, head] = 0
```

or replace the head output with its mean over a clean batch.

Measure change in:

```text
final_accuracy
final_answer_logit_margin
trace_exact_rate              # thinking only
marker_recall                 # thinking only
correct_top1_attention        # if attention recomputed
```

#### 6.5.2 Multi-head ablation

Ablate:

```text
top_1 retrieval head
top_2 retrieval heads
top_4 retrieval heads
all heads with diagonal_dominance >= threshold
```

Report whether behavior drops more than single-head ablation. If single-head ablation has little effect but multi-head ablation drops trace exact or final accuracy, interpret as redundancy.

#### 6.5.3 Targeted attention masking

For thinking models, implement masking for query anchor `index_k_pos` or `marker_k_pos`:

```text
mask attention from trace item k to the correct prompt needle position k
mask attention from trace item k to all prompt needle positions
mask attention from trace item k to all non-needle positions
```

Measure final accuracy and trace generation quality.

If modifying attention masks is too invasive, implement only head ablation and leave attention masking as a clearly marked TODO in the report.

#### 6.5.4 Optional path patching

If the codebase already has activation patching utilities, add Q/K/V/O path patching for the top retrieval head:

```text
patch q at trace query position
patch k at prompt needle position
patch v at prompt needle position
patch o at trace query position
```

This is optional for v3. Do not block the main report on it.

#### 6.5.5 Causal-test outputs

Write:

```text
tables/round3_head_ablation_results.csv
tables/round3_attention_masking_results.csv
figures/round3_head_ablation_effects.png
figures/round3_attention_masking_effects.png
```

Minimum columns:

```text
model_type
seed
checkpoint_step
seq_len_eval
intervention_type
layer
head
query_anchor
mask_type
count_bin
baseline_final_accuracy
intervened_final_accuracy
delta_final_accuracy
baseline_trace_exact
intervened_trace_exact
delta_trace_exact
baseline_logit_margin
intervened_logit_margin
delta_logit_margin
```

---

## 7. Report generation

Generate a single self-contained report:

```text
run_dir/syn_v3_no_loss_report.html
```

The report should include:

1. Config and run metadata.
2. Round 1 hard-eval results.
3. Round 2 corrupted-trace diagnostics.
4. Round 3 probes, attention, and causal tests.
5. A short interpretation section with explicit limitations.

Also write a machine-readable summary:

```text
run_dir/summary.json
```

Required summary keys:

```json
{
  "run_name": "...",
  "preset": "debug|main",
  "train_seq_len": 256,
  "seq_lens_eval": [256, 512, 1024],
  "count_range": [1, 10],
  "seeds": [],
  "non_thinking_final_accuracy_by_len": {},
  "thinking_final_accuracy_by_len": {},
  "thinking_trace_exact_by_len": {},
  "round1_main_takeaway": "...",
  "round2_main_takeaway": "...",
  "round3_main_takeaway": "...",
  "limitations": []
}
```

The interpretation section must distinguish:

```text
behavioral evidence
trace-generation evidence
corrupted-trace evidence
probe evidence
attention evidence
causal-intervention evidence
```

Do not state that a retrieval head is causal unless ablation or masking changes behavior.

---

## 8. File organization

Create or refactor into a clear module structure. Suggested layout:

```text
synthetic_niah_v3/
  __init__.py
  vocab.py
  data.py
  render.py
  model.py
  objectives.py
  train.py
  eval.py
  trace_parse.py
  corrupted_trace.py
  probes.py
  attention.py
  interventions.py
  plots.py
  report.py
  run_v3.py
configs/
  syn_v3_no_loss_debug.yaml
  syn_v3_no_loss_main.yaml
```

Run outputs:

```text
runs/syn_v3_no_loss/{timestamp}_{preset}/
  config.yaml
  vocab.json
  checkpoints/
  metrics/
    train_log.csv
    eval_by_step.csv
    eval_by_count.csv
    eval_by_bin.csv
  tables/
    round1_eval_by_step.csv
    round1_final_checkpoint_by_count.csv
    round1_step_to_thresholds.csv
    round2_corrupted_trace_results.csv
    round2_follow_rule_summary.csv
    round3_probe_results.csv
    round3_attention_head_metrics.csv
    round3_head_ablation_results.csv
    round3_attention_masking_results.csv
  figures/
    *.png
  summary.json
  syn_v3_no_loss_report.html
```

---

## 9. CLI requirements

Implement these commands:

```bash
# quick sanity check
python -m synthetic_niah_v3.run_v3 --preset debug --round all

# main run
python -m synthetic_niah_v3.run_v3 --preset main --round all

# individual rounds
python -m synthetic_niah_v3.run_v3 --preset main --round 1_hard_eval
python -m synthetic_niah_v3.run_v3 --preset main --round 2_corrupted_trace
python -m synthetic_niah_v3.run_v3 --preset main --round 3_mechanistic
```

Also support:

```bash
python -m synthetic_niah_v3.run_v3 --preset main --round all --seeds 1234,1235
python -m synthetic_niah_v3.run_v3 --preset main --round all --device cuda
python -m synthetic_niah_v3.run_v3 --preset main --round all --device cpu
```

If the repo already has a different CLI framework, adapt this interface while preserving these capabilities.

---

## 10. Unit tests and sanity checks

Add unit tests or script-level assertions for:

### Data generation

```text
- generated sequence length is correct;
- count equals number of marker tokens;
- positions are unique and sorted;
- marker identities at positions match metadata;
- count distribution is approximately uniform over a large sample;
- marker distribution is approximately uniform;
- noise distribution is approximately uniform.
```

### Rendering

```text
- non-thinking rendered sequence has exactly one <Ans> and final count after it;
- thinking rendered sequence has <Think/> before trace and </Think> before <Ans>;
- thinking trace exactly matches left-to-right marker order;
- <10> is one token;
- span metadata points to the correct tokens.
```

### Training objectives

```text
- non-thinking objective weights only <Ans> -> final count and final count -> <EOS>;
- thinking objective masks prompt-body positions;
- thinking objective weights <Think/> -> <1>, trace tokens, </Think> -> <Ans>, <Ans> -> final count, and final count -> <EOS>;
- no loss-policy sweep exists in CLI or config.
```

### Evaluation

```text
- non-thinking eval reads logits after <Ans>;
- thinking free-run parser handles valid and invalid generations;
- oracle-trace eval reads logits after <Ans>;
- corrupted trace labels are computed correctly;
- results are grouped by exact count, count bin, and eval sequence length.
```

### Attention/probe

```text
- attention matrices align trace item k to prompt needle j;
- diagonal dominance is high for a manually constructed diagonal matrix;
- position-only probe baseline does not use hidden states;
- index-token probes are marked leakage-prone;
- head ablation changes only the requested layer/head.
```

---

## 11. Acceptance criteria

The implementation is acceptable when:

1. `debug` preset runs end-to-end on CPU or GPU and produces `syn_v3_no_loss_report.html`.
2. The report contains all three rounds.
3. Round 1 shows accuracy and loss curves by step, count bin, and eval sequence length.
4. Round 2 reports corrupted-trace follow-rule diagnostics.
5. Round 3 reports probes and attention retrieval metrics.
6. At least single-head ablation is implemented for the top retrieval head. If attention masking is not implemented, the report must say so explicitly.
7. All generated figures and CSV files are saved under the run directory.
8. The implementation does not overwrite v2 results.
9. The implementation does not include the old Round 2 loss-mask ablation.
10. The final output path is printed at the end of the run.

---

## 12. Expected interpretation template

At the end of the report, use this structure:

```text
Behavior:
- Did thinking outperform non-thinking at longer seq_len?
- Which count bin broke first?
- Does the gap grow from seq_len 256 to 512 to 1024?

Trace:
- Did thinking generate exact traces?
- Did trace quality degrade before final accuracy or vice versa?
- Are failures missing items, extra items, wrong marker identities, wrong indices, or invalid delimiters?

Corrupted trace:
- Does final count follow prompt count, trace pair count, last index, max index, or marker count?
- Does the model fall back to prompt count when the trace is empty or malformed?

Hidden states:
- Which anchors/layers decode final count or prefix count?
- Does probe accuracy exceed position-only and trace-length-only baselines?
- Are index-token probes clearly marked as leakage-prone?

Attention:
- Are there near-diagonal trace-to-prompt retrieval heads?
- Does non-thinking rely on broad <Ans>-to-prompt aggregation?
- Does retrieval geometry degrade at seq_len 512 or 1024?

Causality:
- Does ablating top retrieval heads reduce trace exact or final accuracy?
- If not, is the likely explanation redundancy or non-causal diagnostic attention?

Limitations:
- All data are symbolic.
- Counts are still limited to 1..10.
- The trace exposes count length, so final readout may exploit trace-length or last-index shortcuts.
- Probe decodability is not causal evidence.
- Attention patterns are not causal unless ablation or masking changes behavior.
- There is no loss-mask ablation in this version by design.
```
