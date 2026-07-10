# Synthetic NIAH Counting v4: Hidden-State Probe + Steering Pipeline

You are implementing v4 of the synthetic NIAH counting experiments. The goal of v4 is **not** OOD generalization. Keep the behavioral task in-distribution and saturated if necessary. The goal is to test whether the trained small transformers contain a hidden-state direction or subspace that can be used to **causally steer the final count**.

The main deliverable is a runnable Python package that trains or loads **v2-style GPT-2 models**, caches hidden states, fits deconfounded probes, constructs steering directions, applies hidden-state interventions, and produces an HTML report with plots and tables.

---

## 0. Scope

### Do this

Implement a full hidden-state analysis pipeline for the existing simple symbolic task:

- fixed sequence length;
- 64 noise tokens;
- 10 countable marker tokens;
- count range 1..10;
- one non-thinking model;
- one thinking model;
- no ID/OOD split;
- no loss-mask ablation;
- focus on probe, matched-pair directions, interchange patching, and steering.

### Do not do this in v4

- Do not run 256 -> 512 / 1024 length OOD.
- Do not add realistic city-score records.
- Do not add distractors.
- Do not train a mixed thinking/non-thinking single model; that is v5.
- Do not claim that high probe R2 is a count-vector. Treat probe as diagnostic until steering or patching works.

---

## 1. Task definition

### Vocabulary

Use exactly this symbolic vocabulary unless a config override is supplied.

```text
special tokens:
  <PAD>, <BOS>, <EOS>, <Ans>, <Think/>, </Think>

noise tokens:
  <N0>, <N1>, ..., <N63>

countable marker tokens:
  <A>, <B>, <C>, <D>, <E>, <F>, <G>, <H>, <I>, <J>

numeric tokens:
  <1>, <2>, ..., <10>
```

Use a fixed integer tokenizer mapping. Do not use BPE. v4 follows the v2 marker-trace convention where the same numeric tokens are used for trace indices and final count answers.

### Base example generation

Each base example has:

```python
seq_len: int = 256
count: int in {1, ..., 10}
needle_positions: sorted list[int] of length count
needle_markers: list[str] of length count
seq_tokens: list[str] of length seq_len
```

Sampling rules:

```text
count: uniform over 1..10
needle positions: choose count unique positions uniformly from 0..seq_len-1
needle marker type: each marker independently uniform over <A>..<J>
noise token type: each non-needle position independently uniform over <N0>..<N63>
```

The trace order is strictly left-to-right by prompt position.

Example base sequence:

```text
<N4> <A> <N9> <B> <N2> <C> <N7>
```

Gold count is `3`.

---

## 2. Rendered datasets

Train two separate transformer models, matching the v2 marker-trace setting.

### 2.1 Non-thinking model

Rendered sequence:

```text
<BOS> seq_tokens <Ans> <n> <EOS>
```

Example:

```text
<BOS> <N4> <A> <N9> <B> <N2> <C> <N7> <Ans> <3> <EOS>
```

Training loss:

```text
prompt tokens: <BOS> seq_tokens <Ans>
supervised tokens: <n> <EOS>
```

Use causal next-token prediction with labels masked to `-100` on prompt positions.

Evaluation query:

```text
<BOS> seq_tokens <Ans>
```

Final-count accuracy is computed from the logits at the next token after `<Ans>`, restricted to numeric token ids `<1>.. <10>`.

### 2.2 Thinking model

Rendered sequence:

```text
<BOS> seq_tokens <Think/> <1> marker_1 <2> marker_2 ... <n> marker_n </Think> <Ans> <n> <EOS>
```

Example:

```text
<BOS> <N4> <A> <N9> <B> <N2> <C> <N7>
<Think/> <1> <A> <2> <B> <3> <C> </Think> <Ans> <3> <EOS>
```

Training loss:

```text
prompt tokens: <BOS> seq_tokens <Think/>
supervised tokens: <1> marker_1 ... <n> marker_n </Think> <Ans> <n> <EOS>
```

Evaluation modes:

```text
A. generated trace:
   prompt = <BOS> seq_tokens <Think/>
   greedy-generate until </Think>, then <Ans>, then count.

B. oracle trace final readout:
   prompt = <BOS> seq_tokens <Think/> gold_trace </Think> <Ans>
   evaluate next-token count logits.
```

For steering, make **oracle trace final readout** the primary mode because it isolates count readout from trace generation.

---

## 3. Model

Use the same transformer family as v2: HuggingFace `GPT2LMHeadModel` built from `GPT2Config`.

Important: v4 should **not** use the v3 custom RoPE transformer. The v2 architecture uses learned absolute positional embeddings through GPT-2's `wpe` table. This keeps v4 directly comparable to the previous v2 attention results.

Default main config:

```yaml
model:
  architecture: gpt2_lm_head
  position_embedding: learned_absolute
  n_layer: 4
  n_head: 4
  n_embd: 256
  n_positions: 320
  dropout: 0.0
  activation_function: gelu_new
```

Debug config:

```yaml
model:
  architecture: gpt2_lm_head
  position_embedding: learned_absolute
  n_layer: 2
  n_head: 2
  n_embd: 128
  n_positions: 128
  dropout: 0.0
  activation_function: gelu_new
```

Training defaults:

```yaml
train:
  steps: 10000
  batch_size: 128
  lr: 3e-4
  weight_decay: 0.01
  warmup_steps: 500
  grad_clip: 1.0
  eval_every: 500
  checkpoint_every: 1000
  seed: 1234
```

The package may either train the models from scratch or load v2-compatible GPT-2 checkpoints. The default `run_v4 --stage all` should train if no checkpoints are supplied.

---

## 4. Required package structure

Create this package structure:

```text
src/synthetic_niah_v4/
  __init__.py
  config.py
  vocab.py
  data.py
  model.py
  train.py
  generation.py
  hooks.py
  cache.py
  probes.py
  directions.py
  patching.py
  steering.py
  metrics.py
  plots.py
  report.py
  run_v4.py
```

The top-level run command must be:

```bash
python -m synthetic_niah_v4.run_v4 --preset debug --stage all
python -m synthetic_niah_v4.run_v4 --preset main  --stage all
```

Supported stages:

```text
train
behavior_eval
cache
probe
directions
patching
steering
plots
report
all
```

---

## 5. Hidden-state cache

### 5.1 Hook points

Cache residual stream states at the following hook points:

```text
embed
resid_pre_layer_l     for l = 0..n_layers-1
resid_post_layer_l    for l = 0..n_layers-1
final_norm
```

Primary steering hook:

```text
resid_post_layer_l
```

Secondary optional hook:

```text
resid_pre_layer_l
```

### 5.2 Anchor positions

Store token positions and metadata for these anchors.

#### Non-thinking anchors

```text
prompt_marker_k       position of kth needle in prompt sequence
pre_ans_pos           token immediately before <Ans>
ans_pos               position of <Ans>
```

#### Thinking anchors

```text
prompt_marker_k       position of kth needle in prompt sequence
think_open_pos        position of <Think/>
pre_index_k           token immediately before numeric token <k>
index_k_pos           position of numeric token <k>
marker_k_pos          generated trace marker for kth needle
post_marker_k         token immediately after generated trace marker k, if it exists
think_close_pos       position of </Think>
pre_ans_pos           token immediately before <Ans>
ans_pos               position of <Ans>
```

Do not treat `index_k_pos` as clean counter evidence by itself because the numeric token `<k>` directly leaks prefix count. It can still be used as an intervention site.

### 5.3 Cache metadata schema

For every cached state row, store:

```text
example_id
model_type: non_thinking | thinking
count: int
count_bin: low | mid | high
seq_len
anchor_name
anchor_k: optional int
layer
hook_name
position
absolute_token_id
token_string
hidden: float[d_model]
```

For thinking prefix anchors, also store:

```text
prefix_count = k
final_count = n
trace_length_tokens
```

---

## 6. Probe analysis

The goal is to identify candidate counter states while controlling for leakage.

### 6.1 Probe targets

Fit probes for these targets:

```text
final_count: integer 1..10
prefix_count: integer 1..10, only for k-dependent thinking anchors
```

### 6.2 Probe types

Implement:

```text
ridge_scalar:
  hidden -> scalar count
  metrics: R2, MAE, rounded accuracy

multiclass_logistic:
  hidden -> 10-way count class
  metrics: accuracy, cross entropy
```

Use scikit-learn if available. Otherwise implement minimal ridge closed form and use PyTorch for logistic regression.

### 6.3 Train/test split

Use example-level split, not hidden-row-level split.

Default:

```yaml
probe:
  examples_per_count_train: 1000
  examples_per_count_test: 1000
  train_fraction: 0.5
  standardize_hidden: true
  ridge_alpha_grid: [0.01, 0.1, 1.0, 10.0, 100.0]
  logistic_l2_grid: [0.01, 0.1, 1.0, 10.0]
```

### 6.4 Confound baselines

For each probe result, compute and report:

```text
position-only baseline:
  one-hot or scalar absolute position -> target

token-id-only baseline:
  token id -> target

trace-length-only baseline:
  trace length -> final_count, thinking only

index-token-only baseline:
  index k -> prefix_count, thinking only

shuffled-label baseline:
  hidden -> randomly permuted labels
```

### 6.5 Residualized hidden probes

Implement residualized probes:

1. Fit a linear model from confounds to hidden state on train data.
2. Replace hidden by residual hidden:

```python
h_resid = h - confound_model(confounds)
```

3. Fit the count probe on `h_resid`.

At minimum residualize against:

```text
absolute position
trace length
token id
```

Report both raw-probe and residualized-probe metrics.

### 6.6 Probe outputs

Write:

```text
outputs/v4/tables/probe_results.csv
outputs/v4/tables/probe_baselines.csv
outputs/v4/tables/probe_residualized.csv
outputs/v4/figures/probe_acc_by_layer_anchor.png
outputs/v4/figures/probe_r2_by_layer_anchor.png
outputs/v4/figures/probe_minus_baseline_heatmap.png
```

---

## 7. Steering direction construction

Do not rely on one direction only. Implement several candidate directions.

### 7.1 Ridge direction

For a fitted ridge scalar probe:

```python
v_ridge = normalize(w_ridge)
```

Positive direction should increase predicted count.

### 7.2 Logistic adjacent-count direction

For logistic classifier weights `W[class, dim]`, define:

```python
v_logistic_adjacent = mean_{k=1..9}(W[k+1] - W[k])
```

Normalize to unit norm.

### 7.3 Difference-of-means direction

For an anchor/layer:

```python
mu_k = mean hidden for examples with count k
v_dom = mean_{k=1..9}(mu_{k+1} - mu_k)
v_dom = normalize(v_dom)
```

### 7.4 Matched-pair delta direction

This is the preferred deconfounded direction.

Construct matched base-example pairs that differ by exactly one added needle while preserving as much as possible:

```text
same seq_len
same noise tokens at non-edited positions
same marker vocabulary distribution as much as possible
same answer position
same trace-format positions as much as possible
count differs by +1
```

For each pair `(x_minus, x_plus)` and each anchor/layer:

```python
delta_h = h(x_plus) - h(x_minus)
```

Average deltas:

```python
v_matched_delta = normalize(mean(delta_h))
```

For thinking prefix anchors, only include pairs where the added needle is before or at the relevant prefix anchor when testing prefix-count changes.

### 7.5 Unembedding count direction

For count tokens:

```python
v_unembed_adjacent = mean_{k=1..9}(unembed[<{k+1}>] - unembed[<k>])
```

If embeddings are tied, use the output unembedding matrix. Normalize to unit norm. This direction is mainly a diagnostic; it may not align with internal count states.

### 7.6 Direction diagnostics

For every anchor/layer/direction, report:

```text
norm
projection_mean_by_count
projection_slope_vs_count
projection_R2
cosine with other directions
cosine with unembedding direction
cross-seed cosine if multiple seeds are run
```

Write:

```text
outputs/v4/tables/direction_metrics.csv
outputs/v4/figures/direction_cosine_heatmap.png
outputs/v4/figures/projection_by_count.png
```

---

## 8. Steering interventions

### 8.1 Basic intervention

At a selected hidden state:

```python
h[layer, pos] <- h[layer, pos] + alpha * scale(anchor, layer) * v
```

where:

```text
v: candidate direction
alpha: scalar from grid
scale(anchor, layer): default = std of projection <h, v> on probe train data
```

Alpha grid:

```yaml
steering:
  alpha_grid: [-6, -4, -2, -1, 0, 1, 2, 4, 6]
```

### 8.2 Primary steering evaluation: final-count logits

Use final-count logits to avoid generation artifacts.

#### Non-thinking

Input prefix:

```text
<BOS> seq_tokens <Ans>
```

Intervene at one anchor/layer in the forward pass, then evaluate count-token logits at the next-token position.

#### Thinking

Primary input prefix:

```text
<BOS> seq_tokens <Think/> gold_trace </Think> <Ans>
```

Intervene at one anchor/layer in the forward pass, then evaluate count-token logits at the next-token position.

### 8.3 Secondary steering evaluation: generated trace

Optional but useful.

Thinking input prefix:

```text
<BOS> seq_tokens <Think/>
```

Greedy-generate trace and final answer while applying the intervention when the specified anchor becomes available. If an anchor does not exist in generated text because generation diverges, record `anchor_missing=true` and skip that sample for generated-trace steering.

### 8.4 Steering targets

For each example with gold count `n`, evaluate:

```text
increase steering: alpha > 0 should shift predicted count upward
decrease steering: alpha < 0 should shift predicted count downward
```

Do not require target counts outside 1..10. Clip expected count to range 1..10 only for summary plots, but keep raw predicted count for metrics.

### 8.5 Steering metrics

For each `(model_type, eval_mode, anchor, layer, hook, direction, alpha)` report:

```text
n_examples
base_accuracy
steered_accuracy
mean_pred_count_base
mean_pred_count_steered
mean_count_shift = mean(pred_steered - pred_base)
mean_gold_logit_change
mean_correct_logprob_change
mean_target_plus_one_logprob_change
monotonicity_score over alpha grid
validity_rate
count_token_rate
KL_count_distribution_vs_base
```

Monotonicity score:

```python
SpearmanCorr(alpha, mean_pred_count_steered)
```

Success criterion for a candidate count-vector:

```text
1. positive alpha increases predicted count on average;
2. negative alpha decreases predicted count on average;
3. effect is monotonic over alpha;
4. output remains a valid count token;
5. effect is localized to plausible anchors/layers;
6. effect is stronger than shuffled-direction and random-direction controls;
7. effect survives residualization/matched-pair direction construction.
```

### 8.6 Controls

For each steering run also evaluate:

```text
random_unit_direction with same norm
shuffled_label_probe_direction
orthogonalized_direction_to_count_probe
zero intervention
wrong-anchor intervention
wrong-layer intervention
```

Write:

```text
outputs/v4/tables/steering_results.csv
outputs/v4/figures/steering_heatmap_anchor_layer.png
outputs/v4/figures/steering_dose_response_top_configs.png
outputs/v4/figures/steering_controls.png
```

---

## 9. Interchange patching

Steering may fail even if the state is causal. Implement interchange patching as a stronger causal test.

### 9.1 Pair selection

Create donor and receiver pairs:

```text
same model_type
same eval_mode
same anchor_name
same anchor_k if applicable
same token string at anchor when possible
different final_count or prefix_count
```

Prefer pairs with count difference exactly `+1` or `-1`.

### 9.2 Patch operation

At hook `(layer, hook_name, anchor_pos)`:

```python
receiver_hidden[layer, pos] <- donor_hidden[layer, pos]
```

Then continue the forward pass and evaluate final count logits.

### 9.3 Metrics

Report:

```text
receiver_base_pred
donor_pred
patched_pred
patched_moves_toward_donor: bool
logit_recovery_toward_donor_count
causal_effect_size
```

For thinking prefix anchors, also test whether patching `prefix_count=k` states changes later trace or final count.

### 9.4 Outputs

```text
outputs/v4/tables/interchange_patching_results.csv
outputs/v4/figures/interchange_patch_matrix.png
```

---

## 10. Input-to-state geometry

Implement a light version of the input geometry test.

### 10.1 Perturbations

For a base example, construct paired examples:

```text
delete_one_needle:
  replace one needle marker with a noise token; count decreases by 1

add_one_needle:
  replace one noise token with a marker; count increases by 1

replace_irrelevant_noise:
  change only noise tokens; count unchanged

permute_needle_markers:
  permute marker identities among needle positions; count unchanged

permute_needle_positions:
  move needle positions while preserving count; count unchanged, prompt retrieval path changes
```

### 10.2 Projection trajectory

For each candidate direction `v`, plot projection along token positions or trace anchors:

```python
proj = dot(h[layer, pos], v)
```

Expected for a good counter direction:

```text
add/delete needle causes a step-like shift after the edited item;
noise replacement causes small projection change;
permuting marker identities preserves final-count projection;
permuting needle positions permutes prefix trajectory but preserves final count.
```

Outputs:

```text
outputs/v4/tables/input_geometry_results.csv
outputs/v4/figures/input_geometry_projection_trajectories.png
```

---

## 11. Report

Generate:

```text
outputs/v4/report.html
outputs/v4/report.md
```

The report must contain:

1. setup and hyperparameters;
2. final ID behavioral accuracy for non-thinking and thinking;
3. probe tables with baselines;
4. residualized probe results;
5. direction diagnostics;
6. steering heatmaps and dose-response plots;
7. steering control results;
8. interchange patching matrix;
9. input geometry projection plots;
10. a conclusion section that explicitly says whether steering succeeded or remains weak.

Use cautious language:

```text
- "linearly decodable count information" is allowed.
- "count-vector" is only allowed if monotonic steering or patching succeeds.
- If steering fails, state that decodability is not causal under the tested interventions.
```

---

## 12. Required CSV schemas

### `probe_results.csv`

```text
model_type, eval_mode, anchor_name, anchor_k, target, hook_name, layer,
probe_type, raw_or_residualized, train_n, test_n,
accuracy, r2, mae, ce_loss,
position_baseline_acc, token_baseline_acc, trace_len_baseline_acc,
leakage_prone
```

### `direction_metrics.csv`

```text
model_type, eval_mode, anchor_name, anchor_k, hook_name, layer,
direction_type, target,
norm, projection_slope, projection_r2,
cosine_with_ridge, cosine_with_dom, cosine_with_matched_delta,
cosine_with_unembedding
```

### `steering_results.csv`

```text
model_type, eval_mode, anchor_name, anchor_k, hook_name, layer,
direction_type, alpha, n_examples,
base_accuracy, steered_accuracy,
mean_pred_base, mean_pred_steered, mean_count_shift,
mean_gold_logit_change, mean_correct_logprob_change,
mean_target_plus_one_logprob_change,
monotonicity_score, validity_rate, count_token_rate,
kl_count_distribution_vs_base,
control_type
```

### `interchange_patching_results.csv`

```text
model_type, eval_mode, anchor_name, anchor_k, hook_name, layer,
donor_count, receiver_count,
base_pred, donor_pred, patched_pred,
patched_moves_toward_donor,
logit_recovery_toward_donor_count,
causal_effect_size,
n_examples
```

---

## 13. Acceptance tests

Debug run must satisfy:

```bash
python -m synthetic_niah_v4.run_v4 --preset debug --stage all
```

and produce:

```text
outputs/v4/report.html
outputs/v4/tables/probe_results.csv
outputs/v4/tables/steering_results.csv
outputs/v4/tables/interchange_patching_results.csv
outputs/v4/figures/steering_heatmap_anchor_layer.png
```

Unit tests must cover:

```text
1. tokenizer round-trip;
2. data generator count correctness;
3. non-thinking render and label mask correctness;
4. thinking render and label mask correctness;
5. anchor extraction correctness;
6. hidden cache shape correctness;
7. probe train/test split by example_id;
8. steering hook changes logits when alpha != 0;
9. random direction control runs;
10. report generation does not require manual paths.
```

---

## 14. Default main run

After debug passes, run:

```bash
python -m synthetic_niah_v4.run_v4 --preset main --stage all \
  --seq-len 256 \
  --count-min 1 \
  --count-max 10 \
  --train-steps 10000 \
  --eval-examples-per-count 1000 \
  --probe-examples-per-count 1000 \
  --steering-examples-per-count 300
```

The final report should make one of these three conclusions:

```text
A. Strong steering:
   A localized anchor/layer/direction produces monotonic count shifts.

B. Patching but weak steering:
   Interchange patching moves predictions, but linear steering directions do not.

C. Probe-only:
   Count is decodable, but neither steering nor patching shows a clear causal effect.
```

Do not hide negative results. For v4, a complete null steering grid is a valid outcome.
