# Codex Prompt: Trace Count v3.2, Causal Tests for v2 Attention Mechanisms

Implement v3.2 as an analysis-only causal-testing pipeline for the existing v2 marker-trace experiment.

Do **not** train a new v3.2 model. Do **not** replace the v2 GPT-2 learned-absolute-position architecture. v3.2 should load the already trained v2 thinking checkpoint and test which candidate mechanisms actually affect model behavior.

This pipeline is motivated by the v3 attention deep-dive report:

```text
colab_results/v3_v2_attention_deepdive_seed1234_20260708_053824/syn_v3_report.html
```

The report found:

- On the final trace index token, `L3H3` strongly attends to the final prompt needle:
  - correct final prompt needle attention mass about `0.849`;
  - top-1 retrieval rate `1.000`;
  - previous-index + previous-marker mass about `0.007`.
- `L3H1` is a secondary retrieval-like head.
- `L2H3` is the strongest local trace / plus-one-like head:
  - plus-one score about `0.336`, mostly previous-marker attention.
- Single-head ablation had weak behavioral effect:
  - ablating `L3H3` barely changed final answer accuracy;
  - ablating `L3H3 + L3H1` reduced trace exactness but not catastrophically;
  - ablating `L2H3` had little effect.

Therefore v3.2 should answer the sharper causal question:

> Which model components are necessary or sufficient for trace generation and final count prediction, and are the attention heads diagnostic, redundant, or causally active?

The notebook should be named:

```text
notebooks/Trace_Count_v3_2_Colab.ipynb
```

The main code package can be:

```text
synthetic_niah_v3_2/
```

or, if simpler, v3.2 can be implemented as notebook-local code plus reusable scripts. Prefer modular code if the implementation grows beyond one notebook.

---

## 0. Scientific Questions

v3.2 should separate **correlation**, **necessity**, and **sufficiency**.

### Q1. Are retrieval heads causally necessary?

Attention says `L3H3` and `L3H1` retrieve prompt needles. But ablation suggests redundancy. Test whether removing or weakening these heads changes:

- generated trace marker sequence;
- final count answer;
- answer logit margin;
- trace exactness;
- probability of malformed trace.

### Q2. Are retrieval heads sufficient to transfer correct information?

Patch only retrieval-head outputs from a clean run into a corrupted run. If this restores marker generation or final answer logits, then the head carries causally useful information.

### Q3. Is the model using prompt retrieval, local trace continuation, or both?

Compare interventions on:

- retrieval heads: `L3H3`, `L3H1`, `L3H2`, `L4H1`;
- local trace heads: `L2H3`, `L4H2`, `L1H0`;
- control heads with low retrieval and low plus-one score.

### Q4. What token positions matter?

Patch/ablate at specific positions:

- `index_token_k`, especially final `index_token_n`;
- `marker_token_k`;
- `pre_index_k`;
- `think_end`;
- `<Ans>` position.

Do not average over positions prematurely. The causal effect can be position-specific.

### Q5. Is the final answer read from trace length/index tokens, retrieved prompt needles, or residual counter state?

Use counterfactual traces and prompt edits to distinguish:

- final answer driven by the generated indexed trace;
- final answer driven by prompt needle count;
- final answer driven by local transition sequence;
- final answer driven by residual hidden-state count geometry.

---

## 1. Source Artifacts

Use v2 artifacts from one of these locations:

```text
runs/v2_marker_trace_seed1234_main
runs/v2_marker_trace_seed1234_debug
colab_results/v2_marker_trace_*_seed*/run
colab_results/v2_marker_trace_main_seed1234_20260706_215757/run
a user-specified V2_RUN_DIR_OVERRIDE
```

Expected v2 checkpoint layout:

```text
{v2_run_dir}/checkpoints/final/thinking/
{v2_run_dir}/checkpoints/final/non_thinking/
{v2_run_dir}/checkpoints/final/vocab.json
{v2_run_dir}/config.json or config.yaml
```

Load the thinking model with eager attention:

```python
GPT2LMHeadModel.from_pretrained(thinking_model_dir, attn_implementation="eager")
```

Fallback:

```python
model = GPT2LMHeadModel.from_pretrained(thinking_model_dir)
model.config._attn_implementation = "eager"
```

Also load v3 deep-dive tables if available:

```text
{v3_report_dir}/analysis/tables/head_summary.csv
{v3_report_dir}/analysis/tables/last_index_head_summary.csv
{v3_report_dir}/analysis/tables/head_ablation_results.csv
```

If these files are missing, recompute the minimal head summary from v2 examples.

---

## 2. Keep v2 Data and Model Semantics

v3.2 must use the v2 setup:

- GPT-2 style decoder-only Transformer.
- Learned absolute positional embeddings.
- No RoPE.
- Random-init model already trained in v2.
- Prompt length usually `seq_len = 256`.
- Count range `1..10`.
- Marker tokens `<A>` ... `<J>`.
- Noise tokens `<N0>` ... `<N63>`.
- Thinking sequence format:

```text
<BOS> prompt_tokens
<Think/> <1> marker_1 <2> marker_2 ... <n> marker_n
</Think> <Ans> <n> <EOS>
```

Layer/head convention:

- report layer numbers as **1-based** for readability;
- use Hugging Face module indices as **0-based** internally;
- head numbers are **0-based**.

When reporting `L3H3`, this means:

```python
module layer index = 2
head index = 3
```

unless the existing v3 code already stores layer numbers as 1-based. The notebook must explicitly state the convention and convert consistently.

---

## 3. Core Candidate Heads

Use the v3 report as the initial candidate list.

### Retrieval-like heads

```python
RETRIEVAL_HEADS = [
    (3, 3),  # strongest final-index retrieval
    (3, 1),  # secondary retrieval
    (4, 1),  # later retrieval-like head
    (3, 2),
]
```

### Local trace / plus-one-like heads

```python
PLUS_ONE_HEADS = [
    (2, 3),  # strongest previous-marker / local trace head
    (4, 2),  # previous-index-like head
    (1, 0),
]
```

### Control heads

Choose control heads from low retrieval and low plus-one scores, for example:

```python
CONTROL_HEADS = [
    (2, 0),
    (1, 2),
]
```

The notebook should automatically select heads from `last_index_head_summary.csv` when available:

- top-k by `correct_prompt_needle_mass`;
- top-k by `plus_one_score`;
- bottom-k by both scores as controls.

Use hard-coded defaults only as fallback.

---

## 4. Intervention Infrastructure

Implement hooks for GPT-2 attention head outputs.

### 4.1 Head output hook point

For Hugging Face `GPT2Attention`, the easiest robust hook is usually the input to `attn.c_proj`, because at that point per-head outputs have already been concatenated:

```python
def c_proj_pre_hook(module, inputs):
    hidden = inputs[0]  # [batch, seq, n_embd]
    head_view = hidden.view(batch, seq, n_head, head_dim)
    ...
```

This allows:

- zero-ablation of selected heads;
- mean-ablation;
- scaling head output by alpha;
- replacing head output at selected token positions with cached clean/corrupt values.

The code must validate:

```python
n_embd == n_head * head_dim
```

### 4.2 Position masks

Support intervention positions:

```python
POSITION_SCOPES = [
    "all_positions",
    "trace_positions",
    "index_token_all",
    "index_token_last",
    "marker_token_all",
    "marker_token_last",
    "pre_index_all",
    "pre_index_last",
    "think_end",
    "ans_token",
]
```

For each rendered teacher-forced sequence, build a metadata object containing:

```python
{
  "prompt_needle_positions": list[int],
  "trace_index_positions": list[int],
  "trace_marker_positions": list[int],
  "pre_index_positions": list[int],
  "think_start_pos": int,
  "think_end_pos": int,
  "ans_pos": int,
  "count_pos": int,
}
```

### 4.3 Intervention types

Implement at least:

```python
INTERVENTIONS = [
    "zero_ablation",
    "mean_ablation",
    "scale",
    "clean_to_corrupt_patch",
    "corrupt_to_clean_patch",
    "attention_pattern_patch",
    "value_output_patch",
]
```

If `attention_pattern_patch` is hard to implement robustly, mark it optional and implement `value_output_patch` first.

### 4.4 Residual stream patching

Also implement residual-stream patching at selected positions:

```python
RESIDUAL_PATCH_SITES = [
    "after_block",
    "mlp_output",
    "attn_output",
]
```

Minimum required:

- patch residual after a block at `index_token_last`;
- patch residual after a block at `think_end`;
- patch residual after a block at `<Ans>`.

This tests whether information is localized in the residual stream even if single-head ablations are weak.

---

## 5. Datasets for Causal Tests

Use small but balanced datasets so causal tests are fast.

Default:

```yaml
causal_examples_per_count: 40
patch_pairs_per_count: 40
max_count: 10
seed: 1234
```

Debug:

```yaml
causal_examples_per_count: 5
patch_pairs_per_count: 5
max_count: 10
seed: 1234
```

### 5.1 Balanced base examples

Generate base examples exactly as v2:

- count `n` in `1..10`;
- prompt length `seq_len`;
- marker positions sampled without replacement;
- marker types sampled from `<A>` ... `<J>`;
- noise sampled from `<N0>` ... `<N63>`.

### 5.2 Clean/corrupt pair types

Create paired examples for patching. Each pair should include:

```python
{
  "pair_id": str,
  "pair_type": str,
  "clean_example": BaseExample,
  "corrupt_example": BaseExample,
  "clean_target_count": int,
  "corrupt_target_count": int,
  "expected_effect": str,
}
```

Required pair types:

#### A. Count decrement: remove last needle

Clean has count `n`; corrupt removes the final prompt needle, so count is `n-1`.

Purpose:

- If a component carries final prompt needle evidence, clean-to-corrupt patching should push the corrupt answer upward toward `n`.
- Corrupt-to-clean patching should push clean answer downward toward `n-1`.

Use only `n >= 2`.

#### B. Count increment: add one final needle

Clean has count `n`; corrupt has count `n+1` by adding a final needle at a later prompt position.

Purpose:

- Tests whether the model can be causally pushed to include an extra final needle.

Use only `n <= 9`.

#### C. Last-two-position swap

Keep count fixed but swap the positions of the last two prompt needles while preserving marker identities.

Purpose:

- Tests whether retrieval follows prompt position/order, not just marker token identity.

#### D. Last-marker identity swap

Keep count and positions fixed but replace the last marker type with another marker type.

Purpose:

- Tests whether retrieval head output carries marker identity information needed for the next trace marker.

#### E. Trace-prefix corruption

Keep prompt fixed, but teacher-force an incorrect trace prefix:

- wrong final index token;
- wrong previous marker token;
- shuffled last two trace marker tokens.

Purpose:

- Tests whether final answer and marker generation rely more on trace prefix or prompt retrieval.

### 5.3 Saturation note

The v2 model has near-perfect final answer accuracy. Therefore accuracy alone will be insensitive.

Always report logit-level metrics:

```text
gold_logit
corrupt_gold_logit
logit_margin = logit(clean_target) - logit(corrupt_target)
prob_clean_target
prob_corrupt_target
count_expectation
pred_count
```

Behavioral metrics are still useful, but logit margin is the primary causal effect metric.

---

## 6. Causal Test A: Expanded Head Necessity Scan

Run autoregressive generation under head ablation and under teacher-forced final-answer readout.

### Conditions

At minimum:

```python
HEAD_ABLATION_CONDITIONS = {
    "baseline_no_ablation": [],
    "retrieval_L3H3": [(3, 3)],
    "retrieval_L3H1": [(3, 1)],
    "retrieval_L3H3_L3H1": [(3, 3), (3, 1)],
    "retrieval_top4": RETRIEVAL_HEADS,
    "plus_one_L2H3": [(2, 3)],
    "plus_one_L4H2": [(4, 2)],
    "plus_one_top3": PLUS_ONE_HEADS,
    "retrieval_plus_one_top": [(3, 3), (3, 1), (2, 3), (4, 2)],
    "low_score_controls": CONTROL_HEADS,
}
```

### Position scopes

Run each ablation under at least:

```python
["all_positions", "index_token_last", "index_token_all", "trace_positions"]
```

For autoregressive generation, position-specific intervention requires applying hooks dynamically as tokens are generated. If this is too complex, implement:

- teacher-forced position-specific ablation first;
- autoregressive all-position ablation second.

### Metrics

Save:

```text
v3_2_causal/tables/head_necessity_results.csv
```

Columns:

```text
condition, intervention_type, heads, position_scope,
eval_mode, n_examples,
answer_accuracy, invalid_rate,
trace_exact_match_rate, trace_marker_recall, trace_index_accuracy,
mean_gold_logit_margin, mean_answer_ce,
mean_count_shift, under_rate, over_rate
```

Plot:

```text
v3_2_causal/figures/head_necessity_answer_margin.png
v3_2_causal/figures/head_necessity_trace_exact.png
v3_2_causal/figures/head_necessity_count_shift.png
```

Interpretation:

- A head is **necessary** only if ablating it reliably changes logit margin or behavior beyond controls.
- If attention is strong but ablation effect is small, call it diagnostic or redundant, not necessary.

---

## 7. Causal Test B: Dose-Response Head Scaling

Instead of only zeroing a head, scale selected head outputs:

```python
ALPHAS = [-2, -1, -0.5, 0, 0.25, 0.5, 1, 1.5, 2]
```

For each candidate head and head group:

- multiply selected head output by `alpha`;
- evaluate final-answer logits and generated trace.

Required heads/groups:

```python
[
  "L3H3",
  "L3H1",
  "L3H3_L3H1",
  "L2H3",
  "retrieval_top4",
  "plus_one_top3",
  "control_top2",
]
```

Save:

```text
v3_2_causal/tables/head_dose_response.csv
```

Plots:

```text
v3_2_causal/figures/head_dose_response_answer_margin.png
v3_2_causal/figures/head_dose_response_trace_exact.png
```

Interpretation:

- A smooth monotonic dose-response is stronger causal evidence than one-off ablation.
- Non-monotonic effects suggest distribution shift from intervention or redundant routing.

---

## 8. Causal Test C: Clean-to-Corrupt Activation Patching

This is the main sufficiency test.

### 8.1 Clean/corrupt forward passes

For each pair:

1. Render clean teacher-forced sequence.
2. Render corrupt teacher-forced sequence.
3. Cache clean and corrupt activations at:
   - selected head outputs;
   - attention outputs;
   - MLP outputs;
   - residual stream after each block.

### 8.2 Patch directions

Run:

```python
clean_to_corrupt_patch
corrupt_to_clean_patch
```

For clean-to-corrupt:

- base run is corrupt;
- selected activation from clean is inserted;
- measure whether logits move toward clean target.

For corrupt-to-clean:

- base run is clean;
- selected activation from corrupt is inserted;
- measure whether logits move toward corrupt target.

### 8.3 Patch sites

Required head-output patch sites:

```python
[
  ("head_output", "L3H3", "index_token_last"),
  ("head_output", "L3H1", "index_token_last"),
  ("head_output", "L3H3_L3H1", "index_token_last"),
  ("head_output", "L2H3", "pre_index_last"),
  ("head_output", "L2H3", "index_token_last"),
  ("head_output", "retrieval_top4", "index_token_last"),
  ("head_output", "plus_one_top3", "pre_index_all"),
]
```

Required residual patch sites:

```python
[
  ("resid_after_block", layer, "index_token_last") for layer in all_layers
  ("resid_after_block", layer, "think_end") for layer in all_layers
  ("resid_after_block", layer, "ans_token") for layer in all_layers
]
```

### 8.4 Patch metrics

For each patched run, save:

```text
clean_target_logit
corrupt_target_logit
clean_minus_corrupt_margin
base_margin
patched_margin
margin_delta = patched_margin - base_margin
normalized_recovery =
  (patched_margin - corrupt_margin) / (clean_margin - corrupt_margin + eps)
pred_count
count_shift
trace_exact_match
trace_marker_recall
trace_index_accuracy
```

Save:

```text
v3_2_causal/tables/activation_patching_results.csv
```

Plots:

```text
v3_2_causal/figures/patching_recovery_by_layer_position.png
v3_2_causal/figures/patching_recovery_by_head_group.png
v3_2_causal/figures/patching_count_shift_by_pair_type.png
```

Interpretation:

- If `L3H3` head-output patch at `index_token_last` recovers marker or count logits, this supports sufficiency of that retrieval head output.
- If residual patching at `think_end` or `<Ans>` recovers final count but head patching does not, the count signal may be distributed in residual stream rather than localized in one head.

---

## 9. Causal Test D: Path Patching

Path patching should test whether the path:

```text
prompt final needle -> L3H3 at final index token -> downstream trace/final answer
```

is causally active.

### Minimal path patching

Implement:

1. Cache clean and corrupt attention head outputs.
2. Patch only selected head output at selected query position.
3. Keep all other activations from the base run.

Required paths:

```python
PATHS = [
  "final_prompt_needle_to_L3H3_index_last_to_marker_last",
  "final_prompt_needle_to_L3H3_index_last_to_think_end",
  "final_prompt_needle_to_L3H3_index_last_to_ans",
  "previous_marker_to_L2H3_pre_index_last_to_index_last",
  "previous_index_to_L4H2_index_last_to_ans",
]
```

For each path, measure:

- last marker logit recovery;
- final answer logit recovery;
- trace correctness recovery;
- effect size relative to control paths.

Save:

```text
v3_2_causal/tables/path_patching_results.csv
```

Plots:

```text
v3_2_causal/figures/path_patching_recovery_heatmap.png
```

### Optional: attention-pattern vs value-output patch

If possible, separate:

- attention pattern patch: use clean attention probabilities with corrupt values;
- value patch: use corrupt pattern with clean values;
- full head-output patch.

This tests whether L3H3 matters because of where it attends, or because of value vectors carried from the attended prompt needle.

Save:

```text
v3_2_causal/tables/pattern_vs_value_patch_results.csv
```

---

## 10. Causal Test E: Counterfactual Prompt and Trace Edits

These tests should be run without activation patching first, then with patching.

### E1. Prompt last-needle deletion

Remove last prompt needle but keep the old gold trace in teacher forcing.

Question:

- Does final answer follow prompt count or trace count?

Metrics:

- final answer logits after `<Ans>`;
- marker logits after final index;
- attention from L3H3 to deleted/old position.

### E2. Prompt last-needle insertion

Add a final prompt needle but keep old trace.

Question:

- Does final answer notice the extra prompt needle without trace support?

### E3. Swap last two prompt needle positions

Swap positions of the last two prompt needles while preserving trace order or while updating trace order.

Question:

- Does L3H3 follow order/position or marker identity?

### E4. Wrong final index token

Teacher force:

```text
... <n-1> marker_{n-1} <wrong_index> marker_n ...
```

Question:

- Does marker retrieval depend on the numeric index token identity?

### E5. Wrong previous marker token

Teacher force previous marker as wrong token while keeping index sequence correct.

Question:

- Does local plus-one head drive index continuation or marker generation?

Save:

```text
v3_2_causal/tables/counterfactual_edit_results.csv
```

Plots:

```text
v3_2_causal/figures/counterfactual_prompt_vs_trace_logits.png
v3_2_causal/figures/counterfactual_l3h3_attention_shift.png
```

---

## 11. Causal Test F: Residual Count Direction and Readout

Use existing v2/v4-style count-direction tools if available, but keep this section secondary.

Fit simple linear directions on teacher-forced hidden states:

- ridge direction predicting final count at `<Ans>`;
- ridge direction predicting prefix count at `index_token_k`;
- unembedding adjacent direction `<k+1> - <k>`.

Then test:

1. Does projecting residual states along the direction correlate with count?
2. Does adding/subtracting the direction at `<Ans>` shift final count logits?
3. Does adding/subtracting the direction at `index_token_last` affect trace marker or final answer?

Save:

```text
v3_2_causal/tables/residual_direction_results.csv
v3_2_causal/tables/residual_steering_results.csv
```

Plots:

```text
v3_2_causal/figures/residual_count_direction_projection.png
v3_2_causal/figures/residual_steering_dose_response.png
```

Interpretation:

- A direction that predicts count is not necessarily causal.
- A direction is causal only if intervention shifts logits/behavior in the predicted direction under controls.

---

## 12. Metrics and Tables

All tables should be tidy CSV with enough metadata to plot later.

### Common columns

Every result row should include:

```text
experiment_name
run_id
source_v2_run_dir
seed
example_id
pair_id
pair_type
count_clean
count_corrupt
intervention_type
intervention_name
heads
layer
head
position_scope
token_position_name
eval_mode
```

### Behavioral metrics

```text
pred_count
gold_count
answer_accuracy
invalid
trace_exact_match
trace_marker_recall
trace_index_accuracy
under
over
count_shift
```

### Logit metrics

```text
gold_logit
clean_target_logit
corrupt_target_logit
logit_margin
base_logit_margin
patched_logit_margin
margin_delta
normalized_recovery
answer_ce
count_distribution_entropy
```

### Attention diagnostics

```text
correct_prompt_needle_mass
all_prompt_needles_mass
previous_index_token_mass
previous_marker_token_mass
plus_one_score
retrieval_score
```

---

## 13. Required Figures

Generate clear figures with Chinese captions in the notebook. Save PNGs to:

```text
{v2_run_dir}/v3_2_causal/figures/
```

Required:

```text
head_necessity_answer_margin.png
head_necessity_trace_exact.png
head_dose_response_answer_margin.png
patching_recovery_by_head_group.png
patching_recovery_by_layer_position.png
path_patching_recovery_heatmap.png
counterfactual_prompt_vs_trace_logits.png
counterfactual_l3h3_attention_shift.png
residual_steering_dose_response.png
```

Each figure must state:

- x-axis definition;
- y-axis definition;
- color/group definition;
- which examples are included;
- what a positive/negative value means.

Avoid huge single-panel figures. Prefer grouped panels:

- retrieval vs plus-one vs controls;
- clean-to-corrupt vs corrupt-to-clean;
- low/mid/high count bins.

---

## 14. Notebook Structure

The notebook should be readable as a result report and runnable in Colab.

Suggested sections:

1. **Environment and Repo Setup**
   - install dependencies;
   - load repo;
   - set device;
   - set `V2_RUN_DIR_OVERRIDE`;
   - set `V3_REPORT_DIR_OVERRIDE`;
   - set output directory.

2. **Load Model and Reconstruct v2 Data**
   - load thinking checkpoint;
   - load/build vocab;
   - define render functions;
   - verify a sample sequence.

3. **Reproduce v3 Head Summary**
   - load v3 report tables if present;
   - show candidate heads;
   - explain L3H3/L3H1/L2H3.

4. **Causal Test A: Necessity via Head Ablation**
   - run or load table;
   - show answer margin and trace exactness.

5. **Causal Test B: Dose-Response Scaling**
   - run or load table;
   - show monotonicity or lack thereof.

6. **Causal Test C: Activation Patching**
   - build clean/corrupt pairs;
   - cache activations;
   - patch head/residual sites;
   - show recovery heatmaps.

7. **Causal Test D: Path Patching**
   - targeted path patching for L3H3 and L2H3;
   - compare to controls.

8. **Causal Test E: Counterfactual Prompt/Trace Edits**
   - prompt deletion/insertion/swap;
   - wrong index/wrong marker trace;
   - show whether model follows prompt, trace, or both.

9. **Causal Test F: Residual Count Direction**
   - optional but useful;
   - distinguish decodability from causal steering.

10. **Synthesis**
    - classify each candidate mechanism:
      - diagnostic only;
      - necessary;
      - sufficient;
      - redundant but causal;
      - unsupported.

11. **Save Results to Google Drive**
    - copy output directory to:

```text
/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/Synthetic_CoT_NiaH_Count/colab_results/
```

12. **Optional GitHub Push**
    - disabled by default.

13. **Optional Runtime Disconnect**
    - disabled by default unless user sets `AUTO_DISCONNECT = True`.

Do not require Google Drive login in every cell. Mount only if not already mounted.

---

## 15. Runtime Settings

Use explicit switches to avoid accidental long runs.

```python
PRESET = "debug"  # "debug" or "main"
RUN_NECESSITY = True
RUN_DOSE_RESPONSE = True
RUN_ACTIVATION_PATCHING = True
RUN_PATH_PATCHING = True
RUN_COUNTERFACTUALS = True
RUN_RESIDUAL_DIRECTIONS = False

SKIP_COMPLETED = True
SAVE_TO_DRIVE = True
AUTO_DISCONNECT = False
```

Debug defaults:

```python
EXAMPLES_PER_COUNT = 5
PATCH_PAIRS_PER_COUNT = 5
MAX_GENERATION_EXAMPLES = 100
ALPHAS = [-1, 0, 0.5, 1, 1.5]
```

Main defaults:

```python
EXAMPLES_PER_COUNT = 40
PATCH_PAIRS_PER_COUNT = 40
MAX_GENERATION_EXAMPLES = 400
ALPHAS = [-2, -1, -0.5, 0, 0.25, 0.5, 1, 1.5, 2]
```

The notebook should print expected runtime before running:

```text
Estimated interventions:
  necessity: ...
  dose-response: ...
  activation patching: ...
  path patching: ...
```

---

## 16. Implementation Cautions

### 16.1 Accuracy saturation

If baseline accuracy is near 1.0, do not conclude “no effect” from unchanged accuracy alone.
Use:

- answer logit margin;
- normalized recovery;
- trace exactness;
- marker recall;
- count-shift distribution.

### 16.2 Distribution shift from ablation

Zeroing heads can create out-of-distribution activations. Include:

- mean ablation;
- random/control head ablations;
- dose-response curves.

### 16.3 Attention is diagnostic

Attention mass should be used to select hypotheses, not as proof.
Causal claims require interventions.

### 16.4 Redundancy

If ablating a single head does little but patching it has effect, call it **sufficient but redundant**.
If ablation and patching both do little, call it **diagnostic only**.
If ablation hurts and patching recovers, call it **causally active and important**.

### 16.5 Position specificity

Do not only ablate heads across all positions. Test:

- all positions;
- last index only;
- all index tokens;
- trace positions;
- `<Ans>`/`think_end`.

### 16.6 Clean/corrupt pair validity

Before patching, verify:

- clean and corrupt both render correctly;
- target counts differ when expected;
- baseline model predicts both correctly before patching;
- if baseline is wrong, exclude pair from recovery metrics or mark it separately.

---

## 17. Acceptance Checks

The implementation is complete when:

### Data checks

- Can render v2 thinking examples exactly.
- Can locate all trace index/marker positions.
- Can locate final index token and `<Ans>` token.
- Clean/corrupt pairs are valid and balanced by count.

### Hook checks

- Head-output hook changes only selected heads.
- Position mask changes only selected token positions.
- `alpha = 1` scaling reproduces baseline logits to numerical tolerance.
- No-intervention patch reproduces baseline logits.
- Control head interventions are recorded.

### Causal table checks

The following exist:

```text
v3_2_causal/tables/head_necessity_results.csv
v3_2_causal/tables/head_dose_response.csv
v3_2_causal/tables/activation_patching_results.csv
v3_2_causal/tables/path_patching_results.csv
v3_2_causal/tables/counterfactual_edit_results.csv
```

Optional:

```text
v3_2_causal/tables/residual_direction_results.csv
v3_2_causal/tables/residual_steering_results.csv
```

### Figure checks

Required figures exist and have readable captions in the notebook.

### Interpretation checks

The final notebook conclusion must answer:

1. Is `L3H3` necessary?
2. Is `L3H3` sufficient to transfer final-needle information?
3. Are `L3H3` and `L3H1` redundant retrieval heads?
4. Does `L2H3` causally affect local trace continuation?
5. Does final answer follow prompt count, trace count, or both under counterfactual edits?
6. Are count directions merely decodable or causally steerable?

Use cautious language:

- “supports a causal role” only when patching/ablation changes logits or behavior relative to controls;
- “diagnostic attention pattern” when attention is strong but intervention effects are weak;
- “redundant causal pathway” when patching works but ablation is compensated.

---

## 18. Expected Possible Outcomes

### Outcome A: L3H3 is diagnostic but redundant

Evidence:

- strong attention to final prompt needle;
- single-head ablation small;
- patching L3H3 output causes some logit recovery;
- top retrieval group ablation has larger effect.

Interpretation:

```text
L3H3 carries final-needle information, but the model has redundant retrieval/readout paths.
```

### Outcome B: Retrieval group is necessary

Evidence:

- ablating retrieval_top4 reduces trace marker recall or answer margin;
- patching retrieval_top4 recovers corrupt examples.

Interpretation:

```text
Targeted prompt retrieval is a causal part of trace generation/final answer readout.
```

### Outcome C: Local trace continuation is causal for index generation, not final answer

Evidence:

- L2H3 ablation affects trace index accuracy or next-index logits;
- final answer remains stable.

Interpretation:

```text
Local plus-one-like heads help maintain the trace format but are not the main final-count readout.
```

### Outcome D: Final answer follows trace more than prompt

Evidence:

- counterfactual prompt edits do not change final answer if teacher-forced trace is unchanged;
- wrong trace prefix changes final answer despite correct prompt.

Interpretation:

```text
The model may use the completed trace as a computational scratchpad, with prompt retrieval mainly supporting trace generation.
```

### Outcome E: Final answer follows prompt more than trace

Evidence:

- prompt deletion/insertion changes final answer logits even with old trace;
- trace corruption has little effect.

Interpretation:

```text
The final readout may still directly use prompt-level count evidence, and the trace may be auxiliary.
```

---

## 19. Explicit Non-Goals

Do not add:

- new training regimes;
- new model architectures;
- RoPE;
- OOD splits;
- v4 steering grid as the primary experiment;
- v5 mixed thinking toggle;
- v6 separator trace training;
- natural-language prompts;
- new synthetic task variants.

v3.2 is a causal analysis layer over the existing v2 trained thinking model.

