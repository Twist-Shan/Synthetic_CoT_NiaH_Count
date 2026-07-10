# Codex Prompt: Trace Count v3, v2 Attention-Head Deep Dive

Rewrite v3 as an analysis-only notebook for the existing v2 marker-trace experiment.
Do not train a new v3 model. Do not use the old v3 RoPE/length-generalization
pipeline. The goal is to answer one mechanistic question from the v2 result:

> In the thinking trace, when the model is at the last trace number token
> such as `<3>` in `<Think/> <1> x <2> y <3> z </Think>`, is attention
> primarily retrieving the corresponding final prompt needle, or is it
> mostly looking at the previous trace number/marker and doing a local `+1`
> transition?

The notebook should be named `notebooks/Trace_Count_v3_Colab.ipynb`.

## Source Artifacts

Use v2 artifacts from one of these locations:

- `runs/v2_marker_trace_seed1234_main`
- `runs/v2_marker_trace_seed1234_debug`
- `colab_results/v2_marker_trace_*_seed*/run`
- a user-specified `V2_RUN_DIR_OVERRIDE`

The v2 checkpoint layout is:

```text
{v2_run_dir}/checkpoints/final/thinking/
{v2_run_dir}/checkpoints/final/non_thinking/
{v2_run_dir}/checkpoints/final/vocab.json
{v2_run_dir}/config.json
```

Load the thinking model with eager attention:

```python
GPT2LMHeadModel.from_pretrained(path, attn_implementation="eager")
```

Fallback to normal loading and set `model.config._attn_implementation = "eager"`
when needed.

## Keep v2 Architecture and Data Semantics

This notebook must use the v2 setup:

- GPT-2 style decoder-only Transformer from Hugging Face.
- Learned absolute positional embeddings, not RoPE.
- Prompt length usually `seq_len = 256`.
- Count range `1..10`.
- Marker tokens `<A>` ... `<J>`.
- Noise tokens `<N0>` ... `<N63>`.
- Thinking sequence format:

```text
<BOS> prompt_tokens <Think/> <1> marker_1 <2> marker_2 ... <n> marker_n </Think> <Ans> <n> <EOS>
```

The notebook should define the same v2 vocabulary, example generator, and render
logic so it can run independently after the v2 checkpoints are available.

## Main Analyses

### 1. Total Token Attention Decomposition

For the thinking model, run teacher-forced attention on generated v2 examples and
collect attention for three query anchors:

- `pre_index_k`: the token position immediately before trace index `<k>`.
  This is the position whose next-token logits generate `<k>`.
- `index_token_k`: the trace index token `<k>` itself.
  This is the position whose hidden state helps generate the marker after `<k>`.
- `marker_token_k`: the trace marker token following `<k>`.

For each query, layer, and head, decompose attention mass into interpretable token
groups:

- `correct_prompt_needle`: prompt needle `k`.
- `last_prompt_needle`: prompt needle `n`, especially when `k = n`.
- `previous_prompt_needle`: prompt needle `k-1`.
- `other_prompt_needles`.
- `prompt_noise`.
- `current_index_self`.
- `previous_index_token`.
- `previous_marker_token`.
- `earlier_trace_tokens`.
- `think_open`.
- `bos`.
- `other_context`.

Also compute non-disjoint summary metrics:

- `correct_prompt_needle_mass`
- `last_prompt_needle_mass`
- `previous_prompt_needle_mass`
- `all_prompt_needles_mass`
- `previous_index_token_mass`
- `previous_marker_token_mass`
- `all_previous_index_mass`
- `all_previous_marker_mass`
- `retrieval_score = correct_prompt_needle_mass`
- `plus_one_score = previous_index_token_mass + previous_marker_token_mass`

Save:

```text
{v2_run_dir}/v3_attention_head_deepdive/tables/token_attention_rows.csv
{v2_run_dir}/v3_attention_head_deepdive/tables/head_summary.csv
{v2_run_dir}/v3_attention_head_deepdive/tables/last_index_head_summary.csv
```

### 2. Final Trace Number Analysis

Filter to `is_last_index == True` and focus on `index_token_k`.

The notebook must explicitly answer:

- Does the strongest head attend to the final prompt needle?
- Does any head mostly attend to the previous trace number or previous trace marker?
- Across all heads/layers, is attention more retrieval-like or local-transition-like?

Required plots:

- Heatmap of last-index `correct_prompt_needle_mass` by layer/head.
- Heatmap of last-index `plus_one_score` by layer/head.
- Bar plot of category masses for the best retrieval head.
- Bar plot of mean category masses averaged over all heads/layers.
- Scatter plot of retrieval score vs plus-one score for all heads.

### 3. Head Ablation

Run a lightweight autoregressive ablation on the final thinking checkpoint.
Use the GPT-2 attention `c_proj` pre-hook to zero one or more head outputs.

Conditions:

- `baseline_no_ablation`
- best last-index retrieval head
- top-2 last-index retrieval heads
- best plus-one/local-transition head
- a low-score control head

Metrics:

- final answer accuracy
- invalid rate
- trace exact match rate
- trace marker recall
- trace index accuracy

Save:

```text
{v2_run_dir}/v3_attention_head_deepdive/tables/head_ablation_results.csv
{v2_run_dir}/v3_attention_head_deepdive/figures/head_ablation_results.png
```

## Notebook Presentation

The notebook should be readable as a result report:

- Use Chinese markdown explanations.
- Every figure must say what the x-axis, y-axis, colors/groups, and values mean.
- Make clear that `layer` is 1-based and `head` is 0-based.
- Do not render or generate an HTML report in this notebook.
- Include a Google Drive save cell at the end.
- Include an optional GitHub push cell, disabled by default.

## Interpretation Standard

The key interpretation should separate two possible mechanisms:

- If `pre_index_k` mainly attends to previous trace number/marker, that supports a
  local trace-continuation or `+1` route for generating the next index token.
- If `index_token_k`, especially the final `index_token_n`, attends strongly to
  prompt needle `n`, that supports targeted retrieval from the trace index token
  to the corresponding prompt needle.

Both can be true. The likely mechanistic story may be:

1. the model locally advances the trace index;
2. the current index token then acts as a query to retrieve the corresponding
   prompt marker/needle;
3. the retrieved marker is emitted next;
4. the final answer can be read out from the completed trace.

Do not overclaim causality from attention alone. Use ablation as a causal sanity
check, and state whether the head appears necessary, redundant, or merely
diagnostic.
