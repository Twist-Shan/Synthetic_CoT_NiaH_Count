# v16_2: three-character Tiny Shakespeare counting

`synthetic_counting_v16_2` is an isolated successor to v16. It does not change v16
data, rendering, checkpoints, or reports. The only reused implementation is the
task-independent causal Transformer core.

## Data preparation

Run `prepare` before training. Preparation fingerprints Tiny Shakespeare and creates
contiguous train/validation/test regions separated by `seq_len - 1` character guards.
It computes character frequencies from the training region only, enumerates distinct
three-character sets whose summed frequency is at most
`needle_pool_frequency_threshold`, and samples `needle_pool_size` unique sets across
approximately uniform frequency bins. The split, pool, and all fixed evaluation
manifests are persisted under the run's `data/` directory.

Because an extremely rare training character can be absent from a shorter held-out
region, evaluation samples sets uniformly from that region's viable pool subset (sets
with at least one native occurrence). Training still samples the complete pool uniformly.
`tables/regional_pool_viability.csv` records every set/region viability decision; no
held-out frequencies are used to construct the pool.

The stored `count_max_threshold` is the only mutable count maximum. `cfg.count_max` is
a read-only compatibility property and serialized alias; loading rejects disagreeing
values.

## Training examples

For each example, sample one untouched `seq_len`-character training window. With
probability `1 - task_occurrence_ratio`, use the raw window directly. Otherwise select
one set uniformly from the pool and accept the initial window—or resample only the
window—when its union count satisfies `1 <= n <= count_max_threshold`.

Raw examples contain exactly the corpus characters and provide `seq_len - 1` shifted
next-character targets. Task examples use:

```text
nonthinking: <BOS> <CountChar> A B C <Sep> prompt <Ans> <n> <EOS>
thinking:    <BOS> <CountChar> A B C <Sep> prompt
             <Think> <1> M_1 ... <n> M_n </Think> <Ans> <n> <EOS>
```

The prompt remains an untouched contiguous corpus slice. `M_k` is the actual kth
left-to-right prompt character belonging to the set. The set members are distinct;
their prefix order is shuffled by default while the canonical set ID remains stable.

## Scheduled weighted training objective

Training permits two task-relevant weights:

- `final_count_loss_weight` applies to the final numeric answer token in both modes;
- `cot_trace_loss_weight` applies to thinking trace indices and marker characters only.

The objective is

```text
sum(weight_i * CE_i) / sum(weight_i)
```

over the currently active shifted targets. Both task weights default to 1.0.

`max_steps_for_language_pred` defaults to 1,500. Steps 1–1,500, inclusive, retain the
historical all-sequence objective: control tokens, task-prefix tokens, prompt characters,
and raw-language targets have unit weight unless covered by a task-specific weight.
Starting at step 1,501, training becomes task-output-only with inclusive mode-specific
starts:

```text
nonthinking: <Ans> <count> <EOS>
thinking:    <Think> <trace> </Think> <Ans> <count> <EOS>
```

Raw examples and targets before those starts have zero objective weight. Trace and
final-count weights remain active within the output spans. If a sampled post-transition
batch contains no task example, it is deterministically resampled rather than applying an
AdamW weight-decay-only update. A zero task ratio is invalid when the task-output phase
would occur. The learning-rate schedule and optimizer state do not restart at the phase
boundary.

Fixed-suite cross-entropies, perplexities, and component losses remain unweighted and
full-sequence for comparability. Training tables identify the active phase, objective
token count, and weighted shares. Thinking remains teacher-forced during training, so the
schedule removes language competition without itself eliminating trace exposure bias.

## Optimizer regularization

Training uses AdamW with configurable `weight_decay`, defaulting to `0.01`. The generated
notebook exposes this as `WEIGHT_DECAY`; setting it to `0.0` disables decay. The current
optimizer passes all model parameters in one group, so the coefficient applies to every
trainable parameter, including embeddings, biases, and LayerNorm parameters. Excluding
biases or normalization parameters would change the optimizer definition and is outside
this revision.

Weight decay does not add dropout or early stopping. The existing `0.01` setting did not
prevent held-out language loss from deteriorating after approximately 1,000 steps in the
reported v16_2 runs. Checkpoint selection must therefore continue considering held-out
loss together with autoregressive task accuracy. A different default decay should be
chosen only after a controlled sweep.

## Evaluation and loss curves

Three fixed suites—raw, task, and ratio-matched mixture—are built from both the training
region and held-out validation region. They are evaluated at step 0, every `eval_every`,
and the final step for every available RoPE/RPE and nonthinking/thinking model.

`eval_examples_per_count` controls balanced fixed-suite size. Its main default is 100,
so counts 1–10 produce 1,000 examples per raw/task/mixture suite. A periodic checkpoint
evaluates three train suites, three held-out suites, and the held-out behavioral task
suite—approximately 7,000 teacher-forced sequences per enabled model at the default.
This setting also determines final-test suite size; it does not change the separately
configured autoregressive subset.

The primary curve is the equal-input-sequence mean:

```text
CE_i = total active next-token NLL in sequence i / active targets in sequence i
curve CE = mean_i CE_i
```

Token-weighted cross-entropy is also saved as a secondary training-objective diagnostic.
The primary figure contains raw, task, and mixture panels with train and held-out lines.
The corpus test region is evaluated only at the final checkpoint and saved separately;
repeatedly viewed “test curves” use the validation region and are labeled held-out/test.

## Commands

```bash
python -m synthetic_counting_v16_2.run_v16_2 --preset debug --stage all --device cpu

python -m synthetic_counting_v16_2.run_v16_2 \
  --preset main --stage all --device cuda \
  --task-occurrence-ratio 1.0 \
  --count-max-threshold 10 \
  --weight-decay 0.01 \
  --final-count-loss-weight 1.0 \
  --cot-trace-loss-weight 1.0 \
  --model-variant rope/thinking \
  --model-variant rpe/thinking \
  --train-steps 10000 \
  --max-steps-for-language-pred 1500 \
  --checkpoint-every 500 \
  --eval-examples-per-count 100 \
  --skip-completed
```

The generated Colab notebook exposes both loss weights, `WEIGHT_DECAY`,
`MAX_STEPS_FOR_LANGUAGE_PRED`, four independent model switches, the maximum steps per
enabled model, and evaluation examples per count. At least one model switch must be
enabled. New configs default to the 1,500-step boundary. Legacy v16_2 configs missing the
field use their saved `train_steps` as the boundary, preserving historical all-sequence
training. Other missing legacy fields retain unit task weights, weight decay `0.01`, and
the former position-encoding x mode Cartesian product.

Stages are `prepare -> train -> attention -> state -> plots`. A train or analysis stage
refuses to start when the split, pool, or fixed-suite manifests are missing or when any
fingerprint differs from the configuration/checkpoint.

## Revision-5 checkpoint protocol

The main preset now uses `checkpoint_every=500`. Training writes an initialization
snapshot at step 0, cadence snapshots, an exact objective-boundary snapshot even when
`max_steps_for_language_pred` is not divisible by the cadence, an exact final numeric
snapshot, and the legacy `final` alias. The notebook exposes the cadence as
`CHECKPOINT_EVERY_STEPS`; new default run names include `ckptN`, preventing a 500-step
run from reusing a historical directory with an otherwise identical configuration.
Numeric snapshots are the source of training-dynamics analysis;
the `final` alias is not counted twice. Historical configurations still deserialize with
their stored cadence (typically 1,000), so their available timeline is necessarily
sparser. Keeping all snapshots increases local and Drive storage roughly in proportion
to checkpoint count.

Run the v16_2-specific, resumable orchestration layer after training:

```bash
python scripts/analyze_v16_2_checkpoint_dynamics.py RUN_DIR --device cuda
python scripts/analyze_v16_2_checkpoint_dynamics.py RUN_DIR --device cuda --skip-generated
```

The analyzer validates configuration, vocabulary, pool, split, variant, and checkpoint
step before metric collection. It processes only enabled model variants, loads
checkpoints sequentially, writes an atomic partition under
`analysis/checkpoint_dynamics/parts/`, and resumes completed partitions unless `--force`
is passed. Independent `--skip-attention`, `--skip-states`, `--skip-generated`,
`--skip-counterfactual`, and `--skip-similarity` controls make failures easier to isolate.
The notebook defaults are 20 attention examples/count (10 head selection plus 10 disjoint
reporting), 10 autoregressive examples/count, 40 train examples/count for probe fitting,
and 15 held-out examples/count for state evaluation.

### Attention definitions

At `<Ans>`, both modes report total prompt mass, matching-needle mass, non-needle prompt
mass, needle attention as a fraction of prompt mass, and enrichment relative to the
uniform prompt baseline `n / prompt_length`. Top-n recall and precision use the n largest
prompt-attention positions; needle entropy is normalized by `log(n)`. Thinking additionally
reports mass routed to trace indices/markers at `<Ans>`.

At every thinking trace-index query k, the analyzer reports attention mass on the true kth
prompt needle, whether that needle is top-1 among all n true needles, the `1/n` chance
baseline, top-1-minus-chance, and diagonal dominance (correct-k mass divided by total
needle mass). Ordered trace metrics do not apply to nonthinking. Fixed role heads are
selected only from the 10 selection examples at the final checkpoint and tracked on the
10 held-out reporting examples at every step. Tables retain all layers/heads so a plotted
aggregate can be reconstructed.

### Hidden-state definitions

Layer 0 is the token/position embedding output; layers 1 through `n_layer` are block
outputs. Gold teacher-forced sites are `<Ans>` in both modes and every trace index/marker
in thinking. Nearest-centroid and standardized ridge probes fit only on fixed
training-region task examples and evaluate only on fixed validation-region examples.
Trace labels are progress k; final-answer labels are true counts n.

Geometry includes centroid PC1/count R-squared, adjacent-direction cosine consistency,
PC1 and PC1–PC6 variance, effective dimension, adjacent-centroid distance, and monotonic
order violations. Thinking cross-site transfer applies the trace-progress ridge direction
to answer states and the answer-count direction to trace states, reporting MAE, R-squared,
slope/intercept, and direction cosine. Linear CKA uses the same aligned held-out examples
to compare each site/layer with its previous and final checkpoint.

Generated-prefix states are collected only when generation reaches `<Ans>`; malformed
generation receives an explicit status and no invented state. The counterfactual is
thinking-only and, for true count `n>=2`, removes exactly the final `<n> marker` trace pair
while preserving the task prefix, prompt, close/answer tokens, and true final target. It
reports changes in gold probability, gold-versus-`n-1` logit margin, and the gold-trained
ridge count projection relative to the complete trace.

Interpretation is deliberately limited. Attention is correlational, probe decoding is
not evidence of causal use, teacher forcing can mask generated-trace exposure bias,
position can leak trace progress, and searching many heads creates selection bias. The
disjoint fixed-head split, generated-prefix condition, and counterfactual condition help
separate these concerns but do not eliminate them.

### Revision-5 artifacts

Attention tables are `checkpoint_attention_detail.csv`,
`checkpoint_attention_summary.csv`, `checkpoint_attention_by_count.csv`,
`checkpoint_attention_by_k.csv`, `checkpoint_head_stability.csv`, and
`checkpoint_attention_behavior_link.csv`. Behavioral generation is saved in
`checkpoint_dynamics_autoregressive.csv`. State tables are
`checkpoint_state_probe_summary.csv`, `checkpoint_state_by_count.csv`,
`checkpoint_state_geometry.csv`, `checkpoint_state_cross_site.csv`,
`checkpoint_counterfactual_trace_readout.csv`, `checkpoint_generated_state_status.csv`,
and `checkpoint_state_similarity.csv`.

Figures are `checkpoint_attention_retrieval_emergence.png`,
`checkpoint_answer_routing.png`, `checkpoint_nonthinking_needle_coverage.png`,
`checkpoint_head_role_stability.png`, `checkpoint_final_count_probe_heatmap.png`,
`checkpoint_trace_progress_probe_heatmap.png`,
`checkpoint_cross_site_counter_transfer.png`,
`checkpoint_counterfactual_trace_readout.png`,
`checkpoint_state_geometry_emergence.png`,
`checkpoint_representation_stability.png`, `checkpoint_ordered_trace_retrieval.png`, and
`checkpoint_mechanism_overview.png`. These are derived artifacts: archive them before
deleting checkpoints if later recomputation is not required. The historical final-only
`attention_*.csv` and `state_*.csv` files do not contain training dynamics.

## Runtime instrumentation

`tables/runtime_events.csv` is the common atomic log for pipeline stages, optimizer
intervals, periodic teacher-forced and autoregressive evaluation, checkpoint writes,
final tests, dynamics metric families, aggregation, and plotting. Deterministic event IDs
deduplicate reruns; status distinguishes complete, failed, and cached work. Rows include
scope, block, variant, step, UTC start/end, elapsed seconds, examples/batches, device,
peak CUDA memory when available, cache state, and error type. CUDA is synchronized at
the beginning and end of timed GPU blocks, not on every batch. Console messages expose
the same `[timing:start]` and `[timing:done]` boundaries.

`runtime_breakdown.png` aggregates completed rows by scope/block. Optimizer intervals
exclude periodic evaluation and checkpoint serialization, allowing slow training to be
distinguished from a large fixed suite or Drive I/O. Timings involving Colab and Google
Drive depend on the active runtime, storage cache, and network conditions.
