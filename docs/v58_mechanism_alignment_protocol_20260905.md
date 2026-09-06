# V58 mechanism-completeness and sample-alignment supplement

## Scope and status

Frozen before new inference. Keep both independently trained v58 step-10000
models, count 1–10, 256-character receiver inputs, original separator trace,
weights and training procedure unchanged. Do not tune for a positive outcome.
Sources are the current Non-thinking, Native-thinking and Geometry HTML reports;
their hashes and extracted prose are in `work/v58_alignment_audit_20260905`.

The audit distinguishes experiment coverage, protocol/sample alignment and
positive mechanistic replication. A measured null remains a completed assay.
Architecture-inapplicable contrasts must not be counted as successful replicas.

## Common sample contract

Use the existing external v58 behavior suite, exclude prior progress/continuation
prompts and canonical keys used in historical head selection, sort by prompt
hash within count. Freeze 20 examples/count discovery and 10/count confirmation
(200 + 100 prompts), exact same input multiset for both modes. No selection on
accuracy, NCC or intervention outcomes. This is a registered existing-suite
extension, not a pristine unseen behavioral test. The remaining source pool and
any overlap with earlier mechanism assays must be disclosed.

Within each count, the first 20 retained prompts form discovery and the next 10
form confirmation. Assign an index across counts as a balanced panel block.
Ten confirmation blocks each contain ten distinct prompts. Blocks are a
predefined statistical grouping, NOT ten independently trained model seeds or
the same text regenerated at ten counts. Repeated arms, layers and donor pairs
are paired within block; report both 100 prompt units and ten block units.
All mode-comparable assays require exact realized key and arm multiplicity
equality. Native experiments generally use 20 discovery/10 confirmation seeds,
often counts 1–10; synthetic matches the 200/100 input budget but not natural
language stimulus generation or the number of model seeds.

## Missing primary families to run

1. Source formation/use: equal-length needle-token input replacement, equal-token
   ordinary replacement, clean source-state restoration at embedding and each of
   four post-block depths; frozen clean-centroid retention at prompt occurrences;
   answer accuracy, candidate margin and expected-count absolute error. A
   synthetic needle is one token: full-span and endpoint restoration coincide.
2. Common answer-state layer sweep: self, adjacent-count donor, same-count
   different-context donor at all four depths; report unrestricted pairs and
   clean-correct common support separately. Also count-centroid rank-3 removal
   versus within-count, equal-realized-norm orthogonal removal. Fit bases on
   discovery only. Original token positions stay fixed in the receiver.
3. Retrieval write intervention and ordered source mediation: frozen Top-4 broad
   bank for Non-thinking; fit the bank post-O write's count-centroid rank-3 and
   within-count nuisance basis on discovery. Compare aligned and realized-norm
   orthogonal deletion on natural and source-restored inputs. Also test late
   answer-state removal in the same forward, preserve intermediate readouts and
   late-to-early structural null. Embedding restoration is an explicitly labeled
   input-identity upper bound; do not call it learned running-state mediation.
4. Answer source necessity in both modes: embedding plus all post-block state
   blanking, preserving sequence length and query. Thinking uses its own generated
   trace-to-answer prefix. Compare records, trace, matched ordinary token budget,
   and joint prompt/trace removal; record prefix/generation failures in denominators.
5. Thinking terminal bridge: same damaged trace baseline for terminal full item,
   marker and separator restoration, each against matched-budget ordinary states.
   Cumulative restore starts at fixed L2. Receiver answer query is not patched.
   This is controlled-prefix local bridge evidence, with generated-prefix source
   necessity treated separately.
6. Thinking terminal relay: same directed adjacent-count pairs, full/self terminal
   source at L2 crossed with natural, post-terminal-suffix L3 reset and answer-query
   L3 reset; report source-by-reset margin interaction and remaining source effect.
   Full residual patches retain content/position confounds and do not isolate
   arithmetic. Reset at L3 leaves a downstream block; no final-logit clamp.
7. Read-to-carrier feasibility: mask the fixed last retrieval query's selected
   Top-4 heads and matched complement; measure later item-end residual deformation.
   Last-layer query lesions cannot affect later teacher-forced token states.
   A two-token item also lacks distinct retrieved-marker and later within-item
   commit sites. Preserve this structural null instead of inventing a carrier.

## Statistics, verification and persistence

Implementation clarification, before inference: recompute each mode's role bank
on the new 200-prompt discovery panel and retain historical bank results as
separate assays. Thinking free-rollout head masks apply only to `<Sep>` queries
after `<Think>`, excluding the prompt's query-delimiter `<Sep>`. Three distinct
disjoint controls are feasible at K2; report the actual feasible control count at
K4. Same-count context donors are taken from the fixed discovery block with the
same index, whereas adjacent-count donors share the receiver's confirmation
block. Full-state donor sweeps include all 18 directed adjacent-count edges per
block (180 pairs); the Native-style four-edge subset is a separately reported
40-pair slice, not a new independent experiment.

Use ten balanced confirmation blocks for paired bootstrap (10,000 resamples),
also publish per-count and raw-prompt rates. Label all pointwise intervals as
unadjusted. Missing/non-count output is failure; report probability mass on count
tokens separately from the conditional expected count to avoid hiding off-support
collapse. Self patch must reproduce clean logits; zero delta and temporal-null
controls must pass. Verify actual realized-norm equality and orthogonality.
Save plan, input registry, sample/code/checkpoint hashes, per-prompt metrics,
frozen bases/centroids, actual arm coverage and compact summaries. Do not download
full attention tensors. Outputs go into a new run analysis subdirectory and are
archived locally after completion. Update the report's overbroad alignment claim,
add a per-family coverage/budget table, retain every null and all prior results.

## Intentionally unclaimed

## Whole-report budget unification (user extension, before legacy recomputation)

Additional implementation specification, before its inference: at the same
eleven behavior milestones, measure Top-2 versus three layer-matched disjoint
controls and Thinking value/residual transport on the fixed 100 inputs. Geometry
dynamics uses final-discovery-selected physical depths, checkpoint-specific
discovery-fitted PCA/NCC, all confirmation occurrences and raw-centroid rank-3
variance/effective dimension. Do not confuse refitted probes with a final frozen
probe transferred between checkpoints.

At the final checkpoint add Native-style source blanking immediately before
one retrieval per input (k=max(1,floor(N/2))): records, preceding trace items,
most recent item, all but recent, early half, each with equal-token ordinary
controls. An empty history intervention is an identity, not extra evidence.
Add adjacent-count rank-3 centroid-chord injection in both modes at all four
depths with self and three orthogonal equal-norm directions, on the same 180
directed pairs; report the Native four-edge, 40-pair subset separately. These
assays cannot remove the original model's trace-length/position confound.

The user requested that historical body experiments also use the common budget.
Freeze the supplement's `input_registry.csv` as the sole registry for the new
main report. Recompute clean four-endpoint geometry, frozen-clean NCC after
query-local Top-K, targeted/value and residual transport patching, Thinking
targeted/broad/successor factorial ablation, and progress continuation. Keep old
outputs unchanged and label them historical. Use all 200 discovery and 100
confirmation inputs for applicable families. Running-index endpoints contain
all occurrences (1100 discovery and 550 confirmation states), with identical
input/occurrence keys in both modes. States are not independent prompts.

Progress continuation uses the registry's 20 discovery and 10 confirmation
count-10 inputs, yielding 40 discovery and 60 confirmation directed pairs per
scope. Keep the previous discovery-only layer selection rule and all controls.
Transport uses one predeclared occurrence per input, k=max(1, floor(count/2));
no exclusion of low counts. Frozen role banks use the common discovery inputs.

Training dynamics evaluates all available common step-0/100-step snapshots on
the same 100 confirmation prompts for role attention. Greedy behavior is measured
at steps 0, 100, 200, 400, 800, 1500, 2500, 4000, 6000, 8000, 10000 when available.
No per-checkpoint sample reselection or outcome-dependent checkpoint choice.
All scientific snapshots retain their original checkpoint hashes. Linear and
log-step plots show the same data; step zero is omitted from logarithmic axes.

The new main report must not mix historical larger/smaller assay denominators
with this registry. Historical training loss remains the original training log
(not a held-out sample estimate). Single-token spans and the absence of a
separate post-marker commit remain explicit architecture limitations.

No retraining or seed sweep is authorized by this evaluation plan. No universal
Thinking aggregator, content-free arithmetic counter, independent synthetic
multi-token carrier, or complete/unique circuit is assumed. Qwen/Gemma-specific
OV writers and negative exploratory appendices are not required for the shared
body-mechanism comparison. Corpus/count/marker differences remain explicit.
