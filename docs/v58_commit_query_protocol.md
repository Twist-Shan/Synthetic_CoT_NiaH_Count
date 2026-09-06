# V58 commit-state to next-query protocol (2026-09-05)

Question: does transplanting the completed-item state redirect the next
retrieval query to the donor's successor occurrence? Final answer changes are
secondary, and are not required for this local edge.

This protocol is fixed before inspecting new intervention outcomes. It follows
Native-thinking report section 5.3's full commit vs self / count-subspace /
orthogonal comparisons. It does not assume that synthetic effects will match
the large-model results.

- Existing v58 Thinking final checkpoint, fixed no-index separator grammar.
- Historical progress layer L1 is primary, taken from the existing sufficiency
  discovery selection. L2 is a prespecified sensitivity analysis. L3 (the
  representation-selected layer) and L4 are structural controls: an L3
  post-block commit cannot alter the immediately following query's L4 Q vector,
  although changing the commit K can rescale source probabilities through the
  softmax denominator; an L4 commit has no downstream influence on that query.
- Targeted bank: historical discovery Top-4 L4H5/H0/H1/H4. Top-2 also reported.
- Count 10, completed-item k = 4, 6, 8; donor k-1 and k+1 on the same trajectory.
  Keep all six pairs per prompt, including repeated-marker pairs. No selection
  on attention, logits, generation correctness, or intervention outcome.
- From the existing canonical-disjoint external confirmation examples, sort
  count-10 prompts by SHA256; use first 20 for new discovery basis fitting,
  next 60 for new confirmation. Exclude historical head-selection/reporting
  keys and all historical progress prompts. These are new mechanistic splits
  within a previously behavior-evaluated suite, not unseen behavior data.
- Fit an orthonormal rank-3 basis from centered progress-class centroids on
  discovery states k=1..9 separately for each layer; freeze before confirmation.
  Subspace patch: h_r + P(h_d-h_r). Orthogonal control is norm matched to this
  projected delta. Additional three random directions orthogonal to the count
  basis are norm matched to the *full* delta, strengthening specificity.
- Conditions: clean, self, full donor, count-subspace transplant, projected-
  delta-norm orthogonal, three full-delta-norm orthogonals. Patch only the
  completed marker's post-block residual. Never patch the next query itself.
- Teacher-forced local endpoint: same gold receiver prefix plus the unchanged
  next <Sep>; Y = sum over bank heads of A(donor-successor prompt position)
  minus A(receiver-successor prompt position). Also report per-head QK margin
  log[A(d)/A(r)] and mean within-pair donor share to distinguish redirection
  from softmax denominator changes. Raw bank-summed mass is not comparable
  across differently sized large-model banks.
- Greedy continuation starts after the gold completed-item boundary and has
  no forced tokens thereafter. First marker identity is only informative when
  donor and receiver successor characters differ; direct attention identifies
  occurrences even when characters repeat. Capture routing at every generated
  <Sep> to measure the first three retrievals and whether a shift persists.
  Run rollout at primary L1; evaluate all four layers for the local endpoint.
- Each prompt is the sampling unit. Average the six pairs per prompt before
  10,000 paired bootstrap resamples; report +/- offsets separately as well.
  Main direct contrast: full minus self. Specificity: full minus projected-norm
  orthogonal and full minus mean full-norm orthogonals. No claim of isolated
  arithmetic +1 update, position independence, or serial mediation follows
  solely from a positive full-state effect.
- Freeze sample plan and settings on disk before any outcome evaluation. Save
  per-pair CSVs, compact rollout tokens, per-step routing, aggregate CIs,
  checkpoint/code/input hashes, and validation checks. No retraining.

## Post-primary scope/position sensitivity (specified before its own outcomes)

The current Native-thinking report section 5.3 actually uses a position-aligned
item span, not just a single endpoint. After observing the first single-marker
assay, run one explicitly secondary sensitivity on the same 20/60 prompt split:
retain L1 primary and L2/L3/L4 local depth controls, but patch both `<Sep>` and
marker post-block vectors. Do not relabel these reused prompts as a second
independent confirmation. For donor k+1 delete the last two non-needle filler
tokens from its haystack (254 tokens); for donor k-1 append two copies of its
last non-needle filler token (258 tokens). The ten needle identities/order and
all trace tokens are unchanged; donor item endpoint now exactly equals receiver
item endpoint in absolute position. Receiver input is unchanged. Fit a rank-3
basis on concatenated two-token discovery item states and match orthogonal
controls in that same concatenated space. The intervention holds a two-token
item, not the next query. All endpoints/bootstrap and rollout rules are unchanged.
This removes donor endpoint-position mismatch but introduces a small donor
prompt-length shift and patches a lexical marker; it still does not isolate
a content-free arithmetic counter. Report both assays, irrespective of outcome.
