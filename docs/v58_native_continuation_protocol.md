# V58 Native-aligned continuation confirmation (2026-09-05)

Question: can a contextual item-state transplant control donor-directed free
continuation under the same discovery-only layer selection used in the current
Native-thinking report? No retraining, trace edits, requirement of a changed
final count, or requirement of attention-bank argmax reversal.

## Frozen design

- Fixed v58 Thinking step-10000 dense snapshot. Both scopes are reported:
  item-end marker (one token), and complete `<Sep> marker` item (two tokens).
- Use the existing external count-10 behavior suite. Exclude every prompt and
  canonical (set_id, corpus_start) key used in the previous 80-prompt assay,
  historical head-selection/reporting keys, and historical progress prompts.
  Sort remaining hashes, take first 20 for discovery and next 60 for confirmation.
  These are fresh mechanistic cohorts, drawn from a previously behavior-evaluated
  suite; they are not a new unseen behavior test and not new training seeds.
- Discovery: donor k=6, receiver r=5 and 7. Confirmation: donor k=4,6,8;
  receiver r=k-1 (forward skip) and k+1 (backward rewind), six pairs per prompt.
  This follows the Native donor-index convention; the earlier synthetic assay
  fixed receiver indices instead. Never select on clean accuracy or intervention
  outcome. Keep all pairs, including ambiguous repeated-marker cases.
- Align donor item endpoint to the unchanged receiver endpoint by removing two
  non-needle filler tokens (forward) or appending two non-needle filler tokens
  (backward). Receiver prompt remains 256 characters; donors have 254/258.
  Preserve all ten needle identities/order and original trace text. The prompt
  length perturbation remains a limitation. Both item tokens are position-aligned.

## Layer selection, before any confirmation inference

- Scan synthetic post-block L1-L4 (Native zero-based L0-L3). Exclude L4 from
  selection because it has no downstream decoder block.
- For each scope, use only discovery paired change in donor-successor versus
  receiver-successor marker log-odds after the unchanged next `<Sep>`.
- Unlike Native city candidates, synthetic successor markers can be identical.
  Such pairs have undefined identity-discriminating log-odds and are excluded
  from this selection statistic, with their coverage reported. They are not
  removed from the frozen cohort or the rollout export. This is a necessary
  adaptation, not a literal token-level replica of the Native benchmark.
- Require forward and backward medians to be positive; average paired effects
  within each prompt and take their across-prompt median. Among eligible layers,
  choose the earliest reaching 95% of the maximum median. Attention, generation,
  NCC and confirmation results do not enter this choice. If no layer qualifies,
  report the scope as not selected; do not use a fallback chosen on confirmation.
- Freeze `selected_layers.json` before capturing confirmation states or running
  any confirmation inference. Fit rank-3 progress centroid bases only on these
  20 discovery prompts, for norm-matched control comparisons.

## Confirmation outcomes and controls

- Conditions: clean, self, full donor, rank-3 projected donor, projected-delta
  norm orthogonal, and three full-delta norm orthogonals. Hold the replacement
  at the original item positions while continuing autoregressively; never patch
  the next query or any later generated item. Greedy rollout, maximum 28 tokens.
- Primary behavior: first generated marker equals donor successor, on pairs
  with distinct donor/receiver successor marker identities. A missing marker
  is failure, not an excluded observation. Report raw numerator/denominator and
  prompt-equal rates; use full-self and full-mean(full-norm orthogonals) contrasts.
- Additional donor-specific continuation: find the shortest horizon q<=4 at
  which the gold donor and receiver future marker prefixes differ, requiring
  both prefixes to exist. Match the *entire* generated prefix through q. If no
  such q exists, mark the pair unidentifiable for this metric, not a success.
- Also report exact donor prefixes at h=2,3,4, whenever both gold prefixes exist
  and differ. These are unconditional success rates on input-defined subsets.
  Report continued matches conditional on first-marker transfer separately as
  descriptive statistics, with explicit denominators, matching Native's audit.
- Supplementary: first-query attention/QK shifts, actual needle-position bank
  argmax, final-count accuracy, generated item count, clean correctness, forward
  vs backward and same-commit-marker strata. No additional pass gates imposed
  on the primary continuation claim. These cannot identify an abstract arithmetic
  operator or serial mediation by themselves.
- Average pairs within prompt, then 10,000 prompt-cluster bootstrap resamples.
  Confidence intervals are descriptive and unadjusted for multiple comparisons.
  Preserve fresh confirmation outcomes regardless of their direction or size.
- Save sample/selection/code/checkpoint hashes, per-pair and continuation CSVs,
  control validations and compact plots. No new tuning after confirmation.
