# arXiv draft status and submission blockers

The manuscript currently uses a generic one-column preprint style. All known metadata and reproducibility gaps are explicit in `TODO.md`; source-level gaps are additionally marked with red `\todo{...}` placeholders.

## What is already strong

- One coherent mechanistic thesis connects synthetic and realistic evidence.
- The requested two-route diagram is Figure 1 and the paper's organizing device.
- The controlled model has complementary necessity, local sufficiency, and executability tests.
- The query-position intervention gives a useful, paired interface ablation.
- The realistic study cleanly separates retrieval from aggregation and shows enumeration rescue.

## Highest-priority experiments before conference submission

1. **Multi-seed controlled replication.** Run at least five seeds. Report role/subspace overlap rather than exact head IDs, and replicate top-2 trace-readout synergy plus L2/L4 steering onset.
2. **Disentangle trace length from RoPE.** Keep `<Ans>` and relevant relative positions fixed while varying the number of valid trace pairs with padding/masks.
3. **Directly test the v16.3 query bottleneck.** Probe target-wise counts at query tokens, patch query states, and identify data-to-query heads.
4. **Natural AR interventions.** Repeat key patches on naturally wrong rollouts rather than only teacher-forced or constructed corruptions.
5. **Realistic cross-model causal replication.** Extend ablation/patching/steering beyond Qwen3-8B and beyond 4K context.

## Writing decisions still needed from the authors

- Confirm the corresponding author, public contact email, and whether the controlled and realistic repositories may be released publicly.
- Revisit the title only if later multi-seed results change the strength of the route-rewiring claim.
- Decide whether v16.3 is a main contribution or an ablation; the current draft treats it as a main mechanistic stress test.
- Replace any internal run names with anonymous artifact identifiers before submission.

## Claims that should not be strengthened yet

- Do not claim a universal CoT mechanism.
- Do not claim the final trace readout computes abstract list length independently of position.
- Do not call a high linear-probe score a causal counter.
- Do not interpret six deterministic random-head paths as statistical confidence intervals.
- Do not claim the +1 pp thinking answer gain in v16.3 is stable across training runs.
