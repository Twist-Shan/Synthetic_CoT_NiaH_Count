# arXiv draft TODOs

All unknown or intentionally incomplete fields are marked in red with `\todo{...}` in `main.tex`.

## Required before the first arXiv upload

- [x] Add author names, affiliations, and the July 21, 2026 version date.
- [ ] Confirm the corresponding author and public contact email.
- [ ] Add acknowledgments, funding, compute credits, and disclosure statements.
- [ ] Add the public code, data, and report archive URLs.
- [ ] Record exact model and tokenizer revision hashes, inference-library versions, and licenses for Qwen3-8B and Olmo-Hybrid-7B.
- [ ] Add hardware, wall-clock time, and total compute for synthetic training and realistic inference.
- [ ] Audit every reported number against the frozen result tables and decide which confidence intervals belong in the main text.
- [ ] Export figures as vector PDF/SVG where source plots are available; the current draft uses report PNGs.
- [ ] Run a final citation audit and add any related work discovered during author review.

## Highest-value scientific work before conference submission

- [ ] Repeat role/subspace results across at least five synthetic seeds.
- [ ] Separate query causal visibility, query-to-output RoPE distance, and the five-token data-position shift.
- [ ] Patch and probe query-token states directly in the query-last model.
- [ ] Vary valid trace-pair count while holding answer position and relevant RoPE distances fixed.
- [ ] Repeat interventions on naturally occurring autoregressive errors.
- [ ] Extend the full realistic-model causal battery beyond Qwen3-8B and to 16K/32K contexts.

## Editorial pass

- [ ] Decide whether to use `Direct`/`Trace` or `nonthinking`/`thinking` consistently in every figure.
- [ ] Standardize `CoT`, `trace`, and `enumeration` terminology.
- [ ] Check that the abstract and contribution bullets match the final causal claim strength.
- [ ] Add a reproducibility checklist and a compact table of all evaluation sample sizes.
- [ ] Replace this generic arXiv style only when a target-venue template is selected.
