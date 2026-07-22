# arXiv draft: How Chain-of-Thought Rewires Counting

This folder contains a generic, one-column arXiv-style English paper draft built around the two mechanism routes in the synthetic reports and the realistic 4K NIAH evidence. It does not depend on an ICLR or other conference template.

## Main files

- `main.tex`: manuscript and appendices.
- `main.pdf`: latest compiled draft.
- `references.bib`: literature checked against primary paper pages/arXiv.
- `figures/`: selected report figures, renamed by role.
- `source_map.md`: local evidence provenance and claim-to-artifact mapping.
- `TODO.md`: explicit metadata, reproducibility, experiment, and editorial gaps.
- `DRAFT_STATUS.md`: scientific claim-strength notes and submission blockers.

Unknown fields are marked in red with `\todo{...}` in `main.tex`. Do not silently replace these with guesses.

## Compile

From this directory:

```powershell
latexmk -pdf main.tex
```

or:

```powershell
python C:\Users\HP\.codex\plugins\cache\openai-bundled\latex\0.2.4\scripts\compile_latex.py <absolute-path-to-main.tex>
```

## Claim discipline

The manuscript distinguishes supported causal links in the single-seed controlled transformer, convergent but incomplete query-bottleneck evidence in v16.3, large-model behavioral external validity, and suggestive versus completed mechanistic evidence in Qwen3-8B. Keep the limitations about training-seed variance, trace-span/RoPE confounding, and the missing v16.3 causal battery until those experiments are completed.
