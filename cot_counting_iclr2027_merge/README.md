# How Can Chain-of-Thought Help LLMs Count?

This folder contains the merged arXiv-style mechanism draft.

## Main artifacts

- `main.tex`: current manuscript.
- `main.pdf`: compiled and page-by-page visually checked manuscript.
- `figures/mechanism_overview.tex`: editable TikZ source for Figure 1.
- `figures/mechanism_overview.pdf`: vector Figure 1 used by the manuscript.
- `figures/mechanism_overview.png`: raster preview.
- `merged_outline_cn.md`: Chinese synthesis of the original draft and Pro review.
- `roadmap_review_cn.md`: detailed Chinese roadmap and go/no-go criteria.
- `main_pro_source.tex`: preserved pre-merge Pro manuscript.

## Current argument

The manuscript does **not** assume that Direct uses a noisy running counter. It tests two
competing Direct hypotheses—record-wise running update and answer-time broad aggregation—against
a CoT computation decomposed into targeted source retrieval, semantic progress transition, and
terminal readout. Evidence must pass the ladder

`behavior → decodable state → dynamic update → component intervention → integrated causal chain`.

Figure 1 combines the competing computations, registered hidden-state landmarks, operational
attention roles, and the intervention-to-answer causal bridge. Dashed arrows and red TODOs are
planned tests, not reported findings.

## Reproducibility commitments in the draft

- The frozen Realistic NIAH V2 behavior corpus has 500 base stimuli; all response modes share the
  same passage bytes within a stimulus.
- Mechanism experiments use tokenizer-specific, equal-length clean/corrupt sibling families with
  explicit record delimiters and family-disjoint discovery/validation/test splits.
- Hidden states are extracted at registered token IDs, never by retokenizing raw substrings.
- No hidden states are averaged before probing, PCA, patching, or steering. Statistical summaries
  weight families equally; steering is the one exception where a cross-prompt paired mean defines
  an intervention direction.
- Qwen3-8B and Gemma4-E4B are co-primary mechanism models. Heads and layers are selected on
  discovery/validation families and frozen before the locked test.

## Compilation

From this directory:

```powershell
python C:\Users\HP\.codex\plugins\cache\openai-bundled\latex\0.2.4\scripts\compile_latex.py main.tex
```

The figure can also be rebuilt independently with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error figures\mechanism_overview.tex
```

The current manuscript is a self-contained two-column arXiv draft; it can later be transferred to
the official ICLR 2027 style without changing the experimental protocol.
