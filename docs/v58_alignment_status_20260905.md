# V58 uniform-panel alignment: execution state

User requested both missing body assays and historical body assays be evaluated
at the same scale. Original model checkpoints and trace remain unchanged.

## Frozen registry

`work/v58_final/analysis/v58_alignment_supplement_20260905/input_registry.csv`

SHA256: `64a276f75b5997ddb7f616a0740a0b5a021e7ff9fabd1ef871f5329d39f9dc15`

200 discovery inputs / 100 confirmation inputs; counts 1–10 balanced, paired
between modes. Progress uses the same registry's count-10 subset (20 / 10).

## Completed and copied locally

`v58_alignment_supplement_20260905`: source restoration, answer-state layer
sweep, NT retrieval-write/late-state factorial, own-generated-prefix answer
source necessity, reranked free Top-K, Thinking bridge/relay and structural
read-to-item null. Both modes' common realized trial multisets verified.

Initial descriptive results (confirmation, 100 inputs):

- Free answer accuracy: Thinking 95%, Non-thinking 21%.
- Thinking Top-4 selected: answer 86%, trace exact 51%; clean trace exact 87%.
- Thinking Top-4 matched complement: answer 91%, trace exact 74%.
- Thinking own-generated-prefix source blanking: clean answer 95%, records 95%,
  trace 59%; equal-token ordinary trace-budget control 95%.
- NT broad Top-4 selected answer 7%, matched complement 0%, clean 21%; this
  does not establish selected-bank-specific necessity.
- Thinking gold-prefix terminal bridge: damaged answer 68%; averaged semantic
  restore 69%, ordinary restore 68%. Full scope results retained separately.

These are descriptive values, not a claim that every mechanism replicated.

## Historical-family recomputation

**Completed.** Both remote unified runners finished; both result archives were
downloaded and their SHA256 matched the server values. The realized input,
occurrence-state and directed-pair audit passed, including exact agreement
between step-10000 dynamics NCC and the final-checkpoint frozen clean probe.

Archive hashes:

- legacy: `8042e9e41c9a1076d39fcacd8e4360032a041faa6a142e292542f759755bacbe`
- additional: `f15e884fda88ef42903eb36ca5a09de36592ec9513a0cf34a0cd9089ebc6b5c5`

NCC at discovery NCC-selected layers: NT running 17.6012% (L2), T running
31.3742% (L4), NT final 24% (L4), T final 99% (L2). A separate common decoder
selection chooses NT running L4, NT final L2, T running/final L2. Post-Top-K and
geometry dynamics use those common physical depths; do not conflate selectors.

Progress now selects L1 on the shared 20-prompt discovery subset. Confirmation
has 60 pairs from 10 inputs; first-marker identification has 26 pairs/8 inputs.
Full two-token item donor-minus-full-norm-control effect is 16.04 pp
[4.17, 28.96] for the next marker and 26.04 pp [4.17, 52.08] for the two-marker
prefix. Three-marker CI crosses zero; four-marker CI touches zero. No token caps.

Source-next accuracy: clean 96%, records blank 21% (ordinary control 95%), recent
item/history blank 79% (controls 96%). NT Top-4 complement ablation outputs no
numeric answer on all 100 inputs; selected Top-4 still answers all 100. Therefore
its low control accuracy includes generic output failure and is not a clean
count-specific necessity comparison.

Important dynamic limitation: at fixed L2, T final NCC is already 100% at step
400 while free answer accuracy is 12%; T running NCC peaks at 72.6% at step200
and ends at 34.2%. All geometry uses gold trace; no monotonic compression claim.

Remote script `scripts/run_v58_unified_legacy_remote.sh`, log
`/lambda/nfs/NiaH-Synthetic/v58_unified_legacy_20260905.log`.
Output under the v58 run's `analysis/v58_unified_legacy_20260905`.

Includes four-endpoint geometry, all-layer frozen-clean post-Top-K NCC,
targeted/broad/successor factorial, target V/residual patching, fixed-panel
100-step attention dynamics and eleven behavior/intervention milestones,
followed by Native-style continuation on the shared count-10 subset.

Additional runner `scripts/run_v58_unified_additional.py` covers source-next
necessity, rank-3 adjacent-count chord versus norm controls, and geometry
dynamics at the same eleven milestones. Check completion manifests before
claiming these finished or building the final unified report.

## Report and tests

New builder: `scripts/build_v58_unified_report.py`; staging output defaults to
`reports/NiaH_Synthetic_report_unified.html`. Do not overwrite the main report
until all manifests exist and the staging output has been checked. These checks
have passed; the main report has now been rebuilt. The original main HTML and
manifest are preserved as `NiaH_Synthetic_report_pre_alignment_20260905.html`
and the corresponding `.manifest.json`.

19 focused tests passed (17 intervention/continuation and two new all-state
alignment tests). Original old outputs remain unchanged. The historical
factorial evaluator has been corrected to restrict Sep queries to generated
trace positions; it previously also included the prompt query delimiter.

Static JS syntax, 5200 finite projection points and 16 endpoint×layer panels
passed `scripts/check_v58_report_static.cjs`; scientific PNGs were viewed.
The Browser skill's security policy blocked file-URL preview; no workaround was
attempted, and interactive drag/layer-switch testing is explicitly not claimed.
The updated report retains the established Plotly widget with separate 2D/3D
layer controls and orbital rotation configuration.

Rebuild commands (from repository root):

```
.venv/Scripts/python.exe scripts/audit_v58_unified_outputs.py --analysis work/v58_final/analysis
.venv/Scripts/python.exe scripts/build_v58_unified_report.py --output reports/NiaH_Synthetic_report.html
node scripts/check_v58_report_static.cjs reports/NiaH_Synthetic_report.html
```

## Interpretation constraints

NCC decoding is not causal use. One-token source spans cannot distinguish span
from endpoint. A two-token item has no independent retrieved-marker-to-later
commit site. Last-layer query-only interventions cannot deform later fixed-token
states. Fixed trace length/answer position remains a count confound. Ten balanced
blocks are statistical groups, not ten model-training seeds. Keep pointwise
intervals and input/pair/state denominators explicit.
