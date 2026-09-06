#!/usr/bin/env bash
set -euo pipefail
cd /lambda/nfs/NiaH-Synthetic/repo-v58
RUN=/lambda/nfs/NiaH-Synthetic/runs/v58_count1to10_permuted_grammarw16_width512_heads8_steps10000_fullstarts_independent_L256_pool100_seed1234
PY=/lambda/nfs/NiaH-Synthetic/venv/bin/python
ALIGN="$RUN/analysis/v58_alignment_supplement_20260905"
OUT="$RUN/analysis/v58_unified_legacy_20260905"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PY" scripts/run_v58_unified_legacy.py --run-dir "$RUN" --alignment "$ALIGN" --output "$OUT"
"$PY" scripts/run_v58_native_continuation.py --run-dir "$RUN" --output "$OUT/continuation" --panel-registry "$ALIGN/input_registry.csv" --frozen-sites "$ALIGN/thinking/frozen_sites.json"
echo 'ALL UNIFIED EXPERIMENTS COMPLETE'
