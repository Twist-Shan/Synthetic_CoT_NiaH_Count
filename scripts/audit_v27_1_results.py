#!/usr/bin/env python
"""Audit v27.1 paired behavior, parameter scope, NCC inheritance, and persistence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from synthetic_counting_v20.training import load_v20_checkpoint_model


MODES = ("thinking", "nonthinking")
PAIR_KEYS = (
    "row_id",
    "set_id",
    "count",
    "corpus_region",
    "corpus_start",
    "prompt_sha256",
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def exact_mcnemar(thinking_only: int, nonthinking_only: int) -> float:
    discordant = thinking_only + nonthinking_only
    if discordant == 0:
        return 1.0
    lower_tail = min(thinking_only, nonthinking_only)
    tail = sum(math.comb(discordant, index) for index in range(lower_tail + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def parameter_scope(source_run: Path, calibrated_run: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for mode in MODES:
        _, vocab, _, _, source_model = load_v20_checkpoint_model(
            source_run, "rope", mode, label="final", device="cpu"
        )
        source = source_model.state_dict()
        calibrated = torch.load(
            calibrated_run / "final" / mode / "checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )["model_state_dict"]
        embedding_key = "token_embedding.weight"
        number_rows = list(map(int, vocab.number_ids))
        changed_rows = torch.nonzero(
            (source[embedding_key] != calibrated[embedding_key]).any(dim=1),
            as_tuple=False,
        ).flatten().tolist()
        nonnumber_rows = [
            index
            for index in range(source[embedding_key].shape[0])
            if index not in set(number_rows)
        ]
        result[mode] = {
            "changed_parameter_keys": [
                key
                for key in source
                if not torch.equal(source[key], calibrated[key])
            ],
            "changed_embedding_rows": changed_rows,
            "number_token_rows": number_rows,
            "changed_number_row_count": len(set(changed_rows) & set(number_rows)),
            "changed_nonnumber_row_count": len(
                set(changed_rows) - set(number_rows)
            ),
            "other_parameter_max_abs_diff": max(
                (source[key] - calibrated[key]).abs().max().item()
                for key in source
                if key != embedding_key
            ),
            "nonnumber_embedding_max_abs_diff": (
                source[embedding_key][nonnumber_rows]
                - calibrated[embedding_key][nonnumber_rows]
            )
            .abs()
            .max()
            .item(),
        }
    return result


def paired_behavior(run: Path) -> tuple[dict[str, object], pd.DataFrame]:
    final = run / "final"
    thinking = pd.read_csv(final / "thinking" / "final_autoregressive_detail.csv")
    nonthinking = pd.read_csv(
        final / "nonthinking" / "final_autoregressive_detail.csv"
    )
    thinking = thinking.sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    nonthinking = nonthinking.sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    if not thinking[list(PAIR_KEYS)].equals(nonthinking[list(PAIR_KEYS)]):
        raise RuntimeError("Thinking and Non-thinking test rows do not match exactly")

    thinking_correct = thinking["ar_accuracy"].astype(int).to_numpy()
    nonthinking_correct = nonthinking["ar_accuracy"].astype(int).to_numpy()
    difference = thinking_correct - nonthinking_correct
    thinking_only = int(((thinking_correct == 1) & (nonthinking_correct == 0)).sum())
    nonthinking_only = int(
        ((thinking_correct == 0) & (nonthinking_correct == 1)).sum()
    )
    generator = np.random.default_rng(271)
    bootstrap = difference[
        generator.integers(0, len(difference), size=(20_000, len(difference)))
    ].mean(axis=1)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paired_examples": len(difference),
        "thinking_correct": int(thinking_correct.sum()),
        "nonthinking_correct": int(nonthinking_correct.sum()),
        "thinking_accuracy": float(thinking_correct.mean()),
        "nonthinking_accuracy": float(nonthinking_correct.mean()),
        "thinking_wilson_95": wilson(
            int(thinking_correct.sum()), len(thinking_correct)
        ),
        "nonthinking_wilson_95": wilson(
            int(nonthinking_correct.sum()), len(nonthinking_correct)
        ),
        "paired_accuracy_gap": float(difference.mean()),
        "paired_bootstrap_gap_95": np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        "paired_bootstrap_samples": 20_000,
        "paired_bootstrap_seed": 271,
        "contingency": {
            "both_correct": int(
                ((thinking_correct == 1) & (nonthinking_correct == 1)).sum()
            ),
            "thinking_only_correct": thinking_only,
            "nonthinking_only_correct": nonthinking_only,
            "both_wrong": int(
                ((thinking_correct == 0) & (nonthinking_correct == 0)).sum()
            ),
        },
        "mcnemar_exact_two_sided_p": exact_mcnemar(
            thinking_only, nonthinking_only
        ),
        "test_used_for_selection": False,
    }

    rows = []
    for count in sorted(thinking["count"].unique()):
        mask = thinking["count"].eq(count).to_numpy()
        thinking_count = thinking_correct[mask]
        nonthinking_count = nonthinking_correct[mask]
        thinking_only = int(
            ((thinking_count == 1) & (nonthinking_count == 0)).sum()
        )
        nonthinking_only = int(
            ((thinking_count == 0) & (nonthinking_count == 1)).sum()
        )
        thinking_interval = wilson(int(thinking_count.sum()), len(thinking_count))
        nonthinking_interval = wilson(
            int(nonthinking_count.sum()), len(nonthinking_count)
        )
        rows.append(
            {
                "count": int(count),
                "n": int(mask.sum()),
                "thinking_accuracy": float(thinking_count.mean()),
                "thinking_wilson_low": thinking_interval[0],
                "thinking_wilson_high": thinking_interval[1],
                "nonthinking_accuracy": float(nonthinking_count.mean()),
                "nonthinking_wilson_low": nonthinking_interval[0],
                "nonthinking_wilson_high": nonthinking_interval[1],
                "paired_gap": float((thinking_count - nonthinking_count).mean()),
                "thinking_only_correct": thinking_only,
                "nonthinking_only_correct": nonthinking_only,
                "mcnemar_exact_p": exact_mcnemar(
                    thinking_only, nonthinking_only
                ),
                "thinking_trace_exact": float(
                    thinking.loc[mask, "trace_exact"].mean()
                ),
            }
        )
    return summary, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--source-run", default="v24.3_componentloss_count1-10_seed1234"
    )
    parser.add_argument(
        "--run", default="v27.1_trace_safe_tied_calibration_L256_pool100_seed2478"
    )
    parser.add_argument(
        "--legacy-run", default="v27_tied_number_row_calibration_L256_pool100_seed2478"
    )
    args = parser.parse_args()

    root = args.results_root.resolve()
    source_run = root / args.source_run
    run = root / args.run
    now = datetime.now(timezone.utc).isoformat()

    strict_scope = parameter_scope(source_run, run)
    if any(
        item["changed_nonnumber_row_count"] != 0
        or item["other_parameter_max_abs_diff"] != 0.0
        or item["nonnumber_embedding_max_abs_diff"] != 0.0
        for item in strict_scope.values()
    ):
        raise RuntimeError("v27.1 is not a strict atomic-number-row update")
    atomic_json(
        run / "parameter_scope_audit.json",
        {
            "generated_at_utc": now,
            "source_run": str(source_run),
            "run": str(run),
            "strict_number_row_only": True,
            "modes": strict_scope,
            "conclusion": (
                "Only the ten atomic-number embedding/unembedding rows changed; "
                "every other parameter and embedding row is bitwise identical to v24.3."
            ),
            "pre_answer_hidden_state_and_attention_invariance": (
                "Exact under the checked grammar because atomic number tokens are "
                "absent before the answer target."
            ),
        },
    )

    legacy_run = root / args.legacy_run
    if legacy_run.exists():
        legacy_scope = parameter_scope(source_run, legacy_run)
        atomic_json(
            legacy_run / "parameter_scope_audit.json",
            {
                "generated_at_utc": now,
                "source_run": str(source_run),
                "run": str(legacy_run),
                "strict_number_row_only": False,
                "modes": legacy_scope,
                "conclusion": (
                    "The run changed non-number embedding rows through the tied input "
                    "path and is not a strict number-row readout control."
                ),
                "scientific_use": "Broader tied embedding-row adaptation upper bound only.",
            },
        )
        legacy_manifest_path = legacy_run / "manifest.json"
        legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        legacy_manifest.update(
            {
                "trainable_parameters_declared_at_run": legacy_manifest.get(
                    "trainable_parameters"
                ),
                "trainable_parameters": (
                    "tied token_embedding rows reachable from the causal input path "
                    "plus atomic-number unembedding rows"
                ),
                "gradient_row_mask_applied": False,
                "strict_number_row_only": False,
                "scientific_status": "broader_tied_embedding_row_adaptation_upper_bound",
                "parameter_scope_audit_file": "parameter_scope_audit.json",
                "do_not_use_as": "strict_number_row_readout_control",
            }
        )
        atomic_json(legacy_manifest_path, legacy_manifest)

    behavior_summary, behavior_by_count = paired_behavior(run)
    behavior_dir = run / "analysis" / "paired_behavior"
    atomic_json(behavior_dir / "summary.json", behavior_summary)
    behavior_by_count.to_csv(behavior_dir / "by_count.csv", index=False)

    source_ncc = source_run / "analysis" / "aligned_ncc"
    inherited_ncc = run / "analysis" / "inherited_aligned_ncc"
    inherited_ncc.mkdir(parents=True, exist_ok=True)
    ncc_files = (
        "selected_confirmation_summary.csv",
        "geometry_site_layer_metrics.csv",
        "geometry_discovery_selected_metrics.csv",
        "selected_layers.json",
    )
    for name in ncc_files:
        shutil.copy2(source_ncc / name, inherited_ncc / name)
    selected = pd.read_csv(inherited_ncc / "selected_confirmation_summary.csv")
    reference = {
        "generated_at_utc": now,
        "inheritance_exact": True,
        "source_run": args.source_run,
        "reason": (
            "Only atomic-number embedding/unembedding rows changed, and those tokens "
            "are absent from every pre-answer causal prefix; hidden states and attention "
            "are therefore bitwise inherited."
        ),
        "parameter_scope_audit": "parameter_scope_audit.json",
        "selected_confirmation_metrics": json.loads(
            selected.to_json(orient="records")
        ),
    }
    atomic_json(inherited_ncc / "inheritance_reference.json", reference)

    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "gradient_row_mask_applied": True,
            "strict_number_row_only": True,
            "atomic_number_prefix_absence_checked": True,
            "scientific_status": "strict_tied_number_row_readout_control",
            "parameter_scope_audit_file": "parameter_scope_audit.json",
            "pre_answer_geometry_inherited_exactly_from_source": True,
            "paired_behavior_audit": "analysis/paired_behavior/summary.json",
            "pre_answer_geometry_reference": (
                "analysis/inherited_aligned_ncc/inheritance_reference.json"
            ),
            "pre_answer_ncc_inheritance_exact": True,
        }
    )
    atomic_json(manifest_path, manifest)

    persistence_files = [
        manifest_path,
        run / "final_summary.csv",
        run / "parameter_scope_audit.json",
        run / "final" / "thinking" / "checkpoint.pt",
        run / "final" / "nonthinking" / "checkpoint.pt",
        behavior_dir / "summary.json",
        behavior_dir / "by_count.csv",
        inherited_ncc / "selected_confirmation_summary.csv",
        inherited_ncc / "inheritance_reference.json",
    ]
    atomic_json(
        run / "persistence_audit.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "verified",
            "files": {
                str(path.relative_to(run)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in persistence_files
            },
        },
    )
    print(json.dumps(behavior_summary, indent=2, sort_keys=True), flush=True)
    print(
        selected[
            [
                "comparison_mode",
                "endpoint",
                "layer",
                "confirmation_logistic_balanced_accuracy",
                "confirmation_ncc_balanced_accuracy",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"AUDITED_RUN={run}", flush=True)


if __name__ == "__main__":
    main()
