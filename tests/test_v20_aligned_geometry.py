from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_counting_v20.aligned_geometry import evaluate_geometry_layer


def _toy_metadata(labels: list[int], split: str, start: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": [split] * len(labels),
            "occurrence": labels,
            "group_id": [f"{split}-{start + index}" for index in range(len(labels))],
        }
    )


def test_aligned_geometry_separates_discovery_and_confirmation() -> None:
    rng = np.random.default_rng(7)
    classes = (1, 2, 3)
    discovery_y = [label for label in classes for _ in range(9)]
    confirmation_y = [label for label in classes for _ in range(4)]
    metadata = pd.concat(
        [
            _toy_metadata(discovery_y, "discovery"),
            _toy_metadata(confirmation_y, "confirmation", len(discovery_y)),
        ],
        ignore_index=True,
    )
    labels = np.asarray(discovery_y + confirmation_y)
    states = np.column_stack(
        [
            labels.astype(float) * 5 + rng.normal(0, 0.04, len(labels)),
            labels.astype(float) ** 2 + rng.normal(0, 0.04, len(labels)),
            rng.normal(0, 0.2, len(labels)),
            rng.normal(0, 0.2, len(labels)),
        ]
    ).astype(np.float32)
    result = evaluate_geometry_layer(
        states,
        metadata,
        classes,
        pca_dim=4,
        random_state=11,
        folds=3,
    )
    assert result["confirmation_logistic_balanced_accuracy"] > 0.95
    assert result["confirmation_ncc_balanced_accuracy"] > 0.95
    assert result["discovery_rows"] == 27
    assert result["confirmation_rows"] == 12


def test_aligned_geometry_constant_answer_embedding_is_chance_only() -> None:
    classes = (1, 2, 3)
    discovery_y = [label for label in classes for _ in range(6)]
    confirmation_y = [label for label in classes for _ in range(3)]
    metadata = pd.concat(
        [
            _toy_metadata(discovery_y, "discovery"),
            _toy_metadata(confirmation_y, "confirmation", len(discovery_y)),
        ],
        ignore_index=True,
    )
    states = np.ones((len(metadata), 8), dtype=np.float32)
    result = evaluate_geometry_layer(states, metadata, classes, folds=3)
    assert result["confirmation_logistic_balanced_accuracy"] == 1 / 3
    assert result["confirmation_ncc_balanced_accuracy"] == 1 / 3
    assert np.isnan(result["confirmation_isotropic_snr_db"])
    assert np.isnan(result["confirmation_fisher_trace_frozen"])
