from __future__ import annotations

import pandas as pd

from src.data.dataset_builder import TrainingFrame
from src.model.evaluation import build_walk_forward_splits, compute_classification_metrics, purged_chronological_split


def _build_training_frame(length: int = 20) -> TrainingFrame:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=length, freq="5min"),
            "ret_1": [index / 1000 for index in range(length)],
            "rv_3": [index / 2000 for index in range(length)],
            "target": [index % 2 for index in range(length)],
        }
    )
    return TrainingFrame(frame=frame, feature_columns=["ret_1", "rv_3"])


def test_purged_chronological_split_leaves_gap_between_train_and_valid() -> None:
    training = _build_training_frame(length=20)

    X_train, X_valid, y_train, y_valid, split = purged_chronological_split(
        training,
        validation_fraction=0.25,
        purge_rows=2,
    )

    assert len(X_train) == len(y_train) == 13
    assert len(X_valid) == len(y_valid) == 5
    assert split.train_end == 13
    assert split.valid_start == 15


def test_build_walk_forward_splits_respects_purge_gap() -> None:
    training = _build_training_frame(length=18)

    splits = build_walk_forward_splits(
        training,
        min_train_size=8,
        validation_size=3,
        step_size=3,
        purge_rows=1,
    )

    assert [(split.train_end, split.valid_start, split.valid_end) for split in splits] == [
        (8, 9, 12),
        (11, 12, 15),
        (14, 15, 18),
    ]


def test_compute_classification_metrics_returns_core_probabilistic_scores() -> None:
    y_true = pd.Series([0, 1, 0, 1, 1], dtype="int64")
    probabilities = pd.Series([0.1, 0.8, 0.4, 0.7, 0.6], dtype="float64")

    metrics = compute_classification_metrics(y_true, probabilities)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["brier_score"] < 0.2
    assert metrics["log_loss"] < 0.6
    assert metrics["positive_rate"] > 0.0
    assert metrics["sample_count"] == 5.0
    assert metrics["roc_auc"] == 1.0
