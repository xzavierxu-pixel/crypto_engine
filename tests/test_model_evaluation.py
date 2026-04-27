from __future__ import annotations

import pandas as pd

from src.data.dataset_builder import TrainingFrame
from src.model.evaluation import (
    build_walk_forward_splits,
    compute_return_direction_metrics,
    compute_stage1_coverage,
    compute_two_stage_end_to_end_metrics,
    purged_chronological_split,
    purged_chronological_time_window_split,
    summarize_walk_forward,
)


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
    splits = build_walk_forward_splits(training, min_train_size=8, validation_size=3, step_size=3, purge_rows=1)
    assert [(split.train_end, split.valid_start, split.valid_end) for split in splits] == [
        (8, 9, 12),
        (11, 12, 15),
        (14, 15, 18),
    ]


def test_stage1_coverage_is_predicted_active_ratio_not_probability_mean() -> None:
    probabilities = pd.Series([0.9, 0.8, 0.2, 0.1], dtype="float64")
    assert compute_stage1_coverage(probabilities, threshold=0.5) == 0.5


def test_return_direction_metrics_report_up_and_down_scores() -> None:
    y_true = pd.Series([0.01, -0.02, 0.0, 0.03], dtype="float64")
    predicted_returns = pd.Series([0.005, -0.01, 0.0, 0.02], dtype="float64")
    metrics = compute_return_direction_metrics(y_true, predicted_returns)
    assert metrics["sample_count"] == 4.0
    assert metrics["stage2_trade_count"] == 3.0
    assert metrics["direction_accuracy"] == 1.0
    assert metrics["trade_precision_up"] == 1.0
    assert metrics["trade_precision_down"] == 1.0
    assert "class_pnl.up" in metrics
    assert "trade_pnl.pnl_per_trade" in metrics


def test_end_to_end_metrics_respect_stage1_gate_and_return_sign() -> None:
    y_true = pd.Series([0.02, -0.01, 0.0, 0.03], dtype="float64")
    stage1_probabilities = pd.Series([0.8, 0.75, 0.4, 0.9], dtype="float64")
    stage2_predictions = pd.Series([0.01, -0.02, float("nan"), 0.01], dtype="float64")
    metrics = compute_two_stage_end_to_end_metrics(
        y_true,
        stage1_probabilities,
        stage2_predictions,
        stage1_threshold=0.5,
    )
    assert metrics["stage1_selected_count"] == 3.0
    assert metrics["stage2_trade_count"] == 3.0
    assert metrics["trade_precision_up"] == 1.0
    assert metrics["trade_precision_down"] == 1.0
    assert metrics["trade_recall_up"] == 1.0
    assert metrics["trade_recall_down"] == 1.0
    assert metrics["trade_pnl.pnl_per_sample"] > 0.0


def test_purged_chronological_time_window_split_uses_tail_window() -> None:
    training = _build_training_frame(length=400)
    X_train, X_valid, y_train, y_valid, split = purged_chronological_time_window_split(
        training,
        validation_window_days=1,
        purge_rows=2,
    )
    assert len(X_train) == len(y_train)
    assert len(X_valid) == len(y_valid)
    assert split.train_end < split.valid_start
    assert split.valid_end == 400


def test_summarize_walk_forward_marks_empty_results_as_disabled() -> None:
    assert summarize_walk_forward([]) == {"enabled": False, "fold_count": 0}
