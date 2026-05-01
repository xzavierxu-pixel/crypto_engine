from __future__ import annotations

import pandas as pd

from src.data.dataset_builder import TrainingFrame
from src.model.rolling import build_recent_rolling_splits, summarize_binary_rolling_results


def test_build_recent_rolling_splits_respects_train_validation_windows() -> None:
    rows = 90 * 24 * 12
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="5min"),
            "f1": 1.0,
            "target": [0, 1] * (rows // 2),
            "stage1_sample_weight": 1.0,
        }
    )
    training = TrainingFrame(
        frame=frame,
        feature_columns=["f1"],
        target_column="target",
        sample_weight_column="stage1_sample_weight",
    )
    splits = build_recent_rolling_splits(
        training,
        train_days_list=[30, 60],
        validation_days=15,
        fold_count=2,
        step_days=15,
        purge_rows=1,
    )

    assert len(splits) == 4
    first = splits[0]
    assert first.train_days == 30
    assert first.fold_index == 0
    assert len(first.development.frame) == 30 * 24 * 12 - 1
    assert len(first.validation.frame) == 15 * 24 * 12 + 1
    assert first.development.frame["timestamp"].max() < first.validation.frame["timestamp"].min()


def test_summarize_binary_rolling_results_ranks_by_balanced_precision() -> None:
    summary = summarize_binary_rolling_results(
        [
            {
                "train_days": 30,
                "balanced_precision": 0.55,
                "coverage": 0.60,
                "precision_up": 0.57,
                "precision_down": 0.53,
                "accepted_sample_accuracy": 0.55,
                "roc_auc": 0.51,
                "t_up": 0.52,
                "t_down": 0.48,
                "constraint_satisfied": True,
                "side_guardrail_constraint_satisfied": False,
            },
            {
                "train_days": 60,
                "balanced_precision": 0.58,
                "coverage": 0.62,
                "precision_up": 0.60,
                "precision_down": 0.56,
                "accepted_sample_accuracy": 0.58,
                "roc_auc": 0.54,
                "t_up": 0.53,
                "t_down": 0.47,
                "constraint_satisfied": True,
                "side_guardrail_constraint_satisfied": True,
            },
        ]
    )

    assert summary["result_count"] == 2
    assert summary["best_train_days"] == 60
    assert summary["by_train_days"][0]["constraint_pass_rate"] == 1.0
