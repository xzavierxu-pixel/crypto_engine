from __future__ import annotations

import pandas as pd

from scripts.run_balanced_precision_holdout_experiment import _metric_dict, _window_dict
from src.core.config import load_settings


def test_metric_dict_includes_required_report_fields() -> None:
    settings = load_settings()
    y_true = pd.Series([1, 0, 1, 0, 1, 0])
    proba = pd.Series([0.8, 0.2, 0.55, 0.45, 0.7, 0.3])

    metrics = _metric_dict(y_true, proba, t_up=0.55, t_down=0.45, settings=settings)

    required = {
        "sample_count",
        "coverage",
        "precision_up",
        "precision_down",
        "balanced_precision",
        "all_sample_accuracy",
        "accepted_sample_accuracy",
        "share_up_predictions",
        "share_down_predictions",
        "selected_t_up",
        "selected_t_down",
        "accepted_count",
        "up_prediction_count",
        "down_prediction_count",
        "roc_auc",
        "brier_score",
        "log_loss",
        "up_signal_count",
        "down_signal_count",
        "total_signal_count",
        "signal_coverage",
        "overall_signal_accuracy",
    }
    assert required.issubset(metrics)
    assert metrics["selected_t_up"] == 0.55
    assert metrics["selected_t_down"] == 0.45
    assert metrics["accepted_count"] == metrics["total_signal_count"]
    assert metrics["up_prediction_count"] == metrics["up_signal_count"]
    assert metrics["down_prediction_count"] == metrics["down_signal_count"]
    assert metrics["coverage"] == metrics["signal_coverage"]
    assert metrics["accepted_sample_accuracy"] == metrics["overall_signal_accuracy"]


def test_window_dict_uses_report_window_shape() -> None:
    split_info = {
        "development": {
            "rows": 10,
            "start": "2026-02-01T00:00:00+00:00",
            "end": "2026-02-02T00:00:00+00:00",
            "target_mean": 0.5,
        }
    }

    assert _window_dict(split_info, "development") == {
        "row_count": 10,
        "start": "2026-02-01T00:00:00+00:00",
        "end": "2026-02-02T00:00:00+00:00",
    }
