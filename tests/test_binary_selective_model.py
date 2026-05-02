from __future__ import annotations

import pandas as pd

from src.core.config import load_settings
from src.data.dataset_builder import TrainingFrame, compute_sample_weight, infer_feature_columns
from src.model.evaluation import compute_selective_binary_metrics, search_selective_binary_thresholds
from src.model.train import split_recent_train_validation_frame


def test_linear_ramp_sample_weight_matches_prd_boundaries() -> None:
    settings = load_settings()
    weights = compute_sample_weight(
        pd.Series([0.0, 0.0001, 0.0002, 0.0003, 0.001], dtype="float64"),
        settings=settings,
    )
    assert list(weights.round(2)) == [0.35, 0.57, 0.78, 1.00, 1.00]


def test_selective_binary_metrics_reports_prd_fields() -> None:
    y_true = pd.Series([1, 1, 0, 0, 1], dtype="int64")
    probabilities = pd.Series([0.61, 0.55, 0.39, 0.45, 0.52], dtype="float64")
    metrics = compute_selective_binary_metrics(y_true, probabilities, t_up=0.60, t_down=0.40)
    assert metrics["coverage"] == 0.4
    assert metrics["precision_up"] == 1.0
    assert metrics["precision_down"] == 1.0
    assert metrics["balanced_precision"] == 1.0
    assert metrics["share_up_predictions"] == 0.5
    assert metrics["share_down_predictions"] == 0.5


def test_threshold_search_prefers_higher_coverage_inside_precision_tie() -> None:
    y_true = pd.Series([1, 1, 0, 0], dtype="int64")
    probabilities = pd.Series([0.61, 0.56, 0.39, 0.44], dtype="float64")
    t_up, t_down, frontier, best = search_selective_binary_thresholds(
        y_true,
        probabilities,
        t_up_min=0.55,
        t_up_max=0.60,
        t_down_min=0.40,
        t_down_max=0.45,
        step=0.05,
        min_coverage=0.50,
        tie_tolerance=0.002,
    )
    assert t_up == 0.55
    assert t_down == 0.45
    assert best["coverage"] == 1.0
    assert not frontier.empty


def test_recent_split_uses_30_day_train_and_30_day_validation_windows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T00:00:00Z", periods=61 * 24 * 12, freq="5min"),
            "f1": 1.0,
            "target": [0, 1] * (61 * 24 * 6),
        }
    )
    training = TrainingFrame(frame=frame, feature_columns=["f1"])
    development, validation = split_recent_train_validation_frame(
        training,
        train_days=30,
        validation_days=30,
        purge_rows=1,
    )
    assert len(development.frame) == 30 * 24 * 12 - 1
    assert len(validation.frame) == 30 * 24 * 12 + 1
    assert development.frame["timestamp"].max() < validation.frame["timestamp"].min()


def test_raw_metadata_columns_are_not_selected_as_features() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01T00:00:00Z")],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
            "quote_volume": [100.0],
            "taker_buy_volume": [50.0],
            "taker_buy_quote_volume": [5000.0],
            "count": [10],
            "open_time": [123],
            "ret_1": [0.01],
            "rv_5": [0.02],
        }
    )
    assert infer_feature_columns(frame) == ["ret_1", "rv_5"]


def test_completed_bar_microstructure_features_are_legal_features() -> None:
    from src.features.prd_microstructure import CompletedBarMicrostructureFeaturePack

    settings = load_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="1min"),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 102.0, 103.0, 104.0],
            "volume": [10.0, 20.0, 30.0, 40.0],
            "quote_volume": [1000.0, 2200.0, 3300.0, 4400.0],
            "count": [5.0, 10.0, 15.0, 20.0],
            "taker_buy_volume": [6.0, 15.0, 12.0, 30.0],
            "taker_buy_quote_volume": [600.0, 1700.0, 1300.0, 3300.0],
        }
    )
    features = CompletedBarMicrostructureFeaturePack().transform(
        frame,
        settings,
        settings.features.get_profile("core_5m"),
    )

    assert features.loc[2, "prev_bar_taker_buy_ratio"] == 0.75
    assert features.loc[2, "prev_bar_taker_imbalance"] == 0.5
    assert "legal_prev_trade_count" not in features.columns
    assert "legal_prev_taker_buy_base_volume" not in features.columns
    assert "legal_prev_taker_buy_base_volume_share" not in features.columns
    assert features.loc[3, "legal_prev_trade_count_sum_3"] == 30.0
    assert features.loc[3, "legal_prev_relative_taker_buy_base_volume_3"] == 12.0 / 11.0
    assert "prev_bar_return" in features.columns
    assert infer_feature_columns(features) == list(features.columns)
