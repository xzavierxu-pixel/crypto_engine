from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.model.train import train_two_stage_model
from src.services.signal_service import SignalService


def test_signal_service_emits_signal_from_latest_grid_row() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=5000, freq="1min"),
            "open": [100 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "high": [101 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "low": [99 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "close": [100 + np.sin(index / 3.0) * 4 for index in range(5000)],
            "volume": [10 + index for index in range(5000)],
        }
    )

    training = build_training_frame(frame, settings, horizon_name="5m")
    artifacts = train_two_stage_model(training, settings, validation_window_days=1)
    service = SignalService(
        settings,
        stage1_model=artifacts.stage1_model,
        stage2_model=artifacts.stage2_model,
        stage1_calibrator=artifacts.stage1_calibrator,
        stage2_calibrator=artifacts.stage2_calibrator,
        feature_columns=artifacts.feature_columns,
        stage2_feature_columns=artifacts.stage2_feature_columns,
        stage1_threshold=artifacts.stage1_threshold,
        buy_threshold=artifacts.buy_threshold,
    )

    signal = service.predict_from_latest_frame(frame, horizon_name="5m")
    expected_grid_id = frame["timestamp"].iloc[-1].floor("5min").strftime("%Y%m%d%H%M")

    assert signal.asset == "BTC/USDT"
    assert signal.horizon == "5m"
    assert 0.0 <= signal.p_up <= 1.0
    assert 0.0 <= float(signal.p_active or 0.0) <= 1.0
    assert signal.feature_version == "v4"
    assert signal.decision_context["grid_id"] == expected_grid_id
    assert signal.decision_context["stage1_drift"] is None


def test_signal_service_emits_15m_signal_from_latest_grid_row() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=5000, freq="1min"),
            "open": [100 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "high": [101 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "low": [99 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "close": [100 + np.sin(index / 3.0) * 4 for index in range(5000)],
            "volume": [10 + index for index in range(5000)],
        }
    )

    training = build_training_frame(frame, settings, horizon_name="15m")
    artifacts = train_two_stage_model(training, settings, validation_window_days=1)
    service = SignalService(
        settings,
        stage1_model=artifacts.stage1_model,
        stage2_model=artifacts.stage2_model,
        stage1_calibrator=artifacts.stage1_calibrator,
        stage2_calibrator=artifacts.stage2_calibrator,
        feature_columns=artifacts.feature_columns,
        stage2_feature_columns=artifacts.stage2_feature_columns,
        stage1_threshold=artifacts.stage1_threshold,
        buy_threshold=artifacts.buy_threshold,
    )

    signal = service.predict_from_latest_frame(frame, horizon_name="15m")
    expected_grid_id = frame["timestamp"].iloc[-1].floor("15min").strftime("%Y%m%d%H%M")

    assert signal.asset == "BTC/USDT"
    assert signal.horizon == "15m"
    assert 0.0 <= signal.p_up <= 1.0
    assert 0.0 <= float(signal.p_active or 0.0) <= 1.0
    assert signal.feature_version == "v4"
    assert signal.decision_context["grid_id"] == expected_grid_id


def test_signal_service_predicts_from_preheated_snapshot() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=5000, freq="1min"),
            "open": [100 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "high": [101 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "low": [99 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "close": [100 + np.sin(index / 3.0) * 4 for index in range(5000)],
            "volume": [10 + index for index in range(5000)],
        }
    )

    training = build_training_frame(frame, settings, horizon_name="5m")
    artifacts = train_two_stage_model(training, settings, validation_window_days=1)
    service = SignalService(
        settings,
        stage1_model=artifacts.stage1_model,
        stage2_model=artifacts.stage2_model,
        stage1_calibrator=artifacts.stage1_calibrator,
        stage2_calibrator=artifacts.stage2_calibrator,
        feature_columns=artifacts.feature_columns,
        stage2_feature_columns=artifacts.stage2_feature_columns,
        stage1_threshold=artifacts.stage1_threshold,
        buy_threshold=artifacts.buy_threshold,
    )

    direct_signal = service.predict_from_latest_frame(frame, horizon_name="5m")
    service.preheat_features(frame, horizon_name="5m")
    preheated_signal = service.predict_from_preheated_snapshot("5m")

    assert round(direct_signal.p_up, 8) == round(preheated_signal.p_up, 8)
    assert round(float(direct_signal.p_active or 0.0), 8) == round(float(preheated_signal.p_active or 0.0), 8)
    assert direct_signal.decision_context["grid_id"] == preheated_signal.decision_context["grid_id"]
    assert preheated_signal.decision_context["preheated"] is True
