from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import replace

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.model.drift import Stage1DriftMonitor
from src.model.train import train_binary_selective_model
from src.services.signal_service import SignalService


def _build_frame(length: int = 3500) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=length, freq="1min"),
            "open": [100 + np.sin(index / 4.0) * 3 for index in range(length)],
            "high": [101 + np.sin(index / 4.0) * 3 for index in range(length)],
            "low": [99 + np.sin(index / 4.0) * 3 for index in range(length)],
            "close": [100 + np.sin(index / 3.0) * 4 for index in range(length)],
            "volume": [10 + index for index in range(length)],
        }
    )


def _train_binary_artifacts():
    settings = load_settings()
    settings = replace(settings, derivatives=replace(settings.derivatives, enabled=False))
    frame = _build_frame()
    training = build_training_frame(frame, settings, horizon_name="5m")
    return settings, frame, train_binary_selective_model(
        training,
        settings,
        train_days=1,
        validation_days=1,
        purge_rows=1,
    )


def test_signal_service_emits_binary_signal_from_latest_grid_row() -> None:
    settings, frame, artifacts = _train_binary_artifacts()
    service = SignalService(
        settings,
        model=artifacts.model,
        calibrator=artifacts.calibrator,
        feature_columns=artifacts.feature_columns,
        t_up=artifacts.t_up,
        t_down=artifacts.t_down,
    )
    signal = service.predict_from_latest_frame(frame, horizon_name="5m")
    expected_grid_id = frame["timestamp"].iloc[-1].floor("5min").strftime("%Y%m%d%H%M")
    assert signal.asset == "BTC/USDT"
    assert signal.horizon == "5m"
    assert 0.0 <= float(signal.p_up or 0.0) <= 1.0
    assert 0.0 <= float(signal.p_down or 0.0) <= 1.0
    assert signal.p_flat is None
    assert signal.decision_context["grid_id"] == expected_grid_id
    assert signal.decision_context["t_up"] == artifacts.t_up
    assert signal.decision_context["t_down"] == artifacts.t_down


def test_signal_service_predicts_from_preheated_snapshot() -> None:
    settings, frame, artifacts = _train_binary_artifacts()
    service = SignalService(
        settings,
        model=artifacts.model,
        calibrator=artifacts.calibrator,
        feature_columns=artifacts.feature_columns,
        t_up=artifacts.t_up,
        t_down=artifacts.t_down,
    )
    direct_signal = service.predict_from_latest_frame(frame, horizon_name="5m")
    service.preheat_features(frame, horizon_name="5m")
    preheated_signal = service.predict_from_preheated_snapshot("5m")
    assert round(float(direct_signal.p_up or 0.0), 8) == round(float(preheated_signal.p_up or 0.0), 8)
    assert direct_signal.decision_context["grid_id"] == preheated_signal.decision_context["grid_id"]
    assert preheated_signal.decision_context["preheated"] is True


def test_signal_service_updates_binary_probability_drift() -> None:
    settings, frame, artifacts = _train_binary_artifacts()
    service = SignalService(
        settings,
        model=artifacts.model,
        calibrator=artifacts.calibrator,
        feature_columns=artifacts.feature_columns,
        t_up=artifacts.t_up,
        t_down=artifacts.t_down,
        stage1_drift_monitor=Stage1DriftMonitor(
            pd.Series([0.1, 0.2, 0.3] * 50, dtype="float64"),
            threshold=0.0,
            window_size=10,
            min_history=1,
            alert_consecutive=1,
        ),
    )
    signal = service.predict_from_latest_frame(frame, horizon_name="5m")
    assert signal.decision_context["p_up_drift"] is not None
