from __future__ import annotations

import math
import numpy as np
import pandas as pd

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.model.drift import Stage2DirectionDriftMonitor
from src.model.train import train_two_stage_model
from src.services.signal_service import SignalService


def _build_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=2500, freq="1min"),
            "open": [100 + np.sin(index / 4.0) * 3 for index in range(2500)],
            "high": [101 + np.sin(index / 4.0) * 3 for index in range(2500)],
            "low": [99 + np.sin(index / 4.0) * 3 for index in range(2500)],
            "close": [100 + np.sin(index / 3.0) * 4 for index in range(2500)],
            "volume": [10 + index for index in range(2500)],
        }
    )


def test_signal_service_emits_signal_from_latest_grid_row() -> None:
    settings = load_settings()
    frame = _build_frame()
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
        up_threshold=artifacts.up_threshold,
        down_threshold=artifacts.down_threshold,
        margin_threshold=artifacts.margin_threshold,
    )
    signal = service.predict_from_latest_frame(frame, horizon_name="5m")
    expected_grid_id = frame["timestamp"].iloc[-1].floor("5min").strftime("%Y%m%d%H%M")
    assert signal.asset == "BTC/USDT"
    assert signal.horizon == "5m"
    assert 0.0 <= float(signal.p_active or 0.0) <= 1.0
    if signal.decision_context["stage1_rejected"]:
        assert math.isnan(float(signal.p_up))
        assert math.isnan(float(signal.p_down))
        assert math.isnan(float(signal.p_flat))
    else:
        assert 0.0 <= float(signal.p_up or 0.0) <= 1.0
        assert 0.0 <= float(signal.p_down or 0.0) <= 1.0
        assert 0.0 <= float(signal.p_flat or 0.0) <= 1.0
    assert signal.feature_version == "v4"
    assert signal.decision_context["grid_id"] == expected_grid_id
    assert signal.decision_context["stage1_drift"] is None


def test_signal_service_predicts_from_preheated_snapshot() -> None:
    settings = load_settings()
    frame = _build_frame()
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
        up_threshold=artifacts.up_threshold,
        down_threshold=artifacts.down_threshold,
        margin_threshold=artifacts.margin_threshold,
    )
    direct_signal = service.predict_from_latest_frame(frame, horizon_name="5m")
    service.preheat_features(frame, horizon_name="5m")
    preheated_signal = service.predict_from_preheated_snapshot("5m")
    assert round(float(direct_signal.p_active or 0.0), 8) == round(float(preheated_signal.p_active or 0.0), 8)
    assert direct_signal.decision_context["grid_id"] == preheated_signal.decision_context["grid_id"]
    assert preheated_signal.decision_context["preheated"] is True


def test_signal_service_skips_stage2_when_stage1_rejects(monkeypatch) -> None:
    settings = load_settings()
    frame = _build_frame()
    training = build_training_frame(frame, settings, horizon_name="5m")
    artifacts = train_two_stage_model(training, settings, validation_window_days=1)

    class RejectStage1Model:
        def predict_proba(self, X):
            return pd.Series(0.0, index=X.index, dtype="float64")

    class FailingStage2Model:
        def predict_proba_multiclass(self, X):
            raise AssertionError("Stage 2 should not be called when Stage 1 rejects.")

    service = SignalService(
        settings,
        stage1_model=RejectStage1Model(),
        stage2_model=FailingStage2Model(),
        stage1_calibrator=artifacts.stage1_calibrator,
        stage2_calibrator=artifacts.stage2_calibrator,
        feature_columns=artifacts.feature_columns,
        stage2_feature_columns=artifacts.stage2_feature_columns,
        stage1_threshold=artifacts.stage1_threshold,
        up_threshold=artifacts.up_threshold,
        down_threshold=artifacts.down_threshold,
        margin_threshold=artifacts.margin_threshold,
    )
    signal = service.predict_from_latest_frame(frame, horizon_name="5m")
    assert signal.decision_context["stage1_rejected"] is True
    assert math.isnan(float(signal.p_up))
    assert math.isnan(float(signal.p_down))
    assert math.isnan(float(signal.p_flat))


def test_signal_service_updates_stage2_drift_when_stage2_runs() -> None:
    settings = load_settings()
    frame = _build_frame()
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
        stage1_threshold=0.0,
        up_threshold=artifacts.up_threshold,
        down_threshold=artifacts.down_threshold,
        margin_threshold=artifacts.margin_threshold,
        stage2_drift_monitor=Stage2DirectionDriftMonitor(
            pd.Series([0.0, 0.1, 0.2] * 50, dtype="float64"),
            threshold=0.0,
            window_size=10,
            min_history=1,
            alert_consecutive=1,
        ),
    )
    signal = service.predict_from_latest_frame(frame, horizon_name="5m")
    assert signal.decision_context["stage1_rejected"] is False
    assert signal.decision_context["stage2_drift"] is not None
