from __future__ import annotations

import pandas as pd

from src.calibration.base import CalibrationPlugin
from src.core.config import Settings
from src.core.schemas import Signal
from src.core.versioning import hash_config
from src.model.base import ModelPlugin
from src.model.drift import Stage1DriftMonitor
from src.model.infer import predict_frame as infer_frame
from src.services.feature_service import FeatureService


class SignalService:
    def __init__(
        self,
        settings: Settings,
        model: ModelPlugin | None = None,
        stage1_model: ModelPlugin | None = None,
        stage2_model: ModelPlugin | None = None,
        calibrator: CalibrationPlugin | None = None,
        stage1_calibrator: CalibrationPlugin | None = None,
        stage2_calibrator: CalibrationPlugin | None = None,
        feature_columns: list[str] | None = None,
        stage2_feature_columns: list[str] | None = None,
        model_version: str | None = None,
        t_up: float | None = None,
        t_down: float | None = None,
        stage1_threshold: float | None = None,
        up_threshold: float | None = None,
        down_threshold: float | None = None,
        margin_threshold: float | None = None,
        stage1_drift_monitor: Stage1DriftMonitor | None = None,
    ) -> None:
        self.settings = settings
        self.model = model or stage1_model
        if self.model is None:
            raise ValueError("model must be provided for binary signal inference.")
        self.calibrator = calibrator or stage1_calibrator
        self.feature_columns = feature_columns
        self.feature_service = FeatureService(settings)
        self.model_version = model_version or f"{settings.model.resolve_plugin(stage='binary')}:{hash_config(settings)}"
        self.t_up = t_up if t_up is not None else up_threshold
        self.t_down = t_down if t_down is not None else down_threshold
        self.drift_monitor = stage1_drift_monitor

    def _predict_from_feature_frame(self, feature_frame: pd.DataFrame) -> pd.Series:
        return infer_frame(
            feature_frame,
            self.model,
            calibrator=self.calibrator,
            feature_columns=self.feature_columns,
        )

    def _build_signal_from_feature_frame(
        self,
        feature_frame: pd.DataFrame,
        probabilities: pd.Series,
    ) -> Signal:
        latest_row = feature_frame.iloc[-1]
        latest_probability = float(probabilities.iloc[-1])
        drift_state = (
            self.drift_monitor.update(latest_probability)
            if self.drift_monitor is not None
            else None
        )
        return Signal(
            asset=str(latest_row["asset"]),
            horizon=str(latest_row["horizon"]),
            t0=latest_row["timestamp"].to_pydatetime(),
            p_down=1.0 - latest_probability,
            p_flat=None,
            p_up=latest_probability,
            p_active=None,
            model_version=self.model_version,
            feature_version=str(latest_row["feature_version"]),
            decision_context={
                "grid_id": latest_row["grid_id"],
                "timestamp": latest_row["timestamp"].isoformat(),
                "t_up": self.t_up,
                "t_down": self.t_down,
                "p_up_drift": drift_state,
            },
        )

    def preheat_features(
        self,
        frame: pd.DataFrame,
        horizon_name: str = "5m",
        derivatives_frame: pd.DataFrame | None = None,
    ) -> pd.Series:
        snapshot = self.feature_service.preheat_latest_feature_snapshot(
            frame,
            horizon_name=horizon_name,
            derivatives_frame=derivatives_frame,
        )
        return snapshot.row

    def predict_frame(
        self,
        frame: pd.DataFrame,
        horizon_name: str = "5m",
        derivatives_frame: pd.DataFrame | None = None,
    ) -> pd.Series:
        feature_frame = self.feature_service.build_feature_frame(
            frame,
            horizon_name=horizon_name,
            select_grid_only=True,
            derivatives_frame=derivatives_frame,
        )
        return self._predict_from_feature_frame(feature_frame)

    def predict_from_latest_frame(
        self,
        frame: pd.DataFrame,
        horizon_name: str = "5m",
        derivatives_frame: pd.DataFrame | None = None,
    ) -> Signal:
        feature_frame = self.feature_service.build_feature_frame(
            frame,
            horizon_name=horizon_name,
            select_grid_only=True,
            derivatives_frame=derivatives_frame,
        )
        probabilities = self._predict_from_feature_frame(feature_frame)
        return self._build_signal_from_feature_frame(feature_frame, probabilities)

    def predict_from_preheated_snapshot(
        self,
        horizon_name: str = "5m",
    ) -> Signal:
        snapshot = self.feature_service.get_preheated_snapshot(horizon_name)
        feature_frame = pd.DataFrame([snapshot.row])
        probabilities = self._predict_from_feature_frame(feature_frame)
        signal = self._build_signal_from_feature_frame(feature_frame, probabilities)
        signal.decision_context["source_timestamp"] = snapshot.source_timestamp.isoformat()
        signal.decision_context["preheated"] = True
        return signal
