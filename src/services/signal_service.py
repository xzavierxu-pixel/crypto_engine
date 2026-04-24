from __future__ import annotations

import pandas as pd

from src.calibration.base import CalibrationPlugin
from src.core.config import Settings
from src.core.constants import (
    DEFAULT_STAGE1_PROBABILITY_COLUMN,
    DEFAULT_STAGE2_PROBABILITY_DOWN_COLUMN,
    DEFAULT_STAGE2_PROBABILITY_FLAT_COLUMN,
    DEFAULT_STAGE2_PROBABILITY_UP_COLUMN,
)
from src.core.schemas import Signal
from src.core.versioning import hash_config
from src.model.base import ModelPlugin
from src.model.drift import Stage1DriftMonitor, Stage2DirectionDriftMonitor
from src.model.infer import predict_frame as infer_frame
from src.model.infer import predict_frame_multiclass
from src.services.feature_service import FeatureService


class SignalService:
    def __init__(
        self,
        settings: Settings,
        stage1_model: ModelPlugin,
        stage2_model: ModelPlugin,
        stage1_calibrator: CalibrationPlugin | None = None,
        stage2_calibrator: CalibrationPlugin | None = None,
        feature_columns: list[str] | None = None,
        stage2_feature_columns: list[str] | None = None,
        model_version: str | None = None,
        stage1_threshold: float | None = None,
        up_threshold: float | None = None,
        down_threshold: float | None = None,
        margin_threshold: float | None = None,
        stage1_drift_monitor: Stage1DriftMonitor | None = None,
        stage2_drift_monitor: Stage2DirectionDriftMonitor | None = None,
    ) -> None:
        self.settings = settings
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.stage1_calibrator = stage1_calibrator
        self.stage2_calibrator = stage2_calibrator
        self.feature_columns = feature_columns
        self.stage2_feature_columns = stage2_feature_columns or [*(feature_columns or []), DEFAULT_STAGE1_PROBABILITY_COLUMN]
        self.feature_service = FeatureService(settings)
        self.model_version = model_version or (
            f"{settings.model.resolve_plugin(stage='stage1')}+{settings.model.resolve_plugin(stage='stage2')}:"
            f"{hash_config(settings)}"
        )
        self.stage1_threshold = stage1_threshold
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.margin_threshold = margin_threshold
        self.stage1_drift_monitor = stage1_drift_monitor
        self.stage2_drift_monitor = stage2_drift_monitor

    def _predict_from_feature_frame(self, feature_frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        stage1_probabilities = infer_frame(
            feature_frame,
            self.stage1_model,
            calibrator=self.stage1_calibrator,
            feature_columns=self.feature_columns,
        )
        stage2_probabilities = pd.DataFrame(
            {
                DEFAULT_STAGE2_PROBABILITY_DOWN_COLUMN: pd.Series(float("nan"), index=feature_frame.index, dtype="float64"),
                DEFAULT_STAGE2_PROBABILITY_FLAT_COLUMN: pd.Series(float("nan"), index=feature_frame.index, dtype="float64"),
                DEFAULT_STAGE2_PROBABILITY_UP_COLUMN: pd.Series(float("nan"), index=feature_frame.index, dtype="float64"),
            }
        )
        if self.stage1_threshold is None:
            raise ValueError("stage1_threshold must be provided for online inference.")
        active_mask = stage1_probabilities >= float(self.stage1_threshold)
        if active_mask.any():
            stage2_frame = feature_frame.loc[active_mask].copy()
            stage2_frame[DEFAULT_STAGE1_PROBABILITY_COLUMN] = stage1_probabilities.loc[active_mask]
            stage2_predictions = predict_frame_multiclass(
                stage2_frame,
                self.stage2_model,
                feature_columns=self.stage2_feature_columns,
            )
            stage2_probabilities.loc[active_mask, stage2_predictions.columns] = stage2_predictions
        return stage1_probabilities, stage2_probabilities

    def _build_signal_from_feature_frame(
        self,
        feature_frame: pd.DataFrame,
        stage1_probabilities: pd.Series,
        probabilities: pd.DataFrame,
    ) -> Signal:
        latest_row = feature_frame.iloc[-1]
        latest_stage1_probability = float(stage1_probabilities.iloc[-1])
        latest_probabilities = probabilities.iloc[-1]
        drift_state = (
            self.stage1_drift_monitor.update(latest_stage1_probability)
            if self.stage1_drift_monitor is not None
            else None
        )
        stage1_rejected = latest_stage1_probability < float(self.stage1_threshold)
        stage2_drift_state = None
        if (
            not stage1_rejected
            and self.stage2_drift_monitor is not None
            and pd.notna(latest_probabilities.get(DEFAULT_STAGE2_PROBABILITY_UP_COLUMN))
            and pd.notna(latest_probabilities.get(DEFAULT_STAGE2_PROBABILITY_DOWN_COLUMN))
        ):
            stage2_drift_state = self.stage2_drift_monitor.update(
                float(latest_probabilities.get(DEFAULT_STAGE2_PROBABILITY_UP_COLUMN)),
                float(latest_probabilities.get(DEFAULT_STAGE2_PROBABILITY_DOWN_COLUMN)),
            )
        return Signal(
            asset=str(latest_row["asset"]),
            horizon=str(latest_row["horizon"]),
            t0=latest_row["timestamp"].to_pydatetime(),
            p_down=float(latest_probabilities.get(DEFAULT_STAGE2_PROBABILITY_DOWN_COLUMN)),
            p_flat=float(latest_probabilities.get(DEFAULT_STAGE2_PROBABILITY_FLAT_COLUMN)),
            p_up=float(latest_probabilities.get(DEFAULT_STAGE2_PROBABILITY_UP_COLUMN)),
            p_active=latest_stage1_probability,
            model_version=self.model_version,
            feature_version=str(latest_row["feature_version"]),
            decision_context={
                "grid_id": latest_row["grid_id"],
                "timestamp": latest_row["timestamp"].isoformat(),
                "stage1_threshold": self.stage1_threshold,
                "up_threshold": self.up_threshold,
                "down_threshold": self.down_threshold,
                "margin_threshold": self.margin_threshold,
                "stage1_rejected": stage1_rejected,
                "stage1_drift": drift_state,
                "stage2_drift": stage2_drift_state,
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
        _, probabilities = self._predict_from_feature_frame(feature_frame)
        return probabilities

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
        stage1_probabilities, probabilities = self._predict_from_feature_frame(feature_frame)
        return self._build_signal_from_feature_frame(feature_frame, stage1_probabilities, probabilities)

    def predict_from_preheated_snapshot(
        self,
        horizon_name: str = "5m",
    ) -> Signal:
        snapshot = self.feature_service.get_preheated_snapshot(horizon_name)
        feature_frame = pd.DataFrame([snapshot.row])
        stage1_probabilities, probabilities = self._predict_from_feature_frame(feature_frame)
        signal = self._build_signal_from_feature_frame(feature_frame, stage1_probabilities, probabilities)
        signal.decision_context["source_timestamp"] = snapshot.source_timestamp.isoformat()
        signal.decision_context["preheated"] = True
        return signal
