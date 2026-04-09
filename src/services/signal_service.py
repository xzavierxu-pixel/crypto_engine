from __future__ import annotations

import pandas as pd

from src.calibration.base import CalibrationPlugin
from src.core.config import Settings
from src.core.schemas import Signal
from src.core.versioning import hash_config
from src.model.base import ModelPlugin
from src.model.infer import predict_frame
from src.services.feature_service import FeatureService


class SignalService:
    def __init__(
        self,
        settings: Settings,
        model: ModelPlugin,
        calibrator: CalibrationPlugin | None = None,
        feature_columns: list[str] | None = None,
        model_version: str | None = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.calibrator = calibrator
        self.feature_columns = feature_columns
        self.feature_service = FeatureService(settings)
        self.model_version = model_version or f"{settings.model.active_plugin}:{hash_config(settings)}"

    def predict_frame(self, frame: pd.DataFrame, horizon_name: str = "5m") -> pd.Series:
        feature_frame = self.feature_service.build_feature_frame(
            frame,
            horizon_name=horizon_name,
            select_grid_only=True,
        )
        return predict_frame(
            feature_frame,
            self.model,
            calibrator=self.calibrator,
            feature_columns=self.feature_columns,
        )

    def predict_from_latest_frame(self, frame: pd.DataFrame, horizon_name: str = "5m") -> Signal:
        feature_frame = self.feature_service.build_feature_frame(
            frame,
            horizon_name=horizon_name,
            select_grid_only=True,
        )
        probabilities = predict_frame(
            feature_frame,
            self.model,
            calibrator=self.calibrator,
            feature_columns=self.feature_columns,
        )
        latest_row = feature_frame.iloc[-1]
        latest_probability = float(probabilities.iloc[-1])
        return Signal(
            asset=str(latest_row["asset"]),
            horizon=str(latest_row["horizon"]),
            t0=latest_row["timestamp"].to_pydatetime(),
            p_up=latest_probability,
            model_version=self.model_version,
            feature_version=str(latest_row["feature_version"]),
            decision_context={
                "grid_id": latest_row["grid_id"],
                "timestamp": latest_row["timestamp"].isoformat(),
            },
        )
