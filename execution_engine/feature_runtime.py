from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from execution_engine.artifacts import BaselineArtifact
from src.calibration.registry import load_calibration_plugin
from src.core.config import Settings
from src.core.constants import DEFAULT_TIMESTAMP_COLUMN
from src.core.schemas import Signal
from src.data.second_level_features import (
    build_second_level_kline_feature_store,
    sample_second_level_feature_store,
)
from src.features.builder import build_feature_frame
from src.model.infer import predict_frame
from src.model.registry import load_model_plugin


@dataclass(frozen=True)
class FeatureBuildResult:
    feature_frame: pd.DataFrame
    second_level_frame: pd.DataFrame
    signal: Signal


class RuntimeInferenceEngine:
    def __init__(self, settings: Settings, baseline: BaselineArtifact, horizon_name: str = "5m") -> None:
        self.settings = settings
        self.baseline = baseline
        self.horizon_name = horizon_name
        self.model = load_model_plugin(baseline.model_plugin, str(baseline.model_path))
        self.calibrator = load_calibration_plugin(baseline.calibration_plugin, str(baseline.calibrator_path))

    def build_feature_frame(self, minute_frame: pd.DataFrame, second_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        decision_frame = minute_frame[[DEFAULT_TIMESTAMP_COLUMN]].copy()
        second_store = build_second_level_kline_feature_store(
            kline_frame=second_frame,
            feature_profile=self.settings.second_level.get_profile_payload(),
        )
        sampled_second = sample_second_level_feature_store(decision_frame, second_store)
        feature_frame = build_feature_frame(
            minute_frame,
            self.settings,
            horizon_name=self.horizon_name,
            select_grid_only=True,
            second_level_features_frame=sampled_second,
        )
        self._validate_feature_columns(feature_frame)
        return feature_frame, sampled_second

    def predict(self, minute_frame: pd.DataFrame, second_frame: pd.DataFrame) -> FeatureBuildResult:
        feature_frame, sampled_second = self.build_feature_frame(minute_frame, second_frame)
        probabilities = predict_frame(
            feature_frame,
            self.model,
            calibrator=self.calibrator,
            feature_columns=self.baseline.feature_columns,
        )
        latest = feature_frame.iloc[-1]
        p_up = float(probabilities.iloc[-1])
        signal = Signal(
            asset=str(latest["asset"]),
            horizon=str(latest["horizon"]),
            t0=latest[DEFAULT_TIMESTAMP_COLUMN].to_pydatetime(),
            p_up=p_up,
            p_down=1.0 - p_up,
            p_flat=None,
            p_active=None,
            model_version=f"{self.baseline.model_plugin}:{self.baseline.artifact_dir.name}",
            feature_version=str(latest["feature_version"]),
            decision_context={
                "grid_id": latest["grid_id"],
                "timestamp": latest[DEFAULT_TIMESTAMP_COLUMN].isoformat(),
                "t_up": self.baseline.t_up,
                "t_down": self.baseline.t_down,
                "baseline_artifact_dir": str(self.baseline.artifact_dir),
            },
        )
        return FeatureBuildResult(feature_frame=feature_frame, second_level_frame=sampled_second, signal=signal)

    def _validate_feature_columns(self, feature_frame: pd.DataFrame) -> None:
        missing = [column for column in self.baseline.feature_columns if column not in feature_frame.columns]
        if missing:
            preview = ", ".join(missing[:20])
            raise ValueError(
                f"Runtime feature frame is missing {len(missing)} baseline features: {preview}"
            )
