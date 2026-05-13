from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from execution_engine.artifacts import BaselineArtifact
from src.calibration.registry import load_calibration_plugin
from src.core.config import Settings
from src.core.constants import DEFAULT_TIMESTAMP_COLUMN
from src.core.schemas import Signal
from src.data.second_level_features import (
    build_second_level_feature_store,
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
    def __init__(
        self,
        settings: Settings,
        baseline: BaselineArtifact,
        horizon_name: str = "5m",
        t_up: float | None = None,
        t_down: float | None = None,
    ) -> None:
        self.settings = settings
        self.baseline = baseline
        self.horizon_name = horizon_name
        self.t_up = baseline.t_up if t_up is None else float(t_up)
        self.t_down = baseline.t_down if t_down is None else float(t_down)
        self.model = load_model_plugin(baseline.model_plugin, str(baseline.model_path))
        self.calibrator = load_calibration_plugin(baseline.calibration_plugin, str(baseline.calibrator_path))

    def build_feature_frame(
        self,
        minute_frame: pd.DataFrame,
        second_frame: pd.DataFrame,
        agg_trades_frame: pd.DataFrame | None = None,
        select_grid_only: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        decision_frame = minute_frame[[DEFAULT_TIMESTAMP_COLUMN]].copy()
        second_store = build_second_level_feature_store(
            kline_frame=second_frame,
            agg_trades_frame=agg_trades_frame,
            feature_profile=self.settings.second_level.get_profile_payload(),
        )
        sampled_second = sample_second_level_feature_store(decision_frame, second_store)
        feature_frame = build_feature_frame(
            minute_frame,
            self.settings,
            horizon_name=self.horizon_name,
            select_grid_only=select_grid_only,
            second_level_features_frame=sampled_second,
        )
        self._validate_feature_columns(feature_frame)
        return feature_frame, sampled_second

    def predict(
        self,
        minute_frame: pd.DataFrame,
        second_frame: pd.DataFrame,
        agg_trades_frame: pd.DataFrame | None = None,
        signal_t0: pd.Timestamp | None = None,
        use_latest_available_before_signal: bool = False,
    ) -> FeatureBuildResult:
        feature_frame, sampled_second = self.build_feature_frame(
            minute_frame,
            second_frame,
            agg_trades_frame,
            select_grid_only=not use_latest_available_before_signal,
        )
        probabilities = predict_frame(
            feature_frame,
            self.model,
            calibrator=self.calibrator,
            feature_columns=self.baseline.feature_columns,
        )
        if signal_t0 is None:
            row_index = feature_frame.index[-1]
        else:
            target_t0 = pd.Timestamp(signal_t0).tz_convert("UTC")
            feature_timestamps = pd.to_datetime(feature_frame[DEFAULT_TIMESTAMP_COLUMN], utc=True)
            if use_latest_available_before_signal:
                matches = feature_frame.index[feature_timestamps < target_t0]
                if matches.empty:
                    raise RuntimeError(
                        "Feature frame does not include a closed row before requested "
                        f"signal_t0 '{target_t0.isoformat()}'."
                    )
                row_index = matches[-1]
            else:
                matches = feature_frame.index[feature_timestamps == target_t0]
                if matches.empty:
                    raise RuntimeError(f"Feature frame does not include requested signal_t0 '{target_t0.isoformat()}'.")
                row_index = matches[-1]
        latest = feature_frame.loc[row_index]
        p_up = float(probabilities.loc[row_index])
        signal_timestamp = (
            pd.Timestamp(signal_t0).tz_convert("UTC").to_pydatetime()
            if signal_t0 is not None and use_latest_available_before_signal
            else latest[DEFAULT_TIMESTAMP_COLUMN].to_pydatetime()
        )
        signal = Signal(
            asset=str(latest["asset"]),
            horizon=str(latest["horizon"]),
            t0=signal_timestamp,
            p_up=p_up,
            p_down=1.0 - p_up,
            p_flat=None,
            p_active=None,
            model_version=f"{self.baseline.model_plugin}:{self.baseline.artifact_dir.name}",
            feature_version=str(latest["feature_version"]),
            decision_context={
                "grid_id": latest["grid_id"],
                "timestamp": signal_timestamp.isoformat(),
                "feature_timestamp": latest[DEFAULT_TIMESTAMP_COLUMN].isoformat(),
                "t_up": self.t_up,
                "t_down": self.t_down,
                "artifact_t_up": self.baseline.t_up,
                "artifact_t_down": self.baseline.t_down,
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
