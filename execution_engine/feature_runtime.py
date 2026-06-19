from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import pandas as pd
import numpy as np

from execution_engine.artifacts import BaselineArtifact, PriceEstimatorArtifact
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
    row_index: Any | None = None


class RuntimeInferenceEngine:
    def __init__(
        self,
        settings: Settings,
        baseline: BaselineArtifact,
        price_estimator: PriceEstimatorArtifact | None = None,
        horizon_name: str = "5m",
        t_up: float | None = None,
        t_down: float | None = None,
    ) -> None:
        self.settings = settings
        self.baseline = baseline
        self.price_estimator = price_estimator
        self.horizon_name = horizon_name
        self.t_up = baseline.t_up if t_up is None else float(t_up)
        self.t_down = baseline.t_down if t_down is None else float(t_down)
        self.model = load_model_plugin(baseline.model_plugin, str(baseline.model_path))
        self.calibrator = load_calibration_plugin(baseline.calibration_plugin, str(baseline.calibrator_path))

    def predict_price_q80(
        self,
        feature_frame: pd.DataFrame,
        *,
        row_index: Any,
        selected_side: str,
        p_up: float | None = None,
    ) -> dict[str, Any] | None:
        if self.price_estimator is None:
            return None
        artifact = self.price_estimator
        side_value = artifact.yes_value if selected_side == "YES" else artifact.no_value
        frame = feature_frame.copy()
        frame[artifact.selected_side_column] = side_value
        if p_up is not None:
            p_up_value = float(p_up)
            p_side = p_up_value if selected_side == "YES" else 1.0 - p_up_value
            frame["p_up"] = p_up_value
            frame["p_side"] = p_side
            frame["direction_confidence"] = abs(p_up_value - 0.5)
            frame["p_bin"] = _p_side_bucket(p_side)
            frame["p_side_bucket"] = _p_side_bucket(p_side)
            timestamp = pd.to_datetime(frame.loc[row_index, DEFAULT_TIMESTAMP_COLUMN], utc=True)
            frame["market_time_bucket"] = _market_time_bucket(timestamp)
        missing = [column for column in artifact.feature_columns if column not in frame.columns]
        if missing:
            preview = ", ".join(missing[:20])
            raise ValueError(
                f"Runtime feature frame is missing {len(missing)} price estimator features: {preview}"
            )
        row = frame.loc[[row_index], artifact.feature_columns]
        raw_prediction = artifact.model.predict(row)
        raw_price = _extract_single_prediction(raw_prediction, prediction_column=artifact.prediction_column)
        if artifact.prediction_transform == "sigmoid":
            raw_price = float(1.0 / (1.0 + np.exp(-raw_price)))
        rounded_price = _round_price(raw_price, artifact.round_decimals)
        context = {
            "price_estimator_enabled": True,
            "price_estimator_artifact_dir": str(artifact.artifact_dir),
            "price_estimator_model_path": str(artifact.model_path),
            "price_estimator_prediction_column": artifact.prediction_column,
            "price_estimator_selected_side_column": artifact.selected_side_column,
            "price_estimator_selected_side": side_value,
            "price_estimator_q80_raw": raw_price,
            "price_estimator_q80_rounded": rounded_price,
            "price_estimator_round_decimals": artifact.round_decimals,
            "price_estimator_best_ask_offset": artifact.best_ask_offset,
            "price_estimator_fallback_price_mode": artifact.fallback_price_mode,
            "price_estimator_prediction_transform": artifact.prediction_transform,
        }
        if artifact.prediction_column == "p_pred":
            context.update(
                {
                    "price_estimator_safe_gap_raw": raw_price,
                    "price_estimator_safe_gap_rounded": rounded_price,
                }
            )
            if isinstance(raw_prediction, pd.DataFrame):
                for column in ("safe_gap_action", "safe_gap_conf_ok", "safe_gap_f_model"):
                    if column in raw_prediction.columns:
                        context[f"price_estimator_{column}"] = raw_prediction[column].iloc[0]
        if artifact.prediction_column == "expected_return_bid" and isinstance(raw_prediction, pd.DataFrame):
            for column in (
                "expected_return_bid",
                "expected_return_ev",
                "expected_return_fill_probability",
                "expected_return_eligible",
            ):
                value = raw_prediction[column].iloc[0]
                context[f"price_estimator_{column}"] = value.item() if isinstance(value, np.generic) else value
        return context

    def _thresholds_for_signal(self, signal_t0: pd.Timestamp | None) -> tuple[float, float, dict[str, str | None]]:
        policy = self.baseline.threshold_policy or {}
        if policy.get("type") != "utc_day_session_coordinate" or signal_t0 is None:
            return self.t_up, self.t_down, {"threshold_policy": policy.get("type"), "threshold_regime": None}
        timestamp = pd.Timestamp(signal_t0).tz_convert("UTC")
        hour = int(timestamp.hour)
        if hour <= 7:
            session = "asia"
        elif hour <= 15:
            session = "europe"
        else:
            session = "us"
        regime = f"d{int(timestamp.dayofweek)}_{session}"
        payload = (policy.get("thresholds") or {}).get(regime)
        if isinstance(payload, dict) and payload.get("t_up") is not None and payload.get("t_down") is not None:
            return (
                float(payload["t_up"]),
                float(payload["t_down"]),
                {"threshold_policy": str(policy.get("type")), "threshold_regime": regime},
            )
        return (
            float(policy.get("fallback_t_up", self.t_up)),
            float(policy.get("fallback_t_down", self.t_down)),
            {"threshold_policy": str(policy.get("type")), "threshold_regime": regime},
        )

    def build_feature_frame(
        self,
        minute_frame: pd.DataFrame,
        second_frame: pd.DataFrame,
        agg_trades_frame: pd.DataFrame | None = None,
        select_grid_only: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        decision_frame = minute_frame[[DEFAULT_TIMESTAMP_COLUMN]].copy()
        if self.settings.second_level.enabled:
            second_store = build_second_level_feature_store(
                kline_frame=second_frame,
                agg_trades_frame=agg_trades_frame,
                feature_profile=self.settings.second_level.get_profile_payload(),
            )
            sampled_second = sample_second_level_feature_store(decision_frame, second_store)
        else:
            sampled_second = decision_frame.iloc[0:0].copy()
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
        runtime_context: dict | None = None,
    ) -> FeatureBuildResult:
        runtime_context = runtime_context or {}
        feature_offset_minutes = int(runtime_context.get("feature_offset_minutes", 0) or 0)
        feature_frame, sampled_second = self.build_feature_frame(
            minute_frame,
            second_frame,
            agg_trades_frame,
            select_grid_only=not use_latest_available_before_signal and feature_offset_minutes == 0,
        )
        probabilities = predict_frame(
            feature_frame,
            self.model,
            calibrator=self.calibrator,
            feature_columns=self.baseline.feature_columns,
            target_semantics=self.baseline.target_semantics,
        )
        if signal_t0 is None:
            row_index = feature_frame.index[-1]
        else:
            market_t0 = pd.Timestamp(signal_t0).tz_convert("UTC")
            context_feature_timestamp = (runtime_context or {}).get("feature_timestamp")
            if context_feature_timestamp is not None and not use_latest_available_before_signal:
                target_t0 = pd.Timestamp(context_feature_timestamp).tz_convert("UTC")
            else:
                target_t0 = market_t0
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
                    raise RuntimeError(
                        "Feature frame does not include requested feature timestamp "
                        f"'{target_t0.isoformat()}'."
                    )
                row_index = matches[-1]
        latest = feature_frame.loc[row_index]
        p_up = float(probabilities.loc[row_index])
        signal_timestamp = (
            pd.Timestamp(signal_t0).tz_convert("UTC").to_pydatetime()
            if signal_t0 is not None
            else latest[DEFAULT_TIMESTAMP_COLUMN].to_pydatetime()
        )
        resolved_t_up, resolved_t_down, threshold_context = self._thresholds_for_signal(pd.Timestamp(signal_timestamp))
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
                "market_t0": signal_timestamp.isoformat(),
                "decision_time": runtime_context.get("decision_time", latest[DEFAULT_TIMESTAMP_COLUMN].isoformat()),
                "feature_timestamp": latest[DEFAULT_TIMESTAMP_COLUMN].isoformat(),
                "row_policy": (
                    "latest_available_before_signal"
                    if use_latest_available_before_signal
                    else runtime_context.get("row_policy", "exact_signal_t0")
                ),
                "t_up": resolved_t_up,
                "t_down": resolved_t_down,
                "artifact_t_up": self.baseline.t_up,
                "artifact_t_down": self.baseline.t_down,
                "baseline_artifact_dir": str(self.baseline.artifact_dir),
                **threshold_context,
                **runtime_context,
            },
        )
        return FeatureBuildResult(
            feature_frame=feature_frame,
            second_level_frame=sampled_second,
            signal=signal,
            row_index=row_index,
        )

    def _validate_feature_columns(self, feature_frame: pd.DataFrame) -> None:
        missing = [column for column in self.baseline.feature_columns if column not in feature_frame.columns]
        if missing:
            preview = ", ".join(missing[:20])
            raise ValueError(
                f"Runtime feature frame is missing {len(missing)} baseline features: {preview}"
            )


def _extract_single_prediction(prediction: Any, *, prediction_column: str) -> float:
    if isinstance(prediction, pd.DataFrame):
        if prediction_column in prediction.columns:
            return float(prediction[prediction_column].iloc[0])
        if prediction.shape[1] == 1:
            return float(prediction.iloc[0, 0])
        raise ValueError(f"Price estimator prediction does not include column '{prediction_column}'.")
    if isinstance(prediction, pd.Series):
        return float(prediction.iloc[0])
    try:
        return float(prediction[0])
    except (TypeError, KeyError, IndexError):
        return float(prediction)


def _round_price(price: float, decimals: int) -> float:
    quantizer = Decimal("1").scaleb(-int(decimals))
    return float(Decimal(str(price)).quantize(quantizer, rounding=ROUND_HALF_UP))


def _p_side_bucket(p_side: float) -> str:
    if p_side < 0.5:
        return "missing"
    if p_side < 0.55:
        return "0.50_0.55"
    if p_side < 0.60:
        return "0.55_0.60"
    if p_side < 0.65:
        return "0.60_0.65"
    if p_side < 0.70:
        return "0.65_0.70"
    if p_side <= 1.0:
        return "0.70_1.00"
    return "missing"


def _market_time_bucket(timestamp: pd.Timestamp) -> str:
    hour = int(timestamp.hour)
    if hour <= 7:
        return "asia"
    if hour <= 15:
        return "europe"
    return "us"
