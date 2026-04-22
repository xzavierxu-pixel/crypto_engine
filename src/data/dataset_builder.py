from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.config import Settings
from src.core.constants import (
    DEFAULT_ASSET_COLUMN,
    DEFAULT_GRID_ID_COLUMN,
    DEFAULT_HORIZON_COLUMN,
    DEFAULT_SAMPLE_WEIGHT_COLUMN,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TIMESTAMP_COLUMN,
)
from src.core.validation import normalize_ohlcv_frame
from src.data.preprocess import drop_incomplete_samples, filter_by_timerange
from src.features.builder import build_feature_frame
from src.horizons.registry import get_horizon_spec
from src.labels.registry import get_label_builder


BASE_DATASET_COLUMNS = {
    DEFAULT_TIMESTAMP_COLUMN,
    "open",
    "high",
    "low",
    "close",
    "volume",
    DEFAULT_ASSET_COLUMN,
    DEFAULT_HORIZON_COLUMN,
    DEFAULT_GRID_ID_COLUMN,
    "grid_t0",
    "is_grid_t0",
    "feature_version",
    "label_version",
    DEFAULT_TARGET_COLUMN,
    DEFAULT_SAMPLE_WEIGHT_COLUMN,
}


@dataclass(frozen=True)
class TrainingFrame:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str = DEFAULT_TARGET_COLUMN
    sample_weight_column: str | None = None

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.feature_columns]

    @property
    def y(self) -> pd.Series:
        return self.frame[self.target_column]

    @property
    def sample_weight(self) -> pd.Series | None:
        if self.sample_weight_column is None or self.sample_weight_column not in self.frame.columns:
            return None
        return self.frame[self.sample_weight_column]


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in BASE_DATASET_COLUMNS]


def _apply_sample_quality_filter(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    config = settings.dataset.sample_quality_filter
    if not config or not config.get("enabled", False):
        return frame

    filtered = frame.copy()
    mask = pd.Series(True, index=filtered.index)
    min_nz_volume_share_20 = config.get("min_nz_volume_share_20")
    max_flat_share_20 = config.get("max_flat_share_20")

    if min_nz_volume_share_20 is not None and "nz_volume_share_20" in filtered.columns:
        mask &= filtered["nz_volume_share_20"] >= float(min_nz_volume_share_20)
    if max_flat_share_20 is not None and "flat_share_20" in filtered.columns:
        mask &= filtered["flat_share_20"] <= float(max_flat_share_20)

    return filtered.loc[mask].reset_index(drop=True)


def _build_sample_weights(frame: pd.DataFrame, settings: Settings) -> pd.Series | None:
    config = settings.dataset.sample_weighting
    if not config or not config.get("enabled", False):
        return None

    weights = pd.Series(float(config.get("base_weight", 1.0)), index=frame.index, dtype="float64")

    if "nz_volume_share_20" in frame.columns:
        weights += float(config.get("nz_volume_share_20_weight", 0.0)) * frame["nz_volume_share_20"].clip(0.0, 1.0)
    if "flat_share_20" in frame.columns:
        weights -= float(config.get("flat_share_20_weight", 0.0)) * frame["flat_share_20"].clip(0.0, 1.0)
    if "abs_ret_mean_20" in frame.columns:
        abs_ret_scale = float(config.get("abs_ret_mean_20_scale", 0.001))
        weights += float(config.get("abs_ret_mean_20_weight", 0.0)) * (
            frame["abs_ret_mean_20"] / max(abs_ret_scale, 1e-12)
        ).clip(0.0, 1.0)
    if "dollar_vol_mean_20" in frame.columns:
        dollar_scale = float(config.get("dollar_vol_mean_20_scale", 1.0))
        weights += float(config.get("dollar_vol_mean_20_weight", 0.0)) * (
            np.log1p(frame["dollar_vol_mean_20"].clip(lower=0.0))
            / np.log1p(max(dollar_scale, 1e-12))
        ).clip(0.0, 1.0)

    min_weight = float(config.get("min_weight", 0.5))
    max_weight = float(config.get("max_weight", 3.0))
    return weights.clip(lower=min_weight, upper=max_weight)


def build_training_frame(
    raw_df: pd.DataFrame,
    settings: Settings,
    horizon_name: str | None = None,
    derivatives_frame: pd.DataFrame | None = None,
) -> TrainingFrame:
    horizon = get_horizon_spec(settings, horizon_name)
    normalized = normalize_ohlcv_frame(raw_df, timestamp_column=DEFAULT_TIMESTAMP_COLUMN, require_volume=False)

    feature_frame = build_feature_frame(
        normalized,
        settings,
        horizon_name=horizon.name,
        select_grid_only=True,
        derivatives_frame=derivatives_frame,
    )
    label_builder = get_label_builder(horizon.label_builder)
    label_frame = label_builder.build(normalized, settings, horizon, select_grid_only=True)

    training_frame = feature_frame.merge(
        label_frame[[DEFAULT_TIMESTAMP_COLUMN, DEFAULT_TARGET_COLUMN, "label_version"]],
        on=DEFAULT_TIMESTAMP_COLUMN,
        how="left",
        validate="one_to_one",
    )
    training_frame = filter_by_timerange(
        training_frame,
        start=settings.dataset.train_start,
        end=settings.dataset.train_end,
        timestamp_column=DEFAULT_TIMESTAMP_COLUMN,
    )

    feature_columns = infer_feature_columns(training_frame)
    if settings.dataset.drop_incomplete_candles:
        training_frame = drop_incomplete_samples(
            training_frame,
            feature_columns=feature_columns,
            target_column=DEFAULT_TARGET_COLUMN,
        )
    training_frame = _apply_sample_quality_filter(training_frame, settings)
    sample_weights = _build_sample_weights(training_frame, settings)
    sample_weight_column = None
    if sample_weights is not None:
        training_frame[DEFAULT_SAMPLE_WEIGHT_COLUMN] = sample_weights
        sample_weight_column = DEFAULT_SAMPLE_WEIGHT_COLUMN

    return TrainingFrame(
        frame=training_frame.reset_index(drop=True),
        feature_columns=feature_columns,
        target_column=DEFAULT_TARGET_COLUMN,
        sample_weight_column=sample_weight_column,
    )
