from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.core.config import Settings
from src.core.constants import (
    DEFAULT_ABS_RETURN_COLUMN,
    DEFAULT_ASSET_COLUMN,
    DEFAULT_GRID_ID_COLUMN,
    DEFAULT_HORIZON_COLUMN,
    DEFAULT_SIGNED_RETURN_COLUMN,
    DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN,
    DEFAULT_STAGE2_TARGET_COLUMN,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TIMESTAMP_COLUMN,
)
from src.core.validation import normalize_ohlcv_frame
from src.data.preprocess import drop_incomplete_samples, filter_by_timerange
from src.features.builder import build_feature_frame
from src.horizons.registry import get_horizon_spec
from src.labels.abs_return import build_abs_return_frame, compute_stage1_boundary_weight
from src.labels.three_class_direction import build_three_class_direction_target
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
    DEFAULT_ABS_RETURN_COLUMN,
    DEFAULT_SIGNED_RETURN_COLUMN,
    DEFAULT_STAGE2_TARGET_COLUMN,
    DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN,
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
    abs_return_frame = build_abs_return_frame(normalized, horizon)
    training_frame = training_frame.merge(
        abs_return_frame,
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
    tau = float(settings.labels.two_stage.active_return_threshold)
    training_frame[DEFAULT_STAGE2_TARGET_COLUMN] = build_three_class_direction_target(
        training_frame[DEFAULT_SIGNED_RETURN_COLUMN],
        tau=tau,
    )
    boundary_weight = compute_stage1_boundary_weight(training_frame[DEFAULT_ABS_RETURN_COLUMN], tau=tau)
    training_frame[DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN] = boundary_weight

    return TrainingFrame(
        frame=training_frame.reset_index(drop=True),
        feature_columns=feature_columns,
        target_column=DEFAULT_TARGET_COLUMN,
        sample_weight_column=None,
    )
