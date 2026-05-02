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
from src.labels.abs_return import build_abs_return_frame
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

RAW_METADATA_FEATURE_COLUMNS = {
    "raw_timestamp",
    "date",
    "open_time",
    "close_time",
    "quote_volume",
    "quote_asset_volume",
    "taker_buy_volume",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_volume",
    "taker_buy_quote_asset_volume",
    "count",
    "number_of_trades",
    "trade_count",
    "taker_buy_base_volume",
    "ignore",
    "symbol",
    "market_family",
    "data_type",
    "interval",
    "source_file",
    "source_date",
    "source_granularity",
    "source_version",
    "checksum_status",
    "expected_checksum",
    "actual_checksum",
    "ingested_at",
    "download_status",
}

RAW_METADATA_PREFIXES = (
    "raw_",
    "source_",
    "checksum_",
)


def is_allowed_feature_column(column: str) -> bool:
    if column in BASE_DATASET_COLUMNS or column in RAW_METADATA_FEATURE_COLUMNS:
        return False
    for suffix in ("_x", "_y"):
        if column.endswith(suffix) and column[: -len(suffix)] in RAW_METADATA_FEATURE_COLUMNS:
            return False
    if any(column.startswith(prefix) for prefix in RAW_METADATA_PREFIXES):
        return False
    return True


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
    return [column for column in df.columns if is_allowed_feature_column(column)]


def assert_feature_schema(feature_columns: list[str]) -> None:
    forbidden = [column for column in feature_columns if not is_allowed_feature_column(column)]
    if forbidden:
        raise ValueError(f"Forbidden raw metadata columns selected as features: {forbidden}")


def build_training_frame(
    raw_df: pd.DataFrame,
    settings: Settings,
    horizon_name: str | None = None,
    derivatives_frame: pd.DataFrame | None = None,
    second_level_features_frame: pd.DataFrame | None = None,
) -> TrainingFrame:
    horizon = get_horizon_spec(settings, horizon_name)
    normalized = normalize_ohlcv_frame(raw_df, timestamp_column=DEFAULT_TIMESTAMP_COLUMN, require_volume=False)

    feature_frame = build_feature_frame(
        normalized,
        settings,
        horizon_name=horizon.name,
        select_grid_only=True,
        derivatives_frame=derivatives_frame,
        second_level_features_frame=second_level_features_frame,
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
    assert_feature_schema(feature_columns)
    if settings.dataset.drop_incomplete_candles:
        training_frame = drop_incomplete_samples(
            training_frame,
            feature_columns=feature_columns,
            target_column=DEFAULT_TARGET_COLUMN,
        )
    training_frame[DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN] = compute_sample_weight(
        training_frame[DEFAULT_ABS_RETURN_COLUMN],
        settings=settings,
    )

    return TrainingFrame(
        frame=training_frame.reset_index(drop=True),
        feature_columns=feature_columns,
        target_column=DEFAULT_TARGET_COLUMN,
        sample_weight_column=DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN if settings.sample_weighting.enabled else None,
    )


def compute_sample_weight(abs_return: pd.Series, settings: Settings) -> pd.Series:
    config = settings.sample_weighting
    weights = pd.Series(float(config.max_weight), index=abs_return.index, dtype="float64")
    if not config.enabled:
        return weights
    if config.mode != "linear_ramp":
        raise ValueError(f"Unsupported sample weighting mode: {config.mode}")

    values = abs_return.astype("float64")
    ramp_denominator = float(config.full_weight_abs_return)
    if ramp_denominator <= 0:
        raise ValueError("sample_weighting.full_weight_abs_return must be > 0.")

    ramp = float(config.min_weight) + (float(config.max_weight) - float(config.min_weight)) * values / ramp_denominator
    weights = ramp.clip(lower=float(config.min_weight), upper=float(config.max_weight))
    weights = weights.mask(values < float(config.min_abs_return), float(config.min_weight))
    return weights.fillna(float(config.min_weight)).astype("float64")
