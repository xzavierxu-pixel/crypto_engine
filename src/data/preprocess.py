from __future__ import annotations

import pandas as pd

from src.core.config import Settings
from src.core.constants import DEFAULT_TARGET_COLUMN, DEFAULT_TIMESTAMP_COLUMN
from src.core.validation import normalize_ohlcv_frame


def filter_by_timerange(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    filtered = df.copy()
    if start is not None:
        start_ts = pd.Timestamp(start, tz="UTC")
        filtered = filtered.loc[filtered[timestamp_column] >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end, tz="UTC")
        filtered = filtered.loc[filtered[timestamp_column] <= end_ts]
    return filtered.reset_index(drop=True)


def sanitize_ohlcv_for_training(
    df: pd.DataFrame,
    settings: Settings,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    normalized = normalize_ohlcv_frame(df, timestamp_column=timestamp_column, require_volume=False)
    return filter_by_timerange(
        normalized,
        start=settings.dataset.train_start,
        end=settings.dataset.train_end,
        timestamp_column=timestamp_column,
    )


def drop_incomplete_samples(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    required_columns = [*feature_columns, target_column]
    return df.dropna(subset=required_columns).reset_index(drop=True)
