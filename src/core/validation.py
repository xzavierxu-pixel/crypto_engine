from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.core.constants import DEFAULT_TIMESTAMP_COLUMN


REQUIRED_OHLC_COLUMNS = ("open", "high", "low", "close")


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_ohlcv_frame(
    df: pd.DataFrame,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
    require_volume: bool = False,
) -> pd.DataFrame:
    required_columns = [timestamp_column, *REQUIRED_OHLC_COLUMNS]
    if require_volume:
        required_columns.append("volume")

    ensure_columns(df, required_columns)

    normalized = df.copy()
    normalized[timestamp_column] = pd.to_datetime(normalized[timestamp_column], utc=True)
    normalized = normalized.sort_values(timestamp_column).drop_duplicates(
        subset=[timestamp_column],
        keep="last",
    )

    if not normalized[timestamp_column].is_monotonic_increasing:
        raise ValueError("Timestamp column must be monotonic increasing after sorting.")

    return normalized.reset_index(drop=True)
