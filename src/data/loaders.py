from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.constants import DEFAULT_TIMESTAMP_COLUMN
from src.core.validation import normalize_ohlcv_frame


def _normalize_loaded_frame(
    frame: pd.DataFrame,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    resolved_timestamp_column = timestamp_column
    if resolved_timestamp_column not in frame.columns and "date" in frame.columns:
        resolved_timestamp_column = "date"
    normalized = normalize_ohlcv_frame(
        frame,
        timestamp_column=resolved_timestamp_column,
        require_volume=False,
    )
    if resolved_timestamp_column != DEFAULT_TIMESTAMP_COLUMN:
        normalized = normalized.rename(columns={resolved_timestamp_column: DEFAULT_TIMESTAMP_COLUMN})
    return normalized


def load_ohlcv_csv(
    path: str | Path,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return _normalize_loaded_frame(frame, timestamp_column=timestamp_column)


def load_ohlcv_parquet(
    path: str | Path,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return _normalize_loaded_frame(frame, timestamp_column=timestamp_column)


def load_ohlcv_feather(
    path: str | Path,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    frame = pd.read_feather(path)
    return _normalize_loaded_frame(frame, timestamp_column=timestamp_column)
