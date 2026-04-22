from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.constants import DEFAULT_FUNDING_EFFECTIVE_TIME_COLUMN, DEFAULT_TIMESTAMP_COLUMN
from src.core.validation import ensure_columns


def normalize_funding_frame(
    frame: pd.DataFrame,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    resolved_timestamp_column = timestamp_column
    if resolved_timestamp_column not in frame.columns and "date" in frame.columns:
        resolved_timestamp_column = "date"

    ensure_columns(frame, [resolved_timestamp_column, "funding_rate"])

    normalized = frame.copy()
    normalized[resolved_timestamp_column] = pd.to_datetime(normalized[resolved_timestamp_column], utc=True)
    if DEFAULT_FUNDING_EFFECTIVE_TIME_COLUMN in normalized.columns:
        normalized[DEFAULT_FUNDING_EFFECTIVE_TIME_COLUMN] = pd.to_datetime(
            normalized[DEFAULT_FUNDING_EFFECTIVE_TIME_COLUMN],
            utc=True,
        )
    else:
        normalized[DEFAULT_FUNDING_EFFECTIVE_TIME_COLUMN] = normalized[resolved_timestamp_column]

    if resolved_timestamp_column != DEFAULT_TIMESTAMP_COLUMN:
        normalized = normalized.rename(columns={resolved_timestamp_column: DEFAULT_TIMESTAMP_COLUMN})

    normalized = normalized.sort_values(DEFAULT_TIMESTAMP_COLUMN).drop_duplicates(
        subset=[DEFAULT_TIMESTAMP_COLUMN],
        keep="last",
    )

    for column in ("exchange", "symbol", "source_version"):
        if column not in normalized.columns:
            normalized[column] = pd.NA

    return normalized.reset_index(drop=True)


def load_funding_frame(
    path: str | Path,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    resolved_path = Path(path)
    suffix = resolved_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(resolved_path)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(resolved_path)
    elif suffix in {".feather", ".ft"}:
        frame = pd.read_feather(resolved_path)
    else:
        raise ValueError(f"Unsupported funding input format: {resolved_path.suffix}")
    return normalize_funding_frame(frame, timestamp_column=timestamp_column)
