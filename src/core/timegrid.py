from __future__ import annotations

import pandas as pd

from src.core.constants import DEFAULT_TIMESTAMP_COLUMN


def floor_to_grid(timestamp: pd.Timestamp, grid_minutes: int) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.floor(f"{grid_minutes}min")


def is_grid_timestamp(timestamp: pd.Timestamp, grid_minutes: int) -> bool:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts == floor_to_grid(ts, grid_minutes)


def select_grid_rows(
    df: pd.DataFrame,
    grid_minutes: int,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    mask = df[timestamp_column].map(lambda value: is_grid_timestamp(value, grid_minutes))
    return df.loc[mask].reset_index(drop=True)


def add_grid_columns(
    df: pd.DataFrame,
    grid_minutes: int,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    enriched = df.copy()
    grid_ts = enriched[timestamp_column].map(lambda value: floor_to_grid(value, grid_minutes))
    enriched["grid_t0"] = grid_ts
    enriched["grid_id"] = grid_ts.dt.strftime("%Y%m%d%H%M")
    enriched["is_grid_t0"] = enriched[timestamp_column].map(
        lambda value: is_grid_timestamp(value, grid_minutes)
    )
    return enriched
