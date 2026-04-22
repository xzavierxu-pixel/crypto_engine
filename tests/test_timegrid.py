from __future__ import annotations

import pandas as pd

from src.core.timegrid import add_grid_columns, floor_to_grid, is_grid_timestamp, select_grid_rows


def test_floor_to_grid_and_membership() -> None:
    ts = pd.Timestamp("2024-01-01T12:03:27Z")
    assert floor_to_grid(ts, 5) == pd.Timestamp("2024-01-01T12:00:00Z")
    assert is_grid_timestamp(pd.Timestamp("2024-01-01T12:05:00Z"), 5)
    assert not is_grid_timestamp(ts, 5)


def test_select_grid_rows_keeps_only_5m_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=8, freq="1min"),
            "open": range(8),
            "high": range(8),
            "low": range(8),
            "close": range(8),
        }
    )
    grid = select_grid_rows(frame, grid_minutes=5)
    assert list(grid["timestamp"]) == [
        pd.Timestamp("2024-01-01T12:00:00Z"),
        pd.Timestamp("2024-01-01T12:05:00Z"),
    ]


def test_select_grid_rows_keeps_only_15m_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=31, freq="1min"),
            "open": range(31),
            "high": range(31),
            "low": range(31),
            "close": range(31),
        }
    )
    grid = select_grid_rows(frame, grid_minutes=15)
    assert list(grid["timestamp"]) == [
        pd.Timestamp("2024-01-01T12:00:00Z"),
        pd.Timestamp("2024-01-01T12:15:00Z"),
        pd.Timestamp("2024-01-01T12:30:00Z"),
    ]


def test_add_grid_columns_populates_grid_metadata() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01T12:00:00Z", "2024-01-01T12:03:00Z"],
                utc=True,
            ),
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
        }
    )
    enriched = add_grid_columns(frame, grid_minutes=5)
    assert enriched.loc[0, "grid_id"] == "202401011200"
    assert enriched.loc[1, "grid_t0"] == pd.Timestamp("2024-01-01T12:00:00Z")
    assert enriched.loc[1, "is_grid_t0"] == False
