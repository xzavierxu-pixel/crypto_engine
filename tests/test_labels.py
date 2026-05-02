from __future__ import annotations

import pandas as pd

from src.core.config import load_settings
from src.core.constants import DEFAULT_ABS_RETURN_COLUMN, DEFAULT_SIGNED_RETURN_COLUMN
from src.horizons.registry import get_horizon_spec
from src.labels.abs_return import build_abs_return_frame
from src.labels.grid_direction import GridDirectionLabelBuilder


def test_grid_direction_label_uses_t0_open_and_t4_close_for_5m_market() -> None:
    settings = load_settings()
    horizon = get_horizon_spec(settings, "5m")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=11, freq="1min"),
            "open": [100.0] * 11,
            "high": [101.0] * 11,
            "low": [99.0] * 11,
            "close": [100.0, 99.0, 99.0, 99.0, 101.0, 99.0, 98.0, 98.0, 98.0, 100.0, 200.0],
        }
    )

    labeled = GridDirectionLabelBuilder().build(frame, settings, horizon)

    assert list(labeled["timestamp"]) == [
        pd.Timestamp("2024-01-01T12:00:00Z"),
        pd.Timestamp("2024-01-01T12:05:00Z"),
        pd.Timestamp("2024-01-01T12:10:00Z"),
    ]
    assert horizon.future_close_offset == 4
    assert labeled.loc[0, "target"] == 1.0
    assert labeled.loc[1, "target"] == 1.0
    assert pd.isna(labeled.loc[2, "target"])
    assert labeled.loc[0, "label_version"] == "settlement_direction_t0_open_to_t4_close_tie_up_v2"


def test_abs_return_uses_same_t4_close_as_5m_direction_label() -> None:
    settings = load_settings()
    horizon = get_horizon_spec(settings, "5m")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=11, freq="1min"),
            "open": [100.0] * 11,
            "high": [101.0] * 11,
            "low": [99.0] * 11,
            "close": [100.0, 99.0, 99.0, 99.0, 101.0, 99.0, 98.0, 98.0, 98.0, 100.0, 200.0],
        }
    )

    returns = build_abs_return_frame(frame, horizon)

    assert returns.loc[0, DEFAULT_SIGNED_RETURN_COLUMN] == 0.01
    assert returns.loc[0, DEFAULT_ABS_RETURN_COLUMN] == 0.01
    assert returns.loc[5, DEFAULT_SIGNED_RETURN_COLUMN] == 0.0
    assert pd.isna(returns.loc[10, DEFAULT_SIGNED_RETURN_COLUMN])


def test_grid_direction_label_supports_15m_horizon() -> None:
    settings = load_settings()
    horizon = get_horizon_spec(settings, "15m")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=32, freq="1min"),
            "open": [100 + index for index in range(32)],
            "high": [101 + index for index in range(32)],
            "low": [99 + index for index in range(32)],
            "close": [100 + index for index in range(32)],
        }
    )

    labeled = GridDirectionLabelBuilder().build(frame, settings, horizon)

    assert list(labeled["timestamp"]) == [
        pd.Timestamp("2024-01-01T12:00:00Z"),
        pd.Timestamp("2024-01-01T12:15:00Z"),
        pd.Timestamp("2024-01-01T12:30:00Z"),
    ]
    assert labeled.loc[0, "target"] == 1.0
    assert labeled.loc[1, "target"] == 1.0
    assert pd.isna(labeled.loc[2, "target"])
    assert (labeled["horizon"] == "15m").all()


def test_grid_direction_label_uses_pure_direction_without_threshold_multiplier() -> None:
    settings = load_settings()
    horizon = get_horizon_spec(settings, "5m")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=10, freq="1min"),
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0, 100.05, 100.06, 100.07, 100.08, 100.09, 100.0, 100.0, 100.0, 100.0],
        }
    )

    labeled = GridDirectionLabelBuilder().build(frame, settings, horizon)

    assert labeled.loc[0, "target"] == 1.0
    assert labeled.loc[1, "target"] == 1.0
    assert labeled.loc[0, "label_version"] == "settlement_direction_t0_open_to_t4_close_tie_up_v2"
