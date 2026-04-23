from __future__ import annotations

import pandas as pd

from src.core.config import load_settings
from src.horizons.registry import get_horizon_spec
from src.labels.grid_direction import GridDirectionLabelBuilder


def test_grid_direction_label_uses_grid_open_and_future_close() -> None:
    settings = load_settings()
    horizon = get_horizon_spec(settings, "5m")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=10, freq="1min"),
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [100, 103, 101, 102, 106, 105, 104, 108, 107, 111],
        }
    )

    labeled = GridDirectionLabelBuilder().build(frame, settings, horizon)

    assert list(labeled["timestamp"]) == [
        pd.Timestamp("2024-01-01T12:00:00Z"),
        pd.Timestamp("2024-01-01T12:05:00Z"),
    ]
    assert labeled.loc[0, "target"] == 1.0
    assert labeled.loc[1, "target"] == 1.0
    assert labeled.loc[0, "label_version"] == "v2"


def test_grid_direction_label_applies_threshold_multiplier_for_5m_v2() -> None:
    settings = load_settings()
    horizon = get_horizon_spec(settings, "5m")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=10, freq="1min"),
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0, 100.05, 100.06, 100.07, 100.08, 100.09, 100.0, 100.0, 100.0, 100.0],
        }
    )

    labeled = GridDirectionLabelBuilder().build(frame, settings, horizon)

    assert labeled.loc[0, "target"] == 0.0
    assert labeled.loc[1, "target"] == 0.0
    assert labeled.loc[0, "label_version"] == "v2"
