from __future__ import annotations

import math

import pandas as pd

from src.core.config import load_settings
from src.features.builder import build_feature_frame


def test_lagged_feature_pack_builds_selected_lags() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=240, freq="1min"),
            "open": [100 + index for index in range(240)],
            "high": [101 + index for index in range(240)],
            "low": [99 + index for index in range(240)],
            "close": [100 + index for index in range(240)],
            "volume": [50 + index for index in range(240)],
        }
    )

    features = build_feature_frame(frame, settings, horizon_name="5m", select_grid_only=False)
    row = features.loc[features["timestamp"] == pd.Timestamp("2024-01-01T12:06:00Z")].iloc[0]

    assert "ret_1_lag1" in features.columns
    assert "rv_5_lag3" in features.columns
    expected_ret_1_at_1205 = (104 / 103) - 1
    assert math.isclose(row["ret_1_lag1"], expected_ret_1_at_1205, rel_tol=1e-9)
