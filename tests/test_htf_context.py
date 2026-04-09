from __future__ import annotations

import math

import pandas as pd

from src.core.config import load_settings
from src.features.builder import build_feature_frame


def test_htf_context_uses_last_completed_higher_timeframe_candle() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=240, freq="1min"),
            "open": [100 + index for index in range(240)],
            "high": [101 + index for index in range(240)],
            "low": [99 + index for index in range(240)],
            "close": [100 + index for index in range(240)],
            "volume": [1000 + index for index in range(240)],
        }
    )

    features = build_feature_frame(frame, settings, horizon_name="5m", select_grid_only=False)
    row = features.loc[features["timestamp"] == pd.Timestamp("2024-01-01T12:15:00Z")].iloc[0]

    expected_htf_ret_5m = (114 / 109) - 1
    current_incomplete_5m_ret = (119 / 114) - 1

    assert math.isclose(row["htf_ret_5m_1"], expected_htf_ret_5m, rel_tol=1e-9)
    assert not math.isclose(row["htf_ret_5m_1"], current_incomplete_5m_ret, rel_tol=1e-9)
    assert "htf_regime_trend_strength_15m" in features.columns
