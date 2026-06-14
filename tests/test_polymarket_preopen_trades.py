from __future__ import annotations

import pandas as pd

from src.data.polymarket_trades import build_preopen_trade_feature_frame


def test_preopen_trade_features_use_strict_pre_market_window() -> None:
    trades = pd.DataFrame(
        {
            "price": [0.45, 0.55, 0.80, 0.10, 0.90],
            "timestamp": [
                1_700_000_000 - 61,
                1_700_000_000 - 60,
                1_700_000_000 - 1,
                1_700_000_000,
                1_700_000_000 + 1,
            ],
            "market_start_ts": [1_700_000_000] * 5,
            "outcome": ["up", "up", "down", "up", "down"],
        }
    )

    features = build_preopen_trade_feature_frame(trades, preopen_window_seconds=60)

    assert len(features) == 1
    row = features.iloc[0]
    assert row["timestamp"] == pd.Timestamp(1_700_000_000, unit="s", tz="UTC")
    assert row["pm_preopen_trade_count"] == 2.0
    assert row["pm_preopen_up_trade_count"] == 1.0
    assert row["pm_preopen_down_trade_count"] == 1.0
    assert row["pm_preopen_up_price_last"] == 0.55
    assert row["pm_preopen_down_price_last"] == 0.80
    assert row["pm_preopen_last_price_gap"] == -0.25
