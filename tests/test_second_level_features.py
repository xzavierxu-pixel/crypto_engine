from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.data.second_level_features import build_second_level_feature_frame


def test_second_level_trade_features_use_only_events_at_or_before_decision() -> None:
    decisions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01T00:01:00Z", "2024-01-01T00:02:00Z"],
                utc=True,
            )
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:55Z",
                    "2024-01-01T00:01:00Z",
                    "2024-01-01T00:01:01Z",
                    "2024-01-01T00:01:55Z",
                    "2024-01-01T00:02:00Z",
                ],
                utc=True,
            ),
            "price": [100.0, 101.0, 120.0, 102.0, 103.0],
            "quantity": [1.0, 2.0, 100.0, 1.0, 1.0],
            "quote_quantity": [100.0, 202.0, 12000.0, 102.0, 103.0],
            "is_buyer_maker": [False, False, False, True, True],
        }
    )

    features = build_second_level_feature_frame(decisions, trades_frame=trades)

    assert features.loc[0, "sl_trade_count_5s"] == 1
    assert features.loc[0, "sl_taker_buy_volume_5s"] == 2.0
    assert features.loc[0, "sl_dollar_volume_5s"] == 202.0
    assert round(float(features.loc[0, "sl_return_5s"]), 6) == 0.01
    assert features.loc[1, "sl_trade_count_5s"] == 1
    assert features.loc[1, "sl_taker_sell_volume_5s"] == 1.0


def test_second_level_book_features_align_to_decisions() -> None:
    decisions = pd.DataFrame({"timestamp": pd.date_range("2024-01-01T00:01:00Z", periods=2, freq="1min")})
    book = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:58Z",
                    "2024-01-01T00:01:00Z",
                    "2024-01-01T00:01:30Z",
                    "2024-01-01T00:02:00Z",
                ],
                utc=True,
            ),
            "bid_price": [99.0, 100.0, 101.0, 101.5],
            "bid_qty": [2.0, 3.0, 4.0, 5.0],
            "ask_price": [101.0, 102.0, 103.0, 103.5],
            "ask_qty": [1.0, 1.0, 2.0, 3.0],
        }
    )

    features = build_second_level_feature_frame(decisions, book_frame=book)

    assert features.loc[0, "sl_best_bid"] == 100.0
    assert features.loc[0, "sl_bid_ask_qty_imbalance"] == 0.5
    assert features.loc[1, "sl_best_ask"] == 103.5
    assert "sl_microprice_premium" in features.columns


def test_training_frame_accepts_second_level_features_without_raw_columns() -> None:
    rows = 900
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="1min"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": [100.0 + index * 0.01 for index in range(rows)],
            "volume": 10.0,
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T00:00:00Z", periods=rows * 2, freq="30s"),
            "price": 100.0,
            "quantity": 1.0,
            "quote_quantity": 100.0,
            "is_buyer_maker": [False, True] * rows,
        }
    )
    settings = load_settings()
    settings = replace(settings, derivatives=replace(settings.derivatives, enabled=False))
    second_level = build_second_level_feature_frame(source, trades_frame=trades)
    training = build_training_frame(source, settings, horizon_name="5m", second_level_features_frame=second_level)

    assert any(column.startswith("sl_") for column in training.feature_columns)
    assert "price" not in training.feature_columns
    assert "quantity" not in training.feature_columns
