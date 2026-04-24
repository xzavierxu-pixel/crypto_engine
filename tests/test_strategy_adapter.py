from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.strategies.BTCGridFreqAIStrategy import BTCGridFreqAIStrategy


def test_strategy_builds_freqai_columns_and_sets_grid_target() -> None:
    strategy = BTCGridFreqAIStrategy(config={"candle_type_def": "spot"})
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01T12:00:00Z", periods=30, freq="1min"),
            "open": [100 + index for index in range(30)],
            "high": [101 + index for index in range(30)],
            "low": [99 + index for index in range(30)],
            "close": [100 + index for index in range(30)],
            "volume": [10 + index for index in range(30)],
        }
    )

    indicators = strategy.feature_engineering_standard(frame.copy(), metadata={"pair": "BTC/USDT"})
    assert len(indicators) == len(frame)
    assert "%-ret_1" in indicators.columns
    assert "%-relative_volume_5" in indicators.columns
    assert "%-low_volume_flag_share_20" in indicators.columns
    assert indicators["is_grid_t0"].sum() == 6

    labeled = strategy.set_freqai_targets(frame.copy(), metadata={"pair": "BTC/USDT"})
    assert len(labeled) == len(frame)
    assert "&s-up_or_down" in labeled.columns
    assert "grid_dir_target" in labeled.columns
    assert labeled.loc[labeled["date"] == pd.Timestamp("2024-01-01T12:01:00Z"), "grid_dir_target"].isna().all()
    assert labeled.loc[labeled["date"] == pd.Timestamp("2024-01-01T12:05:00Z"), "&s-up_or_down"].iloc[0] == "up"


def test_strategy_custom_exit_matches_5m_horizon() -> None:
    strategy = BTCGridFreqAIStrategy(config={"candle_type_def": "spot"})
    trade = SimpleNamespace(open_date_utc=pd.Timestamp("2024-01-01T12:00:00Z").to_pydatetime())

    assert strategy.custom_exit("BTC/USDT", trade, pd.Timestamp("2024-01-01T12:04:00Z").to_pydatetime(), 0, 0) is None
    assert strategy.custom_exit("BTC/USDT", trade, pd.Timestamp("2024-01-01T12:05:00Z").to_pydatetime(), 0, 0) == "horizon_timeout"


def test_strategy_populate_indicators_accepts_timestamp_alias() -> None:
    strategy = BTCGridFreqAIStrategy(config={"candle_type_def": "spot"})
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=20, freq="1min"),
            "open": [100 + index for index in range(20)],
            "high": [101 + index for index in range(20)],
            "low": [99 + index for index in range(20)],
            "close": [100 + index for index in range(20)],
            "volume": [10 + index for index in range(20)],
        }
    )

    populated = strategy.populate_indicators(frame.copy(), metadata={"pair": "BTC/USDT"})

    assert "&s-up_or_down" in populated.columns
    assert "date" in populated.columns


def test_strategy_entry_uses_probability_and_activity_filters() -> None:
    strategy = BTCGridFreqAIStrategy(
        config={
            "candle_type_def": "spot",
            "freqai_signal": {
                "entry_probability_threshold": 0.58,
                "entry_probability_margin": 0.16,
                "exit_probability_threshold": 0.55,
            },
        }
    )
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01T12:00:00Z", periods=3, freq="1min"),
            "do_predict": [1, 1, 1],
            "is_grid_t0": [True, True, True],
            "up": [0.62, 0.60, 0.70],
            "down": [0.35, 0.47, 0.10],
            "&s-up_or_down": ["up", "up", "up"],
        }
    )

    entries = strategy.populate_entry_trend(frame.copy(), metadata={"pair": "BTC/USDT"})
    exits = strategy.populate_exit_trend(frame.copy(), metadata={"pair": "BTC/USDT"})

    assert entries["enter_long"].tolist() == [1, 0, 1]
    assert entries.loc[entries["enter_long"] == 1, "enter_tag"].iloc[0] == "freqai_up_5m_prob"
    assert exits["exit_long"].tolist() == [0, 0, 0]


def test_strategy_custom_exit_matches_15m_horizon() -> None:
    strategy = BTCGridFreqAIStrategy(config={"candle_type_def": "spot"}, horizon_name="15m")
    trade = SimpleNamespace(open_date_utc=pd.Timestamp("2024-01-01T12:00:00Z").to_pydatetime())

    assert strategy.custom_exit("BTC/USDT", trade, pd.Timestamp("2024-01-01T12:14:00Z").to_pydatetime(), 0, 0) is None
    assert strategy.custom_exit("BTC/USDT", trade, pd.Timestamp("2024-01-01T12:15:00Z").to_pydatetime(), 0, 0) == "horizon_timeout"


def test_strategy_entry_tag_uses_selected_horizon_name() -> None:
    strategy = BTCGridFreqAIStrategy(
        config={
            "candle_type_def": "spot",
            "freqai_signal": {
                "entry_probability_threshold": 0.58,
                "entry_probability_margin": 0.16,
                "exit_probability_threshold": 0.55,
            },
        },
        horizon_name="15m",
    )
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01T12:00:00Z", periods=1, freq="1min"),
            "do_predict": [1],
            "is_grid_t0": [True],
            "up": [0.62],
            "down": [0.35],
            "&s-up_or_down": ["up"],
        }
    )

    entries = strategy.populate_entry_trend(frame.copy(), metadata={"pair": "BTC/USDT"})

    assert entries.loc[0, "enter_tag"] == "freqai_up_15m_prob"
