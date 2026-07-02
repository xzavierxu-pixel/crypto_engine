from __future__ import annotations

import pandas as pd
import pytest

from src.data.polymarket_l2 import (
    PolymarketL2Config,
    assert_feature_schema_safe,
    build_first_minute_features,
    build_trade_products,
    join_l2_features,
    market_t0_from_slug,
)


def _events() -> pd.DataFrame:
    t0 = pd.Timestamp("2026-02-13T17:00:00Z")
    rows = []
    for offset, price, mirror in [
        (0, 0.52, False),
        (60, 0.53, False),
        (60, 0.47, True),
        (61, 0.40, False),
        (62, 0.60, True),
        (300, 0.39, False),
    ]:
        timestamp = t0 + pd.Timedelta(seconds=offset)
        rows.append(
            {
                "market_slug": "btc-updown-5m-1771002000",
                "timestamp": timestamp,
                "local_timestamp": timestamp + pd.Timedelta(milliseconds=5),
                "event_type": "last_trade_price",
                "trade_price": price,
                "trade_is_mirror": mirror,
            }
        )
    return pd.DataFrame(rows)


def test_slug_epoch_and_trade_window_boundaries_are_exact() -> None:
    config = PolymarketL2Config(primary_token_side="UP")
    assert market_t0_from_slug("btc-updown-5m-1771002000") == pd.Timestamp("2026-02-13T17:00:00Z")
    refs, lows = build_trade_products(_events(), config)
    assert refs.loc[0, "up_last_trade_price_1m"] == 0.53
    assert refs.loc[0, "down_last_trade_price_1m"] == pytest.approx(0.53)
    assert lows.loc[0, "up_future_low_4m"] == 0.39
    assert lows.loc[0, "up_future_low_time_4m"] == pd.Timestamp("2026-02-13T17:05:00Z")
    assert lows.loc[0, "down_future_low_4m"] == pytest.approx(0.40)


def test_appending_post_cutoff_events_does_not_change_price_reference() -> None:
    config = PolymarketL2Config(primary_token_side="UP")
    base = _events().iloc[:3].copy()
    refs_before, _ = build_trade_products(base, config)
    refs_after, _ = build_trade_products(_events(), config)
    pd.testing.assert_frame_equal(refs_before, refs_after)


def test_forbidden_columns_and_duplicate_join_keys_fail_closed() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_feature_schema_safe(["pm_l2_1m_spread", "up_future_low_4m"])
    frame = pd.DataFrame({"market_t0": [1], "target": [0]})
    duplicated = pd.DataFrame({"market_t0": [1, 1], "pm_l2_1m_has_trade": [1, 0]})
    with pytest.raises(ValueError, match="duplicate"):
        join_l2_features(frame, duplicated)


def test_side_mapping_must_be_explicit_and_valid() -> None:
    with pytest.raises(ValueError, match="primary_token_side"):
        PolymarketL2Config(primary_token_side="UNKNOWN")


def test_book_replay_and_binary_complement_features() -> None:
    events = _events()
    t0 = events.loc[0, "timestamp"]
    book = {
        "market_slug": events.loc[0, "market_slug"],
        "timestamp": t0 + pd.Timedelta(seconds=5),
        "local_timestamp": t0 + pd.Timedelta(seconds=5, milliseconds=5),
        "event_type": "book",
        "bid_prices": [0.50, 0.49],
        "bid_sizes": [10.0, 20.0],
        "ask_prices": [0.52, 0.53],
        "ask_sizes": [15.0, 25.0],
    }
    change = {
        "market_slug": events.loc[0, "market_slug"],
        "timestamp": t0 + pd.Timedelta(seconds=10),
        "local_timestamp": t0 + pd.Timedelta(seconds=10, milliseconds=5),
        "event_type": "price_change",
        "pc_price": 0.50,
        "pc_size": 30.0,
        "pc_side": "BUY",
    }
    frame = pd.concat([events, pd.DataFrame([book, change])], ignore_index=True)
    features = build_first_minute_features(frame, PolymarketL2Config())
    assert features.loc[0, "pm_l2_1m_up_best_bid"] == 0.50
    assert features.loc[0, "pm_l2_1m_up_best_bid_size"] == 30.0
    assert features.loc[0, "pm_l2_1m_down_best_ask"] == 0.50
    assert features.loc[0, "pm_l2_1m_mid_complement_deviation"] == pytest.approx(0.0)
    assert features.loc[0, "max_feature_event_time"] <= features.loc[0, "feature_cutoff_time"]


def test_feature_hash_is_unchanged_by_future_events() -> None:
    config = PolymarketL2Config()
    base = _events().iloc[:3].copy()
    before = build_first_minute_features(base, config)
    after = build_first_minute_features(_events(), config)
    pd.testing.assert_frame_equal(before, after)
