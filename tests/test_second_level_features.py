from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.data.second_level_features import (
    build_second_level_feature_frame,
    build_second_level_feature_store,
    build_second_level_source_tables,
    load_second_level_frame,
    load_sampled_second_level_features,
    sample_second_level_feature_store,
    write_partitioned_second_level_feature_store,
)
from src.data.second_level_feature_packs import (
    SecondLevelFeatureProfile,
    get_second_level_feature_pack,
)


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


def test_second_level_feature_store_uses_kline_backbone_and_samples_backward() -> None:
    kline = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01T00:00:00Z", periods=6, freq="1s"),
            "open": [100, 100, 101, 102, 103, 104],
            "high": [100, 101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
            "volume": [10, 10, 10, 10, 10, 10],
            "quote_volume": [1000, 1010, 1020, 1030, 1040, 1050],
            "trade_count": [2, 2, 2, 2, 2, 2],
            "taker_buy_base_volume": [6, 7, 8, 9, 10, 10],
            "taker_buy_quote_volume": [600, 707, 816, 927, 1040, 1050],
        }
    )
    book = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T00:00:00Z", periods=6, freq="1s"),
            "bid_price": [99, 100, 101, 102, 103, 104],
            "bid_qty": [5, 6, 7, 5, 4, 8],
            "ask_price": [101, 102, 103, 104, 105, 106],
            "ask_qty": [5, 4, 3, 4, 6, 2],
        }
    )
    agg = pd.DataFrame(
        {
            "T": pd.to_datetime(
                ["2024-01-01T00:00:01.100Z", "2024-01-01T00:00:01.150Z", "2024-01-01T00:00:05.000Z"],
                utc=True,
            ),
            "p": [101, 101, 105],
            "q": [0.1, 5.0, 0.2],
            "m": [False, False, True],
        }
    )
    decisions = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01T00:00:04.500Z"], utc=True)})
    depth = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T00:00:00Z", periods=6, freq="1s"),
            "bid_qty_1": [2, 2, 3, 3, 4, 4],
            "ask_qty_1": [1, 2, 2, 3, 3, 4],
            "bid_price_1": [99, 100, 101, 102, 103, 104],
            "ask_price_1": [101, 102, 103, 104, 105, 106],
            "bid_qty_2": [1, 1, 1, 1, 1, 1],
            "ask_qty_2": [1, 1, 1, 1, 1, 1],
            "bid_price_2": [98, 99, 100, 101, 102, 103],
            "ask_price_2": [102, 103, 104, 105, 106, 107],
            "bid_qty_3": [1, 1, 1, 1, 1, 1],
            "ask_qty_3": [1, 1, 1, 1, 1, 1],
            "bid_price_3": [97, 98, 99, 100, 101, 102],
            "ask_price_3": [103, 104, 105, 106, 107, 108],
            "bid_qty_4": [1, 1, 1, 1, 1, 1],
            "ask_qty_4": [1, 1, 1, 1, 1, 1],
            "bid_price_4": [96, 97, 98, 99, 100, 101],
            "ask_price_4": [104, 105, 106, 107, 108, 109],
            "bid_qty_5": [1, 1, 1, 1, 1, 1],
            "ask_qty_5": [1, 1, 1, 1, 1, 1],
            "bid_price_5": [95, 96, 97, 98, 99, 100],
            "ask_price_5": [105, 106, 107, 108, 109, 110],
        }
    )
    perp = kline.copy()
    perp["close"] = [101, 102, 103, 104, 105, 106]
    eth = kline.copy()
    eth["close"] = [50, 51, 52, 53, 54, 55]
    perp_book = book.copy()
    perp_book["bid_qty"] = [3, 3, 4, 4, 5, 5]

    store = build_second_level_feature_store(
        kline_frame=kline,
        agg_trades_frame=agg,
        book_frame=book,
        depth_frame=depth,
        cross_market_frame=perp,
        cross_market_book_frame=perp_book,
        eth_kline_frame=eth,
    )
    sampled = sample_second_level_feature_store(decisions, store)
    source_tables = build_second_level_source_tables(kline_frame=kline, agg_trades_frame=agg, book_frame=book)

    assert "sec_close" in store.columns
    assert "decision_grid_name" in store.columns
    assert "sec_median_trade_size" in store.columns
    assert "sec_bid_price" in store.columns
    assert "sl_ofi_5s" in sampled.columns
    assert "sl_large_trade_count_10s" in sampled.columns
    assert "sl_last_n_trades_buy_share" in sampled.columns
    assert "sl_buy_run_length_10s" in sampled.columns
    assert "sl_intrasecond_flow_concentration_10s" in sampled.columns
    assert "sl_price_minus_vwap_30s" in sampled.columns
    assert "sl_quote_update_asymmetry_10s" in sampled.columns
    assert "sl_shock_continuation_flag" in sampled.columns
    assert "sl_bid_depth_5" in sampled.columns
    assert "sl_book_slope_bid" in sampled.columns
    assert "sl_perp_return_5s" in sampled.columns
    assert "sl_spot_minus_perp_book_imbalance" in sampled.columns
    assert "sl_perp_book_imbalance_30s" in sampled.columns
    assert "sl_btc_minus_eth_return_30s" in sampled.columns
    assert "sl_crypto_beta_residual_return_30s" in sampled.columns
    assert "agg_trade_1s_event_summary" in source_tables
    assert "book_ticker_1s_quote_state" in source_tables
    assert "sec_close" not in sampled.columns
    assert sampled.loc[0, "timestamp"] == pd.Timestamp("2024-01-01T00:00:04.500Z")


def test_second_level_v2_pack_profile_expands_mirror_features() -> None:
    kline = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T00:00:00Z", periods=20, freq="1s"),
            "open": [100 + index for index in range(20)],
            "high": [101 + index for index in range(20)],
            "low": [99 + index for index in range(20)],
            "close": [100 + index for index in range(20)],
            "volume": [10.0 + index for index in range(20)],
            "quote_volume": [1000.0 + index for index in range(20)],
            "trade_count": [2 + index for index in range(20)],
            "taker_buy_base_volume": [5.0 + index for index in range(20)],
            "taker_buy_quote_volume": [500.0 + index for index in range(20)],
        }
    )
    profile = SecondLevelFeatureProfile(
        packs=[
            "second_level_momentum",
            "second_level_volatility",
            "second_level_volume",
            "second_level_candle_structure",
            "second_level_path_structure",
            "second_level_lagged",
        ],
        windows=[1, 5, 10],
        compact_windows=[5, 10],
        slope_windows=[5],
        range_windows=[5],
        lagged_feature_names=["sl_mirror_ret_5s"],
        lagged_feature_lags=[1, 3],
    )

    store = build_second_level_feature_store(kline_frame=kline, feature_profile=profile)
    sampled = sample_second_level_feature_store(
        pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01T00:00:15Z"], utc=True)}),
        store,
    )

    assert get_second_level_feature_pack("second_level_momentum").name == "second_level_momentum"
    assert "sl_mirror_ret_5s" in store.columns
    assert "sl_mirror_rv_5s" in store.columns
    assert "sl_mirror_relative_volume_5s" in store.columns
    assert "sl_mirror_body_pct_1s" in store.columns
    assert "sl_mirror_ret_5s_lag1s" in store.columns
    assert "sec_close" not in sampled.columns
    assert "sl_mirror_ret_5s" in sampled.columns


def test_partitioned_second_level_feature_store_writes_trimmed_chunks(tmp_path) -> None:
    kline = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T23:59:55Z", periods=12, freq="1s"),
            "open": [100 + index for index in range(12)],
            "high": [100 + index for index in range(12)],
            "low": [100 + index for index in range(12)],
            "close": [100 + index for index in range(12)],
            "volume": [1.0] * 12,
            "quote_volume": [100.0] * 12,
            "trade_count": [1] * 12,
            "taker_buy_base_volume": [0.5] * 12,
            "taker_buy_quote_volume": [50.0] * 12,
        }
    )
    manifest = write_partitioned_second_level_feature_store(
        kline_frame=kline,
        output_dir=tmp_path / "data_v2" / "second_level" / "version=second_level_v2" / "market=BTCUSDT" / "second_features",
        partition_frequency="daily",
        warmup_seconds=5,
    )
    loaded = load_second_level_frame(tmp_path / "data_v2" / "second_level" / "version=second_level_v2" / "market=BTCUSDT" / "second_features")
    sampled = load_sampled_second_level_features(
        pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01T23:59:59Z", "2024-01-02T00:00:04Z"], utc=True)}),
        tmp_path / "data_v2" / "second_level" / "version=second_level_v2" / "market=BTCUSDT" / "second_features",
    )

    assert manifest["feature_version"] == "second_level_v2"
    assert manifest["feature_profile"] == "expanded_v2"
    assert "second_level_momentum" in manifest["feature_packs"]
    assert manifest["partitioned"] is True
    assert manifest["row_count"] == len(kline)
    assert len(manifest["partitions"]) == 2
    assert len(loaded) == len(kline)
    assert loaded["timestamp"].min() == kline["timestamp"].min()
    assert loaded["timestamp"].max() == kline["timestamp"].max()
    assert len(sampled) == 2
    assert "sl_return_1s" in sampled.columns
