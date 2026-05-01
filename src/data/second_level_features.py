from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.constants import DEFAULT_TIMESTAMP_COLUMN


SECOND_LEVEL_WINDOWS = (1, 3, 5, 10, 15, 30, 60, 120, 300)
COMPACT_WINDOWS = (5, 10, 30, 60, 300)
BOOK_DYNAMICS_WINDOWS = (5, 10, 30)


def load_second_level_frame(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(resolved)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    if suffix in {".feather", ".ft"}:
        return pd.read_feather(resolved)
    raise ValueError(f"Unsupported second-level input format: {resolved.suffix}")


def normalize_trade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp_column = DEFAULT_TIMESTAMP_COLUMN
    if timestamp_column not in frame.columns:
        if "transact_time" in frame.columns:
            timestamp_column = "transact_time"
        elif "date" in frame.columns:
            timestamp_column = "date"
    if timestamp_column not in frame.columns:
        raise ValueError("Trade frame requires timestamp, date, or transact_time.")
    if "price" not in frame.columns or "quantity" not in frame.columns:
        raise ValueError("Trade frame requires price and quantity columns.")

    normalized = frame.copy()
    normalized[DEFAULT_TIMESTAMP_COLUMN] = pd.to_datetime(normalized[timestamp_column], utc=True)
    normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
    normalized["quantity"] = pd.to_numeric(normalized["quantity"], errors="coerce")
    if "quote_quantity" in normalized.columns:
        normalized["quote_quantity"] = pd.to_numeric(normalized["quote_quantity"], errors="coerce")
    else:
        normalized["quote_quantity"] = normalized["price"] * normalized["quantity"]
    if "is_buyer_maker" in normalized.columns:
        maker = normalized["is_buyer_maker"]
        if maker.dtype == "object":
            maker = maker.astype(str).str.lower().isin({"true", "1", "t"})
        normalized["is_taker_buy"] = ~maker.astype(bool)
    elif "is_taker_buy" in normalized.columns:
        normalized["is_taker_buy"] = normalized["is_taker_buy"].astype(bool)
    else:
        normalized["is_taker_buy"] = pd.NA
    normalized = normalized.dropna(subset=[DEFAULT_TIMESTAMP_COLUMN, "price", "quantity"])
    return normalized.sort_values(DEFAULT_TIMESTAMP_COLUMN).reset_index(drop=True)


def is_preaggregated_trade_second_frame(frame: pd.DataFrame) -> bool:
    return {
        DEFAULT_TIMESTAMP_COLUMN,
        "price",
        "trade_count",
        "total_volume",
        "dollar_volume",
        "taker_buy_volume",
        "taker_sell_volume",
        "taker_buy_count",
        "taker_sell_count",
        "signed_dollar_flow",
    }.issubset(frame.columns)


def normalize_preaggregated_trade_second_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[DEFAULT_TIMESTAMP_COLUMN] = pd.to_datetime(normalized[DEFAULT_TIMESTAMP_COLUMN], utc=True)
    numeric_columns = [
        "price",
        "trade_count",
        "total_volume",
        "dollar_volume",
        "taker_buy_volume",
        "taker_sell_volume",
        "taker_buy_count",
        "taker_sell_count",
        "signed_dollar_flow",
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.dropna(subset=[DEFAULT_TIMESTAMP_COLUMN]).sort_values(DEFAULT_TIMESTAMP_COLUMN).reset_index(drop=True)


def normalize_second_book_frame(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp_column = DEFAULT_TIMESTAMP_COLUMN
    if timestamp_column not in frame.columns:
        if "transaction_time" in frame.columns:
            timestamp_column = "transaction_time"
        elif "event_time" in frame.columns:
            timestamp_column = "event_time"
        elif "date" in frame.columns:
            timestamp_column = "date"
    required = {"bid_price", "bid_qty", "ask_price", "ask_qty"}
    if timestamp_column not in frame.columns or not required.issubset(frame.columns):
        raise ValueError("Book frame requires timestamp/date and bid_price, bid_qty, ask_price, ask_qty.")
    normalized = frame.copy()
    normalized[DEFAULT_TIMESTAMP_COLUMN] = pd.to_datetime(normalized[timestamp_column], utc=True)
    for column in required:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=[DEFAULT_TIMESTAMP_COLUMN, *required])
    return normalized.sort_values(DEFAULT_TIMESTAMP_COLUMN).reset_index(drop=True)


def _decision_index(decision_timestamps: pd.Series) -> pd.DatetimeIndex:
    timestamps = pd.to_datetime(decision_timestamps, utc=True)
    return pd.DatetimeIndex(timestamps).sort_values()


def _second_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.floor("s"), end.ceil("s"), freq="1s", tz="UTC")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _asof_to_decisions(features: pd.DataFrame, decision_timestamps: pd.Series) -> pd.DataFrame:
    decisions = pd.DataFrame({DEFAULT_TIMESTAMP_COLUMN: pd.to_datetime(decision_timestamps, utc=True)})
    aligned = pd.merge_asof(
        decisions.sort_values(DEFAULT_TIMESTAMP_COLUMN),
        features.sort_index().reset_index().rename(columns={"index": DEFAULT_TIMESTAMP_COLUMN}),
        on=DEFAULT_TIMESTAMP_COLUMN,
        direction="backward",
    )
    return aligned.set_index(decisions.index).drop(columns=[DEFAULT_TIMESTAMP_COLUMN])


def _sample_second_series(
    series: pd.Series,
    decision_index: pd.DatetimeIndex,
    output_index: pd.Index,
) -> pd.Series:
    positions = series.index.searchsorted(decision_index, side="right") - 1
    sampled = np.full(len(decision_index), np.nan, dtype="float64")
    valid = positions >= 0
    if valid.any():
        sampled[valid] = series.to_numpy(dtype="float64", copy=False)[positions[valid]]
    return pd.Series(sampled, index=output_index)


def build_trade_second_level_features(
    decision_timestamps: pd.Series,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    decisions = _decision_index(decision_timestamps)
    if is_preaggregated_trade_second_frame(trades):
        normalized = normalize_preaggregated_trade_second_frame(trades)
        if normalized.empty:
            return pd.DataFrame(index=decision_timestamps.index)
        start = min(normalized[DEFAULT_TIMESTAMP_COLUMN].min(), decisions.min()) - pd.Timedelta(seconds=max(SECOND_LEVEL_WINDOWS))
        end = max(normalized[DEFAULT_TIMESTAMP_COLUMN].max(), decisions.max())
        one_second_index = _second_index(start, end)
        per_second = normalized.set_index(DEFAULT_TIMESTAMP_COLUMN).sort_index().reindex(one_second_index)
    else:
        normalized = normalize_trade_frame(trades)
        if normalized.empty:
            return pd.DataFrame(index=decision_timestamps.index)

        start = min(normalized[DEFAULT_TIMESTAMP_COLUMN].min(), decisions.min()) - pd.Timedelta(seconds=max(SECOND_LEVEL_WINDOWS))
        end = max(normalized[DEFAULT_TIMESTAMP_COLUMN].max(), decisions.max())
        one_second_index = _second_index(start, end)
        events = normalized.set_index(DEFAULT_TIMESTAMP_COLUMN).sort_index()

        is_taker_buy = events["is_taker_buy"].fillna(False).astype(bool)
        buy_quantity = events["quantity"].where(is_taker_buy, 0.0)
        sell_quantity = events["quantity"].where(~is_taker_buy, 0.0)
        buy_count = is_taker_buy.astype("int64")
        sell_count = (~is_taker_buy).astype("int64")
        signed_dollar = events["quote_quantity"].where(is_taker_buy, -events["quote_quantity"])
        per_second = pd.DataFrame(
            {
                "price": events["price"].resample("1s").last(),
                "trade_count": events["quantity"].resample("1s").count(),
                "total_volume": events["quantity"].resample("1s").sum(),
                "dollar_volume": events["quote_quantity"].resample("1s").sum(),
                "taker_buy_volume": buy_quantity.resample("1s").sum(),
                "taker_sell_volume": sell_quantity.resample("1s").sum(),
                "taker_buy_count": buy_count.resample("1s").sum(),
                "taker_sell_count": sell_count.resample("1s").sum(),
                "signed_dollar_flow": signed_dollar.resample("1s").sum(),
            }
        ).reindex(one_second_index)
    flow_columns = [column for column in per_second.columns if column != "price"]
    per_second[flow_columns] = per_second[flow_columns].fillna(0.0)
    per_second["price"] = per_second["price"].ffill()
    second_ret = per_second["price"].pct_change(fill_method=None).fillna(0.0)
    second_log_ret = np.log(per_second["price"] / per_second["price"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    output_index = decision_timestamps.index
    sampled_features: dict[str, pd.Series] = {}
    full_features: dict[str, pd.Series] = {}

    def add_feature(name: str, values: pd.Series, *, keep_full: bool = False) -> None:
        sampled_features[name] = _sample_second_series(values, decisions, output_index)
        if keep_full:
            full_features[name] = values

    for window in SECOND_LEVEL_WINDOWS:
        price_lag = per_second["price"].shift(window)
        keep_return = window in {5, 10, 30, 60, 300}
        keep_rv = window in {10, 30, 60, 300}
        add_feature(f"sl_return_{window}s", per_second["price"] / price_lag - 1.0, keep_full=keep_return)
        add_feature(f"sl_log_return_{window}s", np.log(per_second["price"] / price_lag).replace([np.inf, -np.inf], np.nan))
        add_feature(f"sl_rv_{window}s", second_ret.rolling(window, min_periods=1).std(ddof=0), keep_full=keep_rv)
        add_feature(f"sl_rvar_{window}s", second_ret.rolling(window, min_periods=1).var(ddof=0))
        add_feature(f"sl_mean_abs_return_{window}s", second_ret.abs().rolling(window, min_periods=1).mean())
        add_feature(f"sl_max_abs_return_{window}s", second_ret.abs().rolling(window, min_periods=1).max())
        price_range = per_second["price"].rolling(window, min_periods=1).max() / per_second["price"].rolling(window, min_periods=1).min() - 1.0
        add_feature(f"sl_range_{window}s", price_range)

    log_price = np.log(per_second["price"].replace(0, np.nan))
    for window in (5, 10, 30, 60):
        price_slope = (per_second["price"] - per_second["price"].shift(window)) / window
        add_feature(f"sl_price_slope_{window}s", price_slope, keep_full=window in {5, 10, 30, 60})
        add_feature(f"sl_log_price_slope_{window}s", (log_price - log_price.shift(window)) / window)
    add_feature("sl_return_5s_minus_60s", full_features["sl_return_5s"] - full_features["sl_return_60s"])
    add_feature("sl_return_10s_minus_60s", full_features["sl_return_10s"] - full_features["sl_return_60s"])
    add_feature("sl_return_30s_minus_300s", full_features["sl_return_30s"] - full_features["sl_return_300s"])
    add_feature("sl_slope_5s_minus_slope_30s", full_features["sl_price_slope_5s"] - full_features["sl_price_slope_30s"])
    add_feature("sl_slope_10s_minus_slope_60s", full_features["sl_price_slope_10s"] - full_features["sl_price_slope_60s"])
    add_feature("sl_return_accel_5s_30s", full_features["sl_return_5s"] - full_features["sl_return_30s"] * (5.0 / 30.0))
    add_feature("sl_late_window_acceleration_flag", (full_features["sl_return_5s"] > full_features["sl_return_30s"] * (5.0 / 30.0)).astype(float))
    add_feature("sl_return_5s_sign", np.sign(full_features["sl_return_5s"]))
    add_feature("sl_return_30s_sign", np.sign(full_features["sl_return_30s"]))
    add_feature("sl_return_5s_30s_sign_disagreement", (np.sign(full_features["sl_return_5s"]) != np.sign(full_features["sl_return_30s"])).astype(float))

    ret_sign = np.sign(second_ret)
    flip = (ret_sign != ret_sign.shift(1)).astype(float).where(ret_sign != 0, 0.0)
    for window in (10, 30, 60):
        add_feature(f"sl_direction_flips_{window}s", flip.rolling(window, min_periods=1).sum())
        same_direction = ret_sign == np.sign(full_features[f"sl_return_{window}s"])
        add_feature(f"sl_momentum_persistence_{window}s", same_direction.astype(float).rolling(window, min_periods=1).mean())
        path_length = second_ret.abs().rolling(window, min_periods=1).sum()
        directional_efficiency = _safe_divide(full_features[f"sl_return_{window}s"].abs(), path_length)
        add_feature(f"sl_directional_efficiency_{window}s", directional_efficiency)
        choppiness = 1.0 - directional_efficiency.clip(0.0, 1.0)
        add_feature(f"sl_choppiness_{window}s", choppiness, keep_full=window == 30)
    rolling_high_60 = per_second["price"].rolling(60, min_periods=1).max()
    rolling_low_60 = per_second["price"].rolling(60, min_periods=1).min()
    add_feature("sl_max_adverse_move_60s", per_second["price"] / rolling_high_60 - 1.0)
    add_feature("sl_rebound_from_local_low_60s", per_second["price"] / rolling_low_60 - 1.0)
    add_feature("sl_pullback_from_local_high_60s", per_second["price"] / rolling_high_60 - 1.0)
    add_feature("sl_last_second_reversal_flag", (ret_sign != ret_sign.shift(1)).astype(float))

    for window in COMPACT_WINDOWS:
        total_volume = per_second["total_volume"].rolling(window, min_periods=1).sum()
        dollar_volume = per_second["dollar_volume"].rolling(window, min_periods=1).sum()
        trade_count = per_second["trade_count"].rolling(window, min_periods=1).sum()
        buy_volume = per_second["taker_buy_volume"].rolling(window, min_periods=1).sum()
        sell_volume = per_second["taker_sell_volume"].rolling(window, min_periods=1).sum()
        buy_count_window = per_second["taker_buy_count"].rolling(window, min_periods=1).sum()
        sell_count_window = per_second["taker_sell_count"].rolling(window, min_periods=1).sum()
        signed_flow = per_second["signed_dollar_flow"].rolling(window, min_periods=1).sum()
        keep = window in {10, 30, 60, 300}
        add_feature(f"sl_taker_buy_volume_{window}s", buy_volume, keep_full=window in {30, 300})
        add_feature(f"sl_taker_sell_volume_{window}s", sell_volume, keep_full=window in {30, 300})
        add_feature(f"sl_taker_buy_count_{window}s", buy_count_window, keep_full=window in {30, 300})
        add_feature(f"sl_taker_sell_count_{window}s", sell_count_window, keep_full=window in {30, 300})
        add_feature(f"sl_taker_buy_ratio_{window}s", _safe_divide(buy_volume, total_volume))
        add_feature(f"sl_taker_sell_ratio_{window}s", _safe_divide(sell_volume, total_volume))
        taker_imbalance = _safe_divide(buy_volume - sell_volume, buy_volume + sell_volume)
        add_feature(f"sl_taker_imbalance_{window}s", taker_imbalance, keep_full=window in {30, 300})
        add_feature(f"sl_taker_count_imbalance_{window}s", _safe_divide(buy_count_window - sell_count_window, buy_count_window + sell_count_window))
        add_feature(f"sl_signed_dollar_flow_{window}s", signed_flow, keep_full=keep)
        add_feature(f"sl_signed_dollar_flow_ratio_{window}s", _safe_divide(signed_flow, dollar_volume))
        add_feature(f"sl_trade_count_{window}s", trade_count, keep_full=window in {30, 300})
        add_feature(f"sl_trades_per_second_{window}s", trade_count / window)
        add_feature(f"sl_total_volume_{window}s", total_volume, keep_full=window in {30, 300})
        add_feature(f"sl_dollar_volume_{window}s", dollar_volume)
        add_feature(f"sl_avg_trade_size_{window}s", _safe_divide(total_volume, trade_count))
        add_feature(f"sl_price_impact_proxy_{window}s", _safe_divide(full_features[f"sl_return_{window}s"].abs(), dollar_volume))

    add_feature(
        "sl_taker_imbalance_30s_z_300s",
        _safe_divide(
            full_features["sl_taker_imbalance_30s"] - full_features["sl_taker_imbalance_300s"],
            full_features["sl_taker_imbalance_300s"].rolling(300, min_periods=30).std(ddof=0),
        ),
    )
    add_feature(
        "sl_signed_dollar_flow_30s_z_300s",
        _safe_divide(
            full_features["sl_signed_dollar_flow_30s"] - full_features["sl_signed_dollar_flow_300s"],
            full_features["sl_signed_dollar_flow_300s"].rolling(300, min_periods=30).std(ddof=0),
        ),
    )
    add_feature("sl_signed_dollar_flow_acceleration", full_features["sl_signed_dollar_flow_10s"] - full_features["sl_signed_dollar_flow_60s"] * (10.0 / 60.0))
    add_feature(
        "sl_trade_intensity_z_300s",
        _safe_divide(
            full_features["sl_trade_count_30s"] - full_features["sl_trade_count_300s"] * 0.1,
            full_features["sl_trade_count_300s"].rolling(300, min_periods=30).std(ddof=0),
        ),
    )
    volume_burst_30s = (full_features["sl_total_volume_30s"] > full_features["sl_total_volume_300s"] * 0.15).astype(float)
    add_feature("sl_volume_burst_30s", volume_burst_30s, keep_full=True)
    add_feature("sl_buy_volume_burst_flag", (full_features["sl_taker_buy_volume_30s"] > full_features["sl_taker_buy_volume_300s"] * 0.15).astype(float))
    add_feature("sl_sell_volume_burst_flag", (full_features["sl_taker_sell_volume_30s"] > full_features["sl_taker_sell_volume_300s"] * 0.15).astype(float))
    add_feature("sl_buy_trade_count_burst_flag", (full_features["sl_taker_buy_count_30s"] > full_features["sl_taker_buy_count_300s"] * 0.15).astype(float))
    add_feature("sl_sell_trade_count_burst_flag", (full_features["sl_taker_sell_count_30s"] > full_features["sl_taker_sell_count_300s"] * 0.15).astype(float))
    add_feature("sl_last_3s_buy_dominance", (per_second["taker_buy_volume"].rolling(3, min_periods=1).sum() > per_second["taker_sell_volume"].rolling(3, min_periods=1).sum()).astype(float))
    add_feature("sl_last_3s_sell_dominance", (per_second["taker_sell_volume"].rolling(3, min_periods=1).sum() > per_second["taker_buy_volume"].rolling(3, min_periods=1).sum()).astype(float))
    add_feature("sl_rv_10s_div_60s", _safe_divide(full_features["sl_rv_10s"], full_features["sl_rv_60s"]))
    add_feature("sl_rv_30s_div_300s", _safe_divide(full_features["sl_rv_30s"], full_features["sl_rv_300s"]))
    add_feature("sl_abs_return_10s_div_rv_60s", _safe_divide(full_features["sl_return_10s"].abs(), full_features["sl_rv_60s"]))
    add_feature("sl_abs_return_30s_div_rv_300s", _safe_divide(full_features["sl_return_30s"].abs(), full_features["sl_rv_300s"]))
    add_feature("sl_return_10s_x_volume_burst", full_features["sl_return_10s"] * volume_burst_30s)
    add_feature("sl_return_30s_x_choppiness", full_features["sl_return_30s"] * full_features["sl_choppiness_30s"])
    add_feature("sl_positive_taker_imbalance_30s", full_features["sl_taker_imbalance_30s"].clip(lower=0.0))
    add_feature("sl_negative_taker_imbalance_30s", -full_features["sl_taker_imbalance_30s"].clip(upper=0.0))
    add_feature("sl_positive_signed_flow_30s", full_features["sl_signed_dollar_flow_30s"].clip(lower=0.0))
    add_feature("sl_negative_signed_flow_30s", -full_features["sl_signed_dollar_flow_30s"].clip(upper=0.0))
    return pd.DataFrame(sampled_features, index=output_index)


def build_book_second_level_features(
    decision_timestamps: pd.Series,
    book: pd.DataFrame,
) -> pd.DataFrame:
    normalized = normalize_second_book_frame(book)
    if normalized.empty:
        return pd.DataFrame(index=decision_timestamps.index)

    decisions = _decision_index(decision_timestamps)
    start = min(normalized[DEFAULT_TIMESTAMP_COLUMN].min(), decisions.min()) - pd.Timedelta(seconds=max(SECOND_LEVEL_WINDOWS))
    end = max(normalized[DEFAULT_TIMESTAMP_COLUMN].max(), decisions.max())
    one_second_index = _second_index(start, end)
    events = normalized.set_index(DEFAULT_TIMESTAMP_COLUMN).sort_index()
    per_second = events[["bid_price", "bid_qty", "ask_price", "ask_qty"]].resample("1s").last().reindex(one_second_index).ffill()
    mid = (per_second["bid_price"] + per_second["ask_price"]) / 2.0
    total_qty = (per_second["bid_qty"] + per_second["ask_qty"]).replace(0, np.nan)
    microprice = (per_second["ask_price"] * per_second["bid_qty"] + per_second["bid_price"] * per_second["ask_qty"]) / total_qty

    features = pd.DataFrame(index=one_second_index)
    features["sl_best_bid"] = per_second["bid_price"]
    features["sl_best_ask"] = per_second["ask_price"]
    features["sl_mid_price"] = mid
    features["sl_spread"] = per_second["ask_price"] - per_second["bid_price"]
    features["sl_spread_bps"] = _safe_divide(features["sl_spread"], mid) * 10000.0
    features["sl_bid_qty"] = per_second["bid_qty"]
    features["sl_ask_qty"] = per_second["ask_qty"]
    features["sl_bid_ask_qty_imbalance"] = _safe_divide(per_second["bid_qty"] - per_second["ask_qty"], total_qty)
    features["sl_microprice"] = microprice
    features["sl_microprice_premium"] = _safe_divide(microprice - mid, mid)
    quote_update_count = normalized.set_index(DEFAULT_TIMESTAMP_COLUMN)["bid_price"].resample("1s").count().reindex(one_second_index).fillna(0.0)

    for window in BOOK_DYNAMICS_WINDOWS:
        features[f"sl_bid_qty_change_{window}s"] = per_second["bid_qty"] - per_second["bid_qty"].shift(window)
        features[f"sl_ask_qty_change_{window}s"] = per_second["ask_qty"] - per_second["ask_qty"].shift(window)
        features[f"sl_book_imbalance_change_{window}s"] = features["sl_bid_ask_qty_imbalance"] - features["sl_bid_ask_qty_imbalance"].shift(window)
        features[f"sl_spread_bps_change_{window}s"] = features["sl_spread_bps"] - features["sl_spread_bps"].shift(window)
        features[f"sl_microprice_drift_{window}s"] = microprice / microprice.shift(window) - 1.0
        features[f"sl_mid_price_drift_{window}s"] = mid / mid.shift(window) - 1.0
        features[f"sl_quote_update_count_{window}s"] = quote_update_count.rolling(window, min_periods=1).sum()
        features[f"sl_spread_widening_{window}s"] = (features[f"sl_spread_bps_change_{window}s"] > 0).astype(float)
        features[f"sl_spread_tightening_{window}s"] = (features[f"sl_spread_bps_change_{window}s"] < 0).astype(float)
        features[f"sl_bid_qty_depletion_{window}s"] = (-features[f"sl_bid_qty_change_{window}s"]).clip(lower=0.0)
        features[f"sl_ask_qty_depletion_{window}s"] = (-features[f"sl_ask_qty_change_{window}s"]).clip(lower=0.0)

    for window in (10, 30, 60):
        features[f"sl_spread_bps_mean_{window}s"] = features["sl_spread_bps"].rolling(window, min_periods=1).mean()
        features[f"sl_spread_bps_max_{window}s"] = features["sl_spread_bps"].rolling(window, min_periods=1).max()
        features[f"sl_spread_bps_vol_{window}s"] = features["sl_spread_bps"].rolling(window, min_periods=1).std(ddof=0)
    features["sl_imbalance_persistence_30s"] = (np.sign(features["sl_bid_ask_qty_imbalance"]) == np.sign(features["sl_bid_ask_qty_imbalance"].shift(1))).astype(float).rolling(30, min_periods=1).mean()
    features["sl_book_imbalance_x_microprice_premium"] = features["sl_bid_ask_qty_imbalance"] * features["sl_microprice_premium"]
    features["sl_positive_microprice_premium"] = features["sl_microprice_premium"].clip(lower=0.0)
    features["sl_negative_microprice_premium"] = -features["sl_microprice_premium"].clip(upper=0.0)
    return _asof_to_decisions(features, decision_timestamps)


def build_second_level_feature_frame(
    decision_frame: pd.DataFrame,
    *,
    trades_frame: pd.DataFrame | None = None,
    book_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    decision_timestamps = decision_frame[DEFAULT_TIMESTAMP_COLUMN]
    pieces: list[pd.DataFrame] = []
    if trades_frame is not None and not trades_frame.empty:
        pieces.append(build_trade_second_level_features(decision_timestamps, trades_frame))
    if book_frame is not None and not book_frame.empty:
        pieces.append(build_book_second_level_features(decision_timestamps, book_frame))
    if not pieces:
        return pd.DataFrame(index=decision_frame.index)
    combined = pd.concat(pieces, axis=1)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined[DEFAULT_TIMESTAMP_COLUMN] = pd.to_datetime(decision_timestamps, utc=True).to_numpy()
    return combined.reset_index(drop=True)
