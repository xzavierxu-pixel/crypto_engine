from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from collections.abc import Iterator

import numpy as np
import pandas as pd

from src.core.constants import DEFAULT_TIMESTAMP_COLUMN
from src.data.second_level_feature_packs import (
    DEFAULT_SECOND_LEVEL_PACKS,
    SecondLevelFeatureProfile,
    build_second_level_pack_features,
)


SECOND_LEVEL_WINDOWS = (1, 3, 5, 10, 15, 30, 60, 120, 300)
COMPACT_WINDOWS = (5, 10, 30, 60, 300)
BOOK_DYNAMICS_WINDOWS = (5, 10, 30)
SECOND_LEVEL_FEATURE_STORE_VERSION = "second_level_v2"
DEFAULT_FEATURE_STORE_WARMUP_SECONDS = max(SECOND_LEVEL_WINDOWS)


def resolve_second_level_feature_profile(payload: dict[str, Any] | None = None) -> SecondLevelFeatureProfile:
    return SecondLevelFeatureProfile(**(payload or {}))


def load_second_level_frame(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if resolved.is_dir():
        parquet_files = sorted(
            file
            for file in resolved.rglob("*.parquet")
            if "source_tables" not in file.parts
        )
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under second-level input directory: {resolved}")
        return pd.concat((pd.read_parquet(file) for file in parquet_files), ignore_index=True)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(resolved)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    if suffix in {".feather", ".ft"}:
        return pd.read_feather(resolved)
    raise ValueError(f"Unsupported second-level input format: {resolved.suffix}")


def _to_datetime_utc(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        median = numeric.dropna().abs().median()
        unit = "us" if median > 10_000_000_000_000 else "ms" if median > 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def _first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def normalize_trade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp_column = DEFAULT_TIMESTAMP_COLUMN
    if timestamp_column not in frame.columns:
        timestamp_column = _first_existing(frame.columns, ("transact_time", "date", "time", "T")) or timestamp_column
    if timestamp_column not in frame.columns:
        raise ValueError("Trade frame requires timestamp, date, or transact_time.")
    normalized = frame.copy()
    rename_map = {}
    if "p" in normalized.columns and "price" not in normalized.columns:
        rename_map["p"] = "price"
    if "q" in normalized.columns and "quantity" not in normalized.columns:
        rename_map["q"] = "quantity"
    if "m" in normalized.columns and "is_buyer_maker" not in normalized.columns:
        rename_map["m"] = "is_buyer_maker"
    if rename_map:
        normalized = normalized.rename(columns=rename_map)
    if "price" not in normalized.columns or "quantity" not in normalized.columns:
        raise ValueError("Trade frame requires price and quantity columns.")

    normalized[DEFAULT_TIMESTAMP_COLUMN] = _to_datetime_utc(normalized[timestamp_column])
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
    normalized[DEFAULT_TIMESTAMP_COLUMN] = _to_datetime_utc(normalized[DEFAULT_TIMESTAMP_COLUMN])
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
        timestamp_column = _first_existing(frame.columns, ("transaction_time", "event_time", "date", "E", "T")) or timestamp_column
    normalized = frame.copy()
    rename_map = {}
    if "b" in normalized.columns and "bid_price" not in normalized.columns:
        rename_map["b"] = "bid_price"
    if "B" in normalized.columns and "bid_qty" not in normalized.columns:
        rename_map["B"] = "bid_qty"
    if "a" in normalized.columns and "ask_price" not in normalized.columns:
        rename_map["a"] = "ask_price"
    if "A" in normalized.columns and "ask_qty" not in normalized.columns:
        rename_map["A"] = "ask_qty"
    if rename_map:
        normalized = normalized.rename(columns=rename_map)
    required = {"bid_price", "bid_qty", "ask_price", "ask_qty"}
    if timestamp_column not in normalized.columns or not required.issubset(normalized.columns):
        raise ValueError("Book frame requires timestamp/date and bid_price, bid_qty, ask_price, ask_qty.")
    normalized[DEFAULT_TIMESTAMP_COLUMN] = _to_datetime_utc(normalized[timestamp_column])
    for column in required:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=[DEFAULT_TIMESTAMP_COLUMN, *required])
    return normalized.sort_values(DEFAULT_TIMESTAMP_COLUMN).reset_index(drop=True)


def normalize_second_kline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    timestamp_column = _first_existing(normalized.columns, (DEFAULT_TIMESTAMP_COLUMN, "open_time", "date", "t"))
    if timestamp_column is None:
        raise ValueError("1s kline frame requires timestamp, open_time, date, or t.")
    rename_map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "q": "quote_volume",
        "n": "trade_count",
        "V": "taker_buy_base_volume",
        "Q": "taker_buy_quote_volume",
    }
    normalized = normalized.rename(columns={k: v for k, v in rename_map.items() if k in normalized.columns and v not in normalized.columns})
    if "quote_asset_volume" in normalized.columns and "quote_volume" not in normalized.columns:
        normalized = normalized.rename(columns={"quote_asset_volume": "quote_volume"})
    if "number_of_trades" in normalized.columns and "trade_count" not in normalized.columns:
        normalized = normalized.rename(columns={"number_of_trades": "trade_count"})
    if "count" in normalized.columns and "trade_count" not in normalized.columns:
        normalized = normalized.rename(columns={"count": "trade_count"})
    if "taker_buy_volume" in normalized.columns and "taker_buy_base_volume" not in normalized.columns:
        normalized = normalized.rename(columns={"taker_buy_volume": "taker_buy_base_volume"})
    if "taker_buy_base_asset_volume" in normalized.columns and "taker_buy_base_volume" not in normalized.columns:
        normalized = normalized.rename(columns={"taker_buy_base_asset_volume": "taker_buy_base_volume"})
    if "taker_buy_quote_asset_volume" in normalized.columns and "taker_buy_quote_volume" not in normalized.columns:
        normalized = normalized.rename(columns={"taker_buy_quote_asset_volume": "taker_buy_quote_volume"})
    required = {"open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume"}
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise ValueError(f"1s kline frame missing required columns: {missing}")
    normalized[DEFAULT_TIMESTAMP_COLUMN] = _to_datetime_utc(normalized[timestamp_column])
    for column in required:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=[DEFAULT_TIMESTAMP_COLUMN, "open", "high", "low", "close"])
    return normalized.sort_values(DEFAULT_TIMESTAMP_COLUMN).reset_index(drop=True)


def _decision_index(decision_timestamps: pd.Series) -> pd.DatetimeIndex:
    timestamps = pd.to_datetime(decision_timestamps, utc=True)
    return pd.DatetimeIndex(timestamps).sort_values()


def _second_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.floor("s"), end.ceil("s"), freq="1s", tz="UTC")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _append_columns(frame: pd.DataFrame, columns: dict[str, pd.Series | np.ndarray | float | bool]) -> pd.DataFrame:
    if not columns:
        return frame
    additions = pd.DataFrame(columns, index=frame.index)
    return pd.concat([frame, additions], axis=1).copy()


def _normalize_partition_frequency(value: str) -> str:
    normalized = value.lower()
    if normalized in {"month", "monthly", "ms"}:
        return "MS"
    if normalized in {"day", "daily", "d"}:
        return "D"
    raise ValueError("partition_frequency must be one of: monthly, daily.")


def _iter_time_partitions(start: pd.Timestamp, end: pd.Timestamp, frequency: str) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    freq = _normalize_partition_frequency(frequency)
    start = pd.Timestamp(start).floor("s")
    end = pd.Timestamp(end).floor("s")
    anchors = pd.date_range(start.floor("D"), end.ceil("D"), freq=freq, tz="UTC")
    if len(anchors) == 0 or anchors[0] > start:
        anchors = anchors.insert(0, start.floor("D"))
    for index, anchor in enumerate(anchors):
        next_anchor = anchors[index + 1] if index + 1 < len(anchors) else end + pd.Timedelta(seconds=1)
        chunk_start = max(start, anchor)
        chunk_end = min(end, next_anchor - pd.Timedelta(seconds=1))
        if chunk_start <= chunk_end:
            yield chunk_start, chunk_end


def _timestamp_bounds(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    if DEFAULT_TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(f"Frame requires {DEFAULT_TIMESTAMP_COLUMN} column for partitioned feature-store build.")
    timestamps = pd.to_datetime(frame[DEFAULT_TIMESTAMP_COLUMN], utc=True)
    return timestamps.min(), timestamps.max()


def _load_partitioned_frame_by_time(path: str | Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Partitioned second-level input does not exist: {resolved}")
    if resolved.is_file():
        frame = load_second_level_frame(resolved)
        return _slice_frame_by_time(frame, start, end)

    frames: list[pd.DataFrame] = []
    for day in pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC"):
        partition_dir = resolved / f"date={day.strftime('%Y-%m-%d')}"
        candidates = sorted(partition_dir.glob("*.parquet")) if partition_dir.exists() else []
        if not candidates:
            candidates = sorted(resolved.glob(f"*{day.strftime('%Y-%m-%d')}*.parquet"))
        for candidate in candidates:
            frames.append(pd.read_parquet(candidate))
    if not frames:
        return None
    return _slice_frame_by_time(pd.concat(frames, ignore_index=True), start, end)


def _slice_frame_by_time(
    frame: pd.DataFrame | str | Path | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    if isinstance(frame, (str, Path)):
        return _load_partitioned_frame_by_time(frame, start, end)
    if frame is None or frame.empty:
        return frame
    if DEFAULT_TIMESTAMP_COLUMN not in frame.columns:
        return frame
    timestamps = pd.to_datetime(frame[DEFAULT_TIMESTAMP_COLUMN], utc=True)
    return frame.loc[(timestamps >= start) & (timestamps <= end)].copy()


def _partition_label(start: pd.Timestamp, frequency: str) -> str:
    freq = _normalize_partition_frequency(frequency)
    if freq == "MS":
        return start.strftime("%Y-%m")
    return start.strftime("%Y-%m-%d")


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

    spread = per_second["ask_price"] - per_second["bid_price"]
    spread_bps = _safe_divide(spread, mid) * 10000.0
    qty_imbalance = _safe_divide(per_second["bid_qty"] - per_second["ask_qty"], total_qty)
    microprice_premium = _safe_divide(microprice - mid, mid)
    feature_columns: dict[str, pd.Series] = {
        "sl_best_bid": per_second["bid_price"],
        "sl_best_ask": per_second["ask_price"],
        "sl_mid_price": mid,
        "sl_spread": spread,
        "sl_spread_bps": spread_bps,
        "sl_bid_qty": per_second["bid_qty"],
        "sl_ask_qty": per_second["ask_qty"],
        "sl_bid_ask_qty_imbalance": qty_imbalance,
        "sl_microprice": microprice,
        "sl_microprice_premium": microprice_premium,
    }
    quote_update_count = normalized.set_index(DEFAULT_TIMESTAMP_COLUMN)["bid_price"].resample("1s").count().reindex(one_second_index).fillna(0.0)
    bid_price_delta = per_second["bid_price"].diff()
    ask_price_delta = per_second["ask_price"].diff()
    bid_qty_delta = per_second["bid_qty"].diff().fillna(0.0)
    ask_qty_delta = per_second["ask_qty"].diff().fillna(0.0)
    sec_ofi = (
        bid_qty_delta.where(bid_price_delta >= 0, -bid_qty_delta)
        - ask_qty_delta.where(ask_price_delta <= 0, -ask_qty_delta)
    ).fillna(0.0)
    bid_replenishment = bid_qty_delta.where((bid_qty_delta > 0) & (bid_price_delta >= 0), 0.0)
    ask_replenishment = ask_qty_delta.where((ask_qty_delta > 0) & (ask_price_delta <= 0), 0.0)
    bid_depletion = (-bid_qty_delta).where((bid_qty_delta < 0) & (bid_price_delta <= 0), 0.0)
    ask_depletion = (-ask_qty_delta).where((ask_qty_delta < 0) & (ask_price_delta >= 0), 0.0)
    feature_columns.update(
        {
            "sl_ofi_1s": sec_ofi,
            "sl_bid_replenishment_1s": bid_replenishment,
            "sl_ask_replenishment_1s": ask_replenishment,
            "sl_bid_depletion_1s": bid_depletion,
            "sl_ask_depletion_1s": ask_depletion,
        }
    )
    bid_uptick = (bid_price_delta > 0).astype(float)
    bid_downtick = (bid_price_delta < 0).astype(float)
    ask_uptick = (ask_price_delta > 0).astype(float)
    ask_downtick = (ask_price_delta < 0).astype(float)

    for window in BOOK_DYNAMICS_WINDOWS:
        bid_qty_change = per_second["bid_qty"] - per_second["bid_qty"].shift(window)
        ask_qty_change = per_second["ask_qty"] - per_second["ask_qty"].shift(window)
        spread_bps_change = spread_bps - spread_bps.shift(window)
        bid_replenishment_rate = bid_replenishment.rolling(window, min_periods=1).sum() / window
        ask_replenishment_rate = ask_replenishment.rolling(window, min_periods=1).sum() / window
        bid_depletion_rate = bid_depletion.rolling(window, min_periods=1).sum() / window
        ask_depletion_rate = ask_depletion.rolling(window, min_periods=1).sum() / window
        bid_uptick_count = bid_uptick.rolling(window, min_periods=1).sum()
        bid_downtick_count = bid_downtick.rolling(window, min_periods=1).sum()
        ask_uptick_count = ask_uptick.rolling(window, min_periods=1).sum()
        ask_downtick_count = ask_downtick.rolling(window, min_periods=1).sum()
        bid_revision_pressure = bid_uptick_count - bid_downtick_count
        ask_revision_pressure = ask_downtick_count - ask_uptick_count
        feature_columns.update(
            {
                f"sl_bid_qty_change_{window}s": bid_qty_change,
                f"sl_ask_qty_change_{window}s": ask_qty_change,
                f"sl_book_imbalance_change_{window}s": qty_imbalance - qty_imbalance.shift(window),
                f"sl_spread_bps_change_{window}s": spread_bps_change,
                f"sl_microprice_drift_{window}s": microprice / microprice.shift(window) - 1.0,
                f"sl_mid_price_drift_{window}s": mid / mid.shift(window) - 1.0,
                f"sl_quote_update_count_{window}s": quote_update_count.rolling(window, min_periods=1).sum(),
                f"sl_spread_widening_{window}s": (spread_bps_change > 0).astype(float),
                f"sl_spread_tightening_{window}s": (spread_bps_change < 0).astype(float),
                f"sl_bid_qty_depletion_{window}s": (-bid_qty_change).clip(lower=0.0),
                f"sl_ask_qty_depletion_{window}s": (-ask_qty_change).clip(lower=0.0),
                f"sl_ofi_{window}s": sec_ofi.rolling(window, min_periods=1).sum(),
                f"sl_bid_replenishment_rate_{window}s": bid_replenishment_rate,
                f"sl_ask_replenishment_rate_{window}s": ask_replenishment_rate,
                f"sl_bid_depletion_rate_{window}s": bid_depletion_rate,
                f"sl_ask_depletion_rate_{window}s": ask_depletion_rate,
                f"sl_bid_depletion_minus_replenishment_{window}s": bid_depletion_rate - bid_replenishment_rate,
                f"sl_ask_depletion_minus_replenishment_{window}s": ask_depletion_rate - ask_replenishment_rate,
                f"sl_bid_uptick_count_{window}s": bid_uptick_count,
                f"sl_bid_downtick_count_{window}s": bid_downtick_count,
                f"sl_ask_uptick_count_{window}s": ask_uptick_count,
                f"sl_ask_downtick_count_{window}s": ask_downtick_count,
                f"sl_bid_revision_pressure_{window}s": bid_revision_pressure,
                f"sl_ask_revision_pressure_{window}s": ask_revision_pressure,
                f"sl_quote_update_asymmetry_{window}s": bid_revision_pressure - ask_revision_pressure,
            }
        )
    ofi_30s = feature_columns["sl_ofi_30s"]
    feature_columns["sl_ofi_z_300s"] = _safe_divide(
        ofi_30s - ofi_30s.rolling(300, min_periods=30).mean(),
        ofi_30s.rolling(300, min_periods=30).std(ddof=0),
    )
    feature_columns["sl_ofi_persistence_30s"] = (np.sign(sec_ofi) == np.sign(sec_ofi.shift(1))).astype(float).rolling(30, min_periods=1).mean()
    feature_columns["sl_ofi_acceleration"] = feature_columns["sl_ofi_5s"] - feature_columns["sl_ofi_30s"] * (5.0 / 30.0)
    feature_columns["sl_bid_wall_disappear_flag"] = (
        feature_columns["sl_bid_depletion_rate_10s"] > feature_columns["sl_bid_replenishment_rate_10s"] * 3.0
    ).astype(float)
    feature_columns["sl_ask_wall_disappear_flag"] = (
        feature_columns["sl_ask_depletion_rate_10s"] > feature_columns["sl_ask_replenishment_rate_10s"] * 3.0
    ).astype(float)
    feature_columns["sl_microprice_improvement_persistence_30s"] = (
        np.sign(microprice_premium.diff()) == np.sign(microprice_premium.diff().shift(1))
    ).astype(float).rolling(30, min_periods=1).mean()

    for window in (10, 30, 60):
        feature_columns[f"sl_spread_bps_mean_{window}s"] = spread_bps.rolling(window, min_periods=1).mean()
        feature_columns[f"sl_spread_bps_max_{window}s"] = spread_bps.rolling(window, min_periods=1).max()
        feature_columns[f"sl_spread_bps_vol_{window}s"] = spread_bps.rolling(window, min_periods=1).std(ddof=0)
    feature_columns["sl_imbalance_persistence_30s"] = (
        np.sign(qty_imbalance) == np.sign(qty_imbalance.shift(1))
    ).astype(float).rolling(30, min_periods=1).mean()
    feature_columns["sl_book_imbalance_x_microprice_premium"] = qty_imbalance * microprice_premium
    feature_columns["sl_positive_microprice_premium"] = microprice_premium.clip(lower=0.0)
    feature_columns["sl_negative_microprice_premium"] = -microprice_premium.clip(upper=0.0)
    features = pd.DataFrame(feature_columns, index=one_second_index)
    return _asof_to_decisions(features, decision_timestamps)


def build_agg_trade_enrichment_features(
    decision_timestamps: pd.Series,
    trades: pd.DataFrame,
    *,
    large_trade_quantile: float = 0.95,
    large_trade_window_seconds: int = 300,
    cluster_threshold_ms: float = 100.0,
) -> pd.DataFrame:
    normalized = normalize_trade_frame(trades)
    if normalized.empty:
        return pd.DataFrame(index=decision_timestamps.index)

    decisions = _decision_index(decision_timestamps)
    start = min(normalized[DEFAULT_TIMESTAMP_COLUMN].min(), decisions.min()) - pd.Timedelta(seconds=max(SECOND_LEVEL_WINDOWS))
    end = max(normalized[DEFAULT_TIMESTAMP_COLUMN].max(), decisions.max())
    one_second_index = _second_index(start, end)
    per_second = _build_agg_trade_second_summary(
        normalized,
        one_second_index,
        large_trade_quantile=large_trade_quantile,
        large_trade_window_seconds=large_trade_window_seconds,
        cluster_threshold_ms=cluster_threshold_ms,
    )

    feature_columns: dict[str, pd.Series] = {}
    for window in (10, 30, 60):
        large_count = per_second["sec_large_trade_count"].rolling(window, min_periods=1).sum()
        large_buy_count = per_second["sec_large_buy_trade_count"].rolling(window, min_periods=1).sum()
        large_sell_count = per_second["sec_large_sell_trade_count"].rolling(window, min_periods=1).sum()
        large_notional = per_second["sec_large_trade_notional"].rolling(window, min_periods=1).sum()
        total_notional = per_second["sec_trade_notional"].rolling(window, min_periods=1).sum()
        large_buy_notional = per_second["sec_large_buy_notional"].rolling(window, min_periods=1).sum()
        large_sell_notional = per_second["sec_large_sell_notional"].rolling(window, min_periods=1).sum()
        mean_interarrival = per_second["sec_mean_interarrival_ms"].rolling(window, min_periods=1).mean()
        feature_columns.update(
            {
                f"sl_median_trade_size_{window}s": per_second["sec_median_trade_size"].rolling(window, min_periods=1).median(),
                f"sl_large_trade_count_{window}s": large_count,
                f"sl_large_trade_volume_share_{window}s": _safe_divide(large_notional, total_notional),
                f"sl_large_buy_trade_count_{window}s": large_buy_count,
                f"sl_large_sell_trade_count_{window}s": large_sell_count,
                f"sl_large_buy_volume_share_{window}s": _safe_divide(large_buy_notional, total_notional),
                f"sl_large_sell_volume_share_{window}s": _safe_divide(large_sell_notional, total_notional),
                f"sl_large_trade_imbalance_{window}s": _safe_divide(large_buy_notional - large_sell_notional, large_buy_notional + large_sell_notional),
                f"sl_mean_interarrival_ms_{window}s": mean_interarrival,
                f"sl_min_interarrival_ms_{window}s": per_second["sec_min_interarrival_ms"].rolling(window, min_periods=1).min(),
                f"sl_interarrival_cv_{window}s": _safe_divide(
            per_second["sec_interarrival_std_ms"].rolling(window, min_periods=1).mean(),
            mean_interarrival,
                ),
                f"sl_trade_cluster_score_{window}s": per_second["sec_trade_cluster_score"].rolling(window, min_periods=1).mean(),
                f"sl_buy_run_length_{window}s": per_second["sec_buy_run_length"].rolling(window, min_periods=1).max(),
                f"sl_sell_run_length_{window}s": per_second["sec_sell_run_length"].rolling(window, min_periods=1).max(),
                f"sl_intrasecond_flow_concentration_{window}s": per_second["sec_intrasecond_flow_concentration"].rolling(window, min_periods=1).mean(),
                f"sl_trade_arrival_burst_flag_{window}s": (per_second["sec_trade_cluster_score"].rolling(window, min_periods=1).mean() > 0.5).astype(float),
            }
        )
    for window in (10, 30):
        feature_columns[f"sl_buy_trade_cluster_score_{window}s"] = per_second["sec_buy_trade_cluster_score"].rolling(window, min_periods=1).mean()
        feature_columns[f"sl_sell_trade_cluster_score_{window}s"] = per_second["sec_sell_trade_cluster_score"].rolling(window, min_periods=1).mean()
    feature_columns["sl_last_n_trades_buy_share"] = per_second["sec_last_trades_buy_share"]
    feature_columns["sl_last_n_trades_sell_share"] = per_second["sec_last_trades_sell_share"]
    features = pd.DataFrame(feature_columns, index=one_second_index)
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


def _kline_to_canonical_trade_seconds(kline_frame: pd.DataFrame) -> pd.DataFrame:
    kline = normalize_second_kline_frame(kline_frame)
    total_volume = kline["volume"].fillna(0.0)
    buy_volume = kline["taker_buy_base_volume"].fillna(0.0)
    sell_volume = (total_volume - buy_volume).clip(lower=0.0)
    quote_volume = kline["quote_volume"].fillna(0.0)
    buy_quote = kline["taker_buy_quote_volume"].fillna(0.0)
    sell_quote = (quote_volume - buy_quote).clip(lower=0.0)
    buy_ratio = _safe_divide(buy_volume, total_volume).fillna(0.0).clip(0.0, 1.0)
    trade_count = kline["trade_count"].fillna(0.0)
    return pd.DataFrame(
        {
            DEFAULT_TIMESTAMP_COLUMN: kline[DEFAULT_TIMESTAMP_COLUMN],
            "price": kline["close"],
            "trade_count": trade_count,
            "total_volume": total_volume,
            "dollar_volume": quote_volume,
            "taker_buy_volume": buy_volume,
            "taker_sell_volume": sell_volume,
            "taker_buy_count": trade_count * buy_ratio,
            "taker_sell_count": trade_count * (1.0 - buy_ratio),
            "signed_dollar_flow": buy_quote - sell_quote,
        }
    )


def _build_agg_trade_second_summary(
    trades: pd.DataFrame,
    one_second_index: pd.DatetimeIndex,
    *,
    large_trade_quantile: float = 0.95,
    large_trade_window_seconds: int = 300,
    cluster_threshold_ms: float = 100.0,
) -> pd.DataFrame:
    normalized = normalize_trade_frame(trades)
    events = normalized.set_index(DEFAULT_TIMESTAMP_COLUMN).sort_index()
    is_taker_buy = events["is_taker_buy"].fillna(False).astype(bool)
    notional = events["quote_quantity"].astype("float64")
    threshold = notional.rolling(f"{large_trade_window_seconds}s", min_periods=20).quantile(large_trade_quantile)
    threshold = threshold.fillna(notional.expanding(min_periods=1).quantile(large_trade_quantile))
    is_large = notional > threshold
    interarrival_ms = events.index.to_series().diff().dt.total_seconds().mul(1000.0)
    clustered = interarrival_ms < cluster_threshold_ms
    signed_large_notional = notional.where(is_taker_buy, -notional).where(is_large, 0.0)
    side_group = (is_taker_buy != is_taker_buy.shift(fill_value=is_taker_buy.iloc[0])).cumsum()
    run_length = is_taker_buy.groupby(side_group).cumcount() + 1
    second_bucket = events.index.floor("s")
    per_event = pd.DataFrame(
        {
            "trade_size": events["quantity"].astype("float64"),
            "notional": notional,
            "is_large": is_large.astype(float),
            "large_buy": (is_large & is_taker_buy).astype(float),
            "large_sell": (is_large & ~is_taker_buy).astype(float),
            "large_notional": notional.where(is_large, 0.0),
            "large_buy_notional": notional.where(is_large & is_taker_buy, 0.0),
            "large_sell_notional": notional.where(is_large & ~is_taker_buy, 0.0),
            "signed_large_notional": signed_large_notional,
            "interarrival_ms": interarrival_ms,
            "clustered": clustered.astype(float),
            "buy_clustered": (clustered & is_taker_buy).astype(float),
            "sell_clustered": (clustered & ~is_taker_buy).astype(float),
            "buy_run_length": run_length.where(is_taker_buy, 0.0).astype(float),
            "sell_run_length": run_length.where(~is_taker_buy, 0.0).astype(float),
        },
        index=events.index,
    )
    last_n_share = (
        pd.DataFrame({"second": second_bucket, "is_buy": is_taker_buy.astype(float)})
        .groupby("second")["is_buy"]
        .apply(lambda values: values.tail(10).mean())
    )
    max_notional = per_event["notional"].resample("1s").max()
    total_notional_second = per_event["notional"].resample("1s").sum()
    per_second = pd.DataFrame(
        {
            "sec_median_trade_size": per_event["trade_size"].resample("1s").median(),
            "sec_large_trade_count": per_event["is_large"].resample("1s").sum(),
            "sec_large_buy_trade_count": per_event["large_buy"].resample("1s").sum(),
            "sec_large_sell_trade_count": per_event["large_sell"].resample("1s").sum(),
            "sec_large_trade_notional": per_event["large_notional"].resample("1s").sum(),
            "sec_large_buy_notional": per_event["large_buy_notional"].resample("1s").sum(),
            "sec_large_sell_notional": per_event["large_sell_notional"].resample("1s").sum(),
            "sec_signed_large_notional": per_event["signed_large_notional"].resample("1s").sum(),
            "sec_trade_notional": total_notional_second,
            "sec_mean_interarrival_ms": per_event["interarrival_ms"].resample("1s").mean(),
            "sec_min_interarrival_ms": per_event["interarrival_ms"].resample("1s").min(),
            "sec_interarrival_std_ms": per_event["interarrival_ms"].resample("1s").std(ddof=0),
            "sec_trade_cluster_score": per_event["clustered"].resample("1s").mean(),
            "sec_buy_trade_cluster_score": per_event["buy_clustered"].resample("1s").mean(),
            "sec_sell_trade_cluster_score": per_event["sell_clustered"].resample("1s").mean(),
            "sec_buy_run_length": per_event["buy_run_length"].resample("1s").max(),
            "sec_sell_run_length": per_event["sell_run_length"].resample("1s").max(),
            "sec_intrasecond_flow_concentration": _safe_divide(max_notional, total_notional_second),
            "sec_last_trades_buy_share": last_n_share,
            "sec_last_trades_sell_share": 1.0 - last_n_share,
        },
        index=one_second_index,
    ).fillna(0.0)
    per_second["sec_large_trade_volume_share"] = _safe_divide(
        per_second["sec_large_trade_notional"],
        per_second["sec_trade_notional"],
    )
    per_second["sec_interarrival_cv"] = _safe_divide(
        per_second["sec_interarrival_std_ms"],
        per_second["sec_mean_interarrival_ms"],
    )
    return per_second


def _build_book_second_summary(book: pd.DataFrame, one_second_index: pd.DatetimeIndex) -> pd.DataFrame:
    normalized = normalize_second_book_frame(book)
    events = normalized.set_index(DEFAULT_TIMESTAMP_COLUMN).sort_index()
    per_second = events[["bid_price", "bid_qty", "ask_price", "ask_qty"]].resample("1s").last().reindex(one_second_index).ffill()
    mid = (per_second["bid_price"] + per_second["ask_price"]) / 2.0
    total_qty = (per_second["bid_qty"] + per_second["ask_qty"]).replace(0, np.nan)
    microprice = (per_second["ask_price"] * per_second["bid_qty"] + per_second["bid_price"] * per_second["ask_qty"]) / total_qty
    bid_price_delta = per_second["bid_price"].diff()
    ask_price_delta = per_second["ask_price"].diff()
    bid_qty_delta = per_second["bid_qty"].diff().fillna(0.0)
    ask_qty_delta = per_second["ask_qty"].diff().fillna(0.0)
    sec_ofi = (
        bid_qty_delta.where(bid_price_delta >= 0, -bid_qty_delta)
        - ask_qty_delta.where(ask_price_delta <= 0, -ask_qty_delta)
    ).fillna(0.0)
    return pd.DataFrame(
        {
            "sec_bid_price": per_second["bid_price"],
            "sec_bid_qty": per_second["bid_qty"],
            "sec_ask_price": per_second["ask_price"],
            "sec_ask_qty": per_second["ask_qty"],
            "sec_spread_bps": _safe_divide(per_second["ask_price"] - per_second["bid_price"], mid) * 10000.0,
            "sec_microprice": microprice,
            "sec_microprice_premium": _safe_divide(microprice - mid, mid),
            "sec_bid_ask_qty_imbalance": _safe_divide(per_second["bid_qty"] - per_second["ask_qty"], total_qty),
            "sec_quote_update_count": normalized.set_index(DEFAULT_TIMESTAMP_COLUMN)["bid_price"].resample("1s").count().reindex(one_second_index).fillna(0.0),
            "sec_ofi": sec_ofi,
            "sec_bid_depletion": (-bid_qty_delta).where((bid_qty_delta < 0) & (bid_price_delta <= 0), 0.0),
            "sec_ask_depletion": (-ask_qty_delta).where((ask_qty_delta < 0) & (ask_price_delta >= 0), 0.0),
            "sec_bid_replenishment": bid_qty_delta.where((bid_qty_delta > 0) & (bid_price_delta >= 0), 0.0),
            "sec_ask_replenishment": ask_qty_delta.where((ask_qty_delta > 0) & (ask_price_delta <= 0), 0.0),
        },
        index=one_second_index,
    )


def _add_vwap_deviation_features(store: pd.DataFrame) -> pd.DataFrame:
    if not {"sec_quote_volume", "sec_volume", "sec_close", "sl_signed_dollar_flow_30s"}.issubset(store.columns):
        return store
    additions: dict[str, pd.Series] = {}
    for window in (10, 30, 60):
        vwap = _safe_divide(
            store["sec_quote_volume"].rolling(window, min_periods=1).sum(),
            store["sec_volume"].rolling(window, min_periods=1).sum(),
        )
        additions[f"sl_vwap_{window}s"] = vwap
        additions[f"sl_price_minus_vwap_{window}s"] = store["sec_close"] - vwap
    price_minus_vwap_30s = additions["sl_price_minus_vwap_30s"]
    additions["sl_price_minus_vwap_z_300s"] = _safe_divide(
        price_minus_vwap_30s - price_minus_vwap_30s.rolling(300, min_periods=30).mean(),
        price_minus_vwap_30s.rolling(300, min_periods=30).std(ddof=0),
    )
    additions["sl_buy_flow_with_positive_vwap_deviation"] = (
        (store["sl_signed_dollar_flow_30s"] > 0) & (price_minus_vwap_30s > 0)
    ).astype(float)
    additions["sl_sell_flow_with_negative_vwap_deviation"] = (
        (store["sl_signed_dollar_flow_30s"] < 0) & (price_minus_vwap_30s < 0)
    ).astype(float)
    return _append_columns(store, additions)


def _add_shock_features(store: pd.DataFrame) -> pd.DataFrame:
    if "sec_close" not in store.columns:
        return store
    second_return = store["sec_close"].pct_change(fill_method=None).fillna(0.0)
    jump_threshold = second_return.abs().rolling(300, min_periods=30).quantile(0.95)
    positive_jump = second_return > jump_threshold
    negative_jump = second_return < -jump_threshold
    any_jump = positive_jump | negative_jump
    positions = pd.Series(np.arange(len(store), dtype="float64"), index=store.index)
    positive_position = positions.where(positive_jump).ffill()
    negative_position = positions.where(negative_jump).ffill()
    last_jump_price = store["sec_close"].where(any_jump).ffill()
    last_jump_return = second_return.where(any_jump).ffill()
    post_jump_return = store["sec_close"] / last_jump_price - 1.0
    same_direction = np.sign(post_jump_return) == np.sign(last_jump_return)
    opposite_direction = np.sign(post_jump_return) == -np.sign(last_jump_return)
    recent_jump = any_jump.rolling(30, min_periods=1).max().astype(bool)
    return _append_columns(
        store,
        {
            "sl_time_since_last_positive_jump": positions - positive_position,
            "sl_time_since_last_negative_jump": positions - negative_position,
            "sl_positive_jump_count_30s": positive_jump.astype(float).rolling(30, min_periods=1).sum(),
            "sl_negative_jump_count_30s": negative_jump.astype(float).rolling(30, min_periods=1).sum(),
            "sl_shock_continuation_flag": (recent_jump & same_direction).astype(float),
            "sl_shock_reversal_flag": (recent_jump & opposite_direction).astype(float),
            "sl_post_jump_followthrough_ratio": _safe_divide(post_jump_return.clip(lower=0.0).abs(), last_jump_return.abs()),
            "sl_post_jump_pullback_ratio": _safe_divide((-post_jump_return.clip(upper=0.0)), last_jump_return.abs()),
        },
    )


def _add_cross_source_qa(store: pd.DataFrame, kline: pd.DataFrame, agg_trades_frame: pd.DataFrame | None) -> pd.DataFrame:
    additions: dict[str, pd.Series | np.ndarray | float] = {
        "kline_gap_flag": kline[DEFAULT_TIMESTAMP_COLUMN].duplicated().astype(float).to_numpy(),
        "agg_trade_gap_flag": 0.0,
        "book_gap_flag": 0.0,
        "cross_source_volume_delta": np.nan,
        "cross_source_trade_count_delta": np.nan,
    }
    if agg_trades_frame is None or agg_trades_frame.empty:
        return _append_columns(store, additions)
    agg = normalize_trade_frame(agg_trades_frame)
    per_second = agg.set_index(DEFAULT_TIMESTAMP_COLUMN).sort_index()
    agg_summary = pd.DataFrame(
        {
            "agg_volume": per_second["quantity"].resample("1s").sum(),
            "agg_trade_count": per_second["quantity"].resample("1s").count(),
        }
    )
    joined = pd.DataFrame({DEFAULT_TIMESTAMP_COLUMN: store[DEFAULT_TIMESTAMP_COLUMN]}).merge(
        agg_summary.reset_index(),
        on=DEFAULT_TIMESTAMP_COLUMN,
        how="left",
    )
    additions.update(
        {
            "cross_source_volume_delta": (joined["agg_volume"].fillna(0.0) - store["sec_volume"]).to_numpy(),
            "cross_source_trade_count_delta": (joined["agg_trade_count"].fillna(0.0) - store["sec_trade_count"]).to_numpy(),
            "agg_trade_gap_flag": joined["agg_volume"].isna().astype(float).to_numpy(),
        }
    )
    return _append_columns(store, additions)


def _add_cross_market_features(
    store: pd.DataFrame,
    cross_market_frame: pd.DataFrame | None,
    cross_market_book_frame: pd.DataFrame | None = None,
    eth_kline_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    additions: dict[str, pd.Series] = {}
    if cross_market_frame is None or cross_market_frame.empty:
        perp_close = None
    else:
        cross = normalize_second_kline_frame(cross_market_frame)
        aligned = pd.merge_asof(
            store[[DEFAULT_TIMESTAMP_COLUMN]].sort_values(DEFAULT_TIMESTAMP_COLUMN),
            cross[[DEFAULT_TIMESTAMP_COLUMN, "close"]].rename(columns={"close": "perp_close"}).sort_values(DEFAULT_TIMESTAMP_COLUMN),
            on=DEFAULT_TIMESTAMP_COLUMN,
            direction="backward",
        )
        perp_close = aligned["perp_close"]
        spot_close = store["sec_close"]
        for window in (5, 10, 30):
            additions[f"sl_perp_return_{window}s"] = perp_close / perp_close.shift(window) - 1.0
        additions["sl_spot_minus_perp_return_10s"] = store["sl_return_10s"] - additions["sl_perp_return_10s"]
        basis = _safe_divide(perp_close - spot_close, spot_close)
        additions["sl_basis_change_10s"] = basis - basis.shift(10)
    if cross_market_book_frame is not None and not cross_market_book_frame.empty:
        perp_book = build_book_second_level_features(store[DEFAULT_TIMESTAMP_COLUMN], cross_market_book_frame)
        for column in ("sl_bid_ask_qty_imbalance", "sl_microprice_premium"):
            if column in perp_book.columns:
                additions[f"sl_perp_{column.removeprefix('sl_')}"] = perp_book[column]
        if "sl_bid_ask_qty_imbalance" in perp_book.columns:
            additions["sl_perp_book_imbalance_30s"] = perp_book["sl_bid_ask_qty_imbalance"].rolling(30, min_periods=1).mean()
        if "sl_microprice_premium" in perp_book.columns:
            additions["sl_perp_microprice_premium_10s"] = perp_book["sl_microprice_premium"].rolling(10, min_periods=1).mean()
        if "sl_perp_bid_ask_qty_imbalance" in additions and "sl_bid_ask_qty_imbalance" in store.columns:
            additions["sl_spot_minus_perp_book_imbalance"] = store["sl_bid_ask_qty_imbalance"] - additions["sl_perp_bid_ask_qty_imbalance"]
        if "sl_perp_microprice_premium" in additions and "sl_microprice_premium" in store.columns:
            additions["sl_spot_minus_perp_microprice_premium"] = store["sl_microprice_premium"] - additions["sl_perp_microprice_premium"]
    if eth_kline_frame is not None and not eth_kline_frame.empty:
        eth = normalize_second_kline_frame(eth_kline_frame)
        aligned_eth = pd.merge_asof(
            store[[DEFAULT_TIMESTAMP_COLUMN]].sort_values(DEFAULT_TIMESTAMP_COLUMN),
            eth[[DEFAULT_TIMESTAMP_COLUMN, "close"]].rename(columns={"close": "eth_close"}).sort_values(DEFAULT_TIMESTAMP_COLUMN),
            on=DEFAULT_TIMESTAMP_COLUMN,
            direction="backward",
        )
        eth_return_30s = aligned_eth["eth_close"] / aligned_eth["eth_close"].shift(30) - 1.0
        btc_return_30s = store["sl_return_30s"]
        beta = _safe_divide(
            btc_return_30s.rolling(300, min_periods=30).cov(eth_return_30s),
            eth_return_30s.rolling(300, min_periods=30).var(ddof=0),
        )
        additions["sl_btc_minus_eth_return_30s"] = btc_return_30s - eth_return_30s
        additions["sl_crypto_beta_residual_return_30s"] = btc_return_30s - beta * eth_return_30s
    return _append_columns(store, additions)


def _add_depth_features(store: pd.DataFrame, depth_frame: pd.DataFrame | None) -> pd.DataFrame:
    if depth_frame is None or depth_frame.empty:
        return store
    depth = depth_frame.copy()
    timestamp_column = _first_existing(depth.columns, (DEFAULT_TIMESTAMP_COLUMN, "event_time", "transaction_time", "date", "E", "T"))
    if timestamp_column is None:
        raise ValueError("Depth frame requires timestamp/date/event_time.")
    depth[DEFAULT_TIMESTAMP_COLUMN] = _to_datetime_utc(depth[timestamp_column])
    levels = []
    for index in range(1, 11):
        bid_qty = _first_existing(depth.columns, (f"bid_qty_{index}", f"bid_quantity_{index}", f"bid{index}_qty"))
        ask_qty = _first_existing(depth.columns, (f"ask_qty_{index}", f"ask_quantity_{index}", f"ask{index}_qty"))
        bid_price = _first_existing(depth.columns, (f"bid_price_{index}", f"bid{index}_price"))
        ask_price = _first_existing(depth.columns, (f"ask_price_{index}", f"ask{index}_price"))
        if bid_qty and ask_qty:
            levels.append((index, bid_price, bid_qty, ask_price, ask_qty))
    if not levels:
        return store
    for _, bid_price, bid_qty, ask_price, ask_qty in levels:
        depth[bid_qty] = pd.to_numeric(depth[bid_qty], errors="coerce")
        depth[ask_qty] = pd.to_numeric(depth[ask_qty], errors="coerce")
        if bid_price:
            depth[bid_price] = pd.to_numeric(depth[bid_price], errors="coerce")
        if ask_price:
            depth[ask_price] = pd.to_numeric(depth[ask_price], errors="coerce")
    per_second = depth.set_index(DEFAULT_TIMESTAMP_COLUMN).sort_index().resample("1s").last()
    aligned = pd.merge_asof(
        store[[DEFAULT_TIMESTAMP_COLUMN]].sort_values(DEFAULT_TIMESTAMP_COLUMN),
        per_second.reset_index().sort_values(DEFAULT_TIMESTAMP_COLUMN),
        on=DEFAULT_TIMESTAMP_COLUMN,
        direction="backward",
    )
    additions: dict[str, pd.Series] = {}
    for count in (5, 10):
        selected = levels[:count]
        if not selected:
            continue
        bid_depth = sum(aligned[bid_qty].fillna(0.0) for _, _, bid_qty, _, _ in selected)
        ask_depth = sum(aligned[ask_qty].fillna(0.0) for _, _, _, _, ask_qty in selected)
        additions[f"sl_bid_depth_{count}"] = bid_depth
        additions[f"sl_ask_depth_{count}"] = ask_depth
        additions[f"sl_depth_imbalance_{count}"] = _safe_divide(bid_depth - ask_depth, bid_depth + ask_depth)
        weights = np.array([1.0 / level[0] for level in selected])
        bid_weighted = sum(aligned[bid_qty].fillna(0.0) * weight for weight, (_, _, bid_qty, _, _) in zip(weights, selected))
        ask_weighted = sum(aligned[ask_qty].fillna(0.0) * weight for weight, (_, _, _, _, ask_qty) in zip(weights, selected))
        additions[f"sl_weighted_depth_imbalance_{count}"] = _safe_divide(bid_weighted - ask_weighted, bid_weighted + ask_weighted)
    if {"sl_bid_depth_5", "sl_ask_depth_5"}.issubset(additions):
        additions["sl_near_mid_depth_imbalance"] = _safe_divide(
            additions["sl_bid_depth_5"] - additions["sl_ask_depth_5"],
            additions["sl_bid_depth_5"] + additions["sl_ask_depth_5"],
        )
    price_levels = [level for level in levels if level[1] and level[3]]
    if price_levels:
        bid_wall_level = max(price_levels, key=lambda item: float(aligned[item[2]].median(skipna=True) or 0.0))
        ask_wall_level = max(price_levels, key=lambda item: float(aligned[item[4]].median(skipna=True) or 0.0))
        mid = store.get("sl_mid_price", (store["sec_high"] + store["sec_low"]) / 2.0)
        additions["sl_bid_wall_size"] = aligned[bid_wall_level[2]].fillna(0.0)
        additions["sl_ask_wall_size"] = aligned[ask_wall_level[4]].fillna(0.0)
        additions["sl_distance_to_large_bid_wall"] = _safe_divide(mid - aligned[bid_wall_level[1]], mid)
        additions["sl_distance_to_large_ask_wall"] = _safe_divide(aligned[ask_wall_level[3]] - mid, mid)
        first = price_levels[0]
        last = price_levels[-1]
        additions["sl_book_slope_bid"] = _safe_divide(aligned[first[2]] - aligned[last[2]], (aligned[first[1]] - aligned[last[1]]).abs())
        additions["sl_book_slope_ask"] = _safe_divide(aligned[first[4]] - aligned[last[4]], (aligned[first[3]] - aligned[last[3]]).abs())
        additions["sl_book_convexity"] = additions["sl_weighted_depth_imbalance_5"] - additions.get("sl_weighted_depth_imbalance_10", additions["sl_weighted_depth_imbalance_5"])
    return _append_columns(store, additions)


def build_second_level_feature_store(
    *,
    kline_frame: pd.DataFrame,
    agg_trades_frame: pd.DataFrame | None = None,
    book_frame: pd.DataFrame | None = None,
    depth_frame: pd.DataFrame | None = None,
    cross_market_frame: pd.DataFrame | None = None,
    cross_market_book_frame: pd.DataFrame | None = None,
    eth_kline_frame: pd.DataFrame | None = None,
    market: str = "BTCUSDT",
    exchange: str = "binance",
    source_manifest_id: str = "",
    large_trade_quantile: float = 0.95,
    large_trade_window_seconds: int = 300,
    feature_profile: SecondLevelFeatureProfile | dict[str, Any] | None = None,
) -> pd.DataFrame:
    kline = normalize_second_kline_frame(kline_frame)
    if kline.empty:
        raise ValueError("Cannot build second-level feature store from an empty 1s kline frame.")
    decisions = pd.DataFrame({DEFAULT_TIMESTAMP_COLUMN: kline[DEFAULT_TIMESTAMP_COLUMN]})
    canonical_trade_seconds = _kline_to_canonical_trade_seconds(kline)
    pieces = [
        build_trade_second_level_features(decisions[DEFAULT_TIMESTAMP_COLUMN], canonical_trade_seconds),
    ]
    has_agg = agg_trades_frame is not None and not agg_trades_frame.empty
    has_book = book_frame is not None and not book_frame.empty
    if has_agg:
        pieces.append(
            build_agg_trade_enrichment_features(
                decisions[DEFAULT_TIMESTAMP_COLUMN],
                agg_trades_frame,
                large_trade_quantile=large_trade_quantile,
                large_trade_window_seconds=large_trade_window_seconds,
            )
        )
    if has_book:
        pieces.append(build_book_second_level_features(decisions[DEFAULT_TIMESTAMP_COLUMN], book_frame))
    source_state_pieces: list[pd.DataFrame] = []
    one_second_index = pd.DatetimeIndex(kline[DEFAULT_TIMESTAMP_COLUMN])
    if has_agg:
        agg_second_summary = _build_agg_trade_second_summary(
            agg_trades_frame,
            one_second_index,
            large_trade_quantile=large_trade_quantile,
            large_trade_window_seconds=large_trade_window_seconds,
        )
        agg_store_columns = [
            "sec_median_trade_size",
            "sec_large_trade_count",
            "sec_large_trade_volume_share",
            "sec_large_buy_trade_count",
            "sec_large_sell_trade_count",
            "sec_buy_run_length",
            "sec_sell_run_length",
            "sec_mean_interarrival_ms",
            "sec_min_interarrival_ms",
            "sec_interarrival_cv",
            "sec_trade_cluster_score",
            "sec_intrasecond_flow_concentration",
            "sec_last_trades_buy_share",
            "sec_last_trades_sell_share",
        ]
        source_state_pieces.append(agg_second_summary[agg_store_columns].reset_index(drop=True))
    if has_book:
        source_state_pieces.append(_build_book_second_summary(book_frame, one_second_index).reset_index(drop=True))
    metadata = pd.DataFrame(
        {
            DEFAULT_TIMESTAMP_COLUMN: decisions[DEFAULT_TIMESTAMP_COLUMN].to_numpy(),
            "market": market,
            "exchange": exchange,
            "second_level_feature_version": SECOND_LEVEL_FEATURE_STORE_VERSION,
            "source_manifest_id": source_manifest_id,
            "decision_grid_name": "1s_backbone",
            "has_1s_kline": True,
            "has_agg_trade_enrichment": bool(has_agg),
            "has_book_ticker": bool(has_book),
            "sec_open": kline["open"].to_numpy(),
            "sec_high": kline["high"].to_numpy(),
            "sec_low": kline["low"].to_numpy(),
            "sec_close": kline["close"].to_numpy(),
            "sec_volume": kline["volume"].to_numpy(),
            "sec_quote_volume": kline["quote_volume"].to_numpy(),
            "sec_trade_count": kline["trade_count"].to_numpy(),
            "sec_taker_buy_base_volume": kline["taker_buy_base_volume"].to_numpy(),
            "sec_taker_buy_quote_volume": kline["taker_buy_quote_volume"].to_numpy(),
            "sec_taker_sell_base_volume": (kline["volume"] - kline["taker_buy_base_volume"]).clip(lower=0.0).to_numpy(),
            "sec_taker_sell_quote_volume": (kline["quote_volume"] - kline["taker_buy_quote_volume"]).clip(lower=0.0).to_numpy(),
            "sec_vwap": _safe_divide(kline["quote_volume"], kline["volume"]).to_numpy(),
        }
    )
    store = pd.concat([metadata, *source_state_pieces, *pieces], axis=1).replace([np.inf, -np.inf], np.nan).copy()
    interaction_columns: dict[str, pd.Series] = {}
    if "sl_signed_dollar_flow_30s" in store.columns and "sl_bid_ask_qty_imbalance" in store.columns:
        interaction_columns["sl_flow_x_book_imbalance_30s"] = store["sl_signed_dollar_flow_30s"] * store["sl_bid_ask_qty_imbalance"]
        interaction_columns["sl_flow_x_microprice_premium_30s"] = store["sl_signed_dollar_flow_30s"] * store["sl_microprice_premium"]
        interaction_columns["sl_taker_imbalance_x_spread_30s"] = store["sl_taker_imbalance_30s"] * store["sl_spread_bps"]
        interaction_columns["sl_signed_flow_x_ask_depletion_10s"] = store["sl_signed_dollar_flow_10s"] * store["sl_ask_depletion_rate_10s"]
        interaction_columns["sl_signed_flow_x_bid_replenishment_10s"] = store["sl_signed_dollar_flow_10s"] * store["sl_bid_replenishment_rate_10s"]
        tight_spread = store["sl_spread_bps"] <= store["sl_spread_bps"].rolling(300, min_periods=30).quantile(0.25)
        interaction_columns["sl_buy_pressure_with_tight_spread_flag"] = ((store["sl_taker_imbalance_30s"] > 0) & tight_spread).astype(float)
        interaction_columns["sl_sell_pressure_with_tight_spread_flag"] = ((store["sl_taker_imbalance_30s"] < 0) & tight_spread).astype(float)
    if "sl_signed_dollar_flow_30s" in store.columns and "sl_return_30s" in store.columns:
        abs_flow = store["sl_signed_dollar_flow_30s"].abs()
        abs_return = store["sl_return_30s"].abs()
        quiet_return = abs_return < abs_return.rolling(300, min_periods=30).quantile(0.25)
        interaction_columns["sl_flow_to_price_efficiency_30s"] = _safe_divide(abs_return, abs_flow)
        interaction_columns["sl_buy_absorption_30s"] = _safe_divide(store["sl_signed_dollar_flow_30s"].clip(lower=0.0), abs_return.replace(0, np.nan))
        interaction_columns["sl_sell_absorption_30s"] = _safe_divide((-store["sl_signed_dollar_flow_30s"].clip(upper=0.0)), abs_return.replace(0, np.nan))
        interaction_columns["sl_price_to_flow_divergence_30s"] = np.sign(store["sl_return_30s"]) - np.sign(store["sl_signed_dollar_flow_30s"])
        interaction_columns["sl_exhaustion_after_buy_burst_flag"] = ((store["sl_signed_dollar_flow_30s"] > 0) & quiet_return).astype(float)
        interaction_columns["sl_exhaustion_after_sell_burst_flag"] = ((store["sl_signed_dollar_flow_30s"] < 0) & quiet_return).astype(float)
    store = _append_columns(store, interaction_columns)
    store = _add_vwap_deviation_features(store)
    store = _add_shock_features(store)
    store = _add_cross_market_features(store, cross_market_frame, cross_market_book_frame, eth_kline_frame)
    store = _add_depth_features(store, depth_frame)
    store = _add_cross_source_qa(store, kline, agg_trades_frame)
    resolved_profile = (
        feature_profile
        if isinstance(feature_profile, SecondLevelFeatureProfile)
        else resolve_second_level_feature_profile(feature_profile)
    )
    store = build_second_level_pack_features(store, resolved_profile)
    return store.reset_index(drop=True)


def sample_second_level_feature_store(decision_frame: pd.DataFrame, feature_store: pd.DataFrame) -> pd.DataFrame:
    if DEFAULT_TIMESTAMP_COLUMN not in feature_store.columns:
        raise ValueError("Second-level feature store requires a timestamp column.")
    training_columns = [DEFAULT_TIMESTAMP_COLUMN, *[column for column in feature_store.columns if column.startswith("sl_")]]
    sampled = pd.merge_asof(
        decision_frame[[DEFAULT_TIMESTAMP_COLUMN]].assign(
            **{DEFAULT_TIMESTAMP_COLUMN: pd.to_datetime(decision_frame[DEFAULT_TIMESTAMP_COLUMN], utc=True)}
        ).sort_values(DEFAULT_TIMESTAMP_COLUMN),
        feature_store[training_columns].sort_values(DEFAULT_TIMESTAMP_COLUMN),
        on=DEFAULT_TIMESTAMP_COLUMN,
        direction="backward",
    )
    sl_columns = [column for column in sampled.columns if column.startswith("sl_")]
    if sl_columns:
        sampled[sl_columns] = sampled[sl_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return sampled.set_index(decision_frame.index).reset_index(drop=True)


def load_sampled_second_level_features(decision_frame: pd.DataFrame, feature_store_path: str | Path) -> pd.DataFrame:
    resolved = Path(feature_store_path)
    if resolved.is_dir():
        decisions = decision_frame[[DEFAULT_TIMESTAMP_COLUMN]].copy()
        decisions[DEFAULT_TIMESTAMP_COLUMN] = pd.to_datetime(decisions[DEFAULT_TIMESTAMP_COLUMN], utc=True)
        sampled_parts: list[pd.DataFrame] = []
        for parquet_path in sorted(resolved.rglob("*.parquet")):
            if "source_tables" in parquet_path.parts or parquet_path.name != "second_features.parquet":
                continue
            partition = pd.read_parquet(parquet_path)
            if DEFAULT_TIMESTAMP_COLUMN not in partition.columns:
                continue
            partition[DEFAULT_TIMESTAMP_COLUMN] = pd.to_datetime(partition[DEFAULT_TIMESTAMP_COLUMN], utc=True)
            start = partition[DEFAULT_TIMESTAMP_COLUMN].min()
            end = partition[DEFAULT_TIMESTAMP_COLUMN].max()
            decision_mask = (decisions[DEFAULT_TIMESTAMP_COLUMN] >= start) & (decisions[DEFAULT_TIMESTAMP_COLUMN] <= end)
            if not decision_mask.any():
                continue
            decision_slice = decisions.loc[decision_mask]
            sampled = sample_second_level_feature_store(decision_slice, partition)
            sampled.index = decision_slice.index
            sampled_parts.append(sampled)
        if not sampled_parts:
            return pd.DataFrame({DEFAULT_TIMESTAMP_COLUMN: decisions[DEFAULT_TIMESTAMP_COLUMN]}, index=decision_frame.index).reset_index(drop=True)
        combined = pd.concat(sampled_parts, axis=0).sort_index()
        missing_index = decision_frame.index.difference(combined.index)
        if len(missing_index) > 0:
            missing = pd.DataFrame({DEFAULT_TIMESTAMP_COLUMN: decisions.loc[missing_index, DEFAULT_TIMESTAMP_COLUMN]}, index=missing_index)
            combined = pd.concat([combined, missing], axis=0).sort_index()
        return combined.reset_index(drop=True)
    return sample_second_level_feature_store(decision_frame, load_second_level_frame(feature_store_path))


def build_second_level_source_tables(
    *,
    kline_frame: pd.DataFrame,
    agg_trades_frame: pd.DataFrame | None = None,
    book_frame: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    kline = normalize_second_kline_frame(kline_frame)
    one_second_index = pd.DatetimeIndex(kline[DEFAULT_TIMESTAMP_COLUMN])
    tables = {
        "kline_1s_backbone": kline.reset_index(drop=True),
    }
    if agg_trades_frame is not None and not agg_trades_frame.empty:
        tables["agg_trade_1s_event_summary"] = (
            _build_agg_trade_second_summary(agg_trades_frame, one_second_index)
            .reset_index()
            .rename(columns={"index": DEFAULT_TIMESTAMP_COLUMN})
        )
    if book_frame is not None and not book_frame.empty:
        tables["book_ticker_1s_quote_state"] = (
            _build_book_second_summary(book_frame, one_second_index)
            .reset_index()
            .rename(columns={"index": DEFAULT_TIMESTAMP_COLUMN})
        )
    return tables


def write_second_level_source_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, str]:
    resolved = Path(output_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for name, frame in tables.items():
        path = resolved / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        outputs[name] = str(path)
    return outputs


def write_second_level_feature_store(
    feature_store: pd.DataFrame,
    output_path: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    feature_store.to_parquet(resolved, index=False)
    payload = {
        "feature_version": SECOND_LEVEL_FEATURE_STORE_VERSION,
        "feature_profile": "expanded_v2",
        "feature_packs": list(DEFAULT_SECOND_LEVEL_PACKS),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(feature_store)),
        "start": str(feature_store[DEFAULT_TIMESTAMP_COLUMN].min()) if not feature_store.empty else None,
        "end": str(feature_store[DEFAULT_TIMESTAMP_COLUMN].max()) if not feature_store.empty else None,
        "feature_count": int(sum(column.startswith("sl_") for column in feature_store.columns)),
        "source_coverage_ratios": {
            column: float(feature_store[column].mean())
            for column in ("has_1s_kline", "has_agg_trade_enrichment", "has_book_ticker")
            if column in feature_store.columns
        },
        "missing_data_policy": "Backward-looking rolling features may be null until sufficient history exists; training drops incomplete samples.",
        "aggregation_rules": {
            "canonical_second_grid": "1s klines",
            "aggTrades": "event-structure enrichment only",
            "bookTicker": "top-of-book liquidity state",
            "sampler": "backward asof to decision timestamps",
        },
        "schema": list(feature_store.columns),
    }
    if manifest:
        payload.update(manifest)
    manifest_path = resolved.with_name("manifest.json")
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    qa_payload = {
        "missing_ratio_by_column": feature_store.isna().mean().round(6).to_dict(),
        "duplicate_timestamps": int(feature_store[DEFAULT_TIMESTAMP_COLUMN].duplicated().sum()),
        "gap_flag_counts": {
            column: int(feature_store[column].fillna(0).sum())
            for column in ("agg_trade_gap_flag", "book_gap_flag", "kline_gap_flag")
            if column in feature_store.columns
        },
    }
    resolved.with_name("qa_report.json").write_text(json.dumps(qa_payload, indent=2), encoding="utf-8")
    return payload


def write_partitioned_second_level_feature_store(
    *,
    kline_frame: pd.DataFrame,
    output_dir: str | Path,
    partition_frequency: str = "monthly",
    warmup_seconds: int = DEFAULT_FEATURE_STORE_WARMUP_SECONDS,
    agg_trades_frame: pd.DataFrame | None = None,
    book_frame: pd.DataFrame | None = None,
    depth_frame: pd.DataFrame | None = None,
    cross_market_frame: pd.DataFrame | None = None,
    cross_market_book_frame: pd.DataFrame | None = None,
    eth_kline_frame: pd.DataFrame | None = None,
    market: str = "BTCUSDT",
    exchange: str = "binance",
    source_manifest_id: str = "",
    large_trade_quantile: float = 0.95,
    large_trade_window_seconds: int = 300,
    feature_profile: SecondLevelFeatureProfile | dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    kline = normalize_second_kline_frame(kline_frame)
    start, end = _timestamp_bounds(kline)
    partitions: list[dict[str, Any]] = []
    total_rows = 0
    feature_count = 0
    schema: list[str] = []
    for chunk_start, chunk_end in _iter_time_partitions(start, end, partition_frequency):
        warm_start = chunk_start - pd.Timedelta(seconds=warmup_seconds)
        warm_end = chunk_end
        chunk_kline = _slice_frame_by_time(kline, warm_start, warm_end)
        if chunk_kline is None or chunk_kline.empty:
            continue
        chunk_store = build_second_level_feature_store(
            kline_frame=chunk_kline,
            agg_trades_frame=_slice_frame_by_time(agg_trades_frame, warm_start, warm_end),
            book_frame=_slice_frame_by_time(book_frame, warm_start, warm_end),
            depth_frame=_slice_frame_by_time(depth_frame, warm_start, warm_end),
            cross_market_frame=_slice_frame_by_time(cross_market_frame, warm_start, warm_end),
            cross_market_book_frame=_slice_frame_by_time(cross_market_book_frame, warm_start, warm_end),
            eth_kline_frame=_slice_frame_by_time(eth_kline_frame, warm_start, warm_end),
            market=market,
            exchange=exchange,
            source_manifest_id=source_manifest_id,
            large_trade_quantile=large_trade_quantile,
            large_trade_window_seconds=large_trade_window_seconds,
            feature_profile=feature_profile,
        )
        timestamps = pd.to_datetime(chunk_store[DEFAULT_TIMESTAMP_COLUMN], utc=True)
        chunk_store = chunk_store.loc[(timestamps >= chunk_start) & (timestamps <= chunk_end)].reset_index(drop=True)
        if chunk_store.empty:
            continue
        label = _partition_label(chunk_start, partition_frequency)
        partition_dir = output_root / f"date={label}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        partition_path = partition_dir / "second_features.parquet"
        chunk_store.to_parquet(partition_path, index=False)
        total_rows += len(chunk_store)
        feature_count = int(sum(column.startswith("sl_") for column in chunk_store.columns))
        schema = list(chunk_store.columns)
        partitions.append(
            {
                "label": label,
                "path": str(partition_path),
                "start": str(chunk_store[DEFAULT_TIMESTAMP_COLUMN].min()),
                "end": str(chunk_store[DEFAULT_TIMESTAMP_COLUMN].max()),
                "row_count": int(len(chunk_store)),
                "warmup_seconds": int(warmup_seconds),
            }
        )
    if not partitions:
        raise ValueError("Partitioned feature-store build produced no partitions.")
    payload = {
        "feature_version": SECOND_LEVEL_FEATURE_STORE_VERSION,
        "feature_profile": "expanded_v2",
        "feature_packs": list(DEFAULT_SECOND_LEVEL_PACKS),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "partitioned": True,
        "partition_frequency": partition_frequency,
        "warmup_seconds": int(warmup_seconds),
        "row_count": int(total_rows),
        "start": partitions[0]["start"],
        "end": partitions[-1]["end"],
        "feature_count": feature_count,
        "schema": schema,
        "partitions": partitions,
    }
    if manifest:
        payload.update(manifest)
    (output_root / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    qa_payload = {
        "partition_count": len(partitions),
        "row_count": int(total_rows),
        "duplicate_partition_labels": int(pd.Series([item["label"] for item in partitions]).duplicated().sum()),
    }
    (output_root / "qa_report.json").write_text(json.dumps(qa_payload, indent=2), encoding="utf-8")
    return payload
