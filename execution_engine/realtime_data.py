from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from execution_engine.config import BinanceConfig
from src.core.constants import DEFAULT_TIMESTAMP_COLUMN


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

AGG_TRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
]


class BinanceRealtimeClient:
    def __init__(self, config: BinanceConfig, session: requests.Session | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.base_url = config.base_url.rstrip("/")

    def server_time(self) -> pd.Timestamp:
        response = self.session.get(
            f"{self.base_url}/api/v3/time",
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return pd.to_datetime(response.json()["serverTime"], unit="ms", utc=True)

    def fetch_klines(
        self,
        interval: str,
        start_time: datetime | pd.Timestamp,
        end_time: datetime | pd.Timestamp | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_time).tz_convert(UTC)
        end = pd.Timestamp(end_time).tz_convert(UTC) if end_time is not None else None
        frames: list[pd.DataFrame] = []
        while True:
            params: dict[str, Any] = {
                "symbol": self.config.symbol,
                "interval": interval,
                "startTime": int(start.timestamp() * 1000),
                "limit": limit,
            }
            if end is not None:
                params["endTime"] = int(end.timestamp() * 1000)
            response = self.session.get(
                f"{self.base_url}/api/v3/klines",
                params=params,
                timeout=self.config.request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload:
                break
            frame = normalize_binance_klines(payload, require_closed=False)
            frames.append(frame)
            last_close = pd.to_datetime(frame["close_time"].iloc[-1], utc=True)
            next_start = last_close + pd.Timedelta(milliseconds=1)
            if len(payload) < limit or (end is not None and next_start > end) or next_start <= start:
                break
            start = next_start
        if not frames:
            return pd.DataFrame(columns=[DEFAULT_TIMESTAMP_COLUMN, *KLINE_COLUMNS[1:-1]])
        return pd.concat(frames, ignore_index=True).drop_duplicates(DEFAULT_TIMESTAMP_COLUMN).sort_values(
            DEFAULT_TIMESTAMP_COLUMN
        ).reset_index(drop=True)

    def fetch_recent_klines(
        self,
        interval: str,
        lookback: timedelta,
        end_time: datetime | pd.Timestamp | None = None,
        require_closed: bool | None = None,
    ) -> pd.DataFrame:
        end = pd.Timestamp(end_time or self.server_time()).tz_convert(UTC)
        start = end - lookback
        frame = self.fetch_klines(interval, start_time=start, end_time=end)
        if require_closed if require_closed is not None else self.config.require_closed_kline:
            frame = filter_closed_klines(frame, server_time=end)
        return frame

    def fetch_agg_trades(
        self,
        start_time: datetime | pd.Timestamp,
        end_time: datetime | pd.Timestamp,
        limit: int = 1000,
        from_id: int | None = None,
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_time).tz_convert(UTC)
        end = pd.Timestamp(end_time).tz_convert(UTC)
        frames: list[pd.DataFrame] = []
        next_from_id = from_id
        while start <= end:
            params: dict[str, Any] = {"symbol": self.config.symbol, "limit": limit}
            if next_from_id is None:
                params["startTime"] = int(start.timestamp() * 1000)
                params["endTime"] = int(end.timestamp() * 1000)
            else:
                params["fromId"] = int(next_from_id)
            response = self.session.get(
                f"{self.base_url}/api/v3/aggTrades",
                params=params,
                timeout=self.config.request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload:
                break
            frame = normalize_binance_agg_trades(payload)
            if next_from_id is not None:
                frame = _filter_before(frame, end, inclusive=True)
                if frame.empty:
                    break
            frames.append(frame)
            last_time = pd.to_datetime(frame["transact_time"].iloc[-1], utc=True)
            last_id = int(frame["agg_trade_id"].dropna().iloc[-1])
            next_from_id = last_id + 1
            next_start = last_time + pd.Timedelta(milliseconds=1)
            if len(payload) < limit or next_start <= start:
                break
            start = next_start
        if not frames:
            return pd.DataFrame(columns=[DEFAULT_TIMESTAMP_COLUMN, *AGG_TRADE_COLUMNS])
        return pd.concat(frames, ignore_index=True).drop_duplicates("agg_trade_id").sort_values(
            DEFAULT_TIMESTAMP_COLUMN
        ).reset_index(drop=True)

    def fetch_recent_agg_trades(
        self,
        lookback: timedelta,
        end_time: datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        end = pd.Timestamp(end_time or self.server_time()).tz_convert(UTC)
        return self.fetch_agg_trades(start_time=end - lookback, end_time=end)

    def fetch_agg_trades_from_id(
        self,
        from_id: int,
        end_time: datetime | pd.Timestamp,
        lookback_start: datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        return self.fetch_agg_trades(
            start_time=lookback_start,
            end_time=end_time,
            from_id=from_id,
        )

    def _cache_dir(self) -> Path | None:
        if not self.config.cache_path:
            return None
        path = Path(self.config.cache_path)
        return path if path.suffix == "" else path.with_suffix("")

    def _read_cache_frame(self, name: str) -> pd.DataFrame | None:
        cache_dir = self._cache_dir()
        if cache_dir is None:
            return None
        path = cache_dir / f"{name}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _write_cache_frame(self, name: str, frame: pd.DataFrame) -> None:
        cache_dir = self._cache_dir()
        if cache_dir is None:
            return
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{name}.parquet"
        tmp_path = cache_dir / f".{name}.{os.getpid()}.tmp.parquet"
        frame.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)

    @staticmethod
    def _merge_cached_frame(
        cached: pd.DataFrame | None,
        fresh: pd.DataFrame,
        lookback_start: pd.Timestamp,
        dedupe_column: str = DEFAULT_TIMESTAMP_COLUMN,
    ) -> pd.DataFrame:
        frames = [frame for frame in (cached, fresh) if frame is not None and not frame.empty]
        if not frames:
            return fresh
        merged = pd.concat(frames, ignore_index=True)
        merged[DEFAULT_TIMESTAMP_COLUMN] = pd.to_datetime(merged[DEFAULT_TIMESTAMP_COLUMN], utc=True)
        merged = merged.loc[merged[DEFAULT_TIMESTAMP_COLUMN] >= lookback_start]
        return (
            merged.drop_duplicates(dedupe_column, keep="last")
            .sort_values(DEFAULT_TIMESTAMP_COLUMN)
            .reset_index(drop=True)
        )

    @staticmethod
    def _incremental_start(cached: pd.DataFrame | None, fallback: pd.Timestamp, overlap: pd.Timedelta) -> pd.Timestamp:
        if cached is None or cached.empty:
            return fallback
        latest = pd.to_datetime(cached[DEFAULT_TIMESTAMP_COLUMN], utc=True).max()
        return max(fallback, latest - overlap)

    def fetch_runtime_frames(self, end_time: datetime | pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        lookback = timedelta(minutes=self.config.lookback_minutes)
        end = pd.Timestamp(end_time or self.server_time()).tz_convert(UTC)
        lookback_start = end - lookback
        cached_minute = self._read_cache_frame("minute")
        cached_second = self._read_cache_frame("second")
        cached_agg_trades = self._read_cache_frame("agg_trades")

        minute_start = self._incremental_start(cached_minute, lookback_start, pd.Timedelta(minutes=2))
        second_start = self._incremental_start(cached_second, lookback_start, pd.Timedelta(seconds=10))
        agg_start = self._incremental_start(cached_agg_trades, lookback_start, pd.Timedelta(seconds=10))

        one_minute_fresh = self.fetch_klines(
            self.config.one_minute_interval,
            start_time=minute_start,
            end_time=end,
        )
        one_minute_fresh = filter_closed_klines(one_minute_fresh, server_time=end)
        one_second_fresh = self.fetch_klines(
            self.config.one_second_interval,
            start_time=second_start,
            end_time=end,
        )
        one_second_fresh = filter_closed_klines(one_second_fresh, server_time=end)
        agg_trades_fresh = self.fetch_agg_trades(start_time=agg_start, end_time=end)

        one_minute = self._merge_cached_frame(cached_minute, one_minute_fresh, lookback_start)
        one_second = self._merge_cached_frame(cached_second, one_second_fresh, lookback_start)
        agg_trades = self._merge_cached_frame(cached_agg_trades, agg_trades_fresh, lookback_start, "agg_trade_id")

        self._write_cache_frame("minute", one_minute)
        self._write_cache_frame("second", one_second)
        self._write_cache_frame("agg_trades", agg_trades)
        return one_minute, one_second, agg_trades

    def _fetch_agg_trade_tail_until(
        self,
        agg_trades: pd.DataFrame,
        required_timestamp: pd.Timestamp,
        end_time: datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if agg_trades.empty or "agg_trade_id" not in agg_trades.columns:
            return agg_trades
        latest = pd.to_datetime(agg_trades[DEFAULT_TIMESTAMP_COLUMN], utc=True).max()
        if latest >= required_timestamp:
            return agg_trades
        latest_id = pd.to_numeric(agg_trades["agg_trade_id"], errors="coerce").dropna()
        if latest_id.empty:
            return agg_trades
        end = pd.Timestamp(end_time or self.server_time()).tz_convert(UTC)
        lookback_start = end - timedelta(minutes=self.config.lookback_minutes)
        fresh = self.fetch_agg_trades_from_id(
            int(latest_id.max()) + 1,
            end_time=end,
            lookback_start=lookback_start,
        )
        merged = self._merge_cached_frame(agg_trades, fresh, lookback_start, "agg_trade_id")
        self._write_cache_frame("agg_trades", merged)
        return merged

    def wait_for_closed_runtime_frames(
        self,
        end_time: datetime | pd.Timestamp | None = None,
        max_wait_seconds: float = 20.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        deadline = time.monotonic() + max_wait_seconds
        last_error: Exception | None = None
        while True:
            try:
                one_minute, one_second, agg_trades = self.fetch_runtime_frames(end_time=end_time)
                if not one_minute.empty and not one_second.empty and not agg_trades.empty:
                    return one_minute, one_second, agg_trades
            except Exception as exc:  # pragma: no cover - exercised by integration smoke only.
                last_error = exc
            if time.monotonic() >= deadline:
                if last_error is not None:
                    raise last_error
                raise RuntimeError("Timed out waiting for closed Binance klines.")
            time.sleep(0.2)

    def wait_for_signal_runtime_frames(
        self,
        signal_t0: datetime | pd.Timestamp,
        end_time: datetime | pd.Timestamp | None = None,
        max_wait_seconds: float = 20.0,
        feature_offset_minutes: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        started = time.monotonic()
        deadline = started + max_wait_seconds
        agg_deadline = started + min(max_wait_seconds, self.config.agg_trade_wait_seconds)
        last_error: Exception | None = None
        while True:
            try:
                one_minute, one_second, agg_trades = self.fetch_runtime_frames(end_time=end_time)
                target = decision_timestamp(signal_t0, feature_offset_minutes=feature_offset_minutes)
                required_second = target - pd.Timedelta(seconds=1)
                required_agg = required_second - pd.Timedelta(seconds=self.config.max_agg_trade_lag_seconds)
                if self.config.require_agg_trade_through_last_second:
                    agg_trades = self._fetch_agg_trade_tail_until(agg_trades, required_agg, end_time=end_time)
                finalized = finalize_runtime_frames_for_signal(
                    one_minute,
                    one_second,
                    agg_trades,
                    signal_t0=signal_t0,
                    feature_offset_minutes=feature_offset_minutes,
                    require_agg_trade_through_last_second=self.config.require_agg_trade_through_last_second,
                    max_agg_trade_lag_seconds=self.config.max_agg_trade_lag_seconds,
                )
                return finalized
            except Exception as exc:  # pragma: no cover - exercised by integration smoke only.
                last_error = exc
            if (
                self.config.require_agg_trade_through_last_second
                and last_error is not None
                and "agg trade" in str(last_error).lower()
                and time.monotonic() >= agg_deadline
            ):
                raise last_error
            if time.monotonic() >= deadline:
                if last_error is not None:
                    raise last_error
                raise RuntimeError("Timed out waiting for Binance runtime frames aligned to signal_t0.")
            time.sleep(0.2)


def normalize_binance_klines(payload: list[list[Any]], require_closed: bool = False) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(columns=[DEFAULT_TIMESTAMP_COLUMN, *KLINE_COLUMNS[1:-1]])
    frame = pd.DataFrame(payload, columns=KLINE_COLUMNS)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[DEFAULT_TIMESTAMP_COLUMN] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame.drop(columns=["open_time", "ignore"])
    if require_closed:
        frame = filter_closed_klines(frame, server_time=pd.Timestamp.now(tz=UTC))
    return frame.sort_values(DEFAULT_TIMESTAMP_COLUMN).reset_index(drop=True)


def normalize_binance_agg_trades(payload: list[dict[str, Any]]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(columns=[DEFAULT_TIMESTAMP_COLUMN, *AGG_TRADE_COLUMNS])
    frame = pd.DataFrame(payload).rename(
        columns={
            "a": "agg_trade_id",
            "p": "price",
            "q": "quantity",
            "f": "first_trade_id",
            "l": "last_trade_id",
            "T": "transact_time",
            "m": "is_buyer_maker",
            "M": "is_best_match",
        }
    )
    for column in ("agg_trade_id", "first_trade_id", "last_trade_id"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in ("price", "quantity"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[DEFAULT_TIMESTAMP_COLUMN] = pd.to_datetime(frame["transact_time"], unit="ms", utc=True)
    frame["transact_time"] = frame[DEFAULT_TIMESTAMP_COLUMN]
    frame["quote_quantity"] = frame["price"] * frame["quantity"]
    if "is_buyer_maker" in frame.columns:
        frame["is_buyer_maker"] = frame["is_buyer_maker"].astype(bool)
    if "is_best_match" in frame.columns:
        frame["is_best_match"] = frame["is_best_match"].astype(bool)
    return frame.sort_values(DEFAULT_TIMESTAMP_COLUMN).reset_index(drop=True)


def filter_closed_klines(frame: pd.DataFrame, server_time: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or "close_time" not in frame.columns:
        return frame
    server_ts = pd.Timestamp(server_time).tz_convert(UTC)
    closed = frame.loc[pd.to_datetime(frame["close_time"], utc=True) <= server_ts]
    return closed.reset_index(drop=True)


def decision_timestamp(
    signal_t0: datetime | pd.Timestamp,
    *,
    feature_offset_minutes: int = 0,
) -> pd.Timestamp:
    if feature_offset_minutes < 0:
        raise ValueError("feature_offset_minutes must be >= 0.")
    target = pd.Timestamp(signal_t0).tz_convert(UTC).floor("min")
    return target + pd.Timedelta(minutes=int(feature_offset_minutes))


def expected_latest_closed_minute(
    signal_t0: datetime | pd.Timestamp,
    *,
    feature_offset_minutes: int = 0,
) -> pd.Timestamp:
    target = decision_timestamp(signal_t0, feature_offset_minutes=feature_offset_minutes)
    return target - pd.Timedelta(minutes=1)


def _latest_timestamp(frame: pd.DataFrame) -> str | None:
    if frame.empty or DEFAULT_TIMESTAMP_COLUMN not in frame.columns:
        return None
    return pd.to_datetime(frame[DEFAULT_TIMESTAMP_COLUMN], utc=True).max().isoformat()


def _filter_before(frame: pd.DataFrame, timestamp: pd.Timestamp, *, inclusive: bool) -> pd.DataFrame:
    if frame.empty or DEFAULT_TIMESTAMP_COLUMN not in frame.columns:
        return frame.copy()
    timestamps = pd.to_datetime(frame[DEFAULT_TIMESTAMP_COLUMN], utc=True)
    mask = timestamps <= timestamp if inclusive else timestamps < timestamp
    return frame.loc[mask].copy().reset_index(drop=True)


def _append_synthetic_decision_minute(minute_frame: pd.DataFrame, decision_time: pd.Timestamp) -> pd.DataFrame:
    if minute_frame.empty:
        raise RuntimeError("Cannot append signal decision row to an empty 1m frame.")
    decision_row = minute_frame.iloc[[-1]].copy()
    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ):
        if column in decision_row.columns:
            decision_row[column] = float("nan")
    decision_row[DEFAULT_TIMESTAMP_COLUMN] = decision_time
    if "close_time" in decision_row.columns:
        decision_row["close_time"] = pd.NaT
    return pd.concat([minute_frame, decision_row], ignore_index=True)


def finalize_runtime_frames_for_signal(
    minute_frame: pd.DataFrame,
    second_frame: pd.DataFrame,
    agg_trades_frame: pd.DataFrame,
    signal_t0: datetime | pd.Timestamp,
    *,
    feature_offset_minutes: int = 0,
    require_agg_trade_through_last_second: bool = True,
    max_agg_trade_lag_seconds: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    market_t0 = pd.Timestamp(signal_t0).tz_convert(UTC).floor("min")
    target = decision_timestamp(market_t0, feature_offset_minutes=feature_offset_minutes)
    required_minute = expected_latest_closed_minute(
        market_t0,
        feature_offset_minutes=feature_offset_minutes,
    )
    required_second = target - pd.Timedelta(seconds=1)
    required_agg_trade = required_second - pd.Timedelta(seconds=max_agg_trade_lag_seconds)

    original_minute_latest = _latest_timestamp(minute_frame)
    original_second_latest = _latest_timestamp(second_frame)
    original_agg_latest = _latest_timestamp(agg_trades_frame)

    safe_minute = _filter_before(minute_frame, required_minute, inclusive=True)
    safe_second = _filter_before(second_frame, target, inclusive=False)
    safe_agg = _filter_before(agg_trades_frame, target, inclusive=False)

    if safe_minute.empty:
        raise RuntimeError(
            "Binance 1m frame has no closed rows available before signal_t0 "
            f"{market_t0.isoformat()} and decision_time {target.isoformat()}."
        )
    minute_timestamps = pd.to_datetime(safe_minute[DEFAULT_TIMESTAMP_COLUMN], utc=True)
    if not minute_timestamps.eq(required_minute).any():
        raise RuntimeError(
            "Binance 1m frame is not aligned to decision_time. "
            f"Required closed minute {required_minute.isoformat()}, "
            f"latest available {_latest_timestamp(safe_minute)}."
        )
    if safe_second.empty:
        raise RuntimeError(
            "Binance 1s frame has no safe rows before decision_time "
            f"{target.isoformat()}."
        )
    second_latest = pd.to_datetime(safe_second[DEFAULT_TIMESTAMP_COLUMN], utc=True).max()
    if second_latest < required_second:
        raise RuntimeError(
            "Binance 1s frame is not complete through the last pre-signal second. "
            f"Required {required_second.isoformat()}, latest available {second_latest.isoformat()}."
        )
    agg_latest = None if safe_agg.empty else pd.to_datetime(safe_agg[DEFAULT_TIMESTAMP_COLUMN], utc=True).max()
    agg_lag_seconds = None if agg_latest is None else (required_second - agg_latest).total_seconds()
    if require_agg_trade_through_last_second:
        if agg_latest is None:
            raise RuntimeError(
                "Binance agg trade frame has no safe rows before decision_time "
                f"{target.isoformat()}."
            )
        if agg_latest < required_agg_trade:
            raise RuntimeError(
                "Binance agg trade frame is not complete through the required pre-signal time. "
                f"Required at least {required_agg_trade.isoformat()}, "
                f"latest available {agg_latest.isoformat()}."
            )

    finalized_minute = _append_synthetic_decision_minute(safe_minute, target)
    alignment = {
        "signal_t0": market_t0.isoformat(),
        "market_t0": market_t0.isoformat(),
        "decision_time": target.isoformat(),
        "decision_alignment_mode": "delayed_feature_offset" if feature_offset_minutes else "exact_signal_t0",
        "feature_offset_minutes": int(feature_offset_minutes),
        "row_policy": (
            f"delayed_{int(feature_offset_minutes)}m_synthetic_decision_row"
            if int(feature_offset_minutes) > 0
            else "exact_signal_t0_with_synthetic_decision_row"
        ),
        "feature_timestamp": target.isoformat(),
        "market_window_start": market_t0.isoformat(),
        "market_window_end": (market_t0 + pd.Timedelta(minutes=5)).isoformat(),
        "required_latest_closed_minute": required_minute.isoformat(),
        "required_latest_closed_second": required_second.isoformat(),
        "required_latest_agg_trade": required_agg_trade.isoformat(),
        "prewarm_base_until": (required_minute - pd.Timedelta(minutes=1)).isoformat(),
        "minute_latest_before_finalize": original_minute_latest,
        "second_latest_before_finalize": original_second_latest,
        "agg_trade_latest_before_finalize": original_agg_latest,
        "minute_latest": _latest_timestamp(safe_minute),
        "second_latest": _latest_timestamp(safe_second),
        "agg_trade_latest": _latest_timestamp(safe_agg),
        "agg_trade_lag_seconds": agg_lag_seconds,
        "max_agg_trade_lag_seconds": float(max_agg_trade_lag_seconds),
        "post_signal_second_rows_dropped": max(len(second_frame) - len(safe_second), 0),
        "post_signal_agg_trade_rows_dropped": max(len(agg_trades_frame) - len(safe_agg), 0),
        "synthetic_decision_row": True,
    }
    return finalized_minute, safe_second, safe_agg, alignment
