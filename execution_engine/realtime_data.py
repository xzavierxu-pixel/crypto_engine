from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
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
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_time).tz_convert(UTC)
        end = pd.Timestamp(end_time).tz_convert(UTC)
        frames: list[pd.DataFrame] = []
        while start <= end:
            params: dict[str, Any] = {
                "symbol": self.config.symbol,
                "startTime": int(start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
                "limit": limit,
            }
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
            frames.append(frame)
            last_time = pd.to_datetime(frame["transact_time"].iloc[-1], utc=True)
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

    def fetch_runtime_frames(self, end_time: datetime | pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        lookback = timedelta(minutes=self.config.lookback_minutes)
        end = pd.Timestamp(end_time or self.server_time()).tz_convert(UTC)
        one_minute = self.fetch_recent_klines(
            self.config.one_minute_interval,
            lookback=lookback,
            end_time=end,
        )
        one_second = self.fetch_recent_klines(
            self.config.one_second_interval,
            lookback=lookback,
            end_time=end,
        )
        agg_trades = self.fetch_recent_agg_trades(lookback=lookback, end_time=end)
        return one_minute, one_second, agg_trades

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
