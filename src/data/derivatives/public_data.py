from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests

BINANCE_FAPI_BASE_URL = "https://fapi.binance.com"
DERIBIT_BASE_URL = "https://www.deribit.com/api/v2"
BINANCE_SYMBOL = "BTCUSDT"
DERIBIT_CURRENCY = "BTC"


def _parse_utc_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _to_milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _request_json(
    session: requests.Session,
    url: str,
    params: dict[str, object],
) -> object:
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "error" in payload and payload["error"]:
        raise ValueError(f"API error from {url}: {payload['error']}")
    return payload


def _normalize_funding_records(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=["timestamp", "funding_rate", "funding_effective_time", "exchange", "symbol", "source_version"]
        )

    frame = pd.DataFrame(records)
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["fundingTime"], unit="ms", utc=True),
            "funding_rate": pd.to_numeric(frame["fundingRate"], errors="coerce"),
            "funding_effective_time": pd.to_datetime(frame["fundingTime"], unit="ms", utc=True),
            "exchange": "binance",
            "symbol": frame["symbol"].astype("string"),
            "source_version": "binance_fapi_funding_v1",
        }
    )
    return normalized.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)


def _normalize_basis_records(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["timestamp", "mark_price", "index_price", "premium_index", "exchange", "symbol", "source_version"])

    frame = pd.DataFrame(records)
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], unit="ms", utc=True),
            "mark_price": pd.to_numeric(frame["futuresPrice"], errors="coerce"),
            "index_price": pd.to_numeric(frame["indexPrice"], errors="coerce"),
            "premium_index": pd.to_numeric(frame["basisRate"], errors="coerce"),
            "exchange": "binance",
            "symbol": frame["pair"].astype("string"),
            "source_version": "binance_fapi_basis_v1",
        }
    )
    return normalized.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)


def _normalize_oi_records(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["timestamp", "open_interest", "oi_notional", "exchange", "symbol", "source_version"])

    frame = pd.DataFrame(records)
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], unit="ms", utc=True),
            "open_interest": pd.to_numeric(frame["sumOpenInterest"], errors="coerce"),
            "oi_notional": pd.to_numeric(frame["sumOpenInterestValue"], errors="coerce"),
            "exchange": "binance",
            "symbol": frame["symbol"].astype("string"),
            "source_version": "binance_fapi_open_interest_hist_v1",
        }
    )
    return normalized.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)


def _normalize_deribit_vol_rows(rows: list[list[object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "atm_iv_near", "exchange", "symbol", "source_version"])

    frame = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close"])
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True),
            "atm_iv_near": pd.to_numeric(frame["close"], errors="coerce") / 100.0,
            "exchange": "deribit",
            "symbol": DERIBIT_CURRENCY,
            "source_version": "deribit_volatility_index_v1",
        }
    )
    return normalized.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)


def _fetch_binance_funding(
    session: requests.Session,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    cursor = start_ms
    url = f"{BINANCE_FAPI_BASE_URL}/fapi/v1/fundingRate"

    while cursor <= end_ms:
        payload = _request_json(
            session,
            url,
            {
                "symbol": BINANCE_SYMBOL,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not payload:
            break
        batch = list(payload)
        records.extend(batch)
        last_timestamp = int(batch[-1]["fundingTime"])
        next_cursor = last_timestamp + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.1)

    return _normalize_funding_records(records)


def _fetch_binance_time_series(
    session: requests.Session,
    endpoint: str,
    params: dict[str, object],
    timestamp_key: str = "timestamp",
    max_pages: int = 64,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    cursor = int(params["startTime"])
    end_ms = int(params["endTime"])
    page_count = 0

    while cursor <= end_ms:
        page_params = {**params, "startTime": cursor, "endTime": end_ms}
        payload = _request_json(session, f"{BINANCE_FAPI_BASE_URL}{endpoint}", page_params)
        if not payload:
            break
        batch = list(payload)
        records.extend(batch)
        last_timestamp = int(batch[-1][timestamp_key])
        next_cursor = last_timestamp + 1
        page_count += 1
        if next_cursor <= cursor or page_count >= max_pages:
            break
        cursor = next_cursor
        time.sleep(0.1)

    return records


def _fetch_binance_basis(
    session: requests.Session,
    start_ms: int,
    end_ms: int,
    period: str,
) -> pd.DataFrame:
    records = _fetch_binance_time_series(
        session,
        "/futures/data/basis",
        {
            "pair": BINANCE_SYMBOL,
            "contractType": "PERPETUAL",
            "period": period,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 500,
        },
    )
    return _normalize_basis_records(records)


def _fetch_binance_oi(
    session: requests.Session,
    start_ms: int,
    end_ms: int,
    period: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    cursor_end = end_ms

    while cursor_end >= start_ms:
        payload = _request_json(
            session,
            f"{BINANCE_FAPI_BASE_URL}/futures/data/openInterestHist",
            {
                "symbol": BINANCE_SYMBOL,
                "period": period,
                "startTime": start_ms,
                "endTime": cursor_end,
                "limit": 500,
            },
        )
        if not payload:
            break

        batch = list(payload)
        records.extend(batch)
        earliest_timestamp = int(batch[0]["timestamp"])
        next_cursor_end = earliest_timestamp - 1
        if next_cursor_end >= cursor_end:
            break
        cursor_end = next_cursor_end
        time.sleep(0.1)

    return _normalize_oi_records(records)


def _fetch_deribit_options_proxy(
    session: requests.Session,
    start_ms: int,
    end_ms: int,
    resolution_seconds: int,
) -> pd.DataFrame:
    payload = _request_json(
        session,
        f"{DERIBIT_BASE_URL}/public/get_volatility_index_data",
        {
            "currency": DERIBIT_CURRENCY,
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
            "resolution": resolution_seconds,
        },
    )
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    return _normalize_deribit_vol_rows(result.get("data", []))
