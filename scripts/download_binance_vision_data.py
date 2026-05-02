from __future__ import annotations

import argparse
import calendar
import io
import sys
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BINANCE_VISION_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
BINANCE_VISION_DAILY_BASE_URL = "https://data.binance.vision/data/spot/daily/klines"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


@dataclass(frozen=True)
class MonthSpec:
    year: int
    month: int

    @property
    def ym(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def _parse_month(value: str) -> MonthSpec:
    year, month = value.split("-", maxsplit=1)
    return MonthSpec(year=int(year), month=int(month))


def _iter_months(start: MonthSpec, end: MonthSpec) -> list[MonthSpec]:
    months: list[MonthSpec] = []
    year = start.year
    month = start.month
    while (year, month) <= (end.year, end.month):
        months.append(MonthSpec(year=year, month=month))
        month += 1
        if month > 12:
            year += 1
            month = 1
    return months


def _pair_to_binance_symbol(pair: str) -> str:
    return pair.replace("/", "").replace(":", "")


def _build_month_url(symbol: str, timeframe: str, month: MonthSpec) -> str:
    filename = f"{symbol}-{timeframe}-{month.ym}.zip"
    return f"{BINANCE_VISION_BASE_URL}/{symbol}/{timeframe}/{filename}"


def _build_day_url(symbol: str, timeframe: str, day_value: date) -> str:
    day_str = day_value.isoformat()
    filename = f"{symbol}-{timeframe}-{day_str}.zip"
    return f"{BINANCE_VISION_DAILY_BASE_URL}/{symbol}/{timeframe}/{filename}"


def _download_zip_frame(session: requests.Session, url: str) -> pd.DataFrame:
    response = session.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_members = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_members:
            raise ValueError(f"No CSV file found in archive: {url}")
        with archive.open(csv_members[0]) as handle:
            frame = pd.read_csv(handle, header=None, names=KLINE_COLUMNS)

    frame = frame[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]
    ].copy()
    frame["date"] = pd.to_numeric(frame["open_time"], errors="raise").astype("int64")

    # Official Binance archive uses microseconds for spot data from 2025-01-01 onward.
    if frame["date"].max() > 10**13:
        frame["date"] = (frame["date"] // 1000).astype("int64")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")

    return frame[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]
    ]


def _download_month_frame(session: requests.Session, symbol: str, timeframe: str, month: MonthSpec) -> pd.DataFrame:
    return _download_zip_frame(session, _build_month_url(symbol, timeframe, month))


def _iter_days_for_month(month: MonthSpec, start_date: date | None, end_date: date | None) -> list[date]:
    last_day = calendar.monthrange(month.year, month.month)[1]
    month_start = date(month.year, month.month, 1)
    month_end = date(month.year, month.month, last_day)
    effective_start = max(month_start, start_date) if start_date else month_start
    effective_end = min(month_end, end_date) if end_date else month_end
    if effective_start > effective_end:
        return []
    return [effective_start.fromordinal(ordinal) for ordinal in range(effective_start.toordinal(), effective_end.toordinal() + 1)]


def _download_daily_frames(
    session: requests.Session,
    symbol: str,
    timeframe: str,
    month: MonthSpec,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    day_frames = []
    for day_value in _iter_days_for_month(month, start_date=start_date, end_date=end_date):
        try:
            day_frames.append(_download_zip_frame(session, _build_day_url(symbol, timeframe, day_value)))
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise
            continue
    if not day_frames:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    return pd.concat(day_frames, ignore_index=True)


def _filter_timerange(frame: pd.DataFrame, start_date: date | None, end_date: date | None) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame["date"], unit="ms", utc=True)
    mask = pd.Series(True, index=frame.index)
    if start_date is not None:
        mask &= timestamps >= pd.Timestamp(start_date, tz="UTC")
    if end_date is not None:
        mask &= timestamps < pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    return frame.loc[mask].reset_index(drop=True)


def _save_freqtrade_feather(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    freqtrade_columns = ["date", "open", "high", "low", "close", "volume"]
    frame[freqtrade_columns].to_feather(output_path, compression="lz4", compression_level=9)


def _save_parquet_copy(frame: pd.DataFrame, output_path: Path) -> None:
    parquet_frame = frame.copy()
    parquet_frame["timestamp"] = pd.to_datetime(parquet_frame["date"], unit="ms", utc=True)
    parquet_frame = parquet_frame.rename(
        columns={
            "quote_asset_volume": "quote_volume",
            "number_of_trades": "trade_count",
            "taker_buy_base_asset_volume": "taker_buy_base_volume",
            "taker_buy_quote_asset_volume": "taker_buy_quote_volume",
        }
    )
    parquet_frame = parquet_frame[
        [
            "timestamp",
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
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_frame.to_parquet(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance Vision monthly spot klines.")
    parser.add_argument("--pair", default="BTC/USDT", help="Trading pair, e.g. BTC/USDT.")
    parser.add_argument("--timeframe", default="1m", help="Binance kline interval.")
    parser.add_argument("--start-month", required=True, help="Start month in YYYY-MM.")
    parser.add_argument("--end-month", required=True, help="End month in YYYY-MM.")
    parser.add_argument("--start-date", help="Optional inclusive UTC date filter in YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Optional inclusive UTC date filter in YYYY-MM-DD.")
    parser.add_argument("--data-root", default="artifacts/data_v2", help="Unified data root for new outputs.")
    parser.add_argument(
        "--freqtrade-output",
        default=None,
        help="Output feather path in Freqtrade format.",
    )
    parser.add_argument(
        "--parquet-output",
        default=None,
        help="Optional parquet copy for shared training pipeline.",
    )
    args = parser.parse_args()

    start_month = _parse_month(args.start_month)
    end_month = _parse_month(args.end_month)
    symbol = _pair_to_binance_symbol(args.pair)
    market_dir = symbol
    timeframe = args.timeframe
    freqtrade_output = args.freqtrade_output or str(
        Path(args.data_root) / "normalized" / "binance" / "spot" / market_dir / f"{symbol}-{timeframe}.feather"
    )
    parquet_output = args.parquet_output or str(
        Path(args.data_root) / "normalized" / "binance" / "spot" / market_dir / f"{symbol}-{timeframe}.parquet"
    )
    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    end_date = date.fromisoformat(args.end_date) if args.end_date else None

    session = requests.Session()
    month_frames = []
    for month in _iter_months(start_month, end_month):
        try:
            month_frames.append(_download_month_frame(session, symbol=symbol, timeframe=args.timeframe, month=month))
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise
            month_frames.append(
                _download_daily_frames(
                    session,
                    symbol=symbol,
                    timeframe=args.timeframe,
                    month=month,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
    combined = pd.concat(month_frames, ignore_index=True).drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)
    combined = _filter_timerange(combined, start_date=start_date, end_date=end_date)

    _save_freqtrade_feather(combined, Path(freqtrade_output))
    if parquet_output:
        _save_parquet_copy(combined, Path(parquet_output))

    print(f"rows={len(combined)}")
    print(f"freqtrade_output={Path(freqtrade_output).resolve()}")
    if parquet_output:
        print(f"parquet_output={Path(parquet_output).resolve()}")


if __name__ == "__main__":
    main()
