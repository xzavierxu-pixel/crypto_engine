from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.constants import DEFAULT_TIMESTAMP_COLUMN


PREOPEN_TRADE_FEATURE_PREFIX = "pm_preopen_"


def _to_datetime_utc(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        median = numeric.dropna().abs().median()
        unit = "ms" if median > 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def load_polymarket_trade_frame(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if resolved.is_dir():
        parquet_files = sorted(resolved.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under Polymarket trade path: {resolved}")
        return pd.concat((pd.read_parquet(file) for file in parquet_files), ignore_index=True)
    if resolved.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    if resolved.suffix.lower() == ".csv":
        return pd.read_csv(resolved)
    raise ValueError(f"Unsupported Polymarket trade input format: {resolved.suffix}")


def normalize_polymarket_trade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"price", "outcome", "market_start_ts"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Polymarket trade frame missing required columns: {missing}")
    if "trade_time" not in frame.columns and "timestamp" not in frame.columns:
        raise ValueError("Polymarket trade frame requires trade_time or timestamp.")

    normalized = frame.copy()
    normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
    normalized["outcome"] = normalized["outcome"].astype(str).str.lower()
    normalized["market_start"] = pd.to_datetime(
        pd.to_numeric(normalized["market_start_ts"], errors="coerce"),
        unit="s",
        utc=True,
        errors="coerce",
    )
    trade_time_column = "trade_time" if "trade_time" in normalized.columns else "timestamp"
    normalized["trade_time"] = _to_datetime_utc(normalized[trade_time_column])
    normalized = normalized.loc[normalized["outcome"].isin(["up", "down"])]
    return normalized.dropna(subset=["price", "market_start", "trade_time"]).sort_values("trade_time")


def _side_aggregates(side: pd.DataFrame, side_name: str) -> pd.Series:
    prices = pd.to_numeric(side["price"], errors="coerce").dropna()
    if prices.empty:
        return pd.Series(
            {
                f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_trade_count": 0.0,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_mean": np.nan,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_std": 0.0,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_min": np.nan,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_max": np.nan,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_first": np.nan,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_last": np.nan,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_change": 0.0,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_seconds_since_last": np.nan,
            }
        )
    ordered = side.sort_values("trade_time")
    market_start = ordered["market_start"].iloc[0]
    seconds_since_last = (market_start - ordered["trade_time"].iloc[-1]).total_seconds()
    return pd.Series(
        {
            f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_trade_count": float(len(prices)),
            f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_mean": float(prices.mean()),
            f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_std": float(prices.std(ddof=0)) if len(prices) > 1 else 0.0,
            f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_min": float(prices.min()),
            f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_max": float(prices.max()),
            f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_first": float(ordered["price"].iloc[0]),
            f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_last": float(ordered["price"].iloc[-1]),
            f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_price_change": float(ordered["price"].iloc[-1] - ordered["price"].iloc[0]),
            f"{PREOPEN_TRADE_FEATURE_PREFIX}{side_name}_seconds_since_last": float(seconds_since_last),
        }
    )


def build_preopen_trade_feature_frame(
    trades: pd.DataFrame,
    *,
    preopen_window_seconds: int = 60,
) -> pd.DataFrame:
    if preopen_window_seconds <= 0:
        raise ValueError("preopen_window_seconds must be > 0.")
    normalized = normalize_polymarket_trade_frame(trades)
    if normalized.empty:
        return pd.DataFrame(columns=[DEFAULT_TIMESTAMP_COLUMN])

    seconds_to_open = (normalized["market_start"] - normalized["trade_time"]).dt.total_seconds()
    window = normalized.loc[(seconds_to_open > 0) & (seconds_to_open <= float(preopen_window_seconds))].copy()
    if window.empty:
        return pd.DataFrame(columns=[DEFAULT_TIMESTAMP_COLUMN])

    rows: list[dict[str, float | pd.Timestamp]] = []
    for market_start, group in window.groupby("market_start", sort=True):
        up = _side_aggregates(group.loc[group["outcome"] == "up"], "up")
        down = _side_aggregates(group.loc[group["outcome"] == "down"], "down")
        row = {**up.to_dict(), **down.to_dict()}
        up_count = float(row[f"{PREOPEN_TRADE_FEATURE_PREFIX}up_trade_count"])
        down_count = float(row[f"{PREOPEN_TRADE_FEATURE_PREFIX}down_trade_count"])
        total_count = up_count + down_count
        up_last = row[f"{PREOPEN_TRADE_FEATURE_PREFIX}up_price_last"]
        down_last = row[f"{PREOPEN_TRADE_FEATURE_PREFIX}down_price_last"]
        up_mean = row[f"{PREOPEN_TRADE_FEATURE_PREFIX}up_price_mean"]
        down_mean = row[f"{PREOPEN_TRADE_FEATURE_PREFIX}down_price_mean"]
        row.update(
            {
                DEFAULT_TIMESTAMP_COLUMN: market_start,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}has_trade": 1.0,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}trade_count": float(total_count),
                f"{PREOPEN_TRADE_FEATURE_PREFIX}up_trade_share": float(up_count / total_count) if total_count else 0.0,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}down_trade_share": float(down_count / total_count) if total_count else 0.0,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}trade_count_imbalance": (
                    float((up_count - down_count) / total_count) if total_count else 0.0
                ),
                f"{PREOPEN_TRADE_FEATURE_PREFIX}last_price_gap": float(up_last - down_last)
                if pd.notna(up_last) and pd.notna(down_last)
                else 0.0,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}mean_price_gap": float(up_mean - down_mean)
                if pd.notna(up_mean) and pd.notna(down_mean)
                else 0.0,
                f"{PREOPEN_TRADE_FEATURE_PREFIX}last_price_sum_minus_one": float(up_last + down_last - 1.0)
                if pd.notna(up_last) and pd.notna(down_last)
                else 0.0,
            }
        )
        rows.append(row)

    features = pd.DataFrame.from_records(rows)
    feature_columns = [column for column in features.columns if column.startswith(PREOPEN_TRADE_FEATURE_PREFIX)]
    features[feature_columns] = features[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features.sort_values(DEFAULT_TIMESTAMP_COLUMN).reset_index(drop=True)
