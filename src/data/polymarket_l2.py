from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


L2_FEATURE_PREFIX = "pm_l2_1m_"
FORBIDDEN_FEATURE_TOKENS = (
    "future_low",
    "future_four_minute",
    "winning_outcome",
    "market_resolved",
    "target",
    "correct",
    "winner",
)
_SLUG_EPOCH = re.compile(r"-(?P<epoch>\d{10})$")


@dataclass(frozen=True)
class PolymarketL2Config:
    market_duration_seconds: int = 300
    feature_window_seconds: int = 60
    availability_lag_ms: int = 0
    future_low_start_inclusive: bool = False
    future_low_end_inclusive: bool = True
    primary_token_side: str = "UP"
    book_semantics: str = "primary_token_complement"
    require_explicit_trade_mirror: bool = True
    depth_levels: tuple[int, ...] = (1, 2, 3, 5, 10)
    rolling_windows_seconds: tuple[int, ...] = (5, 15, 30, 60)
    safe_divide_epsilon: float = 1.0e-9

    def __post_init__(self) -> None:
        if self.primary_token_side not in {"UP", "DOWN"}:
            raise ValueError("primary_token_side must be explicitly UP or DOWN")
        if not 0 < self.feature_window_seconds < self.market_duration_seconds:
            raise ValueError("feature window must be inside the market window")
        if self.availability_lag_ms < 0:
            raise ValueError("availability_lag_ms cannot be negative")
        if self.book_semantics != "primary_token_complement":
            raise ValueError("book_semantics must be explicitly primary_token_complement")
        if not self.depth_levels or min(self.depth_levels) <= 0:
            raise ValueError("depth_levels must contain positive levels")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


def market_t0_from_slug(slug: str) -> pd.Timestamp:
    match = _SLUG_EPOCH.search(str(slug))
    if match is None:
        raise ValueError(f"market slug has no terminal epoch: {slug!r}")
    return pd.to_datetime(int(match.group("epoch")), unit="s", utc=True)


def market_times(slug: str, config: PolymarketL2Config) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    t0 = market_t0_from_slug(slug)
    cutoff = t0 + pd.Timedelta(seconds=config.feature_window_seconds) - pd.Timedelta(
        milliseconds=config.availability_lag_ms
    )
    end = t0 + pd.Timedelta(seconds=config.market_duration_seconds)
    return t0, cutoff, end


def stable_event_order(events: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "local_timestamp"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"missing event ordering columns: {sorted(missing)}")
    ordered = events.copy()
    ordered["_source_row"] = range(len(ordered))
    for column in ("timestamp", "local_timestamp"):
        ordered[column] = pd.to_datetime(ordered[column], utc=True)
    return ordered.sort_values(
        ["timestamp", "local_timestamp", "_source_row"], kind="stable"
    ).reset_index(drop=True)


def canonical_trade_side(is_mirror: object, config: PolymarketL2Config) -> str | None:
    if pd.isna(is_mirror):
        return None
    primary = config.primary_token_side
    if bool(is_mirror):
        return "DOWN" if primary == "UP" else "UP"
    return primary


def canonical_trade_price(price: object, is_mirror: object) -> float | None:
    """Recover the token-local price from PMData's primary-token-normalized stream."""
    if pd.isna(price) or pd.isna(is_mirror):
        return None
    value = float(price)
    return 1.0 - value if bool(is_mirror) else value


def canonicalize_events(events: pd.DataFrame, config: PolymarketL2Config) -> pd.DataFrame:
    ordered = stable_event_order(events)
    if "market_slug" not in ordered or ordered["market_slug"].nunique(dropna=False) != 1:
        raise ValueError("canonicalize_events requires exactly one market_slug")
    slug = str(ordered["market_slug"].iloc[0])
    t0, cutoff, end = market_times(slug, config)
    ordered["market_t0"] = t0
    ordered["feature_cutoff_time"] = cutoff
    ordered["market_end_time"] = end
    mirror = ordered.get("trade_is_mirror", pd.Series(index=ordered.index, dtype="boolean"))
    trade_rows = ordered["event_type"].eq("last_trade_price")
    if config.require_explicit_trade_mirror and mirror.loc[trade_rows].isna().any():
        raise ValueError(
            "last_trade_price rows lack trade_is_mirror; UP/DOWN mapping is not reproducible"
        )
    ordered["canonical_side"] = [canonical_trade_side(value, config) for value in mirror]
    prices = ordered.get("trade_price", pd.Series(index=ordered.index, dtype="float64"))
    ordered["canonical_trade_price"] = [
        canonical_trade_price(price, is_mirror) for price, is_mirror in zip(prices, mirror)
    ]
    return ordered


def split_event_windows(
    events: pd.DataFrame, config: PolymarketL2Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = canonicalize_events(events, config)
    t0 = canonical["market_t0"].iloc[0]
    cutoff = canonical["feature_cutoff_time"].iloc[0]
    end = canonical["market_end_time"].iloc[0]
    feature = canonical.loc[canonical["timestamp"].between(t0, cutoff, inclusive="both")].copy()
    left = canonical["timestamp"].ge(cutoff) if config.future_low_start_inclusive else canonical["timestamp"].gt(cutoff)
    right = canonical["timestamp"].le(end) if config.future_low_end_inclusive else canonical["timestamp"].lt(end)
    future = canonical.loc[left & right].copy()
    return feature, future


def build_trade_products(events: pd.DataFrame, config: PolymarketL2Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_events, future_events = split_event_windows(events, config)
    slug = str(feature_events["market_slug"].iloc[0]) if len(feature_events) else str(events["market_slug"].iloc[0])
    t0, cutoff, end = market_times(slug, config)
    feature_trades = feature_events.loc[
        feature_events["event_type"].eq("last_trade_price")
        & feature_events["canonical_trade_price"].notna()
        & feature_events["canonical_side"].notna()
    ]
    future_trades = future_events.loc[
        future_events["event_type"].eq("last_trade_price")
        & future_events["canonical_trade_price"].notna()
        & future_events["canonical_side"].notna()
    ]
    refs: dict[str, object] = {
        "market_slug": slug,
        "market_t0": t0,
        "feature_cutoff_time": cutoff,
        "max_feature_event_time": feature_events["timestamp"].max() if len(feature_events) else pd.NaT,
    }
    lows: dict[str, object] = {"market_slug": slug, "market_t0": t0, "market_end_time": end}
    for side in ("UP", "DOWN"):
        side_key = side.lower()
        prior = feature_trades.loc[feature_trades["canonical_side"].eq(side)]
        refs[f"{side_key}_last_trade_price_1m"] = prior["canonical_trade_price"].iloc[-1] if len(prior) else pd.NA
        refs[f"{side_key}_last_trade_time_1m"] = prior["timestamp"].iloc[-1] if len(prior) else pd.NaT
        later = future_trades.loc[future_trades["canonical_side"].eq(side)]
        if len(later):
            minimum = later["canonical_trade_price"].min()
            first_low = later.loc[later["canonical_trade_price"].eq(minimum)].iloc[0]
            lows[f"{side_key}_future_low_4m"] = minimum
            lows[f"{side_key}_future_low_time_4m"] = first_low["timestamp"]
            lows[f"{side_key}_future_trade_count_4m"] = len(later)
        else:
            lows[f"{side_key}_future_low_4m"] = pd.NA
            lows[f"{side_key}_future_low_time_4m"] = pd.NaT
            lows[f"{side_key}_future_trade_count_4m"] = 0
    return pd.DataFrame([refs]), pd.DataFrame([lows])


def _safe_divide(numerator: float, denominator: float, epsilon: float) -> float:
    return float(numerator / denominator) if abs(denominator) > epsilon else 0.0


def _book_from_snapshot(row: object) -> tuple[dict[float, float], dict[float, float]]:
    def values(name: str) -> list[float]:
        value = getattr(row, name, None) if not isinstance(row, pd.Series) else row.get(name)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        return list(value)

    bids = {
        float(price): float(size)
        for price, size in zip(values("bid_prices"), values("bid_sizes"))
        if pd.notna(price) and pd.notna(size) and float(size) > 0
    }
    asks = {
        float(price): float(size)
        for price, size in zip(values("ask_prices"), values("ask_sizes"))
        if pd.notna(price) and pd.notna(size) and float(size) > 0
    }
    return bids, asks


def _complement_book(
    bids: Mapping[float, float], asks: Mapping[float, float]
) -> tuple[dict[float, float], dict[float, float]]:
    return (
        {round(1.0 - price, 10): size for price, size in asks.items()},
        {round(1.0 - price, 10): size for price, size in bids.items()},
    )


def _book_stats(
    bids: Mapping[float, float], asks: Mapping[float, float], config: PolymarketL2Config
) -> dict[str, float]:
    bid_levels = sorted(bids.items(), reverse=True)
    ask_levels = sorted(asks.items())
    best_bid, bid_size = bid_levels[0] if bid_levels else (np.nan, 0.0)
    best_ask, ask_size = ask_levels[0] if ask_levels else (np.nan, 0.0)
    both = np.isfinite(best_bid) and np.isfinite(best_ask)
    mid = (best_bid + best_ask) / 2.0 if both else np.nan
    spread = best_ask - best_bid if both else np.nan
    microprice = (
        _safe_divide(best_ask * bid_size + best_bid * ask_size, bid_size + ask_size, config.safe_divide_epsilon)
        if both
        else np.nan
    )
    stats: dict[str, float] = {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
        "relative_spread": _safe_divide(spread, mid, config.safe_divide_epsilon) if both else np.nan,
        "best_bid_size": bid_size,
        "best_ask_size": ask_size,
        "top_imbalance": _safe_divide(bid_size - ask_size, bid_size + ask_size, config.safe_divide_epsilon),
        "microprice": microprice,
        "microprice_deviation": microprice - mid if both else np.nan,
    }
    for depth in config.depth_levels:
        bid_depth = sum(size for _, size in bid_levels[:depth])
        ask_depth = sum(size for _, size in ask_levels[:depth])
        stats[f"bid_depth_{depth}"] = bid_depth
        stats[f"ask_depth_{depth}"] = ask_depth
        stats[f"depth_imbalance_{depth}"] = _safe_divide(
            bid_depth - ask_depth, bid_depth + ask_depth, config.safe_divide_epsilon
        )
    return stats


def _linear_slope(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    x = np.arange(len(clean), dtype=float)
    return float(np.polyfit(x, clean.to_numpy(), 1)[0])


def build_first_minute_features(events: pd.DataFrame, config: PolymarketL2Config) -> pd.DataFrame:
    """Build cutoff-safe features from one market's ordered event stream.

    Raw book/price-change rows describe the configured primary token. The opposite
    token book is derived by the explicitly configured binary complement mapping.
    No event after ``feature_cutoff_time`` is inspected by this function.
    """
    feature_events, _ = split_event_windows(events, config)
    slug = str(events["market_slug"].iloc[0])
    t0, cutoff, _ = market_times(slug, config)
    prefix = L2_FEATURE_PREFIX
    out: dict[str, object] = {
        "market_slug": slug,
        "market_t0": t0,
        "feature_cutoff_time": cutoff,
        "max_feature_event_time": feature_events["timestamp"].max() if len(feature_events) else pd.NaT,
    }
    out[prefix + "available"] = float(len(feature_events) > 0)
    duration = max(config.feature_window_seconds - config.availability_lag_ms / 1000.0, config.safe_divide_epsilon)
    for event_type in ("book", "price_change", "last_trade_price", "tick_size_change"):
        count = int(feature_events["event_type"].eq(event_type).sum())
        out[prefix + event_type + "_count"] = float(count)
        out[prefix + event_type + "_rate"] = count / duration
    out[prefix + "active_second_count"] = float(feature_events["timestamp"].dt.floor("s").nunique())
    if len(feature_events):
        out[prefix + "first_event_seconds"] = (feature_events["timestamp"].iloc[0] - t0).total_seconds()
        out[prefix + "last_event_seconds"] = (cutoff - feature_events["timestamp"].iloc[-1]).total_seconds()
        latency = (feature_events["local_timestamp"] - feature_events["timestamp"]).dt.total_seconds() * 1000.0
        out[prefix + "latency_mean_ms"] = float(latency.mean())
        out[prefix + "latency_p50_ms"] = float(latency.quantile(0.50))
        out[prefix + "latency_p95_ms"] = float(latency.quantile(0.95))
        out[prefix + "latency_max_ms"] = float(latency.max())
        out[prefix + "timestamp_reversal_count"] = float((feature_events["timestamp"].diff().dt.total_seconds() < 0).sum())
        out[prefix + "duplicate_event_count"] = float(
            feature_events.duplicated(subset=["timestamp", "event_type"], keep=False).sum()
        )

    primary_bids: dict[float, float] = {}
    primary_asks: dict[float, float] = {}
    book_history: list[dict[str, object]] = []
    add_volume = cancel_volume = 0.0
    bid_updates = ask_updates = 0
    for row in feature_events.itertuples(index=False):
        event_type = row.event_type
        if event_type == "book":
            primary_bids, primary_asks = _book_from_snapshot(row)
        elif event_type == "price_change" and pd.notna(row.pc_price) and pd.notna(row.pc_size):
            side = str(row.pc_side or "").upper()
            target = primary_bids if side == "BUY" else primary_asks if side == "SELL" else None
            if target is not None:
                price, new_size = float(row.pc_price), float(row.pc_size)
                old_size = target.get(price, 0.0)
                delta = new_size - old_size
                if delta >= 0:
                    add_volume += delta
                else:
                    cancel_volume += -delta
                if side == "BUY":
                    bid_updates += 1
                else:
                    ask_updates += 1
                if new_size > 0:
                    target[price] = new_size
                else:
                    target.pop(price, None)
        if event_type in {"book", "price_change"} and (primary_bids or primary_asks):
            stats = _book_stats(primary_bids, primary_asks, config)
            stats["timestamp"] = row.timestamp
            book_history.append(stats)

    side_books = {config.primary_token_side: (primary_bids, primary_asks)}
    opposite = "DOWN" if config.primary_token_side == "UP" else "UP"
    side_books[opposite] = _complement_book(primary_bids, primary_asks)
    for side in ("UP", "DOWN"):
        key = side.lower()
        stats = _book_stats(*side_books[side], config)
        for name, value in stats.items():
            out[f"{prefix}{key}_{name}"] = value

    history = pd.DataFrame(book_history)
    if len(history):
        for window in config.rolling_windows_seconds:
            recent = history.loc[history["timestamp"] >= cutoff - pd.Timedelta(seconds=window)]
            mid = recent["mid"].dropna()
            out[f"{prefix}up_mid_return_{window}s"] = (
                _safe_divide(float(mid.iloc[-1] - mid.iloc[0]), float(mid.iloc[0]), config.safe_divide_epsilon)
                if len(mid) >= 2
                else 0.0
            )
            out[f"{prefix}up_mid_slope_{window}s"] = _linear_slope(mid)
            out[f"{prefix}up_mid_volatility_{window}s"] = float(mid.pct_change().std(ddof=0)) if len(mid) >= 2 else 0.0
            out[f"{prefix}up_ofi_{window}s"] = float(recent["top_imbalance"].mean()) if len(recent) else 0.0
            # Binary complement makes DOWN returns/slopes the signed counterpart.
            down_mid = 1.0 - mid
            out[f"{prefix}down_mid_return_{window}s"] = (
                _safe_divide(float(down_mid.iloc[-1] - down_mid.iloc[0]), float(down_mid.iloc[0]), config.safe_divide_epsilon)
                if len(down_mid) >= 2
                else 0.0
            )
            out[f"{prefix}down_mid_slope_{window}s"] = _linear_slope(down_mid)
            out[f"{prefix}down_mid_volatility_{window}s"] = float(down_mid.pct_change().std(ddof=0)) if len(down_mid) >= 2 else 0.0
            out[f"{prefix}down_ofi_{window}s"] = -out[f"{prefix}up_ofi_{window}s"]
        out[prefix + "up_mid_high"] = float(history["mid"].max())
        out[prefix + "up_mid_low"] = float(history["mid"].min())
        out[prefix + "spread_widen_count"] = float((history["spread"].diff() > 0).sum())
        out[prefix + "spread_narrow_count"] = float((history["spread"].diff() < 0).sum())
    out[prefix + "book_add_volume"] = add_volume
    out[prefix + "book_cancel_volume"] = cancel_volume
    out[prefix + "book_add_cancel_ratio"] = _safe_divide(add_volume, cancel_volume, config.safe_divide_epsilon)
    out[prefix + "bid_update_count"] = float(bid_updates)
    out[prefix + "ask_update_count"] = float(ask_updates)
    out[prefix + "order_flow_imbalance"] = _safe_divide(
        bid_updates - ask_updates, bid_updates + ask_updates, config.safe_divide_epsilon
    )

    trades = feature_events.loc[
        feature_events["event_type"].eq("last_trade_price")
        & feature_events["canonical_trade_price"].notna()
        & feature_events["canonical_side"].notna()
    ].copy()
    trade_sizes = trades.get("trade_size", pd.Series(0.0, index=trades.index))
    trades["trade_size"] = pd.to_numeric(trade_sizes, errors="coerce").fillna(0.0)
    for side in ("UP", "DOWN"):
        key = side.lower()
        side_trades = trades.loc[trades["canonical_side"].eq(side)]
        for window in config.rolling_windows_seconds:
            recent = side_trades.loc[side_trades["timestamp"] >= cutoff - pd.Timedelta(seconds=window)]
            prices = recent["canonical_trade_price"].astype(float)
            sizes = recent["trade_size"].astype(float)
            out[f"{prefix}{key}_trade_count_{window}s"] = float(len(recent))
            out[f"{prefix}{key}_trade_volume_{window}s"] = float(sizes.sum())
            out[f"{prefix}{key}_trade_vwap_{window}s"] = (
                _safe_divide(float((prices * sizes).sum()), float(sizes.sum()), config.safe_divide_epsilon)
                if len(recent)
                else np.nan
            )
            out[f"{prefix}{key}_trade_price_range_{window}s"] = float(prices.max() - prices.min()) if len(prices) else 0.0
        out[f"{prefix}{key}_has_trade"] = float(len(side_trades) > 0)
        out[f"{prefix}{key}_last_trade"] = float(side_trades["canonical_trade_price"].iloc[-1]) if len(side_trades) else np.nan

    up = {name: out.get(f"{prefix}up_{name}", np.nan) for name in ("mid", "best_bid", "best_ask", "spread", "last_trade")}
    down = {name: out.get(f"{prefix}down_{name}", np.nan) for name in ("mid", "best_bid", "best_ask", "spread", "last_trade")}
    out[prefix + "mid_complement_deviation"] = up["mid"] + down["mid"] - 1.0
    out[prefix + "last_complement_deviation"] = up["last_trade"] + down["last_trade"] - 1.0
    out[prefix + "buyable_pair_cost"] = up["best_ask"] + down["best_ask"] - 1.0
    out[prefix + "sellable_pair_value"] = up["best_bid"] + down["best_bid"] - 1.0
    out[prefix + "spread_difference"] = up["spread"] - down["spread"]
    feature_columns = [column for column in out if column.startswith(prefix)]
    assert_feature_schema_safe(feature_columns)
    return pd.DataFrame([out])


def assert_feature_schema_safe(columns: Iterable[str]) -> None:
    violations = sorted(
        column for column in columns if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    if violations:
        raise ValueError(f"forbidden future/label feature columns: {violations}")


def join_l2_features(frame: pd.DataFrame, features: pd.DataFrame, *, key: str = "market_t0") -> pd.DataFrame:
    assert_feature_schema_safe(features.columns)
    if features[key].duplicated().any():
        raise ValueError(f"duplicate L2 feature key: {key}")
    before = len(frame)
    result = frame.merge(features, on=key, how="left", validate="many_to_one")
    if len(result) != before:
        raise AssertionError("L2 feature join changed row count")
    return result


def load_first_minute_feature_frame(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    files = sorted(resolved.rglob("*.parquet")) if resolved.is_dir() else [resolved]
    if not files:
        raise FileNotFoundError(f"no L2 feature parquet files under {resolved}")
    frames = [pd.read_parquet(file) for file in files]
    combined = pd.concat(frames, ignore_index=True)
    required = {"market_t0", "feature_cutoff_time", "max_feature_event_time"}
    missing = required.difference(combined.columns)
    if missing:
        raise ValueError(f"L2 feature store missing columns: {sorted(missing)}")
    if combined["market_t0"].duplicated().any():
        raise ValueError("duplicate market_t0 in L2 feature store")
    if (combined["max_feature_event_time"] > combined["feature_cutoff_time"]).any():
        raise ValueError("L2 feature store contains post-cutoff events")
    feature_columns = [column for column in combined if column.startswith(L2_FEATURE_PREFIX)]
    assert_feature_schema_safe(feature_columns)
    return combined[["market_t0", *feature_columns]].rename(columns={"market_t0": "timestamp"})


def source_fingerprint(paths: Iterable[Path]) -> str:
    records: list[Mapping[str, object]] = []
    for path in sorted((Path(path) for path in paths), key=lambda value: value.as_posix()):
        stat = path.stat()
        records.append({"path": path.as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()
