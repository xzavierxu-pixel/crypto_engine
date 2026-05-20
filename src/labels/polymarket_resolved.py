from __future__ import annotations

import ast
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.core.config import Settings
from src.core.constants import DEFAULT_TARGET_COLUMN, DEFAULT_TIMESTAMP_COLUMN
from src.core.timegrid import add_grid_columns, select_grid_rows
from src.core.validation import normalize_ohlcv_frame
from src.horizons.base import HorizonSpec
from src.labels.base import LabelBuilder


GAMMA_BASE = "https://gamma-api.polymarket.com"
MARKETS_KEYSET_URL = f"{GAMMA_BASE}/markets/keyset"
MARKETS_URL = f"{GAMMA_BASE}/markets"
DEFAULT_LABEL_VERSION = "polymarket_resolved_gamma_v1"
DEFAULT_SLUG_TEMPLATE = "btc-updown-5m-{epoch}"

POLYMARKET_LABEL_AUDIT_COLUMNS = {
    "polymarket_slug",
    "polymarket_label_status",
    "source_frame_path",
    "original_target",
    "original_btc_direction_target",
    "polymarket_target",
    "label_mismatch",
    "label_mismatch_vs_btc_direction",
    "market_id",
    "condition_id",
    "question",
    "endDate",
    "closedTime",
    "closed",
    "umaResolutionStatus",
    "outcomes",
    "outcomePrices",
    "winner",
    "label_source",
    "label_store_path",
    "unresolved_or_missing_label_count",
    "fetched_at",
    "source",
}


def parse_json_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def utc_timestamp_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True)


def polymarket_slug_for_market_t0(
    market_t0: pd.Series,
    *,
    slug_template: str = DEFAULT_SLUG_TEMPLATE,
) -> pd.Series:
    timestamps = utc_timestamp_series(market_t0)
    epochs = timestamps.astype("int64") // 10**9
    return pd.Series(
        [slug_template.format(epoch=int(epoch)) for epoch in epochs],
        index=market_t0.index,
        dtype="object",
    )


def add_polymarket_slugs(
    frame: pd.DataFrame,
    *,
    slug_template: str = DEFAULT_SLUG_TEMPLATE,
) -> pd.DataFrame:
    if "market_t0" in frame.columns:
        market_t0 = frame["market_t0"]
    elif DEFAULT_TIMESTAMP_COLUMN in frame.columns:
        market_t0 = frame[DEFAULT_TIMESTAMP_COLUMN]
    else:
        raise KeyError("Polymarket slug mapping requires market_t0 or timestamp.")
    enriched = frame.copy()
    enriched["market_t0"] = utc_timestamp_series(market_t0)
    enriched["polymarket_slug"] = polymarket_slug_for_market_t0(
        enriched["market_t0"],
        slug_template=slug_template,
    )
    return enriched


def resolved_btc_up_label(market: dict[str, Any], *, win_threshold: float) -> tuple[int | None, str, str | None]:
    if not bool(market.get("closed")):
        return None, "not_closed", None
    outcomes = parse_json_list(market.get("outcomes"))
    prices = parse_json_list(market.get("outcomePrices"))
    if not outcomes or not prices or len(outcomes) != 2 or len(prices) != 2:
        return None, "missing_or_non_binary_outcomes", None
    try:
        prices_f = [float(price) for price in prices]
    except Exception:
        return None, "bad_outcome_prices", None
    max_price = max(prices_f)
    min_price = min(prices_f)
    if max_price < win_threshold or min_price > (1.0 - win_threshold):
        return None, "ambiguous_outcome_prices", None
    winner = str(outcomes[prices_f.index(max_price)]).strip()
    winner_key = winner.lower()
    if "up" in winner_key or winner_key == "yes":
        return 1, "resolved", winner
    if "down" in winner_key or winner_key == "no":
        return 0, "resolved", winner
    return None, f"unknown_winner:{winner_key}", winner


def label_frame_from_markets(
    markets: list[dict[str, Any]],
    *,
    wanted_slugs: set[str] | None = None,
    win_threshold: float = 0.99,
    label_version: str = DEFAULT_LABEL_VERSION,
    fetched_at: pd.Timestamp | None = None,
    source: str = "polymarket_gamma",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    wanted = set(wanted_slugs) if wanted_slugs is not None else None
    fetched_ts = fetched_at or pd.Timestamp.now(tz=UTC)
    for market in markets:
        slug = str(market.get("slug") or "")
        if not slug or (wanted is not None and slug not in wanted):
            continue
        target, status, winner = resolved_btc_up_label(market, win_threshold=win_threshold)
        rows.append(
            {
                "polymarket_slug": slug,
                "market_t0": market_t0_from_slug(slug),
                DEFAULT_TARGET_COLUMN: target,
                "polymarket_label_status": status,
                "market_id": market.get("id"),
                "condition_id": market.get("conditionId"),
                "question": market.get("question"),
                "endDate": market.get("endDate"),
                "closedTime": market.get("closedTime"),
                "closed": market.get("closed"),
                "umaResolutionStatus": market.get("umaResolutionStatus"),
                "outcomes": market.get("outcomes"),
                "outcomePrices": market.get("outcomePrices"),
                "winner": winner,
                "label_version": label_version,
                "fetched_at": fetched_ts,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def market_t0_from_slug(slug: str) -> pd.Timestamp | pd.NaT:
    try:
        epoch = int(str(slug).rsplit("-", 1)[-1])
    except Exception:
        return pd.NaT
    return pd.Timestamp(epoch, unit="s", tz=UTC)


def require_unique_label_slugs(label_store: pd.DataFrame) -> pd.DataFrame:
    if "polymarket_slug" not in label_store.columns:
        raise KeyError("Resolved Polymarket label store requires polymarket_slug.")
    duplicates = label_store[label_store["polymarket_slug"].duplicated(keep=False)]
    if duplicates.empty:
        return label_store
    comparable = duplicates.drop(columns=[column for column in ("fetched_at",) if column in duplicates.columns])
    inconsistent = comparable.drop_duplicates()
    if len(inconsistent) != duplicates["polymarket_slug"].nunique():
        duplicate_slugs = sorted(duplicates["polymarket_slug"].dropna().astype(str).unique())
        raise ValueError(f"Duplicate Polymarket slugs with inconsistent content: {duplicate_slugs[:10]}")
    return label_store.drop_duplicates(subset=["polymarket_slug"], keep="first").reset_index(drop=True)


def load_label_store(path: str | Path) -> pd.DataFrame:
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"Resolved Polymarket label store does not exist: {label_path}")
    store = pd.read_parquet(label_path)
    store = require_unique_label_slugs(store)
    if "market_t0" in store.columns:
        store["market_t0"] = pd.to_datetime(store["market_t0"], utc=True)
    else:
        store["market_t0"] = store["polymarket_slug"].map(market_t0_from_slug)
    if DEFAULT_TARGET_COLUMN not in store.columns and "polymarket_target" in store.columns:
        store[DEFAULT_TARGET_COLUMN] = store["polymarket_target"]
    if "polymarket_label_status" not in store.columns:
        store["polymarket_label_status"] = "resolved"
    return store


def build_btc_direction_audit_frame(df: pd.DataFrame, horizon: HorizonSpec) -> pd.DataFrame:
    labeled = add_grid_columns(df, grid_minutes=horizon.grid_minutes)
    future_close = labeled["close"].shift(-horizon.future_close_offset)
    target = (future_close >= labeled["open"]).astype("float64")
    target[future_close.isna()] = pd.NA
    target[~labeled["is_grid_t0"]] = pd.NA
    labeled["original_btc_direction_target"] = target
    return select_grid_rows(labeled, grid_minutes=horizon.grid_minutes)[
        [DEFAULT_TIMESTAMP_COLUMN, "grid_id", "original_btc_direction_target"]
    ].reset_index(drop=True)


def apply_resolved_labels(
    frame: pd.DataFrame,
    label_store: pd.DataFrame,
    *,
    unresolved_policy: str = "drop",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if unresolved_policy != "drop":
        raise ValueError("Only unresolved_policy=drop is supported for resolved Polymarket labels.")
    if frame["polymarket_slug"].duplicated().any():
        raise ValueError("Training frame has duplicate Polymarket slugs before label join.")
    labels = require_unique_label_slugs(label_store.copy())
    labels = labels.copy()
    labels["polymarket_slug"] = labels["polymarket_slug"].astype(str)
    merged = frame.merge(labels, on="polymarket_slug", how="left", validate="one_to_one", suffixes=("", "_label"))
    source_rows = len(merged)
    resolved_mask = merged["polymarket_label_status"].eq("resolved") & merged[DEFAULT_TARGET_COLUMN].notna()
    resolved = merged.loc[resolved_mask].copy()
    resolved[DEFAULT_TARGET_COLUMN] = resolved[DEFAULT_TARGET_COLUMN].astype("int8")
    if "original_btc_direction_target" in resolved.columns:
        resolved["label_mismatch_vs_btc_direction"] = (
            resolved[DEFAULT_TARGET_COLUMN].astype(float) != resolved["original_btc_direction_target"].astype(float)
        )
        resolved["label_mismatch"] = resolved["label_mismatch_vs_btc_direction"]
    report = summarize_resolved_label_join(resolved, source_rows=source_rows)
    return resolved.reset_index(drop=True), report


def summarize_resolved_label_join(frame: pd.DataFrame, *, source_rows: int) -> dict[str, Any]:
    mismatch = frame.get("label_mismatch_vs_btc_direction")
    return {
        "label_source": "polymarket_resolved",
        "label_version": str(frame["label_version"].dropna().iloc[0]) if "label_version" in frame.columns and frame["label_version"].notna().any() else DEFAULT_LABEL_VERSION,
        "resolved_label_count": int(len(frame)),
        "unresolved_or_missing_label_count": int(source_rows - len(frame)),
        "label_mismatch_count_vs_btc_direction": int(mismatch.sum()) if mismatch is not None and len(mismatch) else 0,
        "label_mismatch_rate_vs_btc_direction": float(mismatch.mean()) if mismatch is not None and len(mismatch) else None,
        "polymarket_target_mean": float(frame[DEFAULT_TARGET_COLUMN].mean()) if len(frame) else None,
    }


def _iso_utc(ts: pd.Timestamp | datetime) -> str:
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _request_with_retry(session: requests.Session, url: str, params: dict[str, Any], retries: int = 6) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=30)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(min(2**attempt, 20))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Gamma request failed: {last_error}")


def fetch_closed_markets_by_slug(slugs: list[str], *, max_workers: int = 16) -> list[dict[str, Any]]:
    def fetch_one(slug: str) -> dict[str, Any] | None:
        session = requests.Session()
        payload = _request_with_retry(session, MARKETS_URL, {"slug": slug, "closed": "true"})
        if not payload:
            return None
        return payload[0]

    markets: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_one, slug) for slug in slugs]
        for future in as_completed(futures):
            market = future.result()
            if market is not None:
                markets.append(market)
    return markets


def fetch_closed_markets_by_time_window(
    start_dt: datetime,
    end_dt: datetime,
    *,
    sleep_sec: float = 0.05,
    limit: int = 100,
) -> list[dict[str, Any]]:
    session = requests.Session()
    cursor: str | None = None
    markets: list[dict[str, Any]] = []
    while True:
        params: dict[str, Any] = {
            "closed": "true",
            "limit": limit,
            "order": "endDate",
            "ascending": "true",
            "end_date_min": _iso_utc(start_dt),
            "end_date_max": _iso_utc(end_dt),
        }
        if cursor:
            params["after_cursor"] = cursor
        payload = _request_with_retry(session, MARKETS_KEYSET_URL, params)
        batch = payload.get("markets") or []
        if not batch:
            break
        markets.extend(batch)
        cursor = payload.get("next_cursor")
        if not cursor:
            break
        time.sleep(sleep_sec)
    return markets


def fetch_closed_markets_by_time_slices(
    start_dt: datetime,
    end_dt: datetime,
    *,
    slice_minutes: int = 60,
    limit: int = 100,
    max_workers: int = 16,
) -> list[dict[str, Any]]:
    slices: list[tuple[datetime, datetime]] = []
    delta = timedelta(minutes=slice_minutes)
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + delta, end_dt)
        slices.append((chunk_start, chunk_end))
        chunk_start = chunk_end

    markets: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(fetch_closed_markets_by_time_window, chunk_start, chunk_end, limit=limit)
            for chunk_start, chunk_end in slices
        ]
        for future in as_completed(futures):
            markets.extend(future.result())
    return markets


def build_label_store_report(
    *,
    requested_slug_count: int,
    fetched_market_count: int,
    label_store: pd.DataFrame,
) -> dict[str, Any]:
    status_counts = (
        label_store["polymarket_label_status"].value_counts(dropna=False).to_dict()
        if "polymarket_label_status" in label_store.columns
        else {}
    )
    duplicate_count = int(label_store["polymarket_slug"].duplicated().sum()) if "polymarket_slug" in label_store.columns else 0
    resolved = label_store[label_store["polymarket_label_status"].eq("resolved")] if "polymarket_label_status" in label_store.columns else label_store
    return {
        "requested_slug_count": int(requested_slug_count),
        "fetched_market_count": int(fetched_market_count),
        "resolved_count": int(len(resolved)),
        "unresolved_count": int(sum(count for status, count in status_counts.items() if status != "resolved")),
        "ambiguous_count": int(status_counts.get("ambiguous_outcome_prices", 0)),
        "duplicate_count": duplicate_count,
        "target_mean": float(resolved[DEFAULT_TARGET_COLUMN].mean()) if len(resolved) else None,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
    }


class PolymarketResolvedLabelBuilder(LabelBuilder):
    name = "polymarket_resolved"

    def build(
        self,
        df: pd.DataFrame,
        settings: Settings,
        horizon: HorizonSpec,
        select_grid_only: bool | None = None,
    ) -> pd.DataFrame:
        params = horizon.label_params or {}
        label_store_path = params.get("label_store_path")
        if not label_store_path:
            raise ValueError("polymarket_resolved label builder requires label_params.label_store_path.")
        slug_template = str(params.get("slug_template", DEFAULT_SLUG_TEMPLATE))
        unresolved_policy = str(params.get("unresolved_policy", "drop"))
        label_version = str(params.get("label_version", DEFAULT_LABEL_VERSION))

        normalized = normalize_ohlcv_frame(df, timestamp_column=DEFAULT_TIMESTAMP_COLUMN, require_volume=False)
        audit = build_btc_direction_audit_frame(normalized, horizon)
        base = add_polymarket_slugs(audit, slug_template=slug_template)
        store = load_label_store(label_store_path)
        store["label_store_path"] = str(label_store_path)
        if "label_version" not in store.columns:
            store["label_version"] = label_version
        store["label_source"] = "polymarket_resolved"
        resolved, report = apply_resolved_labels(base, store, unresolved_policy=unresolved_policy)
        resolved["unresolved_or_missing_label_count"] = int(report["unresolved_or_missing_label_count"])
        resolved[DEFAULT_TIMESTAMP_COLUMN] = pd.to_datetime(resolved["market_t0"], utc=True)
        resolved["asset"] = settings.market.pair
        resolved["horizon"] = horizon.name
        resolved["label_version"] = resolved["label_version"].fillna(label_version)
        if select_grid_only is None:
            select_grid_only = settings.dataset.strict_grid_only
        if select_grid_only:
            resolved = select_grid_rows(resolved, grid_minutes=horizon.grid_minutes)
        output = resolved.reset_index(drop=True)
        output.attrs["label_metadata_report"] = report
        return output
