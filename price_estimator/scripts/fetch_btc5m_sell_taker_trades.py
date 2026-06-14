#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from price_estimator_common import load_config, resolve_path


KEEP_TRADE_COLUMNS = [
    "price",
    "timestamp",
    "condition_id",
    "slug",
    "asset",
    "outcome",
    "market_start_ts",
    "market_start_dt_utc",
    "final_outcome",
]


def get_json(session: requests.Session, url: str, params: dict[str, Any] | None, timeout: int, retries: int) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(min(2**attempt, 30))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(min(2**attempt, 30))
    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to fetch {url}")


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return json.loads(value)
    return []


def norm_outcome(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"up", "yes"}:
        return "up"
    if text in {"down", "no"}:
        return "down"
    return None


def build_base_refs(config: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for split_name, key in [("train", "train_frame"), ("validation", "validation_frame")]:
        cols = ["polymarket_slug", "condition_id", "market_t0", "target", "endDate", "outcomes"]
        df = pd.read_parquet(resolve_path(config["paths"][key]), columns=cols)
        df["split_source"] = split_name
        frames.append(df)
    refs = pd.concat(frames, ignore_index=True)
    refs = refs.drop_duplicates(subset=["condition_id", "polymarket_slug"]).copy()
    refs["market_t0"] = pd.to_datetime(refs["market_t0"], utc=True)
    refs["endDate"] = pd.to_datetime(refs["endDate"], utc=True, errors="coerce")
    refs["final_outcome"] = refs["target"].map({1: "up", 0: "down"})
    refs = refs[refs["final_outcome"].isin(["up", "down"])].reset_index(drop=True)
    return refs


def enrich_one_market(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    session = requests.Session()
    slug = row["polymarket_slug"]
    url = config["api"]["gamma_market_slug_url"].format(slug=slug)
    data = get_json(
        session,
        url,
        None,
        int(config["api"]["request_timeout_seconds"]),
        int(config["api"]["max_retries"]),
    )
    clob_ids = parse_json_list(data.get("clobTokenIds"))
    outcomes = [str(x).lower() for x in parse_json_list(data.get("outcomes") or row.get("outcomes"))]
    if len(clob_ids) < 2 or len(outcomes) < 2:
        raise ValueError(f"Missing token ids/outcomes for {slug}")
    token_by_outcome = dict(zip(outcomes, map(str, clob_ids), strict=False))
    row["up_token_id"] = token_by_outcome.get("up") or token_by_outcome.get("yes")
    row["down_token_id"] = token_by_outcome.get("down") or token_by_outcome.get("no")
    if not row["up_token_id"] or not row["down_token_id"]:
        raise ValueError(f"Could not map up/down token ids for {slug}: {outcomes}")
    row["gamma_market_id"] = data.get("id")
    return row


def build_market_refs(config: dict[str, Any], workers: int, limit_markets: int | None = None) -> pd.DataFrame:
    out_path = resolve_path(config["paths"]["market_refs"])
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    refs = build_base_refs(config)
    if limit_markets:
        refs = refs.sort_values("market_t0").head(limit_markets).copy()
    if not bool(config["api"].get("enrich_market_refs", False)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat([existing, refs], ignore_index=True) if not existing.empty else refs
        combined = combined.drop_duplicates(subset=["condition_id"])
        combined.to_parquet(out_path, index=False)
        return combined
    if not existing.empty:
        existing_keys = set(existing["condition_id"].astype(str))
        refs = refs[~refs["condition_id"].astype(str).isin(existing_keys)].copy()
    rows: list[dict[str, Any]] = existing.to_dict("records") if not existing.empty else []
    if refs.empty:
        return pd.DataFrame(rows)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(enrich_one_market, row, config) for row in refs.to_dict("records")]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetch market refs"):
            rows.append(future.result())
            if len(rows) % 500 == 0:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).drop_duplicates(subset=["condition_id"]).to_parquet(out_path, index=False)
    result = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.drop_duplicates(subset=["condition_id"]).to_parquet(out_path, index=False)
    return result


def normalize_trade(trade: dict[str, Any], ref: pd.Series) -> dict[str, Any] | None:
    side = str(trade.get("side", "")).upper()
    if side != "SELL":
        return None
    asset = str(trade.get("asset") or trade.get("token") or trade.get("tokenId") or "")
    outcome = norm_outcome(trade.get("outcome"))
    if "up_token_id" in ref and outcome is None and asset == str(ref["up_token_id"]):
        outcome = "up"
    if "down_token_id" in ref and outcome is None and asset == str(ref["down_token_id"]):
        outcome = "down"
    if outcome not in {"up", "down"}:
        return None
    return {
        "price": trade.get("price"),
        "timestamp": trade.get("timestamp"),
        "condition_id": ref["condition_id"],
        "slug": ref["polymarket_slug"],
        "asset": asset,
        "outcome": outcome,
        "market_start_ts": int(pd.Timestamp(ref["market_t0"]).timestamp()),
        "market_start_dt_utc": pd.Timestamp(ref["market_t0"]).isoformat(),
        "final_outcome": ref["final_outcome"],
    }


def fetch_trades_for_market(ref_dict: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    ref = pd.Series(ref_dict)
    session = requests.Session()
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = {
            "market": ref["condition_id"],
            "limit": int(config["api"]["limit"]),
            "offset": offset,
            "takerOnly": "true",
            "side": "SELL",
        }
        data = get_json(
            session,
            config["api"]["data_trades_url"],
            params,
            int(config["api"]["request_timeout_seconds"]),
            int(config["api"]["max_retries"]),
        )
        if not isinstance(data, list):
            break
        for trade in data:
            row = normalize_trade(trade, ref)
            if row is not None:
                rows.append(row)
        if len(data) < int(config["api"]["limit"]):
            break
        offset += int(config["api"]["limit"])
        sleep_s = float(config["api"].get("sleep_seconds", 0.0))
        if sleep_s:
            time.sleep(sleep_s)
    return rows


def write_daily_parquets(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=KEEP_TRADE_COLUMNS)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["price", "timestamp", "condition_id", "outcome"])
    df["trade_time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["trade_date"] = df["trade_time"].dt.strftime("%Y-%m-%d")
    dedupe_cols = ["condition_id", "asset", "outcome", "price", "timestamp"]
    df = df.drop_duplicates(subset=dedupe_cols)
    out_dir.mkdir(parents=True, exist_ok=True)
    for date, part in df.groupby("trade_date", sort=True):
        path = out_dir / f"date={date}.parquet"
        write_part = part.drop(columns=["trade_date"])
        if path.exists():
            old = pd.read_parquet(path)
            write_part = pd.concat([old, write_part], ignore_index=True).drop_duplicates(subset=dedupe_cols)
        write_part.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="price_estimator/configs/catboost_quantile_baseline.yaml")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit-markets", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    workers = args.workers or int(config["api"]["workers"])
    refs = build_market_refs(config, workers=workers, limit_markets=args.limit_markets)
    refs = refs.sort_values("market_t0").reset_index(drop=True)
    out_dir = resolve_path(config["paths"]["trades_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / "_done_condition_ids.txt"
    done = set(progress_file.read_text(encoding="utf-8").splitlines()) if progress_file.exists() else set()
    todo = refs[~refs["condition_id"].isin(done)].copy()

    all_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_trades_for_market, row, config): row["condition_id"] for row in todo.to_dict("records")}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetch SELL taker trades"):
            condition_id = futures[future]
            rows = future.result()
            all_rows.extend(rows)
            with progress_file.open("a", encoding="utf-8") as f:
                f.write(f"{condition_id}\n")
            if len(all_rows) >= 100_000:
                write_daily_parquets(all_rows, out_dir)
                all_rows.clear()
    write_daily_parquets(all_rows, out_dir)

    files = sorted(out_dir.glob("date=*.parquet"))
    row_count = sum(len(pd.read_parquet(path, columns=["price"])) for path in files)
    print(f"Wrote {row_count:,} compact SELL/takerOnly trades across {len(files)} daily parquet files in {out_dir}")


if __name__ == "__main__":
    main()
