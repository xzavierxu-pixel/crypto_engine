from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.labels.polymarket_resolved import (
    add_polymarket_slugs,
    apply_resolved_labels,
    label_frame_from_markets,
    parse_json_list,
    resolved_btc_up_label,
)

GAMMA_BASE = "https://gamma-api.polymarket.com"
MARKETS_KEYSET_URL = f"{GAMMA_BASE}/markets/keyset"
MARKETS_URL = f"{GAMMA_BASE}/markets"
TRAINING_AUDIT_COLUMNS = {
    "source_frame_path",
    "polymarket_slug",
    "original_target",
    "label_mismatch",
    "market_id",
    "condition_id",
    "question",
    "endDate",
    "closedTime",
    "closed",
    "umaResolutionStatus",
    "outcomes",
    "outcomePrices",
    "polymarket_target",
    "polymarket_label_status",
}


def _parse_json_list(value: Any) -> list[Any] | None:
    return parse_json_list(value)


def _iso_utc(ts: pd.Timestamp | datetime) -> str:
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _request_with_retry(session: requests.Session, params: dict[str, Any], retries: int = 6) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(MARKETS_KEYSET_URL, params=params, timeout=30)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(min(2**attempt, 20))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Gamma request failed: {last_error}")


def _request_market_slug_with_retry(slug: str, retries: int = 6) -> dict[str, Any] | None:
    session = requests.Session()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(MARKETS_URL, params={"slug": slug, "closed": "true"}, timeout=30)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(min(2**attempt, 20))
                continue
            response.raise_for_status()
            payload = response.json()
            if not payload:
                return None
            return payload[0]
        except Exception as exc:
            last_error = exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Gamma slug request failed for {slug}: {last_error}")


def _fetch_closed_markets(start_dt: datetime, end_dt: datetime, *, sleep_sec: float, limit: int) -> list[dict[str, Any]]:
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
        payload = _request_with_retry(session, params)
        batch = payload.get("markets") or []
        if not batch:
            break
        markets.extend(batch)
        cursor = payload.get("next_cursor")
        if not cursor:
            break
        time.sleep(sleep_sec)
    return markets


def _fetch_closed_markets_by_slug(slugs: list[str], *, max_workers: int) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_slug = {executor.submit(_request_market_slug_with_retry, slug): slug for slug in slugs}
        for future in as_completed(future_to_slug):
            market = future.result()
            if market is not None:
                markets.append(market)
    return markets


def _fetch_closed_markets_by_date_chunks(
    start_dt: datetime,
    end_dt: datetime,
    *,
    sleep_sec: float,
    limit: int,
    chunk_days: float,
) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    chunk_start = start_dt
    delta = timedelta(days=chunk_days)
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + delta, end_dt)
        markets.extend(_fetch_closed_markets(chunk_start, chunk_end, sleep_sec=sleep_sec, limit=limit))
        chunk_start = chunk_end
    return markets


def _fetch_closed_market_slice(start_dt: datetime, end_dt: datetime, *, limit: int) -> list[dict[str, Any]]:
    session = requests.Session()
    payload = _request_with_retry(
        session,
        {
            "closed": "true",
            "limit": limit,
            "order": "endDate",
            "ascending": "true",
            "end_date_min": _iso_utc(start_dt),
            "end_date_max": _iso_utc(end_dt),
        },
    )
    return payload.get("markets") or []


def _fetch_closed_markets_by_time_slices(
    start_dt: datetime,
    end_dt: datetime,
    *,
    slice_minutes: int,
    limit: int,
    max_workers: int,
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
            executor.submit(_fetch_closed_market_slice, chunk_start, chunk_end, limit=limit)
            for chunk_start, chunk_end in slices
        ]
        for future in as_completed(futures):
            markets.extend(future.result())
    return markets


def _resolved_btc_up_label(market: dict[str, Any], *, win_threshold: float) -> tuple[int | None, str]:
    target, status, _ = resolved_btc_up_label(market, win_threshold=win_threshold)
    return target, status


def _load_frames(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["source_frame_path"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No input frames were provided.")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def _frame_with_polymarket_slugs(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = add_polymarket_slugs(frame)
    enriched["original_target"] = enriched["target"]
    return enriched


def _label_frame_from_markets(
    markets: list[dict[str, Any]],
    *,
    wanted_slugs: set[str],
    win_threshold: float,
) -> pd.DataFrame:
    labels = label_frame_from_markets(
        markets,
        wanted_slugs=wanted_slugs,
        win_threshold=win_threshold,
    )
    if "target" in labels.columns:
        labels["polymarket_target"] = labels["target"]
    return labels


def _apply_resolved_labels(frame: pd.DataFrame, label_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if label_frame.empty:
        raise RuntimeError("No matching Polymarket labels were fetched.")

    merged = frame.merge(label_frame, on="polymarket_slug", how="left", validate="many_to_one")
    source_rows = len(merged)
    merged = merged[merged["polymarket_label_status"].eq("resolved")].copy()
    merged["target"] = merged["polymarket_target"].astype("int8")
    merged["label_mismatch"] = merged["target"].astype(float) != merged["original_target"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    split_counts = merged["source_frame_path"].value_counts().to_dict() if "source_frame_path" in merged.columns else {}
    report = {
        "source_rows": int(source_rows),
        "resolved_rows": int(len(merged)),
        "missing_or_unresolved_rows": int(source_rows - len(merged)),
        "mismatch_count": int(merged["label_mismatch"].sum()),
        "mismatch_rate": float(merged["label_mismatch"].mean()) if len(merged) else None,
        "resolved_target_mean": float(merged["target"].mean()) if len(merged) else None,
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
    }
    return merged, report


def _write_resolved_splits(merged: pd.DataFrame, input_paths: list[Path], output_split_dir: Path) -> dict[str, str]:
    output_split_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for input_path in input_paths:
        source_path = str(input_path)
        split_name = input_path.stem
        if split_name not in {"development_frame", "validation_frame"}:
            if "development" in split_name:
                split_name = "development_frame"
            elif "validation" in split_name:
                split_name = "validation_frame"
        if split_name not in {"development_frame", "validation_frame"}:
            continue
        split = merged[merged["source_frame_path"].eq(source_path)].copy()
        split = split.drop(columns=[column for column in TRAINING_AUDIT_COLUMNS if column in split.columns])
        output_path = output_split_dir / f"{split_name}.parquet"
        split.to_parquet(output_path, index=False)
        outputs[split_name] = str(output_path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-frame", type=Path, action="append", required=True)
    parser.add_argument("--output-frame", type=Path, required=True)
    parser.add_argument(
        "--seed-resolved-frame",
        type=Path,
        help="Optional existing resolved frame. Its slugs are reused and only missing slugs are fetched.",
    )
    parser.add_argument(
        "--output-split-dir",
        type=Path,
        help="Optional cached split dir to write development_frame.parquet and validation_frame.parquet.",
    )
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--sleep-sec", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--fetch-mode", choices=["time-slices", "date-chunks", "slug", "date"], default="time-slices")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--chunk-days", type=float, default=1.0)
    parser.add_argument("--slice-minutes", type=int, default=60)
    parser.add_argument("--win-threshold", type=float, default=0.99)
    args = parser.parse_args()

    frame = _frame_with_polymarket_slugs(_load_frames(args.input_frame))
    market_t0 = pd.to_datetime(frame["market_t0"] if "market_t0" in frame.columns else frame["timestamp"], utc=True)

    fetch_start = market_t0.min().to_pydatetime() + timedelta(minutes=5)
    fetch_end = market_t0.max().to_pydatetime() + timedelta(minutes=5)
    wanted_slugs = sorted(set(frame["polymarket_slug"]))
    seed_frame = None
    seed_slugs: set[str] = set()
    if args.seed_resolved_frame and args.seed_resolved_frame.exists():
        seed_frame = pd.read_parquet(args.seed_resolved_frame)
        if "polymarket_slug" in seed_frame.columns:
            seed_slugs = set(seed_frame["polymarket_slug"].dropna().astype(str))
    slugs_to_fetch = sorted(set(wanted_slugs) - seed_slugs)
    if args.fetch_mode == "slug":
        markets = _fetch_closed_markets_by_slug(slugs_to_fetch, max_workers=args.max_workers)
    elif args.fetch_mode == "time-slices":
        markets = _fetch_closed_markets_by_time_slices(
            fetch_start,
            fetch_end,
            slice_minutes=args.slice_minutes,
            limit=args.limit,
            max_workers=args.max_workers,
        )
    elif args.fetch_mode == "date-chunks":
        markets = _fetch_closed_markets_by_date_chunks(
            fetch_start,
            fetch_end,
            sleep_sec=args.sleep_sec,
            limit=args.limit,
            chunk_days=args.chunk_days,
        )
    else:
        markets = _fetch_closed_markets(fetch_start, fetch_end, sleep_sec=args.sleep_sec, limit=args.limit)

    label_frame = _label_frame_from_markets(
        markets,
        wanted_slugs=set(slugs_to_fetch),
        win_threshold=args.win_threshold,
    )
    if seed_frame is not None:
        if label_frame.empty and slugs_to_fetch:
            fetched_resolved = pd.DataFrame(columns=seed_frame.columns)
        else:
            fetched_resolved, _ = _apply_resolved_labels(
                frame[frame["polymarket_slug"].isin(slugs_to_fetch)].copy(),
                label_frame,
            )
        merged = (
            pd.concat([seed_frame, fetched_resolved], ignore_index=True)
            .drop_duplicates(subset=["polymarket_slug"], keep="first")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        source_rows = len(frame)
        label_report = {
            "source_rows": int(source_rows),
            "resolved_rows": int(len(merged)),
            "missing_or_unresolved_rows": int(source_rows - len(merged)),
            "mismatch_count": int(merged["label_mismatch"].sum()),
            "mismatch_rate": float(merged["label_mismatch"].mean()) if len(merged) else None,
            "resolved_target_mean": float(merged["target"].mean()) if len(merged) else None,
            "split_counts": {
                str(key): int(value) for key, value in merged["source_frame_path"].value_counts().to_dict().items()
            },
        }
    else:
        merged, label_report = _apply_resolved_labels(frame, label_frame)

    args.output_frame.parent.mkdir(parents=True, exist_ok=True)
    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output_frame, index=False)
    audit_columns = [
        "timestamp",
        "market_t0",
        "polymarket_slug",
        "original_target",
        "target",
        "label_mismatch",
        "question",
        "outcomes",
        "outcomePrices",
        "polymarket_label_status",
    ]
    merged[[column for column in audit_columns if column in merged.columns]].to_csv(args.audit_csv, index=False)
    split_outputs = _write_resolved_splits(merged, args.input_frame, args.output_split_dir) if args.output_split_dir else {}

    report = {
        "input_frames": [str(path) for path in args.input_frame],
        "output_frame": str(args.output_frame),
        "output_split_dir": str(args.output_split_dir) if args.output_split_dir else None,
        "split_outputs": split_outputs,
        "audit_csv": str(args.audit_csv),
        **label_report,
        "fetch": {
            "start": _iso_utc(fetch_start),
            "end": _iso_utc(fetch_end),
            "market_count": int(len(markets)),
            "seed_resolved_count": int(len(seed_frame)) if seed_frame is not None else 0,
            "fetch_slug_count": int(len(slugs_to_fetch)),
            "matched_slug_count": int(label_frame["polymarket_slug"].nunique()) if "polymarket_slug" in label_frame else 0,
            "win_threshold": args.win_threshold,
        },
    }
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
