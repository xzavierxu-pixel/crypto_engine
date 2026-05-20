from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.labels.polymarket_resolved import (
    DEFAULT_LABEL_VERSION,
    add_polymarket_slugs,
    build_label_store_report,
    fetch_closed_markets_by_slug,
    fetch_closed_markets_by_time_slices,
    fetch_closed_markets_by_time_window,
    label_frame_from_markets,
    require_unique_label_slugs,
)


def _load_requested_slugs(input_frames: list[Path]) -> tuple[list[str], pd.Series]:
    frames = []
    for path in input_frames:
        frame = pd.read_parquet(path)
        frames.append(add_polymarket_slugs(frame))
    if not frames:
        raise ValueError("At least one --input-frame is required when --slug is not provided.")
    combined = pd.concat(frames, ignore_index=True)
    return sorted(combined["polymarket_slug"].dropna().astype(str).unique()), pd.to_datetime(combined["market_t0"], utc=True)


def _read_seed(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return require_unique_label_slugs(pd.read_parquet(path))


def _utc_datetime(value: str) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    return ts.tz_convert(UTC).to_pydatetime()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the canonical Polymarket resolved BTC 5m label store.")
    parser.add_argument("--input-frame", type=Path, action="append", default=[])
    parser.add_argument("--slug", action="append", default=[])
    parser.add_argument("--seed-label-store", type=Path)
    parser.add_argument(
        "--output-label-store",
        type=Path,
        default=Path("artifacts/data_v2/labels/polymarket_resolved/btc_updown_5m.parquet"),
    )
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--fetch-mode", choices=["slug", "time-window", "time-slices"], default="slug")
    parser.add_argument("--start", help="UTC start time for time-window/time-slices fetch.")
    parser.add_argument("--end", help="UTC end time for time-window/time-slices fetch.")
    parser.add_argument("--win-threshold", type=float, default=0.99)
    parser.add_argument("--label-version", default=DEFAULT_LABEL_VERSION)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sleep-sec", type=float, default=0.05)
    parser.add_argument("--slice-minutes", type=int, default=60)
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    input_slugs, market_t0 = _load_requested_slugs(args.input_frame) if args.input_frame else ([], pd.Series(dtype="datetime64[ns, UTC]"))
    requested_slugs = sorted(set(input_slugs).union(str(slug) for slug in args.slug))
    if not requested_slugs and not (args.start and args.end):
        raise ValueError("Provide --input-frame/--slug or an explicit --start/--end fetch window.")

    seed = _read_seed(args.seed_label_store or args.output_label_store)
    seed_slugs = set(seed["polymarket_slug"].dropna().astype(str)) if not seed.empty and "polymarket_slug" in seed else set()
    slugs_to_fetch = sorted(set(requested_slugs) - seed_slugs)

    if args.fetch_mode == "slug":
        markets = fetch_closed_markets_by_slug(slugs_to_fetch, max_workers=args.max_workers)
    else:
        if args.start and args.end:
            start_dt = _utc_datetime(args.start)
            end_dt = _utc_datetime(args.end)
        elif not market_t0.empty:
            start_dt = market_t0.min().to_pydatetime() + timedelta(minutes=5)
            end_dt = market_t0.max().to_pydatetime() + timedelta(minutes=5)
        else:
            raise ValueError("time-window fetch requires --start/--end or --input-frame.")
        if args.fetch_mode == "time-slices":
            markets = fetch_closed_markets_by_time_slices(
                start_dt,
                end_dt,
                slice_minutes=args.slice_minutes,
                limit=args.limit,
                max_workers=args.max_workers,
            )
        else:
            markets = fetch_closed_markets_by_time_window(
                start_dt,
                end_dt,
                sleep_sec=args.sleep_sec,
                limit=args.limit,
            )

    fetched = label_frame_from_markets(
        markets,
        wanted_slugs=set(slugs_to_fetch) if requested_slugs else None,
        win_threshold=args.win_threshold,
        label_version=args.label_version,
    )
    output = require_unique_label_slugs(pd.concat([seed, fetched], ignore_index=True)) if not seed.empty else require_unique_label_slugs(fetched)
    args.output_label_store.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output_label_store, index=False)

    report = {
        "label_store_path": str(args.output_label_store),
        "label_version": args.label_version,
        "seed_label_count": int(len(seed)),
        "fetch_slug_count": int(len(slugs_to_fetch)),
        **build_label_store_report(
            requested_slug_count=len(requested_slugs),
            fetched_market_count=len(markets),
            label_store=output,
        ),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
