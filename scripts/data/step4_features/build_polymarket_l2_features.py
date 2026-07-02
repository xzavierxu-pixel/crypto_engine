from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.polymarket_l2 import (  # noqa: E402
    PolymarketL2Config,
    build_first_minute_features,
    build_trade_products,
    canonicalize_events,
    market_t0_from_slug,
    source_fingerprint,
)


RAW_COLUMNS = [
    "market_slug", "timestamp", "local_timestamp", "event_type",
    "ask_prices", "ask_sizes", "bid_prices", "bid_sizes",
    "best_ask", "best_bid", "pc_price", "pc_size", "pc_side",
    "new_tick_size", "trade_price", "trade_size", "trade_side",
    "trade_is_mirror", "winning_outcome",
]


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_config(path: Path) -> tuple[PolymarketL2Config, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = payload.get("polymarket_l2", payload)
    fields = set(PolymarketL2Config.__dataclass_fields__)
    kwargs = {key: value for key, value in section.items() if key in fields}
    for key in ("depth_levels", "rolling_windows_seconds"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    return PolymarketL2Config(**kwargs), section


def _market_date(path: Path) -> str:
    return market_t0_from_slug(path.stem).strftime("%Y-%m-%d")


def _canonical_output(events: pd.DataFrame) -> pd.DataFrame:
    # winning_outcome is retained only in canonical audit data and never passed to features.
    columns = [column for column in events.columns if column != "_source_row"]
    return events[columns]


def _process_market(task: tuple[str, str, str, PolymarketL2Config, bool]) -> dict[str, Any]:
    path_text, date, output_root_text, config, write_canonical = task
    path = Path(path_text)
    output_root = Path(output_root_text)
    schema = pq.read_schema(path)
    columns = [column for column in RAW_COLUMNS if column in schema.names]
    frame = pd.read_parquet(path, columns=columns)
    counts = {str(key): int(value) for key, value in frame["event_type"].value_counts().items()}
    canonical = canonicalize_events(frame, config)
    feature = build_first_minute_features(frame, config)
    reference, future_low = build_trade_products(frame, config)
    if write_canonical:
        destination = output_root / "canonical_events" / f"date={date}" / path.name
        _atomic_parquet(_canonical_output(canonical), destination)
    return {
        "path": str(path),
        "schema": str(schema),
        "input_rows": len(frame),
        "event_counts": counts,
        "feature": feature,
        "reference": reference,
        "future_low": future_low,
    }


def build_day(
    date: str,
    paths: list[Path],
    output_root: Path,
    config: PolymarketL2Config,
    *,
    write_canonical: bool,
    workers: int,
) -> dict[str, Any]:
    features: list[pd.DataFrame] = []
    references: list[pd.DataFrame] = []
    lows: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    event_counts: dict[str, int] = {}
    input_rows = 0
    source_schema = None
    tasks = [
        (str(path), date, str(output_root), config, write_canonical)
        for path in sorted(paths)
    ]
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    iterator = executor.map(_process_market, tasks) if executor else map(_process_market, tasks)
    for path, item in zip(sorted(paths), iterator):
        try:
            source_schema = source_schema or item["schema"]
            input_rows += item["input_rows"]
            for event_type, count in item["event_counts"].items():
                event_counts[event_type] = event_counts.get(event_type, 0) + count
            features.append(item["feature"])
            references.append(item["reference"])
            lows.append(item["future_low"])
        except Exception as exc:  # one corrupt market must not discard an otherwise complete day
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    if executor:
        executor.shutdown()

    products = {
        "first_minute_features": pd.concat(features, ignore_index=True) if features else pd.DataFrame(),
        "first_minute_price_reference": pd.concat(references, ignore_index=True) if references else pd.DataFrame(),
        "future_four_minute_lows": pd.concat(lows, ignore_index=True) if lows else pd.DataFrame(),
    }
    if not features:
        raise RuntimeError(f"all {len(paths)} markets failed for {date}: {failures[:3]}")
    for name, frame in products.items():
        if len(frame):
            if frame["market_t0"].duplicated().any():
                raise ValueError(f"duplicate market_t0 in {name} for {date}")
            frame = frame.sort_values("market_t0", kind="stable").reset_index(drop=True)
        _atomic_parquet(frame, output_root / name / f"date={date}" / "part-00000.parquet")

    feature_frame = products["first_minute_features"]
    reference_frame = products["first_minute_price_reference"]
    low_frame = products["future_four_minute_lows"]
    qa = {
        "date": date,
        "source_market_count": len(paths),
        "built_market_count": len(feature_frame),
        "failed_market_count": len(failures),
        "failures": failures,
        "max_feature_time_violation_count": int(
            (feature_frame["max_feature_event_time"] > feature_frame["feature_cutoff_time"]).sum()
        ) if len(feature_frame) else 0,
        "future_low_time_violation_count": int(sum(
            (low_frame[f"{side}_future_low_time_4m"].notna() &
             ((low_frame[f"{side}_future_low_time_4m"] <= low_frame["market_t0"] + pd.Timedelta(seconds=config.feature_window_seconds)) |
              (low_frame[f"{side}_future_low_time_4m"] > low_frame["market_end_time"]))).sum()
            for side in ("up", "down")
        )) if len(low_frame) else 0,
        "up_price_reference_coverage": float(reference_frame["up_last_trade_price_1m"].notna().mean()) if len(reference_frame) else 0.0,
        "down_price_reference_coverage": float(reference_frame["down_last_trade_price_1m"].notna().mean()) if len(reference_frame) else 0.0,
        "up_future_low_coverage": float(low_frame["up_future_low_4m"].notna().mean()) if len(low_frame) else 0.0,
        "down_future_low_coverage": float(low_frame["down_future_low_4m"].notna().mean()) if len(low_frame) else 0.0,
    }
    _atomic_json(qa, output_root / "qa" / f"date={date}.json")
    manifest = {
        "builder_version": "polymarket_l2_first_minute_v1",
        "date": date,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": asdict(config),
        "config_hash": config.config_hash,
        "source_fingerprint": source_fingerprint(paths),
        "source_schema": source_schema,
        "input_file_count": len(paths),
        "input_row_count": input_rows,
        "event_counts": event_counts,
        "output_market_count": len(feature_frame),
        "failed_markets": failures,
        "deterministic_order": "timestamp,local_timestamp,source_row",
    }
    _atomic_json(manifest, output_root / "manifests" / f"date={date}.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cutoff-safe Polymarket L2 data products")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--limit-markets", type=int)
    parser.add_argument("--eligible-markets", type=Path)
    parser.add_argument("--skip-canonical", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config, raw_config = _load_config(args.config)
    source_dir = args.source_dir or ROOT / raw_config["source_dir"]
    output_root = args.output_root or ROOT / "artifacts/data_v2/polymarket_l2"
    paths = sorted(source_dir.rglob("*.parquet"))
    if args.eligible_markets:
        eligible_slugs = set(pd.read_parquet(args.eligible_markets, columns=["market_slug"])["market_slug"].astype(str))
        paths = [path for path in paths if path.stem in eligible_slugs]
    if args.limit_markets:
        paths = paths[: args.limit_markets]
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        date = _market_date(path)
        if args.start_date and date < args.start_date:
            continue
        if args.end_date and date > args.end_date:
            continue
        grouped.setdefault(date, []).append(path)
    if not grouped:
        raise SystemExit("no source markets selected")
    manifests = []
    for date, day_paths in sorted(grouped.items()):
        manifest_path = output_root / "manifests" / f"date={date}.json"
        if args.resume and manifest_path.exists():
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            products_exist = all(
                (output_root / name / f"date={date}" / "part-00000.parquet").exists()
                for name in ("first_minute_features", "first_minute_price_reference", "future_four_minute_lows")
            )
            canonical_complete = args.skip_canonical or (
                len(list((output_root / "canonical_events" / f"date={date}").glob("*.parquet")))
                == len(day_paths)
            )
            if (
                products_exist
                and canonical_complete
                and prior.get("config_hash") == config.config_hash
                and prior.get("source_fingerprint") == source_fingerprint(day_paths)
            ):
                manifests.append(prior)
                continue
        manifests.append(
            build_day(
                date, day_paths, output_root, config,
                write_canonical=not args.skip_canonical,
                workers=args.workers,
            )
        )
    eligible = []
    for feature_file in sorted((output_root / "first_minute_features").rglob("*.parquet")):
        schema = pq.read_schema(feature_file)
        if {"market_slug", "market_t0"}.issubset(schema.names):
            eligible.append(pd.read_parquet(feature_file, columns=["market_slug", "market_t0"]))
    if not eligible:
        raise RuntimeError("build produced no eligible markets")
    eligible_frame = pd.concat(eligible, ignore_index=True).drop_duplicates("market_t0").sort_values("market_t0")
    _atomic_parquet(eligible_frame, output_root / "l2_eligible_markets.parquet")
    _atomic_json(
        {
            "builder_version": "polymarket_l2_first_minute_v1",
            "config_hash": config.config_hash,
            "day_count": len(manifests),
            "eligible_market_count": len(eligible_frame),
            "start": eligible_frame["market_t0"].min(),
            "end": eligible_frame["market_t0"].max(),
        },
        output_root / "manifests" / "build_summary.json",
    )


if __name__ == "__main__":
    main()
