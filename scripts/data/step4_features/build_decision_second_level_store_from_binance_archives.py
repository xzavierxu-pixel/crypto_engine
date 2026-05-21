from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings  # noqa: E402
from src.core.constants import DEFAULT_TIMESTAMP_COLUMN  # noqa: E402
from src.data.binance_public.normalizer import AGG_TRADES_COLUMNS, KLINE_COLUMNS  # noqa: E402
from src.data.second_level_features import (  # noqa: E402
    SECOND_LEVEL_FEATURE_STORE_VERSION,
    build_second_level_feature_store,
    resolve_second_level_feature_profile,
    sample_second_level_feature_store,
)


BINANCE_DATA_ROOT = "https://data.binance.vision/data/spot/daily"


def _download_zip_csv(session: requests.Session, url: str) -> bytes:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_members = [member for member in archive.infolist() if not member.is_dir() and member.filename.endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(f"Expected exactly one CSV in {url}, found {len(csv_members)}.")
        with archive.open(csv_members[0]) as handle:
            return handle.read()


def _read_csv_bytes(payload: bytes, columns: list[str]) -> pd.DataFrame:
    first_line = payload.splitlines()[0].decode("utf-8", errors="ignore") if payload else ""
    first_token = first_line.split(",", 1)[0].strip()
    has_header = any(character.isalpha() for character in first_token)
    return pd.read_csv(
        io.BytesIO(payload),
        header=0 if has_header else None,
        names=None if has_header else columns,
        na_filter=False,
        memory_map=False,
    )


def _numeric_epoch_to_utc(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    max_value = int(numeric.max()) if numeric.notna().any() else 0
    unit = "us" if max_value >= 10**15 else "ms"
    return pd.to_datetime(numeric, unit=unit, utc=True)


def _fetch_daily_kline(session: requests.Session, symbol: str, day: pd.Timestamp) -> pd.DataFrame:
    day_label = day.strftime("%Y-%m-%d")
    url = f"{BINANCE_DATA_ROOT}/klines/{symbol}/1s/{symbol}-1s-{day_label}.zip"
    frame = _read_csv_bytes(_download_zip_csv(session, url), KLINE_COLUMNS)
    frame[DEFAULT_TIMESTAMP_COLUMN] = _numeric_epoch_to_utc(frame["open_time"])
    rename_map = {
        "count": "trade_count",
        "taker_buy_volume": "taker_buy_base_volume",
    }
    frame = frame.rename(columns=rename_map)
    return frame[
        [
            DEFAULT_TIMESTAMP_COLUMN,
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
    ].copy()


def _fetch_daily_agg_trades(session: requests.Session, symbol: str, day: pd.Timestamp) -> pd.DataFrame:
    day_label = day.strftime("%Y-%m-%d")
    url = f"{BINANCE_DATA_ROOT}/aggTrades/{symbol}/{symbol}-aggTrades-{day_label}.zip"
    frame = _read_csv_bytes(_download_zip_csv(session, url), AGG_TRADES_COLUMNS)
    frame[DEFAULT_TIMESTAMP_COLUMN] = _numeric_epoch_to_utc(frame["transact_time"])
    return frame[
        [
            DEFAULT_TIMESTAMP_COLUMN,
            "agg_trade_id",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "transact_time",
            "is_buyer_maker",
            "is_best_match",
        ]
    ].copy()


def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame[DEFAULT_TIMESTAMP_COLUMN], utc=True)
    return frame.loc[(timestamps >= start) & (timestamps <= end)].copy()


def _write_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_decision_store(
    *,
    decision_input: Path,
    output_dir: Path,
    config: Path,
    symbol: str = "BTCUSDT",
    warmup_seconds: int = 900,
    start_date: str | None = None,
    end_date: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    settings = load_settings(config)
    feature_profile = resolve_second_level_feature_profile(settings.second_level.get_profile_payload())
    decision_source = pd.read_parquet(decision_input, columns=[DEFAULT_TIMESTAMP_COLUMN])
    decisions = pd.DataFrame(
        {
            DEFAULT_TIMESTAMP_COLUMN: pd.to_datetime(decision_source[DEFAULT_TIMESTAMP_COLUMN], utc=True).dropna().sort_values().drop_duplicates()
        }
    )
    if start_date:
        decisions = decisions.loc[decisions[DEFAULT_TIMESTAMP_COLUMN] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        decisions = decisions.loc[decisions[DEFAULT_TIMESTAMP_COLUMN] <= pd.Timestamp(end_date, tz="UTC")]
    if decisions.empty:
        raise ValueError("No decision timestamps selected.")

    output_dir.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(
        decisions[DEFAULT_TIMESTAMP_COLUMN].min().floor("D"),
        decisions[DEFAULT_TIMESTAMP_COLUMN].max().floor("D"),
        freq="D",
        tz="UTC",
    )
    partitions: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    previous_kline: pd.DataFrame | None = None
    previous_agg: pd.DataFrame | None = None

    with requests.Session() as session:
        for day in days:
            label = day.strftime("%Y-%m-%d")
            partition_dir = output_dir / f"date={label}"
            partition_path = partition_dir / "second_features.parquet"
            if resume and partition_path.exists():
                existing = pd.read_parquet(partition_path, columns=[DEFAULT_TIMESTAMP_COLUMN])
                partitions.append(
                    {
                        "label": label,
                        "path": str(partition_path),
                        "row_count": int(len(existing)),
                        "start": existing[DEFAULT_TIMESTAMP_COLUMN].min().isoformat(),
                        "end": existing[DEFAULT_TIMESTAMP_COLUMN].max().isoformat(),
                        "status": "reused_existing",
                    }
                )
                continue

            day_start = day
            day_end = day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            day_decisions = _slice(decisions, day_start, day_end)
            if day_decisions.empty:
                continue
            warm_start = day_start - pd.Timedelta(seconds=warmup_seconds)
            try:
                current_kline = _fetch_daily_kline(session, symbol, day)
                current_agg = _fetch_daily_agg_trades(session, symbol, day)
                kline_parts = [frame for frame in (previous_kline, current_kline) if frame is not None and not frame.empty]
                agg_parts = [frame for frame in (previous_agg, current_agg) if frame is not None and not frame.empty]
                kline = _slice(pd.concat(kline_parts, ignore_index=True), warm_start, day_end)
                agg = _slice(pd.concat(agg_parts, ignore_index=True), warm_start, day_end)
                full_store = build_second_level_feature_store(
                    kline_frame=kline,
                    agg_trades_frame=agg,
                    market=settings.second_level.market,
                    exchange=settings.second_level.exchange,
                    source_manifest_id="binance_daily_stream",
                    large_trade_quantile=settings.second_level.large_trade_quantile,
                    large_trade_window_seconds=settings.second_level.large_trade_window_seconds,
                    feature_profile=feature_profile,
                )
                sampled = sample_second_level_feature_store(day_decisions.reset_index(drop=True), full_store)
                partition_dir.mkdir(parents=True, exist_ok=True)
                sampled.to_parquet(partition_path, index=False)
                sl_columns = [column for column in sampled.columns if column.startswith(("sl_", "fm_"))]
                partitions.append(
                    {
                        "label": label,
                        "path": str(partition_path),
                        "row_count": int(len(sampled)),
                        "start": sampled[DEFAULT_TIMESTAMP_COLUMN].min().isoformat(),
                        "end": sampled[DEFAULT_TIMESTAMP_COLUMN].max().isoformat(),
                        "feature_count": int(len(sl_columns)),
                        "status": "built",
                    }
                )
                print(f"built_partition={label} rows={len(sampled)} features={len(sl_columns)}", flush=True)
                previous_kline = _slice(current_kline, day_end - pd.Timedelta(seconds=warmup_seconds), day_end)
                previous_agg = _slice(current_agg, day_end - pd.Timedelta(seconds=warmup_seconds), day_end)
            except Exception as exc:  # noqa: BLE001 - continue so the manifest records coverage gaps.
                failures.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
                print(f"failed_partition={label} error={type(exc).__name__}: {exc}", flush=True)
                previous_kline = None
                previous_agg = None

    if not partitions:
        raise ValueError(f"No partitions were written. failures={failures[:3]}")
    schema = list(pd.read_parquet(partitions[0]["path"]).columns)
    manifest = {
        "feature_version": SECOND_LEVEL_FEATURE_STORE_VERSION,
        "feature_profile": settings.second_level.feature_profile,
        "feature_packs": list(feature_profile.packs),
        "store_name": "second_features_decision_sampled",
        "partitioned": True,
        "partition_frequency": "daily",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "decision_input": str(decision_input.resolve()),
        "config": str(config.resolve()),
        "warmup_seconds": int(warmup_seconds),
        "row_count": int(sum(item["row_count"] for item in partitions)),
        "partition_count": int(len(partitions)),
        "failure_count": int(len(failures)),
        "start": partitions[0]["start"],
        "end": partitions[-1]["end"],
        "schema": schema,
        "feature_count": int(sum(column.startswith(("sl_", "fm_")) for column in schema)),
        "has_1s_kline": True,
        "has_agg_trade_enrichment": True,
        "has_book_ticker": False,
        "partitions": partitions,
        "failures": failures,
    }
    _write_manifest(output_dir, manifest)
    (output_dir / "qa_report.json").write_text(
        json.dumps(
            {
                "row_count": manifest["row_count"],
                "partition_count": manifest["partition_count"],
                "failure_count": manifest["failure_count"],
                "passed": manifest["failure_count"] == 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact decision-row second-level feature store from Binance daily archives.")
    parser.add_argument(
        "--decision-input",
        default="artifacts/data_v2/normalized/binance/spot/BTCUSDT/klines/BTCUSDT-1m.parquet",
        help="Parquet file with timestamp decisions to sample.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT",
        help="Partitioned output directory.",
    )
    parser.add_argument("--config", default="experiments/configs/20260521_regime_reversal_second_agg_features.yaml")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--warmup-seconds", type=int, default=900)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = build_decision_store(
        decision_input=Path(args.decision_input),
        output_dir=Path(args.output),
        config=Path(args.config),
        symbol=args.symbol,
        warmup_seconds=args.warmup_seconds,
        start_date=args.start_date,
        end_date=args.end_date,
        resume=args.resume,
    )
    print(json.dumps({"output": str(Path(args.output).resolve()), **manifest}, indent=2))


if __name__ == "__main__":
    main()
