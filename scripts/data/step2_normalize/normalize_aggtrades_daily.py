from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'src').is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


AGG_TRADES_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
]
OUTPUT_COLUMNS = [
    "timestamp",
    "price",
    "quantity",
    "quote_quantity",
    "is_buyer_maker",
    "agg_trade_id",
    "first_trade_id",
    "last_trade_id",
]


def _read_csv_chunks(path: Path, chunksize: int):
    first_line = path.open("r", encoding="utf-8", errors="ignore").readline().strip()
    has_header = first_line.split(",", maxsplit=1)[0] == "agg_trade_id"
    return pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else AGG_TRADES_COLUMNS,
        chunksize=chunksize,
        memory_map=True,
        na_filter=False,
    )


def _timestamp_unit(values: pd.Series) -> str:
    median = pd.to_numeric(values, errors="coerce").dropna().abs().median()
    return "us" if median > 10_000_000_000_000 else "ms" if median > 10_000_000_000 else "s"


def _normalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    normalized = chunk.copy()
    for column in ("agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["timestamp"] = pd.to_datetime(
        normalized["transact_time"],
        unit=_timestamp_unit(normalized["transact_time"]),
        utc=True,
        errors="coerce",
    )
    maker = normalized["is_buyer_maker"]
    if maker.dtype == "object":
        maker = maker.astype(str).str.lower().isin({"true", "1", "t"})
    normalized["is_buyer_maker"] = maker.astype(bool)
    normalized["quote_quantity"] = normalized["price"] * normalized["quantity"]
    normalized = normalized.dropna(subset=["timestamp", "price", "quantity"])
    return normalized[OUTPUT_COLUMNS]


def _date_label(timestamp: pd.Timestamp) -> str:
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw Binance Spot aggTrades CSVs into daily event parquet partitions.")
    parser.add_argument("--input", action="append", required=True, help="Raw aggTrades CSV path. Can be provided multiple times.")
    parser.add_argument("--data-root", default="artifacts/data_v2", help="Unified data root for new outputs.")
    parser.add_argument("--output-dir", help="Output partition directory. Defaults to data-root/normalized/binance/spot/BTCUSDT/aggTrades.")
    parser.add_argument("--start", help="Inclusive UTC start timestamp/date.")
    parser.add_argument("--end", help="Exclusive UTC end timestamp/date.")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()

    output_root = Path(args.output_dir) if args.output_dir else (
        Path(args.data_root) / "normalized" / "binance" / "spot" / "BTCUSDT" / "aggTrades"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end = pd.Timestamp(args.end, tz="UTC") if args.end else None
    writers: dict[str, pq.ParquetWriter] = {}
    row_count = 0
    source_files = [str(Path(item).resolve()) for item in args.input]

    try:
        for raw_path in [Path(item) for item in args.input]:
            for chunk in _read_csv_chunks(raw_path, args.chunksize):
                normalized = _normalize_chunk(chunk)
                if start is not None:
                    normalized = normalized.loc[normalized["timestamp"] >= start]
                if end is not None:
                    normalized = normalized.loc[normalized["timestamp"] < end]
                if normalized.empty:
                    continue
                normalized["date"] = normalized["timestamp"].dt.strftime("%Y-%m-%d")
                for label, group in normalized.groupby("date", sort=True):
                    partition_dir = output_root / f"date={label}"
                    partition_dir.mkdir(parents=True, exist_ok=True)
                    path = partition_dir / "agg_trades.parquet"
                    table = pa.Table.from_pandas(group.drop(columns=["date"]), preserve_index=False)
                    if label not in writers:
                        writers[label] = pq.ParquetWriter(path, table.schema)
                    writers[label].write_table(table)
                    row_count += len(group)
    finally:
        for writer in writers.values():
            writer.close()

    manifest = {
        "source": "binance_spot_aggTrades",
        "source_files": source_files,
        "output_dir": str(output_root.resolve()),
        "partition_count": len(writers),
        "row_count": int(row_count),
        "start": str(start) if start is not None else None,
        "end": str(end) if end is not None else None,
        "schema": OUTPUT_COLUMNS,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
