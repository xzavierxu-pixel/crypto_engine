from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas import CategoricalDtype
from pandas.api.types import is_numeric_dtype

from src.core.constants import BINANCE_PUBLIC_SCHEMA_VERSION

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

METADATA_COLUMNS = [
    "timestamp",
    "raw_timestamp",
    "symbol",
    "market_family",
    "data_type",
    "interval",
    "source_file",
    "source_date",
    "source_granularity",
    "source_version",
    "checksum_status",
    "ingested_at",
]

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

TRADES_COLUMNS = [
    "trade_id",
    "price",
    "quantity",
    "quote_quantity",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
]

BOOK_TICKER_COLUMNS = [
    "update_id",
    "bid_price",
    "bid_qty",
    "ask_price",
    "ask_qty",
    "transaction_time",
    "event_time",
]

CHUNKED_DATA_TYPES = {"aggTrades", "trades"}
CSV_CHUNK_SIZE = 500_000
CSV_READ_COMMON_KWARGS = {
    "na_filter": False,
    "memory_map": True,
}

FLOAT_LIKE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "price",
    "quantity",
    "quote_quantity",
    "bid_price",
    "bid_qty",
    "ask_price",
    "ask_qty",
    "last_funding_rate",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
    "percentage",
    "depth",
    "notional",
    "index_value",
}

INT_LIKE_COLUMNS = {
    "open_time",
    "close_time",
    "count",
    "agg_trade_id",
    "first_trade_id",
    "last_trade_id",
    "trade_id",
    "transact_time",
    "transaction_time",
    "event_time",
    "update_id",
    "calc_time",
    "funding_interval_hours",
    "ignore",
}

CATEGORY_LIKE_COLUMNS = {
    "symbol",
    "market_family",
    "data_type",
    "interval",
    "source_file",
    "source_date",
    "source_granularity",
    "source_version",
    "checksum_status",
    "download_status",
    "expected_checksum",
    "actual_checksum",
    "base_asset",
    "quote_asset",
}


@dataclass(frozen=True)
class RawFileDescriptor:
    file_path: Path
    market_family: str
    data_type: str
    symbol: str
    interval: str | None
    source_granularity: str
    source_date: str


@dataclass(frozen=True)
class ChecksumEntry:
    checksum_status: str
    status: str
    expected_checksum: str | None
    actual_checksum: str | None


def _iter_raw_file_descriptors(raw_root: Path) -> list[RawFileDescriptor]:
    descriptors: list[RawFileDescriptor] = []
    for file_path in raw_root.rglob("*.csv"):
        relative_parts = file_path.relative_to(raw_root).parts
        if len(relative_parts) == 6:
            market_family, data_type, symbol, source_granularity, source_date, _ = relative_parts
            interval = None
        elif len(relative_parts) == 7:
            market_family, data_type, symbol, interval, source_granularity, source_date, _ = relative_parts
        else:
            continue
        descriptors.append(
            RawFileDescriptor(
                file_path=file_path,
                market_family=market_family,
                data_type=data_type,
                symbol=symbol,
                interval=interval,
                source_granularity=source_granularity,
                source_date=source_date,
            )
        )
    return sorted(descriptors, key=lambda item: str(item.file_path))


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    if is_numeric_dtype(series):
        numeric = series
    else:
        numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        max_value = int(numeric.max()) if len(numeric) else 0
        unit = "us" if max_value >= 10**15 else "ms"
        return pd.to_datetime(numeric.astype("int64"), unit=unit, utc=True)
    return pd.to_datetime(series, utc=True)


def _coerce_numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns or is_numeric_dtype(frame[column]):
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _read_csv_with_optional_header(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    first_token = first_line.split(",", 1)[0].strip()
    has_header = any(character.isalpha() for character in first_token)
    return pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else expected_columns,
        **CSV_READ_COMMON_KWARGS,
    )


def _read_csv_chunks_with_optional_header(path: Path, expected_columns: list[str], chunksize: int) -> Iterable[pd.DataFrame]:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    first_token = first_line.split(",", 1)[0].strip()
    has_header = any(character.isalpha() for character in first_token)
    return pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else expected_columns,
        chunksize=chunksize,
        **CSV_READ_COMMON_KWARGS,
    )


def _read_kline_frame(descriptor: RawFileDescriptor) -> pd.DataFrame:
    frame = _read_csv_with_optional_header(descriptor.file_path, KLINE_COLUMNS)
    frame["timestamp"] = _parse_timestamp_series(frame["open_time"])
    frame["raw_timestamp"] = frame["open_time"].astype(str)
    return _coerce_numeric_columns(
        frame,
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )


def _read_funding_rate_frame(descriptor: RawFileDescriptor) -> pd.DataFrame:
    frame = pd.read_csv(descriptor.file_path)
    frame["timestamp"] = _parse_timestamp_series(frame["calc_time"])
    frame["raw_timestamp"] = frame["calc_time"].astype(str)
    return _coerce_numeric_columns(frame, ("funding_interval_hours", "last_funding_rate"))


def _read_metrics_frame(descriptor: RawFileDescriptor) -> pd.DataFrame:
    frame = pd.read_csv(descriptor.file_path)
    frame["timestamp"] = _parse_timestamp_series(frame["create_time"])
    frame["raw_timestamp"] = frame["create_time"].astype(str)
    for column in frame.columns:
        if column in {"create_time", "symbol", "timestamp", "raw_timestamp"}:
            continue
        try:
            frame[column] = pd.to_numeric(frame[column])
        except (TypeError, ValueError):
            continue
    return frame


def _read_agg_trades_frame(descriptor: RawFileDescriptor) -> pd.DataFrame:
    frame = _read_csv_with_optional_header(descriptor.file_path, AGG_TRADES_COLUMNS)
    frame = _coerce_numeric_columns(
        frame,
        ("agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time"),
    )
    frame["timestamp"] = _parse_timestamp_series(frame["transact_time"])
    frame["raw_timestamp"] = frame["transact_time"].astype(str)
    return frame


def _read_trades_frame(descriptor: RawFileDescriptor) -> pd.DataFrame:
    frame = _read_csv_with_optional_header(descriptor.file_path, TRADES_COLUMNS)
    frame = _coerce_numeric_columns(
        frame,
        ("trade_id", "price", "quantity", "quote_quantity", "transact_time"),
    )
    frame["timestamp"] = _parse_timestamp_series(frame["transact_time"])
    frame["raw_timestamp"] = frame["transact_time"].astype(str)
    return frame


def _read_book_ticker_frame(descriptor: RawFileDescriptor) -> pd.DataFrame:
    frame = _read_csv_with_optional_header(descriptor.file_path, BOOK_TICKER_COLUMNS)
    timestamp_column = "transaction_time" if "transaction_time" in frame.columns else "event_time"
    frame = _coerce_numeric_columns(
        frame,
        ("update_id", "bid_price", "bid_qty", "ask_price", "ask_qty", "transaction_time", "event_time"),
    )
    frame["timestamp"] = _parse_timestamp_series(frame[timestamp_column])
    frame["raw_timestamp"] = frame[timestamp_column].astype(str)
    return frame


def _read_book_depth_frame(descriptor: RawFileDescriptor) -> pd.DataFrame:
    frame = pd.read_csv(descriptor.file_path)
    frame["timestamp"] = _parse_timestamp_series(frame["timestamp"])
    frame["raw_timestamp"] = frame["timestamp"].astype(str)
    return _coerce_numeric_columns(frame, ("percentage", "depth", "notional"))


def _read_bvol_index_frame(descriptor: RawFileDescriptor) -> pd.DataFrame:
    frame = pd.read_csv(descriptor.file_path)
    frame["timestamp"] = _parse_timestamp_series(frame["calc_time"])
    frame["raw_timestamp"] = frame["calc_time"].astype(str)
    return _coerce_numeric_columns(frame, ("index_value",))


def _read_generic_timestamped_frame(
    descriptor: RawFileDescriptor,
    timestamp_candidates: list[str],
) -> pd.DataFrame:
    frame = pd.read_csv(descriptor.file_path)
    timestamp_column = next((column for column in timestamp_candidates if column in frame.columns), None)
    if timestamp_column is None:
        raise ValueError(f"No timestamp column found for {descriptor.data_type} in {descriptor.file_path}.")
    frame["timestamp"] = _parse_timestamp_series(frame[timestamp_column])
    frame["raw_timestamp"] = frame[timestamp_column].astype(str)
    return frame


def _load_checksum_map(manifests_root: Path) -> dict[str, ChecksumEntry]:
    download_manifest_path = manifests_root / "download_manifest.json"
    if not download_manifest_path.exists():
        return {}

    payload = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    checksum_map: dict[str, ChecksumEntry] = {}
    for section in ("downloaded", "unavailable_by_listing", "missing_or_failed"):
        for entry in payload.get(section, []):
            for extracted_file in entry.get("extracted_files", []):
                checksum_map[str(Path(extracted_file).resolve())] = ChecksumEntry(
                    checksum_status=entry.get("checksum_status", "unknown"),
                    status=entry.get("status", "unknown"),
                    expected_checksum=entry.get("expected_checksum"),
                    actual_checksum=entry.get("actual_checksum"),
                )
    return checksum_map


def _apply_checksum_metadata(frame: pd.DataFrame, descriptor: RawFileDescriptor, checksum_map: dict[str, ChecksumEntry]) -> pd.DataFrame:
    entry = checksum_map.get(str(descriptor.file_path.resolve()))
    if entry is None:
        frame["checksum_status"] = "unknown"
        frame["download_status"] = "unknown"
        frame["expected_checksum"] = None
        frame["actual_checksum"] = None
        return frame
    frame["checksum_status"] = entry.checksum_status
    frame["download_status"] = entry.status
    frame["expected_checksum"] = entry.expected_checksum
    frame["actual_checksum"] = entry.actual_checksum
    return frame


def _stabilize_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        if column in FLOAT_LIKE_COLUMNS:
            if not is_numeric_dtype(frame[column]) or str(frame[column].dtype) != "float64":
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        elif column in INT_LIKE_COLUMNS:
            if str(frame[column].dtype) != "Int64":
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
        elif column in CATEGORY_LIKE_COLUMNS:
            if not isinstance(frame[column].dtype, CategoricalDtype):
                frame[column] = frame[column].astype("category")
    return frame


def _sort_by_timestamp_if_needed(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or frame["timestamp"].is_monotonic_increasing:
        return frame
    return frame.sort_values("timestamp").reset_index(drop=True)


def _apply_common_metadata(
    frame: pd.DataFrame,
    descriptor: RawFileDescriptor,
    ingested_at: str,
    checksum_map: dict[str, ChecksumEntry],
) -> pd.DataFrame:
    if "symbol" not in frame.columns:
        frame["symbol"] = descriptor.symbol
    else:
        frame["symbol"] = frame["symbol"].fillna(descriptor.symbol).astype(str)

    frame["market_family"] = descriptor.market_family
    frame["data_type"] = descriptor.data_type
    frame["interval"] = descriptor.interval
    frame["source_file"] = str(descriptor.file_path.resolve())
    frame["source_date"] = descriptor.source_date
    frame["source_granularity"] = descriptor.source_granularity
    frame["source_version"] = BINANCE_PUBLIC_SCHEMA_VERSION
    frame["ingested_at"] = ingested_at
    frame = _apply_checksum_metadata(frame, descriptor, checksum_map)
    frame = _stabilize_dtypes(frame)

    non_metadata_columns = [column for column in frame.columns if column not in METADATA_COLUMNS]
    ordered_columns = METADATA_COLUMNS + non_metadata_columns
    return frame[ordered_columns]


def _normalize_file(
    descriptor: RawFileDescriptor,
    ingested_at: str,
    checksum_map: dict[str, ChecksumEntry],
) -> pd.DataFrame:
    if descriptor.data_type.endswith("Klines") or descriptor.data_type == "klines":
        frame = _read_kline_frame(descriptor)
    elif descriptor.data_type == "fundingRate":
        frame = _read_funding_rate_frame(descriptor)
    elif descriptor.data_type == "metrics":
        frame = _read_metrics_frame(descriptor)
    elif descriptor.data_type == "bookTicker":
        frame = _read_book_ticker_frame(descriptor)
    elif descriptor.data_type == "bookDepth":
        frame = _read_book_depth_frame(descriptor)
    elif descriptor.data_type == "BVOLIndex":
        frame = _read_bvol_index_frame(descriptor)
    elif descriptor.data_type == "EOHSummary":
        frame = _read_generic_timestamped_frame(descriptor, ["timestamp", "calc_time", "create_time"])
    elif descriptor.data_type == "liquidationSnapshot":
        frame = _read_generic_timestamped_frame(descriptor, ["timestamp", "time", "update_time"])
    elif descriptor.data_type == "aggTrades":
        frame = _read_agg_trades_frame(descriptor)
    elif descriptor.data_type == "trades":
        frame = _read_trades_frame(descriptor)
    else:
        raise ValueError(f"Unsupported data_type '{descriptor.data_type}' for normalization.")
    return _apply_common_metadata(frame, descriptor, ingested_at, checksum_map)


def _normalize_file_chunks(
    descriptor: RawFileDescriptor,
    ingested_at: str,
    checksum_map: dict[str, ChecksumEntry],
) -> Iterable[pd.DataFrame]:
    if descriptor.data_type == "aggTrades":
        for chunk in _read_csv_chunks_with_optional_header(descriptor.file_path, AGG_TRADES_COLUMNS, CSV_CHUNK_SIZE):
            chunk = _coerce_numeric_columns(
                chunk,
                ("agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time"),
            )
            chunk["timestamp"] = _parse_timestamp_series(chunk["transact_time"])
            chunk["raw_timestamp"] = chunk["transact_time"].astype(str)
            yield _apply_common_metadata(chunk, descriptor, ingested_at, checksum_map)
        return
    if descriptor.data_type == "trades":
        for chunk in _read_csv_chunks_with_optional_header(descriptor.file_path, TRADES_COLUMNS, CSV_CHUNK_SIZE):
            chunk = _coerce_numeric_columns(
                chunk,
                ("trade_id", "price", "quantity", "quote_quantity", "transact_time"),
            )
            chunk["timestamp"] = _parse_timestamp_series(chunk["transact_time"])
            chunk["raw_timestamp"] = chunk["transact_time"].astype(str)
            yield _apply_common_metadata(chunk, descriptor, ingested_at, checksum_map)
        return
    yield _normalize_file(descriptor, ingested_at, checksum_map)


def _group_key(descriptor: RawFileDescriptor) -> tuple[str, str, str, str | None]:
    return (descriptor.market_family, descriptor.data_type, descriptor.symbol, descriptor.interval)


def _descriptor_sort_key(descriptor: RawFileDescriptor) -> tuple[int, str, str, str]:
    granularity_rank = 0 if descriptor.source_granularity == "monthly" else 1
    return (granularity_rank, descriptor.source_date, descriptor.interval or "", str(descriptor.file_path))


def _group_output_path(normalized_root: Path, key: tuple[str, str, str, str | None]) -> Path:
    market_family, data_type, symbol, interval = key
    filename = f"{symbol}"
    if interval:
        filename = f"{filename}-{interval}"
    filename = f"{filename}.parquet"
    output_dir = normalized_root / market_family / data_type
    return output_dir / filename


@dataclass
class FrameAccumulator:
    columns: list[str] | None = None
    dtypes: dict[str, str] | None = None
    row_count: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    null_count_by_column: dict[str, int] | None = None
    duplicate_timestamp_symbol_rows: int = 0
    timestamp_monotonic_increasing: bool = True
    strict_1m_continuity: bool | None = None
    gap_count_gt_1m: int | None = None
    previous_last_timestamp: pd.Timestamp | None = None
    checksum_statuses: set[str] | None = None


def _update_accumulator(
    accumulator: FrameAccumulator,
    frame: pd.DataFrame,
    interval: str | None,
    data_type: str,
) -> None:
    if accumulator.columns is None:
        accumulator.columns = list(frame.columns)
        accumulator.dtypes = {column: str(dtype) for column, dtype in frame.dtypes.items()}
        accumulator.null_count_by_column = {column: 0 for column in frame.columns}
        accumulator.checksum_statuses = set()
        if interval == "1m" and (data_type.endswith("Klines") or data_type == "klines"):
            accumulator.strict_1m_continuity = True
            accumulator.gap_count_gt_1m = 0
        else:
            accumulator.strict_1m_continuity = None
            accumulator.gap_count_gt_1m = None

    accumulator.row_count += int(len(frame))
    if frame.empty:
        return

    null_counts = frame.isna().sum()
    for column, null_count in null_counts.items():
        accumulator.null_count_by_column[column] += int(null_count)
    if "checksum_status" in frame.columns:
        accumulator.checksum_statuses.update(frame["checksum_status"].dropna().astype(str).tolist())

    timestamps = frame["timestamp"]
    frame_start = timestamps.iloc[0]
    frame_end = timestamps.iloc[-1]
    accumulator.start = frame_start if accumulator.start is None else min(accumulator.start, frame_start)
    accumulator.end = frame_end if accumulator.end is None else max(accumulator.end, frame_end)

    accumulator.duplicate_timestamp_symbol_rows += int(frame.duplicated(subset=["timestamp", "symbol"]).sum())
    if not timestamps.is_monotonic_increasing:
        accumulator.timestamp_monotonic_increasing = False

    if accumulator.previous_last_timestamp is not None and frame_start < accumulator.previous_last_timestamp:
        accumulator.timestamp_monotonic_increasing = False

    if accumulator.strict_1m_continuity is not None:
        diffs = timestamps.diff().dropna()
        gap_count = int((diffs > pd.Timedelta(minutes=1)).sum())
        if accumulator.previous_last_timestamp is not None:
            boundary_diff = frame_start - accumulator.previous_last_timestamp
            if boundary_diff > pd.Timedelta(minutes=1):
                gap_count += 1
            if boundary_diff != pd.Timedelta(minutes=1):
                accumulator.strict_1m_continuity = False
        if not diffs.empty and not (diffs == pd.Timedelta(minutes=1)).all():
            accumulator.strict_1m_continuity = False
        accumulator.gap_count_gt_1m += gap_count

    accumulator.previous_last_timestamp = frame_end


def _accumulator_summary(accumulator: FrameAccumulator) -> dict[str, Any]:
    return {
        "row_count": accumulator.row_count,
        "columns": accumulator.columns or [],
        "dtypes": accumulator.dtypes or {},
        "start": accumulator.start.isoformat() if accumulator.start is not None else None,
        "end": accumulator.end.isoformat() if accumulator.end is not None else None,
        "qa": {
            "duplicate_timestamp_symbol_rows": accumulator.duplicate_timestamp_symbol_rows,
            "timestamp_monotonic_increasing": accumulator.timestamp_monotonic_increasing,
            "null_count_by_column": accumulator.null_count_by_column or {},
            "strict_1m_continuity": accumulator.strict_1m_continuity,
            "gap_count_gt_1m": accumulator.gap_count_gt_1m,
        },
    }


def normalize_binance_public_history(output_root: Path) -> dict[str, Any]:
    raw_root = output_root / "raw"
    normalized_root = output_root / "normalized"
    manifests_root = output_root / "manifests"
    normalized_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    descriptors = _iter_raw_file_descriptors(raw_root)
    ingested_at = datetime.now(timezone.utc).isoformat()
    checksum_map = _load_checksum_map(manifests_root)

    grouped_descriptors: dict[tuple[str, str, str, str | None], list[RawFileDescriptor]] = {}
    unsupported_files: list[str] = []
    for descriptor in descriptors:
        if descriptor.data_type not in {
            "klines",
            "markPriceKlines",
            "indexPriceKlines",
            "premiumIndexKlines",
            "fundingRate",
            "metrics",
            "aggTrades",
            "trades",
            "bookTicker",
            "bookDepth",
            "BVOLIndex",
            "EOHSummary",
            "liquidationSnapshot",
        }:
            unsupported_files.append(str(descriptor.file_path.resolve()))
            continue
        grouped_descriptors.setdefault(_group_key(descriptor), []).append(descriptor)

    outputs: list[dict[str, Any]] = []
    for key, file_group in sorted(grouped_descriptors.items(), key=lambda item: item[0]):
        file_group = sorted(file_group, key=_descriptor_sort_key)
        output_path = _group_output_path(normalized_root, key)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer: pq.ParquetWriter | None = None
        accumulator = FrameAccumulator()
        try:
            for descriptor in file_group:
                for frame in _normalize_file_chunks(descriptor, ingested_at=ingested_at, checksum_map=checksum_map):
                    frame = _sort_by_timestamp_if_needed(frame)
                    _update_accumulator(accumulator, frame, interval=key[3], data_type=key[1])
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema)
                    writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        outputs.append(
            {
                "market_family": key[0],
                "data_type": key[1],
                "symbol": key[2],
                "interval": key[3],
                "output_path": str(output_path.resolve()),
                "source_files": [str(descriptor.file_path.resolve()) for descriptor in file_group],
                "schema": _accumulator_summary(accumulator),
                "checksum_statuses": sorted((accumulator.checksum_statuses or set())),
            }
        )

    manifest = {
        "generated_at": ingested_at,
        "source_version": BINANCE_PUBLIC_SCHEMA_VERSION,
        "output_root": str(output_root.resolve()),
        "normalized_outputs": outputs,
        "unsupported_files": unsupported_files,
    }
    (manifests_root / "schema_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
