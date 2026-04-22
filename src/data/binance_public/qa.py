from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


COMMON_REQUIRED_COLUMNS = {
    "timestamp",
    "raw_timestamp",
    "symbol",
    "market_family",
    "data_type",
    "source_file",
    "source_date",
    "source_granularity",
    "source_version",
    "checksum_status",
    "ingested_at",
}

REQUIRED_COLUMNS_BY_DATA_TYPE: dict[str, set[str]] = {
    "klines": {"open", "high", "low", "close", "volume"},
    "markPriceKlines": {"open", "high", "low", "close"},
    "indexPriceKlines": {"open", "high", "low", "close"},
    "premiumIndexKlines": {"open", "high", "low", "close"},
    "fundingRate": {"last_funding_rate", "funding_interval_hours"},
    "metrics": {"timestamp"},
    "bookTicker": {"bid_price", "bid_qty", "ask_price", "ask_qty"},
    "aggTrades": {"price", "quantity", "transact_time"},
    "trades": {"price", "quantity", "transact_time"},
    "bookDepth": {"percentage", "depth", "notional"},
    "BVOLIndex": {"index_value"},
    "EOHSummary": {"timestamp"},
    "liquidationSnapshot": {"timestamp"},
}

NON_NEGATIVE_COLUMNS_BY_DATA_TYPE: dict[str, set[str]] = {
    "klines": {"open", "high", "low", "close", "volume"},
    "markPriceKlines": {"open", "high", "low", "close"},
    "indexPriceKlines": {"open", "high", "low", "close"},
    "premiumIndexKlines": {"open", "high", "low", "close"},
    "fundingRate": {"funding_interval_hours"},
    "bookTicker": {"bid_price", "bid_qty", "ask_price", "ask_qty"},
    "aggTrades": {"price", "quantity"},
    "trades": {"price", "quantity", "quote_quantity"},
    "bookDepth": {"depth", "notional"},
    "BVOLIndex": {"index_value"},
}

EVENT_STREAM_TYPES = {"bookTicker", "aggTrades", "trades", "bookDepth", "liquidationSnapshot"}
KLINE_TYPES = {"klines", "markPriceKlines", "indexPriceKlines", "premiumIndexKlines"}


@dataclass(frozen=True)
class QaTableResult:
    market_family: str
    data_type: str
    symbol: str
    interval: str | None
    file_path: str
    row_count: int
    checks: dict[str, Any]

    @property
    def passed(self) -> bool:
        positive_check_names = {
            name
            for name, value in self.checks.items()
            if isinstance(value, bool) and not name.startswith("has_")
        }
        return all(bool(self.checks[name]) for name in positive_check_names)


def _infer_symbol_and_interval(path: Path) -> tuple[str, str | None]:
    stem = path.stem
    if path.parent.name in KLINE_TYPES:
        symbol, interval = stem.rsplit("-", 1)
        return symbol, interval
    return stem, None


def _iter_normalized_files(normalized_root: Path) -> list[Path]:
    return sorted(normalized_root.rglob("*.parquet"))


def _required_columns_for(data_type: str) -> set[str]:
    return COMMON_REQUIRED_COLUMNS | REQUIRED_COLUMNS_BY_DATA_TYPE.get(data_type, set())


def _strict_1m_continuity(frame: pd.DataFrame, *, interval: str | None, data_type: str) -> tuple[bool | None, int | None]:
    if interval != "1m" or data_type not in KLINE_TYPES:
        return None, None
    if len(frame) <= 1:
        return True, 0
    diffs = frame["timestamp"].sort_values().diff().dropna()
    gap_count = int((diffs > pd.Timedelta(minutes=1)).sum())
    return bool((diffs == pd.Timedelta(minutes=1)).all()), gap_count


def _has_non_negative_violations(frame: pd.DataFrame, data_type: str) -> bool:
    for column in NON_NEGATIVE_COLUMNS_BY_DATA_TYPE.get(data_type, set()):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if (values.dropna() < 0).any():
            return True
    return False


def _table_checks(file_path: Path) -> QaTableResult:
    market_family = file_path.parent.parent.name
    data_type = file_path.parent.name
    symbol, interval = _infer_symbol_and_interval(file_path)
    frame = pd.read_parquet(file_path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    required_columns = _required_columns_for(data_type)
    missing_required_columns = sorted(required_columns - set(frame.columns))
    duplicate_subset = ["timestamp", "symbol"] if {"timestamp", "symbol"}.issubset(frame.columns) else ["timestamp"]
    duplicate_rows = int(frame.duplicated(subset=duplicate_subset).sum()) if "timestamp" in frame.columns else len(frame)
    monotonic = bool(frame["timestamp"].is_monotonic_increasing) if "timestamp" in frame.columns else False
    continuity_ok, gap_count = _strict_1m_continuity(frame, interval=interval, data_type=data_type)
    non_empty = len(frame) > 0
    event_stream_ready = None
    if data_type in EVENT_STREAM_TYPES:
        event_stream_ready = non_empty and monotonic and not missing_required_columns

    checks: dict[str, Any] = {
        "required_columns_present": not missing_required_columns,
        "missing_required_columns": missing_required_columns,
        "non_empty": non_empty,
        "timestamp_monotonic_increasing": monotonic,
        "duplicate_timestamp_symbol_rows": duplicate_rows,
        "no_duplicate_timestamp_symbol_rows": duplicate_rows == 0,
        "has_negative_value_violation": _has_non_negative_violations(frame, data_type),
        "no_negative_value_violation": not _has_non_negative_violations(frame, data_type),
    }
    if continuity_ok is not None:
        checks["strict_1m_continuity"] = continuity_ok
        checks["gap_count_gt_1m"] = gap_count
    if event_stream_ready is not None:
        checks["event_stream_aggregatable"] = event_stream_ready

    return QaTableResult(
        market_family=market_family,
        data_type=data_type,
        symbol=symbol,
        interval=interval,
        file_path=str(file_path.resolve()),
        row_count=int(len(frame)),
        checks=checks,
    )


def _load_frame_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def _timestamp_overlap(left: pd.DataFrame | None, right: pd.DataFrame | None) -> dict[str, Any]:
    if left is None or right is None or left.empty or right.empty:
        return {
            "left_present": left is not None and not left.empty,
            "right_present": right is not None and not right.empty,
            "overlap_count": 0,
            "left_only_count": None,
            "right_only_count": None,
            "alignable": False,
        }
    left_ts = pd.Index(pd.to_datetime(left["timestamp"], utc=True).drop_duplicates())
    right_ts = pd.Index(pd.to_datetime(right["timestamp"], utc=True).drop_duplicates())
    overlap = left_ts.intersection(right_ts)
    return {
        "left_present": True,
        "right_present": True,
        "overlap_count": int(len(overlap)),
        "left_only_count": int(len(left_ts.difference(right_ts))),
        "right_only_count": int(len(right_ts.difference(left_ts))),
        "alignable": len(overlap) > 0,
    }


def _funding_forward_fill_ready(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"present": False, "monotonic": False, "duplicate_timestamp_rows": None, "forward_fill_ready": False}
    duplicates = int(frame.duplicated(subset=["timestamp", "symbol"]).sum()) if "symbol" in frame.columns else 0
    monotonic = bool(frame["timestamp"].is_monotonic_increasing)
    return {
        "present": True,
        "monotonic": monotonic,
        "duplicate_timestamp_rows": duplicates,
        "forward_fill_ready": monotonic and duplicates == 0,
    }


def _build_cross_table_checks(normalized_root: Path) -> list[dict[str, Any]]:
    spot_klines = _load_frame_if_exists(normalized_root / "spot" / "klines" / "BTCUSDT-1m.parquet")
    um_klines = _load_frame_if_exists(normalized_root / "futures_um" / "klines" / "BTCUSDT-1m.parquet")
    mark = _load_frame_if_exists(normalized_root / "futures_um" / "markPriceKlines" / "BTCUSDT-1m.parquet")
    index = _load_frame_if_exists(normalized_root / "futures_um" / "indexPriceKlines" / "BTCUSDT-1m.parquet")
    premium = _load_frame_if_exists(normalized_root / "futures_um" / "premiumIndexKlines" / "BTCUSDT-1m.parquet")
    funding = _load_frame_if_exists(normalized_root / "futures_um" / "fundingRate" / "BTCUSDT.parquet")
    bvol = _load_frame_if_exists(normalized_root / "option" / "BVOLIndex" / "BTCBVOLUSDT.parquet")

    checks = [
        {"name": "spot_vs_um_klines_alignment", **_timestamp_overlap(spot_klines, um_klines)},
        {"name": "mark_vs_um_klines_alignment", **_timestamp_overlap(mark, um_klines)},
        {"name": "index_vs_um_klines_alignment", **_timestamp_overlap(index, um_klines)},
        {"name": "premium_vs_um_klines_alignment", **_timestamp_overlap(premium, um_klines)},
        {"name": "funding_forward_fill_readiness", **_funding_forward_fill_ready(funding)},
        {"name": "bvol_vs_spot_alignment", **_timestamp_overlap(bvol, spot_klines)},
    ]
    return checks


def run_binance_public_qa(output_root: Path) -> dict[str, Any]:
    normalized_root = output_root / "normalized"
    manifests_root = output_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    table_results = [_table_checks(path) for path in _iter_normalized_files(normalized_root)]
    cross_table_checks = _build_cross_table_checks(normalized_root)
    generated_at = datetime.now(timezone.utc).isoformat()

    payload = {
        "generated_at": generated_at,
        "output_root": str(output_root.resolve()),
        "summary": {
            "table_count": len(table_results),
            "table_pass_count": sum(1 for result in table_results if result.passed),
            "table_fail_count": sum(1 for result in table_results if not result.passed),
            "cross_table_check_count": len(cross_table_checks),
            "cross_table_alignable_count": sum(1 for check in cross_table_checks if check.get("alignable") is True),
        },
        "tables": [
            {
                "market_family": result.market_family,
                "data_type": result.data_type,
                "symbol": result.symbol,
                "interval": result.interval,
                "file_path": result.file_path,
                "row_count": result.row_count,
                "passed": result.passed,
                "checks": result.checks,
            }
            for result in table_results
        ],
        "cross_table_checks": cross_table_checks,
    }

    (manifests_root / "qa_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
