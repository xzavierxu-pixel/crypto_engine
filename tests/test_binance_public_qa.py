from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import src.data.binance_public.qa as qa_module
from src.data.binance_public.qa import run_binance_public_qa


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _base_metadata(frame: pd.DataFrame, *, symbol: str, market_family: str, data_type: str, interval: str | None) -> pd.DataFrame:
    result = frame.copy()
    result["raw_timestamp"] = result["timestamp"].astype(str)
    result["symbol"] = symbol
    result["market_family"] = market_family
    result["data_type"] = data_type
    result["interval"] = interval
    result["source_file"] = f"/tmp/{symbol}-{data_type}.csv"
    result["source_date"] = "2026-01-01"
    result["source_granularity"] = "daily"
    result["source_version"] = "v1"
    result["checksum_status"] = "verified"
    result["ingested_at"] = "2026-04-11T00:00:00+00:00"
    return result


def test_run_binance_public_qa_writes_manifest_with_table_and_cross_table_checks(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1min")

    _write_parquet(
        tmp_path / "normalized" / "spot" / "klines" / "BTCUSDT-1m.parquet",
        _base_metadata(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [10.0, 11.0, 12.0],
                }
            ),
            symbol="BTCUSDT",
            market_family="spot",
            data_type="klines",
            interval="1m",
        ),
    )
    for data_type, close_values in (
        ("klines", [100.4, 101.4, 102.4]),
        ("markPriceKlines", [100.3, 101.3, 102.3]),
        ("indexPriceKlines", [100.2, 101.2, 102.2]),
        ("premiumIndexKlines", [0.001, 0.002, 0.003]),
    ):
        _write_parquet(
            tmp_path / "normalized" / "futures_um" / data_type / "BTCUSDT-1m.parquet",
            _base_metadata(
                pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "open": close_values,
                        "high": close_values,
                        "low": close_values,
                        "close": close_values,
                        "volume": [1.0, 1.0, 1.0],
                    }
                ),
                symbol="BTCUSDT",
                market_family="futures_um",
                data_type=data_type,
                interval="1m",
            ),
        )

    _write_parquet(
        tmp_path / "normalized" / "futures_um" / "fundingRate" / "BTCUSDT.parquet",
        _base_metadata(
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T08:00:00Z"], utc=True),
                    "last_funding_rate": [0.0001, 0.0002],
                    "funding_interval_hours": [8, 8],
                }
            ),
            symbol="BTCUSDT",
            market_family="futures_um",
            data_type="fundingRate",
            interval=None,
        ),
    )
    _write_parquet(
        tmp_path / "normalized" / "futures_um" / "bookTicker" / "BTCUSDT.parquet",
        _base_metadata(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "bid_price": [100.0, 101.0, 102.0],
                    "bid_qty": [10.0, 10.5, 11.0],
                    "ask_price": [100.2, 101.2, 102.2],
                    "ask_qty": [9.0, 9.5, 10.0],
                }
            ),
            symbol="BTCUSDT",
            market_family="futures_um",
            data_type="bookTicker",
            interval=None,
        ),
    )
    _write_parquet(
        tmp_path / "normalized" / "option" / "BVOLIndex" / "BTCBVOLUSDT.parquet",
        _base_metadata(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "index_value": [0.45, 0.46, 0.47],
                }
            ),
            symbol="BTCBVOLUSDT",
            market_family="option",
            data_type="BVOLIndex",
            interval=None,
        ),
    )

    manifest = run_binance_public_qa(tmp_path)

    assert manifest["summary"]["table_count"] == 8
    assert manifest["summary"]["table_fail_count"] == 0
    assert manifest["summary"]["cross_table_check_count"] == 6

    qa_manifest_path = tmp_path / "manifests" / "qa_manifest.json"
    assert qa_manifest_path.exists()
    payload = json.loads(qa_manifest_path.read_text(encoding="utf-8"))
    assert payload["summary"]["table_pass_count"] == 8

    spot_entry = next(entry for entry in payload["tables"] if entry["market_family"] == "spot")
    assert spot_entry["checks"]["strict_1m_continuity"] is True

    funding_entry = next(entry for entry in payload["tables"] if entry["data_type"] == "fundingRate")
    assert funding_entry["checks"]["timestamp_monotonic_increasing"] is True

    cross_check = next(entry for entry in payload["cross_table_checks"] if entry["name"] == "spot_vs_um_klines_alignment")
    assert cross_check["alignable"] is True
    assert cross_check["overlap_count"] == 3


def test_run_binance_public_qa_flags_kline_gaps_and_event_stream_schema_issues(tmp_path: Path) -> None:
    kline_timestamps = pd.to_datetime(
        ["2026-01-01T00:00:00Z", "2026-01-01T00:02:00Z"],
        utc=True,
    )
    _write_parquet(
        tmp_path / "normalized" / "spot" / "klines" / "BTCUSDT-1m.parquet",
        _base_metadata(
            pd.DataFrame(
                {
                    "timestamp": kline_timestamps,
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.5, 101.5],
                    "volume": [10.0, 11.0],
                }
            ),
            symbol="BTCUSDT",
            market_family="spot",
            data_type="klines",
            interval="1m",
        ),
    )

    broken_event = _base_metadata(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1min"),
                "bid_price": [100.0, 101.0],
                "bid_qty": [10.0, 10.5],
                "ask_price": [100.2, 101.2],
                "ask_qty": [9.0, 9.5],
            }
        ),
        symbol="BTCUSDT",
        market_family="futures_um",
        data_type="bookTicker",
        interval=None,
    ).drop(columns=["bid_qty"])
    _write_parquet(tmp_path / "normalized" / "futures_um" / "bookTicker" / "BTCUSDT.parquet", broken_event)

    manifest = run_binance_public_qa(tmp_path)

    assert manifest["summary"]["table_fail_count"] == 2
    spot_entry = next(entry for entry in manifest["tables"] if entry["market_family"] == "spot")
    assert spot_entry["checks"]["strict_1m_continuity"] is False
    assert spot_entry["checks"]["gap_count_gt_1m"] == 1

    book_entry = next(entry for entry in manifest["tables"] if entry["data_type"] == "bookTicker")
    assert book_entry["checks"]["required_columns_present"] is False
    assert "bid_qty" in book_entry["checks"]["missing_required_columns"]
    assert book_entry["checks"]["event_stream_aggregatable"] is False


def test_large_event_stream_qa_uses_streaming_bounded_memory_checks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(qa_module, "DUPLICATE_EXACT_CHECK_ROW_LIMIT", 1)
    timestamps = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:01Z",
            "2026-01-01T00:00:01Z",
        ],
        utc=True,
    )
    _write_parquet(
        tmp_path / "normalized" / "spot" / "bookTicker" / "BTCUSDT.parquet",
        _base_metadata(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "bid_price": [100.0, 101.0, 101.0],
                    "bid_qty": [10.0, 11.0, 11.0],
                    "ask_price": [100.2, 101.2, 101.2],
                    "ask_qty": [9.0, 10.0, 10.0],
                }
            ),
            symbol="BTCUSDT",
            market_family="spot",
            data_type="bookTicker",
            interval=None,
        ),
    )

    manifest = run_binance_public_qa(tmp_path)

    entry = manifest["tables"][0]
    assert entry["checks"]["large_table_check_mode"] == "streaming_bounded_memory"
    assert entry["checks"]["duplicate_timestamp_symbol_rows"] == 1
    assert entry["checks"]["no_duplicate_timestamp_symbol_rows"] is False
