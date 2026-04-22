from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.normalize_binance_public_history import run_normalize
from src.data.binance_public import normalizer
from src.data.binance_public.normalizer import normalize_binance_public_history


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_normalize_binance_public_history_writes_parquet_outputs_and_manifest(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"

    _write_text(
        raw_root / "spot" / "klines" / "BTCUSDT" / "1m" / "monthly" / "2026-01" / "BTCUSDT-1m-2026-01.csv",
        "1767225600000000,87648.21,87648.22,87632.74,87648.00,4.08,1767225659999999,357625.77,1193,1.79,157651.51,0\n",
    )
    _write_text(
        raw_root / "futures_um" / "fundingRate" / "BTCUSDT" / "monthly" / "2026-01" / "BTCUSDT-fundingRate-2026-01.csv",
        "calc_time,funding_interval_hours,last_funding_rate\n1767225600008,8,0.00010000\n",
    )
    _write_text(
        raw_root / "futures_um" / "metrics" / "BTCUSDT" / "daily" / "2026-04-01" / "BTCUSDT-metrics-2026-04-01.csv",
        "create_time,symbol,sum_open_interest,sum_open_interest_value\n2026-04-01 00:05:00,BTCUSDT,89202.368,6079801476.7232\n",
    )

    manifest = normalize_binance_public_history(tmp_path)

    assert len(manifest["normalized_outputs"]) == 3
    schema_manifest_path = tmp_path / "manifests" / "schema_manifest.json"
    assert schema_manifest_path.exists()

    payload = json.loads(schema_manifest_path.read_text(encoding="utf-8"))
    assert payload["source_version"] == "v1"
    assert payload["unsupported_files"] == []

    spot_path = tmp_path / "normalized" / "spot" / "klines" / "BTCUSDT-1m.parquet"
    funding_path = tmp_path / "normalized" / "futures_um" / "fundingRate" / "BTCUSDT.parquet"
    metrics_path = tmp_path / "normalized" / "futures_um" / "metrics" / "BTCUSDT.parquet"
    assert spot_path.exists()
    assert funding_path.exists()
    assert metrics_path.exists()

    spot_frame = pd.read_parquet(spot_path)
    funding_frame = pd.read_parquet(funding_path)
    metrics_frame = pd.read_parquet(metrics_path)

    assert spot_frame["timestamp"].iloc[0].isoformat() == "2026-01-01T00:00:00+00:00"
    assert spot_frame["market_family"].iloc[0] == "spot"
    assert spot_frame["data_type"].iloc[0] == "klines"
    assert funding_frame["funding_interval_hours"].iloc[0] == 8
    assert funding_frame["data_type"].iloc[0] == "fundingRate"
    assert metrics_frame["symbol"].iloc[0] == "BTCUSDT"
    assert metrics_frame["sum_open_interest"].iloc[0] == 89202.368
    assert "qa" in payload["normalized_outputs"][0]["schema"]


def test_normalize_binance_public_history_records_unsupported_files(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "raw" / "futures_um" / "unknownType" / "BTCUSDT" / "daily" / "2026-04-01" / "BTCUSDT-unknownType-2026-04-01.csv",
        "a,b,c\n1,1,2\n",
    )

    manifest = normalize_binance_public_history(tmp_path)

    assert manifest["normalized_outputs"] == []
    assert len(manifest["unsupported_files"]) == 1


def test_normalize_binance_public_history_supports_additional_types_and_checksum_metadata(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    manifests_root = tmp_path / "manifests"

    agg_path = raw_root / "spot" / "aggTrades" / "BTCUSDT" / "daily" / "2026-04-01" / "BTCUSDT-aggTrades-2026-04-01.csv"
    trade_path = raw_root / "spot" / "trades" / "BTCUSDT" / "daily" / "2026-04-01" / "BTCUSDT-trades-2026-04-01.csv"
    cm_kline_path = raw_root / "futures_cm" / "klines" / "BTCUSD_PERP" / "1m" / "monthly" / "2026-01" / "BTCUSD_PERP-1m-2026-01.csv"
    bvol_path = raw_root / "option" / "BVOLIndex" / "BTCBVOLUSDT" / "daily" / "2026-04-01" / "BTCBVOLUSDT-BVOLIndex-2026-04-01.csv"
    depth_path = raw_root / "futures_um" / "bookDepth" / "BTCUSDT" / "daily" / "2026-04-01" / "BTCUSDT-bookDepth-2026-04-01.csv"

    _write_text(
        agg_path,
        "3921930718,68284.49,0.00007,6172995294,6172995294,1775001600062762,False,True\n",
    )
    _write_text(
        trade_path,
        "6172995294,68284.49,0.00007,4.77991430,1775001600062762,False,True\n",
    )
    _write_text(
        cm_kline_path,
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n1767225600000,87500.7,87505.0,87483.8,87504.9,3192,1767225659999,3.64848660,135,1764,2.01632785,0\n",
    )
    _write_text(
        bvol_path,
        "calc_time,symbol,base_asset,quote_asset,index_value\n1775001600002,BTCBVOLUSDT,BTCBVOL,USDT,54.0601\n",
    )
    _write_text(
        depth_path,
        "timestamp,percentage,depth,notional\n2026-04-01 00:00:08,-5.00,8003.53600000,533657412.37580000\n",
    )

    download_manifest = {
        "downloaded": [],
        "unavailable_by_listing": [],
        "missing_or_failed": [],
    }
    download_manifest["downloaded"].append(
        {
            "status": "downloaded",
            "checksum_status": "verified",
            "expected_checksum": "abc",
            "actual_checksum": "abc",
            "extracted_files": [str(agg_path.resolve())],
        }
    )
    download_manifest["downloaded"].append(
        {
            "status": "downloaded",
            "checksum_status": "verified",
            "expected_checksum": "def",
            "actual_checksum": "def",
            "extracted_files": [str(cm_kline_path.resolve())],
        }
    )
    manifests_root.mkdir(parents=True, exist_ok=True)
    (manifests_root / "download_manifest.json").write_text(json.dumps(download_manifest), encoding="utf-8")

    manifest = normalize_binance_public_history(tmp_path)

    output_paths = {entry["data_type"]: Path(entry["output_path"]) for entry in manifest["normalized_outputs"]}
    assert "aggTrades" in output_paths
    assert "trades" in output_paths
    assert "BVOLIndex" in output_paths
    assert "bookDepth" in output_paths
    assert any(entry["market_family"] == "futures_cm" and entry["data_type"] == "klines" for entry in manifest["normalized_outputs"])

    agg_frame = pd.read_parquet(output_paths["aggTrades"])
    cm_frame = pd.read_parquet(next(Path(entry["output_path"]) for entry in manifest["normalized_outputs"] if entry["market_family"] == "futures_cm"))
    bvol_frame = pd.read_parquet(output_paths["BVOLIndex"])
    depth_frame = pd.read_parquet(output_paths["bookDepth"])

    assert agg_frame["checksum_status"].iloc[0] == "verified"
    assert agg_frame["download_status"].iloc[0] == "downloaded"
    assert cm_frame["checksum_status"].iloc[0] == "verified"
    assert bvol_frame["index_value"].iloc[0] == 54.0601
    assert depth_frame["percentage"].iloc[0] == -5.0


def test_run_normalize_also_writes_qa_manifest(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_text(
        raw_root / "spot" / "klines" / "BTCUSDT" / "1m" / "daily" / "2026-04-01" / "BTCUSDT-1m-2026-04-01.csv",
        "\n".join(
            [
                "1775001600000000,87648.21,87648.22,87632.74,87648.00,4.08,1775001659999999,357625.77,1193,1.79,157651.51,0",
                "1775001660000000,87648.00,87660.00,87640.00,87655.00,5.08,1775001719999999,357700.77,1200,2.00,157700.51,0",
            ]
        )
        + "\n",
    )

    result = run_normalize("config/settings.yaml", output_root=tmp_path)

    assert len(result["normalize_manifest"]["normalized_outputs"]) == 1
    assert result["qa_manifest"]["summary"]["table_count"] == 1
    assert result["qa_manifest"]["summary"]["table_fail_count"] == 0
    assert (tmp_path / "manifests" / "schema_manifest.json").exists()
    assert (tmp_path / "manifests" / "qa_manifest.json").exists()


def test_normalize_binance_public_history_handles_chunked_event_streams_across_multiple_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    monkeypatch.setattr(normalizer, "CSV_CHUNK_SIZE", 2)

    _write_text(
        raw_root / "spot" / "aggTrades" / "BTCUSDT" / "daily" / "2026-04-01" / "BTCUSDT-aggTrades-2026-04-01.csv",
        "\n".join(
            [
                "3921930718,68284.49,0.00007,6172995294,6172995294,1775001600062762,False,True",
                "3921930719,68284.50,0.10000,6172995295,6172995295,1775001601062762,True,True",
                "3921930720,68284.51,0.20000,6172995296,6172995296,1775001602062762,False,True",
            ]
        )
        + "\n",
    )
    _write_text(
        raw_root / "spot" / "trades" / "BTCUSDT" / "daily" / "2026-04-01" / "BTCUSDT-trades-2026-04-01.csv",
        "\n".join(
            [
                "6172995294,68284.49,0.00007,4.77991430,1775001600062762,False,True",
                "6172995295,68284.50,0.10000,6828.45000000,1775001601062762,True,True",
                "6172995296,68284.51,0.20000,13656.90200000,1775001602062762,False,True",
            ]
        )
        + "\n",
    )

    manifest = normalize_binance_public_history(tmp_path)

    agg_output = next(Path(entry["output_path"]) for entry in manifest["normalized_outputs"] if entry["data_type"] == "aggTrades")
    trade_output = next(Path(entry["output_path"]) for entry in manifest["normalized_outputs"] if entry["data_type"] == "trades")
    agg_frame = pd.read_parquet(agg_output)
    trade_frame = pd.read_parquet(trade_output)

    assert len(agg_frame) == 3
    assert len(trade_frame) == 3
    assert agg_frame["timestamp"].is_monotonic_increasing
    assert trade_frame["timestamp"].is_monotonic_increasing
    assert agg_frame["price"].dtype == "float64"
    assert trade_frame["trade_id"].dtype.name == "Int64"
