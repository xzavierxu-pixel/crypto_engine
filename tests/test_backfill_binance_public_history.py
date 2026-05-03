from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scripts.data.step1_acquire.backfill_binance_public_history import (
    DownloadRequest,
    DownloadResult,
    _determine_exit_code,
    _write_manifests,
    build_download_requests,
)
from src.core.config import load_settings


def _sample_request(tmp_path: Path, **overrides: object) -> DownloadRequest:
    payload = {
        "market_family": "spot",
        "data_type": "klines",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "granularity": "monthly",
        "period_label": "2026-01",
        "url": "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2026-01.zip",
        "checksum_url": "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2026-01.zip.CHECKSUM",
        "raw_dir": tmp_path / "raw",
        "object_key": "data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2026-01.zip",
        "object_prefix": "data/spot/monthly/klines/BTCUSDT/1m/",
        "filename": "BTCUSDT-1m-2026-01.zip",
    }
    payload.update(overrides)
    return DownloadRequest(**payload)


def test_build_download_requests_uses_monthly_for_full_months_and_daily_for_open_month_tail(tmp_path: Path) -> None:
    settings = load_settings()

    requests_to_run = build_download_requests(
        output_root=tmp_path,
        backfill_config=settings.data_backfill,
        as_of_date=date(2026, 4, 11),
    )

    spot_monthly = [
        request
        for request in requests_to_run
        if request.market_family == "spot"
        and request.data_type == "klines"
        and request.granularity == "monthly"
    ]
    spot_daily = [
        request
        for request in requests_to_run
        if request.market_family == "spot"
        and request.data_type == "klines"
        and request.granularity == "daily"
    ]

    assert [request.period_label for request in spot_monthly] == ["2026-01", "2026-02", "2026-03"]
    assert spot_daily[0].period_label == "2026-04-01"
    assert spot_daily[-1].period_label == "2026-04-10"


def test_build_download_requests_generates_expected_paths_for_um_metrics_and_option(tmp_path: Path) -> None:
    settings = load_settings()

    requests_to_run = build_download_requests(
        output_root=tmp_path,
        backfill_config=settings.data_backfill,
        as_of_date=date(2026, 4, 11),
    )

    metrics_request = next(
        request
        for request in requests_to_run
        if request.market_family == "futures_um"
        and request.data_type == "metrics"
        and request.period_label == "2026-04-01"
    )
    option_request = next(
        request
        for request in requests_to_run
        if request.market_family == "option"
        and request.data_type == "BVOLIndex"
        and request.period_label == "2026-04-01"
    )

    assert "/futures/um/daily/metrics/BTCUSDT/" in metrics_request.url
    assert metrics_request.interval is None
    assert metrics_request.raw_dir == tmp_path / "raw" / "futures_um" / "metrics" / "BTCUSDT" / "daily" / "2026-04-01"
    spot_request = next(
        request
        for request in requests_to_run
        if request.market_family == "spot"
        and request.data_type == "klines"
        and request.period_label == "2026-01"
    )
    um_mark_request = next(
        request
        for request in requests_to_run
        if request.market_family == "futures_um"
        and request.data_type == "markPriceKlines"
        and request.period_label == "2026-01"
    )

    assert option_request.url.endswith("/option/daily/BVOLIndex/BTCBVOLUSDT/BTCBVOLUSDT-BVOLIndex-2026-04-01.zip")
    assert spot_request.url.endswith("/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2026-01.zip")
    assert um_mark_request.url.endswith(
        "/futures/um/monthly/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2026-01.zip"
    )


def test_build_download_requests_uses_daily_history_for_daily_only_sources(tmp_path: Path) -> None:
    settings = load_settings()

    requests_to_run = build_download_requests(
        output_root=tmp_path,
        backfill_config=settings.data_backfill,
        as_of_date=date(2026, 4, 11),
    )

    metrics_daily = [
        request
        for request in requests_to_run
        if request.market_family == "futures_um"
        and request.data_type == "metrics"
        and request.granularity == "daily"
    ]
    bvol_daily = [
        request
        for request in requests_to_run
        if request.market_family == "option"
        and request.data_type == "BVOLIndex"
        and request.granularity == "daily"
    ]

    assert metrics_daily[0].period_label == "2026-01-01"
    assert metrics_daily[-1].period_label == "2026-04-10"
    assert bvol_daily[0].period_label == "2026-01-01"
    assert bvol_daily[-1].period_label == "2026-04-10"


def test_build_download_requests_keeps_funding_rate_and_book_ticker_as_desired_requests(tmp_path: Path) -> None:
    settings = load_settings()

    requests_to_run = build_download_requests(
        output_root=tmp_path,
        backfill_config=settings.data_backfill,
        as_of_date=date(2026, 4, 11),
    )

    funding_daily = [
        request
        for request in requests_to_run
        if request.market_family == "futures_um"
        and request.data_type == "fundingRate"
        and request.granularity == "daily"
    ]
    um_book_ticker_requests = [
        request
        for request in requests_to_run
        if request.market_family == "futures_um"
        and request.data_type == "bookTicker"
    ]
    assert len(funding_daily) > 0
    assert len(um_book_ticker_requests) > 0
    assert um_book_ticker_requests[0].url.startswith("https://data.binance.vision/data/futures/um/")


def test_write_manifests_records_missing_and_checksum_failures(tmp_path: Path) -> None:
    settings = load_settings()
    ok_result = DownloadResult(
        request=_sample_request(tmp_path),
        status="downloaded",
        checksum_status="verified",
        extracted_files=[str((tmp_path / "raw" / "file.csv").resolve())],
        expected_checksum="abc",
        actual_checksum="abc",
    )
    missing_result = DownloadResult(
        request=_sample_request(tmp_path, period_label="2026-04-01", granularity="daily"),
        status="missing",
        checksum_status="not_attempted",
        extracted_files=[],
        message="zip missing",
    )
    mismatch_result = DownloadResult(
        request=_sample_request(tmp_path, market_family="futures_um", data_type="fundingRate", interval=None),
        status="checksum_mismatch",
        checksum_status="mismatch",
        extracted_files=[],
        expected_checksum="expected",
        actual_checksum="actual",
        message="checksum mismatch",
    )

    _write_manifests(
        output_root=tmp_path,
        results=[ok_result, missing_result, mismatch_result],
        config=settings.data_backfill,
        as_of_date=date(2026, 4, 11),
    )

    download_manifest = json.loads((tmp_path / "manifests" / "download_manifest.json").read_text(encoding="utf-8"))
    checksum_manifest = json.loads((tmp_path / "manifests" / "file_checksums.json").read_text(encoding="utf-8"))

    assert download_manifest["summary"]["total_requests"] == 3
    assert download_manifest["summary"]["failed_requests"] == 2
    assert download_manifest["zip_retention"] == "deleted_after_extract"
    assert len(download_manifest["downloaded"]) == 1
    assert len(download_manifest["missing_or_failed"]) == 2
    assert download_manifest["unavailable_by_listing"] == []
    assert checksum_manifest["results"][0]["checksum_status"] == "verified"
    assert checksum_manifest["results"][2]["status"] == "checksum_mismatch"


def test_determine_exit_code_returns_non_zero_when_any_request_failed(tmp_path: Path) -> None:
    ok_result = DownloadResult(
        request=_sample_request(tmp_path),
        status="downloaded",
        checksum_status="verified",
        extracted_files=[],
    )
    failed_result = DownloadResult(
        request=_sample_request(tmp_path, period_label="2026-04-01", granularity="daily"),
        status="missing",
        checksum_status="not_attempted",
        extracted_files=[],
    )

    assert _determine_exit_code([ok_result]) == 0
    assert _determine_exit_code([ok_result, failed_result]) == 1


def test_unavailable_by_listing_is_reported_but_not_treated_as_execution_failure(tmp_path: Path) -> None:
    unavailable_result = DownloadResult(
        request=_sample_request(tmp_path),
        status="unavailable_by_listing",
        checksum_status="not_attempted",
        extracted_files=[],
        message="object not present in Binance bucket listing for prefix",
    )

    assert unavailable_result.is_error is False
    assert _determine_exit_code([unavailable_result]) == 0


def test_load_settings_exposes_data_backfill_config() -> None:
    settings = load_settings()

    assert settings.data_backfill.provider == "binance_public"
    assert settings.data_backfill.spot.symbols == ["BTCUSDT"]
    assert settings.data_backfill.futures_cm.symbols == ["BTCUSD_PERP"]
    assert settings.data_backfill.option.symbols["BVOLIndex"] == ["BTCBVOLUSDT"]
