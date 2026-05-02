from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import (  # noqa: E402
    DataBackfillConfig,
    DataBackfillMarketConfig,
    DataBackfillOptionConfig,
    load_settings,
)


BINANCE_VISION_ROOT = "https://data.binance.vision/data"
BINANCE_VISION_BUCKET_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CHECKSUM_SUFFIX = ".CHECKSUM"

SUPPORTED_DATA_TYPES: dict[str, dict[str, dict[str, Any]]] = {
    "spot": {
        "klines": {"granularities": ("monthly", "daily"), "interval_required": True},
        "aggTrades": {"granularities": ("monthly", "daily"), "interval_required": False},
        "trades": {"granularities": ("monthly", "daily"), "interval_required": False},
        "bookTicker": {"granularities": ("monthly", "daily"), "interval_required": False},
    },
    "futures_um": {
        "klines": {"granularities": ("monthly", "daily"), "interval_required": True},
        "markPriceKlines": {"granularities": ("monthly", "daily"), "interval_required": True},
        "indexPriceKlines": {"granularities": ("monthly", "daily"), "interval_required": True},
        "premiumIndexKlines": {"granularities": ("monthly", "daily"), "interval_required": True},
        "fundingRate": {"granularities": ("monthly", "daily"), "interval_required": False},
        "bookTicker": {"granularities": ("monthly", "daily"), "interval_required": False},
        "metrics": {"granularities": ("daily",), "interval_required": False},
        "aggTrades": {"granularities": ("monthly", "daily"), "interval_required": False},
        "trades": {"granularities": ("monthly", "daily"), "interval_required": False},
        "bookDepth": {"granularities": ("daily",), "interval_required": False},
        "liquidationSnapshot": {"granularities": ("daily",), "interval_required": False},
    },
    "futures_cm": {
        "klines": {"granularities": ("monthly", "daily"), "interval_required": True},
        "markPriceKlines": {"granularities": ("monthly", "daily"), "interval_required": True},
        "indexPriceKlines": {"granularities": ("monthly", "daily"), "interval_required": True},
        "premiumIndexKlines": {"granularities": ("monthly", "daily"), "interval_required": True},
        "fundingRate": {"granularities": ("monthly", "daily"), "interval_required": False},
        "bookTicker": {"granularities": ("monthly", "daily"), "interval_required": False},
        "metrics": {"granularities": ("daily",), "interval_required": False},
        "aggTrades": {"granularities": ("monthly", "daily"), "interval_required": False},
        "trades": {"granularities": ("monthly", "daily"), "interval_required": False},
        "bookDepth": {"granularities": ("daily",), "interval_required": False},
        "liquidationSnapshot": {"granularities": ("daily",), "interval_required": False},
    },
    "option": {
        "BVOLIndex": {"granularities": ("daily",), "interval_required": False},
        "EOHSummary": {"granularities": ("daily",), "interval_required": False},
    },
}

MARKET_URL_SEGMENTS = {
    "spot": ("spot",),
    "futures_um": ("futures", "um"),
    "futures_cm": ("futures", "cm"),
    "option": ("option",),
}


@dataclass(frozen=True)
class PeriodSpec:
    granularity: str
    label: str
    day: date | None = None


@dataclass(frozen=True)
class DownloadRequest:
    market_family: str
    data_type: str
    symbol: str
    interval: str | None
    granularity: str
    period_label: str
    url: str
    checksum_url: str
    raw_dir: Path
    object_key: str
    object_prefix: str
    filename: str


@dataclass
class DownloadResult:
    request: DownloadRequest
    status: str
    checksum_status: str
    extracted_files: list[str]
    message: str | None = None
    expected_checksum: str | None = None
    actual_checksum: str | None = None
    zip_deleted: bool = True

    @property
    def is_error(self) -> bool:
        return self.status not in {"downloaded", "skipped", "unavailable_by_listing"}


def _month_start(day_value: date) -> date:
    return day_value.replace(day=1)


def _next_month(day_value: date) -> date:
    if day_value.month == 12:
        return date(day_value.year + 1, 1, 1)
    return date(day_value.year, day_value.month + 1, 1)


def _iter_days(start_date: date, end_date: date) -> list[date]:
    if end_date < start_date:
        return []
    current = start_date
    days: list[date] = []
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)
    return days


def _iter_month_starts(start_date: date, end_date: date) -> list[date]:
    if end_date < start_date:
        return []
    current = _month_start(start_date)
    months: list[date] = []
    while current <= end_date:
        months.append(current)
        current = _next_month(current)
    return months


def _latest_daily_available_date(as_of_date: date) -> date:
    return as_of_date - timedelta(days=1)


def _full_month_periods(start_date: date, end_date: date) -> list[PeriodSpec]:
    periods: list[PeriodSpec] = []
    for month_start in _iter_month_starts(start_date, end_date):
        month_end = _next_month(month_start) - timedelta(days=1)
        if month_start < start_date or month_end > end_date:
            continue
        periods.append(PeriodSpec(granularity="monthly", label=month_start.strftime("%Y-%m")))
    return periods


def _daily_periods(start_date: date, end_date: date) -> list[PeriodSpec]:
    return [
        PeriodSpec(granularity="daily", label=day_value.isoformat(), day=day_value)
        for day_value in _iter_days(start_date, end_date)
    ]


def _plan_periods(start_date: date, as_of_date: date, use_monthly: bool, use_daily_tail: bool) -> list[PeriodSpec]:
    latest_daily = _latest_daily_available_date(as_of_date)
    if latest_daily < start_date:
        return []

    periods: list[PeriodSpec] = []
    if use_monthly:
        periods.extend(_full_month_periods(start_date, latest_daily))

    if use_daily_tail:
        open_month_start = _month_start(as_of_date)
        daily_start = max(start_date, open_month_start)
        periods.extend(_daily_periods(daily_start, latest_daily))

    unique_periods = {(period.granularity, period.label): period for period in periods}
    return sorted(unique_periods.values(), key=lambda item: (item.label, item.granularity))


def _plan_periods_for_supported_granularities(
    start_date: date,
    as_of_date: date,
    *,
    supported_granularities: tuple[str, ...],
    use_monthly: bool,
    use_daily_tail: bool,
    use_daily_full_month_fallback: bool = False,
) -> list[PeriodSpec]:
    latest_daily = _latest_daily_available_date(as_of_date)
    if latest_daily < start_date:
        return []

    supports_monthly = "monthly" in supported_granularities
    supports_daily = "daily" in supported_granularities

    if supports_daily and not supports_monthly:
        return _daily_periods(start_date, latest_daily)

    periods = [
        period
        for period in _plan_periods(
            start_date=start_date,
            as_of_date=as_of_date,
            use_monthly=use_monthly,
            use_daily_tail=use_daily_tail,
        )
        if period.granularity in supported_granularities
    ]
    if use_daily_full_month_fallback and supports_daily:
        full_month_days: list[PeriodSpec] = []
        for monthly_period in [period for period in periods if period.granularity == "monthly"]:
            month_start = date.fromisoformat(f"{monthly_period.label}-01")
            month_end = _next_month(month_start) - timedelta(days=1)
            full_month_days.extend(_daily_periods(max(start_date, month_start), min(month_end, latest_daily)))
        periods.extend(full_month_days)
    unique_periods = {(period.granularity, period.label): period for period in periods}
    return sorted(unique_periods.values(), key=lambda item: (item.label, item.granularity))


def _build_filename(market_family: str, symbol: str, data_type: str, period_label: str, interval: str | None) -> str:
    if data_type.endswith("Klines") or data_type == "klines":
        if interval is None:
            raise ValueError(f"{data_type} filename requires interval.")
        return f"{symbol}-{interval}-{period_label}.zip"
    if interval:
        return f"{symbol}-{data_type}-{interval}-{period_label}.zip"
    return f"{symbol}-{data_type}-{period_label}.zip"


def _build_request(
    output_root: Path,
    market_family: str,
    data_type: str,
    symbol: str,
    interval: str | None,
    period: PeriodSpec,
) -> DownloadRequest:
    url_parts = [BINANCE_VISION_ROOT, *MARKET_URL_SEGMENTS[market_family], period.granularity, data_type, symbol]
    raw_dir = output_root / "raw" / market_family / data_type / symbol
    if interval:
        url_parts.append(interval)
        raw_dir = raw_dir / interval
    filename = _build_filename(
        market_family=market_family,
        symbol=symbol,
        data_type=data_type,
        period_label=period.label,
        interval=interval,
    )
    object_prefix = "/".join((*MARKET_URL_SEGMENTS[market_family], period.granularity, data_type, symbol))
    if interval:
        object_prefix = f"{object_prefix}/{interval}"
    object_key = f"data/{object_prefix}/{filename}"
    url = "/".join((*url_parts, filename))
    checksum_url = f"{url}{CHECKSUM_SUFFIX}"
    raw_dir = raw_dir / period.granularity / period.label
    return DownloadRequest(
        market_family=market_family,
        data_type=data_type,
        symbol=symbol,
        interval=interval,
        granularity=period.granularity,
        period_label=period.label,
        url=url,
        checksum_url=checksum_url,
        raw_dir=raw_dir,
        object_key=object_key,
        object_prefix=f"data/{object_prefix}/",
        filename=filename,
    )


def _build_market_requests(
    output_root: Path,
    market_family: str,
    market_config: DataBackfillMarketConfig,
    start_date: date,
    as_of_date: date,
    use_monthly: bool,
    use_daily_tail: bool,
    use_daily_full_month_fallback: bool,
) -> list[DownloadRequest]:
    if not market_config.enabled:
        return []

    requests_to_run: list[DownloadRequest] = []
    for data_type, data_type_config in market_config.data_types.items():
        support = SUPPORTED_DATA_TYPES[market_family].get(data_type)
        if support is None:
            raise ValueError(f"Unsupported data_type '{data_type}' for market_family '{market_family}'.")
        periods = _plan_periods_for_supported_granularities(
            start_date,
            as_of_date,
            supported_granularities=support["granularities"],
            use_monthly=use_monthly,
            use_daily_tail=use_daily_tail,
            use_daily_full_month_fallback=use_daily_full_month_fallback,
        )
        intervals = list(data_type_config.get("intervals", [])) if support["interval_required"] else [None]
        if support["interval_required"] and not intervals:
            raise ValueError(f"data_type '{data_type}' for market_family '{market_family}' requires intervals.")
        for symbol in market_config.symbols:
            for interval in intervals:
                for period in periods:
                    if period.granularity not in support["granularities"]:
                        continue
                    requests_to_run.append(
                        _build_request(
                            output_root=output_root,
                            market_family=market_family,
                            data_type=data_type,
                            symbol=symbol,
                            interval=interval,
                            period=period,
                        )
                    )
    return requests_to_run


def _build_option_requests(
    output_root: Path,
    option_config: DataBackfillOptionConfig,
    start_date: date,
    as_of_date: date,
    use_monthly: bool,
    use_daily_tail: bool,
) -> list[DownloadRequest]:
    if not option_config.enabled:
        return []

    requests_to_run: list[DownloadRequest] = []
    for data_type, symbols in option_config.symbols.items():
        support = SUPPORTED_DATA_TYPES["option"].get(data_type)
        if support is None:
            raise ValueError(f"Unsupported option data_type '{data_type}'.")
        periods = _plan_periods_for_supported_granularities(
            start_date,
            as_of_date,
            supported_granularities=support["granularities"],
            use_monthly=use_monthly,
            use_daily_tail=use_daily_tail,
            use_daily_full_month_fallback=False,
        )
        for symbol in symbols:
            for period in periods:
                if period.granularity not in support["granularities"]:
                    continue
                requests_to_run.append(
                    _build_request(
                        output_root=output_root,
                        market_family="option",
                        data_type=data_type,
                        symbol=symbol,
                        interval=None,
                        period=period,
                    )
                )
    return requests_to_run


def build_download_requests(
    output_root: Path,
    backfill_config: DataBackfillConfig,
    as_of_date: date,
) -> list[DownloadRequest]:
    start_date = date.fromisoformat(backfill_config.start_date)
    requests_to_run: list[DownloadRequest] = []
    requests_to_run.extend(
        _build_market_requests(
            output_root,
            "spot",
            backfill_config.spot,
            start_date,
            as_of_date,
            backfill_config.use_monthly_for_full_months,
            backfill_config.use_daily_for_open_month_tail,
            backfill_config.use_daily_for_full_month_fallback,
        )
    )
    requests_to_run.extend(
        _build_market_requests(
            output_root,
            "futures_um",
            backfill_config.futures_um,
            start_date,
            as_of_date,
            backfill_config.use_monthly_for_full_months,
            backfill_config.use_daily_for_open_month_tail,
            backfill_config.use_daily_for_full_month_fallback,
        )
    )
    requests_to_run.extend(
        _build_market_requests(
            output_root,
            "futures_cm",
            backfill_config.futures_cm,
            start_date,
            as_of_date,
            backfill_config.use_monthly_for_full_months,
            backfill_config.use_daily_for_open_month_tail,
            backfill_config.use_daily_for_full_month_fallback,
        )
    )
    requests_to_run.extend(
        _build_option_requests(
            output_root,
            backfill_config.option,
            start_date,
            as_of_date,
            backfill_config.use_monthly_for_full_months,
            backfill_config.use_daily_for_open_month_tail,
        )
    )
    return requests_to_run


def _parse_checksum(payload: str) -> str:
    return payload.strip().split()[0]


def _compute_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _extract_archive(content: bytes, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_files: list[str] = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            archive.extract(member, path=target_dir)
            extracted_files.append(str((target_dir / member.filename).resolve()))
    if not extracted_files:
        raise ValueError("Archive did not contain any files.")
    return extracted_files


def _existing_extracted_files(target_dir: Path) -> list[str]:
    if not target_dir.exists():
        return []
    files = sorted(path for path in target_dir.rglob("*.csv") if path.is_file())
    return [str(path.resolve()) for path in files]


def _download_content(session: requests.Session, url: str) -> bytes:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def _list_bucket_keys(session: requests.Session, prefix: str) -> set[str]:
    keys: set[str] = set()
    marker: str | None = None
    while True:
        params = {"prefix": prefix}
        if marker:
            params["marker"] = marker
        response = session.get(BINANCE_VISION_BUCKET_LIST_URL, params=params, timeout=60)
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        batch = [element.text for element in root.findall("s3:Contents/s3:Key", namespace) if element.text]
        keys.update(batch)
        is_truncated = root.findtext("s3:IsTruncated", default="false", namespaces=namespace) == "true"
        if not is_truncated or not batch:
            break
        marker = batch[-1]
    return keys


def _discover_available_requests(
    session: requests.Session,
    requests_to_run: list[DownloadRequest],
) -> tuple[list[DownloadRequest], list[DownloadResult]]:
    by_prefix: dict[str, list[DownloadRequest]] = defaultdict(list)
    for request in requests_to_run:
        by_prefix[request.object_prefix].append(request)

    executable_requests: list[DownloadRequest] = []
    unavailable_results: list[DownloadResult] = []
    for prefix, grouped_requests in by_prefix.items():
        available_keys = _list_bucket_keys(session, prefix)
        for request in grouped_requests:
            if request.object_key in available_keys:
                executable_requests.append(request)
                continue
            unavailable_results.append(
                DownloadResult(
                    request=request,
                    status="unavailable_by_listing",
                    checksum_status="not_attempted",
                    extracted_files=[],
                    message="object not present in Binance bucket listing for prefix",
                )
            )
    return executable_requests, unavailable_results


def execute_request(
    session: requests.Session,
    request: DownloadRequest,
    verify_checksum: bool,
) -> DownloadResult:
    existing_files = _existing_extracted_files(request.raw_dir)
    if existing_files:
        return DownloadResult(
            request=request,
            status="skipped",
            checksum_status="not_rechecked_existing_extract",
            extracted_files=existing_files,
            message="raw csv already extracted",
        )
    zip_content: bytes
    try:
        zip_content = _download_content(session, request.url)
    except requests.HTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code == 404:
            return DownloadResult(
                request=request,
                status="missing",
                checksum_status="not_attempted",
                extracted_files=[],
                message="zip missing",
            )
        return DownloadResult(
            request=request,
            status="error",
            checksum_status="not_attempted",
            extracted_files=[],
            message=f"zip download failed: {exc}",
        )

    expected_checksum: str | None = None
    actual_checksum = _compute_sha256(zip_content)
    checksum_status = "skipped"
    if verify_checksum:
        try:
            checksum_payload = _download_content(session, request.checksum_url).decode("utf-8")
            expected_checksum = _parse_checksum(checksum_payload)
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code == 404:
                return DownloadResult(
                    request=request,
                    status="checksum_missing",
                    checksum_status="missing",
                    extracted_files=[],
                    message="checksum missing",
                    actual_checksum=actual_checksum,
                )
            return DownloadResult(
                request=request,
                status="error",
                checksum_status="error",
                extracted_files=[],
                message=f"checksum download failed: {exc}",
                actual_checksum=actual_checksum,
            )
        if expected_checksum.lower() != actual_checksum.lower():
            return DownloadResult(
                request=request,
                status="checksum_mismatch",
                checksum_status="mismatch",
                extracted_files=[],
                message="checksum mismatch",
                expected_checksum=expected_checksum,
                actual_checksum=actual_checksum,
            )
        checksum_status = "verified"

    try:
        extracted_files = _extract_archive(zip_content, request.raw_dir)
    except (zipfile.BadZipFile, ValueError) as exc:
        return DownloadResult(
            request=request,
            status="extract_failed",
            checksum_status=checksum_status,
            extracted_files=[],
            message=str(exc),
            expected_checksum=expected_checksum,
            actual_checksum=actual_checksum,
        )

    return DownloadResult(
        request=request,
        status="downloaded",
        checksum_status=checksum_status,
        extracted_files=extracted_files,
        expected_checksum=expected_checksum,
        actual_checksum=actual_checksum,
    )


def _manifest_summary(results: list[DownloadResult]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    for result in results:
        by_status[result.status] = by_status.get(result.status, 0) + 1
    error_statuses = [result.status for result in results if result.is_error]
    return {
        "total_requests": len(results),
        "successful_requests": by_status.get("downloaded", 0),
        "failed_requests": len(error_statuses),
        "status_counts": by_status,
        "has_errors": bool(error_statuses),
    }


def _result_to_manifest_entry(result: DownloadResult) -> dict[str, Any]:
    request_payload = asdict(result.request)
    request_payload["raw_dir"] = str(result.request.raw_dir.resolve())
    return {
        "request": request_payload,
        "status": result.status,
        "checksum_status": result.checksum_status,
        "expected_checksum": result.expected_checksum,
        "actual_checksum": result.actual_checksum,
        "zip_deleted": result.zip_deleted,
        "message": result.message,
        "extracted_files": result.extracted_files,
    }


def _write_manifests(output_root: Path, results: list[DownloadResult], config: DataBackfillConfig, as_of_date: date) -> None:
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    summary = _manifest_summary(results)
    downloaded_entries = [
        _result_to_manifest_entry(result)
        for result in results
        if result.status == "downloaded"
    ]
    unavailable_entries = [
        _result_to_manifest_entry(result)
        for result in results
        if result.status == "unavailable_by_listing"
    ]
    missing_entries = [
        _result_to_manifest_entry(result)
        for result in results
        if result.status in {"missing", "checksum_missing", "checksum_mismatch", "extract_failed", "error"}
    ]

    download_manifest = {
        "generated_at": generated_at,
        "provider": config.provider,
        "start_date": config.start_date,
        "as_of_date": as_of_date.isoformat(),
        "zip_retention": "deleted_after_extract",
        "summary": summary,
        "downloaded": downloaded_entries,
        "unavailable_by_listing": unavailable_entries,
        "missing_or_failed": missing_entries,
    }
    (manifests_dir / "download_manifest.json").write_text(
        json.dumps(download_manifest, indent=2),
        encoding="utf-8",
    )

    checksum_manifest = {
        "generated_at": generated_at,
        "results": [
            {
                "market_family": result.request.market_family,
                "data_type": result.request.data_type,
                "symbol": result.request.symbol,
                "interval": result.request.interval,
                "granularity": result.request.granularity,
                "period_label": result.request.period_label,
                "checksum_status": result.checksum_status,
                "expected_checksum": result.expected_checksum,
                "actual_checksum": result.actual_checksum,
                "status": result.status,
            }
            for result in results
        ],
    }
    (manifests_dir / "file_checksums.json").write_text(
        json.dumps(checksum_manifest, indent=2),
        encoding="utf-8",
    )


def _determine_exit_code(results: list[DownloadResult]) -> int:
    return 1 if any(result.is_error for result in results) else 0


def run_backfill(settings_path: Path, output_root: Path | None, as_of_date: date) -> int:
    settings = load_settings(settings_path)
    backfill_config = settings.data_backfill
    resolved_output_root = output_root or Path(settings.paths.artifacts_dir) / "data" / "binance_public"

    requests_to_run = build_download_requests(
        output_root=resolved_output_root,
        backfill_config=backfill_config,
        as_of_date=as_of_date,
    )

    with requests.Session() as session:
        executable_requests, unavailable_results = _discover_available_requests(session, requests_to_run)
        execution_results = [
            execute_request(session=session, request=request, verify_checksum=backfill_config.verify_checksum)
            for request in executable_requests
        ]
        results = execution_results + unavailable_results

    _write_manifests(resolved_output_root, results, backfill_config, as_of_date)

    summary = _manifest_summary(results)
    print(f"output_root={resolved_output_root.resolve()}")
    print(f"desired_requests={len(requests_to_run)}")
    print(f"downloadable_requests={len(executable_requests)}")
    print(f"unavailable_by_listing={len(unavailable_results)}")
    print(f"total_requests={summary['total_requests']}")
    print(f"successful_requests={summary['successful_requests']}")
    print(f"failed_requests={summary['failed_requests']}")
    return _determine_exit_code(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Binance public history archives.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to the project settings file.")
    parser.add_argument("--output-root", help="Override output root directory.")
    parser.add_argument("--as-of-date", help="UTC date used to derive monthly/daily windows in YYYY-MM-DD.")
    args = parser.parse_args()

    as_of_date = date.fromisoformat(args.as_of_date) if args.as_of_date else datetime.now(timezone.utc).date()
    output_root = Path(args.output_root) if args.output_root else None
    raise SystemExit(run_backfill(Path(args.settings), output_root=output_root, as_of_date=as_of_date))


if __name__ == "__main__":
    main()
