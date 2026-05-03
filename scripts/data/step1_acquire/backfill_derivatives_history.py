from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'src').is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.derivatives.public_data import (  # noqa: E402
    _fetch_binance_basis,
    _fetch_binance_funding,
    _fetch_binance_oi,
    _fetch_deribit_options_proxy,
    _normalize_oi_records,
    _parse_utc_date,
    _to_milliseconds,
)


@dataclass(frozen=True)
class WindowSpec:
    start: datetime
    end: datetime

    @property
    def start_label(self) -> str:
        return self.start.date().isoformat()

    @property
    def end_label(self) -> str:
        return self.end.date().isoformat()


def _iter_windows(start: datetime, end: datetime, chunk_days: int) -> list[WindowSpec]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be > 0.")
    if end < start:
        raise ValueError("end must be >= start.")

    windows: list[WindowSpec] = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=chunk_days) - timedelta(days=1), end)
        windows.append(WindowSpec(start=cursor, end=window_end))
        cursor = window_end + timedelta(days=1)
    return windows


def _source_directories(output_root: Path) -> dict[str, Path]:
    return {
        "funding": output_root / "funding",
        "basis": output_root / "basis",
        "oi": output_root / "oi",
        "options": output_root / "options",
        "manifests": output_root / "manifests",
        "chunks": output_root / "chunks",
    }


def _final_output_path(output_root: Path, source_name: str, start_label: str, end_label: str) -> Path:
    directories = _source_directories(output_root)
    if source_name == "funding":
        return directories["funding"] / f"binance_btcusdt_funding_{start_label}_{end_label}.parquet"
    if source_name == "basis":
        return directories["basis"] / f"binance_btcusdt_basis_{start_label}_{end_label}.parquet"
    if source_name == "oi":
        return directories["oi"] / f"binance_btcusdt_oi_{start_label}_{end_label}.parquet"
    if source_name == "options":
        return directories["options"] / f"deribit_btc_iv_{start_label}_{end_label}.parquet"
    raise ValueError(f"Unknown source_name '{source_name}'.")


def _chunk_output_path(output_root: Path, source_name: str, window: WindowSpec) -> Path:
    chunk_dir = _source_directories(output_root)["chunks"] / source_name
    return chunk_dir / f"{source_name}_{window.start_label}_{window.end_label}.parquet"


def _ensure_directories(output_root: Path) -> None:
    for path in _source_directories(output_root).values():
        path.mkdir(parents=True, exist_ok=True)


def _save_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def _load_chunk_frames(chunk_paths: list[Path]) -> pd.DataFrame:
    if not chunk_paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in chunk_paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return frame.reset_index(drop=True)


def _frame_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "row_count": 0,
            "start": None,
            "end": None,
            "columns": [],
            "source_versions": [],
        }

    summary = {
        "row_count": int(len(frame)),
        "start": str(frame["timestamp"].min()) if "timestamp" in frame.columns else None,
        "end": str(frame["timestamp"].max()) if "timestamp" in frame.columns else None,
        "columns": list(frame.columns),
        "source_versions": [],
    }
    if "source_version" in frame.columns:
        values = frame["source_version"].dropna().astype(str).unique().tolist()
        summary["source_versions"] = sorted(values)
    return summary


def _write_manifest(
    output_root: Path,
    start_label: str,
    end_label: str,
    funding_frame: pd.DataFrame,
    basis_frame: pd.DataFrame,
    oi_frame: pd.DataFrame,
    options_frame: pd.DataFrame,
) -> Path:
    manifest_path = _source_directories(output_root)["manifests"] / "binance_btcusdt_derivatives_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": str(datetime.now(timezone.utc)),
        "range": {
            "start": start_label,
            "end": end_label,
        },
        "sources": {
            "funding": _frame_summary(funding_frame),
            "basis": _frame_summary(basis_frame),
            "oi": _frame_summary(oi_frame),
            "options": _frame_summary(options_frame),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _fetch_window(
    session: requests.Session,
    window: WindowSpec,
    include_options: bool,
    basis_period: str,
    oi_period: str,
    options_resolution_seconds: int,
) -> dict[str, pd.DataFrame]:
    start_ms = _to_milliseconds(window.start.replace(hour=0, minute=0, second=0, microsecond=0))
    end_of_day = window.end.replace(hour=23, minute=59, second=59, microsecond=999000)
    end_ms = _to_milliseconds(end_of_day)
    return {
        "funding": _fetch_binance_funding(session, start_ms=start_ms, end_ms=end_ms),
        "basis": _fetch_binance_basis(session, start_ms=start_ms, end_ms=end_ms, period=basis_period),
        "oi": _fetch_oi_with_fallback(session, start_ms=start_ms, end_ms=end_ms, period=oi_period),
        "options": (
            _fetch_deribit_options_proxy(
                session,
                start_ms=start_ms,
                end_ms=end_ms,
                resolution_seconds=options_resolution_seconds,
            )
            if include_options
            else pd.DataFrame()
        ),
    }


def _fetch_oi_with_fallback(
    session: requests.Session,
    start_ms: int,
    end_ms: int,
    period: str,
) -> pd.DataFrame:
    try:
        return _fetch_binance_oi(session, start_ms=start_ms, end_ms=end_ms, period=period)
    except requests.HTTPError as exc:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code != 400:
            raise
        payload = session.get(
            "https://fapi.binance.com/futures/data/openInterestHist",
            params={
                "symbol": "BTCUSDT",
                "period": period,
                "limit": 500,
            },
            timeout=30,
        )
        payload.raise_for_status()
        rows = payload.json()
        return _normalize_oi_records(list(rows) if rows else [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill full-history BTC derivatives archives.")
    parser.add_argument("--start-date", default="2024-01-01", help="Inclusive UTC start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Inclusive UTC end date in YYYY-MM-DD.")
    parser.add_argument(
        "--output-root",
        default="artifacts/data_v2/normalized/binance/futures_um/BTCUSDT/derivatives",
        help="Output root directory.",
    )
    parser.add_argument("--chunk-days", type=int, default=30, help="Chunk size in days for backfill windows.")
    parser.add_argument("--basis-period", default="5m", help="Binance basis period.")
    parser.add_argument("--oi-period", default="5m", help="Binance OI period.")
    parser.add_argument("--include-options", action="store_true", help="Also backfill the options proxy source.")
    parser.add_argument(
        "--options-resolution-seconds",
        type=int,
        default=3600,
        help="Resolution in seconds for the options proxy source.",
    )
    parser.add_argument(
        "--skip-existing-chunks",
        action="store_true",
        help="Reuse existing chunk parquet files instead of downloading them again.",
    )
    args = parser.parse_args()

    start = _parse_utc_date(args.start_date)
    end = _parse_utc_date(args.end_date)
    output_root = Path(args.output_root)
    _ensure_directories(output_root)
    windows = _iter_windows(start, end, args.chunk_days)

    chunk_paths: dict[str, list[Path]] = {
        "funding": [],
        "basis": [],
        "oi": [],
        "options": [],
    }

    with requests.Session() as session:
        for window in windows:
            existing_chunk_paths = {
                source_name: _chunk_output_path(output_root, source_name, window)
                for source_name in chunk_paths
            }
            if args.skip_existing_chunks and all(
                path.exists() for source_name, path in existing_chunk_paths.items() if source_name != "options" or args.include_options
            ):
                for source_name, path in existing_chunk_paths.items():
                    if source_name == "options" and not args.include_options:
                        continue
                    chunk_paths[source_name].append(path)
                continue

            fetched = _fetch_window(
                session=session,
                window=window,
                include_options=args.include_options,
                basis_period=args.basis_period,
                oi_period=args.oi_period,
                options_resolution_seconds=args.options_resolution_seconds,
            )
            for source_name, frame in fetched.items():
                if source_name == "options" and not args.include_options:
                    continue
                chunk_path = existing_chunk_paths[source_name]
                _save_frame(frame, chunk_path)
                chunk_paths[source_name].append(chunk_path)

    funding_frame = _load_chunk_frames(chunk_paths["funding"])
    basis_frame = _load_chunk_frames(chunk_paths["basis"])
    oi_frame = _load_chunk_frames(chunk_paths["oi"])
    options_frame = _load_chunk_frames(chunk_paths["options"]) if args.include_options else pd.DataFrame()

    start_label = start.date().isoformat()
    end_label = end.date().isoformat()

    funding_path = _final_output_path(output_root, "funding", start_label, end_label)
    basis_path = _final_output_path(output_root, "basis", start_label, end_label)
    oi_path = _final_output_path(output_root, "oi", start_label, end_label)

    _save_frame(funding_frame, funding_path)
    _save_frame(basis_frame, basis_path)
    _save_frame(oi_frame, oi_path)

    options_path = None
    if args.include_options:
        options_path = _final_output_path(output_root, "options", start_label, end_label)
        _save_frame(options_frame, options_path)

    manifest_path = _write_manifest(
        output_root=output_root,
        start_label=start_label,
        end_label=end_label,
        funding_frame=funding_frame,
        basis_frame=basis_frame,
        oi_frame=oi_frame,
        options_frame=options_frame,
    )

    print(f"funding_rows={len(funding_frame)} funding_path={funding_path.resolve()}")
    print(f"basis_rows={len(basis_frame)} basis_path={basis_path.resolve()}")
    print(f"oi_rows={len(oi_frame)} oi_path={oi_path.resolve()}")
    if options_path is not None:
        print(f"options_rows={len(options_frame)} options_path={options_path.resolve()}")
    print(f"manifest_path={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
