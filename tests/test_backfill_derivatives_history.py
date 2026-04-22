from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.backfill_derivatives_history import (
    _final_output_path,
    _frame_summary,
    _iter_windows,
    _load_chunk_frames,
    _write_manifest,
)


def test_iter_windows_splits_full_range_into_fixed_day_chunks() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 2, 5, tzinfo=timezone.utc)

    windows = _iter_windows(start, end, chunk_days=14)

    assert len(windows) == 3
    assert windows[0].start.date().isoformat() == "2024-01-01"
    assert windows[0].end.date().isoformat() == "2024-01-14"
    assert windows[-1].start.date().isoformat() == "2024-01-29"
    assert windows[-1].end.date().isoformat() == "2024-02-05"


def test_load_chunk_frames_merges_and_deduplicates_timestamps(tmp_path: Path) -> None:
    chunk_a = tmp_path / "a.parquet"
    chunk_b = tmp_path / "b.parquet"

    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"], utc=True),
            "funding_rate": [0.1, 0.2],
        }
    ).to_parquet(chunk_a, index=False)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:05:00Z", "2024-01-01T00:10:00Z"], utc=True),
            "funding_rate": [0.25, 0.3],
        }
    ).to_parquet(chunk_b, index=False)

    merged = _load_chunk_frames([chunk_a, chunk_b])

    assert len(merged) == 3
    assert merged["timestamp"].iloc[-1].isoformat() == "2024-01-01T00:10:00+00:00"
    assert merged.loc[merged["timestamp"] == pd.Timestamp("2024-01-01T00:05:00Z"), "funding_rate"].iloc[0] == 0.25


def test_final_output_path_uses_expected_archive_layout(tmp_path: Path) -> None:
    path = _final_output_path(tmp_path, "basis", "2024-01-01", "2026-04-08")

    assert path == tmp_path / "basis" / "binance_btcusdt_basis_2024-01-01_2026-04-08.parquet"


def test_write_manifest_records_source_summaries(tmp_path: Path) -> None:
    funding = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T08:00:00Z"], utc=True),
            "funding_rate": [0.1, 0.2],
            "source_version": ["funding_v1", "funding_v1"],
        }
    )
    basis = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "mark_price": [100.0],
            "source_version": ["basis_v1"],
        }
    )
    oi = pd.DataFrame(columns=["timestamp", "open_interest", "source_version"])
    options = pd.DataFrame(columns=["timestamp", "atm_iv_near", "source_version"])

    manifest_path = _write_manifest(
        output_root=tmp_path,
        start_label="2024-01-01",
        end_label="2024-01-31",
        funding_frame=funding,
        basis_frame=basis,
        oi_frame=oi,
        options_frame=options,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["range"]["start"] == "2024-01-01"
    assert payload["sources"]["funding"]["row_count"] == 2
    assert payload["sources"]["basis"]["source_versions"] == ["basis_v1"]
    assert payload["sources"]["oi"]["row_count"] == 0


def test_frame_summary_handles_empty_frame() -> None:
    summary = _frame_summary(pd.DataFrame())

    assert summary["row_count"] == 0
    assert summary["columns"] == []
