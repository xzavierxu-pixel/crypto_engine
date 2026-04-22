from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.core.config import load_settings
from src.data.derivatives.feature_store import load_derivatives_frame_from_paths
from src.data.loaders import load_ohlcv_feather


def test_load_ohlcv_feather_accepts_freqtrade_date_column(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1min"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
        }
    )
    path = tmp_path / "sample.feather"
    source.to_feather(path)

    loaded = load_ohlcv_feather(path)

    assert "timestamp" in loaded.columns
    assert loaded["timestamp"].tolist() == list(source["date"])


def test_load_derivatives_frame_from_paths_merges_all_derivatives_sources(tmp_path: Path) -> None:
    funding = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1min"),
            "funding_rate": [0.001, 0.002],
        }
    )
    basis = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1min"),
            "mark_price": [100.1, 100.2],
            "index_price": [100.0, 100.1],
            "premium_index": [0.001, 0.0015],
        }
    )
    oi = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1min"),
            "open_interest": [1000.0, 1005.0],
            "oi_notional": [100000.0, 100500.0],
        }
    )
    options = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1min"),
            "atm_iv_near": [0.45, 0.46],
            "iv_term_slope": [0.01, 0.015],
        }
    )
    funding_path = tmp_path / "funding.parquet"
    basis_path = tmp_path / "basis.parquet"
    oi_path = tmp_path / "oi.parquet"
    options_path = tmp_path / "options.parquet"
    funding.to_parquet(funding_path, index=False)
    basis.to_parquet(basis_path, index=False)
    oi.to_parquet(oi_path, index=False)
    options.to_parquet(options_path, index=False)

    merged = load_derivatives_frame_from_paths(
        funding_path=funding_path,
        basis_path=basis_path,
        oi_path=oi_path,
        options_path=options_path,
    )

    assert merged is not None
    assert "funding_rate" in merged.columns
    assert "mark_price" in merged.columns
    assert "open_interest" in merged.columns
    assert "atm_iv_near" in merged.columns


def test_load_derivatives_frame_from_settings_uses_archive_paths_when_selected(tmp_path: Path) -> None:
    settings = load_settings()
    archive_funding = tmp_path / "funding_archive.parquet"
    archive_basis = tmp_path / "basis_archive.parquet"

    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1min"),
            "funding_rate": [0.001, 0.002],
        }
    ).to_parquet(archive_funding, index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1min"),
            "mark_price": [100.1, 100.2],
            "index_price": [100.0, 100.1],
            "premium_index": [0.001, 0.0015],
        }
    ).to_parquet(archive_basis, index=False)

    archive_settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            enabled=True,
            path_mode="archive",
            funding=replace(settings.derivatives.funding, enabled=True, archive_path=str(archive_funding)),
            basis=replace(settings.derivatives.basis, enabled=True, archive_path=str(archive_basis)),
            oi=replace(settings.derivatives.oi, enabled=False),
            options=replace(settings.derivatives.options, enabled=False),
        ),
    )

    from src.data.derivatives.feature_store import load_derivatives_frame_from_settings

    merged = load_derivatives_frame_from_settings(archive_settings)

    assert merged is not None
    assert "funding_rate" in merged.columns
    assert "mark_price" in merged.columns


def test_load_derivatives_frame_from_settings_prefers_cli_override_over_archive_mode(tmp_path: Path) -> None:
    settings = load_settings()
    override_funding = tmp_path / "funding_override.parquet"

    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1min"),
            "funding_rate": [0.123, 0.456],
        }
    ).to_parquet(override_funding, index=False)

    archive_settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            enabled=True,
            path_mode="archive",
            funding=replace(settings.derivatives.funding, enabled=True, archive_path=None),
            basis=replace(settings.derivatives.basis, enabled=False),
            oi=replace(settings.derivatives.oi, enabled=False),
            options=replace(settings.derivatives.options, enabled=False),
        ),
    )

    from src.data.derivatives.feature_store import load_derivatives_frame_from_settings

    merged = load_derivatives_frame_from_settings(
        archive_settings,
        funding_path=override_funding,
    )

    assert merged is not None
    assert merged["funding_rate"].tolist() == [0.123, 0.456]


def test_load_derivatives_frame_from_settings_respects_top_level_disable_flag() -> None:
    settings = load_settings()

    from src.data.derivatives.feature_store import load_derivatives_frame_from_settings

    merged = load_derivatives_frame_from_settings(settings)

    assert merged is None
