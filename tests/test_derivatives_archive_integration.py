from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.data.derivatives.feature_store import load_derivatives_frame_from_settings
from src.services.feature_service import FeatureService


def _write_archive_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _build_archive_root(tmp_path: Path) -> Path:
    root = tmp_path / "normalized"
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=240, freq="1min")

    _write_archive_frame(
        root / "futures_um" / "fundingRate" / "BTCUSDT.parquet",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "last_funding_rate": [0.001 + index * 0.0001 for index in range(240)],
                "funding_interval_hours": [8] * 240,
                "symbol": ["BTCUSDT"] * 240,
                "source_version": ["v1"] * 240,
            }
        ),
    )
    _write_archive_frame(
        root / "futures_um" / "markPriceKlines" / "BTCUSDT-1m.parquet",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.2 + index for index in range(240)],
                "source_version": ["v1"] * 240,
            }
        ),
    )
    _write_archive_frame(
        root / "futures_um" / "indexPriceKlines" / "BTCUSDT-1m.parquet",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.1 + index for index in range(240)],
                "source_version": ["v1"] * 240,
            }
        ),
    )
    _write_archive_frame(
        root / "futures_um" / "premiumIndexKlines" / "BTCUSDT-1m.parquet",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "close": [0.001 + index * 0.0001 for index in range(240)],
                "source_version": ["v1"] * 240,
            }
        ),
    )
    _write_archive_frame(
        root / "futures_um" / "metrics" / "BTCUSDT.parquet",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "sum_open_interest": [1000.0 + index * 2 for index in range(240)],
                "sum_open_interest_value": [100000.0 + index * 200 for index in range(240)],
                "symbol": ["BTCUSDT"] * 240,
                "source_version": ["v1"] * 240,
            }
        ),
    )
    _write_archive_frame(
        root / "option" / "BVOLIndex" / "BTCBVOLUSDT.parquet",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "index_value": [0.40 + index * 0.001 for index in range(240)],
                "symbol": ["BTCBVOLUSDT"] * 240,
                "source_version": ["v1"] * 240,
            }
        ),
    )
    _write_archive_frame(
        root / "futures_um" / "bookTicker" / "BTCUSDT.parquet",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "bid_price": [100.0 + index for index in range(240)],
                "bid_qty": [10.0 + (index % 5) for index in range(240)],
                "ask_price": [100.2 + index for index in range(240)],
                "ask_qty": [9.0 + (index % 3) for index in range(240)],
                "symbol": ["BTCUSDT"] * 240,
                "source_version": ["v1"] * 240,
            }
        ),
    )
    return root


def _build_archive_settings(tmp_path: Path):
    archive_root = _build_archive_root(tmp_path)
    settings = load_settings()
    return replace(
        settings,
        dataset=replace(
            settings.dataset,
            train_start="2026-01-01T00:00:00Z",
            train_end="2026-01-01T03:59:00Z",
        ),
        derivatives=replace(
            settings.derivatives,
            enabled=True,
            path_mode="archive",
            funding=replace(
                settings.derivatives.funding,
                enabled=True,
                archive_path=str(archive_root),
                zscore_window=3,
            ),
            basis=replace(
                settings.derivatives.basis,
                enabled=True,
                archive_path=str(archive_root),
                zscore_window=3,
            ),
            oi=replace(settings.derivatives.oi, enabled=False),
            options=replace(settings.derivatives.options, enabled=False),
        ),
    )


def _build_spot_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=240, freq="1min"),
            "open": [100 + index for index in range(240)],
            "high": [101 + index for index in range(240)],
            "low": [99 + index for index in range(240)],
            "close": [100.5 + index for index in range(240)],
            "volume": [10 + index for index in range(240)],
        }
    )


def test_load_derivatives_frame_from_settings_reads_binance_public_archive(tmp_path: Path) -> None:
    settings = _build_archive_settings(tmp_path)

    loaded = load_derivatives_frame_from_settings(settings)

    assert loaded is not None
    assert "funding_rate" in loaded.columns
    assert "mark_price" in loaded.columns
    assert "index_price" in loaded.columns
    assert "premium_index" in loaded.columns
    assert loaded["funding_rate"].iloc[0] == 0.001
    assert loaded["mark_price"].iloc[0] == 100.2


def test_load_derivatives_frame_from_settings_reads_archive_metrics_as_oi(tmp_path: Path) -> None:
    settings = _build_archive_settings(tmp_path)
    settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            oi=replace(
                settings.derivatives.oi,
                enabled=True,
                archive_path=str(tmp_path / "normalized"),
                zscore_window=3,
                change_windows=[5, 60],
                slope_window=3,
            ),
        ),
    )

    loaded = load_derivatives_frame_from_settings(settings)

    assert loaded is not None
    assert "open_interest" in loaded.columns
    assert "oi_notional" in loaded.columns
    assert loaded["open_interest"].iloc[10] == 1020.0
    assert loaded["oi_notional"].iloc[10] == 102000.0


def test_load_derivatives_frame_from_settings_reads_archive_bvol_as_options(tmp_path: Path) -> None:
    settings = _build_archive_settings(tmp_path)
    settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            options=replace(
                settings.derivatives.options,
                enabled=True,
                archive_path=str(tmp_path / "normalized"),
                zscore_window=3,
                change_window=2,
                regime_zscore_threshold=0.5,
            ),
        ),
    )

    loaded = load_derivatives_frame_from_settings(settings)

    assert loaded is not None
    assert "atm_iv_near" in loaded.columns
    assert "iv_term_slope" in loaded.columns
    assert loaded["atm_iv_near"].dropna().iloc[0] == 0.40
    assert loaded["iv_term_slope"].dropna().iloc[0] == 0.0


def test_load_archive_options_frame_downsamples_bvolindex_to_minute_and_scales_percent_values(tmp_path: Path) -> None:
    from src.data.binance_public.derivatives_archive import load_archive_options_frame

    normalized_root = tmp_path / "normalized"
    path = normalized_root / "option" / "BVOLIndex" / "BTCBVOLUSDT.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:01Z",
                    "2026-01-01T00:00:30Z",
                    "2026-01-01T00:01:00Z",
                ],
                utc=True,
            ),
            "index_value": [44.0, 45.0, 46.0],
            "symbol": ["BTCBVOLUSDT", "BTCBVOLUSDT", "BTCBVOLUSDT"],
            "source_version": ["v1", "v1", "v1"],
        }
    ).to_parquet(path, index=False)

    loaded = load_archive_options_frame(normalized_root, symbol="BTCBVOLUSDT")

    assert loaded["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist() == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:01:00Z",
    ]
    assert loaded["atm_iv_near"].tolist() == [0.45, 0.46]


def test_load_archive_options_frame_uses_eoh_summary_when_available(tmp_path: Path) -> None:
    from src.data.binance_public.derivatives_archive import load_archive_options_frame

    normalized_root = tmp_path / "normalized"
    path = normalized_root / "option" / "EOHSummary" / "BTCUSDT.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                ],
                utc=True,
            ),
            "expiry": ["260131", "260228", "260131"],
            "mark_iv": [0.40, 0.55, 0.42],
            "delta": [0.49, 0.80, 0.50],
            "openinterest_usdt": [100.0, 500.0, 200.0],
            "symbol": ["BTCUSDT"] * 3,
            "source_version": ["v1"] * 3,
        }
    ).to_parquet(path, index=False)

    loaded = load_archive_options_frame(normalized_root, symbol="BTCBVOLUSDT")

    assert loaded["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist() == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T01:00:00Z",
    ]
    assert loaded["atm_iv_near"].tolist() == [0.40, 0.42]
    assert round(float(loaded["iv_term_slope"].iloc[0]), 2) == 0.15


def test_load_derivatives_frame_from_settings_reads_archive_book_ticker(tmp_path: Path) -> None:
    settings = _build_archive_settings(tmp_path)
    settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            book_ticker=replace(
                settings.derivatives.book_ticker,
                enabled=True,
                archive_path=str(tmp_path / "normalized"),
                zscore_window=3,
            ),
        ),
    )

    loaded = load_derivatives_frame_from_settings(settings)

    assert loaded is not None
    assert "bid_price" in loaded.columns
    assert "bid_qty" in loaded.columns
    assert "ask_price" in loaded.columns
    assert "ask_qty" in loaded.columns
    assert loaded["bid_price"].iloc[0] == 100.0


def test_train_and_live_feature_paths_match_with_archive_derivatives_inputs(tmp_path: Path) -> None:
    settings = _build_archive_settings(tmp_path)
    spot = _build_spot_frame()

    training = build_training_frame(spot, settings, horizon_name="5m")
    live_feature_frame = FeatureService(settings).build_feature_frame(
        spot,
        horizon_name="5m",
        select_grid_only=True,
    )

    assert not training.frame.empty
    target_timestamp = training.frame["timestamp"].iloc[-1]
    training_row = training.frame.loc[training.frame["timestamp"] == target_timestamp].iloc[0]
    live_row = live_feature_frame.loc[live_feature_frame["timestamp"] == target_timestamp].iloc[0]

    for column in training.feature_columns:
        if pd.isna(training_row[column]) and pd.isna(live_row[column]):
            continue
        assert training_row[column] == live_row[column], column


def test_train_and_live_feature_paths_match_with_archive_oi_inputs(tmp_path: Path) -> None:
    settings = _build_archive_settings(tmp_path)
    settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            oi=replace(
                settings.derivatives.oi,
                enabled=True,
                archive_path=str(tmp_path / "normalized"),
                zscore_window=15,
                change_windows=[5, 60],
                slope_window=15,
            ),
        ),
    )
    spot = _build_spot_frame()

    training = build_training_frame(spot, settings, horizon_name="5m")
    live_feature_frame = FeatureService(settings).build_feature_frame(
        spot,
        horizon_name="5m",
        select_grid_only=True,
    )

    assert not training.frame.empty
    target_timestamp = training.frame["timestamp"].iloc[-1]
    training_row = training.frame.loc[training.frame["timestamp"] == target_timestamp].iloc[0]
    live_row = live_feature_frame.loc[live_feature_frame["timestamp"] == target_timestamp].iloc[0]

    oi_columns = [column for column in training.feature_columns if column.startswith("oi_")]
    assert oi_columns
    assert "source_file_oi" not in training.frame.columns
    assert "checksum_status_oi" not in training.frame.columns
    for column in oi_columns:
        if pd.isna(training_row[column]) and pd.isna(live_row[column]):
            continue
        assert training_row[column] == live_row[column], column


def test_train_and_live_feature_paths_match_with_archive_options_inputs(tmp_path: Path) -> None:
    settings = _build_archive_settings(tmp_path)
    settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            options=replace(
                settings.derivatives.options,
                enabled=True,
                archive_path=str(tmp_path / "normalized"),
                zscore_window=3,
                change_window=2,
                regime_zscore_threshold=0.5,
            ),
        ),
    )
    spot = _build_spot_frame()

    training = build_training_frame(spot, settings, horizon_name="5m")
    live_feature_frame = FeatureService(settings).build_feature_frame(
        spot,
        horizon_name="5m",
        select_grid_only=True,
    )

    assert not training.frame.empty
    target_timestamp = training.frame["timestamp"].iloc[-1]
    training_row = training.frame.loc[training.frame["timestamp"] == target_timestamp].iloc[0]
    live_row = live_feature_frame.loc[live_feature_frame["timestamp"] == target_timestamp].iloc[0]

    option_columns = [column for column in training.feature_columns if column.startswith("iv_") or column == "atm_iv_near"]
    assert option_columns
    for column in option_columns:
        if pd.isna(training_row[column]) and pd.isna(live_row[column]):
            continue
        assert training_row[column] == live_row[column], column


def test_train_and_live_feature_paths_match_with_archive_book_ticker_inputs(tmp_path: Path) -> None:
    settings = _build_archive_settings(tmp_path)
    settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            book_ticker=replace(
                settings.derivatives.book_ticker,
                enabled=True,
                archive_path=str(tmp_path / "normalized"),
                zscore_window=15,
            ),
        ),
    )
    spot = _build_spot_frame()

    training = build_training_frame(spot, settings, horizon_name="5m")
    live_feature_frame = FeatureService(settings).build_feature_frame(
        spot,
        horizon_name="5m",
        select_grid_only=True,
    )

    assert not training.frame.empty
    target_timestamp = training.frame["timestamp"].iloc[-1]
    training_row = training.frame.loc[training.frame["timestamp"] == target_timestamp].iloc[0]
    live_row = live_feature_frame.loc[live_feature_frame["timestamp"] == target_timestamp].iloc[0]

    book_columns = [column for column in training.feature_columns if column.startswith("book_")]
    assert book_columns
    for column in book_columns:
        if pd.isna(training_row[column]) and pd.isna(live_row[column]):
            continue
        assert training_row[column] == live_row[column], column
