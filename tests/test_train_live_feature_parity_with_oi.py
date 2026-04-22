from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.data.derivatives.aligner import merge_derivatives_frames
from src.services.feature_service import FeatureService


def test_train_and_live_feature_paths_match_with_oi() -> None:
    settings = load_settings()
    settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            enabled=True,
            funding=replace(settings.derivatives.funding, enabled=True, zscore_window=3),
            basis=replace(settings.derivatives.basis, enabled=True, zscore_window=3),
            oi=replace(settings.derivatives.oi, enabled=True, zscore_window=15, change_windows=[5, 60], slope_window=15),
        ),
    )
    spot = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=600, freq="1min"),
            "open": [100 + index for index in range(600)],
            "high": [101 + index for index in range(600)],
            "low": [99 + index for index in range(600)],
            "close": [100.5 + index for index in range(600)],
            "volume": [10 + index for index in range(600)],
        }
    )
    funding = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=600, freq="1min"),
            "funding_rate": [0.001 + index * 0.0001 for index in range(600)],
            "funding_effective_time": pd.date_range("2026-01-01T00:00:00Z", periods=600, freq="1min"),
        }
    )
    basis = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=600, freq="1min"),
            "mark_price": [100.2 + index for index in range(600)],
            "index_price": [100.1 + index for index in range(600)],
            "premium_index": [0.001 + index * 0.0001 for index in range(600)],
        }
    )
    oi = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=120, freq="5min"),
            "open_interest": [1000.0 + index * 5 for index in range(120)],
            "oi_notional": [100000.0 + index * 500 for index in range(120)],
        }
    )
    derivatives = merge_derivatives_frames(funding_frame=funding, basis_frame=basis, oi_frame=oi)

    training = build_training_frame(spot, settings, horizon_name="5m", derivatives_frame=derivatives)
    live_feature_frame = FeatureService(settings).build_feature_frame(
        spot,
        horizon_name="5m",
        select_grid_only=True,
        derivatives_frame=derivatives,
    )

    assert not training.frame.empty
    target_timestamp = training.frame["timestamp"].iloc[-1]
    training_row = training.frame.loc[training.frame["timestamp"] == target_timestamp].iloc[0]
    live_row = live_feature_frame.loc[live_feature_frame["timestamp"] == target_timestamp].iloc[0]

    for column in training.feature_columns:
        if pd.isna(training_row[column]) and pd.isna(live_row[column]):
            continue
        assert training_row[column] == live_row[column], column
