from __future__ import annotations

import pandas as pd

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.services.feature_service import FeatureService


def test_train_and_live_feature_paths_match_for_15m_horizon() -> None:
    settings = load_settings()
    spot = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=960, freq="1min"),
            "open": [100 + index for index in range(960)],
            "high": [101 + index for index in range(960)],
            "low": [99 + index for index in range(960)],
            "close": [100.5 + index for index in range(960)],
            "volume": [10 + index for index in range(960)],
        }
    )

    training = build_training_frame(spot, settings, horizon_name="15m")
    live_feature_frame = FeatureService(settings).build_feature_frame(
        spot,
        horizon_name="15m",
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
