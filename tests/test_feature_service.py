from __future__ import annotations

import pandas as pd

from src.core.config import load_settings
from src.services.feature_service import FeatureService


def test_feature_service_returns_latest_grid_row() -> None:
    settings = load_settings()
    service = FeatureService(settings)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=20, freq="1min"),
            "open": [100 + index for index in range(20)],
            "high": [101 + index for index in range(20)],
            "low": [99 + index for index in range(20)],
            "close": [100 + index for index in range(20)],
            "volume": [10 + index for index in range(20)],
        }
    )

    latest = service.build_latest_feature_row(frame, horizon_name="5m")

    assert latest["timestamp"] == pd.Timestamp("2024-01-01T12:15:00Z")
    assert latest["horizon"] == "5m"
    assert "ret_1" in latest.index


def test_feature_service_can_preheat_and_return_snapshot() -> None:
    settings = load_settings()
    service = FeatureService(settings)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=20, freq="1min"),
            "open": [100 + index for index in range(20)],
            "high": [101 + index for index in range(20)],
            "low": [99 + index for index in range(20)],
            "close": [100 + index for index in range(20)],
            "volume": [10 + index for index in range(20)],
        }
    )

    snapshot = service.preheat_latest_feature_snapshot(frame, horizon_name="5m")
    cached = service.get_preheated_snapshot("5m")

    assert snapshot.horizon == "5m"
    assert snapshot.row["timestamp"] == pd.Timestamp("2024-01-01T12:15:00Z")
    assert cached.row["grid_id"] == snapshot.row["grid_id"]
    assert cached.source_timestamp == pd.Timestamp("2024-01-01T12:19:00Z")
