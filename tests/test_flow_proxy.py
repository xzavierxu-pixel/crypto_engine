from __future__ import annotations

from dataclasses import replace
import math

import pandas as pd

from src.core.config import FeaturesConfig, load_settings
from src.features.builder import build_feature_frame


def test_flow_proxy_pack_uses_previous_completed_candle() -> None:
    settings = load_settings()
    base_profile = settings.features.get_profile("core_5m")
    scoped_settings = replace(
        settings,
        features=FeaturesConfig(
            profiles={
                "core_5m": replace(base_profile, packs=["flow_proxy"]),
            }
        ),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=240, freq="1min"),
            "open": [100.0] * 240,
            "high": [110.0] + [101.0] * 239,
            "low": [90.0] + [99.0] * 239,
            "close": [105.0] + [100.0] * 239,
            "volume": [10.0] + [20.0] * 239,
        }
    )

    features = build_feature_frame(frame, scoped_settings, horizon_name="5m", select_grid_only=False)
    row = features.loc[features["timestamp"] == pd.Timestamp("2024-01-01T12:01:00Z")].iloc[0]

    expected_clv = ((105.0 - 90.0) - (110.0 - 105.0)) / (110.0 - 90.0)

    assert math.isclose(row["clv_1"], expected_clv, rel_tol=1e-9)
    assert math.isclose(row["clv_x_volume_1"], expected_clv * 10.0, rel_tol=1e-9)
    assert "signed_dollar_volume_3" in features.columns
    assert "body_x_volume_3" in features.columns
