from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.core.config import FeaturesConfig, load_settings
from src.features.builder import build_feature_frame


def test_asymmetry_pack_emits_directional_volatility_and_imbalance() -> None:
    settings = load_settings()
    base_profile = settings.features.get_profile("core_5m")
    scoped_settings = replace(
        settings,
        features=FeaturesConfig(
            profiles={
                "core_5m": replace(
                    base_profile,
                    packs=["asymmetry"],
                    asymmetry_rv_windows=[5, 20],
                    asymmetry_skew_windows=[10, 20],
                    asymmetry_imbalance_windows=[3, 5],
                ),
            }
        ),
    )
    close = [100 + ((-1) ** index) * (index % 5) for index in range(240)]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=240, freq="1min"),
            "open": close,
            "high": [value + 1.5 for value in close],
            "low": [value - 1.0 for value in close],
            "close": [value + (0.5 if index % 2 == 0 else -0.25) for index, value in enumerate(close)],
            "volume": [100 + index for index in range(240)],
        }
    )

    features = build_feature_frame(frame, scoped_settings, horizon_name="5m", select_grid_only=False)
    valid = features.dropna(subset=["upside_rv_5", "downside_rv_5", "realized_skew_10"])

    assert "upside_rv_20" in features.columns
    assert "downside_rv_20" in features.columns
    assert "wick_imbalance_3" in features.columns
    assert "body_imbalance_5" in features.columns
    assert not valid.empty
    assert (valid["upside_rv_5"] >= 0).all()
    assert (valid["downside_rv_5"] >= 0).all()
