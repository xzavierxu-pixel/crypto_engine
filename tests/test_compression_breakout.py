from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.core.config import FeaturesConfig, load_settings
from src.features.builder import build_feature_frame


def test_compression_breakout_pack_exposes_expected_columns() -> None:
    settings = load_settings()
    base_profile = settings.features.get_profile("core_5m")
    scoped_settings = replace(
        settings,
        features=FeaturesConfig(
            profiles={
                "core_5m": replace(
                    base_profile,
                    packs=["compression_breakout"],
                    compression_window=20,
                    compression_rank_window=100,
                    compression_atr_short_window=5,
                    compression_atr_long_window=20,
                    compression_nr_windows=[4, 7],
                ),
            }
        ),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=300, freq="1min"),
            "open": [100 + (index % 7) * 0.1 for index in range(300)],
            "high": [100.5 + (index % 7) * 0.1 for index in range(300)],
            "low": [99.5 + (index % 7) * 0.1 for index in range(300)],
            "close": [100 + (index % 11) * 0.08 for index in range(300)],
            "volume": [100 + index for index in range(300)],
        }
    )

    features = build_feature_frame(frame, scoped_settings, horizon_name="5m", select_grid_only=False)
    valid = features.dropna(subset=["bb_width_pct_rank_100", "compression_score"])

    assert "bb_width_20" in features.columns
    assert "donchian_width_20" in features.columns
    assert "atr_ratio_5_20" in features.columns
    assert "nr4_flag" in features.columns
    assert "nr7_flag" in features.columns
    assert "breakout_up_dist_20" in features.columns
    assert "breakout_down_dist_20" in features.columns
    assert not valid.empty
    assert valid["bb_width_pct_rank_100"].between(0.0, 1.0).all()
    assert valid["compression_score"].between(0.0, 1.0).all()
