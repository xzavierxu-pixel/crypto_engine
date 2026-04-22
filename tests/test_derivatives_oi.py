from __future__ import annotations

from dataclasses import replace

import math
import pandas as pd

from src.core.config import load_settings
from src.features.derivatives_oi import DerivativesOIFeaturePack


def test_derivatives_oi_feature_pack_uses_prior_visible_values_and_interactions() -> None:
    settings = load_settings()
    settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            enabled=True,
            oi=replace(
                settings.derivatives.oi,
                enabled=True,
                zscore_window=3,
                change_windows=[5, 60],
                slope_window=3,
            ),
        ),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=70, freq="1min"),
            "raw_open_interest": [1000.0 + index * 2 for index in range(70)],
            "raw_oi_notional": [100000.0 + index * 100 for index in range(70)],
            "basis_mark_spot": [0.01 for _ in range(70)],
            "funding_rate": [0.001 for _ in range(70)],
        }
    )

    features = DerivativesOIFeaturePack().transform(frame, settings, settings.features.get_profile("core_5m"))

    assert math.isclose(features.loc[6, "oi_level"], 1010.0, rel_tol=1e-9)
    expected_change_5m = (1012.0 / 1002.0) - 1.0
    assert math.isclose(features.loc[7, "oi_change_5m"], expected_change_5m, rel_tol=1e-9)
    assert "oi_change_1h" in features.columns
    assert "oi_zscore" in features.columns
    assert "oi_slope" in features.columns
    assert math.isclose(features.loc[7, "oi_x_basis"], features.loc[7, "oi_change_5m"] * 0.01, rel_tol=1e-9)
    assert math.isclose(features.loc[7, "oi_x_funding"], features.loc[7, "oi_change_5m"] * 0.001, rel_tol=1e-9)
