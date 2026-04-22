from __future__ import annotations

from dataclasses import replace

import math
import pandas as pd

from src.core.config import load_settings
from src.features.derivatives_funding import DerivativesFundingFeaturePack


def test_derivatives_funding_feature_pack_uses_prior_visible_value() -> None:
    settings = load_settings()
    settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            enabled=True,
            funding=replace(settings.derivatives.funding, enabled=True, zscore_window=3),
        ),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="1min"),
            "raw_funding_rate": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
        }
    )

    features = DerivativesFundingFeaturePack().transform(frame, settings, settings.features.get_profile("core_5m"))

    assert math.isclose(features.loc[2, "funding_rate"], 0.002, rel_tol=1e-9)
    assert math.isclose(features.loc[4, "funding_rate_change_1"], 0.001, rel_tol=1e-9)
    assert "funding_rate_zscore_3" in features.columns
    assert features.loc[4, "funding_is_pos"] == 1.0
