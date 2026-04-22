from __future__ import annotations

from dataclasses import replace

import math
import pandas as pd

from src.core.config import load_settings
from src.features.derivatives_options import DerivativesOptionsFeaturePack


def test_derivatives_options_feature_pack_builds_low_frequency_iv_features() -> None:
    settings = load_settings()
    settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            enabled=True,
            options=replace(
                settings.derivatives.options,
                enabled=True,
                zscore_window=3,
                change_window=3,
                regime_zscore_threshold=0.5,
            ),
        ),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="1h"),
            "raw_atm_iv_near": [0.40, 0.42, 0.44, 0.48, 0.46, 0.45, 0.50, 0.52],
            "raw_iv_term_slope": [0.01, 0.02, 0.03, 0.01, 0.0, -0.01, 0.02, 0.03],
        }
    )

    features = DerivativesOptionsFeaturePack().transform(frame, settings, settings.features.get_profile("core_5m"))

    assert math.isclose(features.loc[3, "atm_iv_near"], 0.44, rel_tol=1e-9)
    assert math.isclose(features.loc[3, "iv_term_slope"], 0.03, rel_tol=1e-9)
    assert "iv_change_1h" in features.columns
    assert "iv_zscore" in features.columns
    assert "iv_regime" in features.columns
    assert features["iv_regime"].isin([-1.0, 0.0, 1.0]).all()


def test_derivatives_options_feature_pack_falls_back_to_neutral_term_slope_when_missing() -> None:
    settings = load_settings()
    settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            enabled=True,
            options=replace(
                settings.derivatives.options,
                enabled=True,
                zscore_window=3,
                change_window=2,
                regime_zscore_threshold=0.5,
            ),
        ),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="1h"),
            "raw_atm_iv_near": [0.40, 0.42, 0.44, 0.48, 0.46, 0.45, 0.50, 0.52],
            "raw_iv_term_slope": [pd.NA] * 8,
        }
    )

    features = DerivativesOptionsFeaturePack().transform(frame, settings, settings.features.get_profile("core_5m"))

    assert features["iv_term_slope"].fillna(0.0).eq(0.0).all()
    assert features["atm_iv_near"].notna().sum() > 0
