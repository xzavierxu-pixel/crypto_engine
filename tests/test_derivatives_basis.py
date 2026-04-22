from __future__ import annotations

from dataclasses import replace

import math
import pandas as pd

from src.core.config import load_settings
from src.features.derivatives_basis import DerivativesBasisFeaturePack


def test_derivatives_basis_feature_pack_uses_prior_visible_values() -> None:
    settings = load_settings()
    settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            enabled=True,
            basis=replace(settings.derivatives.basis, enabled=True, zscore_window=3),
        ),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="1min"),
            "close": [100, 101, 102, 103, 104, 105],
            "raw_mark_price": [100.2, 101.3, 102.4, 103.5, 104.6, 105.7],
            "raw_index_price": [100.1, 101.2, 102.3, 103.4, 104.5, 105.6],
            "raw_premium_index": [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035],
        }
    )

    features = DerivativesBasisFeaturePack().transform(frame, settings, settings.features.get_profile("core_5m"))

    expected = (102.4 / 102.0) - 1.0
    assert math.isclose(features.loc[3, "basis_mark_spot"], expected, rel_tol=1e-9)
    assert math.isclose(features.loc[4, "basis_mark_spot_change_1"], features.loc[4, "basis_mark_spot"] - features.loc[3, "basis_mark_spot"], rel_tol=1e-9)
    assert "basis_mark_spot_zscore_3" in features.columns
    assert "premium_index_zscore_3" in features.columns
