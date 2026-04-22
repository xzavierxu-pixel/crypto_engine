from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd

from src.core.config import load_settings
from src.features.derivatives_book_ticker import DerivativesBookTickerFeaturePack


def test_derivatives_book_ticker_feature_pack_uses_prior_visible_values() -> None:
    settings = load_settings()
    settings = replace(
        settings,
        derivatives=replace(
            settings.derivatives,
            enabled=True,
            book_ticker=replace(
                settings.derivatives.book_ticker,
                enabled=True,
                zscore_window=3,
            ),
        ),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="1min"),
            "raw_bid_price": [100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7],
            "raw_bid_qty": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "raw_ask_price": [100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9],
            "raw_ask_qty": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
        }
    )

    features = DerivativesBookTickerFeaturePack().transform(frame, settings, settings.features.get_profile("core_5m"))

    expected_mid = (100.2 + 100.4) / 2.0
    expected_spread_bps = ((100.4 - 100.2) / expected_mid) * 10000.0
    expected_imbalance = (12.0 - 7.0) / (12.0 + 7.0)
    expected_microprice = ((100.4 * 12.0) + (100.2 * 7.0)) / (12.0 + 7.0)
    expected_microprice_offset_bps = ((expected_microprice - expected_mid) / expected_mid) * 10000.0

    assert math.isclose(features.loc[3, "book_spread_bps"], expected_spread_bps, rel_tol=1e-9)
    assert math.isclose(features.loc[3, "book_imbalance"], expected_imbalance, rel_tol=1e-9)
    assert math.isclose(
        features.loc[3, "book_microprice_offset_bps"],
        expected_microprice_offset_bps,
        rel_tol=1e-9,
    )
    assert math.isclose(features.loc[3, "book_top_depth_total"], 19.0, rel_tol=1e-9)
    assert "book_spread_bps_zscore_3" in features.columns
