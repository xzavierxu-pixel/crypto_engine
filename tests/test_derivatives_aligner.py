from __future__ import annotations

import pandas as pd

from src.data.derivatives.aligner import align_derivatives_to_spot, merge_derivatives_frames


def test_align_derivatives_to_spot_uses_latest_known_values_only() -> None:
    spot = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="1min"),
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "volume": [10, 11, 12, 13, 14, 15],
        }
    )
    derivatives = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:01:00Z", "2026-01-01T00:04:00Z"],
                utc=True,
            ),
            "funding_rate": [0.001, 0.002],
            "mark_price": [101.0, 104.0],
        }
    )

    aligned = align_derivatives_to_spot(spot, derivatives)

    assert pd.isna(aligned.loc[0, "funding_rate"])
    assert aligned.loc[1, "funding_rate"] == 0.001
    assert aligned.loc[3, "funding_rate"] == 0.001
    assert aligned.loc[4, "funding_rate"] == 0.002
    assert aligned.loc[5, "funding_rate"] == 0.002


def test_merge_derivatives_frames_keeps_funding_and_basis_columns() -> None:
    funding = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True),
            "funding_rate": [0.001],
            "source_version": ["funding-v1"],
            "exchange": ["binance"],
            "symbol": ["BTC/USDT:USDT"],
        }
    )
    basis = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True),
            "mark_price": [100.1],
            "basis_source_version": ["basis-v1"],
        }
    )

    merged = merge_derivatives_frames(funding_frame=funding, basis_frame=basis)

    assert "funding_rate" in merged.columns
    assert "mark_price" in merged.columns
    assert "funding_source_version" in merged.columns
    assert "basis_source_version" in merged.columns
    assert merged.loc[0, "exchange"] == "binance"
