from __future__ import annotations

import pandas as pd

from scripts.data.step4_features.build_decision_second_level_store_from_binance_archives import _numeric_epoch_to_utc


def test_numeric_epoch_to_utc_accepts_millisecond_timestamps() -> None:
    result = _numeric_epoch_to_utc(pd.Series([1_767_225_600_000]))

    assert result.iloc[0] == pd.Timestamp("2026-01-01T00:00:00Z")


def test_numeric_epoch_to_utc_accepts_microsecond_timestamps() -> None:
    result = _numeric_epoch_to_utc(pd.Series([1_767_225_600_000_000]))

    assert result.iloc[0] == pd.Timestamp("2026-01-01T00:00:00Z")
