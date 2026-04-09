from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import load_ohlcv_feather


def test_load_ohlcv_feather_accepts_freqtrade_date_column(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1min"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
        }
    )
    path = tmp_path / "sample.feather"
    source.to_feather(path)

    loaded = load_ohlcv_feather(path)

    assert "timestamp" in loaded.columns
    assert loaded["timestamp"].tolist() == list(source["date"])
