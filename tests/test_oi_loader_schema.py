from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.derivatives.oi_loader import load_oi_frame


def test_load_oi_frame_normalizes_schema(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="5min"),
            "open_interest": [1000.0, 1005.0, 1010.0],
            "oi_notional": [100000.0, 100500.0, 101000.0],
            "exchange": ["binance", "binance", "binance"],
            "symbol": ["BTC/USDT:USDT", "BTC/USDT:USDT", "BTC/USDT:USDT"],
        }
    )
    path = tmp_path / "oi.csv"
    source.to_csv(path, index=False)

    loaded = load_oi_frame(path)

    assert loaded["timestamp"].tolist() == list(source["date"])
    assert loaded["open_interest"].tolist() == source["open_interest"].tolist()
    assert loaded["oi_notional"].tolist() == source["oi_notional"].tolist()
