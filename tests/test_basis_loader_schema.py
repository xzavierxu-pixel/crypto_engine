from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.derivatives.basis_loader import load_basis_frame


def test_load_basis_frame_normalizes_schema(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1min"),
            "mark_price": [100.1, 100.2, 100.3],
            "index_price": [100.0, 100.1, 100.2],
            "premium_index": [0.001, 0.002, 0.003],
            "exchange": ["binance", "binance", "binance"],
            "symbol": ["BTC/USDT:USDT", "BTC/USDT:USDT", "BTC/USDT:USDT"],
        }
    )
    path = tmp_path / "basis.feather"
    source.to_feather(path)

    loaded = load_basis_frame(path)

    assert loaded["timestamp"].tolist() == list(source["date"])
    assert loaded["mark_price"].tolist() == source["mark_price"].tolist()
    assert loaded["index_price"].tolist() == source["index_price"].tolist()
    assert loaded["premium_index"].tolist() == source["premium_index"].tolist()
