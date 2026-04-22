from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.derivatives.book_ticker_loader import load_book_ticker_frame


def test_load_book_ticker_frame_normalizes_schema(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1min"),
            "bid_price": [100.0, 100.1, 100.2],
            "bid_qty": [10.0, 11.0, 12.0],
            "ask_price": [100.2, 100.3, 100.4],
            "ask_qty": [9.0, 8.0, 7.0],
            "exchange": ["binance", "binance", "binance"],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        }
    )
    path = tmp_path / "book_ticker.parquet"
    source.to_parquet(path, index=False)

    loaded = load_book_ticker_frame(path)

    assert loaded["timestamp"].tolist() == list(source["date"])
    assert loaded["bid_price"].tolist() == source["bid_price"].tolist()
    assert loaded["bid_qty"].tolist() == source["bid_qty"].tolist()
    assert loaded["ask_price"].tolist() == source["ask_price"].tolist()
    assert loaded["ask_qty"].tolist() == source["ask_qty"].tolist()
