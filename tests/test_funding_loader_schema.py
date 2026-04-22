from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.derivatives.funding_loader import load_funding_frame


def test_load_funding_frame_normalizes_schema_and_effective_time(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1min"),
            "funding_rate": [0.0001, 0.0002, 0.0003],
            "funding_effective_time": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T08:00:00Z",
                "2026-01-01T08:00:00Z",
            ],
            "exchange": ["binance", "binance", "binance"],
            "symbol": ["BTC/USDT:USDT", "BTC/USDT:USDT", "BTC/USDT:USDT"],
            "source_version": ["v1", "v1", "v1"],
        }
    )
    path = tmp_path / "funding.parquet"
    source.to_parquet(path, index=False)

    loaded = load_funding_frame(path)

    assert loaded["timestamp"].tolist() == list(source["date"])
    assert loaded["funding_effective_time"].tolist() == list(pd.to_datetime(source["funding_effective_time"], utc=True))
    assert loaded["funding_rate"].tolist() == source["funding_rate"].tolist()
    assert loaded["exchange"].eq("binance").all()
