from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.binance_public.derivatives_archive import load_archive_oi_frame


def test_load_archive_oi_frame_maps_metrics_columns_to_oi_schema(tmp_path: Path) -> None:
    normalized_root = tmp_path / "normalized"
    path = normalized_root / "futures_um" / "metrics" / "BTCUSDT.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="5min"),
            "sum_open_interest": [1000.0, 1005.0, 1010.0],
            "sum_open_interest_value": [100000.0, 100500.0, 101000.0],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "source_version": ["v1", "v1", "v1"],
        }
    ).to_parquet(path, index=False)

    loaded = load_archive_oi_frame(normalized_root, symbol="BTCUSDT")

    assert loaded["open_interest"].tolist() == [1000.0, 1005.0, 1010.0]
    assert loaded["oi_notional"].tolist() == [100000.0, 100500.0, 101000.0]
    assert loaded["exchange"].eq("binance").all()
