from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.derivatives.options_loader import load_options_frame


def test_load_options_frame_normalizes_schema(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1h"),
            "atm_iv_near": [0.45, 0.47, 0.44],
            "iv_term_slope": [0.01, 0.015, 0.005],
            "exchange": ["deribit", "deribit", "deribit"],
            "symbol": ["BTC", "BTC", "BTC"],
        }
    )
    path = tmp_path / "options.feather"
    source.to_feather(path)

    loaded = load_options_frame(path)

    assert loaded["timestamp"].tolist() == list(source["date"])
    assert loaded["atm_iv_near"].tolist() == source["atm_iv_near"].tolist()
    assert loaded["iv_term_slope"].tolist() == source["iv_term_slope"].tolist()
