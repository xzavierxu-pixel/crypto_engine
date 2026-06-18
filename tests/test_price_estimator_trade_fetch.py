from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "price_estimator" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fetch_btc5m_sell_taker_trades import normalize_trade


def test_normalize_trade_keeps_buy_side_without_response_side_filter() -> None:
    ref = pd.Series(
        {
            "condition_id": "0xabc",
            "polymarket_slug": "btc-updown-5m-1",
            "market_t0": pd.Timestamp("2026-01-01T00:00:00Z"),
            "final_outcome": "up",
        }
    )
    trade = {"price": 0.42, "timestamp": 1, "side": "BUY", "asset": "1", "outcome": "Up"}

    row = normalize_trade(trade, ref)

    assert row is not None
    assert row["side"] == "BUY"
    assert row["outcome"] == "up"


def test_normalize_trade_applies_configured_response_side_filter() -> None:
    ref = pd.Series(
        {
            "condition_id": "0xabc",
            "polymarket_slug": "btc-updown-5m-1",
            "market_t0": pd.Timestamp("2026-01-01T00:00:00Z"),
            "final_outcome": "up",
            "response_side_filter": "SELL",
        }
    )
    trade = {"price": 0.42, "timestamp": 1, "side": "BUY", "asset": "1", "outcome": "Up"}

    assert normalize_trade(trade, ref) is None
