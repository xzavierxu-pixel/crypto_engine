from __future__ import annotations

from datetime import UTC, datetime

from src.core.config import load_settings
from src.core.schemas import Decision, MarketQuote, Signal
from src.execution.guards import build_window_key, evaluate_market_guards


def make_signal() -> Signal:
    return Signal(
        asset="BTC/USDT",
        horizon="5m",
        t0=datetime(2026, 4, 8, 13, 45, tzinfo=UTC),
        p_up=0.57,
        model_version="m1",
        feature_version="v1",
    )


def make_decision() -> Decision:
    return Decision(
        should_trade=True,
        side="YES",
        edge=0.08,
        reason="edge_signal_passed",
        target_size=0.02,
    )


def test_market_guards_reject_wide_spread() -> None:
    settings = load_settings()
    quote = MarketQuote(
        market_id="yes-1",
        yes_price=0.49,
        metadata={"best_bid": 0.10, "best_ask": 0.40, "liquidity_clob": 5000.0},
    )

    result = evaluate_market_guards(make_signal(), make_decision(), quote, settings)

    assert result.allowed is False
    assert result.reason == "spread_above_limit"


def test_market_guards_reject_low_liquidity() -> None:
    settings = load_settings()
    quote = MarketQuote(
        market_id="yes-1",
        yes_price=0.49,
        metadata={"best_bid": 0.45, "best_ask": 0.49, "liquidity_clob": 100.0},
    )

    result = evaluate_market_guards(make_signal(), make_decision(), quote, settings)

    assert result.allowed is False
    assert result.reason == "liquidity_below_minimum"


def test_window_key_prefers_slug_when_available() -> None:
    quote = MarketQuote(
        market_id="yes-1",
        yes_price=0.49,
        metadata={"slug": "btc-updown-5m-123"},
    )

    key = build_window_key(make_signal(), quote)

    assert key == "BTC/USDT:5m:btc-updown-5m-123"
