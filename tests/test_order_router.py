from __future__ import annotations

from datetime import UTC, datetime

from src.core.schemas import Decision, MarketQuote, Signal
from src.execution.order_router import build_order_request


def _make_signal() -> Signal:
    return Signal(
        asset="BTC/USDT",
        horizon="5m",
        t0=datetime(2026, 4, 8, 13, 45, tzinfo=UTC),
        p_down=0.21,
        p_flat=0.19,
        p_up=0.60,
        p_active=0.74,
        model_version="m1",
        feature_version="v1",
        decision_context={
            "stage1_threshold": 0.6,
            "up_threshold": 0.65,
            "down_threshold": 0.64,
            "margin_threshold": 0.08,
            "stage1_rejected": False,
        },
    )


def test_build_order_request_carries_two_stage_context_for_yes_orders() -> None:
    order = build_order_request(
        _make_signal(),
        Decision(should_trade=True, side="YES", edge=0.11, reason="two_stage_signal_passed", target_size=5.0),
        MarketQuote(market_id="yes-1", yes_price=0.48, no_price=0.52, metadata={"no_token_id": "no-1"}),
    )

    assert order.market_id == "yes-1"
    assert order.side == "YES"
    assert order.metadata["p_active"] == 0.74
    assert order.metadata["p_down"] == 0.21
    assert order.metadata["p_flat"] == 0.19
    assert order.metadata["p_up"] == 0.60
    assert order.metadata["stage1_threshold"] == 0.6
    assert order.metadata["up_threshold"] == 0.65


def test_build_order_request_uses_no_token_and_no_price_for_no_orders() -> None:
    order = build_order_request(
        _make_signal(),
        Decision(should_trade=True, side="NO", edge=0.12, reason="two_stage_signal_passed", target_size=5.0),
        MarketQuote(market_id="yes-1", yes_price=0.48, no_price=0.52, metadata={"no_token_id": "no-1"}),
    )

    assert order.market_id == "no-1"
    assert order.side == "NO"
    assert order.price == 0.52
