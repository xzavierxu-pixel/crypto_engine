from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from src.core.schemas import Decision, MarketQuote, Signal
from scripts.run_shadow import (
    _build_shadow_summary,
    _build_stage1_drift_monitor,
    _merge_quote_metadata,
    _resolve_quote,
)


class FakeMapper:
    def map_signal(self, signal: Signal) -> dict:
        return {
            "market_id": "gamma-1",
            "condition_id": "cond-1",
            "slug": "btc-updown-5m-123",
            "window_start": "2026-04-08T13:45:00+00:00",
            "window_end": "2026-04-08T13:50:00+00:00",
            "yes_token_id": "yes-1",
            "no_token_id": "no-1",
        }


class FakeAdapter:
    def get_orderbook(self, token_id: str) -> MarketQuote:
        assert token_id == "yes-1"
        return MarketQuote(
            market_id="yes-1",
            yes_price=0.49,
            no_price=0.51,
            metadata={"best_bid": 0.48, "best_ask": 0.49},
        )


def make_signal() -> Signal:
    return Signal(
        asset="BTC/USDT",
        horizon="5m",
        t0=datetime(2026, 4, 8, 13, 45, tzinfo=UTC),
        p_up=0.57,
        model_version="m1",
        feature_version="v1",
        p_active=0.71,
    )


def test_resolve_quote_uses_manual_inputs_when_provided() -> None:
    signal = make_signal()
    args = SimpleNamespace(market_id="manual-token", yes_price=0.52)

    quote, market = _resolve_quote(signal, args, mapper=FakeMapper(), adapter=FakeAdapter())

    assert quote.market_id == "manual-token"
    assert quote.yes_price == 0.52
    assert market is None


def test_resolve_quote_auto_maps_market_and_merges_metadata() -> None:
    signal = make_signal()
    args = SimpleNamespace(market_id=None, yes_price=None)

    quote, market = _resolve_quote(signal, args, mapper=FakeMapper(), adapter=FakeAdapter())

    assert market["market_id"] == "gamma-1"
    assert quote.market_id == "yes-1"
    assert quote.metadata["slug"] == "btc-updown-5m-123"
    assert quote.metadata["gamma_market_id"] == "gamma-1"


def test_build_shadow_summary_contains_signal_market_decision_and_order() -> None:
    signal = make_signal()
    quote = _merge_quote_metadata(
        MarketQuote(
            market_id="yes-1",
            yes_price=0.49,
            no_price=0.51,
            metadata={"best_bid": 0.48, "best_ask": 0.49},
        ),
        {
            "market_id": "gamma-1",
            "condition_id": "cond-1",
            "slug": "btc-updown-5m-123",
            "window_start": "2026-04-08T13:45:00+00:00",
            "window_end": "2026-04-08T13:50:00+00:00",
            "yes_token_id": "yes-1",
            "no_token_id": "no-1",
        },
    )
    decision = Decision(
        should_trade=True,
        side="YES",
        edge=0.08,
        reason="edge_signal_passed",
        target_size=0.02,
    )

    summary = _build_shadow_summary(
        signal,
        quote,
        decision,
        order={
            "market_id": "yes-1",
            "side": "YES",
            "price": 0.49,
            "size": 0.02,
        },
    )

    assert summary["signal"]["asset"] == "BTC/USDT"
    assert summary["signal"]["p_active"] == 0.71
    assert summary["market"]["slug"] == "btc-updown-5m-123"
    assert summary["decision"]["should_trade"] is True
    assert summary["order"]["market_id"] == "yes-1"


def test_build_stage1_drift_monitor_requires_real_reference_sample() -> None:
    assert _build_stage1_drift_monitor([]) is None

    monitor = _build_stage1_drift_monitor(
        [0.2, 0.4, 0.6, 0.8],
        threshold=0.01,
        window_size=4,
        min_history=2,
        alert_consecutive=2,
    )

    assert monitor is not None
    state = {}
    for value in [0.2, 0.4, 0.6, 0.8]:
        state = monitor.update(value)
    assert state["window_size"] == 4
