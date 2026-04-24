from __future__ import annotations

from datetime import UTC, datetime

from src.core.schemas import Decision, MarketQuote, Signal
from src.execution.audit import decision_evaluated_event, signal_generated_event, stage1_drift_alert_event


def _make_signal() -> Signal:
    return Signal(
        asset="BTC/USDT",
        horizon="5m",
        t0=datetime(2026, 4, 22, 12, 0, tzinfo=UTC),
        p_down=0.22,
        p_flat=0.23,
        p_up=0.55,
        p_active=0.72,
        model_version="m1",
        feature_version="v1",
        decision_context={
            "stage1_drift": {"alert": True, "ks_distance": 0.23},
            "stage1_threshold": 0.6,
            "up_threshold": 0.65,
            "down_threshold": 0.64,
            "margin_threshold": 0.08,
            "stage1_rejected": False,
        },
    )


def test_signal_generated_event_includes_stage1_context() -> None:
    event = signal_generated_event(_make_signal())

    assert event.event_type == "signal_generated"
    assert event.payload["p_active"] == 0.72
    assert event.payload["p_down"] == 0.22
    assert event.payload["stage1_threshold"] == 0.6
    assert event.payload["decision_context"]["stage1_drift"]["ks_distance"] == 0.23


def test_stage1_drift_alert_event_uses_signal_and_drift_state() -> None:
    signal = _make_signal()
    drift_state = {"alert": True, "ks_distance": 0.23, "consecutive_alerts": 3}

    event = stage1_drift_alert_event(signal, drift_state)

    assert event.event_type == "stage1_drift_alert"
    assert event.payload["asset"] == "BTC/USDT"
    assert event.payload["drift"]["consecutive_alerts"] == 3


def test_decision_evaluated_event_includes_thresholds_and_probs() -> None:
    signal = _make_signal()
    quote = MarketQuote(market_id="m1", yes_price=0.49)
    decision = Decision(
        should_trade=True,
        side="YES",
        edge=0.12,
        reason="two_stage_signal_passed",
        target_size=5.0,
    )

    event = decision_evaluated_event(signal, quote, decision)

    assert event.payload["p_down"] == 0.22
    assert event.payload["p_flat"] == 0.23
    assert event.payload["p_up"] == 0.55
    assert event.payload["stage1_threshold"] == 0.6
    assert event.payload["up_threshold"] == 0.65
    assert event.payload["side"] == "YES"
