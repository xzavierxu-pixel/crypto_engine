from __future__ import annotations

from datetime import UTC, datetime

from src.core.schemas import Signal
from src.execution.audit import signal_generated_event, stage1_drift_alert_event


def _make_signal() -> Signal:
    return Signal(
        asset="BTC/USDT",
        horizon="5m",
        t0=datetime(2026, 4, 22, 12, 0, tzinfo=UTC),
        p_up=0.55,
        p_active=0.72,
        model_version="m1",
        feature_version="v1",
        decision_context={"stage1_drift": {"alert": True, "ks_distance": 0.23}},
    )


def test_signal_generated_event_includes_stage1_context() -> None:
    event = signal_generated_event(_make_signal())

    assert event.event_type == "signal_generated"
    assert event.payload["p_active"] == 0.72
    assert event.payload["decision_context"]["stage1_drift"]["ks_distance"] == 0.23


def test_stage1_drift_alert_event_uses_signal_and_drift_state() -> None:
    signal = _make_signal()
    drift_state = {"alert": True, "ks_distance": 0.23, "consecutive_alerts": 3}

    event = stage1_drift_alert_event(signal, drift_state)

    assert event.event_type == "stage1_drift_alert"
    assert event.payload["asset"] == "BTC/USDT"
    assert event.payload["drift"]["consecutive_alerts"] == 3
