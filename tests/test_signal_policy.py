from __future__ import annotations

from datetime import UTC, datetime

from src.core.config import load_settings
from src.core.schemas import RiskState, Signal
from src.signal.decision_engine import evaluate_entry


def test_decision_engine_accepts_yes_signal_when_stage2_yes_threshold_passes() -> None:
    settings = load_settings()
    signal = Signal(
        asset="BTC/USDT",
        horizon="5m",
        t0=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        p_down=0.15,
        p_flat=0.20,
        p_up=0.65,
        predicted_median_return=0.001,
        model_version="m1",
        feature_version="v1",
        p_active=0.74,
        decision_context={
            "stage1_threshold": 0.5,
        },
    )
    decision = evaluate_entry(
        signal,
        yes_price=0.50,
        settings=settings,
        risk_state=RiskState(current_exposure=0.0, max_total_exposure=0.20, active_positions=0),
    )
    assert decision.should_trade is True
    assert decision.side == "YES"
    assert decision.target_size == 5.0


def test_decision_engine_rejects_when_stage1_is_below_threshold() -> None:
    settings = load_settings()
    signal = Signal(
        asset="BTC/USDT",
        horizon="5m",
        t0=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        p_down=None,
        p_flat=None,
        p_up=None,
        predicted_median_return=None,
        model_version="m1",
        feature_version="v1",
        p_active=0.20,
        decision_context={
            "stage1_threshold": 0.5,
        },
    )
    decision = evaluate_entry(
        signal,
        yes_price=0.50,
        settings=settings,
        risk_state=RiskState(current_exposure=0.20, max_total_exposure=0.20, active_positions=1),
    )
    assert decision.should_trade is False
    assert decision.reason == "stage1_below_threshold"
