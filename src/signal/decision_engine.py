from __future__ import annotations

from src.core.config import Settings
from src.core.schemas import Decision, RiskState, Signal
from src.horizons.registry import get_horizon_spec
from src.signal.policies import evaluate_two_stage_signal


def evaluate_entry(
    signal: Signal,
    yes_price: float,
    settings: Settings,
    risk_state: RiskState | None = None,
    horizon_name: str | None = None,
) -> Decision:
    horizon = get_horizon_spec(settings, horizon_name or signal.horizon)
    policy_name = horizon.signal_policy or "two_stage_policy"
    return evaluate_two_stage_signal(
        signal,
        settings=settings,
        policy_name=policy_name,
    )
