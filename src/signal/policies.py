from __future__ import annotations

from dataclasses import dataclass

from src.core.config import Settings
from src.core.schemas import Decision, Signal


@dataclass(frozen=True)
class TwoStagePolicyConfig:
    stage1_threshold: float | None = None
    buy_threshold: float | None = None


def get_policy_config(settings: Settings, policy_name: str) -> TwoStagePolicyConfig:
    try:
        payload = settings.signal.policies[policy_name]
    except KeyError as exc:
        raise KeyError(f"Unknown signal policy '{policy_name}'.") from exc
    return TwoStagePolicyConfig(**payload)


def evaluate_two_stage_signal(
    signal: Signal,
    settings: Settings,
    policy_name: str = "two_stage_policy",
) -> Decision:
    policy = get_policy_config(settings, policy_name)
    stage1_threshold = (
        float(signal.decision_context["stage1_threshold"])
        if signal.decision_context.get("stage1_threshold") is not None
        else float(policy.stage1_threshold or 0.5)
    )
    buy_threshold = (
        float(signal.decision_context["buy_threshold"])
        if signal.decision_context.get("buy_threshold") is not None
        else float(policy.buy_threshold or 0.5)
    )
    p_active = float(signal.p_active or 0.0)
    if p_active < stage1_threshold:
        return Decision(
            should_trade=False,
            side=None,
            edge=None,
            reason="stage1_below_threshold",
            target_size=0.0,
        )

    side = "YES" if float(signal.p_up) >= buy_threshold else "NO"
    confidence = abs(float(signal.p_up) - buy_threshold)
    return Decision(
        should_trade=True,
        side=side,
        edge=confidence,
        reason="two_stage_signal_passed",
        target_size=float(settings.execution.fixed_contract_size),
    )
