from __future__ import annotations

import math

from src.core.config import Settings
from src.core.schemas import Decision, Signal


def get_policy_config(settings: Settings, policy_name: str):
    return settings.signal.get_two_stage_policy(policy_name)


def evaluate_two_stage_signal(
    signal: Signal,
    settings: Settings,
    policy_name: str = "two_stage_policy",
) -> Decision:
    policy = get_policy_config(settings, policy_name)
    if signal.decision_context.get("stage1_threshold") is None and policy.stage1_threshold is None:
        raise ValueError("stage1_threshold must be provided by the artifact or signal context.")

    stage1_threshold = float(signal.decision_context.get("stage1_threshold", policy.stage1_threshold))
    p_active = float(signal.p_active or 0.0)
    if p_active < stage1_threshold:
        return Decision(
            should_trade=False,
            side=None,
            edge=None,
            reason="stage1_below_threshold",
            target_size=0.0,
        )

    predicted_return = signal.predicted_median_return
    if predicted_return is None:
        predicted_return = signal.decision_context.get("predicted_median_return")
    if predicted_return is None:
        return Decision(
            should_trade=False,
            side=None,
            edge=None,
            reason="stage2_missing_predicted_return",
            target_size=0.0,
        )
    predicted_return = float(predicted_return)
    if math.isnan(predicted_return):
        return Decision(
            should_trade=False,
            side=None,
            edge=None,
            reason="stage2_missing_predicted_return",
            target_size=0.0,
        )
    if predicted_return > 0.0:
        side = "YES"
    elif predicted_return < 0.0:
        side = "NO"
    else:
        return Decision(
            should_trade=False,
            side=None,
            edge=None,
            reason="stage2_zero_predicted_return",
            target_size=0.0,
        )

    return Decision(
        should_trade=True,
        side=side,
        edge=abs(predicted_return),
        reason="two_stage_signal_passed",
        target_size=float(settings.execution.fixed_contract_size),
    )
