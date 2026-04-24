from __future__ import annotations

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
    if signal.decision_context.get("up_threshold") is None and policy.up_threshold is None:
        raise ValueError("up_threshold must be provided by the artifact or signal context.")
    if signal.decision_context.get("down_threshold") is None and policy.down_threshold is None:
        raise ValueError("down_threshold must be provided by the artifact or signal context.")
    if signal.decision_context.get("margin_threshold") is None and policy.margin_threshold is None:
        raise ValueError("margin_threshold must be provided by the artifact or signal context.")

    stage1_threshold = float(signal.decision_context.get("stage1_threshold", policy.stage1_threshold))
    up_threshold = float(signal.decision_context.get("up_threshold", policy.up_threshold))
    down_threshold = float(signal.decision_context.get("down_threshold", policy.down_threshold))
    margin_threshold = float(signal.decision_context.get("margin_threshold", policy.margin_threshold))
    p_active = float(signal.p_active or 0.0)
    if p_active < stage1_threshold:
        return Decision(
            should_trade=False,
            side=None,
            edge=None,
            reason="stage1_below_threshold",
            target_size=0.0,
        )

    p_up = float(signal.p_up or 0.0)
    p_down = float(signal.p_down or 0.0)
    up_margin = p_up - p_down
    down_margin = p_down - p_up
    yes_ok = p_up >= up_threshold and up_margin >= margin_threshold
    no_ok = p_down >= down_threshold and down_margin >= margin_threshold
    if yes_ok and no_ok:
        side = "YES" if up_margin >= down_margin else "NO"
    elif yes_ok:
        side = "YES"
    elif no_ok:
        side = "NO"
    else:
        return Decision(
            should_trade=False,
            side=None,
            edge=None,
            reason="stage2_below_threshold",
            target_size=0.0,
        )

    confidence = up_margin if side == "YES" else down_margin
    return Decision(
        should_trade=True,
        side=side,
        edge=confidence,
        reason="two_stage_signal_passed",
        target_size=float(settings.execution.fixed_contract_size),
    )
