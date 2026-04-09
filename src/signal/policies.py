from __future__ import annotations

from dataclasses import dataclass

from src.core.config import Settings
from src.core.schemas import Decision, RiskState, Signal
from src.sizing.fixed_fraction import FixedFractionSizer


@dataclass(frozen=True)
class EdgePolicyConfig:
    yes_price_min: float
    yes_price_max: float
    edge_threshold: float


def get_policy_config(settings: Settings, policy_name: str) -> EdgePolicyConfig:
    try:
        payload = settings.signal.policies[policy_name]
    except KeyError as exc:
        raise KeyError(f"Unknown signal policy '{policy_name}'.") from exc
    return EdgePolicyConfig(**payload)


def get_sizer(settings: Settings) -> FixedFractionSizer:
    plugin_name = settings.sizing.active_plugin
    if plugin_name != "fixed_fraction":
        raise KeyError(f"Unsupported sizing plugin '{plugin_name}'.")
    payload = settings.sizing.plugins[plugin_name]
    return FixedFractionSizer(
        single_position_cap=payload["single_position_cap"],
        max_total_exposure=payload.get("max_total_exposure"),
    )


def evaluate_edge_signal(
    signal: Signal,
    yes_price: float,
    settings: Settings,
    risk_state: RiskState | None = None,
    policy_name: str = "default_edge_policy",
) -> Decision:
    policy = get_policy_config(settings, policy_name)
    if yes_price < policy.yes_price_min or yes_price > policy.yes_price_max:
        return Decision(
            should_trade=False,
            side=None,
            edge=None,
            reason="yes_price_out_of_range",
            target_size=0.0,
        )

    edge = signal.p_up - yes_price
    if edge < policy.edge_threshold:
        return Decision(
            should_trade=False,
            side=None,
            edge=edge,
            reason="edge_below_threshold",
            target_size=0.0,
        )

    sizer = get_sizer(settings)
    target_size = sizer.size(edge, risk_state=risk_state)
    if target_size <= 0:
        return Decision(
            should_trade=False,
            side=None,
            edge=edge,
            reason="risk_capacity_exhausted",
            target_size=0.0,
        )

    return Decision(
        should_trade=True,
        side="YES",
        edge=edge,
        reason="edge_signal_passed",
        target_size=target_size,
    )
