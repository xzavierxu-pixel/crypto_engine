from __future__ import annotations

from datetime import datetime, UTC

from src.core.schemas import AuditEvent, Decision, GuardResult, MarketQuote, OrderRequest, Signal


def signal_generated_event(signal: Signal) -> AuditEvent:
    context = signal.decision_context
    return AuditEvent(
        event_type="signal_generated",
        timestamp=datetime.now(UTC),
        payload={
            "asset": signal.asset,
            "horizon": signal.horizon,
            "t0": signal.t0.isoformat(),
            "p_down": signal.p_down,
            "p_flat": signal.p_flat,
            "p_up": signal.p_up,
            "p_active": signal.p_active,
            "predicted_median_return": signal.predicted_median_return,
            "stage1_threshold": context.get("stage1_threshold"),
            "up_threshold": context.get("up_threshold"),
            "down_threshold": context.get("down_threshold"),
            "margin_threshold": context.get("margin_threshold"),
            "side": context.get("side"),
            "stage1_rejected": context.get("stage1_rejected"),
            "decision_context": signal.decision_context,
        },
    )


def stage1_drift_alert_event(signal: Signal, drift_state: dict) -> AuditEvent:
    return AuditEvent(
        event_type="stage1_drift_alert",
        timestamp=datetime.now(UTC),
        payload={
            "asset": signal.asset,
            "horizon": signal.horizon,
            "t0": signal.t0.isoformat(),
            "p_active": signal.p_active,
            "drift": drift_state,
        },
    )


def stage2_drift_alert_event(signal: Signal, drift_state: dict) -> AuditEvent:
    return AuditEvent(
        event_type="stage2_drift_alert",
        timestamp=datetime.now(UTC),
        payload={
            "asset": signal.asset,
            "horizon": signal.horizon,
            "t0": signal.t0.isoformat(),
            "p_down": signal.p_down,
            "p_flat": signal.p_flat,
            "p_up": signal.p_up,
            "predicted_median_return": signal.predicted_median_return,
            "drift": drift_state,
        },
    )


def market_mapped_event(signal: Signal, quote: MarketQuote) -> AuditEvent:
    return AuditEvent(
        event_type="market_mapped",
        timestamp=datetime.now(UTC),
        payload={
            "asset": signal.asset,
            "market_id": quote.market_id,
            "t0": signal.t0.isoformat(),
            "yes_price": quote.yes_price,
            "metadata": quote.metadata,
        },
    )


def decision_evaluated_event(signal: Signal, quote: MarketQuote, decision: Decision) -> AuditEvent:
    context = signal.decision_context
    return AuditEvent(
        event_type="decision_evaluated",
        timestamp=datetime.now(UTC),
        payload={
            "asset": signal.asset,
            "market_id": quote.market_id,
            "t0": signal.t0.isoformat(),
            "yes_price": quote.yes_price,
            "p_down": signal.p_down,
            "p_flat": signal.p_flat,
            "p_up": signal.p_up,
            "p_active": signal.p_active,
            "predicted_median_return": signal.predicted_median_return,
            "stage1_threshold": context.get("stage1_threshold"),
            "up_threshold": context.get("up_threshold"),
            "down_threshold": context.get("down_threshold"),
            "margin_threshold": context.get("margin_threshold"),
            "should_trade": decision.should_trade,
            "side": decision.side,
            "edge": decision.edge,
            "reason": decision.reason,
            "target_size": decision.target_size,
        },
    )


def guard_evaluated_event(signal: Signal, quote: MarketQuote, guard: GuardResult) -> AuditEvent:
    return AuditEvent(
        event_type="guard_evaluated",
        timestamp=datetime.now(UTC),
        payload={
            "asset": signal.asset,
            "market_id": quote.market_id,
            "t0": signal.t0.isoformat(),
            "allowed": guard.allowed,
            "reason": guard.reason,
            "details": guard.details,
        },
    )


def order_created_event(order: OrderRequest, decision: Decision) -> AuditEvent:
    return AuditEvent(
        event_type="order_created",
        timestamp=datetime.now(UTC),
        payload={
            "market_id": order.market_id,
            "side": order.side,
            "price": order.price,
            "size": order.size,
            "reason": decision.reason,
            "edge": decision.edge,
        },
    )


def execution_skipped_event(signal: Signal, quote: MarketQuote, reason: str, details: dict | None = None) -> AuditEvent:
    return AuditEvent(
        event_type="execution_skipped",
        timestamp=datetime.now(UTC),
        payload={
            "asset": signal.asset,
            "market_id": quote.market_id,
            "t0": signal.t0.isoformat(),
            "reason": reason,
            "details": details or {},
        },
    )


def order_submitted_event(order: OrderRequest, response: dict, mode: str) -> AuditEvent:
    return AuditEvent(
        event_type="order_submitted",
        timestamp=datetime.now(UTC),
        payload={
            "market_id": order.market_id,
            "side": order.side,
            "price": order.price,
            "size": order.size,
            "mode": mode,
            "response": response,
        },
    )
