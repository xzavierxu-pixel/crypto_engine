from __future__ import annotations

from datetime import datetime, UTC

from src.core.schemas import AuditEvent, Decision, GuardResult, MarketQuote, OrderRequest, Signal


def signal_generated_event(signal: Signal) -> AuditEvent:
    return AuditEvent(
        event_type="signal_generated",
        timestamp=datetime.now(UTC),
        payload={
            "asset": signal.asset,
            "horizon": signal.horizon,
            "t0": signal.t0.isoformat(),
            "p_up": signal.p_up,
            "p_active": signal.p_active,
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
    return AuditEvent(
        event_type="decision_evaluated",
        timestamp=datetime.now(UTC),
        payload={
            "asset": signal.asset,
            "market_id": quote.market_id,
            "t0": signal.t0.isoformat(),
            "yes_price": quote.yes_price,
            "p_up": signal.p_up,
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
