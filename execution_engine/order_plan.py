from __future__ import annotations

from dataclasses import dataclass
from math import floor

from execution_engine.config import OrdersConfig
from src.core.schemas import Decision, MarketQuote, OrderRequest, Signal


@dataclass(frozen=True)
class OrderPlanResult:
    orders: list[OrderRequest]
    skipped: list[dict]


def select_target_token(decision: Decision, quote: MarketQuote) -> str:
    if decision.side == "YES":
        return str(quote.metadata.get("yes_token_id", quote.market_id))
    if decision.side == "NO":
        return str(quote.metadata.get("no_token_id", quote.market_id))
    raise ValueError(f"Unsupported decision side for order planning: {decision.side!r}")


def floor_to_tick(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        raise ValueError("tick_size must be positive.")
    return round(floor((price + 1e-12) / tick_size) * tick_size, 10)


def build_two_limit_order_plan(
    signal: Signal,
    decision: Decision,
    quote: MarketQuote,
    config: OrdersConfig,
) -> OrderPlanResult:
    if not decision.should_trade or decision.side is None:
        return OrderPlanResult(orders=[], skipped=[{"reason": decision.reason}])
    best_bid = quote.metadata.get("best_bid")
    if best_bid is None:
        return OrderPlanResult(orders=[], skipped=[{"reason": "missing_best_bid"}])

    tick_size = float(quote.metadata.get("tick_size") or config.tick_size_default)
    target_token = select_target_token(decision, quote)
    orders: list[OrderRequest] = []
    skipped: list[dict] = []

    for name, leg in [("first", config.first), ("second", config.second)]:
        raw_price = min(float(best_bid), leg.price_cap) + leg.offset
        price = floor_to_tick(raw_price, tick_size)
        if price < config.min_price or price > config.max_price:
            action = config.on_invalid_second_order if name == "second" else "skip"
            if action == "clamp":
                price = min(max(price, config.min_price), config.max_price)
                price = floor_to_tick(price, tick_size)
            else:
                skipped.append(
                    {
                        "leg": name,
                        "reason": "invalid_price",
                        "raw_price": raw_price,
                        "price": price,
                    }
                )
                continue
        orders.append(
            OrderRequest(
                market_id=target_token,
                side=str(decision.side),
                price=price,
                size=float(leg.size),
                signal_t0=signal.t0,
                metadata={
                    "leg": name,
                    "best_bid": float(best_bid),
                    "tick_size": tick_size,
                    "p_up": signal.p_up,
                    "p_down": signal.p_down,
                    "t_up": signal.decision_context.get("t_up"),
                    "t_down": signal.decision_context.get("t_down"),
                },
            )
        )
    return OrderPlanResult(orders=orders, skipped=skipped)

