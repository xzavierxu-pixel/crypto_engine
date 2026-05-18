from __future__ import annotations

from dataclasses import dataclass
from math import floor

from execution_engine.config import ExecutionEdgeConfig, OrdersConfig
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
    edge_config: ExecutionEdgeConfig | None = None,
) -> OrderPlanResult:
    if not decision.should_trade or decision.side is None:
        return OrderPlanResult(orders=[], skipped=[{"reason": decision.reason}])
    tick_size = float(quote.metadata.get("tick_size") or config.tick_size_default)
    best_bid = quote.metadata.get("best_bid")
    best_ask = quote.metadata.get("best_ask")
    if best_bid is None and best_ask is None:
        return OrderPlanResult(orders=[], skipped=[{"reason": "missing_quote"}])

    target_token = select_target_token(decision, quote)
    orders: list[OrderRequest] = []
    skipped: list[dict] = []
    edge_config = edge_config or ExecutionEdgeConfig()

    for name, leg in [("first", config.first), ("second", config.second)]:
        if leg.size <= 0:
            skipped.append({"leg": name, "reason": "disabled_leg", "size": float(leg.size)})
            continue
        if best_bid is not None:
            quote_reference = float(best_bid)
            quote_source = "best_bid"
            raw_price = min(quote_reference, leg.price_cap) + leg.offset
        else:
            quote_reference = float(best_ask)
            quote_source = "best_ask"
            raw_price = min(quote_reference, leg.price_cap) + leg.offset - tick_size
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
                        "quote_source": quote_source,
                        "raw_price": raw_price,
                        "price": price,
                    }
                )
                continue
        size = _order_size_for_price(float(leg.size), price, edge_config)
        edge_skip = _edge_skip_reason(
            signal,
            decision,
            price,
            size,
            best_bid=None if best_bid is None else float(best_bid),
            best_ask=None if best_ask is None else float(best_ask),
            leg_name=name,
            config=edge_config,
        )
        if edge_skip is not None:
            skipped.append(edge_skip)
            continue
        orders.append(
            OrderRequest(
                market_id=target_token,
                side=str(decision.side),
                price=price,
                size=size,
                signal_t0=signal.t0,
                metadata={
                    "leg": name,
                    "quote_source": quote_source,
                    "best_bid": None if best_bid is None else float(best_bid),
                    "best_ask": None if best_ask is None else float(best_ask),
                    "tick_size": tick_size,
                    "p_up": signal.p_up,
                    "p_down": signal.p_down,
                    "t_up": signal.decision_context.get("t_up"),
                    "t_down": signal.decision_context.get("t_down"),
                    "configured_size": float(leg.size),
                    "size_to_max_notional": edge_config.size_to_max_notional,
                    "max_order_notional": edge_config.max_order_notional,
                },
            )
        )
    return OrderPlanResult(orders=orders, skipped=skipped)


def _order_size_for_price(size: float, price: float, config: ExecutionEdgeConfig) -> float:
    if not config.enabled or not config.size_to_max_notional or config.max_order_notional is None:
        return size
    if price <= 0:
        return size
    return config.max_order_notional / price


def _edge_skip_reason(
    signal: Signal,
    decision: Decision,
    price: float,
    size: float,
    *,
    best_bid: float | None,
    best_ask: float | None,
    leg_name: str,
    config: ExecutionEdgeConfig,
) -> dict | None:
    if not config.enabled:
        return None
    if leg_name == "first" and not config.apply_to_first_leg:
        return None
    if leg_name == "second" and not config.apply_to_second_leg:
        return None

    p_up = signal.p_up
    if p_up is None:
        return {"leg": leg_name, "reason": "missing_probability"}
    fair_value = float(p_up) if decision.side == "YES" else 1.0 - float(p_up)
    edge = fair_value - price
    spread = None if best_bid is None or best_ask is None else best_ask - best_bid
    notional = price * size

    details = {
        "leg": leg_name,
        "price": price,
        "size": size,
        "notional": notional,
        "side": decision.side,
        "p_up": float(p_up),
        "fair_value": fair_value,
        "edge": edge,
        "min_edge": config.min_edge,
        "max_buy_price": config.max_buy_price,
        "max_spread": config.max_spread,
        "max_order_notional": config.max_order_notional,
        "size_to_max_notional": config.size_to_max_notional,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
    }
    if edge < config.min_edge:
        return {**details, "reason": "edge_below_minimum"}
    if config.max_buy_price is not None and price > config.max_buy_price:
        return {**details, "reason": "price_above_max_buy_price"}
    if config.max_spread is not None and spread is not None and spread > config.max_spread:
        return {**details, "reason": "spread_too_wide"}
    if config.max_order_notional is not None and notional > config.max_order_notional:
        return {**details, "reason": "notional_above_max_order"}
    return None
