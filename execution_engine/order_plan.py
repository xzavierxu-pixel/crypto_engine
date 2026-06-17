from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from math import floor

from execution_engine.config import ExecutionEdgeConfig, OrdersConfig
from execution_engine.limit_configs import down_limit, up_limit
from src.core.schemas import Decision, MarketQuote, OrderRequest, Signal

LOGGER = logging.getLogger(__name__)
LIMIT_CONFIG_BEST_ASK_OFFSET_MODE = "limit_config_best_ask_offset"
Q80_BEST_ASK_OFFSET_MODE = "min_q80_final_price_and_best_ask_offset"
SAFE_GAP_BEST_ASK_OFFSET_MODE = "min_safe_gap_price_and_best_ask_offset"


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


def ceil_to_two_decimals(price: float) -> float:
    return float(Decimal(str(price)).quantize(Decimal("0.01"), rounding=ROUND_CEILING))


def _limit_config_for_side(side: str) -> dict[float, float]:
    if side == "YES":
        return up_limit
    if side == "NO":
        return down_limit
    raise ValueError(f"Unsupported decision side for limit config: {side!r}")


def _limit_config_best_ask_offset_price(
    *,
    decision: Decision,
    best_ask: float | None,
    leg_name: str,
    price_mode: str,
    config: OrdersConfig,
) -> tuple[float | None, float | None, float | None, dict | None]:
    if best_ask is None:
        return None, None, None, {"leg": leg_name, "reason": "missing_best_ask", "price_mode": price_mode}
    lookup_price = ceil_to_two_decimals(float(best_ask))
    offset_lookup = _limit_config_for_side(str(decision.side))
    offset = offset_lookup.get(lookup_price)
    if offset is None:
        return (
            None,
            lookup_price,
            None,
            {
                "leg": leg_name,
                "reason": "missing_limit_offset",
                "price_mode": price_mode,
                "side": decision.side,
                "quote_source": "best_ask",
                "best_ask": float(best_ask),
                "lookup_price": lookup_price,
            },
        )
    raw_price = lookup_price - float(offset)
    if raw_price <= 0 or raw_price < config.min_price or raw_price > config.max_price:
        return (
            None,
            lookup_price,
            float(offset),
            {
                "leg": leg_name,
                "reason": "invalid_limit_config_price",
                "price_mode": price_mode,
                "side": decision.side,
                "quote_source": "best_ask",
                "best_ask": float(best_ask),
                "lookup_price": lookup_price,
                "offset": float(offset),
                "raw_price": raw_price,
                "min_price": config.min_price,
                "max_price": config.max_price,
            },
        )
    return raw_price, lookup_price, float(offset), None


def _price_estimator_min_best_ask_offset_price(
    *,
    signal: Signal,
    decision: Decision,
    best_ask: float | None,
    leg_name: str,
    price_mode: str,
    context_key: str,
    context_label: str,
    fallback_reason: str,
    config: OrdersConfig,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    str | None,
    str,
    float | None,
    float | None,
    dict | None,
]:
    quote_reference = None if best_ask is None else float(best_ask)
    quote_source = f"price_estimator_{context_label}_best_ask"
    estimator_context = signal.decision_context.get(context_key)
    best_ask_offset = float(signal.decision_context.get("price_estimator_best_ask_offset", 0.01))
    if estimator_context is not None and best_ask is not None:
        estimator_price = float(estimator_context)
        best_ask_offset_price = float(best_ask) - best_ask_offset
        return (
            min(estimator_price, best_ask_offset_price),
            estimator_price,
            best_ask_offset_price,
            quote_reference,
            None,
            quote_source,
            None,
            None,
            None,
        )
    fallback = signal.decision_context.get("price_estimator_fallback_price_mode", LIMIT_CONFIG_BEST_ASK_OFFSET_MODE)
    if fallback != LIMIT_CONFIG_BEST_ASK_OFFSET_MODE:
        return (
            None,
            None,
            None,
            quote_reference,
            str(fallback),
            quote_source,
            None,
            None,
            {
                "leg": leg_name,
                "reason": fallback_reason,
                "price_mode": price_mode,
                "fallback_price_mode": fallback,
                "has_best_ask": best_ask is not None,
            },
        )
    raw_price, lookup_price, offset, skip = _limit_config_best_ask_offset_price(
        decision=decision,
        best_ask=None if best_ask is None else float(best_ask),
        leg_name=leg_name,
        price_mode=str(fallback),
        config=config,
    )
    if skip is not None:
        return None, None, None, quote_reference, str(fallback), "best_ask", lookup_price, offset, skip
    return raw_price, None, None, quote_reference, str(fallback), "best_ask", lookup_price, offset, None


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
        if not leg.enabled:
            skipped.append({"leg": name, "reason": "disabled_leg", "enabled": False})
            continue
        if leg.size <= 0:
            skipped.append({"leg": name, "reason": "disabled_leg", "size": float(leg.size)})
            continue

        lookup_price = None
        offset = None
        q80_price = None
        safe_gap_price = None
        best_ask_offset_price = None
        price_estimator_fallback = None
        if leg.price_mode in {Q80_BEST_ASK_OFFSET_MODE, SAFE_GAP_BEST_ASK_OFFSET_MODE}:
            if leg.price_mode == Q80_BEST_ASK_OFFSET_MODE:
                context_key = "price_estimator_q80_rounded"
                context_label = "q80"
                fallback_reason = "missing_price_estimator_q80"
                warning_label = "q80"
            else:
                context_key = "price_estimator_safe_gap_rounded"
                context_label = "safe_gap"
                fallback_reason = "missing_price_estimator_safe_gap"
                warning_label = "safe-gap"
            (
                raw_price,
                estimator_price,
                best_ask_offset_price,
                quote_reference,
                price_estimator_fallback,
                quote_source,
                lookup_price,
                offset,
                skip,
            ) = _price_estimator_min_best_ask_offset_price(
                signal=signal,
                decision=decision,
                best_ask=None if best_ask is None else float(best_ask),
                leg_name=name,
                price_mode=leg.price_mode,
                context_key=context_key,
                context_label=context_label,
                fallback_reason=fallback_reason,
                config=config,
            )
            if skip is not None:
                if skip.get("reason") == fallback_reason:
                    skipped.append(skip)
                else:
                    LOGGER.warning("Skipping order leg because %s fallback price is unavailable: %s", warning_label, skip)
                    skipped.append(skip)
                continue
            if leg.price_mode == Q80_BEST_ASK_OFFSET_MODE:
                q80_price = estimator_price
            else:
                safe_gap_price = estimator_price
        elif leg.price_mode == LIMIT_CONFIG_BEST_ASK_OFFSET_MODE:
            quote_reference = None if best_ask is None else float(best_ask)
            quote_source = "best_ask"
            raw_price, lookup_price, offset, skip = _limit_config_best_ask_offset_price(
                decision=decision,
                best_ask=None if best_ask is None else float(best_ask),
                leg_name=name,
                price_mode=leg.price_mode,
                config=config,
            )
            if skip is not None:
                LOGGER.warning("Skipping order leg because no limit offset is configured: %s", skip)
                skipped.append(skip)
                continue
        elif best_bid is not None:
            quote_reference = float(best_bid)
            quote_source = "best_bid"
            if leg.price_mode == "min_best_bid_offset_and_cap":
                offset = leg.best_bid_offset if leg.best_bid_offset is not None else leg.offset
                raw_price = min(quote_reference + float(offset), leg.price_cap)
            elif leg.price_mode == "reference_multiplier_offset_and_cap":
                raw_price = min((quote_reference * leg.reference_multiplier) + leg.offset, leg.price_cap)
            else:
                skipped.append({"leg": name, "reason": "unsupported_price_mode", "price_mode": leg.price_mode})
                continue
        else:
            quote_reference = float(best_ask)
            quote_source = "best_ask"
            if name == "first":
                raw_price = min(quote_reference - tick_size, leg.price_cap)
            else:
                raw_price = min((quote_reference * leg.reference_multiplier) + leg.offset - tick_size, leg.price_cap)
        if leg.round_decimals is not None:
            raw_price = round(raw_price, int(leg.round_decimals))
        price = floor_to_tick(raw_price, tick_size)
        if price < config.min_price:
            price = floor_to_tick(config.min_price, tick_size)
        if price > config.max_price:
            action = config.on_invalid_second_order if name == "second" else "skip"
            if action == "clamp":
                price = min(price, config.max_price)
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
                    "limit_config_lookup_price": lookup_price,
                    "limit_config_offset": float(offset) if offset is not None else None,
                    "price_estimator_q80_rounded": q80_price,
                    "price_estimator_safe_gap_rounded": safe_gap_price,
                    "price_estimator_safe_gap_action": signal.decision_context.get("price_estimator_safe_gap_action"),
                    "price_estimator_safe_gap_conf_ok": signal.decision_context.get("price_estimator_safe_gap_conf_ok"),
                    "price_estimator_safe_gap_f_model": signal.decision_context.get("price_estimator_safe_gap_f_model"),
                    "price_estimator_best_ask_offset_price": best_ask_offset_price,
                    "price_estimator_fallback_price_mode": price_estimator_fallback,
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
