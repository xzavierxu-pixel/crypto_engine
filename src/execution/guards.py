from __future__ import annotations

from src.core.config import Settings
from src.core.schemas import Decision, GuardResult, MarketQuote, Signal


def build_window_key(signal: Signal, quote: MarketQuote) -> str:
    slug = quote.metadata.get("slug")
    if slug:
        return f"{signal.asset}:{signal.horizon}:{slug}"
    return f"{signal.asset}:{signal.horizon}:{signal.t0.isoformat()}"


def evaluate_market_guards(
    signal: Signal,
    decision: Decision,
    quote: MarketQuote,
    settings: Settings,
) -> GuardResult:
    if not decision.should_trade:
        return GuardResult(allowed=False, reason="decision_rejected", details={"decision_reason": decision.reason})

    safeguards = settings.execution.safeguards
    best_ask = quote.metadata.get("best_ask")
    best_bid = quote.metadata.get("best_bid")
    liquidity_clob = quote.metadata.get("liquidity_clob")

    if safeguards.get("require_best_ask", True) and best_ask is None:
        return GuardResult(allowed=False, reason="missing_best_ask")

    max_spread = safeguards.get("max_spread")
    if max_spread is not None and best_bid is not None and best_ask is not None:
        spread = float(best_ask) - float(best_bid)
        if spread > float(max_spread):
            return GuardResult(
                allowed=False,
                reason="spread_above_limit",
                details={"spread": spread, "max_spread": float(max_spread)},
            )

    min_liquidity = safeguards.get("min_liquidity_clob")
    if min_liquidity is not None and liquidity_clob is not None:
        if float(liquidity_clob) < float(min_liquidity):
            return GuardResult(
                allowed=False,
                reason="liquidity_below_minimum",
                details={
                    "liquidity_clob": float(liquidity_clob),
                    "min_liquidity_clob": float(min_liquidity),
                },
            )

    return GuardResult(allowed=True, reason="market_guards_passed")
