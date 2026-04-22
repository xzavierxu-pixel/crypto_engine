from __future__ import annotations

from src.core.schemas import Decision, MarketQuote, OrderRequest, Signal


def build_order_request(
    signal: Signal,
    decision: Decision,
    quote: MarketQuote,
) -> OrderRequest:
    if not decision.should_trade or decision.side is None:
        raise ValueError("Cannot build order request for a non-trade decision.")

    side = "YES" if decision.side in {"YES", "BUY"} else "NO"
    market_id = quote.market_id
    price = quote.yes_price
    if side == "NO":
        market_id = str(quote.metadata.get("no_token_id", quote.market_id))
        if quote.no_price is None:
            raise ValueError("Cannot build NO-side order without no_price.")
        price = quote.no_price

    return OrderRequest(
        market_id=market_id,
        side=side,
        price=price,
        size=decision.target_size,
        signal_t0=signal.t0,
        metadata={
            "edge": decision.edge,
            "reason": decision.reason,
            "model_version": signal.model_version,
            "feature_version": signal.feature_version,
        },
    )
