from __future__ import annotations

from src.core.schemas import Decision, MarketQuote, OrderRequest, Signal


def build_order_request(
    signal: Signal,
    decision: Decision,
    quote: MarketQuote,
) -> OrderRequest:
    if not decision.should_trade or decision.side is None:
        raise ValueError("Cannot build order request for a non-trade decision.")

    return OrderRequest(
        market_id=quote.market_id,
        side=decision.side,
        price=quote.yes_price,
        size=decision.target_size,
        signal_t0=signal.t0,
        metadata={
            "edge": decision.edge,
            "reason": decision.reason,
            "model_version": signal.model_version,
            "feature_version": signal.feature_version,
        },
    )
