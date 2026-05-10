from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

from execution_engine.config import PolymarketConfig
from src.core.schemas import MarketQuote, OrderRequest


@dataclass(frozen=True)
class PolymarketMarket:
    slug: str
    market_id: str
    yes_token_id: str
    no_token_id: str
    active: bool
    closed: bool
    accepting_orders: bool
    metadata: dict[str, Any]


class PolymarketV2Adapter:
    def __init__(
        self,
        config: PolymarketConfig,
        client: Any | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.client = client or self._build_client()
        self.session = session or requests.Session()

    def _build_client(self) -> Any:
        try:
            from py_clob_client_v2 import ApiCreds, ClobClient
        except ImportError as exc:  # pragma: no cover - depends on optional deployment package.
            raise ImportError(
                "py-clob-client-v2 is required for live Polymarket execution. "
                "Install it in the deployment virtualenv before using live mode."
            ) from exc

        private_key = os.getenv(self.config.private_key_env)
        creds = None
        api_key = os.getenv(self.config.api_key_env)
        api_secret = os.getenv(self.config.api_secret_env)
        api_passphrase = os.getenv(self.config.api_passphrase_env)
        if api_key and api_secret and api_passphrase:
            creds = ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )
        return ClobClient(
            self.config.host,
            chain_id=int(self.config.chain_id),
            key=private_key,
            creds=creds,
            signature_type=self.config.signature_type,
            funder=self.config.funder,
        )

    def get_market_by_slug(self, slug: str) -> PolymarketMarket | None:
        response = self.session.get(
            f"{self.config.gamma_base_url.rstrip('/')}/markets",
            params={"slug": slug},
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return None
        return normalize_gamma_market(payload[0], slug=slug)

    def get_orderbook(self, token_id: str, metadata: dict[str, Any] | None = None) -> MarketQuote:
        orderbook = self.client.get_order_book(token_id)
        bids = getattr(orderbook, "bids", []) or []
        asks = getattr(orderbook, "asks", []) or []
        best_bid = float(bids[0].price) if bids else None
        best_ask = float(asks[0].price) if asks else None
        last_trade = getattr(orderbook, "last_trade_price", None)
        last_trade_price = float(last_trade) if last_trade else None
        yes_price = best_ask if best_ask is not None else last_trade_price
        if yes_price is None:
            raise RuntimeError(f"No ask or last trade price available for token '{token_id}'.")
        return MarketQuote(
            market_id=token_id,
            yes_price=yes_price,
            no_price=1.0 - yes_price,
            metadata={
                **(metadata or {}),
                "asset_id": getattr(orderbook, "asset_id", token_id),
                "market": getattr(orderbook, "market", None),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "tick_size": getattr(orderbook, "tick_size", None),
                "last_trade_price": last_trade_price,
                "hash": getattr(orderbook, "hash", None),
            },
        )

    def place_limit_order(self, order: OrderRequest) -> dict:
        self._ensure_authenticated()
        try:
            from py_clob_client_v2 import OrderArgs, OrderType, PartialCreateOrderOptions, Side
        except ImportError as exc:  # pragma: no cover - depends on optional deployment package.
            raise ImportError("py-clob-client-v2 is required to place live orders.") from exc

        args = OrderArgs(
            token_id=order.market_id,
            price=order.price,
            size=order.size,
            side=Side.BUY,
        )
        options = PartialCreateOrderOptions(tick_size=str(order.metadata.get("tick_size", "0.01")))
        if hasattr(self.client, "create_and_post_order"):
            response = self.client.create_and_post_order(
                order_args=args,
                options=options,
                order_type=OrderType.GTC,
            )
        else:
            signed = self.client.create_order(args)
            response = self.client.post_order(signed, OrderType.GTC)
        return {
            "request": {
                "token_id": order.market_id,
                "side": "BUY",
                "price": order.price,
                "size": order.size,
            },
            "response": response,
        }

    def _ensure_authenticated(self) -> None:
        if getattr(self.client, "creds", None) is not None:
            return
        if getattr(self.client, "signer", None) is None:
            raise RuntimeError(
                f"Polymarket private key is not configured in {self.config.private_key_env}."
            )
        if hasattr(self.client, "create_or_derive_api_key"):
            creds = self.client.create_or_derive_api_key()
            if hasattr(self.client, "set_api_creds"):
                self.client.set_api_creds(creds)
            return
        if hasattr(self.client, "create_or_derive_api_creds"):
            creds = self.client.create_or_derive_api_creds()
            if hasattr(self.client, "set_api_creds"):
                self.client.set_api_creds(creds)


def normalize_gamma_market(market: dict[str, Any], slug: str) -> PolymarketMarket:
    tokens = _extract_tokens(market)
    if len(tokens) < 2:
        raise RuntimeError(f"Gamma market '{slug}' did not include enough token information.")
    yes = next((token for token in tokens if str(token.get("outcome", "")).lower() in {"yes", "up"}), tokens[0])
    no = next((token for token in tokens if str(token.get("outcome", "")).lower() in {"no", "down"}), tokens[1])
    return PolymarketMarket(
        slug=slug,
        market_id=str(market["id"]),
        yes_token_id=str(yes.get("token_id")),
        no_token_id=str(no.get("token_id")),
        active=bool(market.get("active", False)),
        closed=bool(market.get("closed", False)),
        accepting_orders=bool(market.get("acceptingOrders", market.get("accepting_orders", False))),
        metadata={
            "condition_id": market.get("conditionId"),
            "yes_outcome": yes.get("outcome"),
            "no_outcome": no.get("outcome"),
            "best_bid": _optional_float(market.get("bestBid")),
            "best_ask": _optional_float(market.get("bestAsk")),
            "liquidity_clob": _optional_float(market.get("liquidityClob")),
            "order_min_size": market.get("orderMinSize"),
            "tokens": tokens,
        },
    )


def _extract_tokens(market: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = market.get("tokens")
    if isinstance(tokens, list) and tokens:
        return tokens
    import json

    outcomes = _json_list(market.get("outcomes"), json)
    prices = _json_list(market.get("outcomePrices"), json)
    token_ids = _json_list(market.get("clobTokenIds"), json)
    return [
        {
            "token_id": token_ids[index] if index < len(token_ids) else None,
            "outcome": outcome,
            "price": float(prices[index]) if index < len(prices) else None,
        }
        for index, outcome in enumerate(outcomes)
    ]


def _json_list(value: Any, json_module) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return json_module.loads(value)
    raise TypeError(f"Unsupported JSON list payload type: {type(value)!r}")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
