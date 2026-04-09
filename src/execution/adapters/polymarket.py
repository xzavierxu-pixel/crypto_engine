from __future__ import annotations

import os

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType

from src.core.config import Settings
from src.core.schemas import MarketQuote, OrderRequest
from src.execution.adapters.base import ExecutionAdapter


class PolymarketExecutionAdapter(ExecutionAdapter):
    name = "polymarket"

    def __init__(self, settings: Settings, client: ClobClient | None = None) -> None:
        self.settings = settings
        self.config = settings.execution.polymarket
        self.client = client or self._build_client()

    def _build_client(self) -> ClobClient:
        creds = self._load_api_creds_from_env()
        private_key = os.getenv(self.config.get("private_key_env", "POLYMARKET_PRIVATE_KEY"))
        signature_type = self.config.get("signature_type")
        funder = self.config.get("funder")
        return ClobClient(
            self.config["host"],
            chain_id=int(self.config["chain_id"]),
            key=private_key,
            creds=creds,
            signature_type=signature_type,
            funder=funder,
        )

    def _load_api_creds_from_env(self) -> ApiCreds | None:
        api_key = os.getenv(self.config.get("api_key_env", "POLYMARKET_API_KEY"))
        api_secret = os.getenv(self.config.get("api_secret_env", "POLYMARKET_API_SECRET"))
        api_passphrase = os.getenv(
            self.config.get("api_passphrase_env", "POLYMARKET_API_PASSPHRASE")
        )
        if api_key and api_secret and api_passphrase:
            return ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )
        return None

    def _ensure_authenticated(self) -> None:
        if getattr(self.client, "creds", None) is not None:
            return
        if getattr(self.client, "signer", None) is None:
            raise RuntimeError(
                "Polymarket private key is not configured. Set the private key environment variable "
                f"'{self.config.get('private_key_env', 'POLYMARKET_PRIVATE_KEY')}'."
            )
        self.client.set_api_creds(self.client.create_or_derive_api_creds())

    def list_active_markets(self) -> list[dict]:
        markets = self._fetch_active_markets(self.client.get_simplified_markets)
        if markets:
            return markets
        return self._fetch_active_markets(self.client.get_sampling_simplified_markets)

    def _fetch_active_markets(self, fetch_page) -> list[dict]:
        markets: list[dict] = []
        next_cursor = "MA=="
        max_pages = int(self.config.get("max_pages", 3))
        seen_cursors: set[str] = set()

        for _ in range(max_pages):
            seen_cursors.add(next_cursor)
            payload = fetch_page(next_cursor=next_cursor)
            data = payload.get("data", [])
            for market in data:
                if (
                    not market.get("active", False)
                    or market.get("closed", False)
                    or market.get("archived", False)
                    or not market.get("accepting_orders", False)
                ):
                    continue

                tokens = market.get("tokens", [])
                if len(tokens) < 2:
                    continue

                yes_token = next((token for token in tokens if str(token.get("outcome", "")).lower() == "yes"), tokens[0])
                no_token = next(
                    (token for token in tokens if str(token.get("outcome", "")).lower() == "no"),
                    tokens[1],
                )

                markets.append(
                    {
                        "condition_id": market["condition_id"],
                        "market_id": market["condition_id"],
                        "yes_token_id": yes_token["token_id"],
                        "no_token_id": no_token["token_id"],
                        "yes_outcome": yes_token.get("outcome"),
                        "no_outcome": no_token.get("outcome"),
                        "yes_price": float(yes_token.get("price", 0.0)),
                        "no_price": float(no_token.get("price", 0.0)),
                        "active": market.get("active", False),
                        "closed": market.get("closed", False),
                        "archived": market.get("archived", False),
                        "accepting_orders": market.get("accepting_orders", False),
                        "tokens": tokens,
                    }
                )

            new_cursor = payload.get("next_cursor")
            if not data or not new_cursor or new_cursor in seen_cursors or new_cursor == "LTE=":
                break
            next_cursor = new_cursor

        return markets

    def get_orderbook(self, market_id: str) -> MarketQuote:
        orderbook = self.client.get_order_book(market_id)
        best_bid = float(orderbook.bids[0].price) if orderbook.bids else None
        best_ask = float(orderbook.asks[0].price) if orderbook.asks else None
        last_trade = float(orderbook.last_trade_price) if orderbook.last_trade_price else None
        yes_price = best_ask if best_ask is not None else last_trade
        if yes_price is None:
            raise RuntimeError(f"No ask or last trade price available for token '{market_id}'.")

        return MarketQuote(
            market_id=market_id,
            yes_price=yes_price,
            no_price=(1.0 - yes_price) if yes_price is not None else None,
            metadata={
                "market": orderbook.market,
                "asset_id": orderbook.asset_id,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "tick_size": orderbook.tick_size,
                "last_trade_price": last_trade,
                "hash": orderbook.hash,
            },
        )

    def place_limit_order(self, order: OrderRequest) -> dict:
        self._ensure_authenticated()
        clob_side = "BUY" if order.side in {"YES", "NO", "BUY"} else "SELL"
        signed_order = self.client.create_order(
            OrderArgs(
                token_id=order.market_id,
                price=order.price,
                size=order.size,
                side=clob_side,
            )
        )
        response = self.client.post_order(signed_order, OrderType.GTC)
        return {
            "request": {
                "token_id": order.market_id,
                "side": clob_side,
                "price": order.price,
                "size": order.size,
            },
            "response": response,
        }

    def cancel_order(self, order_id: str) -> dict:
        self._ensure_authenticated()
        return self.client.cancel(order_id)
