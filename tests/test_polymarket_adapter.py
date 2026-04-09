from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from py_clob_client.clob_types import OrderBookSummary, OrderSummary

from src.core.config import load_settings
from src.core.schemas import OrderRequest
from src.execution.adapters.polymarket import PolymarketExecutionAdapter


@dataclass
class FakeCreds:
    api_key: str
    api_secret: str
    api_passphrase: str


class FakeClobClient:
    def __init__(self) -> None:
        self.creds = FakeCreds("k", "s", "p")
        self.signer = object()
        self.last_create_order = None
        self.last_post_order = None
        self.last_cancel = None

    def get_simplified_markets(self, next_cursor: str = "MA==") -> dict:
        if next_cursor == "LTE=":
            return {"data": [], "next_cursor": "LTE="}
        return {
            "data": [
                {
                    "condition_id": "cond-1",
                    "tokens": [
                        {"token_id": "yes-1", "outcome": "Yes", "price": 0.52},
                        {"token_id": "no-1", "outcome": "No", "price": 0.48},
                    ],
                    "active": True,
                    "closed": False,
                    "archived": False,
                    "accepting_orders": True,
                }
            ],
            "next_cursor": "LTE=",
        }

    def get_order_book(self, token_id: str) -> OrderBookSummary:
        return OrderBookSummary(
            market="cond-1",
            asset_id=token_id,
            bids=[OrderSummary(price="0.51", size="100")],
            asks=[OrderSummary(price="0.53", size="120")],
            tick_size="0.01",
            last_trade_price="0.52",
            hash="h1",
        )

    def create_order(self, order_args):
        self.last_create_order = order_args
        return {"signed": True, "token_id": order_args.token_id}

    def post_order(self, signed_order, order_type):
        self.last_post_order = (signed_order, order_type)
        return {"success": True, "orderID": "oid-1"}

    def cancel(self, order_id: str):
        self.last_cancel = order_id
        return {"canceled": order_id}


def test_polymarket_adapter_parses_simplified_markets_and_orderbook() -> None:
    settings = load_settings()
    adapter = PolymarketExecutionAdapter(settings, client=FakeClobClient())

    markets = adapter.list_active_markets()
    assert markets[0]["market_id"] == "cond-1"
    assert markets[0]["yes_token_id"] == "yes-1"
    assert markets[0]["no_token_id"] == "no-1"

    quote = adapter.get_orderbook("yes-1")
    assert quote.market_id == "yes-1"
    assert quote.yes_price == 0.53
    assert quote.metadata["best_bid"] == 0.51


def test_polymarket_adapter_places_and_cancels_orders() -> None:
    settings = load_settings()
    client = FakeClobClient()
    adapter = PolymarketExecutionAdapter(settings, client=client)

    response = adapter.place_limit_order(
        OrderRequest(
            market_id="yes-1",
            side="YES",
            price=0.53,
            size=5.0,
            signal_t0=datetime.now(UTC),
        )
    )

    assert client.last_create_order.token_id == "yes-1"
    assert client.last_create_order.side == "BUY"
    assert response["response"]["success"] is True

    cancel = adapter.cancel_order("oid-1")
    assert cancel["canceled"] == "oid-1"
