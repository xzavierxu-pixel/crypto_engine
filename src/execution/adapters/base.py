from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.schemas import MarketQuote, OrderRequest


class ExecutionAdapter(ABC):
    name: str

    @abstractmethod
    def list_active_markets(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def get_orderbook(self, market_id: str) -> MarketQuote:
        raise NotImplementedError

    @abstractmethod
    def place_limit_order(self, order: OrderRequest) -> dict:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict:
        raise NotImplementedError
