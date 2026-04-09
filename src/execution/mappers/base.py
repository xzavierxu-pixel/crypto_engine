from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.schemas import Signal


class MarketMapper(ABC):
    name: str

    @abstractmethod
    def map_signal(self, signal: Signal) -> dict:
        raise NotImplementedError
