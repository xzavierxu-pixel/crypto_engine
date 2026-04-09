from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.schemas import RiskState


class PositionSizer(ABC):
    name: str

    @abstractmethod
    def size(self, edge: float, risk_state: RiskState | None = None) -> float:
        raise NotImplementedError
