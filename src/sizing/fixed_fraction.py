from __future__ import annotations

from src.core.schemas import RiskState
from src.sizing.base import PositionSizer


class FixedFractionSizer(PositionSizer):
    name = "fixed_fraction"

    def __init__(self, single_position_cap: float, max_total_exposure: float | None = None) -> None:
        self.single_position_cap = single_position_cap
        self.max_total_exposure = max_total_exposure

    def size(self, edge: float, risk_state: RiskState | None = None) -> float:
        if edge <= 0:
            return 0.0

        size = self.single_position_cap
        if risk_state is None or self.max_total_exposure is None:
            return size

        remaining = max(0.0, self.max_total_exposure - risk_state.current_exposure)
        return min(size, remaining)
