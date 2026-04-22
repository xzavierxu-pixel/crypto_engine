from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SampleKey:
    asset: str
    horizon: str
    t0: datetime
    grid_id: str


@dataclass(frozen=True)
class Signal:
    asset: str
    horizon: str
    t0: datetime
    p_up: float
    model_version: str
    feature_version: str
    p_active: float | None = None
    decision_context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    should_trade: bool
    side: str | None
    edge: float | None
    reason: str
    target_size: float


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskState:
    current_exposure: float = 0.0
    max_total_exposure: float = 0.0
    active_positions: int = 0


@dataclass(frozen=True)
class MarketQuote:
    market_id: str
    yes_price: float
    no_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderRequest:
    market_id: str
    side: str
    price: float
    size: float
    signal_t0: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
