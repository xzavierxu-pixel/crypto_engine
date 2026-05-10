from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BaselineConfig:
    artifact_dir: str
    settings_path: str = "config/settings.yaml"
    model_file: str | None = None
    calibrator_file: str | None = None
    manifest_file: str = "artifact_manifest.json"


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str = "paper"
    timezone: str = "UTC"
    audit_log: str = "artifacts/logs/execution_engine/live.jsonl"
    summary_dir: str = "artifacts/logs/execution_engine/summaries"
    idempotency_store_path: str = "artifacts/state/execution_engine/idempotency.json"


@dataclass(frozen=True)
class BinanceConfig:
    base_url: str = "https://api.binance.com"
    symbol: str = "BTCUSDT"
    one_minute_interval: str = "1m"
    one_second_interval: str = "1s"
    request_timeout_seconds: float = 5.0
    lookback_minutes: int = 360
    require_closed_kline: bool = True
    max_clock_skew_seconds: float = 2.0
    cache_path: str | None = "artifacts/state/execution_engine/binance_cache.parquet"


@dataclass(frozen=True)
class ScheduleConfig:
    interval_minutes: int = 5
    trigger_delay_seconds: int = 8
    max_data_wait_seconds: int = 20
    prewarm_seconds_before_trigger: int = 45


@dataclass(frozen=True)
class PolymarketConfig:
    host: str = "https://clob.polymarket.com"
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    chain_id: int = 137
    private_key_env: str = "POLYMARKET_PRIVATE_KEY"
    api_key_env: str = "CLOB_API_KEY"
    api_secret_env: str = "CLOB_SECRET"
    api_passphrase_env: str = "CLOB_PASS_PHRASE"
    signature_type: int | None = None
    funder: str | None = None
    timeout_seconds: float = 5.0
    max_pages: int = 3


@dataclass(frozen=True)
class OrderLegConfig:
    price_cap: float = 0.5
    offset: float = 0.0
    size: float = 5.0


@dataclass(frozen=True)
class OrdersConfig:
    enabled: bool = False
    mode: str = "paper"
    first: OrderLegConfig = field(default_factory=OrderLegConfig)
    second: OrderLegConfig = field(
        default_factory=lambda: OrderLegConfig(price_cap=0.5, offset=-0.1, size=5.0)
    )
    min_price: float = 0.01
    max_price: float = 0.99
    tick_size_default: float = 0.01
    on_invalid_second_order: str = "skip"


@dataclass(frozen=True)
class GuardsConfig:
    require_market_accepting_orders: bool = True
    require_best_bid: bool = True
    max_orders_per_window: int = 2
    enforce_idempotency: bool = True


@dataclass(frozen=True)
class ExecutionEngineConfig:
    baseline: BaselineConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    orders: OrdersConfig = field(default_factory=OrdersConfig)
    guards: GuardsConfig = field(default_factory=GuardsConfig)


def _payload_for(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, dict) else {}


def load_execution_config(path: str | Path) -> ExecutionEngineConfig:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    orders_payload = _payload_for(payload, "orders")
    first = OrderLegConfig(**_payload_for(orders_payload, "first"))
    second = OrderLegConfig(**_payload_for(orders_payload, "second"))
    orders = OrdersConfig(
        enabled=orders_payload.get("enabled", False),
        mode=orders_payload.get("mode", "paper"),
        first=first,
        second=second,
        min_price=orders_payload.get("min_price", 0.01),
        max_price=orders_payload.get("max_price", 0.99),
        tick_size_default=orders_payload.get("tick_size_default", 0.01),
        on_invalid_second_order=orders_payload.get("on_invalid_second_order", "skip"),
    )

    return ExecutionEngineConfig(
        baseline=BaselineConfig(**payload["baseline"]),
        runtime=RuntimeConfig(**_payload_for(payload, "runtime")),
        binance=BinanceConfig(**_payload_for(payload, "binance")),
        schedule=ScheduleConfig(**_payload_for(payload, "schedule")),
        polymarket=PolymarketConfig(**_payload_for(payload, "polymarket")),
        orders=orders,
        guards=GuardsConfig(**_payload_for(payload, "guards")),
    )

