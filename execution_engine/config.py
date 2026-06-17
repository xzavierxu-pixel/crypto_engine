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
    active_artifact: str | None = None
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class PriceEstimatorConfig:
    enabled: bool = False
    active_artifact: str | None = None
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifact_dir: str | None = None
    manifest_file: str = "artifact_manifest.json"
    model_file: str | None = None
    prediction_column: str = "pred_q80"
    selected_side_column: str = "selected_side"
    yes_value: str = "YES"
    no_value: str = "NO"
    round_decimals: int = 2
    best_ask_offset: float = 0.01
    fallback_price_mode: str = "limit_config_best_ask_offset"


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
    require_agg_trade_through_last_second: bool = True
    max_agg_trade_lag_seconds: float = 2.0
    agg_trade_wait_seconds: float = 8.0


@dataclass(frozen=True)
class ScheduleConfig:
    interval_minutes: int = 5
    trigger_delay_seconds: int = 68
    max_data_wait_seconds: int = 20
    prewarm_seconds_before_trigger: int = 45


@dataclass(frozen=True)
class ThresholdConfig:
    t_up: float | None = None
    t_down: float | None = None


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
    enabled: bool = True
    price_mode: str = "reference_multiplier_offset_and_cap"
    price_cap: float = 0.75
    offset: float = 0.01
    best_bid_offset: float | None = None
    size: float = 5.0
    reference_multiplier: float = 1.0
    round_decimals: int | None = None


@dataclass(frozen=True)
class OrdersConfig:
    enabled: bool = False
    mode: str = "paper"
    first: OrderLegConfig = field(default_factory=OrderLegConfig)
    second: OrderLegConfig = field(
        default_factory=lambda: OrderLegConfig(
            enabled=False,
            price_cap=0.20,
            offset=0.0,
            size=5.0,
            reference_multiplier=0.25,
            round_decimals=2,
        )
    )
    min_price: float = 0.10
    max_price: float = 0.99
    tick_size_default: float = 0.01
    on_invalid_second_order: str = "skip"
    cancel_unfilled_after_seconds: float = 0.0


@dataclass(frozen=True)
class ExecutionEdgeConfig:
    enabled: bool = False
    min_edge: float = 0.04
    max_buy_price: float | None = 0.8
    max_spread: float | None = 0.08
    max_order_notional: float | None = None
    size_to_max_notional: bool = False
    apply_to_first_leg: bool = True
    apply_to_second_leg: bool = True


@dataclass(frozen=True)
class GuardsConfig:
    require_market_accepting_orders: bool = True
    require_best_bid: bool = True
    max_orders_per_window: int = 2
    enforce_idempotency: bool = True


@dataclass(frozen=True)
class PaperTestConfig:
    max_duration_minutes: int = 60
    stop_when_pnl_gt: float = 25.0
    report_to_user: bool = True


@dataclass(frozen=True)
class ExecutionEngineConfig:
    baseline: BaselineConfig
    price_estimator: PriceEstimatorConfig = field(default_factory=PriceEstimatorConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    orders: OrdersConfig = field(default_factory=OrdersConfig)
    execution_edge: ExecutionEdgeConfig = field(default_factory=ExecutionEdgeConfig)
    guards: GuardsConfig = field(default_factory=GuardsConfig)
    paper_test: PaperTestConfig = field(default_factory=PaperTestConfig)


def _payload_for(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, dict) else {}


def _order_leg_config(payload: dict[str, Any], *, enabled_default: bool) -> OrderLegConfig:
    leg_payload = dict(payload)
    leg_payload.setdefault("enabled", enabled_default)
    return OrderLegConfig(**leg_payload)


def _baseline_config(payload: dict[str, Any]) -> BaselineConfig:
    baseline_payload = dict(payload)
    active_artifact = baseline_payload.get("active_artifact")
    artifacts = baseline_payload.get("artifacts") or {}
    if active_artifact is not None:
        if not isinstance(artifacts, dict) or active_artifact not in artifacts:
            raise ValueError(f"baseline.active_artifact '{active_artifact}' is not defined in baseline.artifacts.")
        selected = artifacts[active_artifact]
        if not isinstance(selected, dict):
            raise ValueError(f"baseline.artifacts.{active_artifact} must be a mapping.")
        selected_dir = selected.get("artifact_dir")
        if not selected_dir:
            raise ValueError(f"baseline.artifacts.{active_artifact}.artifact_dir is required.")
        baseline_payload["artifact_dir"] = selected_dir
        for key in ("model_file", "calibrator_file", "manifest_file", "settings_path"):
            if selected.get(key) is not None:
                baseline_payload[key] = selected[key]
    return BaselineConfig(**baseline_payload)


def _price_estimator_config(payload: dict[str, Any]) -> PriceEstimatorConfig:
    estimator_payload = dict(payload)
    active_artifact = estimator_payload.get("active_artifact")
    artifacts = estimator_payload.get("artifacts") or {}
    if active_artifact is not None:
        if not isinstance(artifacts, dict) or active_artifact not in artifacts:
            raise ValueError(
                f"price_estimator.active_artifact '{active_artifact}' is not defined in price_estimator.artifacts."
            )
        selected = artifacts[active_artifact]
        if not isinstance(selected, dict):
            raise ValueError(f"price_estimator.artifacts.{active_artifact} must be a mapping.")
        selected_dir = selected.get("artifact_dir")
        if not selected_dir:
            raise ValueError(f"price_estimator.artifacts.{active_artifact}.artifact_dir is required.")
        estimator_payload["artifact_dir"] = selected_dir
        for key in (
            "manifest_file",
            "model_file",
            "prediction_column",
            "selected_side_column",
            "yes_value",
            "no_value",
            "round_decimals",
            "best_ask_offset",
            "fallback_price_mode",
        ):
            if selected.get(key) is not None:
                estimator_payload[key] = selected[key]
    return PriceEstimatorConfig(**estimator_payload)


def load_execution_config(path: str | Path) -> ExecutionEngineConfig:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    orders_payload = _payload_for(payload, "orders")
    first = _order_leg_config(_payload_for(orders_payload, "first"), enabled_default=True)
    second = _order_leg_config(_payload_for(orders_payload, "second"), enabled_default=False)
    orders = OrdersConfig(
        enabled=orders_payload.get("enabled", False),
        mode=orders_payload.get("mode", "paper"),
        first=first,
        second=second,
        min_price=orders_payload.get("min_price", 0.01),
        max_price=orders_payload.get("max_price", 0.99),
        tick_size_default=orders_payload.get("tick_size_default", 0.01),
        on_invalid_second_order=orders_payload.get("on_invalid_second_order", "skip"),
        cancel_unfilled_after_seconds=orders_payload.get("cancel_unfilled_after_seconds", 0.0),
    )

    execution_edge_payload = _payload_for(payload, "execution_edge")
    if "max_order_notional" not in execution_edge_payload:
        execution_edge_payload = {**execution_edge_payload, "max_order_notional": orders.first.size}

    return ExecutionEngineConfig(
        baseline=_baseline_config(payload["baseline"]),
        price_estimator=_price_estimator_config(_payload_for(payload, "price_estimator")),
        runtime=RuntimeConfig(**_payload_for(payload, "runtime")),
        binance=BinanceConfig(**_payload_for(payload, "binance")),
        schedule=ScheduleConfig(**_payload_for(payload, "schedule")),
        thresholds=ThresholdConfig(**_payload_for(payload, "thresholds")),
        polymarket=PolymarketConfig(**_payload_for(payload, "polymarket")),
        orders=orders,
        execution_edge=ExecutionEdgeConfig(**execution_edge_payload),
        guards=GuardsConfig(**_payload_for(payload, "guards")),
        paper_test=PaperTestConfig(**_payload_for(payload, "paper_test")),
    )
