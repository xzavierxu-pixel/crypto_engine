from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from execution_engine.config import OrderLegConfig, OrdersConfig, load_execution_config
from execution_engine.order_plan import build_two_limit_order_plan
from src.core.schemas import Decision, MarketQuote, Signal


def _signal() -> Signal:
    return Signal(
        asset="BTC/USDT",
        horizon="5m",
        t0=datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
        p_up=0.7,
        p_down=0.3,
        model_version="test",
        feature_version="test",
        decision_context={"t_up": 0.62, "t_down": 0.415},
    )


def _decision() -> Decision:
    return Decision(
        should_trade=True,
        side="YES",
        edge=None,
        reason="selective_binary_signal_passed",
        target_size=5.0,
    )


def _quote() -> MarketQuote:
    return MarketQuote(
        market_id="market",
        yes_price=0.6,
        metadata={
            "yes_token_id": "yes-token",
            "no_token_id": "no-token",
            "best_bid": 0.6,
            "best_ask": 0.62,
            "tick_size": 0.01,
        },
    )


def test_second_leg_is_disabled_by_default() -> None:
    result = build_two_limit_order_plan(_signal(), _decision(), _quote(), OrdersConfig())

    assert [order.metadata["leg"] for order in result.orders] == ["first"]
    assert {"leg": "second", "reason": "disabled_leg", "enabled": False} in result.skipped


def test_second_leg_can_be_enabled_explicitly() -> None:
    config = OrdersConfig(
        second=OrderLegConfig(
            enabled=True,
            price_cap=0.20,
            offset=0.0,
            size=5.0,
            reference_multiplier=0.25,
            round_decimals=2,
        )
    )

    result = build_two_limit_order_plan(_signal(), _decision(), _quote(), config)

    assert [order.metadata["leg"] for order in result.orders] == ["first", "second"]


def test_first_leg_min_best_bid_offset_and_cap_price_mode() -> None:
    config = OrdersConfig(
        first=OrderLegConfig(
            price_mode="min_best_bid_offset_and_cap",
            best_bid_offset=-0.05,
            price_cap=0.65,
            size=5.0,
        )
    )

    capped = build_two_limit_order_plan(_signal(), _decision(), _quote(), config)
    assert capped.orders[0].price == 0.55

    high_bid_quote = MarketQuote(
        market_id="market",
        yes_price=0.8,
        metadata={
            "yes_token_id": "yes-token",
            "no_token_id": "no-token",
            "best_bid": 0.8,
            "best_ask": 0.82,
            "tick_size": 0.01,
        },
    )
    capped = build_two_limit_order_plan(_signal(), _decision(), high_bid_quote, config)
    assert capped.orders[0].price == 0.65


def test_active_artifact_selects_configured_artifact_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "execution.yaml"
    config_path.write_text(
        """
baseline:
  active_artifact: split_validation
  artifact_dir: execution_engine/deploy/baseline
  artifacts:
    split_validation:
      artifact_dir: artifacts/data_v2/experiments/split
    full_train:
      artifact_dir: execution_engine/deploy/baseline
""",
        encoding="utf-8",
    )

    config = load_execution_config(config_path)

    assert config.baseline.active_artifact == "split_validation"
    assert config.baseline.artifact_dir == "artifacts/data_v2/experiments/split"
