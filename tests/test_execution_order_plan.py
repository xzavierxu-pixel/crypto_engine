from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from execution_engine.config import ExecutionEdgeConfig, OrderLegConfig, OrdersConfig, load_execution_config
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


def _signal_with_q80(price: float, *, offset: float = 0.01) -> Signal:
    signal = _signal()
    return Signal(
        asset=signal.asset,
        horizon=signal.horizon,
        t0=signal.t0,
        p_up=signal.p_up,
        p_down=signal.p_down,
        model_version=signal.model_version,
        feature_version=signal.feature_version,
        decision_context={
            **signal.decision_context,
            "price_estimator_q80_rounded": price,
            "price_estimator_best_ask_offset": offset,
            "price_estimator_fallback_price_mode": "limit_config_best_ask_offset",
        },
    )


def _signal_with_safe_gap(price: float, *, offset: float = 0.01) -> Signal:
    signal = _signal()
    return Signal(
        asset=signal.asset,
        horizon=signal.horizon,
        t0=signal.t0,
        p_up=signal.p_up,
        p_down=signal.p_down,
        model_version=signal.model_version,
        feature_version=signal.feature_version,
        decision_context={
            **signal.decision_context,
            "price_estimator_safe_gap_rounded": price,
            "price_estimator_safe_gap_action": "active",
            "price_estimator_safe_gap_conf_ok": True,
            "price_estimator_safe_gap_f_model": 0.531,
            "price_estimator_best_ask_offset": offset,
            "price_estimator_fallback_price_mode": "limit_config_best_ask_offset",
        },
    )


def _signal_with_expected_return(bid: float, *, eligible: bool = True) -> Signal:
    signal = _signal()
    return Signal(
        asset=signal.asset,
        horizon=signal.horizon,
        t0=signal.t0,
        p_up=signal.p_up,
        p_down=signal.p_down,
        model_version=signal.model_version,
        feature_version=signal.feature_version,
        decision_context={
            **signal.decision_context,
            "price_estimator_expected_return_bid": bid,
            "price_estimator_expected_return_ev": 0.025,
            "price_estimator_expected_return_fill_probability": 0.81,
            "price_estimator_expected_return_eligible": eligible,
        },
    )


def _decision() -> Decision:
    return Decision(
        should_trade=True,
        side="YES",
        edge=None,
        reason="selective_binary_signal_passed",
        target_size=5.0,
    )


def _down_decision() -> Decision:
    return Decision(
        should_trade=True,
        side="NO",
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


def test_best_ask_market_mode_sets_fak_metadata_and_passes_edge() -> None:
    config = OrdersConfig(
        first=OrderLegConfig(
            price_mode="best_ask_market",
            order_type="FAK",
            price_cap=0.99,
            size=5.0,
        )
    )

    result = build_two_limit_order_plan(
        _signal(),
        _decision(),
        _quote(),
        config,
        ExecutionEdgeConfig(enabled=True, min_edge=0.01, max_buy_price=0.99, max_spread=None),
    )

    assert len(result.orders) == 1
    assert result.orders[0].price == 0.62
    assert result.orders[0].metadata["quote_source"] == "best_ask_market"
    assert result.orders[0].metadata["order_type"] == "FAK"


def test_best_ask_market_mode_rejects_edge_at_threshold_boundary() -> None:
    signal = Signal(
        asset="BTC/USDT",
        horizon="5m",
        t0=datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
        p_up=0.75,
        p_down=0.25,
        model_version="test",
        feature_version="test",
        decision_context={"t_up": 0.62, "t_down": 0.415},
    )
    config = OrdersConfig(
        first=OrderLegConfig(
            price_mode="best_ask_market",
            order_type="FAK",
            price_cap=0.99,
            size=5.0,
        )
    )

    result = build_two_limit_order_plan(
        signal,
        _decision(),
        MarketQuote(
            market_id="market",
            yes_price=0.5,
            metadata={
                "yes_token_id": "yes-token",
                "no_token_id": "no-token",
                "best_bid": 0.49,
                "best_ask": 0.5,
                "tick_size": 0.01,
            },
        ),
        config,
        ExecutionEdgeConfig(enabled=True, min_edge=0.25, max_buy_price=0.99, max_spread=None),
    )

    assert result.orders == []
    assert result.skipped[0]["reason"] == "edge_below_minimum"


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


def test_limit_config_best_ask_offset_ceil_lookup_for_up_signal() -> None:
    config = OrdersConfig(
        first=OrderLegConfig(
            price_mode="limit_config_best_ask_offset",
            price_cap=0.99,
            size=5.0,
        )
    )
    quote = MarketQuote(
        market_id="market",
        yes_price=0.801,
        metadata={
            "yes_token_id": "yes-token",
            "no_token_id": "no-token",
            "best_bid": 0.78,
            "best_ask": 0.801,
            "tick_size": 0.01,
        },
    )

    result = build_two_limit_order_plan(_signal(), _decision(), quote, config)

    assert len(result.orders) == 1
    assert result.orders[0].market_id == "yes-token"
    assert result.orders[0].price == 0.8
    assert result.orders[0].metadata["quote_source"] == "best_ask"
    assert result.orders[0].metadata["limit_config_lookup_price"] == 0.81
    assert result.orders[0].metadata["limit_config_offset"] == 0.01
    assert result.skipped == [{"leg": "second", "reason": "disabled_leg", "enabled": False}]


def test_limit_config_best_ask_offset_uses_down_limit_for_no_signal() -> None:
    config = OrdersConfig(
        first=OrderLegConfig(
            price_mode="limit_config_best_ask_offset",
            price_cap=0.99,
            size=5.0,
        )
    )
    quote = MarketQuote(
        market_id="market",
        yes_price=0.351,
        metadata={
            "yes_token_id": "yes-token",
            "no_token_id": "no-token",
            "best_bid": 0.27,
            "best_ask": 0.351,
            "tick_size": 0.01,
        },
    )

    result = build_two_limit_order_plan(_signal(), _down_decision(), quote, config)

    assert len(result.orders) == 1
    assert result.orders[0].market_id == "no-token"
    assert result.orders[0].price == 0.34
    assert result.orders[0].metadata["limit_config_lookup_price"] == 0.36
    assert result.orders[0].metadata["limit_config_offset"] == 0.02


def test_limit_config_best_ask_offset_skips_missing_key_and_invalid_price(caplog) -> None:
    missing_key_config = OrdersConfig(
        first=OrderLegConfig(
            price_mode="limit_config_best_ask_offset",
            price_cap=0.99,
            size=5.0,
        )
    )
    missing_key_quote = MarketQuote(
        market_id="market",
        yes_price=0.961,
        metadata={
            "yes_token_id": "yes-token",
            "best_ask": 0.961,
            "tick_size": 0.01,
        },
    )

    result = build_two_limit_order_plan(_signal(), _decision(), missing_key_quote, missing_key_config)

    assert result.orders == []
    assert result.skipped[0]["reason"] == "missing_limit_offset"
    assert result.skipped[0]["lookup_price"] == 0.97
    assert "no limit offset is configured" in caplog.text

    invalid_price_config = OrdersConfig(
        first=OrderLegConfig(
            price_mode="limit_config_best_ask_offset",
            price_cap=0.99,
            size=5.0,
        ),
        max_price=0.90,
    )
    invalid_price_quote = MarketQuote(
        market_id="market",
        yes_price=0.949,
        metadata={
            "yes_token_id": "yes-token",
            "best_ask": 0.949,
            "tick_size": 0.01,
        },
    )

    result = build_two_limit_order_plan(_signal(), _decision(), invalid_price_quote, invalid_price_config)

    assert result.orders == []
    assert result.skipped[0]["reason"] == "invalid_limit_config_price"
    assert result.skipped[0]["raw_price"] == 0.94


def test_q80_best_ask_offset_mode_uses_lower_q80_price_without_price_cap() -> None:
    config = OrdersConfig(
        first=OrderLegConfig(
            price_mode="min_q80_final_price_and_best_ask_offset",
            price_cap=0.40,
            size=5.0,
        )
    )

    result = build_two_limit_order_plan(_signal_with_q80(0.54), _decision(), _quote(), config)

    assert len(result.orders) == 1
    assert result.orders[0].price == 0.54
    assert result.orders[0].metadata["quote_source"] == "price_estimator_q80_best_ask"
    assert result.orders[0].metadata["price_estimator_q80_rounded"] == 0.54
    assert result.orders[0].metadata["price_estimator_best_ask_offset_price"] == 0.61


def test_q80_best_ask_offset_mode_uses_best_ask_offset_when_lower() -> None:
    config = OrdersConfig(
        first=OrderLegConfig(
            price_mode="min_q80_final_price_and_best_ask_offset",
            price_cap=0.40,
            size=5.0,
        )
    )

    result = build_two_limit_order_plan(_signal_with_q80(0.90), _decision(), _quote(), config)

    assert len(result.orders) == 1
    assert result.orders[0].price == 0.61
    assert result.orders[0].price > config.first.price_cap


def test_q80_best_ask_offset_mode_falls_back_to_limit_config() -> None:
    config = OrdersConfig(
        first=OrderLegConfig(
            price_mode="min_q80_final_price_and_best_ask_offset",
            price_cap=0.40,
            size=5.0,
        )
    )
    quote = MarketQuote(
        market_id="market",
        yes_price=0.801,
        metadata={
            "yes_token_id": "yes-token",
            "no_token_id": "no-token",
            "best_bid": 0.78,
            "best_ask": 0.801,
            "tick_size": 0.01,
        },
    )

    result = build_two_limit_order_plan(_signal(), _decision(), quote, config)

    assert len(result.orders) == 1
    assert result.orders[0].price == 0.8
    assert result.orders[0].metadata["quote_source"] == "best_ask"
    assert result.orders[0].metadata["price_estimator_fallback_price_mode"] == "limit_config_best_ask_offset"
    assert result.orders[0].metadata["limit_config_lookup_price"] == 0.81


def test_safe_gap_best_ask_offset_mode_uses_lower_safe_gap_price_without_price_cap() -> None:
    config = OrdersConfig(
        first=OrderLegConfig(
            price_mode="min_safe_gap_price_and_best_ask_offset",
            price_cap=0.40,
            size=5.0,
        )
    )

    result = build_two_limit_order_plan(_signal_with_safe_gap(0.54), _decision(), _quote(), config)

    assert len(result.orders) == 1
    assert result.orders[0].price == 0.54
    assert result.orders[0].metadata["quote_source"] == "price_estimator_safe_gap_best_ask"
    assert result.orders[0].metadata["price_estimator_safe_gap_rounded"] == 0.54
    assert result.orders[0].metadata["price_estimator_safe_gap_action"] == "active"
    assert result.orders[0].metadata["price_estimator_safe_gap_conf_ok"] is True
    assert result.orders[0].metadata["price_estimator_safe_gap_f_model"] == 0.531
    assert result.orders[0].metadata["price_estimator_best_ask_offset_price"] == 0.61


def test_safe_gap_best_ask_offset_mode_uses_best_ask_offset_when_lower() -> None:
    config = OrdersConfig(
        first=OrderLegConfig(
            price_mode="min_safe_gap_price_and_best_ask_offset",
            price_cap=0.40,
            size=5.0,
        )
    )

    result = build_two_limit_order_plan(_signal_with_safe_gap(0.90), _decision(), _quote(), config)

    assert len(result.orders) == 1
    assert result.orders[0].price == 0.61
    assert result.orders[0].price > config.first.price_cap


def test_expected_return_mode_uses_minimum_of_best_ask_minus_tick_and_optimal_bid() -> None:
    config = OrdersConfig(first=OrderLegConfig(price_mode="expected_return_optimal_bid", size=5.0))

    result = build_two_limit_order_plan(_signal_with_expected_return(0.54), _decision(), _quote(), config)

    assert len(result.orders) == 1
    assert result.orders[0].price == 0.54
    assert result.orders[0].metadata["quote_source"] == "min_best_ask_minus_0p01_and_expected_return_optimal_bid"
    assert result.orders[0].metadata["price_estimator_expected_return_fill_probability"] == 0.81


def test_expected_return_mode_caps_optimal_bid_at_best_ask_minus_tick() -> None:
    config = OrdersConfig(first=OrderLegConfig(price_mode="expected_return_optimal_bid", size=5.0))

    result = build_two_limit_order_plan(_signal_with_expected_return(0.90), _decision(), _quote(), config)

    assert len(result.orders) == 1
    assert result.orders[0].price == 0.61


def test_expected_return_mode_skips_ineligible_candidate() -> None:
    config = OrdersConfig(first=OrderLegConfig(price_mode="expected_return_optimal_bid", size=5.0))

    result = build_two_limit_order_plan(
        _signal_with_expected_return(0.54, eligible=False), _decision(), _quote(), config
    )

    assert result.orders == []
    assert result.skipped[0]["reason"] == "expected_return_policy_rejected"


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
