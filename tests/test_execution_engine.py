from __future__ import annotations

import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import execution_engine.run_once as run_once_module
from execution_engine.config import load_execution_config
from execution_engine.order_plan import build_two_limit_order_plan
from execution_engine.polymarket_v2 import PolymarketV2Adapter, normalize_gamma_market
from execution_engine.realtime_data import normalize_binance_agg_trades, normalize_binance_klines
from execution_engine.run_once import build_btc_5m_slug, build_idempotency_key
from src.core.schemas import Decision, MarketQuote, Signal


def _signal(p_up: float = 0.60) -> Signal:
    return Signal(
        asset="BTC/USDT",
        horizon="5m",
        t0=datetime(2026, 5, 10, 12, 35, tzinfo=UTC),
        p_up=p_up,
        p_down=1.0 - p_up,
        model_version="m",
        feature_version="v",
        decision_context={"t_up": 0.585, "t_down": 0.335},
    )


def test_execution_config_example_loads() -> None:
    config = load_execution_config("execution_engine/config.example.yaml")

    assert config.runtime.mode == "paper"
    assert config.orders.enabled is False
    assert config.orders.first.size == 5.0
    assert config.orders.second.offset == -0.1
    assert config.baseline.artifact_dir == "execution_engine/deploy/baseline"


def test_normalize_binance_klines_outputs_shared_schema() -> None:
    frame = normalize_binance_klines(
        [
            [
                1778400000000,
                "100.0",
                "101.0",
                "99.0",
                "100.5",
                "2.0",
                1778400059999,
                "201.0",
                3,
                "1.2",
                "120.6",
                "0",
            ]
        ]
    )

    assert frame["timestamp"].iloc[0].tzinfo is not None
    assert frame["open"].iloc[0] == 100.0
    assert frame["quote_volume"].iloc[0] == 201.0
    assert frame["trade_count"].iloc[0] == 3
    assert frame["taker_buy_base_volume"].iloc[0] == 1.2


def test_normalize_binance_agg_trades_outputs_trade_schema() -> None:
    frame = normalize_binance_agg_trades(
        [
            {
                "a": 1,
                "p": "100.0",
                "q": "0.2",
                "f": 10,
                "l": 11,
                "T": 1778400000123,
                "m": False,
                "M": True,
            }
        ]
    )

    assert frame["timestamp"].iloc[0].tzinfo is not None
    assert frame["price"].iloc[0] == 100.0
    assert frame["quantity"].iloc[0] == 0.2
    assert frame["quote_quantity"].iloc[0] == 20.0
    assert bool(frame["is_buyer_maker"].iloc[0]) is False


def test_two_limit_order_plan_uses_target_token_best_bid_cap_and_offset() -> None:
    config = load_execution_config("execution_engine/config.example.yaml")
    plan = build_two_limit_order_plan(
        _signal(),
        Decision(True, "YES", 0.1, "selective_binary_signal_passed", 5.0),
        MarketQuote(
            market_id="yes-token",
            yes_price=0.51,
            no_price=0.49,
            metadata={"yes_token_id": "yes-token", "no_token_id": "no-token", "best_bid": 0.57, "tick_size": "0.01"},
        ),
        config.orders,
    )

    assert [order.market_id for order in plan.orders] == ["yes-token", "yes-token"]
    assert [order.price for order in plan.orders] == [0.5, 0.4]
    assert [order.size for order in plan.orders] == [5.0, 5.0]
    assert plan.skipped == []


def test_two_limit_order_plan_skips_without_best_bid() -> None:
    config = load_execution_config("execution_engine/config.example.yaml")
    plan = build_two_limit_order_plan(
        _signal(),
        Decision(True, "NO", 0.1, "selective_binary_signal_passed", 5.0),
        MarketQuote(market_id="no-token", yes_price=0.51, metadata={"no_token_id": "no-token"}),
        config.orders,
    )

    assert plan.orders == []
    assert plan.skipped == [{"reason": "missing_best_bid"}]


def test_slug_and_idempotency_key_are_window_scoped() -> None:
    slug, start, end = build_btc_5m_slug(datetime(2026, 5, 10, 12, 37, 59, tzinfo=UTC))
    key = build_idempotency_key(start, "token-1", "YES")

    assert slug == f"btc-updown-5m-{int(start.timestamp())}"
    assert start == datetime(2026, 5, 10, 12, 35, tzinfo=UTC)
    assert end == datetime(2026, 5, 10, 12, 40, tzinfo=UTC)
    assert key == "2026-05-10T12:35:00+00:00:token-1:YES:two_limit_plan"


def test_gamma_market_normalization_extracts_tokens() -> None:
    market = normalize_gamma_market(
        {
            "id": "m1",
            "active": True,
            "closed": False,
            "acceptingOrders": True,
            "outcomes": '["Up","Down"]',
            "outcomePrices": '["0.52","0.48"]',
            "clobTokenIds": '["yes-token","no-token"]',
        },
        slug="btc-updown-5m-1",
    )

    assert market.yes_token_id == "yes-token"
    assert market.no_token_id == "no-token"
    assert market.accepting_orders is True


def test_polymarket_v2_adapter_places_gtc_buy(monkeypatch) -> None:
    clob_module = types.ModuleType("py_clob_client_v2")

    class OrderArgs:
        def __init__(self, token_id, price, size, side):
            self.token_id = token_id
            self.price = price
            self.size = size
            self.side = side

    class OrderType:
        GTC = "GTC"

    class PartialCreateOrderOptions:
        def __init__(self, tick_size):
            self.tick_size = tick_size

    class Side:
        BUY = "BUY"

    clob_module.OrderArgs = OrderArgs
    clob_module.OrderType = OrderType
    clob_module.PartialCreateOrderOptions = PartialCreateOrderOptions
    clob_module.Side = Side
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", clob_module)

    class FakeClient:
        creds = object()

        def __init__(self) -> None:
            self.last_order = None
            self.last_type = None

        def create_and_post_order(self, order_args, options, order_type):
            self.last_order = order_args
            self.last_options = options
            self.last_type = order_type
            return {"success": True, "orderID": "1"}

    config = load_execution_config("execution_engine/config.example.yaml")
    client = FakeClient()
    adapter = PolymarketV2Adapter(config.polymarket, client=client)
    response = adapter.place_limit_order(
        build_two_limit_order_plan(
            _signal(),
            Decision(True, "YES", 0.1, "selective_binary_signal_passed", 5.0),
            MarketQuote("yes-token", 0.51, metadata={"best_bid": 0.50}),
            config.orders,
        ).orders[0]
    )

    assert client.last_order.token_id == "yes-token"
    assert client.last_order.side == "BUY"
    assert client.last_options.tick_size == "0.01"
    assert client.last_type == "GTC"
    assert response["response"]["success"] is True


def test_run_once_paper_flow_builds_order_plan_without_submit(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
baseline:
  artifact_dir: baseline-dir
runtime:
  mode: paper
  audit_log: {tmp_path.as_posix()}/audit.jsonl
  summary_dir: {tmp_path.as_posix()}/summaries
  idempotency_store_path: {tmp_path.as_posix()}/idempotency.json
orders:
  enabled: false
  mode: paper
  first:
    price_cap: 0.5
    offset: 0.0
    size: 5.0
  second:
    price_cap: 0.5
    offset: -0.1
    size: 5.0
""",
        encoding="utf-8",
    )

    class FakeBaseline:
        artifact_dir = Path("baseline-dir")
        model_plugin = "catboost_lgbm_logit_blend"
        calibration_plugin = "platt_logit"
        feature_columns = ["f1"]
        t_up = 0.585
        t_down = 0.335

    class FakeBinance:
        def __init__(self, config):
            self.config = config

        def wait_for_closed_runtime_frames(self, max_wait_seconds):
            frame = pd.DataFrame({"timestamp": [pd.Timestamp("2026-05-10T12:35:00Z")]})
            return frame, frame, frame

    class FakeInference:
        def __init__(self, settings, baseline):
            self.settings = settings
            self.baseline = baseline

        def predict(self, minute_frame, second_frame, agg_trades_frame=None):
            return types.SimpleNamespace(
                signal=_signal(0.60),
                feature_frame=pd.DataFrame({"f1": [1.0]}),
                second_level_frame=pd.DataFrame({"timestamp": [pd.Timestamp("2026-05-10T12:35:00Z")]}),
            )

    class FakePolymarket:
        def __init__(self, config):
            self.config = config
            self.submitted = []

        def get_market_by_slug(self, slug):
            return types.SimpleNamespace(
                slug=slug,
                market_id="market-1",
                yes_token_id="yes-token",
                no_token_id="no-token",
                active=True,
                closed=False,
                accepting_orders=True,
                metadata={},
            )

        def get_orderbook(self, token_id, metadata=None):
            return MarketQuote(
                market_id=token_id,
                yes_price=0.51,
                metadata={**(metadata or {}), "best_bid": 0.57, "tick_size": "0.01"},
            )

        def place_limit_order(self, order):
            self.submitted.append(order)
            return {"success": True}

    monkeypatch.setattr(run_once_module, "load_baseline_artifact", lambda config: FakeBaseline())
    monkeypatch.setattr(run_once_module, "load_settings", lambda path: object())
    monkeypatch.setattr(run_once_module, "BinanceRealtimeClient", FakeBinance)
    monkeypatch.setattr(run_once_module, "RuntimeInferenceEngine", FakeInference)
    monkeypatch.setattr(run_once_module, "PolymarketV2Adapter", FakePolymarket)
    monkeypatch.setattr(
        run_once_module,
        "evaluate_selective_binary_signal",
        lambda signal, settings: Decision(True, "YES", 0.1, "selective_binary_signal_passed", 5.0),
    )

    summary = run_once_module.run_once(str(config_path), mode_override="paper")

    assert summary["submitted"] is False
    assert summary["market"]["target_token_id"] == "yes-token"
    assert [order["price"] for order in summary["orders"]] == [0.5, 0.4]
    assert summary["skipped"] == [{"reason": "paper_mode_or_orders_disabled"}]
    assert Path(summary["summary_path"]).exists()
