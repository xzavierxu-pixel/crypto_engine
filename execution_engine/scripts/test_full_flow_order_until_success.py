from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution_engine.artifacts import load_baseline_artifact
from execution_engine.config import ExecutionEngineConfig, load_execution_config
from execution_engine.feature_runtime import RuntimeInferenceEngine
from execution_engine.order_plan import build_two_limit_order_plan
from execution_engine.polymarket_v2 import PolymarketV2Adapter
from execution_engine.prewarm import prewarm
from execution_engine.realtime_data import BinanceRealtimeClient
from execution_engine.run_once import build_btc_5m_slug, build_idempotency_key, run_once
from execution_engine.scripts.test_min_price_order import load_env_file
from src.core.config import load_settings
from src.core.schemas import Decision
from src.execution.idempotency import IdempotencyStore


def _is_service_not_ready(exc: Exception) -> bool:
    return "status_code=425" in str(exc) or "service not ready" in str(exc).lower()


def _forced_side(summary: dict[str, Any], fallback: str) -> str:
    signal = summary.get("signal") or {}
    p_up = signal.get("p_up")
    if p_up is None:
        return fallback
    return "YES" if float(p_up) >= 0.5 else "NO"


def submit_forced_order_from_runtime(
    config: ExecutionEngineConfig,
    side: str,
    idempotency_suffix: str,
) -> dict[str, Any]:
    baseline = load_baseline_artifact(config.baseline)
    settings = load_settings(config.baseline.settings_path)
    binance = BinanceRealtimeClient(config.binance)
    minute_frame, second_frame, agg_trades_frame = binance.wait_for_closed_runtime_frames(
        max_wait_seconds=config.schedule.max_data_wait_seconds,
    )
    inference = RuntimeInferenceEngine(
        settings,
        baseline,
        t_up=config.thresholds.t_up,
        t_down=config.thresholds.t_down,
    )
    slug, window_start, window_end = build_btc_5m_slug(datetime.now(UTC), offset_windows=0)
    result = inference.predict(
        minute_frame,
        second_frame,
        agg_trades_frame,
        signal_t0=pd.Timestamp(window_start),
        use_latest_available_before_signal=True,
    )
    signal = result.signal
    decision = Decision(
        should_trade=True,
        side=side,
        edge=None,
        reason="forced_full_flow_order_test",
        target_size=float(config.orders.first.size),
    )

    polymarket = PolymarketV2Adapter(config.polymarket)
    market = polymarket.get_market_by_slug(slug)
    if market is None:
        raise RuntimeError(f"Market not found for slug: {slug}")
    if config.guards.require_market_accepting_orders and (
        not market.active or market.closed or not market.accepting_orders
    ):
        raise RuntimeError(
            "Market is not accepting orders: "
            f"slug={slug} active={market.active} closed={market.closed} "
            f"accepting_orders={market.accepting_orders}"
        )

    token_id = market.yes_token_id if side == "YES" else market.no_token_id
    quote = polymarket.get_orderbook(
        token_id,
        metadata={
            **market.metadata,
            "slug": market.slug,
            "market_id": market.market_id,
            "yes_token_id": market.yes_token_id,
            "no_token_id": market.no_token_id,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
        },
    )
    if config.guards.require_best_bid and quote.metadata.get("best_bid") is None and quote.metadata.get("best_ask") is None:
        raise RuntimeError("missing_quote")

    min_price_orders = replace(
        config.orders,
        first=replace(config.orders.first, price_cap=float(config.orders.min_price), offset=0.0),
        second=replace(config.orders.second, price_cap=0.0, offset=0.0),
        on_invalid_second_order="skip",
    )
    plan = build_two_limit_order_plan(signal, decision, quote, min_price_orders)
    if not plan.orders:
        raise RuntimeError(f"Order plan produced no orders: {plan.skipped}")

    order = plan.orders[0]
    store = IdempotencyStore(config.runtime.idempotency_store_path)
    key = f"{build_idempotency_key(window_start, order.market_id, order.side)}:{idempotency_suffix}"
    if config.guards.enforce_idempotency and store.has(key):
        raise RuntimeError(f"Idempotency key already seen: {key}")

    response = polymarket.place_limit_order(order)
    if config.guards.enforce_idempotency:
        store.record(key, {"market_id": order.market_id, "side": order.side, "price": order.price})

    return {
        "submitted": True,
        "success_type": "forced_order_submitted",
        "signal": {
            "t0": signal.t0.isoformat(),
            "p_up": signal.p_up,
            "p_down": signal.p_down,
            "t_up": signal.decision_context.get("t_up"),
            "t_down": signal.decision_context.get("t_down"),
        },
        "market": {
            "slug": market.slug,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "token_side": order.side,
            "best_bid": quote.metadata.get("best_bid"),
            "best_ask": quote.metadata.get("best_ask"),
        },
        "order": {
            "side": order.side,
            "price": order.price,
            "size": order.size,
            "metadata": order.metadata,
        },
        "response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run prewarm + run_once repeatedly until one live order is submitted."
    )
    parser.add_argument("--config", default="execution_engine/config.yaml")
    parser.add_argument("--secrets", default="execution_engine/secrets.env")
    parser.add_argument("--cache-output", default="artifacts/state/execution_engine/prewarm")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=30.0)
    parser.add_argument("--force-on-abstain", action="store_true")
    parser.add_argument("--forced-side", choices=["YES", "NO", "auto"], default="auto")
    parser.add_argument("--idempotency-suffix", default="full_flow_test")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    load_env_file(Path(args.secrets))
    config = load_execution_config(args.config)
    attempts: list[dict[str, Any]] = []

    for attempt in range(1, args.max_attempts + 1):
        attempt_payload: dict[str, Any] = {"attempt": attempt}
        try:
            attempt_payload["prewarm"] = prewarm(args.config, cache_output=args.cache_output)
            summary = run_once(args.config, mode_override="live")
            attempt_payload["run_once"] = {
                "submitted": bool(summary.get("submitted")),
                "summary_path": summary.get("summary_path"),
                "signal": summary.get("signal"),
                "decision": summary.get("decision"),
                "skipped": summary.get("skipped"),
            }
            if summary.get("submitted"):
                result = {
                    "submitted": True,
                    "success_type": "run_once_submitted",
                    "attempt": attempt,
                    "attempts": attempts + [attempt_payload],
                    "run_once_summary": summary,
                }
                print(json.dumps(result, indent=2 if args.print_json else None, ensure_ascii=False, default=str))
                return

            decision_reason = ((summary.get("decision") or {}).get("reason") or "").lower()
            if args.force_on_abstain and decision_reason == "selective_binary_abstain":
                forced_side = _forced_side(summary, fallback="YES") if args.forced_side == "auto" else args.forced_side
                forced = submit_forced_order_from_runtime(
                    config,
                    forced_side,
                    idempotency_suffix=args.idempotency_suffix,
                )
                result = {
                    "submitted": True,
                    "success_type": "forced_order_submitted",
                    "attempt": attempt,
                    "attempts": attempts + [attempt_payload],
                    "forced_order": forced,
                }
                print(json.dumps(result, indent=2 if args.print_json else None, ensure_ascii=False, default=str))
                return
        except Exception as exc:
            attempt_payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if not _is_service_not_ready(exc):
                attempts.append(attempt_payload)
                raise

        attempts.append(attempt_payload)
        if attempt < args.max_attempts:
            time.sleep(args.sleep_seconds)

    result = {"submitted": False, "success_type": None, "attempts": attempts}
    print(json.dumps(result, indent=2 if args.print_json else None, ensure_ascii=False, default=str))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
