from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution_engine.artifacts import load_baseline_artifact, load_price_estimator_artifact
from execution_engine.config import load_execution_config
from execution_engine.feature_runtime import RuntimeInferenceEngine
from execution_engine.order_plan import build_two_limit_order_plan
from execution_engine.polymarket_v2 import PolymarketV2Adapter
from execution_engine.realtime_data import BinanceRealtimeClient
from src.core.config import load_settings
from src.core.schemas import AuditEvent
from src.execution.idempotency import IdempotencyStore
from src.services.audit_service import AuditService
from src.signal.policies import evaluate_selective_binary_signal


def build_btc_5m_slug(t0: datetime, *, offset_windows: int = 0) -> tuple[str, datetime, datetime]:
    ts = t0.astimezone(UTC).replace(second=0, microsecond=0)
    minute = ts.minute - (ts.minute % 5)
    window_start = ts.replace(minute=minute) + timedelta(minutes=5 * offset_windows)
    window_end = window_start + timedelta(minutes=5)
    return f"btc-updown-5m-{int(window_start.timestamp())}", window_start, window_end


def current_5m_window_start(now: datetime | None = None) -> datetime:
    ts = (now or datetime.now(UTC)).astimezone(UTC).replace(second=0, microsecond=0)
    minute = ts.minute - (ts.minute % 5)
    return ts.replace(minute=minute)


def seconds_until_configured_trigger(now: datetime, *, delay_seconds: int) -> float:
    ts = now.astimezone(UTC)
    trigger_time = current_5m_window_start(ts) + timedelta(seconds=delay_seconds)
    return max(0.0, (trigger_time - ts).total_seconds())


def build_idempotency_key(signal_t0: datetime, token_id: str, side: str, leg: str = "order") -> str:
    return f"{signal_t0.astimezone(UTC).isoformat()}:{token_id}:{side}:{leg}:two_limit_plan"


def audit_event(event_type: str, payload: dict[str, Any]) -> AuditEvent:
    return AuditEvent(event_type=event_type, timestamp=datetime.now(UTC), payload=payload)


def run_once(
    config_path: str,
    mode_override: str | None = None,
    target_window_start: datetime | None = None,
) -> dict[str, Any]:
    config = load_execution_config(config_path)
    mode = mode_override or config.runtime.mode
    if target_window_start is None:
        wait_seconds = seconds_until_configured_trigger(
            datetime.now(UTC),
            delay_seconds=config.schedule.trigger_delay_seconds,
        )
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        target_window_start = current_5m_window_start()
    baseline = load_baseline_artifact(config.baseline)
    price_estimator = load_price_estimator_artifact(config.price_estimator)
    settings = load_settings(config.baseline.settings_path)
    alignment = getattr(settings, "decision_alignment", None)
    feature_offset_minutes = (
        int(getattr(alignment, "feature_offset_minutes", 0))
        if alignment is not None and getattr(alignment, "enabled", False)
        else 0
    )
    audit = AuditService(config.runtime.audit_log)

    binance = BinanceRealtimeClient(config.binance)
    minute_frame, second_frame, agg_trades_frame, frame_alignment = binance.wait_for_signal_runtime_frames(
        signal_t0=target_window_start,
        max_wait_seconds=config.schedule.max_data_wait_seconds,
        feature_offset_minutes=feature_offset_minutes,
    )
    audit.append(
        audit_event(
            "binance_frames_loaded",
            {
                "minute_rows": len(minute_frame),
                "second_rows": len(second_frame),
                "agg_trade_rows": len(agg_trades_frame),
                "minute_latest": None if minute_frame.empty else minute_frame["timestamp"].iloc[-1].isoformat(),
                "second_latest": None if second_frame.empty else second_frame["timestamp"].iloc[-1].isoformat(),
                "agg_trade_latest": None if agg_trades_frame.empty else agg_trades_frame["timestamp"].iloc[-1].isoformat(),
                "frame_alignment": frame_alignment,
            },
        )
    )

    inference = RuntimeInferenceEngine(
        settings,
        baseline,
        price_estimator=price_estimator,
        t_up=config.thresholds.t_up,
        t_down=config.thresholds.t_down,
    )
    result = inference.predict(
        minute_frame,
        second_frame,
        agg_trades_frame,
        signal_t0=pd.Timestamp(target_window_start),
        use_latest_available_before_signal=False,
        runtime_context=frame_alignment,
    )
    signal = result.signal
    t_up = float(signal.decision_context["t_up"])
    t_down = float(signal.decision_context["t_down"])
    audit.append(
        audit_event(
            "signal_generated",
            {
                "asset": signal.asset,
                "horizon": signal.horizon,
                "t0": signal.t0.isoformat(),
                "p_up": signal.p_up,
                "p_down": signal.p_down,
                "t_up": t_up,
                "t_down": t_down,
                "artifact_t_up": baseline.t_up,
                "artifact_t_down": baseline.t_down,
                "feature_count": len(baseline.feature_columns),
                "feature_timestamp": signal.decision_context.get("feature_timestamp"),
                "decision_time": signal.decision_context.get("decision_time"),
                "market_t0": signal.decision_context.get("market_t0"),
                "row_policy": signal.decision_context.get("row_policy"),
                "required_latest_closed_minute": signal.decision_context.get("required_latest_closed_minute"),
                "required_latest_closed_second": signal.decision_context.get("required_latest_closed_second"),
                "post_signal_second_rows_dropped": signal.decision_context.get("post_signal_second_rows_dropped"),
                "post_signal_agg_trade_rows_dropped": signal.decision_context.get("post_signal_agg_trade_rows_dropped"),
                "baseline_artifact_dir": str(baseline.artifact_dir),
            },
        )
    )

    decision = evaluate_selective_binary_signal(signal, settings=settings)
    if decision.should_trade and decision.side is not None:
        try:
            price_context = inference.predict_price_q80(
                result.feature_frame,
                row_index=result.row_index if result.row_index is not None else result.feature_frame.index[-1],
                selected_side=str(decision.side),
                p_up=signal.p_up,
            )
        except Exception as exc:
            price_context = {
                "price_estimator_enabled": price_estimator is not None,
                "price_estimator_error": repr(exc),
                "price_estimator_fallback_price_mode": config.price_estimator.fallback_price_mode,
            }
        if price_context:
            signal = replace(signal, decision_context={**signal.decision_context, **price_context})
    audit.append(audit_event("decision_evaluated", asdict(decision)))

    summary: dict[str, Any] = {
        "mode": mode,
        "orders_enabled": config.orders.enabled,
        "signal": {
            "asset": signal.asset,
            "horizon": signal.horizon,
            "t0": signal.t0.isoformat(),
            "p_up": signal.p_up,
            "p_down": signal.p_down,
            "t_up": t_up,
            "t_down": t_down,
            "artifact_t_up": baseline.t_up,
            "artifact_t_down": baseline.t_down,
            "feature_timestamp": signal.decision_context.get("feature_timestamp"),
            "decision_time": signal.decision_context.get("decision_time"),
            "market_t0": signal.decision_context.get("market_t0"),
            "decision_alignment_mode": signal.decision_context.get("decision_alignment_mode"),
            "feature_offset_minutes": signal.decision_context.get("feature_offset_minutes"),
            "row_policy": signal.decision_context.get("row_policy"),
            "required_latest_closed_minute": signal.decision_context.get("required_latest_closed_minute"),
            "required_latest_closed_second": signal.decision_context.get("required_latest_closed_second"),
            "required_latest_agg_trade": signal.decision_context.get("required_latest_agg_trade"),
            "minute_latest": signal.decision_context.get("minute_latest"),
            "second_latest": signal.decision_context.get("second_latest"),
            "agg_trade_latest": signal.decision_context.get("agg_trade_latest"),
            "agg_trade_lag_seconds": signal.decision_context.get("agg_trade_lag_seconds"),
            "max_agg_trade_lag_seconds": signal.decision_context.get("max_agg_trade_lag_seconds"),
            "prewarm_base_until": signal.decision_context.get("prewarm_base_until"),
            "price_estimator_enabled": signal.decision_context.get("price_estimator_enabled"),
            "price_estimator_q80_raw": signal.decision_context.get("price_estimator_q80_raw"),
            "price_estimator_q80_rounded": signal.decision_context.get("price_estimator_q80_rounded"),
            "price_estimator_safe_gap_raw": signal.decision_context.get("price_estimator_safe_gap_raw"),
            "price_estimator_safe_gap_rounded": signal.decision_context.get("price_estimator_safe_gap_rounded"),
            "price_estimator_safe_gap_action": signal.decision_context.get("price_estimator_safe_gap_action"),
            "price_estimator_safe_gap_conf_ok": signal.decision_context.get("price_estimator_safe_gap_conf_ok"),
            "price_estimator_safe_gap_f_model": signal.decision_context.get("price_estimator_safe_gap_f_model"),
            "price_estimator_best_ask_offset": signal.decision_context.get("price_estimator_best_ask_offset"),
            "price_estimator_selected_side": signal.decision_context.get("price_estimator_selected_side"),
            "price_estimator_artifact_dir": signal.decision_context.get("price_estimator_artifact_dir"),
            "price_estimator_error": signal.decision_context.get("price_estimator_error"),
            "price_estimator_fallback_price_mode": signal.decision_context.get("price_estimator_fallback_price_mode"),
        },
        "decision": asdict(decision),
        "market": None,
        "orders": [],
        "skipped": [],
        "submitted": False,
    }

    if not decision.should_trade:
        summary["skipped"].append({"reason": decision.reason})
        audit.append(audit_event("execution_skipped", {"reason": decision.reason}))
        return write_summary(config.runtime.summary_dir, summary)

    polymarket = PolymarketV2Adapter(config.polymarket)
    slug, window_start, window_end = build_btc_5m_slug(signal.t0, offset_windows=0)
    market = polymarket.get_market_by_slug(slug)
    if market is None:
        reason = "market_not_found"
        details = {"slug": slug, "window_start": window_start.isoformat(), "window_end": window_end.isoformat()}
        summary["skipped"].append({"reason": reason, **details})
        audit.append(audit_event("execution_skipped", {"reason": reason, "details": details}))
        return write_summary(config.runtime.summary_dir, summary)

    if config.guards.require_market_accepting_orders and (
        not market.active or market.closed or not market.accepting_orders
    ):
        reason = "market_not_accepting_orders"
        details = {
            "slug": market.slug,
            "active": market.active,
            "closed": market.closed,
            "accepting_orders": market.accepting_orders,
        }
        summary["market"] = {**details, "market_id": market.market_id}
        summary["skipped"].append({"reason": reason, **details})
        audit.append(audit_event("execution_skipped", {"reason": reason, "details": details}))
        return write_summary(config.runtime.summary_dir, summary)

    token_id = market.yes_token_id if decision.side == "YES" else market.no_token_id
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
    summary["market"] = {
        "slug": market.slug,
        "market_id": market.market_id,
        "target_token_id": token_id,
        "yes_token_id": market.yes_token_id,
        "no_token_id": market.no_token_id,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "best_bid": quote.metadata.get("best_bid"),
        "best_ask": quote.metadata.get("best_ask"),
        "tick_size": quote.metadata.get("tick_size"),
    }
    audit.append(audit_event("market_mapped", summary["market"]))

    if config.guards.require_best_bid and quote.metadata.get("best_bid") is None and quote.metadata.get("best_ask") is None:
        reason = "missing_quote"
        summary["skipped"].append({"reason": reason})
        audit.append(audit_event("execution_skipped", {"reason": reason, "market": summary["market"]}))
        return write_summary(config.runtime.summary_dir, summary)

    order_plan = build_two_limit_order_plan(signal, decision, quote, config.orders, config.execution_edge)
    summary["orders"] = [asdict(order) for order in order_plan.orders]
    summary["skipped"].extend(order_plan.skipped)
    audit.append(audit_event("order_plan_created", {"orders": summary["orders"], "skipped": order_plan.skipped}))

    if not config.orders.enabled or mode != "live":
        summary["submitted"] = False
        summary["skipped"].append({"reason": "paper_mode_or_orders_disabled"})
        audit.append(audit_event("execution_skipped", {"reason": "paper_mode_or_orders_disabled"}))
        return write_summary(config.runtime.summary_dir, summary)

    store = IdempotencyStore(config.runtime.idempotency_store_path)
    responses: list[dict[str, Any]] = []
    order_statuses: list[dict[str, Any]] = []
    submitted_order_ids: list[dict[str, str]] = []
    for order in order_plan.orders[: config.guards.max_orders_per_window]:
        leg = str(order.metadata.get("leg", "order"))
        key = build_idempotency_key(window_start, order.market_id, order.side, leg)
        if config.guards.enforce_idempotency and store.has(key):
            skipped = {"reason": "idempotency_key_already_seen", "key": key}
            summary["skipped"].append(skipped)
            audit.append(audit_event("execution_skipped", skipped))
            continue
        response = polymarket.place_limit_order(order)
        response_payload = _order_response_payload(response)
        order_id = response_payload.get("orderID")
        if not _order_response_success(response):
            skipped = {
                "leg": leg,
                "reason": "order_submission_failed",
                "order_id": order_id,
                "status": response_payload.get("status"),
                "error": response_payload.get("errorMsg") or response_payload.get("error"),
            }
            summary["skipped"].append(skipped)
            audit.append(audit_event("order_submission_failed", {"order": asdict(order), "response": response}))
            responses.append(response)
            continue
        responses.append(response)
        if order_id and hasattr(polymarket, "get_order_status"):
            try:
                status_payload = {"leg": leg, "order_id": order_id, **polymarket.get_order_status(order_id)}
                order_statuses.append(status_payload)
            except Exception as exc:
                order_statuses.append({"leg": leg, "order_id": order_id, "status_lookup_error": repr(exc)})
        if order_id:
            submitted_order_ids.append({"leg": leg, "order_id": str(order_id)})
        if config.guards.enforce_idempotency:
            store.record(key, {"market_id": order.market_id, "side": order.side, "price": order.price, "leg": leg})
        audit.append(audit_event("order_submitted", {"order": asdict(order), "response": response}))

    summary["submitted"] = any(_order_response_success(response) for response in responses)
    summary["responses"] = responses
    cancellations = _cancel_unfilled_orders_after_delay(
        polymarket=polymarket,
        submitted_order_ids=submitted_order_ids,
        delay_seconds=float(config.orders.cancel_unfilled_after_seconds),
        audit=audit,
    )
    if cancellations:
        summary["order_cancellations"] = cancellations
    if order_statuses:
        summary["order_statuses"] = order_statuses
    return write_summary(config.runtime.summary_dir, summary)


def _order_response_payload(response: dict[str, Any]) -> dict[str, Any]:
    payload = response.get("response", response)
    return payload if isinstance(payload, dict) else {}


def _order_response_success(response: dict[str, Any]) -> bool:
    payload = _order_response_payload(response)
    return bool(payload.get("success") is True and payload.get("orderID"))


def _cancel_unfilled_orders_after_delay(
    *,
    polymarket: Any,
    submitted_order_ids: list[dict[str, str]],
    delay_seconds: float,
    audit: AuditService,
) -> list[dict[str, Any]]:
    if delay_seconds <= 0.0 or not submitted_order_ids:
        return []
    if not hasattr(polymarket, "get_order_status") or not hasattr(polymarket, "cancel_order"):
        return [
            {
                "leg": order["leg"],
                "order_id": order["order_id"],
                "cancel_skipped": True,
                "reason": "adapter_missing_status_or_cancel",
            }
            for order in submitted_order_ids
        ]
    time.sleep(delay_seconds)
    cancellations: list[dict[str, Any]] = []
    for order in submitted_order_ids:
        leg = order["leg"]
        order_id = order["order_id"]
        try:
            status = polymarket.get_order_status(order_id)
            if _order_status_has_unfilled_quantity(status):
                cancel_response = polymarket.cancel_order(order_id)
                payload = {
                    "leg": leg,
                    "order_id": order_id,
                    "status_before_cancel": status,
                    "cancel_response": cancel_response,
                }
                cancellations.append(payload)
                audit.append(audit_event("order_cancelled_after_timeout", payload))
            else:
                payload = {
                    "leg": leg,
                    "order_id": order_id,
                    "status_before_cancel": status,
                    "cancel_skipped": True,
                    "reason": "order_filled_or_not_open",
                }
                cancellations.append(payload)
                audit.append(audit_event("order_cancel_skipped_after_timeout", payload))
        except Exception as exc:
            payload = {"leg": leg, "order_id": order_id, "cancel_error": repr(exc)}
            cancellations.append(payload)
            audit.append(audit_event("order_cancel_error_after_timeout", payload))
    return cancellations


def _order_status_has_unfilled_quantity(status: dict[str, Any]) -> bool:
    status_text = str(status.get("status", "")).lower()
    if status_text in {"filled", "matched", "cancelled", "canceled"}:
        return False
    size_matched = _optional_float(status.get("size_matched"))
    original_size = _optional_float(status.get("original_size"))
    if size_matched is not None and original_size is not None:
        return size_matched < original_size
    return status_text in {"", "live", "open", "active", "pending"}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_summary(summary_dir: str, summary: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(summary_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_t0 = summary.get("signal", {}).get("t0", datetime.now(UTC).isoformat())
    safe_t0 = str(signal_t0).replace(":", "").replace("+", "Z")
    path = output_dir / f"{safe_t0}.json"
    summary["summary_path"] = str(path)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one execution-engine cycle.")
    parser.add_argument("--config", default="execution_engine/config.yaml")
    parser.add_argument("--mode", choices=["paper", "live"], default=None)
    parser.add_argument("--target-window-start", default=None)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()
    target_window_start = (
        None
        if args.target_window_start is None
        else pd.Timestamp(args.target_window_start).tz_convert(UTC).to_pydatetime()
    )
    summary = run_once(args.config, mode_override=args.mode, target_window_start=target_window_start)
    if args.print_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"summary_path={summary['summary_path']}")
        print(f"decision={summary['decision']['reason']}")
        print(f"submitted={summary['submitted']}")


if __name__ == "__main__":
    main()
