from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.registry import load_calibration_plugin
from src.core.config import load_settings
from src.core.schemas import RiskState
from src.data.derivatives.feature_store import load_derivatives_frame_from_settings
from src.data.loaders import load_ohlcv_csv, load_ohlcv_feather, load_ohlcv_parquet
from src.execution.adapters.polymarket import PolymarketExecutionAdapter
from src.execution.audit import (
    decision_evaluated_event,
    execution_skipped_event,
    guard_evaluated_event,
    market_mapped_event,
    order_created_event,
    order_submitted_event,
    signal_generated_event,
)
from src.execution.guards import build_window_key, evaluate_market_guards
from src.execution.idempotency import IdempotencyStore
from src.execution.mappers.btc_5m_polymarket import BTC5mPolymarketMapper
from src.execution.order_router import build_order_request
from src.model.registry import load_model_plugin
from src.services.audit_service import AuditService
from src.services.signal_service import SignalService
from src.signal.decision_engine import evaluate_entry


def _load_input(path: Path):
    if path.suffix.lower() == ".csv":
        return load_ohlcv_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return load_ohlcv_parquet(path)
    if path.suffix.lower() in {".feather", ".ft"}:
        return load_ohlcv_feather(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _build_summary(signal, quote, decision, guard, mode, submitted, response=None) -> dict:
    return {
        "mode": mode,
        "signal": {
            "asset": signal.asset,
            "horizon": signal.horizon,
            "t0": signal.t0.isoformat(),
            "p_up": signal.p_up,
        },
        "market": {
            "market_id": quote.market_id,
            "slug": quote.metadata.get("slug"),
            "yes_price": quote.yes_price,
            "best_bid": quote.metadata.get("best_bid"),
            "best_ask": quote.metadata.get("best_ask"),
            "liquidity_clob": quote.metadata.get("liquidity_clob"),
        },
        "decision": {
            "should_trade": decision.should_trade,
            "side": decision.side,
            "edge": decision.edge,
            "reason": decision.reason,
            "target_size": decision.target_size,
        },
        "guard": {
            "allowed": guard.allowed,
            "reason": guard.reason,
            "details": guard.details,
        },
        "submitted": submitted,
        "response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a protected live or paper signal flow.")
    parser.add_argument("--input", required=True, help="Path to OHLCV CSV or parquet input.")
    parser.add_argument("--model", required=True, help="Path to serialized model.")
    parser.add_argument("--calibrator", required=True, help="Path to serialized calibrator.")
    parser.add_argument("--t-up", type=float, default=None, help="UP decision threshold from the training artifact.")
    parser.add_argument("--t-down", type=float, default=None, help="DOWN decision threshold from the training artifact.")
    parser.add_argument("--audit-log", default="artifacts/logs/live.jsonl", help="Audit log path.")
    parser.add_argument("--summary-output", help="Optional path to write JSON summary.")
    parser.add_argument("--print-json", action="store_true", help="Print JSON summary.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--horizon", default="5m", help="Horizon name to run.")
    parser.add_argument("--funding-input", help="Optional funding raw input override.")
    parser.add_argument("--basis-input", help="Optional basis raw input override.")
    parser.add_argument("--oi-input", help="Optional OI raw input override.")
    parser.add_argument("--options-input", help="Optional options raw input override.")
    parser.add_argument("--book-ticker-input", help="Optional bookTicker raw input override.")
    parser.add_argument(
        "--derivatives-path-mode",
        choices=["latest", "archive"],
        default=None,
        help="Override derivatives path mode. Defaults to settings.derivatives.path_mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        help="Override configured execution mode.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    mode = args.mode or settings.execution.mode
    source = _load_input(Path(args.input))
    derivatives_frame = load_derivatives_frame_from_settings(
        settings,
        funding_path=args.funding_input,
        basis_path=args.basis_input,
        oi_path=args.oi_input,
        options_path=args.options_input,
        book_ticker_path=args.book_ticker_input,
        path_mode=args.derivatives_path_mode,
    )
    model = load_model_plugin(settings.model.active_plugin, args.model)
    calibrator = load_calibration_plugin(settings.calibration.active_plugin, args.calibrator)
    signal_service = SignalService(settings, model=model, calibrator=calibrator, t_up=args.t_up, t_down=args.t_down)
    mapper = BTC5mPolymarketMapper(settings)
    adapter = PolymarketExecutionAdapter(settings)
    audit_service = AuditService(args.audit_log)
    idempotency_store = IdempotencyStore(settings.execution.safeguards["idempotency_store_path"])

    signal = signal_service.predict_from_latest_frame(
        source,
        horizon_name=args.horizon,
        derivatives_frame=derivatives_frame,
    )
    audit_service.append(signal_generated_event(signal))

    market = mapper.map_signal(signal)
    quote = adapter.get_orderbook(market["yes_token_id"])
    quote = quote.__class__(
        market_id=quote.market_id,
        yes_price=quote.yes_price,
        no_price=quote.no_price,
        metadata={**quote.metadata, **market},
    )
    audit_service.append(market_mapped_event(signal, quote))

    decision = evaluate_entry(
        signal,
        yes_price=quote.yes_price,
        settings=settings,
        risk_state=RiskState(
            current_exposure=0.0,
            max_total_exposure=settings.sizing.plugins[settings.sizing.active_plugin].get(
                "max_total_exposure",
                0.0,
            ),
            active_positions=0,
        ),
    )
    audit_service.append(decision_evaluated_event(signal, quote, decision))

    guard = evaluate_market_guards(signal, decision, quote, settings)
    audit_service.append(guard_evaluated_event(signal, quote, guard))

    submitted = False
    response = None
    if not guard.allowed:
        audit_service.append(execution_skipped_event(signal, quote, guard.reason, guard.details))
    else:
        key = build_window_key(signal, quote)
        enforce_idempotency = mode == "live" or settings.execution.safeguards.get(
            "enforce_idempotency_in_paper",
            False,
        )
        if enforce_idempotency and idempotency_store.has(key):
            audit_service.append(
                execution_skipped_event(
                    signal,
                    quote,
                    "idempotency_key_already_seen",
                    {"key": key},
                )
            )
            guard = guard.__class__(
                allowed=False,
                reason="idempotency_key_already_seen",
                details={"key": key},
            )
        else:
            order = build_order_request(signal, decision, quote)
            audit_service.append(order_created_event(order, decision))
            if mode == "live":
                response = adapter.place_limit_order(order)
                idempotency_store.record(key, {"market_id": order.market_id, "side": order.side})
                submitted = True
                audit_service.append(order_submitted_event(order, response, mode=mode))
            else:
                response = {
                    "mode": "paper",
                    "would_submit": {
                        "market_id": order.market_id,
                        "side": order.side,
                        "price": order.price,
                        "size": order.size,
                    },
                }

    summary = _build_summary(signal, quote, decision, guard, mode=mode, submitted=submitted, response=response)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.print_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("Live Signal Summary")
        print(f"mode={summary['mode']}")
        print(f"signal.t0={summary['signal']['t0']}")
        print(f"signal.p_up={summary['signal']['p_up']}")
        print(f"market.slug={summary['market']['slug']}")
        print(f"market.yes_price={summary['market']['yes_price']}")
        print(f"market.best_bid={summary['market']['best_bid']}")
        print(f"market.best_ask={summary['market']['best_ask']}")
        print(f"market.liquidity_clob={summary['market']['liquidity_clob']}")
        print(f"decision.should_trade={summary['decision']['should_trade']}")
        print(f"decision.edge={summary['decision']['edge']}")
        print(f"decision.reason={summary['decision']['reason']}")
        print(f"guard.allowed={summary['guard']['allowed']}")
        print(f"guard.reason={summary['guard']['reason']}")
        print(f"submitted={summary['submitted']}")


if __name__ == "__main__":
    main()
