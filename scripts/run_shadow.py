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
from src.core.schemas import MarketQuote, RiskState
from src.data.loaders import load_ohlcv_csv, load_ohlcv_feather, load_ohlcv_parquet
from src.execution.adapters.polymarket import PolymarketExecutionAdapter
from src.execution.audit import (
    decision_evaluated_event,
    market_mapped_event,
    order_created_event,
    signal_generated_event,
)
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


def _merge_quote_metadata(quote: MarketQuote, market: dict | None) -> MarketQuote:
    if market is None:
        return quote
    metadata = dict(quote.metadata)
    metadata.update(
        {
            "gamma_market_id": market.get("market_id"),
            "condition_id": market.get("condition_id"),
            "slug": market.get("slug"),
            "window_start": market.get("window_start"),
            "window_end": market.get("window_end"),
            "yes_token_id": market.get("yes_token_id"),
            "no_token_id": market.get("no_token_id"),
        }
    )
    return MarketQuote(
        market_id=quote.market_id,
        yes_price=quote.yes_price,
        no_price=quote.no_price,
        metadata=metadata,
    )


def _resolve_quote(
    signal,
    args,
    mapper: BTC5mPolymarketMapper,
    adapter: PolymarketExecutionAdapter,
) -> tuple[MarketQuote, dict | None]:
    if args.market_id and args.yes_price is not None:
        return MarketQuote(market_id=args.market_id, yes_price=args.yes_price), None

    market = mapper.map_signal(signal)
    quote = adapter.get_orderbook(market["yes_token_id"])
    return _merge_quote_metadata(quote, market), market


def _build_shadow_summary(
    signal,
    quote: MarketQuote,
    decision,
    order=None,
) -> dict:
    summary = {
        "signal": {
            "asset": signal.asset,
            "horizon": signal.horizon,
            "t0": signal.t0.isoformat(),
            "p_up": round(signal.p_up, 6),
            "model_version": signal.model_version,
            "feature_version": signal.feature_version,
        },
        "market": {
            "market_id": quote.market_id,
            "yes_price": quote.yes_price,
            "no_price": quote.no_price,
            "best_bid": quote.metadata.get("best_bid"),
            "best_ask": quote.metadata.get("best_ask"),
            "slug": quote.metadata.get("slug"),
            "gamma_market_id": quote.metadata.get("gamma_market_id"),
            "condition_id": quote.metadata.get("condition_id"),
            "yes_token_id": quote.metadata.get("yes_token_id"),
            "no_token_id": quote.metadata.get("no_token_id"),
            "window_start": quote.metadata.get("window_start"),
            "window_end": quote.metadata.get("window_end"),
        },
        "decision": {
            "should_trade": decision.should_trade,
            "side": decision.side,
            "edge": round(decision.edge, 6) if decision.edge is not None else None,
            "reason": decision.reason,
            "target_size": decision.target_size,
        },
        "order": order,
    }
    return summary


def _print_shadow_summary(summary: dict) -> None:
    print("Shadow Run Summary")
    print(f"signal.asset={summary['signal']['asset']}")
    print(f"signal.horizon={summary['signal']['horizon']}")
    print(f"signal.t0={summary['signal']['t0']}")
    print(f"signal.p_up={summary['signal']['p_up']}")
    print(f"market.market_id={summary['market']['market_id']}")
    print(f"market.slug={summary['market']['slug']}")
    print(f"market.window_start={summary['market']['window_start']}")
    print(f"market.window_end={summary['market']['window_end']}")
    print(f"market.yes_token_id={summary['market']['yes_token_id']}")
    print(f"market.yes_price={summary['market']['yes_price']}")
    print(f"market.best_bid={summary['market']['best_bid']}")
    print(f"market.best_ask={summary['market']['best_ask']}")
    print(f"decision.should_trade={summary['decision']['should_trade']}")
    print(f"decision.side={summary['decision']['side']}")
    print(f"decision.edge={summary['decision']['edge']}")
    print(f"decision.reason={summary['decision']['reason']}")
    print(f"decision.target_size={summary['decision']['target_size']}")
    if summary["order"] is not None:
        print(f"order.market_id={summary['order']['market_id']}")
        print(f"order.side={summary['order']['side']}")
        print(f"order.price={summary['order']['price']}")
        print(f"order.size={summary['order']['size']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a shadow signal and decision pass.")
    parser.add_argument("--input", required=True, help="Path to OHLCV CSV or parquet input.")
    parser.add_argument("--model", required=True, help="Path to serialized model.")
    parser.add_argument("--calibrator", required=True, help="Path to serialized calibrator.")
    parser.add_argument("--yes-price", type=float, help="Observed Polymarket YES price.")
    parser.add_argument("--market-id", help="Observed Polymarket token identifier.")
    parser.add_argument("--audit-log", default="artifacts/logs/shadow.jsonl", help="Audit log path.")
    parser.add_argument(
        "--summary-output",
        help="Optional path to write a JSON summary of signal, market, decision, and order.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the summary as JSON instead of the default human-readable format.",
    )
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--horizon", default="5m", help="Horizon name to run.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    source = _load_input(Path(args.input))
    model = load_model_plugin(settings.model.active_plugin, args.model)
    calibrator = load_calibration_plugin(settings.calibration.active_plugin, args.calibrator)
    signal_service = SignalService(settings, model=model, calibrator=calibrator)
    audit_service = AuditService(args.audit_log)
    mapper = BTC5mPolymarketMapper(settings)
    adapter = PolymarketExecutionAdapter(settings)

    signal = signal_service.predict_from_latest_frame(source, horizon_name=args.horizon)
    audit_service.append(signal_generated_event(signal))

    quote, market = _resolve_quote(signal, args, mapper=mapper, adapter=adapter)

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

    order_summary = None
    if decision.should_trade:
        order = build_order_request(signal, decision, quote)
        audit_service.append(order_created_event(order, decision))
        order_summary = {
            "market_id": order.market_id,
            "side": order.side,
            "price": order.price,
            "size": order.size,
            "signal_t0": order.signal_t0.isoformat(),
            "metadata": order.metadata,
        }

    summary = _build_shadow_summary(signal, quote, decision, order=order_summary)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.print_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        _print_shadow_summary(summary)


if __name__ == "__main__":
    main()
