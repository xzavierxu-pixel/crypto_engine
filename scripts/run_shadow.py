from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.core.schemas import MarketQuote, RiskState
from src.data.derivatives.feature_store import load_derivatives_frame_from_settings
from src.data.loaders import load_ohlcv_csv, load_ohlcv_feather, load_ohlcv_parquet
from src.execution.adapters.polymarket import PolymarketExecutionAdapter
from src.execution.audit import (
    decision_evaluated_event,
    market_mapped_event,
    order_created_event,
    signal_generated_event,
    stage1_drift_alert_event,
)
from src.execution.mappers.btc_5m_polymarket import BTC5mPolymarketMapper
from src.execution.order_router import build_order_request
from src.model.artifacts import load_two_stage_artifacts
from src.model.drift import Stage1DriftMonitor
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


def _build_stage1_drift_monitor(
    reference_probabilities: list[float],
    *,
    threshold: float = 0.1,
    window_size: int = 500,
    min_history: int = 50,
    alert_consecutive: int = 3,
) -> Stage1DriftMonitor | None:
    if not reference_probabilities:
        return None
    return Stage1DriftMonitor(
        pd.Series(reference_probabilities, dtype="float64"),
        threshold=threshold,
        window_size=window_size,
        min_history=min_history,
        alert_consecutive=alert_consecutive,
    )


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
            "p_active": round(float(signal.p_active or 0.0), 6),
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
    print(f"signal.p_active={summary['signal']['p_active']}")
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
    parser.add_argument("--artifact-dir", help="Directory containing two-stage model artifacts and training_report.json.")
    parser.add_argument("--report", help="Path to training_report.json for two-stage model loading.")
    parser.add_argument("--stage1-model", help="Optional explicit Stage 1 model path.")
    parser.add_argument("--stage2-model", help="Optional explicit Stage 2 model path.")
    parser.add_argument("--stage1-calibrator", help="Optional explicit Stage 1 calibrator path.")
    parser.add_argument("--stage2-calibrator", help="Optional explicit Stage 2 calibrator path.")
    parser.add_argument("--drift-threshold", type=float, default=0.1, help="KS threshold for Stage 1 drift.")
    parser.add_argument("--drift-window-size", type=int, default=500, help="Live window size for Stage 1 drift.")
    parser.add_argument("--drift-min-history", type=int, default=50, help="Minimum history before Stage 1 drift is evaluated.")
    parser.add_argument(
        "--drift-alert-consecutive",
        type=int,
        default=3,
        help="Consecutive threshold breaches required before emitting a Stage 1 drift alert.",
    )
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
    parser.add_argument("--funding-input", help="Optional funding raw input override.")
    parser.add_argument("--basis-input", help="Optional basis raw input override.")
    parser.add_argument("--oi-input", help="Optional OI raw input override.")
    parser.add_argument("--options-input", help="Optional options raw input override.")
    parser.add_argument(
        "--derivatives-path-mode",
        choices=["latest", "archive"],
        default="latest",
        help="Derivatives path mode for shadow runs. Defaults to latest.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    source = _load_input(Path(args.input))
    derivatives_frame = load_derivatives_frame_from_settings(
        settings,
        funding_path=args.funding_input,
        basis_path=args.basis_input,
        oi_path=args.oi_input,
        options_path=args.options_input,
        path_mode=args.derivatives_path_mode,
    )
    bundle = load_two_stage_artifacts(
        settings,
        report_path=args.report,
        artifact_dir=args.artifact_dir,
        stage1_model_path=args.stage1_model,
        stage2_model_path=args.stage2_model,
        stage1_calibrator_path=args.stage1_calibrator,
        stage2_calibrator_path=args.stage2_calibrator,
    )
    drift_monitor = _build_stage1_drift_monitor(
        bundle.stage1_reference_probabilities,
        threshold=args.drift_threshold,
        window_size=args.drift_window_size,
        min_history=args.drift_min_history,
        alert_consecutive=args.drift_alert_consecutive,
    )
    signal_service = SignalService(
        settings,
        stage1_model=bundle.stage1_model,
        stage2_model=bundle.stage2_model,
        stage1_calibrator=bundle.stage1_calibrator,
        stage2_calibrator=bundle.stage2_calibrator,
        feature_columns=bundle.feature_columns,
        stage2_feature_columns=bundle.stage2_feature_columns,
        stage1_threshold=bundle.stage1_threshold,
        buy_threshold=bundle.buy_threshold,
        model_version=bundle.model_version,
        stage1_drift_monitor=drift_monitor,
    )
    audit_service = AuditService(args.audit_log)
    mapper = BTC5mPolymarketMapper(settings)
    adapter = PolymarketExecutionAdapter(settings)

    signal = signal_service.predict_from_latest_frame(
        source,
        horizon_name=args.horizon,
        derivatives_frame=derivatives_frame,
    )
    audit_service.append(signal_generated_event(signal))
    drift_state = signal.decision_context.get("stage1_drift")
    if drift_state and drift_state.get("alert"):
        audit_service.append(stage1_drift_alert_event(signal, drift_state))

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
