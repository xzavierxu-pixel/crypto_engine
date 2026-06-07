from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.registry import load_calibration_plugin
from src.model.registry import load_model_plugin


def _load_label_lookup(label_store_path: Path) -> pd.DataFrame:
    labels = pd.read_parquet(label_store_path)
    labels["market_t0"] = pd.to_datetime(labels["market_t0"], utc=True)
    return labels[["polymarket_slug", "market_t0", "target"]].drop_duplicates(subset=["polymarket_slug"])


def _load_price_store(price_store_path: Path) -> pd.DataFrame:
    prices = pd.read_parquet(price_store_path)
    prices["market_t0"] = pd.to_datetime(prices["market_t0"], utc=True)
    ok = prices[prices["price_status"].eq("ok")].copy()
    return ok[["polymarket_slug", "market_t0", "yes_mid_price", "price_ts", "price_time", "seconds_from_t0"]]


def _selection_metrics(frame: pd.DataFrame) -> dict[str, float]:
    sample_count = int(len(frame))
    accepted = frame[frame["accepted"]]
    accepted_count = int(len(accepted))
    up_count = int((accepted["side"] == "UP").sum())
    down_count = int((accepted["side"] == "DOWN").sum())
    correct_count = int(accepted["correct"].sum()) if accepted_count else 0
    coverage = accepted_count / sample_count if sample_count else 0.0
    accepted_accuracy = correct_count / accepted_count if accepted_count else 0.0
    utility = coverage * (2.0 * accepted_accuracy - 1.0)
    downside_risk = math.sqrt(max(coverage * (1.0 - accepted_accuracy), 0.0))
    return {
        "sample_count": float(sample_count),
        "coverage": float(coverage),
        "accepted_count": float(accepted_count),
        "accepted_sample_accuracy": float(accepted_accuracy),
        "up_prediction_count": float(up_count),
        "down_prediction_count": float(down_count),
        "share_up_predictions": float(up_count / accepted_count) if accepted_count else 0.0,
        "share_down_predictions": float(down_count / accepted_count) if accepted_count else 0.0,
        "wins": float(correct_count),
        "losses": float(accepted_count - correct_count),
        "utility": float(utility),
        "downside_risk": float(downside_risk),
        "selection_score": float(utility / downside_risk) if downside_risk > 0.0 else 0.0,
    }


def _trading_metrics(frame: pd.DataFrame, *, missing_price_samples: int) -> dict[str, float]:
    accepted = frame[frame["accepted"]]
    total = int(len(frame))
    accepted_count = int(len(accepted))
    profit_sum = float(accepted["realized_profit"].sum()) if accepted_count else 0.0
    cost_sum = float(accepted["entry_price"].sum()) if accepted_count else 0.0
    metrics = _selection_metrics(frame)
    metrics.update(
        {
            "missing_price_samples": float(missing_price_samples),
            "realized_profit_sum": profit_sum,
            "trading_utility": float(profit_sum / total) if total else 0.0,
            "avg_profit_per_accepted_trade": float(profit_sum / accepted_count) if accepted_count else 0.0,
            "avg_entry_price": float(accepted["entry_price"].mean()) if accepted_count else 0.0,
            "avg_abs_edge": float(accepted["abs_edge"].mean()) if accepted_count else 0.0,
            "avg_signed_edge": float(accepted["chosen_edge"].mean()) if accepted_count else 0.0,
            "roi_on_cost": float(profit_sum / cost_sum) if cost_sum else 0.0,
        }
    )
    return metrics


def _apply_edge_policy(frame: pd.DataFrame, *, min_ev_threshold: float) -> pd.DataFrame:
    out = frame.copy()
    out["q_up"] = out["p_up"].astype(float)
    out["p_yes"] = out["yes_mid_price"].astype(float)
    out["edge_up"] = out["q_up"] - out["p_yes"]
    out["edge_down"] = -out["edge_up"]
    out["abs_edge"] = out["edge_up"].abs()
    out["side"] = out["edge_up"].map(lambda value: "UP" if value >= 0.0 else "DOWN")
    out["chosen_edge"] = out["abs_edge"]
    out["accepted"] = out["chosen_edge"] > float(min_ev_threshold)
    out.loc[~out["accepted"], "side"] = "ABSTAIN"
    out["entry_price"] = out.apply(
        lambda row: row["p_yes"] if row["side"] == "UP" else 1.0 - row["p_yes"] if row["side"] == "DOWN" else 0.0,
        axis=1,
    )
    out["resolved_up"] = out["target"].astype(float)
    out["resolved_down"] = 1.0 - out["resolved_up"]
    out["realized_profit"] = out.apply(
        lambda row: row["resolved_up"] - row["p_yes"]
        if row["side"] == "UP"
        else row["resolved_down"] - (1.0 - row["p_yes"])
        if row["side"] == "DOWN"
        else 0.0,
        axis=1,
    )
    out["correct"] = (
        ((out["side"] == "UP") & (out["target"].astype(int) == 1))
        | ((out["side"] == "DOWN") & (out["target"].astype(int) == 0))
    ) & out["accepted"]
    return out


def _attach_prices(frame: pd.DataFrame, labels: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = frame.copy()
    if "polymarket_slug" not in out.columns:
        time_column = "market_t0" if "market_t0" in out.columns else "timestamp"
        out[time_column] = pd.to_datetime(out[time_column], utc=True)
        out = out.merge(labels[["polymarket_slug", "market_t0", "target"]], left_on=time_column, right_on="market_t0", how="left")
        if "target_x" in out.columns:
            out["target"] = out["target_x"].fillna(out.get("target_y"))
            out = out.drop(columns=[column for column in ("target_x", "target_y") if column in out])
    else:
        out["polymarket_slug"] = out["polymarket_slug"].astype(str)
        if "target" not in out.columns:
            out = out.merge(labels[["polymarket_slug", "target"]], on="polymarket_slug", how="left")
    joined = out.merge(prices, on="polymarket_slug", how="left", suffixes=("", "_price"))
    missing = int(joined["yes_mid_price"].isna().sum())
    joined = joined[joined["yes_mid_price"].notna() & joined["p_up"].notna() & joined["target"].notna()].copy()
    return joined, missing


def evaluate_prediction_frame(
    *,
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    prices: pd.DataFrame,
    min_ev_threshold: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    joined, missing = _attach_prices(frame, labels, prices)
    evaluated = _apply_edge_policy(joined, min_ev_threshold=min_ev_threshold)
    return _trading_metrics(evaluated, missing_price_samples=missing), evaluated


def _artifact_predictions(artifact_dir: Path) -> pd.DataFrame:
    predictions_path = artifact_dir / "validation_predictions.parquet"
    if predictions_path.exists():
        return pd.read_parquet(predictions_path)
    manifest_path = artifact_dir / "artifact_manifest.json"
    validation_frame_path = artifact_dir / "validation_frame.parquet"
    if not manifest_path.exists() or not validation_frame_path.exists():
        raise FileNotFoundError(f"Missing validation predictions and fallback inputs for {artifact_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_plugin = str(manifest["model_plugin"])
    calibration_plugin = str(manifest["calibration_plugin"])
    feature_columns = list(manifest["feature_columns"])
    frame = pd.read_parquet(validation_frame_path)
    model = load_model_plugin(model_plugin, str(artifact_dir / f"{model_plugin}.binary.pkl"))
    calibrator = load_calibration_plugin(calibration_plugin, str(artifact_dir / f"{calibration_plugin}.binary.pkl"))
    raw_proba = model.predict_proba(frame[feature_columns])
    proba = calibrator.transform(raw_proba)
    out_columns = [
        column
        for column in ("timestamp", "market_t0", "polymarket_slug", "target", "feature_timestamp", "decision_time")
        if column in frame.columns
    ]
    out = frame[out_columns].copy()
    out["p_up"] = proba.to_numpy()
    return out


def evaluate_artifact(
    artifact_dir: Path,
    *,
    labels: pd.DataFrame,
    prices: pd.DataFrame,
    min_ev_threshold: float,
    rows_output_dir: Path,
) -> dict[str, Any]:
    frame = _artifact_predictions(artifact_dir)
    metrics, rows = evaluate_prediction_frame(
        frame=frame,
        labels=labels,
        prices=prices,
        min_ev_threshold=min_ev_threshold,
    )
    run_id = artifact_dir.name
    rows_path = rows_output_dir / f"{run_id}_validation_price_edge.parquet"
    rows_output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(rows_path, index=False)
    report_path = artifact_dir / "report.json"
    original_validation = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        original_validation = report.get("validation_metrics", report.get("validation", {}))
    return {
        "run_id": run_id,
        "kind": "validation",
        "artifact_dir": str(artifact_dir),
        "min_ev_threshold": float(min_ev_threshold),
        "metrics": metrics,
        "original_validation_metrics": {
            key: original_validation.get(key)
            for key in ("coverage", "accepted_sample_accuracy", "selection_score", "utility", "accepted_count")
        },
        "rows_path": str(rows_path),
    }


def _target_from_actual_side(side: str | None) -> int | None:
    if side in {"YES", "UP"}:
        return 1
    if side in {"NO", "DOWN"}:
        return 0
    return None


def evaluate_replay(
    replay_path: Path,
    *,
    labels: pd.DataFrame,
    prices: pd.DataFrame,
    min_ev_threshold: float,
    rows_output_dir: Path,
) -> list[dict[str, Any]]:
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for combo in payload.get("combos", []):
        rows = []
        for row in combo.get("rows", []):
            target = _target_from_actual_side(row.get("actual_side"))
            if target is None:
                continue
            rows.append(
                {
                    "polymarket_slug": row.get("slug") or row.get("polymarket_slug"),
                    "market_t0": row.get("t0"),
                    "target": target,
                    "p_up": row.get("p_up"),
                    "source_side": row.get("side"),
                    "confidence_bucket": row.get("confidence_bucket"),
                }
            )
        if not rows:
            continue
        frame = pd.DataFrame.from_records(rows)
        metrics, evaluated = evaluate_prediction_frame(
            frame=frame,
            labels=labels,
            prices=prices,
            min_ev_threshold=min_ev_threshold,
        )
        run_id = f"{replay_path.stem}_{combo.get('name')}"
        rows_path = rows_output_dir / f"{run_id}_price_edge.parquet"
        rows_output_dir.mkdir(parents=True, exist_ok=True)
        evaluated.to_parquet(rows_path, index=False)
        results.append(
            {
                "run_id": run_id,
                "kind": "replay",
                "replay_path": str(replay_path),
                "combo": combo.get("name"),
                "window_start_utc": payload.get("window_start_utc"),
                "window_end_utc": payload.get("window_end_utc"),
                "min_ev_threshold": float(min_ev_threshold),
                "metrics": metrics,
                "original_replay_metrics": combo.get("metrics", {}),
                "rows_path": str(rows_path),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Polymarket model predictions with price-edge trading utility.")
    parser.add_argument("--price-store", required=True)
    parser.add_argument(
        "--label-store",
        default="artifacts/data_v2/labels/polymarket_resolved/btc_updown_5m.parquet",
    )
    parser.add_argument("--artifact-dir", action="append", default=[])
    parser.add_argument("--replay-json", action="append", default=[])
    parser.add_argument("--min-ev-threshold", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="artifacts/data_v2/reports/price_edge/price_edge_evaluation.json",
    )
    parser.add_argument(
        "--rows-output-dir",
        default="artifacts/data_v2/reports/price_edge/rows",
    )
    args = parser.parse_args()

    labels = _load_label_lookup(Path(args.label_store))
    prices = _load_price_store(Path(args.price_store))
    rows_output_dir = Path(args.rows_output_dir)
    results: list[dict[str, Any]] = []
    for artifact_dir in args.artifact_dir:
        results.append(
            evaluate_artifact(
                Path(artifact_dir),
                labels=labels,
                prices=prices,
                min_ev_threshold=args.min_ev_threshold,
                rows_output_dir=rows_output_dir,
            )
        )
    for replay_json in args.replay_json:
        results.extend(
            evaluate_replay(
                Path(replay_json),
                labels=labels,
                prices=prices,
                min_ev_threshold=args.min_ev_threshold,
                rows_output_dir=rows_output_dir,
            )
        )
    payload = {
        "metric": "trading_utility",
        "definition": "sum(realized_profit for accepted trades) / total_opportunities",
        "price_rule": "YES token first prices-history point in [market_t0, market_t0+300s]; p_down=1-p_up",
        "accept_rule": "abs(model_p_up - polymarket_yes_mid_price) > min_ev_threshold",
        "min_ev_threshold": float(args.min_ev_threshold),
        "result_count": len(results),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
