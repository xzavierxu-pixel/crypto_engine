#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))

from price_estimator_common import load_config, resolve_path  # noqa: E402

from expected_return_common import git_commit, write_json  # noqa: E402
from run_lgbm_ev_policy import LEAKAGE_FEATURE_PATTERN, feature_columns, matrix  # noqa: E402


def window(frame: pd.DataFrame) -> dict[str, object]:
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    return {"row_count": len(frame), "start": str(timestamp.min()), "end": str(timestamp.max())}


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def selection_score(coverage: float, accepted_accuracy: float) -> tuple[float, float, float]:
    utility = coverage * (2.0 * accepted_accuracy - 1.0)
    downside = math.sqrt(max(coverage * (1.0 - accepted_accuracy), 0.0))
    score = utility / downside if downside > 0 else 0.0
    return utility, downside, score


def q_metrics(frame: pd.DataFrame, q: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    y_correct = frame["correct"].astype(bool).to_numpy()
    target = frame["target"].astype(int).to_numpy()
    selected_side = frame["selected_side"].astype(str).to_numpy()
    p_up = np.where(selected_side == "UP", q, 1.0 - q)
    q_clipped = np.clip(q.astype(float), 1e-6, 1.0 - 1e-6)
    if threshold is None:
        accepted = np.ones(len(frame), dtype=bool)
        selected_q_threshold = 0.0
    else:
        accepted = q >= float(threshold)
        selected_q_threshold = float(threshold)

    accepted_count = int(accepted.sum())
    correct_count = int(y_correct[accepted].sum()) if accepted_count else 0
    coverage = safe_ratio(accepted_count, len(frame))
    accepted_accuracy = safe_ratio(correct_count, accepted_count)
    up_mask = accepted & (selected_side == "UP")
    down_mask = accepted & (selected_side == "DOWN")
    up_count = int(up_mask.sum())
    down_count = int(down_mask.sum())
    precision_up = safe_ratio(y_correct[up_mask].sum(), up_count)
    precision_down = safe_ratio(y_correct[down_mask].sum(), down_count)
    balanced_precision = (precision_up + precision_down) / 2.0
    utility, downside, score = selection_score(coverage, accepted_accuracy)
    all_pred_up = p_up >= 0.5
    all_sample_accuracy = float((all_pred_up.astype(int) == target).mean())
    return {
        "sample_count": float(len(frame)),
        "coverage": coverage,
        "precision_up": precision_up,
        "precision_down": precision_down,
        "balanced_precision": balanced_precision,
        "all_sample_accuracy": all_sample_accuracy,
        "accepted_sample_accuracy": accepted_accuracy,
        "share_up_predictions": safe_ratio(up_count, accepted_count),
        "share_down_predictions": safe_ratio(down_count, accepted_count),
        "selected_t_up": float("nan"),
        "selected_t_down": float("nan"),
        "selected_q_threshold": selected_q_threshold,
        "accepted_count": float(accepted_count),
        "up_prediction_count": float(up_count),
        "down_prediction_count": float(down_count),
        "roc_auc": float(roc_auc_score(y_correct.astype(int), q_clipped)),
        "brier_score": float(brier_score_loss(y_correct.astype(int), q_clipped)),
        "log_loss": float(log_loss(y_correct.astype(int), q_clipped, labels=[0, 1])),
        "utility": utility,
        "downside_risk": downside,
        "selection_score": score,
        "up_signal_count": float(up_count),
        "down_signal_count": float(down_count),
        "total_signal_count": float(accepted_count),
        "signal_coverage": coverage,
        "overall_signal_accuracy": accepted_accuracy,
    }


def search_thresholds(frame: pd.DataFrame, q: np.ndarray, min_coverage: float, search_config: dict[str, object]) -> tuple[dict[str, float], list[dict[str, float]]]:
    min_q = float(search_config.get("min_q", 0.5))
    max_q = float(search_config.get("max_q", 0.95))
    step = float(search_config.get("step", 0.001))
    thresholds = np.round(np.arange(min_q, max_q + step / 2.0, step), 10)
    rows: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    for threshold in thresholds:
        metrics = q_metrics(frame, q, float(threshold))
        row = {"q_threshold": float(threshold), **metrics}
        rows.append(row)
        if metrics["coverage"] < min_coverage:
            continue
        if best is None:
            best = row
            continue
        key = (
            metrics["selection_score"],
            metrics["utility"],
            metrics["coverage"],
            metrics["accepted_count"],
            -abs(float(threshold) - 0.5),
        )
        best_key = (
            best["selection_score"],
            best["utility"],
            best["coverage"],
            best["accepted_count"],
            -abs(best["q_threshold"] - 0.5),
        )
        if key > best_key:
            best = row
    if best is None:
        raise ValueError("No threshold satisfies min coverage")
    return best, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_path(args.config), reports_dir / "config_used.yaml")

    train = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))
    validation = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))
    timestamp = pd.to_datetime(train[str(config["split"].get("timestamp_column", "timestamp"))], utc=True)
    cutoff = timestamp.max() - pd.Timedelta(days=int(config["split"]["calibration_tail_days"]))
    fit = train.loc[(timestamp < cutoff) & train["threshold_accepted"].astype(bool)].copy().reset_index(drop=True)
    calibration = train.loc[(timestamp >= cutoff) & train["threshold_accepted"].astype(bool)].copy().reset_index(drop=True)

    forbidden_columns = [str(value) for value in config.get("features", {}).get("forbidden_columns", [])]
    columns = feature_columns(train, forbidden_columns)
    model_config = config["model"]
    model = xgb.XGBClassifier(
        n_estimators=int(model_config["n_estimators"]),
        learning_rate=float(model_config["learning_rate"]),
        max_depth=int(model_config["max_depth"]),
        subsample=float(model_config["subsample"]),
        colsample_bytree=float(model_config["colsample_bytree"]),
        min_child_weight=float(model_config["min_child_weight"]),
        reg_lambda=float(model_config["reg_lambda"]),
        random_state=int(model_config["random_state"]),
        n_jobs=int(model_config["n_jobs"]),
        tree_method=str(model_config["tree_method"]),
        eval_metric="logloss",
    )
    model.fit(
        matrix(fit, columns),
        fit["correct"].astype(int),
        eval_set=[(matrix(calibration, columns), calibration["correct"].astype(int))],
        verbose=False,
    )
    train_q = model.predict_proba(matrix(train, columns))[:, 1]
    validation_q = model.predict_proba(matrix(validation, columns))[:, 1]
    baseline_q = validation["p_side"].astype(float).to_numpy()

    min_coverage = float(config["objective"]["min_coverage"])
    best_threshold, frontier = search_thresholds(validation, validation_q, min_coverage, config["threshold_search"])
    baseline_best_threshold, baseline_frontier = search_thresholds(validation, baseline_q, min_coverage, config["threshold_search"])
    deploy_manifest = json.loads(resolve_path(config["baseline"]["deploy_manifest"]).read_text())

    predictions = validation[["timestamp", "selected_side", "target", "correct", "threshold_accepted", "p_up", "p_side"]].copy()
    predictions["q_model"] = validation_q
    predictions["baseline_q"] = baseline_q
    predictions["accepted_by_best_q_threshold"] = validation_q >= best_threshold["q_threshold"]
    predictions.to_parquet(reports_dir / "full_validation_q_predictions.parquet", index=False)
    pd.DataFrame(frontier).to_csv(reports_dir / "q_threshold_frontier.csv", index=False)
    pd.DataFrame(baseline_frontier).to_csv(reports_dir / "baseline_q_threshold_frontier.csv", index=False)

    report = {
        "experiment_id": config["experiment_id"],
        "git_commit": git_commit(),
        "config_path": str(reports_dir / "config_used.yaml"),
        "report_path": str(reports_dir / "report.json"),
        "primary_metric": "full validation q selection_score with coverage >= 0.70",
        "comparison_note": "New q_model predicts P(correct | deploy selected_side). Baseline q uses deploy p_side=max(p_up,1-p_up). Deploy manifest p_up metrics are included separately.",
        "objective": config["objective"],
        "feature_count": len(columns),
        "excluded_feature_pattern": LEAKAGE_FEATURE_PATTERN.pattern,
        "excluded_feature_forbidden_columns": forbidden_columns,
        "forbidden_columns_present_in_dataset": sorted(set(forbidden_columns).intersection(train.columns)),
        "train_window": window(train),
        "calibration_window": window(calibration),
        "validation_window": window(validation),
        "train_metrics": q_metrics(train, train_q),
        "validation_metrics_full_unfiltered": q_metrics(validation, validation_q),
        "validation_metrics_threshold_tuned": {k: v for k, v in best_threshold.items() if k != "q_threshold"} | {"q_threshold": best_threshold["q_threshold"]},
        "baseline_p_side_metrics_full_unfiltered": q_metrics(validation, baseline_q),
        "baseline_p_side_metrics_threshold_tuned": {k: v for k, v in baseline_best_threshold.items() if k != "q_threshold"} | {"q_threshold": baseline_best_threshold["q_threshold"]},
        "deploy_manifest_validation_metrics": deploy_manifest["offline_validation_metrics"],
        "coverage_constraint_satisfied": bool(best_threshold["coverage"] >= min_coverage),
        "artifacts": {
            "predictions": str(reports_dir / "full_validation_q_predictions.parquet"),
            "q_threshold_frontier": str(reports_dir / "q_threshold_frontier.csv"),
            "baseline_q_threshold_frontier": str(reports_dir / "baseline_q_threshold_frontier.csv"),
        },
        "metadata": config["metadata"],
    }
    write_json(reports_dir / "report.json", report)


if __name__ == "__main__":
    main()
