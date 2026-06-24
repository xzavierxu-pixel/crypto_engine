#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))

from price_estimator_common import load_config, resolve_path  # noqa: E402

from expected_return_common import git_commit, write_json  # noqa: E402
from train_low_cdf_and_backtest import backtest_metrics, backtest_with_bid, write_predictions  # noqa: E402

LEAKAGE_FEATURE_PATTERN = re.compile(
    "target|label|winner|correct|chosen_low|future|closed|endDate|condition|market_id|"
    "question|slug|outcome|fetched|source|time_to|trade_time|timestamp|date|pnl",
    re.IGNORECASE,
)


def window(frame: pd.DataFrame) -> dict[str, object]:
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    return {"row_count": len(frame), "start": str(timestamp.min()), "end": str(timestamp.max())}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.select_dtypes(include=[np.number, bool]).columns
        if not LEAKAGE_FEATURE_PATTERN.search(column)
    ]


def matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].replace([np.inf, -np.inf], np.nan)


def choose_bids(q: np.ndarray, gc: np.ndarray, bid_grid: np.ndarray, min_ev: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ev = q[:, None] * gc * (1.0 - bid_grid[None, :]) - (1.0 - q[:, None]) * bid_grid[None, :]
    best_idx = np.argmax(ev, axis=1)
    best_ev = ev[np.arange(len(q)), best_idx]
    bids = np.where(best_ev > min_ev, bid_grid[best_idx], 0.0)
    fill_prob = gc[np.arange(len(q)), best_idx]
    return bids, best_ev, fill_prob


def fit_lgbm_ev(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    bid_grid: np.ndarray,
    model_config: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    params = {
        "n_estimators": int(model_config.get("n_estimators", 250)),
        "learning_rate": float(model_config.get("learning_rate", 0.04)),
        "num_leaves": int(model_config.get("num_leaves", 31)),
        "subsample": float(model_config.get("subsample", 0.8)),
        "colsample_bytree": float(model_config.get("colsample_bytree", 0.45)),
        "min_child_samples": int(model_config.get("min_child_samples", 80)),
        "random_state": int(model_config.get("random_state", 31)),
        "n_jobs": int(model_config.get("n_jobs", -1)),
        "verbose": -1,
    }
    early_stopping_rounds = int(model_config.get("early_stopping_rounds", 20))
    x_fit = matrix(fit, columns)
    x_cal = matrix(calibration, columns)
    x_val = matrix(validation, columns)

    q_model = lgb.LGBMClassifier(**params)
    q_model.fit(
        x_fit,
        fit["correct"].astype(int),
        eval_set=[(x_cal, calibration["correct"].astype(int))],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    calibration_q = q_model.predict_proba(x_cal)[:, 1]
    validation_q = q_model.predict_proba(x_val)[:, 1]

    correct_fit = fit["correct"].astype(bool)
    correct_cal = calibration["correct"].astype(bool)
    fit_low = pd.to_numeric(fit["chosen_low"], errors="coerce")
    gc_cal: list[np.ndarray] = []
    gc_val: list[np.ndarray] = []
    for bid in bid_grid:
        y_fit = (fit_low[correct_fit] <= bid).astype(int)
        gc_model = lgb.LGBMClassifier(**{**params, "random_state": int(round(float(bid) * 1000.0))})
        gc_model.fit(
            x_fit.loc[correct_fit],
            y_fit,
            eval_set=[
                (
                    x_cal.loc[correct_cal],
                    (pd.to_numeric(calibration.loc[correct_cal, "chosen_low"], errors="coerce") <= bid).astype(int),
                )
            ],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        gc_cal.append(gc_model.predict_proba(x_cal)[:, 1])
        gc_val.append(gc_model.predict_proba(x_val)[:, 1])

    return (
        {"q": calibration_q, "gc": np.vstack(gc_cal).T},
        {"q": validation_q, "gc": np.vstack(gc_val).T},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_path(args.config), reports_dir / "config_used.yaml")

    train_all = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))
    validation_all = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))
    timestamp = pd.to_datetime(train_all[str(config["split"].get("timestamp_column", "timestamp"))], utc=True)
    cutoff = timestamp.max() - pd.Timedelta(days=int(config["split"]["calibration_tail_days"]))
    accepted_train = train_all.loc[train_all["threshold_accepted"].astype(bool)].copy()
    fit = train_all.loc[(timestamp < cutoff) & train_all["threshold_accepted"].astype(bool)].copy().reset_index(drop=True)
    calibration = train_all.loc[(timestamp >= cutoff) & train_all["threshold_accepted"].astype(bool)].copy().reset_index(drop=True)
    validation = validation_all.loc[validation_all["threshold_accepted"].astype(bool)].copy().reset_index(drop=True)

    columns = feature_columns(train_all)
    bid_grid = np.round(
        np.arange(
            float(config["policy_search"]["bid_min"]),
            float(config["policy_search"]["bid_max"]) + 1e-12,
            float(config["policy_search"]["bid_step"]),
        ),
        10,
    )
    calibration_pred, validation_pred = fit_lgbm_ev(
        fit,
        calibration,
        validation,
        columns,
        bid_grid,
        config.get("model", {}),
    )

    search_rows: list[dict[str, float]] = []
    best: tuple[float, float, object] | None = None
    for min_ev in [float(value) for value in config["policy_search"]["min_ev_grid"]]:
        bids, expected_ev, fill_prob = choose_bids(calibration_pred["q"], calibration_pred["gc"], bid_grid, min_ev)
        result = backtest_with_bid(calibration, bids, expected_ev, fill_prob)
        metrics = backtest_metrics(calibration, result, len(calibration))
        row = {"min_ev": min_ev, **metrics}
        search_rows.append(row)
        if metrics["order_count"] < float(config["policy_search"]["min_order_count"]):
            continue
        if best is None or metrics["sum_pnl"] > best[0]:
            best = (float(metrics["sum_pnl"]), min_ev, result)
    if best is None:
        raise ValueError("No min_ev candidate met min_order_count")
    selected_min_ev = best[1]

    validation_bid, validation_ev, validation_fill_prob = choose_bids(
        validation_pred["q"],
        validation_pred["gc"],
        bid_grid,
        selected_min_ev,
    )
    validation_result = backtest_with_bid(validation, validation_bid, validation_ev, validation_fill_prob)
    train_bid = np.zeros(len(accepted_train), dtype=float)
    train_result = backtest_with_bid(accepted_train, train_bid)
    validation_metrics = backtest_metrics(validation, validation_result, len(validation_all))

    pd.DataFrame(search_rows).to_csv(reports_dir / "calibration_min_ev_search.csv", index=False)
    write_predictions(
        validation,
        validation_result,
        reports_dir / "predictions_validation.parquet",
        validation_pred["q"],
    )
    metrics = {
        "experiment_id": config["experiment_id"],
        "git_commit": git_commit(),
        "config_path": str(reports_dir / "config_used.yaml"),
        "report_path": str(reports_dir / "summary_metrics.json"),
        "primary_metric": "validation sum_pnl for no-leak LightGBM EV policy",
        "objective": config["objective"],
        "baseline": config["baseline"],
        "policy": {
            "type": "lgbm_q_plus_bid_gc_ev",
            "selection_source": "calibration",
            "selected_min_ev": selected_min_ev,
            "bid_min": float(config["policy_search"]["bid_min"]),
            "bid_max": float(config["policy_search"]["bid_max"]),
            "bid_step": float(config["policy_search"]["bid_step"]),
            "min_order_count": int(config["policy_search"]["min_order_count"]),
        },
        "leakage_note": "Model and policy selection use fit/calibration labels only. Validation labels are used only for final evaluation.",
        "feature_count": len(columns),
        "excluded_feature_pattern": LEAKAGE_FEATURE_PATTERN.pattern,
        "train_metrics": backtest_metrics(accepted_train, train_result, len(train_all)),
        "train_window": window(train_all),
        "calibration_metrics": search_rows[[row["min_ev"] for row in search_rows].index(selected_min_ev)],
        "calibration_window": window(calibration),
        "validation_metrics": validation_metrics,
        "validation_window": window(validation_all),
        "signal_coverage": validation_metrics["coverage"],
        "coverage_constraint_satisfied": bool(validation_metrics["coverage"] >= float(config["objective"]["min_coverage"])),
        "target_sum_pnl_satisfied": bool(validation_metrics["sum_pnl"] >= float(config["objective"]["target_validation_sum_pnl"])),
        "deploy_training_mode": config["metadata"]["deploy_training_mode"],
        "offline_validation_metric_source": config["metadata"]["offline_validation_metric_source"],
        "artifacts": {
            "calibration_min_ev_search": str(reports_dir / "calibration_min_ev_search.csv"),
            "validation_predictions": str(reports_dir / "predictions_validation.parquet"),
        },
    }
    write_json(reports_dir / "summary_metrics.json", metrics)


if __name__ == "__main__":
    main()
