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
import torch
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "upper_bound_mlp"))

from price_estimator_common import load_config, resolve_path  # noqa: E402
from train_upper_bound_mlp import Preprocessor  # noqa: E402

from expected_return_common import git_commit, write_json  # noqa: E402
from run_lgbm_ev_policy import (  # noqa: E402
    choose_bids,
    feature_columns,
    matrix,
    select_group_min_ev_policy,
    window,
)
from train_low_cdf_and_backtest import (  # noqa: E402
    HazardMLP,
    backtest_metrics,
    backtest_with_bid,
    predict_hazard,
    write_predictions,
)


LEAKAGE_FEATURE_PATTERN = re.compile(
    "target|label|winner|correct|chosen_low|future|closed|endDate|condition|market_id|"
    "question|slug|outcome|fetched|source|time_to|trade_time|timestamp|date|pnl",
    re.IGNORECASE,
)


def make_classifier(family: str, model_config: dict[str, object], seed: int):
    if family == "catboost":
        return CatBoostClassifier(
            iterations=int(model_config.get("iterations", model_config.get("n_estimators", 250))),
            learning_rate=float(model_config.get("learning_rate", 0.04)),
            depth=int(model_config.get("depth", 4)),
            l2_leaf_reg=float(model_config.get("l2_leaf_reg", 5.0)),
            random_seed=seed,
            loss_function="Logloss",
            eval_metric="Logloss",
            thread_count=int(model_config.get("n_jobs", -1)),
            allow_writing_files=False,
            verbose=False,
        )
    if family == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=int(model_config.get("n_estimators", 250)),
            learning_rate=float(model_config.get("learning_rate", 0.04)),
            max_depth=int(model_config.get("max_depth", 4)),
            subsample=float(model_config.get("subsample", 0.8)),
            colsample_bytree=float(model_config.get("colsample_bytree", 0.45)),
            min_child_weight=float(model_config.get("min_child_weight", 20.0)),
            reg_lambda=float(model_config.get("reg_lambda", 5.0)),
            random_state=seed,
            n_jobs=int(model_config.get("n_jobs", -1)),
            eval_metric="logloss",
            tree_method=str(model_config.get("tree_method", "hist")),
        )
    return lgb.LGBMClassifier(
        n_estimators=int(model_config.get("n_estimators", 250)),
        learning_rate=float(model_config.get("learning_rate", 0.04)),
        subsample=float(model_config.get("subsample", 0.8)),
        colsample_bytree=float(model_config.get("colsample_bytree", 0.45)),
        num_leaves=int(model_config.get("num_leaves", 31)),
        min_child_samples=int(model_config.get("min_child_samples", 80)),
        random_state=seed,
        n_jobs=int(model_config.get("n_jobs", -1)),
        verbose=-1,
    )


def fit_classifier(family: str, model, x_train, y_train, x_eval, y_eval, early_stopping_rounds: int) -> None:
    if family == "xgboost":
        model.fit(x_train, y_train, eval_set=[(x_eval, y_eval)], verbose=False)
        return
    if family == "catboost":
        model.fit(
            x_train,
            y_train,
            eval_set=(x_eval, y_eval),
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        return
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_eval, y_eval)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )


def fit_q_predictions(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    model_config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    family = str(model_config.get("family", "lightgbm"))
    if family == "market_mid":
        column = str(model_config.get("probability_column", "pm_l2_selected_mid"))
        if column not in calibration or column not in validation:
            raise ValueError(f"market_mid probability column missing: {column}")
        calibration_q = pd.to_numeric(calibration[column], errors="coerce").fillna(0.5).clip(0.0, 1.0).to_numpy()
        validation_q = pd.to_numeric(validation[column], errors="coerce").fillna(0.5).clip(0.0, 1.0).to_numpy()
        if bool(model_config.get("isotonic_q", False)):
            q_iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            q_iso.fit(calibration_q, calibration["correct"].astype(int))
            calibration_q = np.asarray(q_iso.predict(calibration_q), dtype=float)
            validation_q = np.asarray(q_iso.predict(validation_q), dtype=float)
        return calibration_q, validation_q
    seed = int(model_config.get("random_state", 31))
    early_stopping_rounds = int(model_config.get("early_stopping_rounds", 20))
    model = make_classifier(family, model_config, seed)
    x_fit = matrix(fit, columns)
    x_cal = matrix(calibration, columns)
    x_val = matrix(validation, columns)
    fit_classifier(
        family,
        model,
        x_fit,
        fit["correct"].astype(int),
        x_cal,
        calibration["correct"].astype(int),
        early_stopping_rounds,
    )
    calibration_q = model.predict_proba(x_cal)[:, 1]
    validation_q = model.predict_proba(x_val)[:, 1]
    if bool(model_config.get("isotonic_q", False)):
        q_iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        q_iso.fit(calibration_q, calibration["correct"].astype(int))
        calibration_q = np.asarray(q_iso.predict(calibration_q), dtype=float)
        validation_q = np.asarray(q_iso.predict(validation_q), dtype=float)
    return calibration_q, validation_q


def load_h2_gc(
    checkpoint_path: Path,
    frames: dict[str, pd.DataFrame],
    device_name: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    preprocessor = Preprocessor(**checkpoint["preprocessor"])
    tick_grid = np.asarray(checkpoint["tick_grid"], dtype=float)
    model_config = checkpoint["model"]
    model = HazardMLP(
        input_dim=len(preprocessor.output_columns),
        hidden_dims=[int(v) for v in model_config["hidden_dims"]],
        dropout=[float(v) for v in model_config["dropout"]],
        output_dim=len(tick_grid),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(device_name)
    model.to(device)
    gc = {
        name: predict_hazard(model, preprocessor.transform(frame), device, batch_size)[1]
        for name, frame in frames.items()
    }
    return tick_grid, gc


def cdf_on_bid_grid(h2_tick_grid: np.ndarray, h2_gc: np.ndarray, bid_grid: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(h2_tick_grid, bid_grid, side="left")
    if np.any(indices >= len(h2_tick_grid)) or np.any(np.abs(h2_tick_grid[indices] - bid_grid) > 1e-9):
        raise ValueError("bid_grid must be a subset of the h2 tick_grid")
    return h2_gc[:, indices]


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

    forbidden_columns = [str(value) for value in config.get("features", {}).get("forbidden_columns", [])]
    columns = feature_columns(train_all, forbidden_columns)
    forbidden_feature_overlap = sorted(set(forbidden_columns).intersection(columns))
    if forbidden_feature_overlap:
        raise ValueError(f"Forbidden feature columns present: {forbidden_feature_overlap}")
    forbidden_columns_present = sorted(set(forbidden_columns).intersection(train_all.columns))

    raw_bid_grid = np.round(
        np.arange(
            float(config["policy_search"]["bid_min"]),
            float(config["policy_search"]["bid_max"]) + 1e-12,
            float(config["policy_search"]["bid_step"]),
        ),
        10,
    )
    h2_checkpoint = resolve_path(config["paths"]["h2_checkpoint"])
    tick_grid, h2_gc = load_h2_gc(
        h2_checkpoint,
        {"calibration": calibration, "validation": validation},
        str(config.get("evaluation", {}).get("device", "cpu")),
        int(config.get("evaluation", {}).get("batch_size", 512)),
    )
    bid_grid = raw_bid_grid[raw_bid_grid <= tick_grid.max() + 1e-12]
    if len(bid_grid) == 0:
        raise ValueError("No policy bid grid points are covered by the h2 checkpoint")
    calibration_gc = cdf_on_bid_grid(tick_grid, h2_gc["calibration"], bid_grid)
    validation_gc = cdf_on_bid_grid(tick_grid, h2_gc["validation"], bid_grid)

    calibration_q, validation_q = fit_q_predictions(fit, calibration, validation, columns, config.get("model", {}))
    min_ev_grid = [float(value) for value in config["policy_search"]["min_ev_grid"]]
    bid_offset_grid = [int(value) for value in config["policy_search"].get("bid_offset_steps_grid", [0])]
    min_q_grid = [float(value) for value in config["policy_search"].get("min_q_grid", [0.0])]
    search_rows: list[dict[str, float | str]] = []
    best: tuple[float, float, int, float] | None = None
    for bid_offset_steps in bid_offset_grid:
        for min_q in min_q_grid:
            for min_ev in min_ev_grid:
                bids, expected_ev, fill_prob = choose_bids(
                    calibration_q,
                    calibration_gc,
                    bid_grid,
                    min_ev,
                    bid_offset_steps,
                    min_q,
                )
                result = backtest_with_bid(calibration, bids, expected_ev, fill_prob)
                metrics = backtest_metrics(calibration, result, len(calibration))
                row = {
                    "prediction_candidate": "xgb_q_h2_gc",
                    "min_ev": min_ev,
                    "bid_offset_steps": float(bid_offset_steps),
                    "min_q": min_q,
                    **metrics,
                }
                search_rows.append(row)
                if metrics["order_count"] < float(config["policy_search"]["min_order_count"]):
                    continue
                if best is None or metrics["sum_pnl"] > best[0]:
                    best = (float(metrics["sum_pnl"]), min_ev, bid_offset_steps, min_q)
    if best is None:
        raise ValueError("No policy candidate met min_order_count")
    selected_min_ev = best[1]
    selected_bid_offset_steps = best[2]
    selected_min_q = best[3]

    policy_extra, calibration_metrics, calibration_result, validation_metrics, validation_result = select_group_min_ev_policy(
        calibration,
        validation,
        len(calibration),
        len(validation_all),
        {"q": calibration_q, "gc": calibration_gc},
        {"q": validation_q, "gc": validation_gc},
        bid_grid,
        min_ev_grid,
        selected_min_ev,
        selected_bid_offset_steps,
        selected_min_q,
        config["policy_search"],
    )

    train_bid = np.zeros(len(accepted_train), dtype=float)
    train_result = backtest_with_bid(accepted_train, train_bid)
    pd.DataFrame(search_rows).to_csv(reports_dir / "calibration_min_ev_search.csv", index=False)
    group_policy_path = reports_dir / "group_min_ev_policy.json"
    write_json(group_policy_path, policy_extra)
    write_predictions(
        validation,
        validation_result,
        reports_dir / "predictions_validation.parquet",
        validation_q,
    )
    metrics = {
        "experiment_id": config["experiment_id"],
        "git_commit": git_commit(),
        "config_path": str(reports_dir / "config_used.yaml"),
        "report_path": str(reports_dir / "summary_metrics.json"),
        "primary_metric": "validation sum_pnl for 20260625 lowbid xgb q with frozen h2 GC",
        "objective": config["objective"],
        "baseline": config["baseline"],
        "policy": {
            "type": "q_plus_h2_gc_ev",
            "q_model_family": str(config.get("model", {}).get("family", "unknown")),
            "gc_model_family": "frozen_h2_hazard_survival_cdf",
            "source_h2_checkpoint": str(h2_checkpoint),
            "selection_source": "calibration",
            "global_selected_min_ev": selected_min_ev,
            "global_selected_bid_offset_steps": selected_bid_offset_steps,
            "global_selected_min_q": selected_min_q,
            "bid_min": float(config["policy_search"]["bid_min"]),
            "bid_max_requested": float(config["policy_search"]["bid_max"]),
            "bid_max_evaluated": float(bid_grid.max()),
            "bid_step": float(config["policy_search"]["bid_step"]),
            "min_order_count": int(config["policy_search"]["min_order_count"]),
            "isotonic_q": bool(config.get("model", {}).get("isotonic_q", False)),
            "isotonic_gc": False,
            **policy_extra,
        },
        "leakage_note": "Q model and policy selection use fit/calibration labels only. Validation labels are used only for final evaluation. H2 GC checkpoint is frozen from its source experiment.",
        "grid_limitation": "H2 checkpoint supports GC only through its max tick; requested bid grid points above that price are excluded.",
        "feature_count": len(columns),
        "excluded_feature_pattern": LEAKAGE_FEATURE_PATTERN.pattern,
        "excluded_feature_forbidden_columns": forbidden_columns,
        "forbidden_columns_present_in_dataset": forbidden_columns_present,
        "train_metrics": backtest_metrics(accepted_train, train_result, len(train_all)),
        "train_window": window(train_all),
        "global_calibration_metrics": [
            row
            for row in search_rows
            if row["min_ev"] == selected_min_ev
            and int(row["bid_offset_steps"]) == selected_bid_offset_steps
            and row["min_q"] == selected_min_q
        ][0],
        "calibration_metrics": calibration_metrics,
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
            "group_min_ev_policy": str(group_policy_path),
            "validation_predictions": str(reports_dir / "predictions_validation.parquet"),
        },
    }
    write_json(reports_dir / "summary_metrics.json", metrics)
    print({"validation_metrics": validation_metrics, "policy": metrics["policy"]})


if __name__ == "__main__":
    main()
