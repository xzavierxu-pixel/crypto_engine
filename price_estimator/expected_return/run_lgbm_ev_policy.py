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
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

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


def choose_bids(
    q: np.ndarray,
    gc: np.ndarray,
    bid_grid: np.ndarray,
    min_ev: float,
    bid_offset_steps: int = 0,
    min_q: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ev = q[:, None] * gc * (1.0 - bid_grid[None, :]) - (1.0 - q[:, None]) * bid_grid[None, :]
    best_idx = np.argmax(ev, axis=1)
    selected_idx = np.clip(best_idx + int(bid_offset_steps), 0, len(bid_grid) - 1)
    selected_ev = ev[np.arange(len(q)), selected_idx]
    bids = np.where((selected_ev > min_ev) & (q >= min_q), bid_grid[selected_idx], 0.0)
    fill_prob = gc[np.arange(len(q)), selected_idx]
    return bids, selected_ev, fill_prob


def choose_bids_by_threshold(
    q: np.ndarray,
    gc: np.ndarray,
    bid_grid: np.ndarray,
    min_ev: np.ndarray,
    bid_offset_steps: int = 0,
    min_q: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ev = q[:, None] * gc * (1.0 - bid_grid[None, :]) - (1.0 - q[:, None]) * bid_grid[None, :]
    best_idx = np.argmax(ev, axis=1)
    selected_idx = np.clip(best_idx + int(bid_offset_steps), 0, len(bid_grid) - 1)
    selected_ev = ev[np.arange(len(q)), selected_idx]
    bids = np.where((selected_ev > np.asarray(min_ev, dtype=float)) & (q >= min_q), bid_grid[selected_idx], 0.0)
    fill_prob = gc[np.arange(len(q)), selected_idx]
    return bids, selected_ev, fill_prob


def policy_frame(frame: pd.DataFrame, p_side_bins: list[float] | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["selected_side"] = frame["selected_side"].astype("string").fillna("UNKNOWN")
    out["hour"] = pd.to_datetime(frame["timestamp"], utc=True).dt.hour.astype("int64").astype("string")
    if p_side_bins:
        p_side = pd.to_numeric(frame["p_side"], errors="coerce")
        out["p_side_bin"] = pd.cut(p_side, bins=p_side_bins, include_lowest=True).astype("string").fillna("missing")
    return out


def group_keys(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(["__global__"] * len(frame), index=frame.index, dtype="string")
    return frame[columns].astype("string").agg("|".join, axis=1)


def select_group_min_ev_policy(
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    calibration_available_count: int,
    validation_available_count: int,
    calibration_pred: dict[str, np.ndarray],
    validation_pred: dict[str, np.ndarray],
    bid_grid: np.ndarray,
    min_ev_grid: list[float],
    global_min_ev: float,
    bid_offset_steps: int,
    min_q: float,
    config: dict[str, object],
) -> tuple[dict[str, object], dict[str, float], object, dict[str, float], object]:
    group_config = config.get("group_min_ev")
    if not isinstance(group_config, dict) or not bool(group_config.get("enabled", False)):
        validation_bid, validation_ev, validation_fill_prob = choose_bids(
            validation_pred["q"],
            validation_pred["gc"],
            bid_grid,
            global_min_ev,
            bid_offset_steps,
            min_q,
        )
        validation_result = backtest_with_bid(validation, validation_bid, validation_ev, validation_fill_prob)
        calibration_bid, calibration_ev, calibration_fill_prob = choose_bids(
            calibration_pred["q"],
            calibration_pred["gc"],
            bid_grid,
            global_min_ev,
            bid_offset_steps,
            min_q,
        )
        calibration_result = backtest_with_bid(calibration, calibration_bid, calibration_ev, calibration_fill_prob)
        return (
            {
                "mode": "global_min_ev",
                "selected_min_ev": global_min_ev,
                "bid_offset_steps": bid_offset_steps,
                "min_q": min_q,
            },
            backtest_metrics(calibration, calibration_result, calibration_available_count),
            calibration_result,
            backtest_metrics(validation, validation_result, validation_available_count),
            validation_result,
        )

    p_side_bins = [float(value) for value in group_config.get("p_side_bins", [])]
    min_group_order_count = float(group_config.get("min_group_order_count", 20))
    candidates = group_config.get("group_candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("group_min_ev.group_candidates must be a list")

    cal_policy_frame = policy_frame(calibration, p_side_bins)
    val_policy_frame = policy_frame(validation, p_side_bins)
    per_threshold: dict[float, dict[str, object]] = {}
    for min_ev in min_ev_grid:
        bids, expected_ev, fill_prob = choose_bids(
            calibration_pred["q"],
            calibration_pred["gc"],
            bid_grid,
            min_ev,
            bid_offset_steps,
            min_q,
        )
        result = backtest_with_bid(calibration, bids, expected_ev, fill_prob)
        per_threshold[float(min_ev)] = {
            "bid": bids,
            "expected_ev": expected_ev,
            "fill_prob": fill_prob,
            "pnl": result.pnl,
            "order": bids > 0.0,
        }

    best: tuple[float, dict[str, object], dict[str, float], object, dict[str, float], object] | None = None
    for candidate in candidates:
        columns = [str(value) for value in candidate]
        missing = [column for column in columns if column not in cal_policy_frame.columns]
        if missing:
            raise ValueError(f"Unknown group_min_ev columns: {missing}")
        cal_keys = group_keys(cal_policy_frame, columns)
        val_keys = group_keys(val_policy_frame, columns)
        selected: dict[str, float] = {}
        for key in sorted(cal_keys.dropna().unique()):
            mask = cal_keys.eq(key).to_numpy()
            group_best: tuple[float, float] | None = None
            for min_ev in min_ev_grid:
                values = per_threshold[float(min_ev)]
                order_count = float(np.asarray(values["order"])[mask].sum())
                if order_count < min_group_order_count:
                    continue
                sum_pnl = float(np.asarray(values["pnl"])[mask].sum())
                if group_best is None or sum_pnl > group_best[0]:
                    group_best = (sum_pnl, float(min_ev))
            if group_best is not None:
                selected[str(key)] = group_best[1]
        cal_min_ev = cal_keys.map(selected).fillna(global_min_ev).astype(float).to_numpy()
        cal_bid, cal_ev, cal_fill_prob = choose_bids_by_threshold(
            calibration_pred["q"],
            calibration_pred["gc"],
            bid_grid,
            cal_min_ev,
            bid_offset_steps,
            min_q,
        )
        cal_result = backtest_with_bid(calibration, cal_bid, cal_ev, cal_fill_prob)
        cal_metrics = backtest_metrics(calibration, cal_result, calibration_available_count)
        val_min_ev = val_keys.map(selected).fillna(global_min_ev).astype(float).to_numpy()
        val_bid, val_ev, val_fill_prob = choose_bids_by_threshold(
            validation_pred["q"],
            validation_pred["gc"],
            bid_grid,
            val_min_ev,
            bid_offset_steps,
            min_q,
        )
        val_result = backtest_with_bid(validation, val_bid, val_ev, val_fill_prob)
        val_metrics = backtest_metrics(validation, val_result, validation_available_count)
        policy = {
            "mode": "group_min_ev",
            "selected_group_columns": columns,
            "selected_group_count": len(selected),
            "fallback_min_ev": global_min_ev,
            "bid_offset_steps": bid_offset_steps,
            "min_q": min_q,
            "min_group_order_count": min_group_order_count,
            "group_min_ev": selected,
        }
        if best is None or cal_metrics["sum_pnl"] > best[0]:
            best = (float(cal_metrics["sum_pnl"]), policy, cal_metrics, cal_result, val_metrics, val_result)
    if best is None:
        raise ValueError("No group_min_ev candidate was evaluated")
    return best[1], best[2], best[3], best[4], best[5]


def fit_lgbm_ev(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    bid_grid: np.ndarray,
    model_config: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    family = str(model_config.get("family", "lightgbm"))
    if family == "catboost":
        params = {
            "iterations": int(model_config.get("iterations", model_config.get("n_estimators", 250))),
            "learning_rate": float(model_config.get("learning_rate", 0.04)),
            "depth": int(model_config.get("depth", 4)),
            "l2_leaf_reg": float(model_config.get("l2_leaf_reg", 5.0)),
            "random_seed": int(model_config.get("random_state", 31)),
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "thread_count": int(model_config.get("n_jobs", -1)),
            "allow_writing_files": False,
            "verbose": False,
        }
    else:
        params = {
            "n_estimators": int(model_config.get("n_estimators", 250)),
            "learning_rate": float(model_config.get("learning_rate", 0.04)),
            "subsample": float(model_config.get("subsample", 0.8)),
            "colsample_bytree": float(model_config.get("colsample_bytree", 0.45)),
            "random_state": int(model_config.get("random_state", 31)),
            "n_jobs": int(model_config.get("n_jobs", -1)),
        }
    if family == "xgboost":
        params.update(
            {
                "max_depth": int(model_config.get("max_depth", 4)),
                "min_child_weight": float(model_config.get("min_child_weight", 20.0)),
                "reg_lambda": float(model_config.get("reg_lambda", 5.0)),
                "eval_metric": "logloss",
                "tree_method": str(model_config.get("tree_method", "hist")),
            }
        )
    else:
        params.update(
            {
                "num_leaves": int(model_config.get("num_leaves", 31)),
                "min_child_samples": int(model_config.get("min_child_samples", 80)),
                "verbose": -1,
            }
        )
    early_stopping_rounds = int(model_config.get("early_stopping_rounds", 20))
    base_seed = int(model_config.get("random_state", 31))
    x_fit = matrix(fit, columns)
    x_cal = matrix(calibration, columns)
    x_val = matrix(validation, columns)

    def make_classifier(seed: int):
        if family == "xgboost":
            return xgb.XGBClassifier(**{**params, "random_state": seed})
        if family == "catboost":
            return CatBoostClassifier(**{**params, "random_seed": seed})
        return lgb.LGBMClassifier(**{**params, "random_state": seed})

    def fit_classifier(model, x_train, y_train, x_eval, y_eval) -> None:
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

    q_model = make_classifier(base_seed)
    fit_classifier(q_model, x_fit, fit["correct"].astype(int), x_cal, calibration["correct"].astype(int))
    calibration_q = q_model.predict_proba(x_cal)[:, 1]
    validation_q = q_model.predict_proba(x_val)[:, 1]
    if bool(model_config.get("isotonic_q", False)):
        q_iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        q_iso.fit(calibration_q, calibration["correct"].astype(int))
        calibration_q = np.asarray(q_iso.predict(calibration_q), dtype=float)
        validation_q = np.asarray(q_iso.predict(validation_q), dtype=float)

    correct_fit = fit["correct"].astype(bool)
    correct_cal = calibration["correct"].astype(bool)
    fit_low = pd.to_numeric(fit["chosen_low"], errors="coerce")
    gc_cal: list[np.ndarray] = []
    gc_val: list[np.ndarray] = []
    for bid in bid_grid:
        y_fit = (fit_low[correct_fit] <= bid).astype(int)
        gc_model = make_classifier(int(round(float(bid) * 1000.0)))
        fit_classifier(
            gc_model,
            x_fit.loc[correct_fit],
            y_fit,
            x_cal.loc[correct_cal],
            (pd.to_numeric(calibration.loc[correct_cal, "chosen_low"], errors="coerce") <= bid).astype(int),
        )
        calibration_gc = gc_model.predict_proba(x_cal)[:, 1]
        validation_gc = gc_model.predict_proba(x_val)[:, 1]
        if bool(model_config.get("isotonic_gc", False)):
            gc_iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            gc_iso.fit(
                calibration_gc[correct_cal.to_numpy()],
                (pd.to_numeric(calibration.loc[correct_cal, "chosen_low"], errors="coerce") <= bid).astype(int),
            )
            calibration_gc = np.asarray(gc_iso.predict(calibration_gc), dtype=float)
            validation_gc = np.asarray(gc_iso.predict(validation_gc), dtype=float)
        gc_cal.append(calibration_gc)
        gc_val.append(validation_gc)

    return (
        {"q": calibration_q, "gc": np.vstack(gc_cal).T},
        {"q": validation_q, "gc": np.vstack(gc_val).T},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    family = str(config.get("model", {}).get("family", "lightgbm"))
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
    prediction_sets: list[tuple[str, dict[str, object], dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    model_config = config.get("model", {})
    if family == "blend":
        members = model_config.get("members")
        if not isinstance(members, list) or len(members) != 2:
            raise ValueError("blend model requires exactly two members")
        member_predictions = [
            fit_lgbm_ev(fit, calibration, validation, columns, bid_grid, dict(member))
            for member in members
            if isinstance(member, dict)
        ]
        if len(member_predictions) != 2:
            raise ValueError("blend model members must be mappings")
        member_names = [str(member.get("family", "lightgbm")) for member in members]
        for weight in [float(value) for value in model_config.get("blend_weight_grid", [0.5])]:
            calibration_pred = {
                "q": weight * member_predictions[0][0]["q"] + (1.0 - weight) * member_predictions[1][0]["q"],
                "gc": weight * member_predictions[0][0]["gc"] + (1.0 - weight) * member_predictions[1][0]["gc"],
            }
            validation_pred = {
                "q": weight * member_predictions[0][1]["q"] + (1.0 - weight) * member_predictions[1][1]["q"],
                "gc": weight * member_predictions[0][1]["gc"] + (1.0 - weight) * member_predictions[1][1]["gc"],
            }
            prediction_sets.append(
                (
                    f"blend_{member_names[0]}_{weight:.3f}_{member_names[1]}_{1.0 - weight:.3f}",
                    {
                        "blend_members": member_names,
                        "blend_weight_first": weight,
                        "blend_weight_second": 1.0 - weight,
                    },
                    calibration_pred,
                    validation_pred,
                )
            )
    else:
        calibration_pred, validation_pred = fit_lgbm_ev(
            fit,
            calibration,
            validation,
            columns,
            bid_grid,
            model_config,
        )
        prediction_sets.append((family, {}, calibration_pred, validation_pred))

    all_search_rows: list[dict[str, float | str]] = []
    selected: dict[str, object] | None = None
    bid_offset_grid = [int(value) for value in config["policy_search"].get("bid_offset_steps_grid", [0])]
    min_q_grid = [float(value) for value in config["policy_search"].get("min_q_grid", [0.0])]
    min_ev_grid = [float(value) for value in config["policy_search"]["min_ev_grid"]]
    for prediction_name, prediction_meta, calibration_pred, validation_pred in prediction_sets:
        search_rows: list[dict[str, float | str]] = []
        best: tuple[float, float, int, float, object] | None = None
        for bid_offset_steps in bid_offset_grid:
            for min_q in min_q_grid:
                for min_ev in min_ev_grid:
                    bids, expected_ev, fill_prob = choose_bids(
                        calibration_pred["q"],
                        calibration_pred["gc"],
                        bid_grid,
                        min_ev,
                        bid_offset_steps,
                        min_q,
                    )
                    result = backtest_with_bid(calibration, bids, expected_ev, fill_prob)
                    metrics = backtest_metrics(calibration, result, len(calibration))
                    row = {
                        "prediction_candidate": prediction_name,
                        "min_ev": min_ev,
                        "bid_offset_steps": float(bid_offset_steps),
                        "min_q": min_q,
                        **metrics,
                    }
                    search_rows.append(row)
                    all_search_rows.append(row)
                    if metrics["order_count"] < float(config["policy_search"]["min_order_count"]):
                        continue
                    if best is None or metrics["sum_pnl"] > best[0]:
                        best = (float(metrics["sum_pnl"]), min_ev, bid_offset_steps, min_q, result)
        if best is None:
            continue
        selected_min_ev = best[1]
        selected_bid_offset_steps = best[2]
        selected_min_q = best[3]
        policy_extra, calibration_metrics, calibration_result, validation_metrics, validation_result = select_group_min_ev_policy(
            calibration,
            validation,
            len(calibration),
            len(validation_all),
            calibration_pred,
            validation_pred,
            bid_grid,
            min_ev_grid,
            selected_min_ev,
            selected_bid_offset_steps,
            selected_min_q,
            config["policy_search"],
        )
        if selected is None or calibration_metrics["sum_pnl"] > selected["calibration_metrics"]["sum_pnl"]:  # type: ignore[index]
            selected = {
                "prediction_name": prediction_name,
                "prediction_meta": prediction_meta,
                "calibration_pred": calibration_pred,
                "validation_pred": validation_pred,
                "selected_min_ev": selected_min_ev,
                "selected_bid_offset_steps": selected_bid_offset_steps,
                "selected_min_q": selected_min_q,
                "policy_extra": policy_extra,
                "calibration_metrics": calibration_metrics,
                "calibration_result": calibration_result,
                "validation_metrics": validation_metrics,
                "validation_result": validation_result,
            }
    if selected is None:
        raise ValueError("No prediction candidate met min_order_count")

    selected_min_ev = float(selected["selected_min_ev"])
    selected_bid_offset_steps = int(selected["selected_bid_offset_steps"])
    selected_min_q = float(selected["selected_min_q"])
    policy_extra = dict(selected["policy_extra"])
    calibration_metrics = dict(selected["calibration_metrics"])
    validation_metrics = dict(selected["validation_metrics"])
    validation_result = selected["validation_result"]
    validation_pred = selected["validation_pred"]
    policy_extra.update(
        {
            "prediction_candidate": selected["prediction_name"],
            **dict(selected["prediction_meta"]),
        }
    )
    train_bid = np.zeros(len(accepted_train), dtype=float)
    train_result = backtest_with_bid(accepted_train, train_bid)

    pd.DataFrame(all_search_rows).to_csv(reports_dir / "calibration_min_ev_search.csv", index=False)
    group_policy_path = reports_dir / "group_min_ev_policy.json"
    write_json(group_policy_path, policy_extra)
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
        "primary_metric": f"validation sum_pnl for no-leak {family} EV policy",
        "objective": config["objective"],
        "baseline": config["baseline"],
        "policy": {
            "type": "q_plus_bid_gc_ev",
            "model_family": family,
            "selection_source": "calibration",
            "global_selected_min_ev": selected_min_ev,
            "global_selected_bid_offset_steps": selected_bid_offset_steps,
            "global_selected_min_q": selected_min_q,
            "bid_min": float(config["policy_search"]["bid_min"]),
            "bid_max": float(config["policy_search"]["bid_max"]),
            "bid_step": float(config["policy_search"]["bid_step"]),
            "min_order_count": int(config["policy_search"]["min_order_count"]),
            "isotonic_q": bool(config.get("model", {}).get("isotonic_q", False)),
            "isotonic_gc": bool(config.get("model", {}).get("isotonic_gc", False)),
            **policy_extra,
        },
        "leakage_note": "Model and policy selection use fit/calibration labels only. Validation labels are used only for final evaluation.",
        "feature_count": len(columns),
        "excluded_feature_pattern": LEAKAGE_FEATURE_PATTERN.pattern,
        "train_metrics": backtest_metrics(accepted_train, train_result, len(train_all)),
        "train_window": window(train_all),
        "global_calibration_metrics": [
            row
            for row in all_search_rows
            if row["min_ev"] == selected_min_ev and int(row["bid_offset_steps"]) == selected_bid_offset_steps
            and row["min_q"] == selected_min_q
            and row["prediction_candidate"] == selected["prediction_name"]
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


if __name__ == "__main__":
    main()
