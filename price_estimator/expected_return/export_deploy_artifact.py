#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))

RUNTIME_UNAVAILABLE_COLUMNS = {
    "close_time",
    "count",
    "taker_buy_volume",
    "feature_offset_minutes",
    "stage1_sample_weight",
    "selected_t_up",
    "selected_t_down",
}

from price_estimator_common import resolve_path  # noqa: E402
from run_lgbm_ev_policy import (  # noqa: E402
    feature_columns,
    matrix,
    select_group_min_ev_policy,
    window,
)
from train_low_cdf_and_backtest import backtest_metrics, backtest_with_bid  # noqa: E402


def _make_classifier(model_config: dict[str, object], seed: int):
    family = str(model_config.get("family", "lightgbm"))
    if family != "xgboost":
        raise ValueError(f"Only xgboost expected-return deploy export is supported here; got {family!r}.")
    import xgboost as xgb

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


def _fit_classifier(model, x_train, y_train, x_eval, y_eval) -> None:
    model.fit(x_train, y_train, eval_set=[(x_eval, y_eval)], verbose=False)


def _fit_bid_gc_artifact(config: dict[str, object]) -> tuple[dict[str, object], list[str], dict[str, object]]:
    from sklearn.isotonic import IsotonicRegression

    train_all = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))  # type: ignore[index]
    validation_all = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))  # type: ignore[index]
    split = config.get("split", {})
    timestamp_column = str(split.get("timestamp_column", "timestamp")) if isinstance(split, dict) else "timestamp"
    timestamp = pd.to_datetime(train_all[timestamp_column], utc=True)
    cutoff = timestamp.max() - pd.Timedelta(days=int(split["calibration_tail_days"]))  # type: ignore[index]
    accepted_train = train_all.loc[train_all["threshold_accepted"].astype(bool)].copy()
    fit = train_all.loc[(timestamp < cutoff) & train_all["threshold_accepted"].astype(bool)].copy().reset_index(drop=True)
    calibration = train_all.loc[(timestamp >= cutoff) & train_all["threshold_accepted"].astype(bool)].copy().reset_index(drop=True)
    validation = validation_all.loc[validation_all["threshold_accepted"].astype(bool)].copy().reset_index(drop=True)

    forbidden_columns = [str(value) for value in config.get("features", {}).get("forbidden_columns", [])]  # type: ignore[union-attr]
    forbidden_columns = sorted(set(forbidden_columns).union(RUNTIME_UNAVAILABLE_COLUMNS))
    columns = feature_columns(train_all, forbidden_columns)
    forbidden_feature_overlap = sorted(set(forbidden_columns).intersection(columns))
    if forbidden_feature_overlap:
        raise ValueError(f"Forbidden feature columns present: {forbidden_feature_overlap}")

    policy_search = config["policy_search"]  # type: ignore[index]
    bid_grid = np.round(
        np.arange(
            float(policy_search["bid_min"]),
            float(policy_search["bid_max"]) + 1e-12,
            float(policy_search["bid_step"]),
        ),
        10,
    )
    model_config = dict(config.get("model", {}))
    base_seed = int(model_config.get("random_state", 31))
    x_fit = matrix(fit, columns)
    x_cal = matrix(calibration, columns)
    x_val = matrix(validation, columns)

    q_model = _make_classifier(model_config, base_seed)
    _fit_classifier(q_model, x_fit, fit["correct"].astype(int), x_cal, calibration["correct"].astype(int))
    calibration_q_raw = q_model.predict_proba(x_cal)[:, 1]
    validation_q_raw = q_model.predict_proba(x_val)[:, 1]
    q_calibrator = None
    if bool(model_config.get("isotonic_q", False)):
        q_calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        q_calibrator.fit(calibration_q_raw, calibration["correct"].astype(int))
        calibration_q = np.asarray(q_calibrator.predict(calibration_q_raw), dtype=float)
        validation_q = np.asarray(q_calibrator.predict(validation_q_raw), dtype=float)
    else:
        calibration_q = calibration_q_raw
        validation_q = validation_q_raw

    correct_fit = fit["correct"].astype(bool)
    correct_cal = calibration["correct"].astype(bool)
    fit_low = pd.to_numeric(fit["chosen_low"], errors="coerce")
    gc_models = []
    gc_calibrators = []
    gc_cal = []
    gc_val = []
    for bid in bid_grid:
        y_fit = (fit_low[correct_fit] <= bid).astype(int)
        gc_model = _make_classifier(model_config, int(round(float(bid) * 1000.0)))
        _fit_classifier(
            gc_model,
            x_fit.loc[correct_fit],
            y_fit,
            x_cal.loc[correct_cal],
            (pd.to_numeric(calibration.loc[correct_cal, "chosen_low"], errors="coerce") <= bid).astype(int),
        )
        calibration_gc_raw = gc_model.predict_proba(x_cal)[:, 1]
        validation_gc_raw = gc_model.predict_proba(x_val)[:, 1]
        gc_calibrator = None
        if bool(model_config.get("isotonic_gc", False)):
            gc_calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            gc_calibrator.fit(
                calibration_gc_raw[correct_cal.to_numpy()],
                (pd.to_numeric(calibration.loc[correct_cal, "chosen_low"], errors="coerce") <= bid).astype(int),
            )
            calibration_gc = np.asarray(gc_calibrator.predict(calibration_gc_raw), dtype=float)
            validation_gc = np.asarray(gc_calibrator.predict(validation_gc_raw), dtype=float)
        else:
            calibration_gc = calibration_gc_raw
            validation_gc = validation_gc_raw
        gc_models.append(gc_model)
        gc_calibrators.append(gc_calibrator)
        gc_cal.append(calibration_gc)
        gc_val.append(validation_gc)

    calibration_pred = {"q": calibration_q, "gc": np.vstack(gc_cal).T}
    validation_pred = {"q": validation_q, "gc": np.vstack(gc_val).T}
    min_ev_grid = [float(value) for value in policy_search["min_ev_grid"]]
    bid_offset_grid = [int(value) for value in policy_search.get("bid_offset_steps_grid", [0])]
    min_q_grid = [float(value) for value in policy_search.get("min_q_grid", [0.0])]
    best: tuple[float, float, int, float] | None = None
    for bid_offset_steps in bid_offset_grid:
        for min_q in min_q_grid:
            for min_ev in min_ev_grid:
                from run_lgbm_ev_policy import choose_bids

                bids, expected_ev, fill_prob = choose_bids(calibration_q, calibration_pred["gc"], bid_grid, min_ev, bid_offset_steps, min_q)
                result = backtest_with_bid(calibration, bids, expected_ev, fill_prob)
                metrics = backtest_metrics(calibration, result, len(calibration))
                if metrics["order_count"] < float(policy_search["min_order_count"]):
                    continue
                if best is None or metrics["sum_pnl"] > best[0]:
                    best = (float(metrics["sum_pnl"]), float(min_ev), int(bid_offset_steps), float(min_q))
    if best is None:
        raise ValueError("No policy candidate met min_order_count")
    policy_extra, calibration_metrics, _, validation_metrics, _ = select_group_min_ev_policy(
        calibration,
        validation,
        len(calibration),
        len(validation_all),
        calibration_pred,
        validation_pred,
        bid_grid,
        min_ev_grid,
        best[1],
        best[2],
        best[3],
        policy_search,
    )
    train_result = backtest_with_bid(accepted_train, np.zeros(len(accepted_train), dtype=float))
    model_payload = {
        "q_model": q_model,
        "q_calibrator": q_calibrator,
        "gc_models": gc_models,
        "gc_calibrators": gc_calibrators,
        "bid_grid": bid_grid,
        "policy": {
            "mode": "global_min_ev",
            "selected_min_ev": float(policy_extra.get("selected_min_ev", best[1])),
            "bid_offset_steps": int(policy_extra.get("bid_offset_steps", best[2])),
            "min_q": float(policy_extra.get("min_q", best[3])),
        },
    }
    metrics = {
        "train_metrics": backtest_metrics(accepted_train, train_result, len(train_all)),
        "train_window": window(train_all),
        "calibration_metrics": calibration_metrics,
        "calibration_window": window(calibration),
        "validation_metrics": validation_metrics,
        "validation_window": window(validation_all),
    }
    return model_payload, columns, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an expected-return hazard checkpoint for execution.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--format",
        choices=["auto", "hazard", "bid-gc"],
        default="auto",
        help="Artifact format. auto uses bid-gc when the experiment config has no h2_checkpoint.",
    )
    args = parser.parse_args()

    experiment = (ROOT / args.experiment).resolve()
    output = (ROOT / args.output).resolve()
    config = yaml.safe_load((experiment / "config.yaml").read_text(encoding="utf-8"))
    report = json.loads((experiment / "reports" / "summary_metrics.json").read_text(encoding="utf-8"))
    artifact_format = args.format
    if artifact_format == "auto":
        artifact_format = "hazard" if config.get("paths", {}).get("h2_checkpoint") else "bid-gc"
    if artifact_format == "bid-gc":
        model_payload, feature_columns, recomputed_metrics = _fit_bid_gc_artifact(config)
        output.mkdir(parents=True, exist_ok=True)
        model_file = "expected_return_bid_gc.pkl"
        with (output / model_file).open("wb") as handle:
            pickle.dump(model_payload, handle)
        (output / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
        shutil.copy2(experiment / "config.yaml", output / "source_config.yaml")
        shutil.copy2(experiment / "reports" / "summary_metrics.json", output / "source_summary_metrics.json")
        manifest = {
            "artifact_type": "price_estimator_expected_return_bid_gc",
            "experiment_id": config["experiment_id"],
            "model_format": "expected_return_bid_gc_pickle",
            "model_file": model_file,
            "feature_columns_file": "feature_columns.json",
            "feature_count": len(feature_columns),
            "prediction_column": "expected_return_bid",
            "selected_side_column": "selected_side",
            "yes_value": "UP",
            "no_value": "DOWN",
            "round_decimals": 2,
            "best_ask_offset": 0.01,
            "fallback_price_mode": "skip",
            "order_price_policy": "min(best_ask - 0.01, expected_return_optimal_bid)",
            "order_policy": model_payload["policy"],
            "source_config_path": str(Path(args.experiment) / "config.yaml"),
            "source_report_path": str(Path(args.experiment) / "reports" / "summary_metrics.json"),
            "validation_metrics": recomputed_metrics["validation_metrics"],
            "source_validation_metrics": report["validation_metrics"],
            "recomputed_validation_metrics": recomputed_metrics["validation_metrics"],
            "coverage_constraint_satisfied": report["coverage_constraint_satisfied"],
            "offline_validation_metric_source": report["offline_validation_metric_source"],
        }
        (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (output / "export_report.json").write_text(json.dumps(recomputed_metrics, indent=2), encoding="utf-8")
        return

    checkpoint_path = (ROOT / config["paths"]["h2_checkpoint"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    hidden_indices = sorted({int(key.split(".")[1]) for key in state if key.startswith("trunk.") and key.endswith(".weight")})
    arrays: dict[str, np.ndarray] = {"tick_grid": np.asarray(checkpoint["tick_grid"], dtype=np.float32)}
    for output_index, state_index in enumerate(hidden_indices):
        arrays[f"hidden_{output_index}__weight"] = state[f"trunk.{state_index}.weight"].cpu().numpy()
        arrays[f"hidden_{output_index}__bias"] = state[f"trunk.{state_index}.bias"].cpu().numpy()
    arrays["output__weight"] = state["output.weight"].cpu().numpy()
    arrays["output__bias"] = state["output.bias"].cpu().numpy()

    policy = {
        "min_bid": float(config["target"]["min_bid"]),
        "min_ev": float(report["policy"]["selected_min_ev"]),
        "candidate_gc_strict_floor": float(config["order_policy"]["candidate_gc_strict_floor"]),
    }
    metadata = {
        "preprocessor": checkpoint["preprocessor"],
        "model": {"layer_count": len(hidden_indices)},
        "order_policy": policy,
    }
    output.mkdir(parents=True, exist_ok=True)
    model_file = "expected_return_hazard.npz"
    np.savez_compressed(output / model_file, metadata=json.dumps(metadata), **arrays)
    (output / "feature_columns.json").write_text(
        json.dumps(checkpoint["feature_columns"], indent=2), encoding="utf-8"
    )
    manifest = {
        "artifact_type": "price_estimator_expected_return_hazard",
        "experiment_id": config["experiment_id"],
        "model_format": "expected_return_hazard_numpy",
        "model_file": model_file,
        "feature_columns_file": "feature_columns.json",
        "feature_count": len(checkpoint["feature_columns"]),
        "prediction_column": "expected_return_bid",
        "selected_side_column": "selected_side",
        "yes_value": "UP",
        "no_value": "DOWN",
        "round_decimals": 2,
        "best_ask_offset": 0.01,
        "fallback_price_mode": "skip",
        "order_price_policy": "min(best_ask - 0.01, expected_return_optimal_bid)",
        "order_policy": policy,
        "source_config_path": str(Path(args.experiment) / "config.yaml"),
        "source_report_path": str(Path(args.experiment) / "reports" / "summary_metrics.json"),
        "source_checkpoint_path": config["paths"]["h2_checkpoint"],
        "validation_metrics": report["validation_metrics"],
        "coverage_constraint_satisfied": report["coverage_constraint_satisfied"],
        "offline_validation_metric_source": report["offline_validation_metric_source"],
    }
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
