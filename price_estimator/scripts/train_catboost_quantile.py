#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from evaluate_quantile_model import evaluate_predictions, write_markdown
from price_estimator_common import (
    ensure_no_forbidden_features,
    load_config,
    load_deploy_manifest,
    load_feature_columns,
    resolve_path,
    sigmoid,
    write_json,
)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=resolve_path("."), text=True).strip()
    except Exception:
        return None


def feature_set(config: dict, train: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_columns = load_feature_columns(config)
    added = list(config["features"]["added_columns"])
    available = [c for c in feature_columns if c in train.columns]
    missing = sorted(set(feature_columns) - set(available))
    if missing:
        raise ValueError(f"Missing deploy feature columns in price dataset: {missing[:20]} total={len(missing)}")
    columns = available + added
    ensure_no_forbidden_features(columns, list(config["features"]["forbidden_columns"]))
    cat_cols = [c for c in config["features"]["categorical_columns"] if c in columns]
    return columns, cat_cols


def prepare_pool(df: pd.DataFrame, columns: list[str], cat_cols: list[str]) -> Pool:
    x = df[columns].copy()
    for col in cat_cols:
        x[col] = x[col].astype("string").fillna("missing")
    return Pool(x, label=df["target_logit"].astype(float), cat_features=cat_cols)


def train_one(alpha: float, config: dict, train_pool: Pool, valid_pool: Pool, models_dir: Path) -> CatBoostRegressor:
    params = dict(config["model"]["params"])
    params["loss_function"] = f"Quantile:alpha={alpha}"
    params["eval_metric"] = f"Quantile:alpha={alpha}"
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(models_dir / f"catboost_q{int(alpha * 100)}.cbm")
    return model


PRED_BASE_COLS = [
    "timestamp",
    "decision_time",
    "condition_id",
    "polymarket_slug",
    "selected_side",
    "p_up",
    "p_side",
    "p_bin",
    "p_side_bucket",
    "market_time_bucket",
    "target_raw",
    "lowest_trade_price_next4",
    "lowest_trade_time_next4",
    "time_to_lowest_trade_sec",
    "time_to_lowest_trade_sec_bucket",
]


def make_predictions(df: pd.DataFrame, pool: Pool, models: dict[float, CatBoostRegressor], config: dict) -> pd.DataFrame:
    pred = df[PRED_BASE_COLS].copy()
    raw_quantile_cols = []
    for alpha, model in models.items():
        raw_col = f"raw_pred_q{int(alpha * 100)}"
        raw_quantile_cols.append(raw_col)
        pred[raw_col] = sigmoid(model.predict(pool))
    raw_values = pred[raw_quantile_cols].to_numpy(dtype=float)
    pred["raw_crossing"] = (raw_values[:, 0] > raw_values[:, 1]) | (raw_values[:, 1] > raw_values[:, 2])
    sorted_values = np.sort(raw_values, axis=1)
    for idx, alpha in enumerate(config["model"]["quantiles"]):
        pred[f"pred_q{int(float(alpha) * 100)}"] = sorted_values[:, idx]
    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="price_estimator/configs/catboost_quantile_baseline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_deploy_manifest(config)
    train = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))
    valid = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))
    columns, cat_cols = feature_set(config, train)
    train_pool = prepare_pool(train, columns, cat_cols)
    valid_pool = prepare_pool(valid, columns, cat_cols)
    models_dir = resolve_path(config["paths"]["models_dir"])

    models: dict[float, CatBoostRegressor] = {}
    importances = []
    for alpha in config["model"]["quantiles"]:
        alpha = float(alpha)
        model = train_one(alpha, config, train_pool, valid_pool, models_dir)
        models[alpha] = model
        imp = pd.DataFrame({"feature": columns, f"importance_q{int(alpha * 100)}": model.get_feature_importance(train_pool)})
        importances.append(imp)

    pred = make_predictions(valid, valid_pool, models, config)
    train_pred = make_predictions(train, train_pool, models, config)

    pred_path = resolve_path(config["paths"]["predictions_validation"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(pred_path, index=False)
    train_pred_path = resolve_path(config["paths"]["predictions_train"])
    train_pred.to_parquet(train_pred_path, index=False)

    feature_importance = importances[0]
    for imp in importances[1:]:
        feature_importance = feature_importance.merge(imp, on="feature", how="outer")
    importance_cols = [c for c in feature_importance.columns if c.startswith("importance_")]
    feature_importance["importance_mean"] = feature_importance[importance_cols].mean(axis=1)
    feature_importance = feature_importance.sort_values("importance_mean", ascending=False)
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(reports_dir / "feature_importance.csv", index=False)

    validation_report = evaluate_predictions(config, pred_path, split="validation")
    train_report = evaluate_predictions(config, train_pred_path, split="train")
    report = {
        "experiment_id": config["experiment_id"],
        "validation": validation_report,
        "train": train_report,
    }
    report.update(
        {
            "git_commit_at_training": git_commit(),
            "config_path": args.config,
            "train_dataset": config["paths"]["train_dataset"],
            "validation_dataset": config["paths"]["validation_dataset"],
            "feature_count": len(columns),
            "feature_columns": columns,
            "categorical_columns": cat_cols,
            "feature_importance_path": str(reports_dir / "feature_importance.csv"),
            "deploy_artifact_dir": config["paths"]["deploy_artifact_dir"],
            "deploy_experiment_id": manifest.get("experiment_id"),
            "deploy_training_mode": manifest.get("training_mode"),
            "offline_validation_metric_source": manifest.get("source_report_path"),
            "train_rows": len(train),
            "validation_rows": len(valid),
        }
    )
    write_json(reports_dir / "quantile_metrics.json", report)
    write_markdown(validation_report, reports_dir / "quantile_baseline_report.md")
    print(json.dumps(validation_report["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
