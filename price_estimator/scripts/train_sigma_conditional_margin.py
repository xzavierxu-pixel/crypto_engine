#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from price_estimator_common import (
    apply_sample_filter,
    ensure_no_forbidden_features,
    load_config,
    load_deploy_manifest,
    load_feature_columns,
    resolve_path,
    sigmoid,
    write_json,
)


JOIN_KEYS = ["timestamp", "decision_time", "condition_id"]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=resolve_path("."), text=True).strip()
    except Exception:
        return None


def merge_predictions(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    pred_cols = [c for c in predictions.columns if c.startswith("pred_q") or c.startswith("raw_pred_q")]
    keep = JOIN_KEYS + [c for c in pred_cols + ["raw_crossing"] if c in predictions.columns]
    missing = [c for c in JOIN_KEYS if c not in frame.columns or c not in predictions.columns]
    if missing:
        raise ValueError(f"Missing join keys for prediction merge: {missing}")
    merged = frame.merge(predictions[keep], on=JOIN_KEYS, how="left", validate="one_to_one")
    return merged


def feature_set(config: dict[str, Any], train: pd.DataFrame, variant: dict[str, Any] | None = None) -> tuple[list[str], list[str], dict[str, Any]]:
    columns: list[str] = []
    if variant is None or bool(variant.get("include_deploy_features", True)):
        feature_columns = load_feature_columns(config)
        available = [c for c in feature_columns if c in train.columns]
        missing_deploy = sorted(set(feature_columns) - set(available))
        columns.extend(available)
    else:
        missing_deploy = []

    added = [c for c in config["features"].get("added_columns", []) if c in train.columns]
    columns.extend(added)

    requested_sigma = []
    missing_sigma = []
    if variant is not None and bool(variant.get("include_sigma_feature_set", False)):
        requested_sigma = list(config["sigma"].get("sigma_feature_set", []))
        sigma_available = [c for c in requested_sigma if c in train.columns]
        missing_sigma = sorted(set(requested_sigma) - set(sigma_available))
        columns.extend(sigma_available)

    columns = list(dict.fromkeys(columns))
    ensure_no_forbidden_features(columns, list(config["features"]["forbidden_columns"]))
    cat_cols = [c for c in config["features"].get("categorical_columns", []) if c in columns]
    info = {
        "feature_count": len(columns),
        "categorical_columns": cat_cols,
        "missing_deploy_feature_count": len(missing_deploy),
        "missing_deploy_features_sample": missing_deploy[:25],
        "requested_sigma_features": requested_sigma,
        "missing_sigma_features": missing_sigma,
        "used_sigma_features": [c for c in requested_sigma if c in columns],
    }
    return columns, cat_cols, info


def prepare_pool(df: pd.DataFrame, columns: list[str], cat_cols: list[str], label: pd.Series | np.ndarray | None = None) -> Pool:
    x = df[columns].copy()
    for col in cat_cols:
        x[col] = x[col].astype("string").fillna("missing")
    for col in columns:
        if col not in cat_cols:
            x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return Pool(x, label=label, cat_features=cat_cols)


def no_eval_params(params: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(params)
    cleaned.pop("od_type", None)
    cleaned.pop("od_wait", None)
    cleaned.pop("use_best_model", None)
    return cleaned


def train_center_model(
    config: dict[str, Any],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    columns: list[str],
    cat_cols: list[str],
    models_dir: Path,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    center_cfg = config["center"]
    alpha = float(center_cfg.get("quantile", 0.5))
    params = no_eval_params(dict(center_cfg["params"]))
    params["loss_function"] = f"Quantile:alpha={alpha}"
    params["eval_metric"] = f"Quantile:alpha={alpha}"
    models_dir.mkdir(parents=True, exist_ok=True)

    oof = pd.Series(np.nan, index=train.index, dtype=float)
    sorted_train = train.sort_values("decision_time").copy()
    n = len(sorted_train)
    oof_cfg = center_cfg.get("oof", {})
    if bool(oof_cfg.get("enabled", True)):
        n_splits = int(oof_cfg.get("n_splits", 5))
        min_start = max(1, int(float(oof_cfg.get("min_train_fraction", 0.35)) * n))
        boundaries = np.linspace(min_start, n, n_splits + 1, dtype=int)
        for fold_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
            if end <= start:
                continue
            fit_part = sorted_train.iloc[:start]
            pred_part = sorted_train.iloc[start:end]
            fit_pool = prepare_pool(fit_part, columns, cat_cols, fit_part["target_logit"].astype(float))
            pred_pool = prepare_pool(pred_part, columns, cat_cols)
            model = CatBoostRegressor(**params)
            model.fit(fit_pool)
            model.save_model(models_dir / f"center_oof_fold{fold_idx}_q{int(alpha * 100)}.cbm")
            oof.loc[pred_part.index] = sigmoid(model.predict(pred_pool))

    full_train_pool = prepare_pool(train, columns, cat_cols, train["target_logit"].astype(float))
    valid_pool = prepare_pool(valid, columns, cat_cols)
    full_model = CatBoostRegressor(**params)
    full_model.fit(full_train_pool)
    full_model.save_model(models_dir / f"center_full_q{int(alpha * 100)}.cbm")
    valid_mu = pd.Series(sigmoid(full_model.predict(valid_pool)), index=valid.index, dtype=float)

    fallback_col = None
    missing_oof = oof.isna()
    for col in center_cfg.get("fallback_prediction_columns", []):
        if col in train.columns:
            fallback_col = col
            oof.loc[missing_oof] = pd.to_numeric(train.loc[missing_oof, col], errors="coerce")
            break
    info = {
        "center_quantile": alpha,
        "oof_enabled": bool(oof_cfg.get("enabled", True)),
        "oof_missing_before_fallback": int(missing_oof.sum()),
        "fallback_prediction_column": fallback_col,
        "train_mu_non_null": int(oof.notna().sum()),
        "validation_mu_non_null": int(valid_mu.notna().sum()),
        "model_target": "target_logit",
        "prediction_transform": "sigmoid",
    }
    return oof, valid_mu, info


def train_sigma_model(
    config: dict[str, Any],
    variant_name: str,
    train_sigma: pd.DataFrame,
    valid: pd.DataFrame,
    columns: list[str],
    cat_cols: list[str],
    models_dir: Path,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    sigma_cfg = config["sigma"]
    params = no_eval_params(dict(sigma_cfg["params"]))
    params["loss_function"] = sigma_cfg.get("loss_function", "Huber:delta=0.05")
    params["eval_metric"] = params["loss_function"]
    train_pool = prepare_pool(train_sigma, columns, cat_cols, train_sigma["sigma_target"].astype(float))
    valid_pool = prepare_pool(valid, columns, cat_cols)
    model = CatBoostRegressor(**params)
    model.fit(train_pool)
    model.save_model(models_dir / f"{variant_name}_sigma.cbm")
    floor = float(sigma_cfg.get("sigma_floor", 0.01))
    train_sigma_hat = pd.Series(np.maximum(model.predict(train_pool), floor), index=train_sigma.index, dtype=float)
    valid_sigma_hat = pd.Series(np.maximum(model.predict(valid_pool), floor), index=valid.index, dtype=float)
    info = {
        "sigma_floor": floor,
        "loss_function": params["loss_function"],
        "train_sigma_target_mean": float(train_sigma["sigma_target"].mean()),
        "train_sigma_hat_mean": float(train_sigma_hat.mean()),
        "validation_sigma_hat_mean": float(valid_sigma_hat.mean()),
    }
    return train_sigma_hat, valid_sigma_hat, info


def ceil_tick(values: np.ndarray, tick: float) -> np.ndarray:
    return np.ceil(values / tick - 1e-12) * tick


def floor_tick(values: np.ndarray, tick: float) -> np.ndarray:
    return np.floor(values / tick + 1e-12) * tick


def make_prediction(
    mu: pd.Series,
    sigma_hat: pd.Series,
    p_side: pd.Series,
    q_value: float,
    sigma_high: float,
    tick: float,
    cap_mode: str,
) -> tuple[pd.Series, pd.Series]:
    mu_arr = mu.to_numpy(dtype=float)
    sigma_arr = sigma_hat.to_numpy(dtype=float)
    p_side_arr = p_side.to_numpy(dtype=float)
    fallback = sigma_arr > sigma_high
    raw = mu_arr + q_value * sigma_arr
    raw = np.where(fallback, p_side_arr, np.minimum(raw, p_side_arr))
    pred = ceil_tick(np.clip(raw, 0.0, 1.0), tick)
    if cap_mode == "floor_tick":
        legal_cap = floor_tick(np.clip(p_side_arr, 0.0, 1.0), tick)
    elif cap_mode == "exact_p_side":
        legal_cap = np.clip(p_side_arr, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported calibration.p_side_cap_mode: {cap_mode}")
    pred = np.minimum(pred, legal_cap)
    pred = np.clip(pred, 0.0, 1.0)
    return pd.Series(pred, index=mu.index, dtype=float), pd.Series(fallback, index=mu.index, dtype=bool)


def metric_stats(values: pd.Series) -> dict[str, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return {"q25": float("nan"), "median": float("nan"), "mean": float("nan"), "q75": float("nan")}
    return {
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def evaluate_price_predictions(df: pd.DataFrame, pred: pd.Series, fallback: pd.Series | None, epsilon: float) -> dict[str, Any]:
    target = pd.to_numeric(df["target_raw"], errors="coerce")
    p_side = pd.to_numeric(df["p_side"], errors="coerce")
    lowest_safe = target + epsilon
    feasible = lowest_safe <= p_side
    legal = pred <= p_side + 1e-12
    covered = (pred >= lowest_safe) & legal
    denom = (p_side - lowest_safe).where(lambda s: s > 1e-12)
    norm_gap = ((pred - lowest_safe) / denom).where(covered & feasible)
    low_gap = (pred - lowest_safe).where(covered)
    side_distance = (p_side - pred).where(legal)
    out: dict[str, Any] = {
        "sample_count": float(len(df)),
        "coverage": float(covered.mean()) if len(df) else float("nan"),
        "covered_count": int(covered.sum()),
        "legal_rate": float(legal.mean()) if len(df) else float("nan"),
        "feasible_count": int(feasible.sum()),
        "max_possible_coverage": float(feasible.mean()) if len(df) else float("nan"),
        "coverage_feasible": float(covered[feasible].mean()) if feasible.any() else float("nan"),
        "pred_to_p_side_distance": metric_stats(side_distance),
        "pred_to_lowest_safe_distance": metric_stats(low_gap),
        "normalized_lowest_safe_gap": metric_stats(norm_gap),
        "covered_normalized_lowest_safe_gap_mean": metric_stats(norm_gap)["mean"],
        "covered_normalized_lowest_safe_gap_median": metric_stats(norm_gap)["median"],
        "mean_pred": float(pred.mean()) if len(pred) else float("nan"),
        "mean_lowest_safe": float(lowest_safe.mean()) if len(df) else float("nan"),
    }
    if fallback is not None:
        fallback = fallback.reindex(df.index).fillna(False).astype(bool)
        non_fallback = ~fallback
        out.update(
            {
                "fallback_rate": float(fallback.mean()) if len(df) else float("nan"),
                "fallback_coverage": float(covered[fallback].mean()) if fallback.any() else float("nan"),
                "non_fallback_coverage": float(covered[non_fallback].mean()) if non_fallback.any() else float("nan"),
                "non_fallback_norm_gap": metric_stats(norm_gap[non_fallback]),
            }
        )
    return out


def scan_candidates(
    config: dict[str, Any],
    train_sigma: pd.DataFrame,
    valid: pd.DataFrame,
    train_sigma_hat: pd.Series,
    valid_sigma_hat: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any], pd.Series, pd.Series]:
    epsilon = float(config["target"].get("epsilon", 0.01))
    tick = float(config["calibration"].get("tick_size", 0.01))
    cap_mode = str(config["calibration"].get("p_side_cap_mode", "exact_p_side"))
    min_coverage = float(config["objective"].get("min_coverage", 0.70))
    rows: list[dict[str, Any]] = []
    train_lowest_safe = train_sigma["target_raw"].astype(float) + epsilon
    scores = (train_lowest_safe - train_sigma["mu"]).astype(float) / train_sigma_hat.replace(0.0, np.nan)
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if scores.empty:
        raise ValueError("No finite normalized calibration scores.")

    best: dict[str, Any] | None = None
    best_pred: pd.Series | None = None
    best_fallback: pd.Series | None = None
    for q_quantile in [float(v) for v in config["calibration"]["q_quantiles"]]:
        q_value = float(np.quantile(scores, q_quantile))
        for high_quantile in [float(v) for v in config["calibration"]["sigma_high_quantiles"]]:
            sigma_high = float(np.quantile(train_sigma_hat.to_numpy(dtype=float), high_quantile))
            pred, fallback = make_prediction(valid["mu"], valid_sigma_hat, valid["p_side"], q_value, sigma_high, tick, cap_mode)
            metrics = evaluate_price_predictions(valid, pred, fallback, epsilon)
            row = {
                "q_quantile": q_quantile,
                "q_value": q_value,
                "sigma_high_quantile": high_quantile,
                "sigma_high": sigma_high,
            }
            row.update(flatten_metrics(metrics))
            rows.append(row)
            if metrics["coverage"] < min_coverage or math.isnan(float(metrics["covered_normalized_lowest_safe_gap_mean"])):
                continue
            candidate = {
                **row,
                "selection_metric": metrics["covered_normalized_lowest_safe_gap_mean"],
                "coverage_constraint_satisfied": True,
            }
            if best is None or candidate_key(candidate) < candidate_key(best):
                best = candidate
                best_pred = pred
                best_fallback = fallback

    result = pd.DataFrame(rows)
    if best is None:
        fallback_row = result.sort_values(["coverage", "covered_normalized_lowest_safe_gap_mean"], ascending=[False, True]).iloc[0].to_dict()
        best = {
            **fallback_row,
            "selection_metric": fallback_row.get("covered_normalized_lowest_safe_gap_mean"),
            "coverage_constraint_satisfied": False,
        }
        pred, fallback = make_prediction(
            valid["mu"],
            valid_sigma_hat,
            valid["p_side"],
            float(best["q_value"]),
            float(best["sigma_high"]),
            tick,
            cap_mode,
        )
        best_pred, best_fallback = pred, fallback
    assert best_pred is not None and best_fallback is not None
    return result, best, best_pred, best_fallback


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                out[f"{key}_{sub_key}"] = float(sub_value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = float(value)
    return out


def candidate_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["selection_metric"]),
        float(row.get("covered_normalized_lowest_safe_gap_median", np.inf)),
        -float(row.get("coverage", 0.0)),
        float(row.get("fallback_rate", np.inf)),
    )


def load_baseline(config: dict[str, Any]) -> dict[str, Any] | None:
    path = config.get("paths", {}).get("baseline_predictions_validation")
    if not path:
        return None
    pred_path = resolve_path(path)
    if not pred_path.exists():
        return None
    df = pd.read_parquet(pred_path)
    if "p_pred" not in df.columns:
        return None
    epsilon = float(config["target"].get("epsilon", 0.01))
    metrics = evaluate_price_predictions(df, df["p_pred"].astype(float), None, epsilon)
    summary_path = config.get("paths", {}).get("baseline_summary")
    summary = None
    if summary_path and resolve_path(summary_path).exists():
        with resolve_path(summary_path).open("r", encoding="utf-8") as f:
            summary = json.load(f)
    return {
        "prediction_path": str(pred_path),
        "summary_path": str(resolve_path(summary_path)) if summary_path else None,
        "metrics": metrics,
        "summary_selected": summary.get("selected") if isinstance(summary, dict) else None,
    }


def write_variant_predictions(path: Path, valid: pd.DataFrame, pred: pd.Series, fallback: pd.Series, sigma_hat: pd.Series, epsilon: float) -> None:
    out = valid.copy()
    out["sigma_hat"] = sigma_hat
    out["fallback"] = fallback
    out["p_pred"] = pred
    out["lowest_safe"] = out["target_raw"].astype(float) + epsilon
    out["covered"] = (out["p_pred"] >= out["lowest_safe"]) & (out["p_pred"] <= out["p_side"].astype(float) + 1e-12)
    denom = (out["p_side"].astype(float) - out["lowest_safe"]).where(lambda s: s > 1e-12)
    out["normalized_lowest_safe_gap"] = ((out["p_pred"] - out["lowest_safe"]) / denom).where(out["covered"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="price_estimator/configs/sigma_conditional_margin.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_deploy_manifest(config)

    train_raw = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))
    valid_raw = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))
    train_pred = pd.read_parquet(resolve_path(config["paths"]["predictions_train"]))
    valid_pred = pd.read_parquet(resolve_path(config["paths"]["predictions_validation"]))
    train = merge_predictions(train_raw, train_pred)
    valid = merge_predictions(valid_raw, valid_pred)
    train, train_filter = apply_sample_filter(train, config, manifest)
    valid, validation_filter = apply_sample_filter(valid, config, manifest)

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    center_columns, center_cat_cols, center_feature_info = feature_set(config, train)
    train_mu, valid_mu, center_info = train_center_model(config, train, valid, center_columns, center_cat_cols, models_dir)
    train = train.copy()
    valid = valid.copy()
    train["mu"] = train_mu
    valid["mu"] = valid_mu
    train["sigma_target"] = (train["target_raw"].astype(float) - train["mu"].astype(float)).abs()
    train_sigma = train.dropna(subset=["mu", "sigma_target"]).copy()

    variant_reports = {}
    best_overall: dict[str, Any] | None = None
    baseline = load_baseline(config)
    for variant in config["sigma"]["variants"]:
        variant_name = str(variant["name"])
        columns, cat_cols, feature_info = feature_set(config, train_sigma, variant)
        train_sigma_hat, valid_sigma_hat, sigma_info = train_sigma_model(
            config,
            variant_name,
            train_sigma,
            valid,
            columns,
            cat_cols,
            models_dir,
        )
        candidates, selected, pred, fallback = scan_candidates(config, train_sigma, valid, train_sigma_hat, valid_sigma_hat)
        candidates_path = reports_dir / f"{variant_name}_candidate_metrics.csv"
        candidates.to_csv(candidates_path, index=False)
        pred_path = reports_dir / f"{variant_name}_predictions_validation.parquet"
        epsilon = float(config["target"].get("epsilon", 0.01))
        write_variant_predictions(pred_path, valid, pred, fallback, valid_sigma_hat, epsilon)
        selected_metrics = evaluate_price_predictions(valid, pred, fallback, epsilon)
        variant_report = {
            "variant": variant,
            "feature_info": feature_info,
            "sigma_info": sigma_info,
            "selected": selected,
            "selected_metrics": selected_metrics,
            "candidate_metrics_path": str(candidates_path),
            "predictions_validation_path": str(pred_path),
        }
        variant_reports[variant_name] = variant_report
        candidate = {
            "variant": variant_name,
            "selection_metric": selected.get("selection_metric"),
            "coverage": selected.get("coverage"),
            "coverage_constraint_satisfied": selected.get("coverage_constraint_satisfied"),
            "report": variant_report,
        }
        if bool(candidate["coverage_constraint_satisfied"]):
            if best_overall is None or candidate_key(selected) < candidate_key(best_overall["report"]["selected"]):
                best_overall = candidate

    report = {
        "experiment_id": config["experiment_id"],
        "git_commit_at_training": git_commit(),
        "config_path": args.config,
        "primary_metric": config["objective"]["optimize_metric"],
        "objective": config["objective"],
        "target": config["target"],
        "p_side_cap_mode": config["calibration"].get("p_side_cap_mode", "floor_tick"),
        "deploy_artifact_dir": config["paths"]["deploy_artifact_dir"],
        "deploy_experiment_id": manifest.get("experiment_id"),
        "deploy_training_mode": manifest.get("training_mode"),
        "offline_validation_metric_source": manifest.get("source_report_path"),
        "sample_filter": {"train": train_filter, "validation": validation_filter},
        "center": {**center_info, "feature_info": center_feature_info},
        "train_rows_for_sigma": int(len(train_sigma)),
        "validation_rows": int(len(valid)),
        "variants": variant_reports,
        "selected_variant": best_overall["variant"] if best_overall else None,
        "baseline": baseline,
    }
    if best_overall and baseline:
        after = best_overall["report"]["selected_metrics"]
        before = baseline["metrics"]
        report["comparison"] = {
            "baseline_coverage": before["coverage"],
            "after_coverage": after["coverage"],
            "baseline_covered_normalized_lowest_safe_gap_mean": before["covered_normalized_lowest_safe_gap_mean"],
            "after_covered_normalized_lowest_safe_gap_mean": after["covered_normalized_lowest_safe_gap_mean"],
            "improved_under_coverage_constraint": bool(
                after["coverage"] >= float(config["objective"]["min_coverage"])
                and after["covered_normalized_lowest_safe_gap_mean"] < before["covered_normalized_lowest_safe_gap_mean"]
            ),
        }
    write_json(reports_dir / "summary_metrics.json", report)
    print(json.dumps({k: report[k] for k in ["experiment_id", "selected_variant"] if k in report}, indent=2))
    if "comparison" in report:
        print(json.dumps(report["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
