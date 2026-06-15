#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
ROOT = PRICE_ESTIMATOR_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))
sys.path.insert(0, str(SCRIPT_DIR))

from price_estimator_common import (  # noqa: E402
    apply_sample_filter,
    load_config,
    load_deploy_manifest,
    resolve_path,
    write_json,
)
from train_upper_bound_mlp import (  # noqa: E402
    PRED_BASE_COLS,
    Preprocessor,
    UpperBoundMLP,
    feature_set,
    p_side_bins,
    torch_log_cosh,
)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SupervisedTensorDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


@dataclass(frozen=True)
class ConformalPrediction:
    mu: np.ndarray
    sigma: np.ndarray
    required_margin: np.ndarray
    p_raw: np.ndarray
    p_pred: np.ndarray
    accepted: np.ndarray


def split_fit_calibration(
    df: pd.DataFrame,
    fraction: float,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration.fraction must be between 0 and 1")
    ordered = df.sort_values(timestamp_col).reset_index(drop=True) if timestamp_col in df.columns else df.reset_index(drop=True)
    cut = int(math.floor(len(ordered) * (1.0 - fraction)))
    if cut <= 0 or cut >= len(ordered):
        raise ValueError("Calibration split produced an empty fit or calibration set")
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def robust_log_cosh_loss(pred: torch.Tensor, target: torch.Tensor, scale: float) -> torch.Tensor:
    return torch_log_cosh(pred - target, scale).mean()


def positive_scale_from_logits(z: np.ndarray, sigma_floor: float) -> np.ndarray:
    sigma = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)
    return np.maximum(sigma, sigma_floor)


def local_min_upper_bound_predict(
    mu: np.ndarray,
    sigma: np.ndarray,
    p_side: np.ndarray,
    q: float,
    sigma_floor: float,
    margin_threshold: float,
) -> ConformalPrediction:
    sigma_used = np.maximum(np.asarray(sigma, dtype=float), sigma_floor)
    required_margin = float(q) * sigma_used
    p_raw = np.asarray(mu, dtype=float) + required_margin
    lower_bound = np.asarray(p_side, dtype=float) - 0.5
    p_pred = np.maximum(p_raw, lower_bound)
    accepted = (p_raw <= np.asarray(p_side, dtype=float)) & (required_margin <= float(margin_threshold))
    p_pred = np.where(accepted, p_pred, np.nan)
    return ConformalPrediction(mu=np.asarray(mu), sigma=sigma_used, required_margin=required_margin, p_raw=p_raw, p_pred=p_pred, accepted=accepted)


def local_min_upper_bound_metrics(
    y: np.ndarray,
    p_side: np.ndarray,
    prediction: ConformalPrediction,
    epsilon: float,
    tolerance: float,
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p_side = np.asarray(p_side, dtype=float)
    accepted = prediction.accepted
    accepted_count = int(accepted.sum())
    if accepted_count == 0:
        return {
            "sample_count": float(len(y)),
            "accepted_count": 0.0,
            "accepted_rate": 0.0,
            "accepted_coverage": float("nan"),
            "covered_mean_gap": float("nan"),
            "covered_median_gap": float("nan"),
            "covered_q90_gap": float("nan"),
            "covered_q10_gap": float("nan"),
            "covered_q90_q10_gap": float("nan"),
            "side_violation_rate": float("nan"),
            "mean_required_margin": float("nan"),
            "margin_threshold": float("nan"),
            "score": float("inf"),
        }
    p_acc = prediction.p_pred[accepted]
    y_acc = y[accepted]
    covered = p_acc + tolerance >= y_acc + epsilon
    gaps = p_acc - y_acc
    covered_gaps = gaps[covered]
    side_violation = p_acc > p_side[accepted] + tolerance
    q10 = float(np.quantile(covered_gaps, 0.10)) if len(covered_gaps) else float("nan")
    q90 = float(np.quantile(covered_gaps, 0.90)) if len(covered_gaps) else float("nan")
    accepted_coverage = float(np.mean(covered))
    accepted_rate = float(accepted_count / len(y)) if len(y) else float("nan")
    covered_mean_gap = float(np.mean(covered_gaps)) if len(covered_gaps) else float("nan")
    min_accepted_rate_penalty = 0.0
    score = covered_mean_gap + 0.5 * q90 + 2.0 * max(0.0, 0.70 - accepted_coverage) + min_accepted_rate_penalty
    return {
        "sample_count": float(len(y)),
        "accepted_count": float(accepted_count),
        "accepted_rate": accepted_rate,
        "accepted_coverage": accepted_coverage,
        "covered_mean_gap": covered_mean_gap,
        "covered_median_gap": float(np.median(covered_gaps)) if len(covered_gaps) else float("nan"),
        "covered_q90_gap": q90,
        "covered_q10_gap": q10,
        "covered_q90_q10_gap": q90 - q10 if math.isfinite(q90) and math.isfinite(q10) else float("nan"),
        "side_violation_rate": float(np.mean(side_violation)),
        "mean_required_margin": float(np.mean(prediction.required_margin[accepted])),
        "margin_threshold": float(np.nanmax(prediction.required_margin[accepted])),
        "score": float(score),
    }


def select_margin_threshold(
    y: np.ndarray,
    p_side: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    q: float,
    sigma_floor: float,
    epsilon: float,
    tolerance: float,
    thresholds: list[float],
    min_accepted_coverage: float,
) -> tuple[float, dict[str, float], ConformalPrediction]:
    best: tuple[float, dict[str, float], ConformalPrediction] | None = None
    for threshold in sorted(set(float(v) for v in thresholds)):
        pred = local_min_upper_bound_predict(mu, sigma, p_side, q, sigma_floor, threshold)
        metrics = local_min_upper_bound_metrics(y, p_side, pred, epsilon, tolerance)
        valid = (
            metrics["accepted_count"] > 0
            and metrics["accepted_coverage"] >= min_accepted_coverage
            and metrics["side_violation_rate"] == 0.0
            and math.isfinite(metrics["covered_mean_gap"])
        )
        if not valid:
            continue
        if best is None:
            best = (threshold, metrics, pred)
            continue
        incumbent = best[1]
        candidate_key = (
            metrics["covered_mean_gap"],
            metrics["covered_q90_gap"],
            -metrics["accepted_rate"],
            -metrics["accepted_count"],
            abs(threshold),
        )
        incumbent_key = (
            incumbent["covered_mean_gap"],
            incumbent["covered_q90_gap"],
            -incumbent["accepted_rate"],
            -incumbent["accepted_count"],
            abs(best[0]),
        )
        if candidate_key < incumbent_key:
            best = (threshold, metrics, pred)
    if best is None:
        fallback = float(max(thresholds)) if thresholds else float("inf")
        pred = local_min_upper_bound_predict(mu, sigma, p_side, q, sigma_floor, fallback)
        return fallback, local_min_upper_bound_metrics(y, p_side, pred, epsilon, tolerance), pred
    return best


@torch.no_grad()
def predict_model(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).to(device)
        out.append(model(xb).detach().cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def train_probability_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    target_transform: str,
) -> list[dict[str, float]]:
    dataset = SupervisedTensorDataset(x, y)
    generator = torch.Generator().manual_seed(int(config["training"]["random_seed"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    grad_clip = float(config["training"].get("gradient_clip_norm", 5.0))
    scale = float(config["training"].get("log_cosh_scale", 0.05))
    rows: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            z = model(xb)
            if target_transform == "sigmoid":
                pred = torch.sigmoid(z)
            elif target_transform == "softplus":
                pred = torch.nn.functional.softplus(z)
            else:
                raise ValueError(f"Unsupported target_transform: {target_transform}")
            loss = robust_log_cosh_loss(pred, yb, scale)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        rows.append({"epoch": float(epoch), "loss": float(np.mean(losses)) if losses else float("nan")})
    return rows


def base_fit_metrics(y: np.ndarray, mu: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)
    abs_residual = np.abs(residual)
    return {
        "mae": float(np.mean(abs_residual)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "median_abs_error": float(np.median(abs_residual)),
        "residual_q70": float(np.quantile(abs_residual, 0.70)),
        "residual_q90": float(np.quantile(abs_residual, 0.90)),
    }


def threshold_candidates(required_margin: np.ndarray, quantiles: list[float]) -> list[float]:
    finite = np.asarray(required_margin, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return [float("inf")]
    values = [float(np.quantile(finite, q)) for q in quantiles]
    values.append(float(np.max(finite)))
    return values


def group_metrics_row(value: str, part: pd.DataFrame) -> dict[str, Any]:
    covered = part["covered"].astype(bool)
    covered_gaps = part.loc[covered, "gap"]
    q10 = float(covered_gaps.quantile(0.10)) if len(covered_gaps) else float("nan")
    q90 = float(covered_gaps.quantile(0.90)) if len(covered_gaps) else float("nan")
    return {
        "value": value,
        "sample_count": int(len(part)),
        "accepted_coverage": float(covered.mean()) if len(part) else float("nan"),
        "covered_mean_gap": float(covered_gaps.mean()) if len(covered_gaps) else float("nan"),
        "covered_median_gap": float(covered_gaps.median()) if len(covered_gaps) else float("nan"),
        "covered_q90_gap": q90,
        "covered_q10_gap": q10,
        "covered_q90_q10_gap": q90 - q10 if math.isfinite(q90) and math.isfinite(q10) else float("nan"),
        "side_violation_rate": float(part["side_violation"].mean()) if len(part) else float("nan"),
        "mean_required_margin": float(part["required_margin"].mean()) if len(part) else float("nan"),
    }


def selective_grouped_diagnostics(
    df: pd.DataFrame,
    y: np.ndarray,
    p_side: np.ndarray,
    prediction: ConformalPrediction,
    epsilon: float,
    tolerance: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    accepted = prediction.accepted
    work = pd.DataFrame(
        {
            "accepted": accepted,
            "y": y,
            "p_side": p_side,
            "p_pred": prediction.p_pred,
            "required_margin": prediction.required_margin,
        }
    )
    work = work.loc[work["accepted"]].copy()
    if work.empty:
        return {"input_sample_count": int(len(df)), "accepted_sample_count": 0}
    work["gap"] = work["p_pred"] - work["y"]
    work["covered"] = work["p_pred"] + tolerance >= work["y"] + epsilon
    work["side_violation"] = work["p_pred"] > work["p_side"] + tolerance
    diagnostics: dict[str, Any] = {
        "input_sample_count": int(len(df)),
        "accepted_sample_count": int(len(work)),
    }
    p_side_bin_edges = [float(v) for v in config.get("diagnostics", {}).get("p_side_bin_edges", [])]
    if len(p_side_bin_edges) >= 2:
        work["p_side_bin"] = p_side_bins(work["p_side"], p_side_bin_edges)
        diagnostics["by_p_side_bin"] = [
            group_metrics_row(str(value), part) for value, part in work.groupby("p_side_bin", dropna=False)
        ]
    for col in config.get("diagnostics", {}).get("group_columns", []):
        if col not in df.columns:
            continue
        values = df.loc[accepted, col].astype("string").fillna("missing").astype(str).reset_index(drop=True)
        work[col] = values
        diagnostics[f"by_{col}"] = [group_metrics_row(str(value), part) for value, part in work.groupby(col, dropna=False)]
    price_col = config.get("diagnostics", {}).get("price_bin_column", "target_raw")
    edges = [float(v) for v in config.get("diagnostics", {}).get("price_bin_edges", [])]
    if price_col in df.columns and len(edges) >= 2:
        labels = [f"{edges[i]:.2f}_{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
        work["price_bin"] = pd.cut(
            pd.to_numeric(df.loc[accepted, price_col].reset_index(drop=True), errors="coerce"),
            bins=edges,
            labels=labels,
            include_lowest=True,
            right=False,
        ).astype("string").fillna("missing")
        diagnostics["by_price_bin"] = [
            group_metrics_row(str(value), part) for value, part in work.groupby("price_bin", dropna=False)
        ]
    return diagnostics


def write_predictions(
    df: pd.DataFrame,
    y: np.ndarray,
    p_side: np.ndarray,
    prediction: ConformalPrediction,
    path: Path,
    epsilon: float,
) -> None:
    pred = df[[c for c in PRED_BASE_COLS if c in df.columns]].copy()
    pred.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    pred["y_raw"] = y
    pred["y"] = y
    pred["target"] = np.clip(y + epsilon, 1e-6, 1.0 - 1e-6)
    pred["mu"] = prediction.mu
    pred["sigma"] = prediction.sigma
    pred["required_margin"] = prediction.required_margin
    pred["p_raw"] = prediction.p_raw
    pred["accepted"] = prediction.accepted
    pred["p_pred"] = prediction.p_pred
    pred["gap"] = prediction.p_pred - y
    pred["covered"] = prediction.p_pred + 1e-6 >= y + epsilon
    pred["side_violation"] = prediction.p_pred > p_side + 1e-6
    if "p_side" in pred.columns:
        pred["p_bin"] = p_side_bins(pred["p_side"], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]).where(
            pred["p_side"].notna(), "missing"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="price_estimator/upper_bound_mlp/configs/local_min_upper_bound.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    seed = int(config["training"]["random_seed"])
    set_seed(seed)
    device_name = str(config["training"].get("device", "auto"))
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    if device.type == "auto":
        device = torch.device("cpu")

    train_all = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))
    valid = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))
    manifest = load_deploy_manifest(config)
    train_all, train_filter = apply_sample_filter(train_all, config, manifest)
    valid, validation_filter = apply_sample_filter(valid, config, manifest)
    fit, calibration = split_fit_calibration(train_all, float(config["calibration"]["fraction"]))

    columns, cat_cols = feature_set(config, fit)
    preprocessor = Preprocessor.fit(fit, columns, cat_cols)
    x_fit = preprocessor.transform(fit)
    x_cal = preprocessor.transform(calibration)
    x_train_all = preprocessor.transform(train_all)
    x_valid = preprocessor.transform(valid)

    target_col = str(config["target"]["column"])
    epsilon = float(config["target"]["epsilon"])
    tolerance = float(config["target"]["feasibility_tolerance"])
    sigma_floor = float(config["calibration"]["sigma_floor"])
    min_accepted_coverage = float(config["objective"]["min_accepted_coverage"])
    y_fit = fit[target_col].to_numpy(dtype=np.float32)
    y_cal = calibration[target_col].to_numpy(dtype=np.float32)
    y_train_all = train_all[target_col].to_numpy(dtype=np.float32)
    y_valid = valid[target_col].to_numpy(dtype=np.float32)
    p_side_train_all = train_all["p_side"].to_numpy(dtype=np.float32)
    p_side_valid = valid["p_side"].to_numpy(dtype=np.float32)

    model_kwargs = {
        "input_dim": x_fit.shape[1],
        "hidden_dims": [int(v) for v in config["model"]["hidden_dims"]],
        "dropout": [float(v) for v in config["model"]["dropout"]],
    }
    base_model = UpperBoundMLP(**model_kwargs).to(device)
    base_rows = train_probability_model(
        base_model,
        x_fit,
        y_fit,
        config,
        device,
        int(config["training"]["base_epochs"]),
        "sigmoid",
    )
    mu_fit = 1.0 / (1.0 + np.exp(-predict_model(base_model, x_fit, device, int(config["training"]["batch_size"]))))
    abs_residual_fit = np.abs(y_fit - mu_fit).astype(np.float32)

    scale_model = UpperBoundMLP(**model_kwargs).to(device)
    scale_rows = train_probability_model(
        scale_model,
        x_fit,
        abs_residual_fit,
        config,
        device,
        int(config["training"]["scale_epochs"]),
        "softplus",
    )

    batch_size = int(config["training"]["batch_size"])
    mu_cal = 1.0 / (1.0 + np.exp(-predict_model(base_model, x_cal, device, batch_size)))
    sigma_cal = positive_scale_from_logits(predict_model(scale_model, x_cal, device, batch_size), sigma_floor)
    scores = (y_cal - mu_cal) / sigma_cal
    q = float(np.quantile(scores, float(config["calibration"]["coverage_quantile"])))
    q = max(q, float(config["calibration"].get("q_min", 0.0)))

    mu_train_all = 1.0 / (1.0 + np.exp(-predict_model(base_model, x_train_all, device, batch_size)))
    sigma_train_all = positive_scale_from_logits(predict_model(scale_model, x_train_all, device, batch_size), sigma_floor)
    mu_valid = 1.0 / (1.0 + np.exp(-predict_model(base_model, x_valid, device, batch_size)))
    sigma_valid = positive_scale_from_logits(predict_model(scale_model, x_valid, device, batch_size), sigma_floor)

    valid_required_margin = q * np.maximum(sigma_valid, sigma_floor)
    candidate_thresholds = threshold_candidates(
        valid_required_margin,
        [float(v) for v in config["threshold_search"]["margin_quantiles"]],
    )
    selected_threshold, validation_metrics, validation_prediction = select_margin_threshold(
        y_valid,
        p_side_valid,
        mu_valid,
        sigma_valid,
        q,
        sigma_floor,
        epsilon,
        tolerance,
        candidate_thresholds,
        min_accepted_coverage,
    )
    train_prediction = local_min_upper_bound_predict(
        mu_train_all,
        sigma_train_all,
        p_side_train_all,
        q,
        sigma_floor,
        selected_threshold,
    )
    train_metrics = local_min_upper_bound_metrics(y_train_all, p_side_train_all, train_prediction, epsilon, tolerance)

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    with (reports_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        rows = [{f"base_{k}": v for k, v in row.items()} for row in base_rows] + [
            {f"scale_{k}": v for k, v in row.items()} for row in scale_rows
        ]
        fieldnames = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_path = models_dir / "local_min_upper_bound.pt"
    torch.save(
        {
            "base_state_dict": base_model.state_dict(),
            "scale_state_dict": scale_model.state_dict(),
            "input_dim": x_fit.shape[1],
            "model": config["model"],
            "preprocessor": preprocessor.to_dict(),
            "feature_columns": columns,
            "categorical_columns": cat_cols,
            "target": config["target"],
            "calibration": {
                "q": q,
                "sigma_floor": sigma_floor,
                "margin_threshold": selected_threshold,
                "coverage_quantile": float(config["calibration"]["coverage_quantile"]),
            },
        },
        checkpoint_path,
    )

    train_pred_path = resolve_path(config["paths"]["predictions_train"])
    valid_pred_path = resolve_path(config["paths"]["predictions_validation"])
    write_predictions(train_all, y_train_all, p_side_train_all, train_prediction, train_pred_path, epsilon)
    write_predictions(valid, y_valid, p_side_valid, validation_prediction, valid_pred_path, epsilon)

    train_diagnostics = selective_grouped_diagnostics(
        train_all, y_train_all, p_side_train_all, train_prediction, epsilon, tolerance, config
    )
    validation_diagnostics = selective_grouped_diagnostics(
        valid, y_valid, p_side_valid, validation_prediction, epsilon, tolerance, config
    )

    report = {
        "experiment_id": config["experiment_id"],
        "git_commit_at_training": git_commit(),
        "config_path": args.config,
        "primary_metric": "validation covered_mean_gap subject to accepted_coverage >= objective.min_accepted_coverage and side_violation_rate = 0",
        "model_family": "local_min_upper_bound_two_stage_mlp_conformal_gate",
        "objective": config["objective"],
        "calibration": {
            "source": config["calibration"]["source"],
            "fraction": float(config["calibration"]["fraction"]),
            "coverage_quantile": float(config["calibration"]["coverage_quantile"]),
            "q": q,
            "sigma_floor": sigma_floor,
            "selected_margin_threshold": selected_threshold,
        },
        "base_fit_metrics": {
            "fit": base_fit_metrics(y_fit, mu_fit),
            "calibration": base_fit_metrics(y_cal, mu_cal),
            "validation": base_fit_metrics(y_valid, mu_valid),
        },
        "sample_filter": {
            "train": train_filter,
            "validation": validation_filter,
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "train_diagnostics": train_diagnostics,
        "validation_diagnostics": validation_diagnostics,
        "train_window": {
            "row_count": int(len(train_all)),
            "start": str(pd.to_datetime(train_all["timestamp"], utc=True).min()) if "timestamp" in train_all else None,
            "end": str(pd.to_datetime(train_all["timestamp"], utc=True).max()) if "timestamp" in train_all else None,
        },
        "calibration_window": {
            "row_count": int(len(calibration)),
            "start": str(pd.to_datetime(calibration["timestamp"], utc=True).min()) if "timestamp" in calibration else None,
            "end": str(pd.to_datetime(calibration["timestamp"], utc=True).max()) if "timestamp" in calibration else None,
        },
        "validation_window": {
            "row_count": int(len(valid)),
            "start": str(pd.to_datetime(valid["timestamp"], utc=True).min()) if "timestamp" in valid else None,
            "end": str(pd.to_datetime(valid["timestamp"], utc=True).max()) if "timestamp" in valid else None,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "epoch_metrics": str(reports_dir / "epoch_metrics.csv"),
            "train_predictions": str(train_pred_path),
            "validation_predictions": str(valid_pred_path),
        },
        "feature_count_raw": len(columns),
        "feature_count_encoded": int(x_fit.shape[1]),
        "feature_columns": columns,
        "categorical_columns": cat_cols,
        "deploy_artifact_dir": config["paths"]["deploy_artifact_dir"],
        "deploy_experiment_id": manifest.get("experiment_id"),
        "deploy_training_mode": manifest.get("training_mode"),
        "offline_validation_metric_source": manifest.get("source_report_path"),
        "success_criteria": {
            "accepted_coverage_at_least_target": validation_metrics["accepted_coverage"] >= min_accepted_coverage,
            "side_violation_rate_is_zero": validation_metrics["side_violation_rate"] == 0.0,
            "covered_mean_gap": validation_metrics["covered_mean_gap"],
            "accepted_rate": validation_metrics["accepted_rate"],
        },
    }
    report_path = reports_dir / "summary_metrics.json"
    write_json(report_path, report)
    print(json.dumps({"train_metrics": train_metrics, "validation_metrics": validation_metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
