#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
ROOT = PRICE_ESTIMATOR_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "upper_bound_mlp"))

from price_estimator_common import load_config, load_deploy_manifest, resolve_path  # noqa: E402
from train_upper_bound_mlp import Preprocessor, UpperBoundMLP, feature_set  # noqa: E402

from expected_return_common import empirical_cdf, git_commit, price_grid, write_json  # noqa: E402


class LowPointDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


@dataclass(frozen=True)
class BacktestResult:
    bid: np.ndarray
    expected_ev: np.ndarray
    fill_prob: np.ndarray
    pnl: np.ndarray
    filled: np.ndarray
    printed_filled: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_fit_calibration(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_col = str(config["split"].get("timestamp_column", "timestamp"))
    days = int(config["split"].get("calibration_tail_days", 31))
    ordered = df.sort_values(ts_col).reset_index(drop=True)
    ts = pd.to_datetime(ordered[ts_col], utc=True)
    cutoff = ts.max() - pd.Timedelta(days=days)
    fit = ordered.loc[ts < cutoff].copy()
    calibration = ordered.loc[ts >= cutoff].copy()
    if fit.empty or calibration.empty:
        raise ValueError("Calibration tail split produced an empty fit or calibration set")
    return fit, calibration


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
        z = model(xb).detach().cpu().numpy().reshape(-1)
        out.append(1.0 / (1.0 + np.exp(-z)))
    return np.concatenate(out) if out else np.asarray([], dtype=float)


def train_point_model(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_cal_correct: np.ndarray,
    y_cal_correct: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = UpperBoundMLP(
        input_dim=x_fit.shape[1],
        hidden_dims=[int(v) for v in config["model"]["hidden_dims"]],
        dropout=[float(v) for v in config["model"]["dropout"]],
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loader = DataLoader(
        LowPointDataset(x_fit, y_fit),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    loss_fn = nn.SmoothL1Loss()
    rows: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_cal = float("inf")
    stale = 0
    patience = int(config["training"].get("early_stop_patience", 10))
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = torch.sigmoid(model(xb))
            loss = loss_fn(pred, yb)
            loss.backward()
            clip = float(config["training"].get("gradient_clip_norm", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        cal_pred = predict(model, x_cal_correct, device, int(config["training"]["batch_size"]))
        cal_mae = float(np.mean(np.abs(cal_pred - y_cal_correct))) if len(y_cal_correct) else float("inf")
        rows.append({"epoch": float(epoch), "fit_loss": float(np.mean(losses)), "calibration_mae": cal_mae})
        if cal_mae < best_cal:
            best_cal = cal_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, rows


def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    return {
        "sample_count": float(len(y)),
        "mae": float(np.mean(np.abs(err))) if len(err) else float("nan"),
        "rmse": float(np.sqrt(np.mean(err * err))) if len(err) else float("nan"),
        "bias": float(np.mean(err)) if len(err) else float("nan"),
        "pred_mean": float(np.mean(pred)) if len(pred) else float("nan"),
        "target_mean": float(np.mean(y)) if len(y) else float("nan"),
    }


def choose_expected_return_bids(
    p_side: np.ndarray,
    f_pred: np.ndarray,
    residuals: np.ndarray,
    tick_size: float,
    min_bid: float,
    min_ev: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bids = np.zeros(len(p_side), dtype=float)
    evs = np.zeros(len(p_side), dtype=float)
    fill_probs = np.zeros(len(p_side), dtype=float)
    for i, (q, f) in enumerate(zip(p_side, f_pred)):
        grid = price_grid(float(q), tick_size, min_bid)
        gc = empirical_cdf(residuals, grid - float(f))
        ev = float(q) * gc * (1.0 - grid) - (1.0 - float(q)) * grid
        j = int(np.argmax(ev))
        bids[i] = grid[j] if ev[j] > min_ev else 0.0
        evs[i] = ev[j]
        fill_probs[i] = gc[j]
    return bids, evs, fill_probs


def floor_to_tick(values: np.ndarray, tick_size: float) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=float) / tick_size + 1e-12) * tick_size


def realized_pnl(df: pd.DataFrame, bid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = pd.to_numeric(df["chosen_low"], errors="coerce").to_numpy(dtype=float)
    correct = df["correct"].astype(bool).to_numpy()
    submitted = np.asarray(bid, dtype=float) > 0.0
    printed_filled = submitted & np.isfinite(low) & (low <= (np.asarray(bid, dtype=float) + 1e-12))
    filled = (correct & printed_filled) | ((~correct) & submitted)
    pnl = np.zeros(len(df), dtype=float)
    pnl[filled & correct] = 1.0 - bid[filled & correct]
    pnl[filled & (~correct)] = -bid[filled & (~correct)]
    return pnl, filled, printed_filled


def backtest_with_bid(df: pd.DataFrame, bid: np.ndarray, expected_ev: np.ndarray | None = None, fill_prob: np.ndarray | None = None) -> BacktestResult:
    pnl, filled, printed_filled = realized_pnl(df, bid)
    ev = np.zeros(len(df), dtype=float) if expected_ev is None else np.asarray(expected_ev, dtype=float)
    fp = np.full(len(df), float("nan"), dtype=float) if fill_prob is None else np.asarray(fill_prob, dtype=float)
    return BacktestResult(
        bid=np.asarray(bid, dtype=float),
        expected_ev=ev,
        fill_prob=fp,
        pnl=pnl,
        filled=filled,
        printed_filled=printed_filled,
    )


def backtest_metrics(
    df: pd.DataFrame,
    result: BacktestResult,
    available_sample_count: int | None = None,
) -> dict[str, float]:
    correct = df["correct"].astype(bool).to_numpy()
    filled = result.filled
    pnl = result.pnl
    bid = result.bid
    submitted = bid > 0.0
    wrong_submitted = (~correct) & submitted
    finite_fill_prob = result.fill_prob[np.isfinite(result.fill_prob)]
    available = int(available_sample_count) if available_sample_count is not None else len(df)
    coverage = float(len(df) / available) if available else float("nan")
    accuracy = float(correct.mean()) if len(correct) else float("nan")
    predicted_up = df["selected_side"].astype("string").eq("UP").to_numpy()
    up_count = int(predicted_up.sum())
    down_count = int((~predicted_up).sum())
    precision_up = float(correct[predicted_up].mean()) if predicted_up.any() else float("nan")
    precision_down = float(correct[~predicted_up].mean()) if (~predicted_up).any() else float("nan")
    balanced_precision = float(np.nanmean([precision_up, precision_down]))
    utility = coverage * (2.0 * accuracy - 1.0)
    downside_risk = math.sqrt(max(coverage * (1.0 - accuracy), 0.0))
    p_up = pd.to_numeric(df["p_up"], errors="coerce").to_numpy(dtype=float)
    target = pd.to_numeric(df["target"], errors="coerce").to_numpy(dtype=int)
    metrics = {
        "sample_count": float(available),
        "coverage": coverage,
        "precision_up": precision_up,
        "precision_down": precision_down,
        "balanced_precision": balanced_precision,
        "all_sample_accuracy": float(correct.sum() / available) if available else float("nan"),
        "accepted_sample_accuracy": accuracy,
        "share_up_predictions": float(up_count / len(df)) if len(df) else float("nan"),
        "share_down_predictions": float(down_count / len(df)) if len(df) else float("nan"),
        "selected_t_up": float(pd.to_numeric(df["selected_t_up"], errors="coerce").mean()),
        "selected_t_down": float(pd.to_numeric(df["selected_t_down"], errors="coerce").mean()),
        "accepted_count": float(len(df)),
        "up_prediction_count": float(up_count),
        "down_prediction_count": float(down_count),
        "roc_auc": float(roc_auc_score(target, p_up)) if len(np.unique(target)) > 1 else float("nan"),
        "brier_score": float(brier_score_loss(target, p_up)),
        "log_loss": float(log_loss(target, p_up, labels=[0, 1])),
        "utility": utility,
        "downside_risk": downside_risk,
        "selection_score": float(utility / downside_risk) if downside_risk > 0 else float("nan"),
        "up_signal_count": float(up_count),
        "down_signal_count": float(down_count),
        "total_signal_count": float(len(df)),
        "signal_coverage": coverage,
        "overall_signal_accuracy": accuracy,
        "trade_count": float(filled.sum()),
        "order_count": float(submitted.sum()),
        "order_coverage": float(submitted.mean()) if len(submitted) else float("nan"),
        "fill_rate": float(filled.mean()) if len(filled) else float("nan"),
        "correct_count": float(correct.sum()),
        "correct_fill_rate": float(filled[correct].mean()) if correct.any() else float("nan"),
        "wrong_fill_forced": float(filled[wrong_submitted].mean()) if wrong_submitted.any() else float("nan"),
        "wrong_fill_printed": float(result.printed_filled[wrong_submitted].mean()) if wrong_submitted.any() else float("nan"),
        "sum_pnl": float(pnl.sum()) if len(pnl) else float("nan"),
        "mean_accepted_pnl": float(pnl.mean()) if len(pnl) else float("nan"),
        "mean_pnl_filled": float(pnl[filled].mean()) if filled.any() else float("nan"),
        "win_pnl_sum": float(pnl[correct].sum()) if correct.any() else float("nan"),
        "loss_pnl_sum": float(pnl[~correct].sum()) if (~correct).any() else float("nan"),
        "mean_bid": float(np.mean(bid)) if len(bid) else float("nan"),
        "median_bid": float(np.median(bid)) if len(bid) else float("nan"),
        "mean_expected_ev": float(np.mean(result.expected_ev)) if len(result.expected_ev) else float("nan"),
        "negative_expected_ev_share": float((result.expected_ev < 0.0).mean()) if len(result.expected_ev) else float("nan"),
        "mean_model_fill_prob": float(np.mean(finite_fill_prob)) if len(finite_fill_prob) else float("nan"),
    }
    return metrics


def write_predictions(df: pd.DataFrame, f_pred: np.ndarray, result: BacktestResult, path: Path) -> None:
    base_cols = [
        "timestamp",
        "decision_time",
        "condition_id",
        "polymarket_slug",
        "selected_side",
        "p_up",
        "p_side",
        "selected_t_up",
        "selected_t_down",
        "target",
        "correct",
        "chosen_low",
        "chosen_low_trade_time",
        "time_to_chosen_low_sec",
    ]
    out = df[[c for c in base_cols if c in df.columns]].copy()
    out.insert(0, "sample_id", np.arange(len(out), dtype=np.int64))
    out["gc_point_pred"] = f_pred
    out["bid"] = result.bid
    out["expected_ev"] = result.expected_ev
    out["model_fill_prob"] = result.fill_prob
    out["filled"] = result.filled
    out["printed_filled"] = result.printed_filled
    out["realized_pnl"] = result.pnl
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="price_estimator/expected_return/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_deploy_manifest(config)
    seed = int(config["training"]["random_seed"])
    set_seed(seed)
    device_name = str(config["training"].get("device", "auto"))
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)

    train_all = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))
    validation = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))
    fit_all, calibration_all = split_fit_calibration(train_all, config)
    fit = fit_all.loc[fit_all["correct"].astype(bool) & fit_all["chosen_low"].notna()].copy()
    calibration_correct = calibration_all.loc[
        calibration_all["correct"].astype(bool) & calibration_all["chosen_low"].notna()
    ].copy()
    validation_correct = validation.loc[
        validation["correct"].astype(bool) & validation["chosen_low"].notna()
    ].copy()
    if fit.empty or calibration_correct.empty:
        raise ValueError("Correct-only fit/calibration data is empty")

    columns, cat_cols = feature_set(config, fit_all)
    preprocessor = Preprocessor.fit(fit_all, columns, cat_cols)
    x_fit = preprocessor.transform(fit)
    x_cal_correct = preprocessor.transform(calibration_correct)
    x_train_all = preprocessor.transform(train_all)
    x_cal_all = preprocessor.transform(calibration_all)
    x_validation = preprocessor.transform(validation)
    y_fit = pd.to_numeric(fit["chosen_low"], errors="coerce").to_numpy(dtype=float)
    y_cal_correct = pd.to_numeric(calibration_correct["chosen_low"], errors="coerce").to_numpy(dtype=float)

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_path(args.config), reports_dir / "config_used.yaml")
    target_summary_path = reports_dir / "target_build_summary.json"
    target_summary = json.loads(target_summary_path.read_text(encoding="utf-8")) if target_summary_path.exists() else {}

    model, epoch_rows = train_point_model(x_fit, y_fit, x_cal_correct, y_cal_correct, config, device)
    f_train = predict(model, x_train_all, device, int(config["training"]["batch_size"]))
    f_cal_all = predict(model, x_cal_all, device, int(config["training"]["batch_size"]))
    f_cal_correct = predict(model, x_cal_correct, device, int(config["training"]["batch_size"]))
    f_validation = predict(model, x_validation, device, int(config["training"]["batch_size"]))
    f_validation_correct = f_validation[validation["correct"].astype(bool).to_numpy()]

    residuals = y_cal_correct - f_cal_correct
    tick = float(config["target"]["tick_size"])
    min_bid = float(config["target"].get("min_bid", tick))
    min_ev_grid = [float(value) for value in config["target"]["min_ev_grid"]]

    def run_expected(df: pd.DataFrame, f_pred: np.ndarray, min_ev: float) -> BacktestResult:
        p_side = pd.to_numeric(df["p_side"], errors="coerce").to_numpy(dtype=float)
        bid, ev, fp = choose_expected_return_bids(p_side, f_pred, residuals, tick, min_bid, min_ev)
        return backtest_with_bid(df, bid, ev, fp)

    train_accepted_mask = train_all["threshold_accepted"].astype(bool).to_numpy()
    calibration_accepted_mask = calibration_all["threshold_accepted"].astype(bool).to_numpy()
    validation_accepted_mask = validation["threshold_accepted"].astype(bool).to_numpy()
    train_accepted = train_all.loc[train_accepted_mask].copy()
    calibration_accepted = calibration_all.loc[calibration_accepted_mask].copy()
    validation_accepted = validation.loc[validation_accepted_mask].copy()
    f_train_accepted = f_train[train_accepted_mask]
    f_cal_accepted = f_cal_all[calibration_accepted_mask]
    f_validation_accepted = f_validation[validation_accepted_mask]

    min_ev_search: list[dict[str, float]] = []
    validation_candidates: dict[float, BacktestResult] = {}
    for candidate in min_ev_grid:
        candidate_result = run_expected(validation_accepted, f_validation_accepted, candidate)
        validation_candidates[candidate] = candidate_result
        metrics = backtest_metrics(validation_accepted, candidate_result, len(validation))
        min_ev_search.append({"min_ev": candidate, **metrics})
    selected_min_ev = max(
        min_ev_grid,
        key=lambda value: (
            backtest_metrics(validation_accepted, validation_candidates[value], len(validation))["mean_accepted_pnl"],
            backtest_metrics(validation_accepted, validation_candidates[value], len(validation))["order_count"],
            -value,
        ),
    )

    train_result = run_expected(train_accepted, f_train_accepted, selected_min_ev)
    cal_result = run_expected(calibration_accepted, f_cal_accepted, selected_min_ev)
    validation_result = validation_candidates[selected_min_ev]
    o0_forced_no_abstain = run_expected(validation_accepted, f_validation_accepted, float("-inf"))

    p_validation = pd.to_numeric(validation_accepted["p_side"], errors="coerce").to_numpy(dtype=float)
    baselines: dict[str, dict[str, float]] = {}
    for name, bid in {
        "fixed_0p50_pside": floor_to_tick(0.50 * p_validation, tick),
        "fixed_0p75_pside": floor_to_tick(0.75 * p_validation, tick),
        "pay_pside": floor_to_tick(p_validation, tick),
    }.items():
        bid = np.maximum(bid, min_bid)
        baselines[name] = backtest_metrics(
            validation_accepted, backtest_with_bid(validation_accepted, bid), len(validation)
        )
    baselines["expected_return"] = backtest_metrics(validation_accepted, validation_result, len(validation))

    write_predictions(train_accepted, f_train_accepted, train_result, resolve_path(config["paths"]["predictions_train"]))
    write_predictions(calibration_accepted, f_cal_accepted, cal_result, resolve_path(config["paths"]["predictions_calibration"]))
    write_predictions(validation_accepted, f_validation_accepted, validation_result, resolve_path(config["paths"]["predictions_validation"]))
    pd.DataFrame(epoch_rows).to_csv(reports_dir / "epoch_metrics.csv", index=False)

    checkpoint = {
        "state_dict": model.state_dict(),
        "model": config["model"],
        "preprocessor": preprocessor.to_dict(),
        "feature_columns": columns,
        "categorical_columns": cat_cols,
        "residual_cdf": {
            "source": "calibration_correct",
            "count": int(len(residuals)),
            "residuals": residuals.astype(float).tolist(),
        },
        "target": config["target"],
    }
    checkpoint_path = models_dir / "low_cdf_point_model.pt"
    torch.save(checkpoint, checkpoint_path)

    report = {
        "experiment_id": config["experiment_id"],
        "git_commit_at_training": git_commit(),
        "git_commit": git_commit(),
        "config_path": str(reports_dir / "config_used.yaml"),
        "report_path": str(reports_dir / "summary_metrics.json"),
        "primary_metric": "validation forced mean_accepted_pnl for expected_return policy",
        "model_family": config["model"]["family"],
        "deploy_experiment_id": manifest.get("experiment_id"),
        "deploy_training_mode": manifest.get("training_mode"),
        "offline_validation_metric_source": manifest.get("source_report_path"),
        "deploy_threshold_policy_type": str((manifest.get("threshold_policy") or {}).get("type", "fallback")),
        "order_window": config["target"]["order_window"],
        "selected_min_ev": selected_min_ev,
        "min_ev_selection_source": "validation",
        "validation_optimism_note": "min_ev was selected on validation; reported validation PnL is tuned and optimistic.",
        "min_ev_search": min_ev_search,
        "o0_forced_no_abstain_validation_metrics": backtest_metrics(
            validation_accepted, o0_forced_no_abstain, len(validation)
        ),
        "train_metrics": backtest_metrics(train_accepted, train_result, len(train_all)),
        "calibration_metrics": backtest_metrics(calibration_accepted, cal_result, len(calibration_all)),
        "validation_metrics": backtest_metrics(validation_accepted, validation_result, len(validation)),
        "validation_baselines": baselines,
        "signal_coverage": float(target_summary.get("validation_summary", {}).get("accepted_coverage_vs_source", float("nan"))),
        "coverage_constraint_satisfied": bool(
            float(target_summary.get("validation_summary", {}).get("accepted_coverage_vs_source", float("nan"))) >= 0.70
        )
        if "validation_summary" in target_summary
        else None,
        "coverage_note": "Direction-model threshold coverage is unchanged; expected-return order coverage may be below 0.70.",
        "target_build_summary_path": str(target_summary_path),
        "point_model_metrics": {
            "fit_correct": point_metrics(y_fit, predict(model, x_fit, device, int(config["training"]["batch_size"]))),
            "calibration_correct": point_metrics(y_cal_correct, f_cal_correct),
            "validation_correct": point_metrics(
                pd.to_numeric(validation_correct["chosen_low"], errors="coerce").to_numpy(dtype=float),
                f_validation[validation["correct"].astype(bool).to_numpy() & validation["chosen_low"].notna().to_numpy()],
            ),
        },
        "residual_cdf": {
            "source": "calibration_correct",
            "sample_count": int(len(residuals)),
            "mean": float(np.mean(residuals)) if len(residuals) else float("nan"),
            "std": float(np.std(residuals)) if len(residuals) else float("nan"),
            "q05": float(np.quantile(residuals, 0.05)) if len(residuals) else float("nan"),
            "q50": float(np.quantile(residuals, 0.50)) if len(residuals) else float("nan"),
            "q95": float(np.quantile(residuals, 0.95)) if len(residuals) else float("nan"),
        },
        "windows": {
            "fit": {
                "row_count": int(len(fit_all)),
                "low_head_correct_labeled_count": int(len(fit)),
                "start": str(pd.to_datetime(fit_all["timestamp"], utc=True).min()),
                "end": str(pd.to_datetime(fit_all["timestamp"], utc=True).max()),
            },
            "calibration": {
                "row_count": int(len(calibration_all)),
                "start": str(pd.to_datetime(calibration_all["timestamp"], utc=True).min()),
                "end": str(pd.to_datetime(calibration_all["timestamp"], utc=True).max()),
            },
            "validation": {
                "row_count": int(len(validation)),
                "start": str(pd.to_datetime(validation["timestamp"], utc=True).min()),
                "end": str(pd.to_datetime(validation["timestamp"], utc=True).max()),
            },
        },
        "train_window": {
            "row_count": int(len(train_accepted)),
            "start": str(pd.to_datetime(train_accepted["timestamp"], utc=True).min()),
            "end": str(pd.to_datetime(train_accepted["timestamp"], utc=True).max()),
        },
        "validation_window": {
            "row_count": int(len(validation_accepted)),
            "start": str(pd.to_datetime(validation_accepted["timestamp"], utc=True).min()),
            "end": str(pd.to_datetime(validation_accepted["timestamp"], utc=True).max()),
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "config_snapshot": str(reports_dir / "config_used.yaml"),
            "epoch_metrics": str(reports_dir / "epoch_metrics.csv"),
            "train_predictions": str(resolve_path(config["paths"]["predictions_train"])),
            "calibration_predictions": str(resolve_path(config["paths"]["predictions_calibration"])),
            "validation_predictions": str(resolve_path(config["paths"]["predictions_validation"])),
        },
        "feature_count_raw": len(columns),
        "feature_count_encoded": int(x_fit.shape[1]),
        "feature_columns": columns,
        "categorical_columns": cat_cols,
    }
    write_json(reports_dir / "summary_metrics.json", report)
    print({
        "validation_metrics": report["validation_metrics"],
        "validation_baselines": report["validation_baselines"],
        "point_model_metrics": report["point_model_metrics"],
    })


if __name__ == "__main__":
    main()
